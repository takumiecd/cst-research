"""Common-horizon DST confirmation consumer, with explicit prior-producer imports.

The external f377 and975 benchmarks remain unchanged. This controller never
imports an evaluator or reads confirmation labels. All LR decisions use dev CE.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from . import higgs_dst_comparison as policy
from .higgs_dst_comparison import file_sha, json_sha, read, write
from .higgs_dst_comparison_driver import safe_path
from .higgs_dst_evidence import (
    STAGE1_COMMIT,
    STAGE1_MANIFEST_SHA,
    PoolJob,
    audit_original_qualification,
    audit_raw_cell,
    audit_raw_qualification,
    load_module,
    require,
)

STAGES = (
    "qualify",
    "import-qualification",
    "import-original-qualification",
    "import-hpo",
    "import-original-width",
    "plan-width",
    "retune",
    "retune-control",
    "freeze-width",
    "confirm",
    "confirm-controls",
    "import-original-controls",
    "global-freeze",
)
FILES = ("best-dev.pt", "best-dev.json", "learning.json")


def protocol():
    return {
        **policy.protocol(),
        "version": "dst-comparison-v2-confirmation",
        "prior_hpo_consumer": STAGE1_COMMIT,
        "prior_consumer_manifest_sha256": STAGE1_MANIFEST_SHA,
        "imports": "full pool source/result archives, trusted tracked inventory, completed phase, actual raw tensors; retain producer identity",
        "original_gate": "comparison-owned exact975 original30 freeze proof before global78; primary assess only afterwards; reused controls scored once by primary",
        "raw_state": "minimal selected checkpoints+curves+receipt proof; full raw artifacts retained at external receipt-bound locators",
    }


def control_configurations(width):
    require(width in policy.WIDTHS, "invalid width")
    return [{"id": "dense", "kind": "dense", "width": width, "divisor": 100}] + [
        {"id": f"cst-{d}", "kind": "cst", "width": width, "divisor": d}
        for d in policy.DIVISORS
    ]


def comparable(candidate):
    return {k: v for k, v in candidate.items() if k != "cell_directory"}


def top_two(candidates):
    policy.select(candidates)
    return sorted(
        candidates, key=lambda r: (r["best_development_loss"], r["lr"], r["best_epoch"])
    )[:2]


def validate_selection(selection, candidates, config):
    require(
        selection["configuration"] == config
        and selection["development_seed"] == 17
        and selection["horizon"] == 10
        and selection["confirmation_evaluated"] is False
        and selection["test_evaluated"] is False,
        "HPO selection identity/scope differs",
    )
    require(
        selection.get("base_learning_rates") == list(policy.LRS)
        and selection.get("evaluated_learning_rates")
        == sorted(r["lr"] for r in candidates),
        "declared LR inventory differs",
    )
    base = [r for r in candidates if r["lr"] in policy.LRS]
    require(
        len(base) == 4 and {r["lr"] for r in base} == set(policy.LRS),
        "complete four-LR grid required",
    )
    extra = policy.extension(policy.select(base)["lr"])
    require(
        len(candidates) == 4 + (extra is not None)
        and {r["lr"] for r in candidates}
        == set(policy.LRS) | ({extra} if extra else set())
        and selection["edge_extension"] == extra,
        "single-boundary extension policy differs",
    )
    require(
        {r["lr"]: comparable(r) for r in selection["candidates"]}
        == {r["lr"]: comparable(r) for r in candidates}
        and comparable(selection["selected"]) == comparable(policy.select(candidates)),
        "selection differs from raw admitted cells",
    )


class Campaign:
    def __init__(self, args, invoke=None):
        self.args, self.invoke = args, invoke or subprocess.run
        self.seen_jobs = {}
        self.identity = read(args.identity_file)
        require(
            self.identity["protocol_sha256"] == json_sha(protocol())
            and self.identity["dataset_sha256"] == policy.DATA_SHA256
            and file_sha(args.data) == policy.DATA_SHA256,
            "stage2 identity/data/protocol differs",
        )
        self.state, self.output = args.campaign_state, args.output
        self.state.mkdir(parents=True, exist_ok=True)
        write(self.state / "experiment-identity.json", self.identity, immutable=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cells = self.output / "cells"
        self.cells.mkdir(exist_ok=False)
        self.record = {
            "schema_version": 1,
            "stage": args.stage,
            "identity": self.identity,
            "status": "running",
            "phase_id": uuid.uuid4().hex,
            "width": args.width,
            "kind": args.kind,
            "divisor": args.divisor,
            "seed": args.seed,
            "cells": [],
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        self.persist()
        from .higgs_dst_assessment import verify_sources

        verify_sources(
            args.source_index,
            {
                "source_manifest_sha256": {
                    policy.EXTERNAL_COMMIT: self.identity[
                        "external_source_manifest_sha256"
                    ],
                    policy.ORIGINAL_COMMIT: self.identity[
                        "original_source_manifest_sha256"
                    ],
                }
            },
        )
        self.original = load_module(
            args.original_source,
            "benchmarks/training/higgs_confirmation_campaign.py",
            "_verified_original_campaign",
        )
        self.original_identity = self.original.experiment_identity(
            args.data, file_sha(args.inputs / "original-campaign-source-manifest.json")
        )

    def persist(self):
        write(self.output / "phase.json", self.record)

    def unsealed(self):
        require(
            not (self.state / "confirmation-freeze.json").exists(),
            "global confirmation freeze is immutable",
        )

    def qualified(self):
        marker = read(self.state / "qualification.json")
        require(
            marker["identity"] == self.identity and marker["status"] == "passed",
            "current consumer/source/data/runtime not qualified",
        )

    def qualify(self):
        if (self.state / "qualification.json").exists():
            self.qualified()
            marker = read(self.state / "qualification.json")
            write(self.output / "reused-qualification.json", marker)
            return marker
        destination = self.output / "qualification"
        argv = [
            sys.executable,
            "-m",
            "benchmarks.training.higgs_dst_reference",
            "--qualify-only",
            "--output",
            str(destination.resolve()),
        ]
        with (self.output / "qualification.log").open("w") as log:
            self.invoke(
                argv,
                cwd=self.args.external_source,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=True,
            )
        policy.validate_qualification(destination, self.identity)
        audit = audit_raw_qualification(destination, self.args.external_source)
        # Original controls run through the unchanged qualified975 benchmark. Its
        # separate qualification is required before any fresh control training.
        marker = {
            "status": "passed",
            "identity": self.identity,
            "artifact_hashes": policy.files(destination),
            "raw_audit": audit,
        }
        write(self.state / "qualification.json", marker, immutable=True)
        return marker

    def index(self):
        require(
            self.args.import_index is not None,
            "receipt-bound --import-index is required",
        )
        return read(self.args.import_index)

    def job(self, descriptor, *, stage1=False, original_qualification=False):
        supplied = Path(descriptor["directory"])
        directory = (
            supplied
            if supplied.is_absolute() and getattr(self.args, "offline_import", False)
            else safe_path(self.args.import_index.parent, descriptor["directory"])
        )
        if stage1:
            job = PoolJob(
                directory,
                self.args.inputs / "stage1-consumer-source-manifest.json",
                "src/cst_research/experiments/higgs_dst_comparison_driver.py",
                manifest_sha=STAGE1_MANIFEST_SHA,
            )
            identity = job.phase["identity"]
            require(
                identity["consumer_commit"] == STAGE1_COMMIT
                and identity["consumer_manifest_sha256"] == STAGE1_MANIFEST_SHA
                and identity["external_source_commit"] == policy.EXTERNAL_COMMIT
                and identity["external_source_manifest_sha256"]
                == self.identity["external_source_manifest_sha256"]
                and identity["external_archive_sha256"]
                == self.identity["external_archive_sha256"]
                and identity["protocol_sha256"] == json_sha(policy.protocol())
                and identity["dataset_sha256"] == policy.DATA_SHA256
                and identity["runtime_fingerprint"]
                == self.identity["runtime_fingerprint"],
                "prior HPO producer binding differs",
            )
            job.require_payload("consumer-source-manifest.json", STAGE1_MANIFEST_SHA)
        else:
            job = PoolJob(
                directory,
                self.args.inputs / "original-source-manifest.json",
                "benchmarks/training/higgs_confirmation_bootstrap.py",
                original_qualification=original_qualification,
            )
            require(
                job.phase["identity"] == self.original_identity,
                "original producer full identity differs",
            )
            job.require_payload(
                "source-manifest.json", self.original_identity["source_identity"]
            )
            # Primary identity lacks runtime: bind actual runner and pool receipts.
            verification = read(job.artifacts / "bootstrap-source-verification.json")
            require(
                verification["runtime"] == self.identity["runtime_fingerprint"]
                and verification["source_commit"] == policy.ORIGINAL_COMMIT
                and verification["source_manifest_sha256"]
                == self.original_identity["source_identity"],
                "original full runtime/source verification differs",
            )
            versions = read(job.results / "result.json")["versions"]
            current = self.identity["runtime_fingerprint"]
            require(
                versions["torch"] == current["torch"]
                and versions["cuda"] == current["cuda"],
                "original producer wheel differs",
            )
        job.require_payload("screen.npz", policy.DATA_SHA256)
        job_id = job.spec["id"]
        require(
            job_id not in self.seen_jobs, "duplicate producer job ID in import index"
        )
        self.seen_jobs[job_id] = job.proof
        return job

    def validate(
        self, directory, config, seed, lr, horizon, producer=None, *, raw=True
    ):
        dst = config["kind"] in policy.METHODS
        candidate = (
            policy.validate_cell(
                directory, config, seed, lr, horizon, producer or self.identity
            )
            if dst
            else self.original._validate_cell(
                directory, config, seed, lr, horizon, self.original_identity
            )
        )
        learning = read(directory / "learning.json")
        runtime = self.identity["runtime_fingerprint"]
        require(
            learning["torch"] == runtime["torch"]
            and learning["cuda"] == runtime["cuda"]
            and learning["gpu"] == "NVIDIA L4"
            and learning["TF32"] is False
            and learning["confirmation_evaluated"] is False
            and learning["test_evaluated"] is False,
            "raw cell runtime or heldout scope differs",
        )
        audit = (
            audit_raw_cell(
                directory,
                self.args.external_source if dst else self.args.original_source,
                dst=dst,
                data=self.args.data,
            )
            if raw
            else None
        )
        return candidate, audit

    def import_qualification(self):
        self.unsealed()
        job = self.job(self.index()["qualification_job"], stage1=True)
        require(
            job.phase["stage"] == "qualify", "completed prior qualification required"
        )
        policy.validate_qualification(job.artifacts / "qualification", self.identity)
        audit = audit_raw_qualification(
            job.artifacts / "qualification", self.args.external_source
        )
        marker = {
            "status": "passed",
            "identity": self.identity,
            "producer_identity": job.phase["identity"],
            "raw_proof": job.proof,
            "raw_audit": audit,
            "imported": True,
        }
        write(self.state / "qualification.json", marker, immutable=True)
        write(self.output / "imported-qualification.json", marker)
        return marker

    def import_original_qualification(self):
        self.unsealed()
        job = self.job(
            self.index()["original_qualification_job"], original_qualification=True
        )
        require(
            job.phase["stage"] == "qualify", "completed original qualification required"
        )
        audit = audit_original_qualification(
            job.artifacts,
            self.args.original_source,
            self.original_identity,
            self.identity["runtime_fingerprint"],
        )
        marker = {
            "status": "passed",
            "identity": self.original_identity,
            "runtime_fingerprint": self.identity["runtime_fingerprint"],
            "importer_identity": self.identity,
            "raw_proof": job.proof,
            "raw_audit": audit,
        }
        write(self.state / "original-qualification.json", marker, immutable=True)
        write(self.output / "imported-original-qualification.json", marker)
        return marker

    def import_hpo(self):
        self.unsealed()
        imported = []
        for descriptor in self.index()["hpo_jobs"]:
            job = self.job(descriptor, stage1=True)
            require(
                job.phase["stage"] == "tune",
                "only completed prior tuning phases admitted",
            )
            export = read(job.artifacts / "hpo-export.json")
            producer = job.phase["identity"]
            require(
                export["producer_identity"] == producer
                and export["artifact_kind"] == "dst-development-hpo-group-v1"
                and export["confirmation_evaluated"] is False
                and export["test_evaluated"] is False,
                "HPO export producer/scope differs",
            )
            config = export["configuration"]
            require(
                config
                == policy.configuration(
                    config["width"], config["kind"], config["divisor"]
                ),
                "undeclared imported group",
            )
            selected_path = safe_path(job.artifacts, export["selection_file"])
            require(
                file_sha(selected_path) == export["selection_sha256"],
                "HPO selection bytes differ",
            )
            selection = read(selected_path)
            require(
                selection["identity"] == producer, "selection producer identity differs"
            )
            candidates, audits = [], []
            for row in export["cells"]:
                directory = safe_path(job.artifacts, row["relative_directory"])
                require(
                    policy.files(directory) == row["files"],
                    "imported raw cell inventory differs",
                )
                candidate, audit = self.validate(
                    directory, config, 17, row["candidate"]["lr"], 10, producer
                )
                require(
                    candidate == row["candidate"],
                    "export candidate differs from actual cell",
                )
                candidates.append(candidate)
                audits.append(
                    {
                        "relative_directory": row["relative_directory"],
                        "files": row["files"],
                        "raw_audit": audit,
                    }
                )
            validate_selection(selection, candidates, config)
            require(
                len(export["cells"]) == len(job.phase["cells"])
                and {r["label"]: r["candidate"] for r in export["cells"]}
                == {r["label"]: r["candidate"] for r in job.phase["cells"]}
                and all(r["status"] == "completed" for r in job.phase["cells"]),
                "phase/export cell inventory differs",
            )
            record = {
                "schema_version": 1,
                "producer_identity": producer,
                "selection": selection,
                "raw_proof": job.proof,
                "verified_cells": audits,
                "importer_identity": self.identity,
            }
            write(
                self.state
                / "imported-hpo"
                / str(config["width"])
                / (config["id"] + ".json"),
                record,
                immutable=True,
            )
            imported.append(record)
        write(self.output / "imported-hpo.json", imported)
        return imported

    def original_cells(self, job, configs, *, seed=17):
        admitted = {}
        for row in job.phase["cells"]:
            require(row["status"] == "completed", "original cell incomplete")
            candidate = row["candidate"]
            directory = safe_path(job.artifacts, "cells/" + row["label"])
            learning = read(directory / "learning.json")
            config = next(
                (
                    c
                    for c in configs
                    if all(learning[k] == c[k] for k in ("kind", "width", "divisor"))
                ),
                None,
            )
            require(
                config is not None and learning["seed"] == seed,
                "unexpected original cell configuration",
            )
            actual, audit = self.validate(
                directory, config, seed, learning["lr"], len(learning["epochs"])
            )
            require(
                comparable(candidate) == comparable(actual),
                "original candidate differs from raw evidence",
            )
            admitted[(config["id"], actual["epochs"], actual["lr"])] = (
                actual,
                directory,
                audit,
            )
        return admitted

    def import_original_width(self):
        self.unsealed()
        imported = []
        for descriptor in self.index()["original_width_jobs"]:
            width = descriptor["width"]
            configs = control_configurations(width)
            job = self.job(descriptor)
            require(
                job.phase["stage"] == "freeze-width" and job.phase["width"] == width,
                "wrong original width freeze job",
            )
            frozen = read(job.artifacts / "frozen-width.json")
            desc = {
                "source_commit": policy.ORIGINAL_COMMIT,
                "dataset_sha256": policy.DATA_SHA256,
                "width": width,
                "expected_identity": self.original_identity,
                "source_manifest_sha256": self.original_identity["source_identity"],
                "protocol_sha256": self.original_identity["protocol_sha256"],
            }
            policy.validate_original_width(frozen, desc, width)
            admitted = self.original_cells(job, configs)
            proofs = [job.proof]
            for directory in descriptor["tuning_jobs"]:
                tuning = self.job({"directory": directory})
                require(
                    tuning.phase["stage"] == "tune" and tuning.phase["width"] == width,
                    "wrong original tuning job",
                )
                group = read(tuning.artifacts / "selection.json")
                cells = self.original_cells(tuning, configs)
                cfg = group["configuration"]
                validate_selection(group, [v[0] for v in cells.values()], cfg)
                require(
                    not set(cells) & set(admitted),
                    "duplicate original candidate evidence",
                )
                admitted.update(cells)
                proofs.append(tuning.proof)
            wanted = set()
            for config in frozen["configurations"]:
                for candidate in (
                    config["base_tuning_candidates"] + config["horizon_candidates"]
                ):
                    key = (config["id"], candidate["epochs"], candidate["lr"])
                    require(
                        key in admitted
                        and comparable(candidate) == comparable(admitted[key][0]),
                        "original frozen candidate lacks raw proof",
                    )
                    wanted.add(key)
            require(set(admitted) == wanted, "original width raw inventory differs")
            record = {
                "frozen": frozen,
                "producer_identity": self.original_identity,
                "raw_proofs": proofs,
                "candidate_raw_audits": [
                    {"configuration": list(k), "audit": v[2]}
                    for k, v in admitted.items()
                ],
            }
            write(
                self.state / "original-widths" / f"{width}.json", record, immutable=True
            )
            imported.append(record)
        write(self.output / "imported-original-widths.json", imported)
        return imported

    def plan_width(self):
        width = self.args.width
        require(width in policy.WIDTHS, "declared width required")
        original = read(self.state / "original-widths" / f"{width}.json")["frozen"]
        groups = [
            read(self.state / "imported-hpo" / str(width) / (c["id"] + ".json"))[
                "selection"
            ]
            for c in policy.configurations(width)
        ]
        horizon, triggers = policy.comparison_horizon(
            original["horizon"], groups, width
        )
        plan = {
            "schema_version": 1,
            "identity": self.identity,
            "width": width,
            "original_horizon": original["horizon"],
            "horizon": horizon,
            "dst_late_triggers": triggers,
            "fresh20_controls": original["horizon"] == 10 and horizon == 20,
            "original_width_sha256": file_sha(
                self.state / "original-widths" / f"{width}.json"
            ),
            "hpo_sha256": {
                c["id"]: file_sha(
                    self.state / "imported-hpo" / str(width) / (c["id"] + ".json")
                )
                for c in policy.configurations(width)
            },
        }
        write(self.state / "plans" / f"{width}.json", plan, immutable=True)
        write(self.output / "width-plan.json", plan)
        return plan

    def plan(self, width):
        plan = read(self.state / "plans" / f"{width}.json")
        require(plan["identity"] == self.identity, "comparison plan identity differs")
        require(
            file_sha(self.state / "original-widths" / f"{width}.json")
            == plan["original_width_sha256"],
            "original width changed after plan",
        )
        for cid, digest in plan["hpo_sha256"].items():
            require(
                file_sha(self.state / "imported-hpo" / str(width) / (cid + ".json"))
                == digest,
                "imported HPO changed after plan",
            )
        return plan

    def cell(self, config, seed, lr, horizon):
        self.qualified()
        dst = config["kind"] in policy.METHODS
        if not dst:
            marker = read(self.state / "original-qualification.json")
            require(
                marker["status"] == "passed"
                and marker["identity"] == self.original_identity
                and marker["runtime_fingerprint"]
                == self.identity["runtime_fingerprint"],
                "fresh controls require actual975 qualification",
            )
        label = f"N{config['width']}-{config['id']}-seed{seed}-lr{lr:g}-{horizon}ep"
        target = self.cells / label
        completed = self.state / "completed-development-cells" / (label + ".json")
        if seed == 17 and completed.exists():
            retained = read(completed)
            require(
                retained["identity"] == self.identity
                and retained["configuration"] == config
                and retained["candidate"]["lr"] == lr
                and retained["candidate"]["epochs"] == horizon,
                "completed development-cell identity differs",
            )
            self.record["cells"].append(
                {
                    "label": label,
                    "status": "completed",
                    "reused": True,
                    "candidate": retained["candidate"],
                    "raw_audit": retained["raw_audit"],
                    "raw_files": retained["raw_files"],
                    "origin": retained["origin"],
                }
            )
            self.persist()
            return retained["candidate"], None
        argv = [
            sys.executable,
            "-m",
            "benchmarks.training."
            + ("higgs_dst_reference" if dst else "higgs_confirmation"),
            "--data",
            str(self.args.data.resolve()),
            "--output",
            str(target.resolve()),
            "--kind",
            config["kind"],
            "--width",
            str(config["width"]),
            "--divisor",
            str(config["divisor"]),
            "--seed",
            str(seed),
            "--lr",
            str(lr),
            "--epochs",
            str(horizon),
        ]
        if dst:
            argv += ["--mode", "graph"]
        progress = {"label": label, "status": "running", "argv": argv}
        self.record["cells"].append(progress)
        self.persist()
        environment = os.environ.copy()
        chosen_source = self.args.external_source if dst else self.args.original_source
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(chosen_source / "src"),
                str(chosen_source),
                environment.get("PYTHONPATH", ""),
            )
        )
        with (self.cells / (label + ".log")).open("w") as log:
            self.invoke(
                argv,
                cwd=chosen_source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=14400,
                check=True,
            )
        candidate, audit = self.validate(target, config, seed, lr, horizon)
        progress.update(
            status="completed",
            candidate=candidate,
            raw_audit=audit,
            raw_files=policy.files(target),
            origin={"phase_id": self.record["phase_id"], "output": str(self.output)},
        )
        if seed == 17:
            write(
                completed,
                {
                    "identity": self.identity,
                    "configuration": config,
                    **{
                        k: progress[k]
                        for k in ("candidate", "raw_audit", "raw_files", "origin")
                    },
                },
                immutable=True,
            )
        self.persist()
        return candidate, target

    def retune(self, control=False):
        self.unsealed()
        plan = self.plan(self.args.width)
        require(
            plan["horizon"] == 20 and (not control or plan["fresh20_controls"]),
            "no fresh20 retune required for this group",
        )
        if control:
            config = next(
                c
                for c in control_configurations(self.args.width)
                if c["kind"] == self.args.kind and c["divisor"] == self.args.divisor
            )
            original = read(self.state / "original-widths" / f"{self.args.width}.json")[
                "frozen"
            ]
            base = next(
                c for c in original["configurations"] if c["id"] == config["id"]
            )["base_tuning_candidates"]
        else:
            config = policy.configuration(
                self.args.width, self.args.kind, self.args.divisor
            )
            base = read(
                self.state
                / "imported-hpo"
                / str(self.args.width)
                / (config["id"] + ".json")
            )["selection"]["candidates"]
        selection_path = (
            self.state
            / ("control-retuning" if control else "retuning")
            / str(self.args.width)
            / (config["id"] + ".json")
        )
        if selection_path.exists():
            retained = read(selection_path)
            require(
                retained["identity"] == self.identity
                and retained["configuration"] == config
                and retained["candidate_lrs"] == [r["lr"] for r in top_two(base)]
                and retained["selected"] == policy.select(retained["candidates"])
                and all(r["epochs"] == 20 for r in retained["candidates"]),
                "retained fresh20 selection changed",
            )
            write(self.output / "retuned-selection.json", retained)
            return retained
        candidates = [self.cell(config, 17, r["lr"], 20)[0] for r in top_two(base)]
        record = {
            "identity": self.identity,
            "configuration": config,
            "horizon": 20,
            "fresh_from_initialization": True,
            "candidate_lrs": [r["lr"] for r in top_two(base)],
            "candidates": candidates,
            "selected": policy.select(candidates),
            "admitted_cells": [
                {
                    "label": r["label"],
                    "candidate": comparable(r["candidate"]),
                    "files": r["raw_files"],
                    "origin": r["origin"],
                    "raw_audit": r["raw_audit"],
                }
                for r in self.record["cells"]
            ],
            "raw_phase_proof": {
                "phase_id": self.record["phase_id"],
                "output": str(self.output),
            },
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        write(
            self.state
            / ("control-retuning" if control else "retuning")
            / str(self.args.width)
            / (config["id"] + ".json"),
            record,
            immutable=True,
        )
        write(self.output / "retuned-selection.json", record)
        return record

    def freeze_width(self):
        self.unsealed()
        width = self.args.width
        plan = self.plan(width)
        dst_rows, controls = [], []
        for cfg in policy.configurations(width):
            if plan["horizon"] == 20:
                selected = read(
                    self.state / "retuning" / str(width) / (cfg["id"] + ".json")
                )
                require(
                    selected["identity"] == self.identity and selected["horizon"] == 20,
                    "retune identity differs",
                )
                candidate = selected["selected"]
            else:
                candidate = read(
                    self.state / "imported-hpo" / str(width) / (cfg["id"] + ".json")
                )["selection"]["selected"]
            dst_rows.append(
                {
                    **cfg,
                    "lr": candidate["lr"],
                    "horizon": plan["horizon"],
                    "selected_development": candidate,
                }
            )
        original = read(self.state / "original-widths" / f"{width}.json")["frozen"]
        for cfg in control_configurations(width):
            if plan["fresh20_controls"]:
                selection = read(
                    self.state / "control-retuning" / str(width) / (cfg["id"] + ".json")
                )
                require(
                    selection["identity"] == self.identity,
                    "control retune identity differs",
                )
                candidate = selection["selected"]
            else:
                candidate = next(
                    c for c in original["configurations"] if c["id"] == cfg["id"]
                )["selected_development"]
            controls.append(
                {
                    **cfg,
                    "lr": candidate["lr"],
                    "horizon": plan["horizon"],
                    "reuse": not plan["fresh20_controls"],
                    "selected_development": candidate,
                }
            )
        frozen = {
            "schema_version": 1,
            "identity": self.identity,
            "width": width,
            "horizon": plan["horizon"],
            "plan_sha256": file_sha(self.state / "plans" / f"{width}.json"),
            "dst_configurations": dst_rows,
            "control_configurations": controls,
            "confirmation_seeds": list(policy.SEEDS),
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        write(self.state / "frozen" / f"{width}.json", frozen, immutable=True)
        write(self.output / "frozen-width.json", frozen)
        return frozen

    def retained(self, directory):
        record = read(directory / "complete.json")
        require(
            record["importer_identity"] == self.identity,
            "retained importer identity differs",
        )
        for name in FILES:
            require(
                file_sha(directory / name) == record["files"][name],
                "retained selected model changed",
            )
        return record

    def retain(
        self,
        source,
        destination,
        config,
        seed,
        candidate,
        producer,
        provenance,
        *,
        reuse=False,
    ):
        record = {
            "configuration": config,
            "seed": seed,
            "selected": comparable(candidate),
            "producer_identity": producer,
            "importer_identity": self.identity,
            "files": {name: file_sha(source / name) for name in FILES},
            "provenance": provenance,
            "reuse": reuse,
        }
        if destination.exists():
            require(
                self.retained(destination) == record,
                "refusing to change retained model",
            )
            return record
        pending = self.state / "pending" / uuid.uuid4().hex
        pending.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(source / name, pending / name)
        write(pending / "complete.json", record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending.rename(destination)
        return record

    def confirm(self, controls=False):
        self.unsealed()
        self.qualified()
        width, seed = self.args.width, self.args.seed
        require(seed in policy.SEEDS, "declared paired seed required")
        frozen = read(self.state / "frozen" / f"{width}.json")
        require(frozen["identity"] == self.identity, "width freeze identity differs")
        rows = frozen["control_configurations" if controls else "dst_configurations"]
        if controls:
            require(
                all(not row["reuse"] for row in rows),
                "reuse original controls via import-original-controls",
            )
        retained = []
        for selected in rows:
            config = {k: selected[k] for k in ("id", "kind", "width", "divisor")}
            directory = self.state / "confirmed" / str(width) / str(seed) / config["id"]
            if directory.exists():
                result = self.retained(directory)
                require(
                    result["selected"]["lr"] == selected["lr"]
                    and result["selected"]["epochs"] == frozen["horizon"],
                    "retained model differs from frozen LR/horizon",
                )
            else:
                candidate, source = self.cell(
                    config, seed, selected["lr"], frozen["horizon"]
                )
                result = self.retain(
                    source,
                    directory,
                    config,
                    seed,
                    candidate,
                    self.identity,
                    {
                        "phase_id": self.record["phase_id"],
                        "output": str(self.output),
                        "raw_audit": self.record["cells"][-1]["raw_audit"],
                    },
                )
            retained.append(result)
        write(self.output / "confirmed.json", retained)
        return retained

    def import_original_controls(self):
        self.unsealed()
        records = []
        for descriptor in self.index()["original_confirm_jobs"]:
            width, seed = descriptor["width"], descriptor["seed"]
            require(seed in policy.SEEDS, "undeclared original seed")
            original = read(self.state / "original-widths" / f"{width}.json")["frozen"]
            job = self.job(descriptor)
            require(
                job.phase["stage"] == "confirm"
                and job.phase["width"] == width
                and job.phase["seed"] == seed,
                "wrong original confirm phase",
            )
            admitted = self.original_cells(
                job, control_configurations(width), seed=seed
            )
            require(
                len(admitted) == 5,
                "all five actual original confirmation models required",
            )
            for selected in original["configurations"]:
                config = {k: selected[k] for k in ("id", "kind", "width", "divisor")}
                key = (config["id"], original["horizon"], selected["lr"])
                require(
                    key in admitted,
                    "original model differs from actual frozen LR/horizon",
                )
                candidate, source, audit = admitted[key]
                # Keep original proof30 separate even if comparison needs fresh20.
                destination = (
                    self.state
                    / "original-confirmed"
                    / str(width)
                    / str(seed)
                    / config["id"]
                )
                retained = self.retain(
                    source,
                    destination,
                    config,
                    seed,
                    candidate,
                    self.original_identity,
                    {"pool": job.proof, "raw_audit": audit},
                    reuse=True,
                )
                records.append(retained)
                comparison = read(self.state / "frozen" / f"{width}.json")
                reuse = next(
                    c
                    for c in comparison["control_configurations"]
                    if c["id"] == config["id"]
                )
                if reuse["reuse"]:
                    require(
                        reuse["lr"] == candidate["lr"]
                        and comparison["horizon"] == original["horizon"],
                        "comparison reuse differs from original freeze",
                    )
                    self.retain(
                        source,
                        self.state
                        / "confirmed"
                        / str(width)
                        / str(seed)
                        / config["id"],
                        config,
                        seed,
                        candidate,
                        self.original_identity,
                        {"pool": job.proof, "raw_audit": audit},
                        reuse=True,
                    )
        write(self.output / "imported-original-controls.json", records)
        return records

    def global_freeze(self):
        self.qualified()
        original_qualification = read(self.state / "original-qualification.json")
        require(
            original_qualification["status"] == "passed"
            and original_qualification["identity"] == self.original_identity
            and original_qualification["runtime_fingerprint"]
            == self.identity["runtime_fingerprint"],
            "global gate requires verified original qualification",
        )
        original_rows, rows = [], []
        paired_streams = {}
        for width in policy.WIDTHS:
            frozen = read(self.state / "frozen" / f"{width}.json")
            require(
                frozen["identity"] == self.identity
                and frozen["plan_sha256"]
                == file_sha(self.state / "plans" / f"{width}.json"),
                "width freeze binding changed",
            )
            original = read(self.state / "original-widths" / f"{width}.json")["frozen"]
            for seed in policy.SEEDS:
                for cfg in original["configurations"]:
                    config = {k: cfg[k] for k in ("id", "kind", "width", "divisor")}
                    old = self.retained(
                        self.state
                        / "original-confirmed"
                        / str(width)
                        / str(seed)
                        / cfg["id"]
                    )
                    require(
                        old["producer_identity"] == self.original_identity
                        and old["selected"]["lr"] == cfg["lr"]
                        and old["selected"]["epochs"] == original["horizon"],
                        "original30 proof selection differs",
                    )
                    original_rows.append(
                        {
                            "configuration": config,
                            "seed": seed,
                            "lr": cfg["lr"],
                            "horizon": original["horizon"],
                            "best_epoch": old["selected"]["best_epoch"],
                            "checkpoint_sha256": old["files"]["best-dev.pt"],
                            "best_metadata_sha256": old["files"]["best-dev.json"],
                            "learning_sha256": old["files"]["learning.json"],
                            # Exact original975 semantic path; proof does not mutate primary state.
                            "relative_directory": f"confirmed/{width}/{seed}/{cfg['id']}",
                        }
                    )
                for cfg in (
                    frozen["dst_configurations"] + frozen["control_configurations"]
                ):
                    directory = (
                        self.state / "confirmed" / str(width) / str(seed) / cfg["id"]
                    )
                    record = self.retained(directory)
                    selected = record["selected"]
                    audit = record["provenance"]["raw_audit"]
                    stream = {
                        "ordinary_initialization": audit["ordinary_initialization"],
                        "data_orders": audit["data_orders"],
                    }
                    stream_key = (width, seed)
                    require(
                        stream_key not in paired_streams
                        or paired_streams[stream_key] == stream,
                        "comparison models differ in ordinary initialization or minibatch order",
                    )
                    paired_streams[stream_key] = stream
                    require(
                        selected["lr"] == cfg["lr"]
                        and selected["epochs"] == frozen["horizon"],
                        "global model differs from width freeze",
                    )
                    dst = cfg["kind"] in policy.METHODS
                    require(
                        record["reuse"] == (False if dst else cfg["reuse"]),
                        "global reuse policy differs",
                    )
                    rows.append(
                        {
                            "id": f"N{width}-{cfg['id']}-seed{seed}",
                            "role": "dst" if dst else "control",
                            **{
                                k: cfg[k]
                                for k in ("kind", "width", "divisor", "lr", "horizon")
                            },
                            "seed": seed,
                            "best_epoch": selected["best_epoch"],
                            "checkpoint_sha256": record["files"]["best-dev.pt"],
                            "best_metadata_sha256": record["files"]["best-dev.json"],
                            "learning_sha256": record["files"]["learning.json"],
                            "relative_directory": directory.relative_to(
                                self.state
                            ).as_posix(),
                            "checkpoint_file": "best-dev.pt",
                            "best_metadata_file": "best-dev.json",
                            "learning_file": "learning.json",
                            "source_commit": policy.EXTERNAL_COMMIT
                            if dst
                            else policy.ORIGINAL_COMMIT,
                            "producer_identity": record["producer_identity"],
                            "reuse": record["reuse"],
                            "provenance": record["provenance"],
                        }
                    )
        proof = {
            "schema_version": 1,
            "identity": self.original_identity,
            "configurations": original_rows,
            "configuration_count": 30,
            "confirmation_partition": [50000, 100000],
            "test_evaluated": False,
        }
        proof_path = self.state / "original-confirmation-freeze-proof.json"
        write(proof_path, proof, immutable=True)
        freeze = {
            "schema_version": 1,
            "identity": self.identity,
            "dataset_sha256": policy.DATA_SHA256,
            "configuration_count": 78,
            "dst_count": 48,
            "control_count": 30,
            "comparison_horizons": {
                str(n): read(self.state / "frozen" / f"{n}.json")["horizon"]
                for n in policy.WIDTHS
            },
            "source_manifest_sha256": {
                policy.EXTERNAL_COMMIT: self.identity[
                    "external_source_manifest_sha256"
                ],
                policy.ORIGINAL_COMMIT: self.identity[
                    "original_source_manifest_sha256"
                ],
            },
            "original_confirmation_freeze_file": proof_path.relative_to(
                self.state
            ).as_posix(),
            "original_confirmation_freeze_kind": "comparison-owned-proof",
            "original_confirmation_freeze_sha256": file_sha(proof_path),
            "original_confirmation_freeze_canonical_sha256": json_sha(proof),
            "runtime_fingerprint": self.identity["runtime_fingerprint"],
            "configurations": rows,
            "confirmation_partition": [50000, 100000],
            "test_evaluated": False,
        }
        # Shared assessor performs all78 file/source/family checks without torch or tail access.
        from .higgs_dst_assessment import audit_freeze

        candidate = self.output / "confirmation-freeze.json"
        write(candidate, freeze)
        audit_freeze(
            candidate,
            file_sha(candidate),
            self.state,
            self.args.data,
            self.args.source_index,
        )
        write(self.state / "confirmation-freeze.json", freeze, immutable=True)
        write(
            self.output / "freeze-receipt.json",
            {"sha256": file_sha(candidate), "canonical_sha256": json_sha(freeze)},
        )
        return freeze


def run(args, invoke=None):
    require(args.stage in STAGES, "undeclared phase")
    campaign = Campaign(args, invoke)
    try:
        method = {
            "retune-control": lambda: campaign.retune(True),
            "confirm-controls": lambda: campaign.confirm(True),
        }.get(args.stage)
        result = (
            method() if method else getattr(campaign, args.stage.replace("-", "_"))()
        )
        campaign.record["status"] = "completed"
        return result
    except BaseException as error:
        campaign.record.update(
            status="failed", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        campaign.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    for name in (
        "external-source",
        "original-source",
        "inputs",
        "data",
        "output",
        "campaign-state",
        "identity-file",
        "source-index",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--import-index", type=Path)
    parser.add_argument("--offline-import", action="store_true")
    parser.add_argument("--width", type=int, choices=policy.WIDTHS)
    parser.add_argument("--kind", choices=(*policy.METHODS, "dense", "cst"))
    parser.add_argument("--divisor", type=int, choices=policy.DIVISORS)
    parser.add_argument("--seed", type=int, choices=policy.SEEDS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
