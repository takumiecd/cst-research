"""Development-only DST qualification and learning-rate campaign consumer.

The pinned external benchmark owns model/optimizer mathematics. This module
owns experiment policy, artifact admission and persistent orchestration only.
Later stages currently fail closed pending verified common-horizon controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

EXTERNAL_COMMIT = "f37729e61dc11a57e76103aa2457c2be11c4c306"
ORIGINAL_COMMIT = "97580582ed28c4e494bf5c2656bbe94a85c544c3"
DATA_SHA256 = "62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac"
WIDTHS = (1024, 2048)
DIVISORS = (1000, 500, 200, 100)
METHODS = ("static", "rigl")
SEEDS = (29, 43, 71)
LRS = (0.0001, 0.0003, 0.001, 0.003)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(
                f"refusing to change immutable experiment record: {path.name}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def protocol():
    return {
        "version": "dst-comparison-v1-development",
        "external_source_commit": EXTERNAL_COMMIT,
        "original_control_source_commit": ORIGINAL_COMMIT,
        "dataset_sha256": DATA_SHA256,
        "widths": list(WIDTHS),
        "divisors": list(DIVISORS),
        "methods": list(METHODS),
        "development_seed": 17,
        "confirmation_seeds": list(SEEDS),
        "base_lrs": list(LRS),
        "edge": {"lower": 0.00003, "upper": 0.01, "maximum_per_group": 1},
        "base_epochs": 10,
        "extended_epochs": 20,
        "selection": "minimum development CE; earliest checkpoint epoch; lower LR on exact candidate tie",
        "late_trigger": "selected best_epoch>=9 and CE8-CE10>0.001",
        "comparison_horizon": "max(original frozen width horizon,20 if any of8 DST10ep groups triggers else10)",
        "common20": "fresh top2 LR reruns of all8 DST groups; if original10, separately retune all5 original controls top2 at20 plus fresh3seeds; original family unchanged",
        "batch_size": 4,
        "training_rows": 100000,
        "development": [0, 50000],
        "confirmation": [50000, 100000],
        "rewiring": {
            "frequency": 250,
            "initial_drop": 0.3,
            "end_fraction": 0.75,
            "order": "same-minibatch topology event replaces all-layer AdamW",
        },
        "statistics": {
            "dst_configuration_family": 16,
            "margins": {"accuracy": 0.002, "roc_auc": 0.002},
            "primary": "paired-seed one-sided endpoint t tests,df2; IUT max(endpoint p); Bonferroni16",
            "diagnostic_endpoint_family": 32,
            "original_cst_family": "unchanged separate8-familyFWER.05",
            "direct_cst_dst": "descriptive; no joint24-familyFWER claim",
        },
        "confirmation_freeze": {
            "dst_models": 48,
            "control_references": 30,
            "global_before_any_tail": True,
        },
        "confirmation_evaluated": False,
        "test_evaluated": False,
    }


def configurations(width):
    if type(width) is not int or width not in WIDTHS:
        raise ValueError("choose declared width")
    return [
        {"id": f"{kind}-{divisor}", "kind": kind, "width": width, "divisor": divisor}
        for kind in METHODS
        for divisor in DIVISORS
    ]


def configuration(width, kind, divisor):
    matches = [
        row
        for row in configurations(width)
        if row["kind"] == kind and row["divisor"] == divisor
    ]
    if len(matches) != 1:
        raise ValueError("choose one of the eight declared DST groups")
    return matches[0]


def select(candidates):
    if not candidates or len({row["lr"] for row in candidates}) != len(candidates):
        raise ValueError("nonempty unique LR candidates required")
    for row in candidates:
        if (
            not math.isfinite(row["best_development_loss"])
            or not math.isfinite(row["lr"])
            or row["lr"] <= 0
            or row["best_epoch"] < 1
        ):
            raise ValueError("invalid completed development candidate")
    return min(
        candidates,
        key=lambda row: (row["best_development_loss"], row["lr"], row["best_epoch"]),
    )


def extension(selected_lr):
    return (
        0.00003
        if selected_lr == min(LRS)
        else 0.01
        if selected_lr == max(LRS)
        else None
    )


def late(candidate):
    curve = candidate["development_losses"]
    if len(curve) != 10 or any(not math.isfinite(value) for value in curve):
        raise ValueError("late trigger requires the complete selected10epoch curve")
    return candidate["best_epoch"] >= 9 and curve[7] - curve[9] > 0.001


def comparison_horizon(original_horizon, groups, width):
    expected = {row["id"] for row in configurations(width)}
    if (
        original_horizon not in (10, 20)
        or len(groups) != 8
        or {g["configuration"]["id"] for g in groups} != expected
    ):
        raise ValueError("original horizon and all eight distinct DST groups required")
    if any(
        g["configuration"] not in configurations(width) or g["horizon"] != 10
        for g in groups
    ):
        raise ValueError("DST horizon inputs must be original10epoch groups")
    triggers = [g["configuration"]["id"] for g in groups if late(g["selected"])]
    return max(original_horizon, 20 if triggers else 10), triggers


def validate_original_width(frozen, descriptor, width):
    """Validate actual975 frozen-width JSON after its external SHA is checked.

    Descriptor binds source_commit/dataset/width, the full expected_identity,
    source-manifest/protocol SHA256s and the frozen-width file SHA256.
    It must accompany the receipt/hash-bound staged original artifact; curve-only
    horizon inference is never accepted as a substitute for the actual freeze.
    """
    if (
        descriptor.get("source_commit") != ORIGINAL_COMMIT
        or descriptor.get("dataset_sha256") != DATA_SHA256
        or descriptor.get("width") != width
    ):
        raise ValueError("original selection source/data/width mismatch")
    expected_identity = descriptor.get("expected_identity")
    if (
        not isinstance(expected_identity, dict)
        or frozen.get("identity") != expected_identity
        or set(expected_identity)
        != {
            "source_identity",
            "source_code_sha256",
            "dataset_sha256",
            "protocol_sha256",
        }
    ):
        raise ValueError("original full experiment identity differs")
    if (
        any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in expected_identity.values()
        )
        or descriptor.get("source_manifest_sha256")
        != expected_identity["source_identity"]
        or descriptor.get("protocol_sha256") != expected_identity["protocol_sha256"]
    ):
        raise ValueError("original source-manifest/protocol binding incomplete")
    if (
        frozen.get("schema_version") != 1
        or frozen.get("width") != width
        or frozen.get("horizon") not in (10, 20)
        or frozen.get("identity", {}).get("dataset_sha256") != DATA_SHA256
    ):
        raise ValueError("original frozen-width identity invalid")
    wanted = {"dense", *(f"cst-{d}" for d in DIVISORS)}
    rows = frozen.get("configurations", [])
    if (
        len(rows) != 5
        or {row["id"] for row in rows} != wanted
        or frozen.get("confirmation_seeds") != list(SEEDS)
        or frozen.get("selection_uses_development_only") is not True
        or frozen.get("confirmation_evaluated") is not False
        or frozen.get("test_evaluated") is not False
    ):
        raise ValueError("original freeze family/selection scope invalid")
    for row in rows:
        kind = "dense" if row["id"] == "dense" else "cst"
        divisor = 100 if kind == "dense" else int(row["id"].removeprefix("cst-"))
        if (
            row["kind"] != kind
            or row["width"] != width
            or row["divisor"] != divisor
            or row["horizon"] != frozen["horizon"]
            or row["lr"] != row["selected_development"]["lr"]
        ):
            raise ValueError("original frozen configuration differs")
        base = row["base_tuning_candidates"]
        base_grid = [c for c in base if c["lr"] in LRS]
        if len(base_grid) != 4 or {c["lr"] for c in base_grid} != set(LRS):
            raise ValueError("original base LR grid incomplete")
        extra = extension(select(base_grid)["lr"])
        if {c["lr"] for c in base} != set(LRS) | (
            {extra} if extra is not None else set()
        ) or len(base) != 4 + (extra is not None):
            raise ValueError("original edge extension inventory differs")
        candidates = row["horizon_candidates"]
        if frozen["horizon"] == 10:
            if candidates != base:
                raise ValueError("original10epoch candidates differ from base tuning")
        elif len(candidates) != 2 or {c["lr"] for c in candidates} != {
            c["lr"]
            for c in sorted(
                base,
                key=lambda c: (c["best_development_loss"], c["lr"], c["best_epoch"]),
            )[:2]
        }:
            raise ValueError("original20epoch top2 LR inventory differs")
        for candidate in [*base, *candidates]:
            horizon = candidate["epochs"]
            curve = candidate["development_losses"]
            if (
                horizon not in (10, 20)
                or len(curve) != horizon
                or any(not math.isfinite(x) for x in curve)
                or candidate["best_epoch"] != curve.index(min(curve)) + 1
                or candidate["best_development_loss"] != min(curve)
            ):
                raise ValueError("original selected/candidate curve budget differs")
        if (
            any(c["epochs"] != 10 for c in base)
            or any(c["epochs"] != frozen["horizon"] for c in candidates)
            or row["selected_development"] != select(candidates)
        ):
            raise ValueError("original selection/horizon tie rule differs")
    return frozen["horizon"]


def files(directory):
    result = {}
    for path in Path(directory).rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("only regular artifact files/directories accepted")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = file_sha(path)
    return result


def validate_cell(directory, config, seed, lr, epochs, identity):
    directory = Path(directory)
    learning, best = (
        read(directory / "learning.json"),
        read(directory / "best-dev.json"),
    )
    expected = {
        **{k: config[k] for k in ("width", "kind", "divisor")},
        "seed": seed,
        "lr": lr,
        "mode": "graph",
        "cohort": "higgs_dst_reference_v1",
        "training_rows": 100000,
        "development_rows": 50000,
        "batch_size": 4,
        "dataset_sha256": identity["dataset_sha256"],
        "all_layers_trainable": True,
        "TF32": False,
        "gpu": "NVIDIA L4",
        "total_updates": epochs * 25000,
        "minibatch_step": epochs * 25000,
        "data_examples_seen": epochs * 100000,
        "confirmation_evaluated": False,
        "test_evaluated": False,
    }
    if any(
        learning.get(key) != value for key, value in expected.items()
    ) or learning.get("validation_partition") != {
        "development": [0, 50000],
        "confirmation": [50000, 100000],
    }:
        raise ValueError("cell identity, scope or budget differs")
    rows = learning["epochs"]
    if len(rows) != epochs or [row["epoch"] for row in rows] != list(
        range(1, epochs + 1)
    ):
        raise ValueError("cell lacks its complete epoch curve")
    parameter_names = {"0.weight", "0.bias", "2.weight", "4.weight", "4.bias"}
    state_names = {
        "/model/" + name for name in parameter_names | {"2.mask", "2.row_active_counts"}
    } | {
        f"/optimizer/state/{i}/{name}"
        for i in range(5)
        for name in ("step", "exp_avg", "exp_avg_sq")
    }
    for row in rows:
        if (
            set(row.get("state", {})) != state_names
            or set(row.get("gradients", {})) != parameter_names
        ):
            raise ValueError("recorded complete model/Adam/gradient inventory differs")
        for value in [*row["state"].values(), *row["gradients"].values()]:
            if (
                value.get("dtype") not in ("torch.float32", "torch.int64", "torch.bool")
                or not isinstance(value.get("shape"), list)
                or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))
            ):
                raise ValueError("invalid retained FP32 state metadata")
        if any(
            value["dtype"] != "torch.float32" for value in row["gradients"].values()
        ):
            raise ValueError("gradient precision differs")
    curves = [row["development"]["loss"] for row in rows]
    if (
        any(not math.isfinite(value) for value in curves)
        or best != learning["best_development_checkpoint"]
        or best["epoch"] != curves.index(min(curves)) + 1
        or best["development"] != rows[best["epoch"] - 1]["development"]
    ):
        raise ValueError("checkpoint was not selected by dev CE/earliest tie")
    for name, expected_sha in (
        ("best-dev.pt", best["checkpoint_sha256"]),
        ("best-dev-predictions.npz", best["predictions_sha256"]),
        ("initialization.pt", learning["initialization_sha256"]),
    ):
        if file_sha(directory / name) != expected_sha:
            raise ValueError("retained selected artifact checksum differs")
    if (
        learning["final_checkpoint"] != "final.pt"
        or not (directory / "final.pt").is_file()
        or read(directory / "epochs.json") != rows
        or read(directory / "topology-events.json") != learning["topology_events"]
    ):
        raise ValueError("raw final checkpoint or curve/event copies missing")
    total = epochs * 25000
    event_steps = (
        list(range(250, int(0.75 * total), 250)) if config["kind"] == "rigl" else []
    )
    events = learning["topology_events"]
    if (
        [event["step"] for event in events] != event_steps
        or learning["actual_optimizer_updates"] != total - len(events)
        or learning["graph_replays"] != total - 8
    ):
        raise ValueError("RigL minibatch/optimizer/event counters differ")
    for epoch, row in enumerate(rows, 1):
        step = epoch * 25000
        count = sum(event <= step for event in event_steps)
        if (
            row["updates"] != step
            or row["minibatch_step"] != step
            or row["data_examples_seen"] != 4 * step
            or row["actual_optimizer_updates"] != step - count
            or row["topology_event_count"] != count
            or row["development"]["examples"] != 50000
        ):
            raise ValueError("epoch partition or optimizer counters differ")
    atom_count = config["width"] ** 2 // config["divisor"]
    if (
        learning["atom_count_matched"] != atom_count
        or learning["active_middle_weights"] != 4 * atom_count
    ):
        raise ValueError("middle active4A budget differs")
    rewiring = learning["rewiring"]
    if any(
        rewiring.get(k) != v
        for k, v in {
            "algorithm": config["kind"],
            "update_frequency": 250,
            "initial_drop_fraction": 0.3,
            "end_fraction": 0.75,
            "end_step_exclusive": int(0.75 * total),
        }.items()
    ) or "replaces all-layer AdamW" not in rewiring.get("optimizer_update_order", ""):
        raise ValueError("rewiring protocol changed")
    for key in ("torch", "cuda", "gpu"):
        if str(learning[key]) != str(identity["runtime_fingerprint"][key]):
            raise ValueError("cell runtime differs from admitted campaign")
    return {
        "lr": lr,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_development_loss": min(curves),
        "best_development": best["development"],
        "development_losses": curves,
        "checkpoint_sha256": best["checkpoint_sha256"],
        "best_metadata_sha256": file_sha(directory / "best-dev.json"),
        "learning_sha256": file_sha(directory / "learning.json"),
        "training_seconds": learning["training_seconds"],
        "adoption_seconds": learning["adoption_seconds"],
        "minibatches": total,
        "actual_optimizer_updates": total - len(events),
    }


def validate_qualification(destination, identity):
    destination = Path(destination)
    result = read(destination / "qualification.json")
    expected_oracles = {(17, 51)} | {
        (n, 4 * (n * n // d)) for n in WIDTHS for d in DIVISORS
    }
    expected_comparisons = {
        f"N{n}-D{d}-{kind}" for n in WIDTHS for d in (1000, 100) for kind in METHODS
    }
    if (
        result.get("status") != "passed"
        or len(result.get("oracles", [])) != 9
        or len(result.get("comparisons", [])) != 8
        or result.get("gpu") != "NVIDIA L4"
        or result.get("TF32") is not False
        or result.get("confirmation_evaluated") is not False
        or result.get("test_evaluated") is not False
    ):
        raise ValueError("qualification is incomplete or changed scope")
    if {
        (r["width"], r["active_count"]) for r in result["oracles"]
    } != expected_oracles or {
        r["label"] for r in result["comparisons"]
    } != expected_comparisons:
        raise ValueError(
            "qualification does not cover the declared VJP/trajectory family"
        )
    for key in ("torch", "cuda", "gpu"):
        if str(result[key]) != str(identity["runtime_fingerprint"][key]):
            raise ValueError("qualification runtime differs")
    for row in result["oracles"]:
        name = row["file"]
        path = destination / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or not row["pass"]
            or set(row["checks"]) != {"Y", "dX", "masked_dW", "dense_growth_dW"}
            or not all(row["checks"].values())
            or file_sha(path) != row["sha256"]
        ):
            raise ValueError("qualification oracle/artifact failed")
    for row in result["comparisons"]:
        if (
            not row["pass"]
            or not row["checks"]
            or not all(row["checks"].values())
            or len(row["epoch_equality"]) != 2
            or not all(e["all_byte_equal"] for e in row["epoch_equality"])
        ):
            raise ValueError("qualification trajectory gate failed")
        for mode in ("eager", "graph"):
            path = destination / (row["label"] + "-" + mode)
            if not all(
                (path / name).is_file()
                for name in (
                    "initialization.pt",
                    "epoch-1.pt",
                    "epoch-2.pt",
                    "trajectory.json",
                )
            ):
                raise ValueError("qualification lacks raw trajectory checkpoints")
    return result


class PhaseNotReady(RuntimeError):
    """A later stage lacks independently verified common-budget controls."""


def validate_phase_arguments(args):
    if args.stage == "qualify":
        if any(
            getattr(args, key) is not None
            for key in ("width", "kind", "divisor", "seed")
        ):
            raise ValueError(
                "qualification runs its fixed full family; no cell selector allowed"
            )
    elif args.stage == "tune":
        configuration(args.width, args.kind, args.divisor)
        if args.seed is not None:
            raise ValueError(
                "development tuning always uses seed17, never confirmation seeds"
            )
    elif args.stage == "freeze-width":
        configurations(args.width)
        if any(getattr(args, key) is not None for key in ("kind", "divisor", "seed")):
            raise ValueError("width freeze requires all eight methods/densities")
    elif args.stage == "confirm":
        configurations(args.width)
        if args.seed not in SEEDS or args.kind is not None or args.divisor is not None:
            raise ValueError(
                "confirmation is one width/pairedseed and all eight DST cells"
            )
    elif args.stage == "assess":
        if any(
            getattr(args, key) is not None
            for key in ("width", "kind", "divisor", "seed")
        ):
            raise ValueError("assessment requires the entire globally frozen family")
    else:
        raise ValueError("unknown campaign stage")


class Campaign:
    def __init__(self, args, invoke=None):
        validate_phase_arguments(args)
        self.args, self.identity, self.invoke = (
            args,
            read(args.identity_file),
            invoke or subprocess.run,
        )
        if (
            self.identity["dataset_sha256"] != DATA_SHA256
            or self.identity["external_source_commit"] != EXTERNAL_COMMIT
            or self.identity["protocol_sha256"] != json_sha(protocol())
            or file_sha(args.data) != DATA_SHA256
        ):
            raise ValueError("campaign source/data/protocol identity differs")
        self.state = args.campaign_state
        self.state.mkdir(parents=True, exist_ok=True)
        write(self.state / "experiment-identity.json", self.identity, immutable=True)
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.cells = self.output / "cells"
        self.cells.mkdir(exist_ok=False)
        self.record = {
            "schema_version": 1,
            "stage": args.stage,
            "status": "running",
            "phase_id": uuid.uuid4().hex,
            "identity": self.identity,
            "width": args.width,
            "kind": args.kind,
            "divisor": args.divisor,
            "seed": args.seed,
            "protocol": protocol(),
            "cells": [],
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        self.persist()

    def persist(self):
        write(self.output / "phase.json", self.record)

    def qualified(self):
        record = read(self.state / "qualification.json")
        if record.get("identity") != self.identity or record.get("status") != "passed":
            raise ValueError(
                "matching source/data/runtime has not passed qualification"
            )

    def qualify(self):
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
        result = validate_qualification(destination, self.identity)
        marker = {
            "status": "passed",
            "identity": self.identity,
            "qualification_sha256": file_sha(destination / "qualification.json"),
            "artifact_hashes": files(destination),
            "record": result,
        }
        qualification_path = self.state / "qualification.json"
        if qualification_path.exists():
            self.qualified()
            # Preserve the original passed certificate; this independent repeat
            # remains fully retained in its own phase output.
        else:
            write(qualification_path, marker, immutable=True)
        return marker

    def cell(self, config, seed, lr, epochs):
        label = f"N{config['width']}-{config['id']}-seed{seed}-lr{lr:g}-{epochs}ep"
        target, cached = self.cells / label, self.state / "retained-cells" / label
        origin = {"phase_id": self.record["phase_id"], "output": str(self.output)}
        progress = {"label": label, "status": "running", "reused": cached.exists()}
        self.record["cells"].append(progress)
        self.persist()
        if cached.exists():
            complete = read(cached / "complete.json")
            origin = complete["origin"]
            observed = files(cached)
            observed.pop("complete.json")
            if complete["identity"] != self.identity or observed != complete["files"]:
                raise ValueError("retained cell identity or bytes changed")
            target.mkdir()
            for name in complete["files"]:
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached / name, destination)
        else:
            argv = [
                sys.executable,
                "-m",
                "benchmarks.training.higgs_dst_reference",
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
                str(epochs),
                "--mode",
                "graph",
            ]
            progress["argv"] = argv
            self.persist()
            with (self.cells / (label + ".log")).open("w") as log:
                self.invoke(
                    argv,
                    cwd=self.args.external_source,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=14400,
                    check=True,
                )
        candidate = validate_cell(target, config, seed, lr, epochs, self.identity)
        if not cached.exists():
            pending = self.state / "pending" / uuid.uuid4().hex
            shutil.copytree(target, pending)
            write(
                pending / "complete.json",
                {"identity": self.identity, "files": files(target), "origin": origin},
            )
            cached.parent.mkdir(parents=True, exist_ok=True)
            pending.rename(cached)
        progress.update(status="completed", candidate=candidate, origin=origin)
        self.persist()
        return candidate

    def tune(self):
        self.qualified()
        if (self.state / "confirmation-freeze.json").exists():
            raise ValueError("cannot retune after global confirmation freeze")
        config = configuration(self.args.width, self.args.kind, self.args.divisor)
        candidates = [self.cell(config, 17, lr, 10) for lr in LRS]
        extra = extension(select(candidates)["lr"])
        if extra is not None:
            candidates.append(self.cell(config, 17, extra, 10))
        selected = select(candidates)
        selection = {
            "schema_version": 1,
            "identity": self.identity,
            "configuration": config,
            "development_seed": 17,
            "horizon": 10,
            "base_learning_rates": list(LRS),
            "edge_extension": extra,
            "evaluated_learning_rates": sorted(row["lr"] for row in candidates),
            "selected": selected,
            "candidates": candidates,
            "late_improvement": late(selected),
            "selected_lr_on_evaluated_boundary": selected["lr"]
            in (
                min(row["lr"] for row in candidates),
                max(row["lr"] for row in candidates),
            ),
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        write(
            self.state
            / "tuning"
            / str(config["width"])
            / config["id"]
            / "selection.json",
            selection,
            immutable=True,
        )
        write(self.output / "selection.json", selection)
        write(
            self.output / "hpo-export.json",
            {
                "schema_version": 1,
                "artifact_kind": "dst-development-hpo-group-v1",
                "producer_identity": self.identity,
                "configuration": config,
                "development_seed": 17,
                "horizon": 10,
                "selection_file": "selection.json",
                "selection_sha256": file_sha(self.output / "selection.json"),
                "cells": [
                    {
                        "label": item["label"],
                        "relative_directory": "cells/" + item["label"],
                        "candidate": item["candidate"],
                        "origin": item["origin"],
                        "files": files(self.cells / item["label"]),
                    }
                    for item in self.record["cells"]
                ],
                "import_requirement": "verify original pool result archive/receipt/manifest and completed phase, all cells/selection bytes, exact prior consumer/source/data/runtime/protocol; retain producer identity, never rewrite it",
                "confirmation_evaluated": False,
                "test_evaluated": False,
            },
        )
        return selection

    def later_stage(self):
        if self.args.stage == "freeze-width":
            groups = [
                read(
                    self.state
                    / "tuning"
                    / str(self.args.width)
                    / c["id"]
                    / "selection.json"
                )
                for c in configurations(self.args.width)
            ]
            if any(g["identity"] != self.identity for g in groups):
                raise ValueError("DST tuning identity differs")
            readiness = {
                "ready": False,
                "reason": "verified original frozen-width import and common-horizon control workflow required",
                "dst_groups": len(groups),
                "required_global_dst_models": 48,
                "required_global_control_references": 30,
            }
            if self.args.original_selection_index:
                index = read(self.args.original_selection_index)
                descriptor = index["widths"][str(self.args.width)]
                frozen_path = (
                    self.args.original_selection_index.parent / descriptor["file"]
                )
                if (
                    not descriptor["file"]
                    or Path(descriptor["file"]).is_absolute()
                    or ".." in Path(descriptor["file"]).parts
                    or frozen_path.is_symlink()
                    or file_sha(frozen_path) != descriptor["frozen_width_sha256"]
                ):
                    raise ValueError("original frozen-width path or hash invalid")
                manifest_name = descriptor["source_manifest_file"]
                manifest_path = (
                    self.args.original_selection_index.parent / manifest_name
                )
                if (
                    Path(manifest_name).is_absolute()
                    or ".." in Path(manifest_name).parts
                    or manifest_path.is_symlink()
                    or file_sha(manifest_path) != descriptor["source_manifest_sha256"]
                ):
                    raise ValueError("original source-manifest file/hash differs")
                original_manifest = read(manifest_path)
                if original_manifest.get("commit") != ORIGINAL_COMMIT or not isinstance(
                    original_manifest.get("files"), dict
                ):
                    raise ValueError(
                        "original source-manifest commit/inventory invalid"
                    )
                external_files = read(
                    self.args.external_source / "research-inputs/source-manifest.json"
                )["files"]
                digest = hashlib.sha256()
                for name, value in sorted(original_manifest["files"].items()):
                    if name.startswith(
                        ("src/torchcst/", "benchmarks/training/")
                    ) and Path(name).suffix in {
                        ".py",
                        ".cpp",
                        ".cu",
                        ".cuh",
                        ".h",
                        ".json",
                    }:
                        if external_files.get(name) != value:
                            raise ValueError(
                                "original runtime/benchmark source differs from pinned external bytes"
                            )
                        digest.update(name.encode() + b"\0" + value.encode() + b"\n")
                if (
                    digest.hexdigest()
                    != descriptor["expected_identity"]["source_code_sha256"]
                ):
                    raise ValueError("original code-fingerprint differs")
                original = read(frozen_path)
                original_horizon = validate_original_width(
                    original, descriptor, self.args.width
                )
                horizon, triggers = comparison_horizon(
                    original_horizon, groups, self.args.width
                )
                readiness.update(
                    original_horizon=original_horizon,
                    comparison_horizon=horizon,
                    dst_late_triggers=triggers,
                    original_controls_need_fresh20=original_horizon == 10
                    and horizon == 20,
                )
            write(self.output / "freeze-readiness.json", readiness)
        raise PhaseNotReady(
            "first delivery supports qualification+tune only; no48DST+30control freeze/evaluation is admitted"
        )


def run(args, invoke=None):
    campaign = Campaign(args, invoke)
    try:
        result = (
            campaign.qualify()
            if args.stage == "qualify"
            else campaign.tune()
            if args.stage == "tune"
            else campaign.later_stage()
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
    parser.add_argument(
        "--stage",
        choices=("qualify", "tune", "freeze-width", "confirm", "assess"),
        required=True,
    )
    for name in (
        "external-source",
        "data",
        "output",
        "campaign-state",
        "identity-file",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--width", type=int, choices=WIDTHS)
    parser.add_argument("--kind", choices=METHODS)
    parser.add_argument("--divisor", type=int, choices=DIVISORS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--original-selection-index", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
