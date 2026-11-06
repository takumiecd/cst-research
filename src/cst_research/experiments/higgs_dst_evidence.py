"""Receipt-bound evidence admission for the second, separately frozen consumer.

Only regular, hash-bound artifacts are read. Torch is imported lazily, after the
source and outer pool archives have passed their stdlib checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import re
import sys
import tarfile
from pathlib import Path

from .higgs_dst_comparison import file_sha, read
from .higgs_dst_comparison_driver import inventory, safe_path

STAGE1_COMMIT = "22f2810b1bc51a599e51c8936acac310ca9455ef"
STAGE1_MANIFEST_SHA = "8e0b689790e387bb7adcadc6723905ed210befce3105735df6bf1f16f8b126a6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def archive_files(path, expected):
    """Read every archive member; reject aliases, links, duplicates and extras."""
    seen = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            safe_path(Path("/unused"), member.name.rstrip("/"))
            if member.isdir():
                continue
            require(
                member.isfile() and member.name not in seen and member.name in expected,
                "unsafe or unexpected archive member",
            )
            seen.add(member.name)
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            require(
                digest.hexdigest() == expected[member.name],
                "archive member hash mismatch",
            )
    require(seen == set(expected), "archive inventory incomplete")


def verify_recovery_provenance(directory, spec, receipt, execution):
    """Admit only the two recorded supervisor recovery schemas, without mutation.

    A fresh-proxy recovery has no recorded pool job status. Its intent and local
    termination records authenticate that distinct path; they do not invent an
    ACK or turn local client cleanup into a remote computation result.
    """
    directory = Path(directory)
    require(not directory.is_symlink(), "symlink recovery directory")
    job_id = spec["id"]
    require(
        re.fullmatch(r"l4job-[0-9a-f]{32}", job_id)
        and execution["id"] == job_id
        and execution["returncode"] == 0
        and execution["timed_out"] is False,
        "recovery execution identity differs",
    )
    if (
        directory.name == job_id
        and receipt.get("normal_pool_acknowledgment") is not False
    ):
        return {"kind": "normal-pool-receipt", "job_id": job_id}
    require(
        receipt.get("normal_pool_acknowledgment") is False,
        "renamed job requires explicit supervisor rescue receipt",
    )
    mode = receipt.get("recovery_mode")
    if "archive_bytes" in receipt:
        require(
            type(receipt["archive_bytes"]) is int
            and receipt["archive_bytes"]
            == (directory / "results.tar.gz").stat().st_size,
            "recovery archive byte count differs",
        )
    if mode == "supervisor-owned read-only result recovery":
        require(
            receipt.get("original_pool_job_status") in ("running", "failed", "queued"),
            "legacy supervisor rescue receipt lacks recorded pool status",
        )
        return {
            "kind": "supervisor-read-only-recovery",
            "job_id": job_id,
            "original_pool_job_status": receipt["original_pool_job_status"],
        }
    require(
        mode == "supervisor-owned SDK fresh-proxy read-only recovery"
        and receipt.get("credentials_scope")
        == "memory only; no cached credentials, pool database or slot state changes"
        and type(receipt.get("archive_bytes")) is int,
        "unsupported supervisor rescue receipt schema",
    )
    sidecars = {}
    for name in (
        "recovery-intent.json",
        "local-client-termination-after-full-verification.json",
    ):
        path = safe_path(directory, name)
        require(
            path.is_file() and not path.is_symlink(),
            "SDK recovery sidecar missing or unsafe",
        )
        sidecars[name] = {"sha256": file_sha(path), "record": read(path)}
    intent = sidecars["recovery-intent.json"]["record"]
    termination = sidecars["local-client-termination-after-full-verification.json"][
        "record"
    ]
    require(
        intent.get("mode")
        == "sole-supervisor-owned read-only SDK ContentsClient archive recovery with fresh proxy credentials only in memory"
        and intent.get("job") == job_id
        and intent.get("slot") == receipt.get("slot")
        and type(intent.get("slot")) is int
        and intent["slot"] > 0
        and intent.get("session") == receipt.get("session")
        and isinstance(intent.get("session"), str)
        and intent["session"]
        and intent.get("source_archive_sha256")
        == spec["source_sha256"]
        == execution["source_sha256"]
        and intent.get("cache_or_pool_state_modified") is False,
        "SDK recovery intent identity or no-mutation provenance differs",
    )
    require(
        type(intent.get("local_ack_client_pid")) is int
        and intent["local_ack_client_pid"] > 0
        and intent.get("local_ack_client_state")
        == "SIGSTOP held intentionally until full result verification"
        and termination.get("job") == job_id
        and termination.get("local_client_pid") == intent["local_ack_client_pid"]
        and type(termination.get("supervisor_pid")) is int
        and termination["supervisor_pid"] > 0
        and isinstance(termination.get("prior_state"), str)
        and termination["prior_state"].startswith("T")
        and termination.get("remote_compute_completed_and_verified") is True
        and termination.get("pool_or_session_state_manually_modified") is False
        and termination.get("actions")
        == [
            "one SIGTERM to paused owned local ACK client",
            "SIGCONT to deliver pending termination and return control to existing supervisor cleanup",
        ],
        "SDK recovery local termination identity or no-mutation provenance differs",
    )
    # The physical directory may be staged elsewhere. Preserve the recorded
    # locator rather than treating that machine-specific path as job identity.
    require(
        isinstance(termination.get("verified_result_directory"), str)
        and Path(termination["verified_result_directory"]).is_absolute()
        and isinstance(termination.get("utc"), str)
        and termination["utc"],
        "SDK recovery termination locator or timestamp missing",
    )
    return {
        "kind": "supervisor-sdk-fresh-proxy-recovery",
        "job_id": job_id,
        "original_pool_job_status": receipt.get("original_pool_job_status"),
        "normal_pool_acknowledgment": False,
        "sidecars": sidecars,
    }


class PoolJob:
    """Completed downloaded pool job, verified against a trusted source inventory."""

    def __init__(
        self,
        directory,
        source_manifest,
        driver_name,
        *,
        manifest_sha=None,
        original_qualification=False,
    ):
        self.directory = Path(directory).resolve()
        require(not Path(directory).is_symlink(), "symlink job directory")
        self.spec = read(self.directory / "spec.json")
        self.receipt = read(self.directory / "receipt.json")
        self.results = self.directory / "results"
        self.artifacts = self.results / "artifacts"
        result = read(self.results / "result.json")
        require(
            result["id"] == self.spec["id"]
            and re.fullmatch(r"l4job-[0-9a-f]{32}", self.spec["id"])
            and result["returncode"] == 0
            and result["timed_out"] is False,
            "pool execution incomplete",
        )
        recovery_provenance = verify_recovery_provenance(
            self.directory, self.spec, self.receipt, result
        )
        require(
            file_sha(self.directory / "source.tar.gz")
            == self.spec["source_sha256"]
            == result["source_sha256"],
            "pool source archive changed",
        )
        require(
            file_sha(self.directory / "results.tar.gz")
            == self.receipt["archive_sha256"],
            "pool result receipt mismatch",
        )
        manifest = read(self.results / "manifest.json")
        archive_files(
            self.directory / "results.tar.gz",
            manifest | {"manifest.json": file_sha(self.results / "manifest.json")},
        )
        require(
            set(inventory(self.results)) == set(manifest) | {"manifest.json"},
            "unbound local pool files",
        )
        for name, digest in manifest.items():
            require(
                file_sha(safe_path(self.results, name)) == digest,
                "local pool artifact differs",
            )
        archive_files(self.directory / "source.tar.gz", self.spec["source_files"])
        trusted = read(source_manifest)
        if manifest_sha:
            require(
                file_sha(source_manifest) == manifest_sha,
                "trusted source manifest changed",
            )
        tracked = {
            n: d
            for n, d in self.spec["source_files"].items()
            if not n.startswith("research-inputs/") and n != "__pool_driver__.py"
        }
        require(
            tracked == trusted["files"],
            "pool source differs from trusted producer inventory",
        )
        require(
            self.spec["source_files"].get("__pool_driver__.py")
            == trusted["files"][driver_name],
            "executed driver differs from tracked producer",
        )
        phase_path = self.artifacts / "phase.json"
        if not phase_path.exists() and original_qualification:
            phase_path = self.artifacts / "qualification.json"
            original = read(phase_path)
            bootstrap = read(self.artifacts / "bootstrap-source-verification.json")
            require(
                original.get("stage") == "qualify"
                and original.get("status") == "passed"
                and bootstrap.get("phase") == "qualify"
                and original["test_evaluated"] is False,
                "original qualification phase incomplete",
            )
            self.phase = {
                "stage": "qualify",
                "status": "completed",
                "identity": original["identity"],
                "confirmation_evaluated": False,
                "test_evaluated": False,
            }
        else:
            self.phase = read(phase_path)
        status = read(self.artifacts / "bootstrap-status.json")
        require(
            self.phase["status"] == "completed"
            and status["returncode"] == 0
            and status["failure"] is None,
            "producer phase or bootstrap incomplete",
        )
        require(
            self.phase["confirmation_evaluated"] is False
            and self.phase["test_evaluated"] is False,
            "development/confirmation producer opened heldout data",
        )
        self.proof = {
            "job_id": self.spec["id"],
            "directory": str(self.directory),
            "source_archive_sha256": self.spec["source_sha256"],
            "results_archive_sha256": self.receipt["archive_sha256"],
            "result_manifest_sha256": file_sha(self.results / "manifest.json"),
            "producer_source_manifest_sha256": file_sha(source_manifest),
            "phase_file": phase_path.name,
            "phase_sha256": file_sha(phase_path),
            "receipt": self.receipt,
            "recovery_provenance": recovery_provenance,
        }

    def payload(self, name):
        with tarfile.open(self.directory / "source.tar.gz", "r:gz") as archive:
            return archive.extractfile("research-inputs/" + name).read()

    def require_payload(self, name, digest):
        require(
            hashlib.sha256(self.payload(name)).hexdigest() == digest,
            "producer input payload differs: " + name,
        )


def load_module(source, relative, name):
    """Caller must verify all source bytes before calling this function."""
    spec = importlib.util.spec_from_file_location(name, Path(source) / relative)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def tensor_tools(source):
    return load_module(
        source, "benchmarks/training/higgs_verify.py", "_dst_admitted_verify"
    )


def compare_fields(actual, expected, producer, tools):
    """Preserve cross-version scalar clock dtype differences; reject all others."""
    import torch

    require(set(actual) == set(expected), "retained tensor inventory differs")
    differences = {}
    for name, value in actual.items():
        observed = tools.tensor_hash(value)
        if observed != expected[name]:
            require(
                name.endswith("/step")
                and value.numel() == 1
                and str(torch.__version__) != producer,
                "retained non-clock tensor hash differs",
            )
            differences[name] = {"producer": expected[name], "reader": observed}
    require(not tools.finite_fields(actual), "nonfinite retained tensors")
    return differences


def sparse(model, optimizer, clock, divisor, tools):
    import torch

    mask, weight = model["2.mask"], model["2.weight"]
    n, active = len(mask), 4 * (len(mask) ** 2 // divisor)
    counts = model["2.row_active_counts"]
    require(
        mask.dtype == torch.bool
        and mask.shape == weight.shape == (n, n)
        and int(mask.sum()) == active
        and torch.equal(mask.sum(1), counts)
        and set(counts.tolist()) <= {active // n, active // n + 1},
        "mask exact-count/row balance failed",
    )
    require(bool((weight[~mask] == 0).all()), "inactive weights nonzero")
    require(
        not tools.finite_fields(dict(tools.tensor_fields(model))), "nonfinite model"
    )
    if optimizer:
        require(len(optimizer["state"]) == 5, "five Adam states required")
        middle = optimizer["state"][optimizer["param_groups"][1]["params"][0]]
        require(
            all(bool((middle[k][~mask] == 0).all()) for k in ("exp_avg", "exp_avg_sq")),
            "inactive moments nonzero",
        )
        require(
            all(float(v["step"]) == clock for v in optimizer["state"].values()),
            "Adam clock differs",
        )


def audit_raw_cell(directory, source, *, dst=True, data=None):
    """Inspect actual retained model, optimizer and gradient tensors on CPU.

    Event logs are checked as a chain. Intermediate event moments are not stored
    by the producer and are explicitly not independently verified here.
    """
    import torch

    directory = Path(directory)
    tools = tensor_tools(source)
    learning, best = (
        read(directory / "learning.json"),
        read(directory / "best-dev.json"),
    )
    epochs = learning["epochs"]
    selected = epochs[best["epoch"] - 1]
    differences, records = {}, []
    initial = torch.load(
        directory / "initialization.pt", map_location="cpu", weights_only=True
    )
    require(
        file_sha(directory / "initialization.pt") == learning["initialization_sha256"],
        "raw initialization changed",
    )
    require(
        all(
            initial["args"][k] == learning[k]
            for k in ("kind", "width", "divisor", "seed", "lr")
        )
        and initial["args"]["epochs"] == len(epochs),
        "raw initial configuration differs",
    )
    ordinary = {
        k: tools.tensor_hash(initial["model"][k])
        for k in ("0.weight", "0.bias", "4.weight", "4.bias")
    }
    if dst:
        sparse(initial["model"], None, 0, learning["divisor"], tools)
        mask = initial["model"]["2.mask"]
        bounds = (
            initial["model"]["2.row_active_counts"]
            .double()
            .rsqrt()[:, None]
            .expand_as(mask)
        )
        require(
            bool((initial["model"]["2.weight"].abs()[mask] <= bounds[mask]).all()),
            "density compensated initialization bounds failed",
        )
    events = learning.get("topology_events", [])
    if dst:
        previous = tools.tensor_hash(initial["model"]["2.mask"])["sha256"]
        for index, event in enumerate(events):
            require(
                event["topology_before_sha256"] == previous
                and all(event["checks"].values()),
                "event hash chain or producer invariant failed",
            )
            require(
                event["optimizer_updates"] == event["step"] - index - 1,
                "event optimizer clock differs",
            )
            require(
                event["grown_slots"] == event["pruned_before_growth"]
                and event["survived"] + event["grown_slots"]
                == 4 * (learning["width"] ** 2 // learning["divisor"])
                and event["dropped"] == event["grown"]
                and event["grown_slots"]
                == event["grown"] + event["pruned_then_regrown"],
                "event churn accounting differs",
            )
            expected = (
                0.3
                * (
                    1
                    + math.cos(
                        math.pi * event["step"] / int(0.75 * len(epochs) * 25000)
                    )
                )
                / 2
            )
            require(
                abs(event["drop_fraction"] - expected) < 1e-14, "event cosine differs"
            )
            previous = event["topology_after_sha256"]
    for filename, row in [("best-dev.pt", selected), ("final.pt", epochs[-1])]:
        path = directory / filename
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(
            all(
                state["configuration"][k] == learning[k]
                for k in ("kind", "width", "divisor", "seed", "lr", "dataset_sha256")
            )
            and state["configuration"]["epochs"] == len(epochs),
            "raw checkpoint configuration differs",
        )
        fields = dict(
            tools.tensor_fields(
                {"model": state["model"], "optimizer": state["optimizer"]}
            )
        )
        differences[filename] = compare_fields(
            fields, row["state"], learning["torch"], tools
        )
        differences[filename + ":gradients"] = compare_fields(
            state["gradients"], row["gradients"], learning["torch"], tools
        )
        require(
            state["epoch"] == row["epoch"]
            and state["updates"] == row["updates"]
            and state["development"] == row["development"],
            "raw checkpoint metadata differs",
        )
        if dst:
            prefix = [e for e in events if e["step"] <= row["updates"]]
            clock = row["updates"] - len(prefix)
            require(
                state["topology_events"] == prefix
                and state["actual_optimizer_updates"] == clock
                and state["minibatch_step"] == row["updates"]
                and state["data_examples_seen"] == 4 * row["updates"],
                "raw DST counters differ",
            )
            sparse(
                state["model"], state["optimizer"], clock, learning["divisor"], tools
            )
            require(
                bool(
                    (
                        state["gradients"]["2.weight"][~state["model"]["2.mask"]] == 0
                    ).all()
                ),
                "inactive gradient nonzero",
            )
            require(
                tools.tensor_hash(state["dense_growth_gradient"])["sha256"]
                == row["dense_growth_gradient_sha256"],
                "growth gradient differs",
            )
            if filename == "final.pt":
                require(
                    previous == tools.tensor_hash(state["model"]["2.mask"])["sha256"],
                    "final topology differs",
                )
        else:
            require(
                all(
                    float(s["step"]) == row["updates"]
                    for s in state["optimizer"]["state"].values()
                ),
                "original Adam clock differs",
            )
        records.append(
            {"file": filename, "sha256": file_sha(path), "epoch": row["epoch"]}
        )
    prediction = None
    if data:
        import numpy as np

        from .higgs_dst_assessment import prediction_metrics

        with np.load(data, allow_pickle=False) as dataset:
            labels = dataset["y_val"][:50000].copy()
        with np.load(
            directory / "best-dev-predictions.npz", allow_pickle=False
        ) as values:
            require(
                np.array_equal(values["labels"], labels),
                "development prediction labels differ",
            )
            prediction = prediction_metrics(values["logits"], labels)
        require(
            file_sha(directory / "best-dev-predictions.npz")
            == best["predictions_sha256"],
            "development prediction digest differs",
        )
        require(
            all(
                abs(prediction[k] - best["development"][k]) <= 2e-7
                for k in ("loss", "accuracy", "roc_auc")
            ),
            "development metrics differ from retained predictions",
        )
    return {
        "ordinary_initialization": ordinary,
        "data_orders": [row["order_sha256"] for row in epochs],
        "checkpoints": records,
        "reader_torch": str(torch.__version__),
        "producer_torch": learning["torch"],
        "cross_version_clock_differences": differences,
        "development_metrics": prediction,
        "intermediate_event_moments_independently_verified": False,
    }


def audit_raw_qualification(directory, source):
    import torch

    from .higgs_dst_comparison import validate_qualification

    directory = Path(directory)
    record = read(directory / "qualification.json")
    validate_qualification(
        directory,
        {
            "runtime_fingerprint": {
                "torch": record["torch"],
                "cuda": record["cuda"],
                "gpu": record["gpu"],
            }
        },
    )
    tools = tensor_tools(source)
    differences = []
    for oracle in record["oracles"]:
        raw = torch.load(
            directory / oracle["file"], map_location="cpu", weights_only=True
        )
        require(
            raw["width"] == oracle["width"]
            and raw["active_count"] == oracle["active_count"]
            and len(raw["actual"]) == len(raw["expected"]) == 4
            and all(
                tools.byte_equal(a, b) for a, b in zip(raw["actual"], raw["expected"])
            )
            and not tools.finite_fields(dict(tools.tensor_fields(raw))),
            "retained VJP oracle differs",
        )
    for n in (1024, 2048):
        for divisor in (1000, 100):
            for kind in ("static", "rigl"):
                roots = [
                    directory / f"N{n}-D{divisor}-{kind}-{mode}"
                    for mode in ("eager", "graph")
                ]
                logs = [read(p / "trajectory.json") for p in roots]
                require(
                    logs[0]["events"] == logs[1]["events"]
                    and logs[0]["data_orders"] == logs[1]["data_orders"],
                    "qualification trajectory orders/events differ",
                )
                initials = [
                    torch.load(
                        p / "initialization.pt", map_location="cpu", weights_only=True
                    )["model"]
                    for p in roots
                ]
                require(
                    initials[0].keys() == initials[1].keys()
                    and all(
                        tools.byte_equal(initials[0][k], initials[1][k])
                        for k in initials[0]
                    ),
                    "eager/Graph initial models differ",
                )
                for epoch in (1, 2):
                    states = [
                        torch.load(
                            p / f"epoch-{epoch}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                        for p in roots
                    ]
                    require(
                        states[0].keys() == states[1].keys()
                        and all(
                            tools.byte_equal(states[0][k], states[1][k])
                            for k in states[0]
                        ),
                        "eager/Graph raw states differ",
                    )
                    for state, log in zip(states, logs):
                        differences.append(
                            compare_fields(
                                state,
                                log["state_hashes"][epoch - 1],
                                record["torch"],
                                tools,
                            )
                        )
                        model = {
                            k.removeprefix("/model/"): v
                            for k, v in state.items()
                            if k.startswith("/model/")
                        }
                        sparse(model, None, None, divisor, tools)
                        clock = (
                            epoch * 16 - (4 if epoch == 1 else 5)
                            if kind == "rigl"
                            else epoch * 16
                        )
                        clocks = [
                            v
                            for k, v in state.items()
                            if k.startswith("/optimizer/state/") and k.endswith("/step")
                        ]
                        require(
                            len(clocks) == 5 and all(float(v) == clock for v in clocks),
                            "qualification Adam clocks differ",
                        )
                        moments = [
                            v
                            for k, v in state.items()
                            if k.startswith("/optimizer/state/")
                            and k.endswith(("/exp_avg", "/exp_avg_sq"))
                            and v.shape == model["2.mask"].shape
                        ]
                        require(
                            len(moments) == 2
                            and all(
                                bool((v[~model["2.mask"]] == 0).all()) for v in moments
                            ),
                            "qualification inactive moments nonzero",
                        )
    return {
        "raw_oracles": 9,
        "raw_epoch_pairs": 16,
        "clock_reader_differences": differences,
    }


def audit_original_qualification(directory, source, expected_identity, runtime):
    """Recheck all10 retained numerical gates and16 full eager/Graph pairs."""
    import torch

    directory = Path(directory)
    record = read(directory / "qualification.json")
    require(
        record["status"] == "passed"
        and record["identity"] == expected_identity
        and len(record["oracles"]) == 10
        and len(record["comparisons"]) == 16
        and record["test_evaluated"] is False,
        "original qualification family incomplete",
    )
    require(
        isinstance(record.get("tests"), str)
        and "passed" in record["tests"]
        and "failed" not in record["tests"]
        and "error" not in record["tests"],
        "original GPU test suite did not pass",
    )
    tools = tensor_tools(source)
    expected = {f"N{n}-D{d}" for n in (1024, 2048) for d in (1000, 500, 200, 100)} | {
        f"N{n}-quadrature31" for n in (1024, 2048)
    }
    require(
        {r["name"] for r in record["oracles"]} == expected,
        "original oracle inventory differs",
    )
    oracles = []
    for row in record["oracles"]:
        raw = torch.load(
            directory / (row["name"] + ("-vjp.pt" if "-D" in row["name"] else ".pt")),
            map_location="cpu",
            weights_only=True,
        )
        n = int(row["name"].split("-")[0][1:])
        expected_atoms = (
            n * n // int(row["name"].split("-D")[1]) if "-D" in row["name"] else 31
        )
        require(
            row["atoms"] == expected_atoms, "original all-parameter oracle size differs"
        )
        require(
            tuple(raw["X"].shape) == tuple(raw["dY"].shape) == (4, n)
            and raw["P"].shape == (row["atoms"], 4)
            and len(raw["actual"]) == len(raw["expected"]) == 3,
            "original oracle shapes differ",
        )
        for index, key in enumerate(("Y", "dX", "dP")):
            actual, reference = (
                raw["actual"][index].double(),
                raw["expected"][index].double(),
            )
            require(
                actual.shape
                == reference.shape
                == ((row["atoms"], 4) if key == "dP" else (4, n))
                and torch.isfinite(actual).all()
                and torch.isfinite(reference).all()
                and bool(
                    ((actual - reference).abs() <= 3e-5 + 3e-4 * reference.abs()).all()
                )
                and row[key]["pass"] is True,
                "original numerical tolerance gate failed",
            )
        oracles.append(row["name"])
    pairs = []
    for n in (1024, 2048):
        for d in (1000, 100):
            for kind in ("dense", "cst"):
                for epoch in (1, 2):
                    roots = [
                        directory / "cells" / f"qual-{n}-{d}-{kind}-{mode}"
                        for mode in ("eager", "graph")
                    ]
                    learning = [read(p / "learning.json") for p in roots]
                    require(
                        all(
                            r["torch"] == runtime["torch"]
                            and r["cuda"] == runtime["cuda"]
                            and r["gpu"] == "NVIDIA L4"
                            and r["TF32"] is False
                            for r in learning
                        ),
                        "original qualification runtime differs",
                    )
                    states = [
                        torch.load(
                            p / f"epoch-{epoch}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                        for p in roots
                    ]
                    fields = [dict(tools.tensor_fields(s)) for s in states]
                    require(
                        fields[0].keys() == fields[1].keys()
                        and all(
                            tools.byte_equal(fields[0][k], fields[1][k])
                            for k in fields[0]
                        )
                        and not tools.finite_fields(fields[0])
                        and all(s["updates"] == epoch * 16 for s in states),
                        "original eager/Graph raw states differ",
                    )
                    pairs.append((n, d, kind, epoch))
    require(
        all(r["all_byte_equal"] for r in record["comparisons"]),
        "original producer trajectory gate failed",
    )
    return {
        "gpu_tests": record["tests"],
        "raw_oracles": oracles,
        "raw_epoch_pairs": pairs,
        "reader_torch": str(torch.__version__),
    }
