"""Sealed HIGGS DST assessment and CPU analysis; no runtime vendoring.

Importing this module neither imports torch nor reads confirmation data. Every
78-model gate passes before evaluation workers may import a pinned benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

DST_SOURCE = "f37729e61dc11a57e76103aa2457c2be11c4c306"
CONTROL_SOURCE = "97580582ed28c4e494bf5c2656bbe94a85c544c3"
DATA_SHA256 = "62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac"
WIDTHS = (1024, 2048)
DIVISORS = (1000, 500, 200, 100)
SEEDS = (29, 43, 71)
MARGINS = {"accuracy": 0.002, "roc_auc": 0.002}
PRIMARY_CONFIGURATIONS = 16
DIAGNOSTIC_ENDPOINTS = 32


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def write(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(read(path) == value, "refusing to replace a sealed assessment record")
        return
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def safe(root, name):
    relative = PurePosixPath(name)
    require(
        isinstance(name, str)
        and name
        and "\\" not in name
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == name
        and name != ".",
        "unsafe state-relative path",
    )
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(root), "state path escapes its root")
    for current in (path, *path.parents):
        if current == root.parent:
            break
        require(not current.is_symlink(), "symlink in bound state path")
    return path


def row_key(row):
    config = row.get("configuration", row)
    return (config["width"], config["kind"], config["divisor"], row["seed"])


def expected_keys():
    dst = {
        (n, kind, d, seed)
        for n in WIDTHS
        for kind in ("static", "rigl")
        for d in DIVISORS
        for seed in SEEDS
    }
    controls = {(n, "cst", d, seed) for n in WIDTHS for d in DIVISORS for seed in SEEDS}
    controls |= {(n, "dense", 100, seed) for n in WIDTHS for seed in SEEDS}
    return dst, controls


def verify_sources(source_index, freeze):
    """Verify complete declared source bytes without importing executable code."""
    descriptors = read(source_index)["sources"]
    expected = freeze["source_manifest_sha256"]
    require(
        set(expected) == {DST_SOURCE, CONTROL_SOURCE},
        "both pinned source manifests required",
    )
    verified = {}
    for commit in (DST_SOURCE, CONTROL_SOURCE):
        descriptor = descriptors[commit]
        directory, manifest_path = (
            Path(descriptor["directory"]),
            Path(descriptor["manifest_file"]),
        )
        require(
            directory.is_dir()
            and not directory.is_symlink()
            and not manifest_path.is_symlink(),
            "invalid pinned source location",
        )
        require(
            file_sha(manifest_path) == expected[commit],
            "source manifest digest differs from global freeze",
        )
        manifest = read(manifest_path)
        require(
            manifest.get("source_commit", manifest.get("commit")) == commit,
            "source manifest commit differs",
        )
        files = manifest["files"]
        require(
            isinstance(files, dict)
            and files
            and "benchmarks/training/higgs_confirmation.py" in files,
            "incomplete source manifest",
        )
        if commit == DST_SOURCE:
            require(
                "benchmarks/training/higgs_dst_reference.py" in files,
                "DST benchmark missing from source manifest",
            )
        for name, digest in files.items():
            target = safe(directory, name)
            require(
                target.is_file() and file_sha(target) == digest,
                "pinned source member changed: " + name,
            )
        extra = {
            p.relative_to(directory).as_posix()
            for p in directory.rglob("*")
            if p.is_file()
            and not p.relative_to(directory)
            .as_posix()
            .startswith((".job-deps/", "research-inputs/"))
        }
        require(extra == set(files), "unmanifested file in pinned source directory")
        verified[commit] = {
            "directory": str(directory.resolve()),
            "manifest_file": str(manifest_path.resolve()),
            "manifest_sha256": expected[commit],
            "verified_files": len(files),
        }
    return verified


def check_row_files(row, state, data_sha):
    directory = safe(state, row["relative_directory"])
    names = {
        "checkpoint_sha256": row.get("checkpoint_file", "best-dev.pt"),
        "best_metadata_sha256": row.get("best_metadata_file", "best-dev.json"),
        "learning_sha256": row.get("learning_file", "learning.json"),
    }
    require(
        names
        == {
            "checkpoint_sha256": "best-dev.pt",
            "best_metadata_sha256": "best-dev.json",
            "learning_sha256": "learning.json",
        },
        "only selected best-development checkpoint files are admitted",
    )
    hashes = {}
    for key, name in names.items():
        path = safe(directory, name)
        require(path.is_file(), "frozen model artifact missing: " + row["id"])
        actual = file_sha(path)
        require(
            actual == row[key], "frozen selected model artifact changed: " + row["id"]
        )
        hashes[name] = actual
    best, learning = (
        read(directory / "best-dev.json"),
        read(directory / "learning.json"),
    )
    require(
        best["checkpoint_sha256"] == row["checkpoint_sha256"]
        and best["epoch"] == row["best_epoch"],
        "selected metadata disagrees with freeze",
    )
    require(
        all(
            learning.get(k) == row[k]
            for k in ("width", "kind", "divisor", "seed", "lr")
        )
        and learning["dataset_sha256"] == data_sha
        and learning["batch_size"] == 4
        and learning["training_rows"] == 100000
        and learning["development_rows"] == 50000
        and learning["confirmation_evaluated"] is False
        and learning["test_evaluated"] is False,
        "frozen training identity or development-only scope differs",
    )
    require(
        learning["validation_partition"]
        == {"development": [0, 50000], "confirmation": [50000, 100000]},
        "frozen partition differs",
    )
    epochs = learning["epochs"]
    require(
        len(epochs) == row["horizon"]
        and [e["epoch"] for e in epochs] == list(range(1, row["horizon"] + 1))
        and [e["updates"] for e in epochs]
        == [25000 * i for i in range(1, row["horizon"] + 1)],
        "frozen learning budget differs",
    )
    losses = [e["development"]["loss"] for e in epochs]
    require(
        all(math.isfinite(x) for x in losses), "nonfinite development selection curve"
    )
    selected = epochs[losses.index(min(losses))]
    require(
        row["best_epoch"] == selected["epoch"]
        and best == learning["best_development_checkpoint"]
        and best["development"] == selected["development"]
        and best["development"]["examples"] == 50000,
        "checkpoint was not selected by earliest minimum development CE",
    )
    return hashes


def audit_freeze(frozen_index, expected_sha, campaign_state, data, source_index):
    """Global CPU admission. No torch/NumPy import or confirmation access here."""
    frozen_index, state, data = Path(frozen_index), Path(campaign_state), Path(data)
    require(
        frozen_index.is_file() and file_sha(frozen_index) == expected_sha,
        "global freeze absent or sealed SHA differs",
    )
    freeze = read(frozen_index)
    require(
        freeze.get("schema_version") == 1
        and freeze.get("configuration_count") == 78
        and freeze.get("dst_count") == 48
        and freeze.get("control_count") == 30
        and freeze.get("confirmation_partition") == [50000, 100000]
        and freeze.get("test_evaluated") is False,
        "global78 freeze scope/count differs",
    )
    require(
        freeze["dataset_sha256"] == DATA_SHA256 and file_sha(data) == DATA_SHA256,
        "frozen dataset differs",
    )
    runtime = freeze.get("runtime_fingerprint")
    require(
        isinstance(runtime, dict)
        and all(
            isinstance(runtime.get(k), str) and runtime[k]
            for k in ("torch", "cuda", "gpu")
        ),
        "frozen runtime fingerprint missing/invalid",
    )
    horizons = freeze["comparison_horizons"]
    require(
        set(horizons) == {"1024", "2048"}
        and all(t in (10, 20) for t in horizons.values()),
        "comparison horizons missing/invalid",
    )
    sources = verify_sources(source_index, freeze)
    original_file = safe(state, freeze["original_confirmation_freeze_file"])
    require(
        original_file.is_file()
        and file_sha(original_file) == freeze["original_confirmation_freeze_sha256"],
        "actual original30 freeze missing/changed",
    )
    original = read(original_file)
    require(
        freeze.get("original_confirmation_freeze_kind") == "comparison-owned-proof"
        and json_sha(original)
        == freeze["original_confirmation_freeze_canonical_sha256"],
        "comparison-owned original proof differs",
    )
    require(
        original.get("configuration_count") == 30
        and original.get("confirmation_partition") == [50000, 100000]
        and original.get("test_evaluated") is False
        and original["identity"]["dataset_sha256"] == DATA_SHA256,
        "original30 freeze identity/scope differs",
    )
    original_rows = {row_key(row): row for row in original["configurations"]}
    _, control_keys = expected_keys()
    require(
        len(original["configurations"]) == 30 and set(original_rows) == control_keys,
        "original control freeze is incomplete/duplicated",
    )
    rows = freeze["configurations"]
    require(
        len(rows) == 78 and len({r["id"] for r in rows}) == 78,
        "global model IDs missing/duplicated",
    )
    observed_dst, observed_control, hashes = set(), set(), {}
    paths = set()
    for row in rows:
        key = row_key(row)
        require(
            isinstance(row["id"], str) and bool(row["id"]),
            "model IDs must be nonempty strings",
        )
        require(
            row["kind"] in ("static", "rigl", "dense", "cst")
            and row["seed"] in SEEDS
            and row["width"] in WIDTHS
            and row["divisor"] in DIVISORS
            and type(row["lr"]) in (int, float)
            and math.isfinite(row["lr"])
            and row["lr"] > 0,
            "undeclared comparison model",
        )
        role = "dst" if row["kind"] in ("static", "rigl") else "control"
        require(
            row["role"] == role
            and row["source_commit"]
            == (DST_SOURCE if role == "dst" else CONTROL_SOURCE),
            "frozen role/source mismatch",
        )
        require(
            row["horizon"] == horizons[str(row["width"])]
            and isinstance(row.get("producer_identity"), dict)
            and row["producer_identity"]
            and type(row.get("reuse", False)) is bool,
            "frozen horizon/provenance/reuse differs",
        )
        observed = observed_dst if role == "dst" else observed_control
        require(key not in observed, "duplicate model in global family")
        observed.add(key)
        require(
            row["relative_directory"] not in paths,
            "distinct frozen models alias one directory",
        )
        paths.add(row["relative_directory"])
        hashes[row["id"]] = check_row_files(row, state, DATA_SHA256)
        if row.get("reuse", False):
            require(
                role == "control" and key in original_rows,
                "only verified original controls may be reused",
            )
            old = original_rows[key]
            require(
                all(
                    row[k] == old[k]
                    for k in (
                        "checkpoint_sha256",
                        "best_metadata_sha256",
                        "learning_sha256",
                        "lr",
                        "horizon",
                        "best_epoch",
                    )
                ),
                "reuse reference differs from actual original selected model",
            )
    wanted_dst, wanted_control = expected_keys()
    require(
        observed_dst == wanted_dst and observed_control == wanted_control,
        "global48DST+30control family incomplete",
    )
    return (
        freeze,
        {
            "schema_version": 1,
            "freeze_sha256": expected_sha,
            "dataset_sha256": DATA_SHA256,
            "verified_dst_count": 48,
            "verified_control_count": 30,
            "verified_model_files": hashes,
            "source_manifests": {k: v["manifest_sha256"] for k, v in sources.items()},
            "original_confirmation_freeze_sha256": freeze[
                "original_confirmation_freeze_sha256"
            ],
        },
        sources,
    )


def prediction_metrics(logits, labels):
    import numpy as np

    require(
        logits.shape == (50000, 2)
        and logits.dtype == np.float32
        and labels.shape == (50000,)
        and labels.dtype == np.int64
        and np.isfinite(logits).all()
        and np.isin(labels, (0, 1)).all(),
        "invalid confirmation prediction arrays",
    )
    values = logits.astype(np.float64)
    scores = logits[:, 1] - logits[:, 0]
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    stops = np.r_[starts[1:], len(scores)]
    ranks = np.repeat((starts + 1 + stops) / 2, stops - starts)
    positives = int(labels.sum())
    require(0 < positives < len(labels), "confirmation requires both classes")
    auc = (ranks[labels[order] == 1].sum() - positives * (positives + 1) / 2) / (
        positives * (len(labels) - positives)
    )
    loss = (
        np.logaddexp(values[:, 0], values[:, 1])
        - values[np.arange(len(labels)), labels]
    )
    return {
        "loss": float(loss.mean()),
        "accuracy": float((logits.argmax(1) == labels).mean()),
        "roc_auc": float(auc),
        "examples": len(labels),
    }


def validate_evaluation(directory, row, freeze_sha, data_sha):
    import numpy as np

    directory = Path(directory)
    record = read(directory / "evaluation.json")
    require(
        record["freeze_sha256"] == freeze_sha
        and record["checkpoint_sha256"] == row["checkpoint_sha256"]
        and record["model_id"] == row["id"]
        and record["dataset_sha256"] == data_sha
        and record["validation_partition"] == [50000, 100000]
        and record["test_evaluated"] is False,
        "cached evaluation identity differs",
    )
    require(
        file_sha(directory / "predictions.npz") == record["predictions_sha256"],
        "cached prediction bytes changed",
    )
    with np.load(directory / "predictions.npz", allow_pickle=False) as stored:
        require(
            set(stored.files) == {"logits", "labels"},
            "prediction archive fields differ",
        )
        logits, labels = stored["logits"], stored["labels"]
        actual = prediction_metrics(logits, labels)
        require(
            hashlib.sha256(logits.tobytes()).hexdigest() == record["logits_sha256"]
            and hashlib.sha256(labels.tobytes()).hexdigest() == record["labels_sha256"],
            "prediction array digest differs",
        )
    if record.get("evaluation_origin", "").startswith("imported original975"):
        require(
            file_sha(directory / "original-evaluation.json")
            == record["original_evaluation_sha256"],
            "cached imported original provenance changed",
        )
    for name in ("loss", "accuracy", "roc_auc"):
        require(
            abs(actual[name] - record["confirmation"][name])
            <= (3e-6 if name == "loss" else 2e-7 if name == "accuracy" else 1e-12),
            "cached confirmation metrics disagree with predictions",
        )
    return record


def evaluate_all(args, evaluator=None):
    """Audit all78 first; cache each attempt and never automatically rescore it."""
    freeze, admission, sources = audit_freeze(
        args.frozen_index,
        args.frozen_index_sha256,
        args.campaign_state,
        args.data,
        args.source_index,
    )
    output, state = Path(args.output).resolve(), Path(args.campaign_state).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "global-admission.json", admission, immutable=True)
    frozen_copy = output / "global-freeze.json"
    if frozen_copy.exists():
        require(
            file_sha(frozen_copy) == args.frozen_index_sha256,
            "output freeze copy changed",
        )
    else:
        shutil.copy2(args.frozen_index, frozen_copy)
    reused = [row for row in freeze["configurations"] if row.get("reuse", False)]
    original_evaluations = {}
    for row in freeze["configurations"]:
        pending = safe(
            state, "assessments/" + hashlib.sha256(row["id"].encode()).hexdigest()
        )
        require(
            not pending.exists() or (pending / "complete.json").exists(),
            "previous assessment attempt incomplete; recover explicitly before opening any more tail predictions",
        )
    if reused:
        require(
            getattr(args, "original_assessment_job", None) is not None,
            "reused controls require the verified original975 assessment job; never rescore them",
        )
        original_evaluations = verify_original_assessment(
            args.original_assessment_job, freeze, state, sources
        )
    records = []
    for row in freeze["configurations"]:
        # Check the selected files again immediately before each cached/model use.
        check_row_files(row, state, DATA_SHA256)
        directory = safe(
            state, "assessments/" + hashlib.sha256(row["id"].encode()).hexdigest()
        )
        complete = directory / "complete.json"
        if complete.exists():
            saved = read(complete)
            require(
                saved["freeze_sha256"] == args.frozen_index_sha256
                and saved["evaluation_sha256"]
                == file_sha(directory / "evaluation.json"),
                "cached receipt changed",
            )
            record = validate_evaluation(
                directory, row, args.frozen_index_sha256, DATA_SHA256
            )
        else:
            require(
                not directory.exists(),
                "previous assessment attempt incomplete; preserve evidence and recover explicitly, do not rescore automatically",
            )
            directory.mkdir(parents=True)
            write(
                directory / "started.json",
                {
                    "freeze_sha256": args.frozen_index_sha256,
                    "model_id": row["id"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                },
                immutable=True,
            )
            if row.get("reuse", False):
                import_original_prediction(
                    original_evaluations[row_key(row)],
                    directory,
                    row,
                    args.frozen_index_sha256,
                    freeze,
                )
            elif evaluator is not None:
                evaluator(row, directory, admission, sources)
            else:
                command = [
                    sys.executable,
                    "-m",
                    "cst_research.experiments.higgs_dst_assessment",
                    "--stage",
                    "worker",
                    "--frozen-index",
                    str(Path(args.frozen_index).resolve()),
                    "--frozen-index-sha256",
                    args.frozen_index_sha256,
                    "--campaign-state",
                    str(state.resolve()),
                    "--data",
                    str(Path(args.data).resolve()),
                    "--source-index",
                    str(Path(args.source_index).resolve()),
                    "--output",
                    str(directory.resolve()),
                    "--model-id",
                    row["id"],
                    "--admission-file",
                    str((output / "global-admission.json").resolve()),
                ]
                with (directory / "worker.log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=7200,
                    )
            record = validate_evaluation(
                directory, row, args.frozen_index_sha256, DATA_SHA256
            )
            write(
                complete,
                {
                    "freeze_sha256": args.frozen_index_sha256,
                    "evaluation_sha256": file_sha(directory / "evaluation.json"),
                },
                immutable=True,
            )
        destination = safe(
            output, "models/" + hashlib.sha256(row["id"].encode()).hexdigest()
        )
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("evaluation.json", "predictions.npz", "complete.json"):
            shutil.copy2(directory / name, destination / name)
        for name in ("original-evaluation.json", "source-evaluation.json"):
            if (directory / name).is_file():
                shutil.copy2(directory / name, destination / name)
        records.append(
            {
                "model_id": row["id"],
                "relative_directory": str(destination.relative_to(output)),
                "evaluation_sha256": file_sha(directory / "evaluation.json"),
                "predictions_sha256": record["predictions_sha256"],
            }
        )
        write(
            output / "assessment-progress.json",
            {
                "freeze_sha256": args.frozen_index_sha256,
                "completed": len(records),
                "total": 78,
            },
        )
    index = {
        "schema_version": 1,
        "freeze_sha256": args.frozen_index_sha256,
        "configurations": freeze["configurations"],
        "evaluations": records,
        "configuration_count": 78,
        "confirmation_evaluated": True,
        "test_evaluated": False,
    }
    write(output / "assessment-index.json", index, immutable=True)
    return index


def verify_original_assessment(job, freeze, state, sources):
    """Admit the once-scored original30 bundle, without executing archive code."""
    job = Path(job)
    results, artifacts = job / "results", job / "results/artifacts"
    receipt, spec, execution, manifest = (
        read(p)
        for p in (
            job / "receipt.json",
            job / "spec.json",
            results / "result.json",
            results / "manifest.json",
        )
    )
    from .higgs_dst_evidence import verify_recovery_provenance

    recovery_provenance = verify_recovery_provenance(job, spec, receipt, execution)
    require(
        file_sha(job / "results.tar.gz") == receipt["archive_sha256"]
        and file_sha(job / "source.tar.gz")
        == spec["source_sha256"]
        == execution["source_sha256"],
        "original assessment archive receipt differs",
    )
    for archive, expected in (
        (
            job / "results.tar.gz",
            manifest | {"manifest.json": file_sha(results / "manifest.json")},
        ),
        (job / "source.tar.gz", spec["source_files"]),
    ):
        observed = set()
        with tarfile.open(archive, "r|gz") as bundle:
            for member in bundle:
                if member.isdir():
                    continue
                require(
                    member.isfile() and member.name not in observed,
                    "unsafe original assessment archive",
                )
                safe(results, member.name)
                observed.add(member.name)
                digest = hashlib.sha256()
                with bundle.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                require(
                    digest.hexdigest() == expected.get(member.name),
                    "original archive member hash differs",
                )
        require(observed == set(expected), "original archive inventory differs")
    for name, digest in manifest.items():
        require(
            file_sha(safe(results, name)) == digest,
            "original local result member changed",
        )
    source, phase, status = (
        read(artifacts / n)
        for n in (
            "bootstrap-source-verification.json",
            "phase.json",
            "bootstrap-status.json",
        )
    )
    require(
        source["source_commit"] == CONTROL_SOURCE
        and phase["stage"] == "assess"
        and phase["status"] == "completed"
        and phase["confirmation_evaluated"] is True
        and phase["test_evaluated"] is False
        and status["returncode"] == 0
        and status["failure"] is None,
        "original source/assessment scope differs",
    )
    require(
        source.get("dataset_sha256") == DATA_SHA256
        and phase["identity"]["dataset_sha256"] == DATA_SHA256
        and spec["source_files"].get("research-inputs/screen.npz") == DATA_SHA256,
        "original assessment data identity differs",
    )
    original_manifest = read(sources[CONTROL_SOURCE]["manifest_file"])["files"]
    source_files = {
        name: digest
        for name, digest in spec["source_files"].items()
        if not name.startswith("research-inputs/") and name != "__pool_driver__.py"
    }
    require(
        source_files == original_manifest,
        "original assessment did not execute pinned975 source",
    )
    actual_freeze = read(artifacts / "confirmation-freeze.json")
    require(
        json_sha(actual_freeze)
        == freeze["original_confirmation_freeze_canonical_sha256"],
        "actual primary freeze is not semantically identical to comparison-owned original30 proof",
    )
    proof = read(safe(state, freeze["original_confirmation_freeze_file"]))
    require(actual_freeze == proof, "actual primary freeze differs from proof")
    index = read(artifacts / "assess-index.json")
    require(
        index["confirmation_evaluated"] is True
        and index["test_evaluated"] is False
        and index["confirmation_freeze_sha256"]
        == file_sha(artifacts / "confirmation-freeze.json")
        and len(index["assessments"]) == 30,
        "original evaluation index differs",
    )
    admitted = {}
    original_by_key = {row_key(row): row for row in actual_freeze["configurations"]}
    for entry in index["assessments"]:
        key = row_key(entry)
        require(
            key in original_by_key and key not in admitted,
            "original evaluations missing/duplicated",
        )
        expected = original_by_key[key]
        require(
            entry["checkpoint_sha256"] == expected["checkpoint_sha256"],
            "original evaluated checkpoint differs",
        )
        directory = safe(artifacts, entry["evaluation_directory"])
        evaluation = read(directory / "evaluation.json")
        require(
            file_sha(directory / "evaluation.json") == entry["evaluation_sha256"]
            and file_sha(directory / "predictions.npz")
            == entry["predictions_sha256"]
            == evaluation["predictions_sha256"]
            and evaluation["checkpoint_sha256"] == expected["checkpoint_sha256"]
            and evaluation["dataset_sha256"] == DATA_SHA256
            and evaluation["validation_partition"] == [50000, 100000]
            and evaluation["test_evaluated"] is False,
            "original prediction or evaluation bytes differ",
        )
        import numpy as np

        with np.load(directory / "predictions.npz", allow_pickle=False) as prediction:
            require(
                set(prediction.files) == {"logits", "labels"},
                "original prediction fields differ",
            )
            logits, labels = prediction["logits"], prediction["labels"]
            actual = prediction_metrics(logits, labels)
            require(
                hashlib.sha256(logits.tobytes()).hexdigest()
                == evaluation["logits_sha256"]
                and hashlib.sha256(labels.tobytes()).hexdigest()
                == evaluation["labels_sha256"],
                "original prediction array hashes differ",
            )
            for metric, tolerance in (
                ("loss", 3e-6),
                ("accuracy", 2e-7),
                ("roc_auc", 1e-12),
            ):
                require(
                    abs(actual[metric] - evaluation["confirmation"][metric])
                    <= tolerance,
                    "original confirmation metrics disagree with retained predictions",
                )
        admitted[key] = {
            "directory": directory,
            "evaluation": evaluation,
            "job": spec["id"],
            "recovery_provenance": recovery_provenance,
            "receipt_recovery": {
                key: receipt[key]
                for key in (
                    "normal_pool_acknowledgment",
                    "recovery_mode",
                    "original_pool_job_status",
                )
                if key in receipt
            },
            "result_archive_sha256": receipt["archive_sha256"],
            "original_freeze_sha256": file_sha(artifacts / "confirmation-freeze.json"),
        }
    require(
        set(admitted) == expected_keys()[1], "original30 evaluation family incomplete"
    )
    return admitted


def import_original_prediction(original, directory, row, freeze_sha, freeze):
    evaluation = original["evaluation"]
    require(
        evaluation["checkpoint_sha256"] == row["checkpoint_sha256"],
        "reused prediction is for a different model",
    )
    shutil.copy2(
        original["directory"] / "predictions.npz", directory / "predictions.npz"
    )
    shutil.copy2(
        original["directory"] / "evaluation.json",
        directory / "original-evaluation.json",
    )
    record = dict(evaluation)
    record.update(
        freeze_sha256=freeze_sha,
        model_id=row["id"],
        evaluation_origin="imported original975 assessment; not rescored",
        original_job=original["job"],
        original_receipt_recovery=original.get("receipt_recovery", {}),
        original_recovery_provenance=original.get("recovery_provenance", {}),
        original_result_archive_sha256=original["result_archive_sha256"],
        original_freeze_sha256=original["original_freeze_sha256"],
        original_freeze_canonical_sha256=freeze[
            "original_confirmation_freeze_canonical_sha256"
        ],
        original_evaluation_sha256=file_sha(directory / "original-evaluation.json"),
    )
    write(directory / "evaluation.json", record, immutable=True)


def worker(args):
    """Fresh process: validate admission/source/own checkpoint before tail access."""
    require(
        file_sha(args.frozen_index) == args.frozen_index_sha256, "worker freeze changed"
    )
    freeze, current_admission, sources = audit_freeze(
        args.frozen_index,
        args.frozen_index_sha256,
        args.campaign_state,
        args.data,
        args.source_index,
    )
    admission = read(args.admission_file)
    require(admission == current_admission, "worker global78 admission changed")
    require(
        admission["freeze_sha256"] == args.frozen_index_sha256
        and admission["verified_dst_count"] == 48
        and admission["verified_control_count"] == 30
        and len(admission["verified_model_files"]) == 78,
        "worker lacks global78 admission",
    )
    rows = [row for row in freeze["configurations"] if row["id"] == args.model_id]
    require(
        len(rows) == 1 and not rows[0].get("reuse", False),
        "worker may not rescore a reused original model",
    )
    row = rows[0]
    require(
        check_row_files(row, args.campaign_state, DATA_SHA256)
        == admission["verified_model_files"][row["id"]],
        "worker selected input changed",
    )
    require(file_sha(args.data) == DATA_SHA256, "worker data changed")
    sources = verify_sources(args.source_index, freeze)
    directory = Path(args.output).resolve()
    expected_directory = safe(
        args.campaign_state,
        "assessments/" + hashlib.sha256(row["id"].encode()).hexdigest(),
    )
    require(
        directory == expected_directory
        and (directory / "started.json").is_file()
        and not (directory / "evaluation.json").exists()
        and not (directory / "complete.json").exists(),
        "worker requires a pending once-only parent attempt",
    )
    started = read(directory / "started.json")
    require(
        started
        == {
            "freeze_sha256": args.frozen_index_sha256,
            "model_id": row["id"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        },
        "worker attempt identity differs",
    )
    claim = directory / "worker-claimed.json"
    with claim.open("x") as stream:
        json.dump(
            {"freeze_sha256": args.frozen_index_sha256, "model_id": row["id"]}, stream
        )
    source = Path(sources[row["source_commit"]]["directory"])
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source), str(source / "src")]
    # Source imports happen only after the sealed global gate above.
    import numpy as np
    import torch
    from benchmarks.training import higgs_confirmation as control

    control._configure()
    runtime = {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(),
    }
    require(
        all(str(freeze["runtime_fingerprint"][k]) == runtime[k] for k in runtime),
        "worker runtime differs from frozen campaign",
    )
    directory = Path(args.output)
    checkpoint = safe(args.campaign_state, row["relative_directory"]) / "best-dev.pt"
    selected_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(
        all(
            selected_state["configuration"][k] == row[k]
            for k in ("width", "kind", "divisor", "seed", "lr")
        )
        and selected_state["configuration"]["epochs"] == row["horizon"]
        and selected_state["configuration"]["dataset_sha256"] == DATA_SHA256
        and selected_state["epoch"] == row["best_epoch"],
        "selected checkpoint contents differ from global freeze",
    )
    if row["kind"] in ("cst", "dense"):
        # Reuse the pinned public-eager deferred evaluator for fresh comparison controls.
        namespace = argparse.Namespace(
            data=Path(args.data),
            width=row["width"],
            divisor=row["divisor"],
            seed=row["seed"],
            kind=row["kind"],
        )
        raw = directory / "external-control-evaluation"
        record = control.evaluate_checkpoint(namespace, checkpoint, raw)
        shutil.copy2(raw / "predictions.npz", directory / "predictions.npz")
        shutil.copy2(raw / "evaluation.json", directory / "source-evaluation.json")
    else:
        from benchmarks.training import higgs_dst_reference as dst

        state = selected_state
        require(
            all(
                state["configuration"][k] == row[k]
                for k in ("width", "kind", "divisor", "seed", "lr")
            )
            and state["configuration"]["epochs"] == row["horizon"]
            and state["configuration"]["dataset_sha256"] == DATA_SHA256
            and state["epoch"] == row["best_epoch"],
            "DST checkpoint contents differ from sealed model",
        )
        model = dst.make_model(row["width"], row["divisor"], row["seed"])
        model.load_state_dict(state["model"])
        # This is the first confirmation-data access in the DST evaluation path.
        x, y = control.load_partition(args.data, "confirmation")
        with torch.cuda.stream(torch.cuda.Stream()):
            model = model.cuda()
            metrics, logits, labels = control.evaluate(model, x.cuda(), y.cuda())
            torch.cuda.synchronize()
        np.savez_compressed(directory / "predictions.npz", logits=logits, labels=labels)
        record = {
            "schema_version": 1,
            "kind": row["kind"],
            "width": row["width"],
            "divisor": row["divisor"],
            "seed": row["seed"],
            "lr": row["lr"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_epoch": row["best_epoch"],
            "checkpoint_configuration": state["configuration"],
            "dataset_sha256": DATA_SHA256,
            "validation_partition": [50000, 100000],
            "confirmation": metrics,
            "test_evaluated": False,
            "predictions_sha256": file_sha(directory / "predictions.npz"),
            "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            **runtime,
        }
    record.update(
        freeze_sha256=args.frozen_index_sha256,
        model_id=row["id"],
        evaluation_origin="fresh once-only comparison evaluation",
    )
    write(directory / "evaluation.json", record, immutable=True)
    return record


def t_df2_ppf(probability):
    require(0 < probability < 1 and math.isfinite(probability), "invalid t probability")
    return (2 * probability - 1) / math.sqrt(2 * probability * (1 - probability))


def t_df2_sf(value):
    if value < 0:
        return 1 - t_df2_sf(-value)
    root = math.hypot(value, math.sqrt(2))
    return (1 / root) / (root + value)


def seed_statistics(differences, margin=0.002):
    require(
        len(differences) == 3 and all(math.isfinite(d) for d in differences),
        "three finite paired seed differences required",
    )
    mean = sum(differences) / 3
    sd = math.sqrt(sum((d - mean) ** 2 for d in differences) / 2)
    se = sd / math.sqrt(3)
    statistic = (mean + margin) / se if se else None
    p_value = t_df2_sf(statistic) if se else (0.0 if mean + margin > 0 else 1.0)
    diagnostic = t_df2_ppf(1 - 0.05 / (2 * DIAGNOSTIC_ENDPOINTS))
    return {
        "difference_per_seed": list(differences),
        "mean_difference": mean,
        "seed_standard_deviation": sd,
        "seed_standard_error": se,
        "margin": margin,
        "one_sided_noninferiority_p": p_value,
        "noninferiority_t_statistic": statistic,
        "zero_variance": se == 0,
        "primary_one_sided_seed_t_lower": mean
        - t_df2_ppf(1 - 0.05 / PRIMARY_CONFIGURATIONS) * se,
        "diagnostic_joint32_seed_t_lower": mean - diagnostic * se,
        "diagnostic_joint32_seed_t_upper": mean + diagnostic * se,
        "pointwise_two_sided95_seed_t": [
            mean - t_df2_ppf(0.975) * se,
            mean + t_df2_ppf(0.975) * se,
        ],
    }


def compare_seed_metrics(method, dense):
    metrics = {
        name: seed_statistics(
            [method[i][name] - dense[i][name] for i in range(3)], margin
        )
        for name, margin in MARGINS.items()
    }
    p_config = max(m["one_sided_noninferiority_p"] for m in metrics.values())
    if p_config <= 0.05 / PRIMARY_CONFIGURATIONS:
        status = "supported"
    elif any(
        m["diagnostic_joint32_seed_t_upper"] < -m["margin"] for m in metrics.values()
    ):
        status = "inferior"
    else:
        status = "inconclusive"
    return {
        "metrics": metrics,
        "configuration_p": p_config,
        "bonferroni16_adjusted_p": min(1.0, 16 * p_config),
        "status": status,
    }


def analyze(assessment_index, output):
    """Fixed16 IUT family plus explicitly unadjusted descriptive CST contrasts."""
    index_path = Path(assessment_index)
    root, index = index_path.parent, read(index_path)
    freeze_file = root / "global-freeze.json"
    require(
        file_sha(freeze_file) == index["freeze_sha256"]
        and index["configuration_count"] == 78
        and index["confirmation_evaluated"] is True
        and index["test_evaluated"] is False,
        "analysis lacks complete sealed assessment",
    )
    freeze = read(freeze_file)
    require(
        index["configurations"] == freeze["configurations"],
        "analysis model inventory differs from sealed freeze",
    )
    rows = {row["id"]: row for row in freeze["configurations"]}
    evaluations = index["evaluations"]
    require(
        len(rows) == 78
        and len(evaluations) == 78
        and {e["model_id"] for e in evaluations} == set(rows),
        "analysis requires all78 evaluations",
    )
    require(
        len({row_key(r) for r in rows.values()}) == 78, "analysis model keys duplicated"
    )
    by_key, labels = {}, set()
    # Check every referenced file before reading any prediction arrays.
    for entry in evaluations:
        directory = safe(root, entry["relative_directory"])
        require(
            file_sha(directory / "evaluation.json") == entry["evaluation_sha256"]
            and file_sha(directory / "predictions.npz") == entry["predictions_sha256"],
            "analysis prediction or metadata bytes changed",
        )
    for entry in evaluations:
        row = rows[entry["model_id"]]
        record = validate_evaluation(
            safe(root, entry["relative_directory"]),
            row,
            index["freeze_sha256"],
            DATA_SHA256,
        )
        labels.add(record["labels_sha256"])
        by_key[row_key(row)] = record["confirmation"]
    require(len(labels) == 1, "confirmation ordered labels differ across models")
    require(set(by_key) == set.union(*expected_keys()), "analysis family differs")
    primary, descriptive = [], []
    for width in WIDTHS:
        dense = [by_key[(width, "dense", 100, seed)] for seed in SEEDS]
        for divisor in DIVISORS:
            cst = [by_key[(width, "cst", divisor, seed)] for seed in SEEDS]
            for kind in ("static", "rigl"):
                method = [by_key[(width, kind, divisor, seed)] for seed in SEEDS]
                comparison = compare_seed_metrics(method, dense)
                primary.append(
                    {
                        "width": width,
                        "divisor": divisor,
                        "alpha": 1 / divisor,
                        "kind": kind,
                        "seeds": list(SEEDS),
                        "method_per_seed": method,
                        "dense_per_seed": dense,
                        **comparison,
                    }
                )
                descriptive.append(
                    {
                        "width": width,
                        "divisor": divisor,
                        "kind": kind,
                        "difference_direction": "CST minus DST",
                        "seeds": list(SEEDS),
                        "cst_per_seed": cst,
                        "dst_per_seed": method,
                        "metrics": {
                            name: {
                                k: v
                                for k, v in seed_statistics(
                                    [cst[i][name] - method[i][name] for i in range(3)],
                                    margin,
                                ).items()
                                if k
                                in (
                                    "difference_per_seed",
                                    "mean_difference",
                                    "seed_standard_deviation",
                                    "seed_standard_error",
                                    "pointwise_two_sided95_seed_t",
                                )
                            }
                            for name, margin in MARGINS.items()
                        },
                        "scope": "Descriptive; unadjusted pointwise two-sided95% paired-seed t interval, df2, conditional on fixed examples and normal IID paired-seed approximation; no formal CST-DST claim.",
                    }
                )
    minima = []
    for width in WIDTHS:
        for kind in ("static", "rigl"):
            group = sorted(
                (
                    row
                    for row in primary
                    if row["width"] == width and row["kind"] == kind
                ),
                key=lambda row: row["alpha"],
            )
            supported = [row["alpha"] for row in group if row["status"] == "supported"]
            minimum = min(supported) if supported else None
            minima.append(
                {
                    "width": width,
                    "kind": kind,
                    "minimum_supported_tested_alpha": minimum,
                    "labels": [
                        {"alpha": row["alpha"], "status": row["status"]}
                        for row in group
                    ],
                    "lower_range_endpoint_supported": minimum == 0.001,
                    "finite_grid_boundary_identified": bool(
                        minimum
                        and minimum > 0.001
                        and all(
                            row["status"] == "inferior"
                            for row in group
                            if row["alpha"] < minimum
                        )
                    ),
                    "continuous_or_universal_minimum_claim": False,
                    "monotonicity_assumed": False,
                }
            )
    result = {
        "schema_version": 1,
        "freeze_sha256": index["freeze_sha256"],
        "family": "16 DST-vs-matched-dense configurations; both co-primary endpoints required",
        "primary_method": "paired training-seed t df2 IUT, p_config=max(endpoint p), Bonferroni16; familywise alpha0.05",
        "primary_sources": [
            "https://www.fda.gov/media/162416/download?attachment=",
            "https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm",
        ],
        "margins": MARGINS,
        "primary_configurations": 16,
        "diagnostic_two_sided_endpoints": 32,
        "scope": "Conditional on fixed confirmation benchmark; approximately normal IID paired seed differences; three seeds cannot validate normality and have limited power.",
        "original_eight_configuration_family": "Unchanged and separate; no joint95% guarantee across original8 plus new16.",
        "combined_joint_seed_example_confidence_claim": False,
        "paired_example_bootstrap": {
            "performed": False,
            "scope": "No combined seed/example interval is claimed; primary inference concerns paired training seeds on this fixed benchmark.",
        },
        "primary": primary,
        "minimum_density": minima,
        "cst_dst_descriptive": descriptive,
        "test_evaluated": False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "analysis.json", result, immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("audit", "evaluate", "worker", "analyze"), required=True
    )
    parser.add_argument("--frozen-index", type=Path)
    parser.add_argument("--frozen-index-sha256")
    parser.add_argument("--campaign-state", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--original-assessment-job", type=Path)
    parser.add_argument("--assessment-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--admission-file", type=Path)
    args = parser.parse_args()
    if args.stage == "analyze":
        require(
            args.assessment_index is not None, "analyze requires --assessment-index"
        )
        result = analyze(args.assessment_index, args.output)
    else:
        require(
            all(
                getattr(args, k) is not None
                for k in (
                    "frozen_index",
                    "frozen_index_sha256",
                    "campaign_state",
                    "data",
                    "source_index",
                )
            ),
            "assessment gate arguments are required",
        )
        if args.stage == "audit":
            _, result, _ = audit_freeze(
                args.frozen_index,
                args.frozen_index_sha256,
                args.campaign_state,
                args.data,
                args.source_index,
            )
            write(args.output / "global-admission.json", result, immutable=True)
        elif args.stage == "worker":
            require(
                args.model_id is not None and args.admission_file is not None,
                "worker requires model/admission",
            )
            result = worker(args)
        else:
            result = evaluate_all(args)
    print(
        json.dumps(
            {"stage": args.stage, "completed": True, "output": str(args.output)},
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    main()
