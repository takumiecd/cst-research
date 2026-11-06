"""Read-only inherited4f evidence and exact recursive task-prefix comparison."""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path, PurePosixPath

from .higgs_depth_data import DATA_SHA, MANIFEST_SHA
from .higgs_depth_study import SOURCE, SOURCE_MANIFEST
from .higgs_dst_comparison import file_sha, read

PRIOR_COMMIT = "4f777a5f1ef38cca7031cf9c637d0fa6fc0d5e7b"
PRIOR_MANIFEST = "aa942f52f88d9a26a950eb51203df948216bbb50fb437c410b9dc8d9b86bbd53"
PRIOR_PROTOCOL = "0a267cc7fe634426ada6a515bdddf3459b7f7fa2d34797b84d38f65860cc3d33"
RUNTIME_SHA = "1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b"
PROOFS = {
    "prior-full-CPU-admission.json": "89238d2bece6f2537c68e3053e4da839a138dba7f4f78cd8c0bc92fe54faeba8",
    "prior-file-admission.json": "3d857e153906395c76cf516f0afeb54508bf532acfa375b99e0c69f6e916ab3a",
    "prior-outer-receipt.json": "ae0c2bbb7f000f5465c41b24e66c08223ffdd722a45382d70251db71f905c6ab",
    "prior-launch-intent.json": "bcb31daaff0f84191db1c965484191afe6540425f441d44338db73357b96fcf2",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_member(root, name):
    relative = PurePosixPath(name)
    require(
        isinstance(name, str)
        and name
        and str(relative) == name
        and not relative.is_absolute()
        and ".." not in relative.parts
        and "\\" not in name,
        "unsafe inherited evidence member",
    )
    cursor = Path(root)
    require(not cursor.is_symlink(), "symlink inherited evidence root")
    for part in relative.parts:
        cursor /= part
        require(not cursor.is_symlink(), "symlink inherited evidence member")
    return cursor


def verify_file_inventory(root, files, overrides=None):
    """Stream hashes in place; explicit overrides may relocate exact small receipts only."""
    overrides = overrides or {}
    require(
        set(overrides) <= {"outer-receipt.json", "launch-intent.json"},
        "undeclared prior relocation",
    )
    observed = {}
    for name, entry in files.items():
        canonical = safe_member(root, name)
        path = overrides.get(name, canonical)
        require(
            path.is_file() and not path.is_symlink(),
            "missing/nonregular inherited file: " + name,
        )
        require(
            path.stat().st_size == entry["bytes"] and file_sha(path) == entry["sha256"],
            "inherited file bytes differ: " + name,
        )
        observed[name] = entry
    return observed


def verify_prior(inputs, prior_root, consumer, external_source, runtime, cases):
    for name, digest in PROOFS.items():
        require(
            file_sha(inputs / name) == digest, "inherited proof digest differs: " + name
        )
    full = read(inputs / "prior-full-CPU-admission.json")
    files = read(inputs / "prior-file-admission.json")
    outer = read(inputs / "prior-outer-receipt.json")
    identity = full["identity"]
    require(
        full["status"] == "passed-independent-all151file-nine-case-raw-CPU-admission"
        and full["file_admission_sha256"] == PROOFS["prior-file-admission.json"]
        and files["outer_receipt_sha256"] == PROOFS["prior-outer-receipt.json"]
        and outer["driver_exit_code"] == 0
        and outer["completed_summary_present"] is True,
        "prior full admission/normal receipt incomplete",
    )
    require(
        identity["consumer_commit"] == PRIOR_COMMIT
        and identity["consumer_manifest_sha256"] == PRIOR_MANIFEST
        and identity["protocol_sha256"] == PRIOR_PROTOCOL
        and identity["runtime_pin_sha256"] == RUNTIME_SHA
        and identity["runtime_fingerprint"] == runtime,
        "prior producer identity/runtime differs",
    )
    require(
        identity["data_sha256"] == DATA_SHA
        and identity["data_manifest_sha256"] == MANIFEST_SHA
        and identity["external_source_commit"] == SOURCE
        and identity["external_source_manifest_sha256"] == SOURCE_MANIFEST,
        "prior data/external source identity differs",
    )
    require(
        len(files["files"]) == 151 and files["raw_bytes"] == 2643520928,
        "prior full151 inventory differs",
    )
    observed = verify_file_inventory(
        prior_root,
        files["files"],
        {
            "outer-receipt.json": inputs / "prior-outer-receipt.json",
            "launch-intent.json": inputs / "prior-launch-intent.json",
        },
    )
    manifest_file = safe_member(prior_root, "result/consumer-source-manifest.json")
    require(
        file_sha(manifest_file) == PRIOR_MANIFEST, "prior scientific manifest differs"
    )
    scientific = read(manifest_file)
    require(
        scientific["commit"] == PRIOR_COMMIT and len(scientific["files"]) == 85,
        "prior source inventory differs",
    )
    unchanged = {
        name: digest
        for name, digest in scientific["files"].items()
        if name.startswith("src/cst_research/experiments/")
    }
    require(len(unchanged) == 24, "prior24 numerical modules required")
    for name, digest in unchanged.items():
        require(
            file_sha(safe_member(consumer, name)) == digest,
            "inherited numerical module changed: " + name,
        )
    external_manifest = read(inputs / "original-source-manifest.json")
    step_name = "benchmarks/training/full_step_graph.py"
    require(
        external_manifest["files"][step_name]
        == "83277111c339c9291f0f4b9709a057c5dc11771407927bb8144bf8c72f2dd3d9"
        and file_sha(external_source / step_name)
        == external_manifest["files"][step_name],
        "bare ordinary_step source differs",
    )
    source_context = read(safe_member(prior_root, "result/source-verification.json"))
    require(
        source_context["consumer_commit"] == PRIOR_COMMIT
        and source_context["consumer_manifest_sha256"] == PRIOR_MANIFEST
        and source_context["external_commit"] == SOURCE
        and source_context["external_manifest_sha256"] == SOURCE_MANIFEST,
        "prior source context differs",
    )
    for name, digest in [
        ("consumer-source-manifest.json", PRIOR_MANIFEST),
        ("train-development.npz", DATA_SHA),
        ("data-manifest.json", MANIFEST_SHA),
        ("runtime-pin.json", RUNTIME_SHA),
    ]:
        require(
            source_context["inputs"][name] == digest,
            "prior input context differs: " + name,
        )
    require(
        read(safe_member(prior_root, "result/science/run-manifest.json"))["identity"]
        == identity,
        "prior run context differs",
    )
    summary = read(safe_member(prior_root, "result/science/run-summary.json"))
    require(
        summary["status"] == "completed"
        and summary["identity"] == identity
        and summary["cases"] == list(cases)
        and summary["task_updates_per_case"] == 8
        and summary["development_evaluated"] is False
        and summary["selection_performed"] is False
        and summary["future_confirmation_evaluated"] is False
        and summary["official_test_evaluated"] is False,
        "prior task/scope incomplete",
    )
    calibrations = read(safe_member(prior_root, "result/science/calibration.json"))
    require(set(calibrations) == set(cases), "prior calibration case inventory differs")
    return {
        "kind": "INHERITED_4f_QUALIFICATION_NOT_NEW_ORACLE",
        "producer_identity": identity,
        "proof_sha256": dict(PROOFS),
        "files_verified_in_place": observed,
        "small_receipt_relocation_only": ["outer-receipt.json", "launch-intent.json"],
        "scientific_modules_verified": unchanged,
        "source_ordinary_step_sha256": external_manifest["files"][step_name],
        "prior_root": str(prior_root),
        "old_state8_reused_for_training": False,
        "calibrations": calibrations,
        "full_CPU_audit": full,
        "scope": "Inherited source/runtime/operator/normal oracle qualification; new control flow requires exact new bare task-prefix reproduction for all9 before long learning.",
    }


def assert_equal(actual, expected, path=""):
    """Exact tensors AND recursive metadata; never repeat the old _extra_state bug."""
    import torch

    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        require(
            isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor),
            "tensor type mismatch: " + path,
        )
        require(
            actual.dtype == expected.dtype and actual.shape == expected.shape,
            "tensor metadata mismatch: " + path,
        )
        a = actual.detach().cpu().contiguous()
        b = expected.detach().cpu().contiguous()
        require(
            torch.equal(
                a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
            ),
            "tensor bytes mismatch: " + path,
        )
    elif isinstance(actual, dict) or isinstance(expected, dict):
        require(
            type(actual) is type(expected) and actual.keys() == expected.keys(),
            "mapping keys/type mismatch: " + path,
        )
        require(
            {(type(k), repr(k)) for k in actual}
            == {(type(k), repr(k)) for k in expected},
            "mapping key types differ: " + path,
        )
        for key in actual:
            assert_equal(actual[key], expected[key], path + "/" + repr(key))
    elif isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        require(
            type(actual) is type(expected) and len(actual) == len(expected),
            "sequence mismatch: " + path,
        )
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            assert_equal(a, b, path + "/" + str(i))
    elif isinstance(actual, float) or isinstance(expected, float):
        require(
            type(actual) is type(expected)
            and math.isfinite(actual)
            and math.isfinite(expected)
            and struct.pack("!d", actual) == struct.pack("!d", expected),
            "scalar float mismatch: " + path,
        )
    else:
        require(
            type(actual) is type(expected) and actual == expected,
            "metadata mismatch: " + path,
        )


def fingerprint(value):
    """Compact exact tensor hash maps and typed recursive metadata, no tensor copies on disk."""
    import torch

    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        return {
            "tensor": True,
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "sha256": hashlib.sha256(
                t.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    if isinstance(value, dict):
        return {"dict": [[repr(k), fingerprint(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [fingerprint(v) for v in value]}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    return {"type": type(value).__name__, "value": value}
