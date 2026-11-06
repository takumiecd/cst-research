"""Explicit one-shot post-failure correction: reuse300k, finish selected eval, fresh375k."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import higgs_depth_normalization_learning as parent
from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_normalization_learning_restore as correction
from .higgs_dst_comparison import file_sha, json_sha, read, write

COHORT = "higgs-normalization-learning-recovery-v1"
PARENT_COMMIT = "ea0d5701c45b188d619853ebd303767f224e36b8"
PARENT_MANIFEST = "68bf4e7eb80ed9c48ce4c27fc26bd549f55a309276ddd519d63d00dcf2860fb0"
PARENT_RECEIPT = "18403283955ab2a4d6f5dbdfe9f2094cec54efe9aadbeb3bff15e5285275bcab"
PARENT_BUNDLE = "fd1171acc872d0903b6a03033b0aecf0dfb8c3dccf1e2f4061679eeb7e0917b8"
PARENT_CPU_ADMISSION = (
    "7d69415094ec9bdf31e5ecd66654d3aac2d792e46d92836dd016c2abe780b6b7"
)
PARENT_FILE_ADMISSION = (
    "dbe3081a726d6e175f880d483b298ce86319eb28cf4bdbca98090769553ea4fb"
)
REMAINING = tuple(parent.CASES[4:])
ORIGINAL_EVALUATOR = parent.evaluate_selected


def protocol():
    return {
        "version": COHORT,
        "amendment": "explicit post-failure selected-checkpoint load-contract correction, not training recipe change",
        "parent_consumer_commit": PARENT_COMMIT,
        "parent_consumer_manifest_sha256": PARENT_MANIFEST,
        "failed_parent_outer_receipt_sha256": PARENT_RECEIPT,
        "training_recipe_protocol_sha256": json_sha(parent.protocol()),
        "native_load_rule": correction.RULE,
        "native_source_sha256": correction.ATOM_STATE_SOURCE_SHA,
        "generation_scope": "only actual3CST owners2/4/6 at native load; every other core tensor/metadata literal; evaluation compared toactual POSTLOAD baseline",
        "reuse_complete_selected_cases": list(parent.CASES[:3]),
        "finish_parent_selected_evaluation_case": "cst104-stock",
        "fresh_remaining_cases": list(REMAINING),
        "parent_main_updates": 300000,
        "recovery_main_updates": 375000,
        "combined_main_updates": 675000,
        "parent_prefix_updates": 72,
        "new_prefix_updates": 0,
        "seed": 17,
        "epochs": 3,
        "updates_per_fresh_case": 75000,
        "parent_raw_readonly": True,
        "automatic_retry_or_resume": False,
        "heldout_evaluated": False,
        "HPO": False,
        "additional_recipe_budget_or_selection_change": False,
        "prior27_modules_and9_registries_unchanged": True,
        "required_parent_CPU_report_status": "passed-retained-fourcase-evidence-only-run-INCOMPLETE",
    }


def require_parent_inventory(root, expected):
    """Reject extra artifacts as well as symlinks; never delete or normalize inputs."""
    actual = {}
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if name.startswith("consumer/") or name == "consumer":
            continue
        exact.require(
            not path.is_symlink() and (path.is_file() or path.is_dir()),
            "nonregular failed-parent artifact",
        )
        if path.is_file():
            actual[name] = path
    exact.require(
        set(actual) == set(expected),
        "unexpected failed-parent member; keep audit sidecars outside raw root",
    )
    return actual


def verify_parent(root, report_file, report_sha, current_consumer):
    """Full raw bytes plus independently admitted parent scope; no GPU needed."""
    root = Path(root).resolve(strict=True)
    exact.require(
        report_sha == PARENT_CPU_ADMISSION and file_sha(report_file) == report_sha,
        "owner-sealed actual parent CPU report differs",
    )
    audit = read(report_file)
    for key, expected in {
        "status": protocol()["required_parent_CPU_report_status"],
        "parent_outer_receipt_sha256": PARENT_RECEIPT,
        "parent_consumer_commit": PARENT_COMMIT,
        "parent_consumer_manifest_sha256": PARENT_MANIFEST,
        "completed_main_updates": 300000,
        "prefixes_all9pass": True,
        "complete_selected_cases": list(parent.CASES[:3]),
        "trained_cst104_stock_epochs": 3,
        "heldout_evaluated": False,
        "parent_run_complete": False,
    }.items():
        exact.require(
            audit.get(key) == expected,
            "independent actual parent admission differs: " + key,
        )
    exact.require(
        isinstance(audit.get("file_admission_proof_sha256"), str)
        and re.fullmatch("[0-9a-f]{64}", audit["file_admission_proof_sha256"])
        and audit["file_admission_proof_sha256"] == PARENT_FILE_ADMISSION,
        "parent full-file admission proof missing",
    )
    receipt_path = exact.safe_member(root, "outer-receipt.json")
    exact.require(
        file_sha(receipt_path) == PARENT_RECEIPT, "failed parent receipt differs"
    )
    receipt = read(receipt_path)
    exact.require(
        type(receipt["driver_exit_code"]) is int
        and receipt["driver_exit_code"] == 1
        and receipt["completed_summary_present"] is False
        and receipt["consumer_commit"] == PARENT_COMMIT
        and receipt["bundle_sha256"] == PARENT_BUNDLE
        and receipt["runtime_pin_sha256"] == exact.RUNTIME_SHA
        and receipt["automatic_retry"] is False,
        "failed parent producer identity/status differs",
    )
    files = {
        **receipt["files"],
        "outer-receipt.json": {
            "sha256": PARENT_RECEIPT,
            "bytes": receipt_path.stat().st_size,
        },
    }
    exact.require(
        len(files) == 75
        and len(audit["parent_receipt_files_verified"]) == 74
        and audit["parent_receipt_files_verified"] == receipt["files"],
        "parent75 byte proof differs from actual receipt",
    )
    exact.verify_file_inventory(root, files)
    require_parent_inventory(root, files)
    source_manifest = exact.safe_member(root, "result/consumer-source-manifest.json")
    exact.require(
        file_sha(source_manifest) == PARENT_MANIFEST, "parent source manifest differs"
    )
    old = read(source_manifest)
    old_modules = {
        name: value
        for name, value in old["files"].items()
        if name.startswith("src/cst_research/experiments/")
    }
    old_studies = {
        name: value
        for name, value in old["files"].items()
        if name.startswith("src/cst_research/studies/")
    }
    exact.require(
        old["commit"] == PARENT_COMMIT
        and len(old_modules) == 27
        and len(old_studies) == 9,
        "parent27/9 inventory differs",
    )
    for name, value in {**old_modules, **old_studies}.items():
        exact.require(
            file_sha(exact.safe_member(current_consumer, name)) == value,
            "frozen parent module/registry changed: " + name,
        )
    manifest = read(exact.safe_member(root, "result/science/run-manifest.json"))
    identity = manifest["identity"]
    exact.require(
        identity["consumer_commit"] == PARENT_COMMIT
        and identity["consumer_manifest_sha256"] == PARENT_MANIFEST
        and identity["protocol_sha256"] == json_sha(parent.protocol())
        and identity["data_sha256"] == parent.DATA_SHA
        and identity["data_manifest_sha256"] == parent.MANIFEST_SHA
        and identity["runtime_pin_sha256"] == exact.RUNTIME_SHA,
        "parent scientific identity differs",
    )
    exact.require(
        not (root / "result/science/run-summary.json").exists(),
        "failed parent unexpectedly complete",
    )
    prefix = read(root / "result/science/all9-prefix-reproduction.json")
    exact.require(
        prefix["status"] == "passed-all9-exact-new-bare-prefix"
        and prefix["cases"] == list(parent.CASES)
        and prefix["replay_updates_per_case"] == 8
        and prefix["new_main_training_updates"] == 0,
        "all9 inherited parent prefix incomplete",
    )
    for case in parent.CASES[:4]:
        path = root / "result/science/cases" / case
        learning = read(path / "learning.json")
        rows = learning["epochs"]
        selected = parent.selected_epoch(rows)
        best = read(path / "best-dev.json")
        exact.require(
            [r["updates"] for r in rows] == [25000, 50000, 75000]
            and learning["configuration"]["case"] == case
            and learning["configuration"]["learning_identity"] == identity
            and best["epoch"] == selected["epoch"]
            and best["checkpoint_sha256"] == selected["checkpoint_sha256"]
            and best["development_predictions_sha256"]
            == selected["development_predictions_sha256"],
            "parent case budget/selection differs",
        )
        exact.require(
            (path / "complete.json").exists() is (case in parent.CASES[:3]),
            "parent selected-evaluation completion boundary differs",
        )
    for case in REMAINING:
        exact.require(
            not (root / "result/science/cases" / case).exists(),
            "remaining case already exists; no duplicate training permitted",
        )
    return {
        "kind": "read-only-failed-parent-300k-source-and-independent-CPU-admission",
        "auditor_source_sha256": file_sha(Path(__file__)),
        "auditor_protocol_sha256": json_sha(protocol()),
        "producer_identity": identity,
        "parent_outer_receipt_sha256": PARENT_RECEIPT,
        "independent_CPU_report_sha256": report_sha,
        "independent_CPU_file_admission_proof_sha256": audit[
            "file_admission_proof_sha256"
        ],
        "files": files,
        "parent_modules": old_modules,
        "parent_studies": old_studies,
        "inherited_prefix_updates": 72,
        "reused_main_updates": 300000,
        "parent_complete": False,
    }


@contextlib.contextmanager
def selected_evaluator_amendment(identity, readonly_parent_root):
    exact.require(
        parent.evaluate_selected is ORIGINAL_EVALUATOR,
        "parent evaluator already replaced; no nested or unreviewed seam",
    )
    original = parent.evaluate_selected

    def amended(case, gains, best, root, tx, ty, vx, vy, device):
        return correction.evaluate_selected(
            case,
            gains,
            best,
            root,
            root,
            tx,
            ty,
            vx,
            vy,
            device,
            identity,
            readonly_parent_root=readonly_parent_root,
        )

    provenance = {
        "attribute": "higgs_depth_normalization_learning.evaluate_selected",
        "original_function": original.__module__ + "." + original.__qualname__,
        "original_function_source_sha256": hashlib.sha256(
            inspect.getsource(original).encode()
        ).hexdigest(),
        "replacement_function": correction.evaluate_selected.__module__
        + "."
        + correction.evaluate_selected.__qualname__,
        "replacement_function_source_sha256": hashlib.sha256(
            inspect.getsource(correction.evaluate_selected).encode()
        ).hexdigest(),
        "scope": "only5fresh remaining cases; no model/optimizer/ordinary_step/epochloop/core/prefix override",
        "restored_finally": True,
    }
    parent.evaluate_selected = amended
    try:
        yield provenance
    finally:
        parent.evaluate_selected = original


def run(args):
    from . import higgs_dst_confirmation_driver as common
    from .higgs_coordinate_lr import verify_external

    recovery_started = time.perf_counter()
    consumer = Path(__file__).resolve().parents[3]
    common.verify_consumer(
        consumer,
        args.consumer_manifest,
        args.consumer_commit,
        consumer / common.DRIVER_PATH,
    )
    verify_external(
        SimpleNamespace(
            external_source=args.external_source,
            external_manifest=args.external_manifest,
            identity={"external_source_manifest_sha256": parent.stock.SOURCE_MANIFEST},
        )
    )
    import torch
    from benchmarks.training.higgs_confirmation import _configure

    _configure()
    runtime = parent.prior_runtime.runtime_fingerprint()
    exact.require(
        file_sha(args.inputs / "runtime-pin.json") == exact.RUNTIME_SHA
        and runtime == read(args.inputs / "runtime-pin.json"),
        "unchanged4090 runtime pin required",
    )
    parent.prior_runtime.validate_runtime(runtime)
    args.parent_root = args.parent_root.resolve(strict=True)
    args.output = args.output.resolve()
    exact.require(
        not args.output.exists()
        and not args.output.is_relative_to(args.parent_root)
        and not args.output.is_relative_to(args.prior_root.resolve(strict=True)),
        "fresh output outside both immutable parents required",
    )
    source_runtime_guard_seconds = time.perf_counter() - recovery_started
    phase_started = time.perf_counter()
    inherited = exact.verify_prior(
        args.inputs,
        args.prior_root,
        consumer,
        args.external_source,
        runtime,
        parent.CASES,
    )
    prior151_admission_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    admitted = verify_parent(
        args.parent_root, args.parent_admission, args.parent_admission_sha256, consumer
    )
    exact.assert_equal(
        admitted["producer_identity"]["runtime_fingerprint"],
        runtime,
        "parent-to-current-exact-runtime",
    )
    parent75_admission_seconds = time.perf_counter() - phase_started
    args.output.mkdir(parents=True)
    identity = {
        "amendment_cohort": COHORT,
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "protocol_sha256": json_sha(protocol()),
        "unchanged_training_recipe_protocol_sha256": json_sha(parent.protocol()),
        "parent_producer_identity": admitted["producer_identity"],
        "parent_outer_receipt_sha256": PARENT_RECEIPT,
        "parent_independent_CPU_report_sha256": args.parent_admission_sha256,
        "inherited_qualification_identity": inherited["producer_identity"],
        "external_source_commit": parent.stock.SOURCE,
        "external_source_manifest_sha256": parent.stock.SOURCE_MANIFEST,
        "data_sha256": parent.DATA_SHA,
        "data_manifest_sha256": parent.MANIFEST_SHA,
        "runtime_pin_sha256": exact.RUNTIME_SHA,
        "runtime_fingerprint": runtime,
    }
    write(
        args.output / "run-manifest.json",
        {
            "identity": identity,
            "protocol": protocol(),
            "status": "started-explicit-post-failure-attempt",
            "argv": sys.argv,
        },
    )
    write(args.output / "parent-admission.json", admitted)
    phase_started = time.perf_counter()
    arrays, data_manifest = parent.load_pilot(
        args.inputs / "train-development.npz", args.inputs / "data-manifest.json"
    )
    calibration = read(args.parent_root / "result/science/calibration.json")["cases"]
    for case in parent.CASES:
        exact.assert_equal(
            calibration[case],
            inherited["calibrations"][case],
            "parent-calibration-to-4f",
        )
    device = torch.device("cuda:0")
    tx, ty, vx, vy = [
        torch.from_numpy(arrays[key]).to(device)
        for key in ("x_train", "y_train", "x_development", "y_development")
    ]
    torch.cuda.synchronize()
    preparation = {
        "source_and_runtime_guards_seconds": source_runtime_guard_seconds,
        "inherited151_readonly_admission_seconds": prior151_admission_seconds,
        "failed_parent75_readonly_admission_seconds": parent75_admission_seconds,
        "data_read_calibration_metadata_check_and_device_transfer_seconds": time.perf_counter()
        - phase_started,
        "preparation_total_seconds_inclusive": time.perf_counter() - recovery_started,
        "scope": "components above disjoint; total inclusive/nonadditive and includes manifest I/O; driver archive/cache bootstrap precedes this child and is not separately timed here, available only within outer receipt walltime; no full adoption-time completeness claim",
    }
    write(args.output / "preparation.json", preparation)
    source_stock = args.parent_root / "result/science/cases/cst104-stock"
    fixed_root = args.output / "parent-cst104-stock-selected-evaluation"
    best = read(source_stock / "best-dev.json")
    selected = correction.evaluate_selected(
        "cst104-stock",
        calibration["cst104-stock"]["gains"],
        best,
        source_stock,
        fixed_root,
        tx,
        ty,
        vx,
        vy,
        device,
        identity,
        readonly_parent_root=args.parent_root,
    )
    write(
        args.output / "parent-stock-selection-gate.json",
        {
            "status": "passed-before-any-remaining-training",
            "parent_selected_reference_sha256": file_sha(
                source_stock / "best-dev.json"
            ),
            "corrected_evaluation_sha256": file_sha(
                fixed_root / "selected-evaluation.json"
            ),
            "main_training_updates_in_new_attempt": 0,
            "parent_raw_modified": False,
            "selected_epoch": selected["epoch"],
        },
    )
    completed = []
    cases = args.output / "cases"
    cases.mkdir()
    with selected_evaluator_amendment(identity, args.parent_root) as seam:
        write(args.output / "explicit-evaluator-amendment.json", seam)
        for case in REMAINING:
            completed.append(
                parent.run_case(
                    case,
                    calibration[case],
                    inherited,
                    args.prior_root,
                    identity,
                    tx,
                    ty,
                    vx,
                    vy,
                    device,
                    cases / case,
                )
            )
    # Recheck both every byte AND exact inventory, rejecting unexpected new parent files.
    phase_started = time.perf_counter()
    final_parent = verify_parent(
        args.parent_root, args.parent_admission, args.parent_admission_sha256, consumer
    )
    exact.assert_equal(final_parent, admitted, "final-parent-admission-unchanged")
    final_parent_verification_seconds = time.perf_counter() - phase_started
    write(
        args.output / "run-summary.json",
        {
            "status": "completed-bounded-nine-case-study-by-explicit-recovery-lineage",
            "identity": identity,
            "reused_complete_selected_parent_cases": [
                {
                    "case": case,
                    "producer_identity": admitted["producer_identity"],
                    "parent_complete_sha256": file_sha(
                        args.parent_root
                        / "result/science/cases"
                        / case
                        / "complete.json"
                    ),
                }
                for case in parent.CASES[:3]
            ],
            "parent_cst104_stock_selected_evaluation_sha256": file_sha(
                fixed_root / "selected-evaluation.json"
            ),
            "fresh_cases": completed,
            "preparation": preparation,
            "final_parent75_verification_seconds": final_parent_verification_seconds,
            "new_child_wall_seconds_inclusive": time.perf_counter() - recovery_started,
            "new_child_wall_scope": "inclusive guard/admission/data/selected evaluation/375k training/postprocessing; not additive with component timers or parent300k time; excludes earlier driver bootstrap",
            "parent_main_updates": 300000,
            "new_main_updates": 375000,
            "combined_main_updates": 675000,
            "inherited_parent_prefix_updates": 72,
            "new_prefix_updates": 0,
            "failed_parent_original_exit_code": 1,
            "failed_parent_relabelled_complete": False,
            "all75_parent_members_unchanged": True,
            "data_exposure": data_manifest["exposure"],
            "heldout_evaluated": False,
            "automatic_retry_or_followup": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inputs",
        "prior-root",
        "parent-root",
        "parent-admission",
        "consumer-manifest",
        "external-source",
        "external-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("consumer-commit", "parent-admission-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.external_source / "src"), str(args.external_source)]
    run(args)


if __name__ == "__main__":
    main()
