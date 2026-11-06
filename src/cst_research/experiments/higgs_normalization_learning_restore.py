"""Explicit post-failure native-load contract correction; no training/runtime changes."""

from __future__ import annotations

import copy
import time
from pathlib import Path

from . import higgs_depth_normalization_learning as parent
from . import higgs_depth_normalization_learning_evidence as exact
from .higgs_dst_comparison import file_sha, write

RULE = "public975 AtomState.set_extra_state: loaded=max(before_load+1,serialized)"
ATOM_STATE_SOURCE_SHA = (
    "eaba2a17badebd373b38745ffcbc42e443cd1c519c8972b9fdf766c069b5102c"
)


def owner_generations(model):
    from torchcst.nn import CSTModule

    owners = {
        name + ".atom_state._extra_state": site.atom_state
        for name, site in model.named_modules()
        if isinstance(site, CSTModule)
    }
    expected = (
        set()
        if model._diagnostic_case.startswith("dense-")
        else {str(i) + ".atom_state._extra_state" for i in (2, 4, 6)}
    )
    exact.require(set(owners) == expected, "exact actual CST owner inventory differs")
    for state in owners.values():
        exact.require(
            type(state.layout_version) is int and state.layout_version >= 0,
            "invalid pre-load generation",
        )
    return {key: state.layout_version for key, state in owners.items()}


def expected_native_loaded_core(saved, before):
    """Adjust a comparison COPY only; never alter a live or saved generation."""
    expected = copy.deepcopy(parent.core_state(saved))
    transitions = []
    declared = {
        key for key in expected["model"] if key.endswith(".atom_state._extra_state")
    }
    exact.require(declared == set(before), "checkpoint CST owner inventory differs")
    for key, previous in before.items():
        state = expected["model"][key]
        exact.require(
            set(state) == {"format_version", "shape", "layout_version"}
            and state["format_version"] == 1
            and type(state["layout_version"]) is int
            and state["layout_version"] >= 0
            and type(previous) is int
            and previous >= 0,
            "native atom checkpoint schema differs",
        )
        serialized = state["layout_version"]
        loaded = max(previous + 1, serialized)
        state["layout_version"] = loaded
        transitions.append(
            {
                "path": "model/" + key + "/layout_version",
                "before_load": previous,
                "serialized": serialized,
                "expected_after_native_load": loaded,
            }
        )
    return expected, transitions


def restore_selected(case, gains, saved, device):
    model, optimizer, _ = parent.fresh(case, gains, device)
    before = owner_generations(model)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    for name, parameter in model.named_parameters():
        value = saved["gradients"][name]
        parameter.grad = None if value is None else value.to(device).clone()
    actual = parent.core_state(
        parent.stock.checkpoint(
            model, optimizer, saved["epoch"], saved["updates"], {}, None
        )
    )
    expected, transitions = expected_native_loaded_core(saved, before)
    exact.assert_equal(actual, expected, "native-load-transition-all-other-bytes-exact")
    report = {
        "rule": RULE,
        "source_sha256": ATOM_STATE_SOURCE_SHA,
        "owner_transitions": transitions,
        "all_other_tensor_and_metadata_bytes_exact": True,
        "saved_or_live_counter_written_by_adapter": False,
        "actual_post_load_core": exact.fingerprint(actual),
    }
    return model, optimizer, actual, report


def assert_after_evaluation(model, optimizer, saved, postload_baseline):
    actual = parent.core_state(
        parent.stock.checkpoint(
            model, optimizer, saved["epoch"], saved["updates"], {}, None
        )
    )
    exact.assert_equal(actual, postload_baseline, "evaluation-mutated-postload-state")


def evaluate_selected(
    case,
    gains,
    best,
    source_case_root,
    destination_output,
    tx,
    ty,
    vx,
    vy,
    device,
    amendment_identity,
    *,
    readonly_parent_root,
):
    """Read the source case only; write all new predictions/proofs to fresh output."""
    import numpy as np
    import torch
    from benchmarks.training.higgs_confirmation import evaluate

    total_started = time.perf_counter()
    source_case_root = Path(source_case_root).resolve(strict=True)
    destination_output = Path(destination_output).resolve()
    exact.require(
        not destination_output.is_relative_to(
            Path(readonly_parent_root).resolve(strict=True)
        ),
        "new evaluation output cannot write within immutable failed-parent evidence",
    )
    for name in (
        "selected-train-predictions.npz",
        "selected-dev-predictions.npz",
        "selected-evaluation.json",
    ):
        exact.require(
            not (destination_output / name).exists(),
            "fresh selected-evaluation artifacts required",
        )
    destination_output.mkdir(parents=True, exist_ok=True)
    checkpoint = exact.safe_member(source_case_root, best["checkpoint_file"])
    cached_dev = exact.safe_member(
        source_case_root, best["development_predictions_file"]
    )
    exact.require(
        file_sha(checkpoint) == best["checkpoint_sha256"]
        and file_sha(cached_dev) == best["development_predictions_sha256"],
        "selected parent checkpoint/prediction bytes changed",
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    exact.require(
        saved["epoch"] == best["epoch"] and saved["updates"] == best["epoch"] * 25000,
        "selected epoch/clock differs",
    )
    exact.assert_equal(
        saved["development"], best["development"], "selected-saved-development"
    )
    checkpoint_read_admission_seconds = time.perf_counter() - total_started
    (model, optimizer, baseline, first_load), first_restore_seconds = parent.timed(
        restore_selected, case, gains, saved, device
    )
    (training, logits, labels), train_seconds = parent.timed(evaluate, model, tx, ty)
    integrity_started = time.perf_counter()
    assert_after_evaluation(model, optimizer, saved, baseline)
    first_integrity_seconds = time.perf_counter() - integrity_started
    train_arrays = {
        "logits": logits,
        "labels": labels,
        "row_ids": np.arange(1000000, 1100000, dtype=np.int64),
    }
    read_started = time.perf_counter()
    with np.load(cached_dev, allow_pickle=False) as archive:
        dev_arrays = {name: archive[name].copy() for name in archive.files}
    cached_prediction_read_seconds = time.perf_counter() - read_started
    del model, optimizer
    (model, optimizer, baseline, second_load), second_restore_seconds = parent.timed(
        restore_selected, case, gains, saved, device
    )
    (check_training, check_logits, check_labels), reloaded_train_seconds = parent.timed(
        evaluate, model, tx, ty
    )
    (development, dev_logits, dev_labels), reloaded_dev_seconds = parent.timed(
        evaluate, model, vx, vy
    )
    integrity_started = time.perf_counter()
    parent.assert_prediction_arrays(
        train_arrays,
        {
            "logits": check_logits,
            "labels": check_labels,
            "row_ids": np.arange(1000000, 1100000, dtype=np.int64),
        },
    )
    parent.assert_prediction_arrays(
        dev_arrays,
        {
            "logits": dev_logits,
            "labels": dev_labels,
            "row_ids": np.arange(1100000, 1150000, dtype=np.int64),
        },
    )
    exact.assert_equal(training, check_training, "same-selected-train-metrics")
    exact.assert_equal(
        best["development"], development, "same-selected-development-metrics"
    )
    assert_after_evaluation(model, optimizer, saved, baseline)
    second_integrity_seconds = time.perf_counter() - integrity_started
    # No writes into source_case_root when it belongs to failed-parent evidence.
    write_started = time.perf_counter()
    np.savez_compressed(
        destination_output / "selected-train-predictions.npz", **train_arrays
    )
    np.savez_compressed(
        destination_output / "selected-dev-predictions.npz", **dev_arrays
    )
    prediction_write_seconds = time.perf_counter() - write_started
    record = {
        "case": case,
        "amendment_identity": amendment_identity,
        "source_case_root": str(source_case_root),
        "checkpoint_sha256": best["checkpoint_sha256"],
        "source_checkpoint_sha256": best["checkpoint_sha256"],
        "source_cached_development_predictions_sha256": best[
            "development_predictions_sha256"
        ],
        "epoch": best["epoch"],
        "training": training,
        "development": development,
        "gap_dev_minus_train": {
            k: development[k] - training[k] for k in ("loss", "accuracy", "roc_auc")
        },
        "array_bytes_equal_after_exact_full_state_reload": True,
        "native_loader_transition_qualified": [first_load, second_load],
        "model_optimizer_gradients_and_postload_metadata_unchanged_by_evaluation": True,
        "prediction_arrays": {
            "train": {k: parent.array_record(v) for k, v in train_arrays.items()},
            "development": {k: parent.array_record(v) for k, v in dev_arrays.items()},
        },
        "evaluation_seconds": {
            "selected_train": train_seconds,
            "reloaded_selected_train": reloaded_train_seconds,
            "reloaded_selected_development": reloaded_dev_seconds,
        },
        "restore_and_postprocessing_seconds": {
            "checkpoint_read_hash_admission": checkpoint_read_admission_seconds,
            "first_model_optimizer_restore_and_exact_admission": first_restore_seconds,
            "second_model_optimizer_restore_and_exact_admission": second_restore_seconds,
            "cached_development_prediction_read": cached_prediction_read_seconds,
            "first_post_eval_state_integrity": first_integrity_seconds,
            "reloaded_post_eval_state_and_array_integrity": second_integrity_seconds,
            "prediction_serialization": prediction_write_seconds,
        },
        "selected_evaluation_total_seconds_inclusive": time.perf_counter()
        - total_started,
        "timing_scope": "synchronized evaluation and restore intervals separate; total is inclusive/nonadditive; first selected_train may include lazy native compilation/placement initialization; no complete adoption-time or throughput claim",
        "parent_raw_artifacts_modified": False,
        "selection_uses_training_evaluation": False,
        "heldout_evaluated": False,
    }
    write(destination_output / "selected-evaluation.json", record)
    return record
