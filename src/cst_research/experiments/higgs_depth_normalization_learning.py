"""Bounded three-epoch followup using unchanged4f models and bare975 steps."""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

from . import higgs_depth_normalization as model_contract
from . import higgs_depth_normalization_learning_evidence as evidence
from . import higgs_depth_normalization_run as prior_runtime
from . import higgs_depth_normalization_steps as old_observations
from . import higgs_depth_study as stock
from .higgs_depth_data import DATA_SHA, MANIFEST_SHA, load_pilot
from .higgs_dst_comparison import file_sha, json_sha, read, write

COHORT = "higgs-depth-normalization-learning-v1"
CASES = tuple(model_contract.CASES)


def protocol():
    return {
        "version": COHORT,
        "model_contract_source": evidence.PRIOR_COMMIT,
        "model_contract_protocol_sha256": evidence.PRIOR_PROTOCOL,
        "external_source_commit": stock.SOURCE,
        "cases": list(CASES),
        "architecture": [28, 1024, 1024, 1024, 1024, 2],
        "seed": 17,
        "epochs": 3,
        "batch_size": 4,
        "training_rows": 100000,
        "development_rows": 50000,
        "updates_per_epoch": 25000,
        "updates_per_case": 75000,
        "main_updates_all9": 675000,
        "admission_prefix_updates_per_case": 8,
        "admission_prefix_updates_all9": 72,
        "epoch_order_seeds": [117, 118, 119],
        "mode": "eager bare native ordinary_step",
        "lr": 0.0003,
        "position_multiplier": 1,
        "FP32": True,
        "TF32": False,
        "norm_eps": 1e-5,
        "normalization_affine": False,
        "optimizer": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "ordinary_weight_decay": 0.01,
            "amplitude_weight_decay": 0.01,
            "atom_raw_weight_decay": 0.0,
            "fused": True,
            "amsgrad": False,
            "maximize": False,
            "capturable": False,
        },
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "runtime_pin_sha256": evidence.RUNTIME_SHA,
        "qualification": "INHERITED_4f_QUALIFICATION_NOT_NEW_NATIVE_ORACLE; readonly full151 hashes and source24/runtime/data binding; fresh bare actual-prefix exact replay all9 then fresh0",
        "prefix_comparison": "before params, logits/CE, model/moments/clocks/gains and gradients at1/2/4/8; require prior raw==prepared and compare new post-step grad toboth; no hooks",
        "selection": "earliest minimum full-development CE at epochs1..3 only",
        "retained": "one state0 and epoch1..3; best checkpoint reference; all3dev predictions; selected fulltrain100k/dev50k logits labels rowIDs; exact selected-state reload arraybytes gate",
        "diagnostics": "outside clock fixed256 training-feature forward RMS/rho plus last actual task minibatch retained parameter gradients and native moment/clock summaries; no single-update inference from epoch endpoints",
        "timing": "synchronized ordinary_step loop only; preparation/calibration/admission/evaluation/diagnostics/checkpoint/hash separate",
        "HPO": False,
        "heldout_evaluated": False,
        "official_test_evaluated": False,
        "formal_NI": False,
        "automatic_followup": False,
        "proposed_foreground_timeout": 7200,
        "interpretation": "adaptive single-seed development followup; norm changes function map, fixedgain equivalence remains ideal-real; no minimumA/N/depth law or default adoption",
    }


def selected_epoch(rows):
    import math

    if len(rows) != 3 or [r["epoch"] for r in rows] != [1, 2, 3]:
        raise ValueError("complete three-epoch selection inventory required")
    if any(not math.isfinite(r["development"]["loss"]) for r in rows):
        raise ValueError("nonfinite selection loss")
    return min(rows, key=lambda r: (r["development"]["loss"], r["epoch"]))


def core_state(checkpoint):
    return {
        key: checkpoint[key]
        for key in ("model", "optimizer", "gradients", "epoch", "updates")
    }


def load_prior(prior_root, proof, name):
    import torch

    record = proof["files_verified_in_place"][name]
    path = evidence.safe_member(prior_root, name)
    if file_sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError("inherited raw member changed before read: " + name)
    return torch.load(path, map_location="cpu", weights_only=True)


def fresh(case, gains, device):
    model = model_contract.make_model(case, gains=gains).to(device)
    optimizer, contract = stock.make_optimizer(model)
    return model, optimizer, contract


def assert_initial(
    model, optimizer, expected, case, calibration, prior_identity, contract
):
    config = expected["configuration"]
    evidence.require(
        config["cohort"] == model_contract.COHORT
        and config["architecture"] == protocol()["architecture"]
        and config["inherited_factory_configuration"] == model._depth_configuration
        and config["optimizer_contract"] == contract
        and config["case"] == case
        and config["identity"] == prior_identity
        and config["normalization"] == model_contract.CASES[case][2]
        and config["gains"] == calibration["gains"],
        "prior model-contract configuration differs",
    )
    actual = stock.checkpoint(model, optimizer, 0, 0, {}, None)
    evidence.assert_equal(core_state(actual), core_state(expected), "fresh-state0")
    evidence.require(len(optimizer.state) == 0, "fresh0 has moments/clocks")
    return evidence.fingerprint(core_state(actual))


def reproduce_prefix(
    case, calibration, proof, prior_root, tx, ty, device, output, new_identity
):
    import torch
    from benchmarks.training.full_step_graph import ordinary_step

    model, optimizer, contract = fresh(case, calibration["gains"], device)
    state0 = load_prior(prior_root, proof, f"result/science/{case}/state-0.pt")
    initial = assert_initial(
        model,
        optimizer,
        state0,
        case,
        calibration,
        proof["producer_identity"],
        contract,
    )
    order = torch.randperm(100000, generator=torch.Generator().manual_seed(117))[
        :32
    ].to(device)
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    reports = []
    for step in range(1, 9):
        retained = step in (1, 2, 4, 8)
        before = old_observations.parameters(model) if retained else None
        logits, loss = ordinary_step(
            model, optimizer, tx, ty, order[4 * (step - 1) : 4 * step], totals
        )
        if not retained:
            continue
        expected = load_prior(
            prior_root, proof, f"result/science/{case}/update-{step}.pt"
        )
        actual = stock.checkpoint(model, optimizer, 0, step, {}, None)
        try:
            indices = order[4 * (step - 1) : 4 * step]
            evidence.assert_equal(
                tx[indices].detach().cpu(), expected["inputs"], "task-inputs"
            )
            evidence.assert_equal(
                ty[indices].detach().cpu(), expected["labels"], "task-labels"
            )
            evidence.assert_equal(
                before, expected["before_parameters"], "before-update"
            )
            evidence.assert_equal(
                logits.detach().cpu(), expected["logits"], "task-logits"
            )
            evidence.assert_equal(loss.detach().cpu(), expected["task_CE"], "task-CE")
            evidence.assert_equal(
                expected["raw_gradients"],
                expected["prepared_gradients"],
                "prior-flat-raw-prepared-identity",
            )
            evidence.assert_equal(
                actual["gradients"],
                expected["raw_gradients"],
                "new-post-step-gradient-to-prior-task-gradient",
            )
            evidence.assert_equal(
                actual["gradients"],
                expected["prepared_gradients"],
                "new-post-step-gradient-to-prior-prepared-gradient",
            )
            evidence.assert_equal(
                core_state(actual),
                core_state(expected["accepted"]),
                "accepted-native-state",
            )
        except BaseException:
            torch.save(
                {
                    "actual": actual,
                    "before_parameters": before,
                    "logits": logits.detach().cpu(),
                    "loss": loss.detach().cpu(),
                    "expected_member": f"result/science/{case}/update-{step}.pt",
                    "new_learning_identity": new_identity,
                },
                output / f"{case}-reproduction-failure.pt",
            )
            raise
        reports.append(
            {
                "step": step,
                "status": "all-compared-bytes-exact",
                "actual_state": evidence.fingerprint(core_state(actual)),
                "actual_before_parameters": evidence.fingerprint(before),
                "actual_logits": evidence.fingerprint(logits.detach().cpu()),
                "actual_CE": evidence.fingerprint(loss.detach().cpu()),
                "prior_member_sha256": proof["files_verified_in_place"][
                    f"result/science/{case}/update-{step}.pt"
                ]["sha256"],
            }
        )
        del expected, actual
    record = {
        "status": "passed-fresh-bare-task-prefix-reproduction",
        "case": case,
        "new_learning_identity": new_identity,
        "inherited_producer_identity": proof["producer_identity"],
        "fresh_independent_oracle_claim": False,
        "bare_step_hooks_used": False,
        "gradient_scope": "new retained post-step.grad matches both prior raw and prepared after explicit prior identity check; not a new pre-hook capture",
        "fresh_state0": initial,
        "retained_steps": reports,
        "replay_updates": 8,
        "replay_state_used_for_main_training": False,
    }
    write(output / f"{case}.json", record)
    del model, optimizer, state0
    gc.collect()
    torch.cuda.empty_cache()
    return record


def diagnostics(model, optimizer, probe):
    import torch

    values = {str(i): [] for i in range(len(model))}
    with torch.no_grad():
        for batch in probe.split(4):
            for i, value in enumerate(model_contract.trace(model, batch)):
                values[str(i)].append(value.detach().cpu())
    stats = {i: stock.tensor_statistics(torch.cat(v)) for i, v in values.items()}
    return {
        "probe_rows": 256,
        "activations": stats,
        "relu": {
            str(i): {
                "rho": (stats[str(i + 1)]["rms"] / stats[str(i)]["rms"]) ** 2
                if stats[str(i)]["rms"]
                else None,
                "positive_fraction": 1 - stats[str(i + 1)]["zero_fraction"],
            }
            for i in (2, 4, 6)
        },
        "last_actual_minibatch_gradients": {
            name: None if p.grad is None else stock.tensor_statistics(p.grad)
            for name, p in model.named_parameters()
        },
        "accepted_native_moments": {
            name: {
                key: stock.tensor_statistics(value)
                for key, value in optimizer.state.get(p, {}).items()
                if key in ("exp_avg", "exp_avg_sq", "step")
            }
            for name, p in model.named_parameters()
        },
        "scope": "fixed256 TRAIN features; current forward + last training B4 retained gradients/accepted moments, no backward probe or epoch-endpoint one-step displacement",
    }


def timed(function, *args):
    import torch

    torch.cuda.synchronize()
    begin = time.perf_counter()
    value = function(*args)
    torch.cuda.synchronize()
    return value, time.perf_counter() - begin


def array_record(array):
    import hashlib

    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def assert_prediction_arrays(actual, expected):
    evidence.require(
        actual.keys() == expected.keys(), "prediction array inventory differs"
    )
    for name in actual:
        evidence.require(
            array_record(actual[name]) == array_record(expected[name]),
            "prediction arraybytes differ: " + name,
        )


def restore_selected(case, gains, raw, device):
    model, optimizer, _ = fresh(case, gains, device)
    model.load_state_dict(raw["model"])
    optimizer.load_state_dict(raw["optimizer"])
    for name, p in model.named_parameters():
        p.grad = (
            None
            if raw["gradients"][name] is None
            else raw["gradients"][name].to(device).clone()
        )
    evidence.assert_equal(
        core_state(
            stock.checkpoint(model, optimizer, raw["epoch"], raw["updates"], {}, None)
        ),
        core_state(raw),
        "selected-full-state-restore",
    )
    return model, optimizer


def evaluate_selected(case, gains, best, root, tx, ty, vx, vy, device):
    import numpy as np
    import torch
    from benchmarks.training.higgs_confirmation import evaluate

    checkpoint = root / best["checkpoint_file"]
    evidence.require(
        file_sha(checkpoint) == best["checkpoint_sha256"],
        "selected checkpoint bytes changed",
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model, optimizer = restore_selected(case, gains, saved, device)
    (train_metrics, train_logits, train_labels), first_train_time = timed(
        evaluate, model, tx, ty
    )
    evidence.assert_equal(
        core_state(
            stock.checkpoint(
                model, optimizer, saved["epoch"], saved["updates"], {}, None
            )
        ),
        core_state(saved),
        "selected-evaluation-mutated-state",
    )
    evidence.require(
        file_sha(root / best["development_predictions_file"])
        == best["development_predictions_sha256"],
        "selected development prediction bytes changed",
    )
    with np.load(
        root / best["development_predictions_file"], allow_pickle=False
    ) as archive:
        dev_arrays = {name: archive[name].copy() for name in archive.files}
    train_arrays = {
        "logits": train_logits,
        "labels": train_labels,
        "row_ids": np.arange(1000000, 1100000, dtype=np.int64),
    }
    del model, optimizer
    model, optimizer = restore_selected(case, gains, saved, device)
    (check_train, check_logits, check_labels), reload_train_time = timed(
        evaluate, model, tx, ty
    )
    (dev_metrics, dev_logits, dev_labels), reload_dev_time = timed(
        evaluate, model, vx, vy
    )
    assert_prediction_arrays(
        train_arrays,
        {
            "logits": check_logits,
            "labels": check_labels,
            "row_ids": np.arange(1000000, 1100000, dtype=np.int64),
        },
    )
    assert_prediction_arrays(
        dev_arrays,
        {
            "logits": dev_logits,
            "labels": dev_labels,
            "row_ids": np.arange(1100000, 1150000, dtype=np.int64),
        },
    )
    evidence.assert_equal(train_metrics, check_train, "same-selected-train-metrics")
    evidence.assert_equal(
        best["development"], dev_metrics, "same-selected-development-metrics"
    )
    evidence.assert_equal(
        core_state(
            stock.checkpoint(
                model, optimizer, saved["epoch"], saved["updates"], {}, None
            )
        ),
        core_state(saved),
        "reload-evaluation-mutated-state",
    )
    np.savez_compressed(root / "selected-train-predictions.npz", **train_arrays)
    shutil.copy2(
        root / best["development_predictions_file"],
        root / "selected-dev-predictions.npz",
    )
    record = {
        "checkpoint_sha256": best["checkpoint_sha256"],
        "epoch": best["epoch"],
        "training": train_metrics,
        "development": dev_metrics,
        "gap_dev_minus_train": {
            key: dev_metrics[key] - train_metrics[key]
            for key in ("loss", "accuracy", "roc_auc")
        },
        "array_bytes_equal_after_exact_full_state_reload": True,
        "model_optimizer_gradients_unchanged_by_evaluation": True,
        "prediction_arrays": {
            "train": {k: array_record(v) for k, v in train_arrays.items()},
            "development": {k: array_record(v) for k, v in dev_arrays.items()},
        },
        "evaluation_seconds": {
            "selected_train": first_train_time,
            "reloaded_selected_train": reload_train_time,
            "reloaded_selected_development": reload_dev_time,
        },
        "selection_uses_training_evaluation": False,
        "heldout_evaluated": False,
    }
    write(root / "selected-evaluation.json", record)
    del model, optimizer, saved
    return record


def train_epoch(model, optimizer, tx, ty, indices, totals):
    """Bare loop; peak covers this training interval, before any checkpoint/eval."""
    import torch
    from benchmarks.training.full_step_graph import ordinary_step

    if indices.numel() % 4:
        raise ValueError("complete B4 epoch indices required")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    for step in range(indices.numel() // 4):
        ordinary_step(
            model, optimizer, tx, ty, indices[4 * step : 4 * step + 4], totals
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    return {
        "training_seconds": seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_scope": "this epoch training interval; resident train/dev arrays, model/optimizer/native buffers; reserved allocator blocks may persist from previous evaluation; no Graph; not process memory",
    }


def nonfinite_checkpoint_fields(checkpoint):
    from benchmarks.training.higgs_verify import finite_fields, tensor_fields

    return finite_fields(dict(tensor_fields(checkpoint)))


def run_case(
    case, calibration, proof, prior_root, identity, tx, ty, vx, vy, device, output
):
    import math

    import numpy as np
    import torch
    from benchmarks.training.higgs_confirmation import evaluate
    from benchmarks.training.higgs_verify import tensor_hash

    output.mkdir()
    begin = time.perf_counter()
    model, optimizer, contract = fresh(case, calibration["gains"], device)
    prior0 = load_prior(prior_root, proof, f"result/science/{case}/state-0.pt")
    assert_initial(
        model,
        optimizer,
        prior0,
        case,
        calibration,
        proof["producer_identity"],
        contract,
    )
    config = {
        "learning_cohort": COHORT,
        "learning_identity": identity,
        "case": case,
        "inherited_model_contract_cohort": model_contract.COHORT,
        "inherited_model_contract_source": evidence.PRIOR_COMMIT,
        "inherited_factory_configuration": model._depth_configuration,
        "gains": calibration["gains"],
        "normalization": model_contract.CASES[case][2],
        "optimizer_contract": contract,
        "seed": 17,
        "epochs": 3,
        "batch_size": 4,
    }
    torch.save(
        stock.checkpoint(model, optimizer, 0, 0, config, None), output / "state-0.pt"
    )
    initial, initial_diagnostic_seconds = timed(diagnostics, model, optimizer, tx[:256])
    preparation_seconds = time.perf_counter() - begin
    rows = []
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    for epoch in range(1, 4):
        model.train()
        print(f"{case} epoch{epoch}/3 starting", flush=True)
        start = time.perf_counter()
        order = torch.randperm(
            100000, generator=torch.Generator().manual_seed(116 + epoch)
        )
        order_sha = tensor_hash(order)["sha256"]
        indices = order.to(device)
        for value in totals:
            value.zero_()
        torch.cuda.synchronize()
        epoch_preparation_seconds = time.perf_counter() - start
        training_record = train_epoch(model, optimizer, tx, ty, indices, totals)
        updates = epoch * 25000
        start = time.perf_counter()
        raw = stock.checkpoint(model, optimizer, epoch, updates, config, None)
        if (
            nonfinite_checkpoint_fields(raw)
            or len(optimizer.state) != 7
            or any(float(s["step"]) != updates for s in optimizer.state.values())
        ):
            torch.save(raw, output / f"failed-epoch-{epoch}.pt")
            raise FloatingPointError("learning finite/ownership/clock gate failed")
        checkpoint_snapshot_admission_seconds = time.perf_counter() - start
        (development, logits, labels), dev_seconds = timed(evaluate, model, vx, vy)
        diagnostic, diagnostic_seconds = timed(diagnostics, model, optimizer, tx[:256])
        start = time.perf_counter()
        evidence.assert_equal(
            core_state(stock.checkpoint(model, optimizer, epoch, updates, {}, None)),
            core_state(raw),
            "epoch-evaluation-diagnostic-mutation",
        )
        post_evaluation_integrity_seconds = time.perf_counter() - start
        raw["development"] = development
        online_loss = float(totals[0]) / 100000
        if not math.isfinite(online_loss) or not all(
            math.isfinite(development[k])
            for k in ("loss", "accuracy", "roc_auc", "logit_rms")
        ):
            torch.save(raw, output / f"failed-epoch-{epoch}.pt")
            raise FloatingPointError("nonfinite learning/evaluation metric")
        start = time.perf_counter()
        checkpoint_file = f"epoch-{epoch}.pt"
        prediction_file = f"development-epoch-{epoch}.npz"
        torch.save(raw, output / checkpoint_file)
        np.savez_compressed(
            output / prediction_file,
            logits=logits,
            labels=labels,
            row_ids=np.arange(1100000, 1150000, dtype=np.int64),
        )
        io_seconds = time.perf_counter() - start
        start = time.perf_counter()
        checkpoint_sha = file_sha(output / checkpoint_file)
        prediction_sha = file_sha(output / prediction_file)
        hash_seconds = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "updates": updates,
            "online_training_loss": online_loss,
            "online_training_accuracy": float(totals[1]) / 100000,
            "development": development,
            "order_sha256": order_sha,
            **training_record,
            "epoch_preparation_seconds": epoch_preparation_seconds,
            "development_evaluation_seconds": dev_seconds,
            "diagnostics_seconds": diagnostic_seconds,
            "checkpoint_snapshot_admission_seconds": checkpoint_snapshot_admission_seconds,
            "post_evaluation_integrity_seconds": post_evaluation_integrity_seconds,
            "checkpoint_prediction_IO_seconds": io_seconds,
            "checkpoint_prediction_hash_seconds": hash_seconds,
            "checkpoint_file": checkpoint_file,
            "checkpoint_sha256": checkpoint_sha,
            "development_predictions_file": prediction_file,
            "development_predictions_sha256": prediction_sha,
            "diagnostics": diagnostic,
        }
        rows.append(row)
        # Prefix selection is descriptive progress only; fixed epochs never change.
        prefix_best = min(rows, key=lambda r: (r["development"]["loss"], r["epoch"]))
        write(
            output / "learning.json",
            {
                "cohort": COHORT,
                "configuration": config,
                "initial_diagnostics": initial,
                "preparation_seconds": preparation_seconds,
                "preparation_seconds_scope": "inclusive case setup: model/optimizer construction, prior state0 read/admission, new state0 serialization and initial diagnostics; initial_diagnostic_seconds is a component already included here, not additive",
                "initial_diagnostic_seconds": initial_diagnostic_seconds,
                "epochs": rows,
                "development_selected_epoch_so_far": prefix_best["epoch"],
                "hard_epochs": 3,
                "timing_scope": protocol()["timing"],
                "heldout_evaluated": False,
            },
        )
        print(
            f"{case} epoch{epoch}/3 updates{updates} devCE={development['loss']:.9g}",
            flush=True,
        )
        write(
            output.parent.parent / "progress.json",
            {"stage": "learning", "case": case, "epoch": epoch, "updates": updates},
        )
        del raw
    best = selected_epoch(rows)
    write(
        output / "best-dev.json",
        {
            key: best[key]
            for key in (
                "epoch",
                "updates",
                "development",
                "checkpoint_file",
                "checkpoint_sha256",
                "development_predictions_file",
                "development_predictions_sha256",
            )
        },
    )
    del model, optimizer, prior0
    gc.collect()
    torch.cuda.empty_cache()
    selected = evaluate_selected(
        case, calibration["gains"], best, output, tx, ty, vx, vy, device
    )
    complete = {
        "status": "completed",
        "case": case,
        "identity": identity,
        "updates": 75000,
        "epochs": 3,
        "selected_epoch": best["epoch"],
        "best_reference_sha256": file_sha(output / "best-dev.json"),
        "selected_evaluation_sha256": file_sha(output / "selected-evaluation.json"),
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
        "heldout_evaluated": False,
        "automatic_followup": False,
        "selected_evaluation": selected,
    }
    write(output / "complete.json", complete)
    gc.collect()
    torch.cuda.empty_cache()
    return complete


def run(args):
    from types import SimpleNamespace

    from . import higgs_dst_confirmation_driver as common
    from .higgs_coordinate_lr import verify_external

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
            identity={"external_source_manifest_sha256": stock.SOURCE_MANIFEST},
        )
    )
    sys.path[:0] = [str(args.external_source / "src"), str(args.external_source)]
    import torch
    from benchmarks.training.higgs_confirmation import _configure

    _configure()
    evidence.require(
        file_sha(args.inputs / "runtime-pin.json") == evidence.RUNTIME_SHA,
        "unchanged4090 runtime pin required",
    )
    runtime = prior_runtime.runtime_fingerprint()
    prior_runtime.validate_runtime(runtime)
    evidence.require(
        runtime == read(args.inputs / "runtime-pin.json"),
        "learning runtime differs from inherited4090 qualification",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    proof = evidence.verify_prior(
        args.inputs, args.prior_root, consumer, args.external_source, runtime, CASES
    )
    admission_seconds = time.perf_counter() - start
    identity = {
        "learning_cohort": COHORT,
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "protocol_sha256": json_sha(protocol()),
        "model_contract_source_commit": evidence.PRIOR_COMMIT,
        "inherited_qualification_identity": proof["producer_identity"],
        "external_source_commit": stock.SOURCE,
        "external_source_manifest_sha256": stock.SOURCE_MANIFEST,
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "runtime_pin_sha256": evidence.RUNTIME_SHA,
        "runtime_fingerprint": runtime,
    }
    write(
        args.output / "run-manifest.json",
        {
            "identity": identity,
            "protocol": protocol(),
            "status": "started",
            "argv": sys.argv,
        },
    )
    write(args.output / "inherited-qualification.json", proof)
    arrays, manifest = load_pilot(
        args.inputs / "train-development.npz", args.inputs / "data-manifest.json"
    )
    features = torch.from_numpy(arrays["x_train"])
    start = time.perf_counter()
    dense = model_contract.make_model("dense-stock")
    calibrations = {}
    for case in CASES:
        model = model_contract.make_model(case)
        calibration = model_contract.calibrate(dense, model, features[:256])
        evidence.assert_equal(
            calibration, proof["calibrations"][case], "fresh-calibration-to-actual4f"
        )
        calibrations[case] = calibration
    calibration_seconds = time.perf_counter() - start
    write(
        args.output / "calibration.json",
        {
            "new_learning_identity": identity,
            "model_contract_calibration_protocol_sha256": evidence.PRIOR_PROTOCOL,
            "cases": calibrations,
            "fresh_equal_actual4f": True,
            "calibration_seconds": calibration_seconds,
        },
    )
    del dense, model
    device = torch.device("cuda:0")
    start = time.perf_counter()
    tx = features.to(device)
    ty = torch.from_numpy(arrays["y_train"]).to(device)
    vx = torch.from_numpy(arrays["x_development"]).to(device)
    vy = torch.from_numpy(arrays["y_development"]).to(device)
    torch.cuda.synchronize()
    data_preparation_seconds = time.perf_counter() - start
    prefix = args.output / "prefix-reproduction"
    prefix.mkdir()
    reproductions = {}
    start = time.perf_counter()
    for case in CASES:
        print("fresh prefix reproduction " + case, flush=True)
        write(
            args.output / "progress.json",
            {
                "stage": "fresh-prefix-admission",
                "case": case,
                "main_training_updates": 0,
            },
        )
        reproductions[case] = reproduce_prefix(
            case,
            calibrations[case],
            proof,
            args.prior_root,
            tx,
            ty,
            device,
            prefix,
            identity,
        )
    prefix_seconds = time.perf_counter() - start
    write(
        args.output / "all9-prefix-reproduction.json",
        {
            "status": "passed-all9-exact-new-bare-prefix",
            "cases": list(CASES),
            "replay_updates_per_case": 8,
            "new_main_training_updates": 0,
            "new_native_oracle_claim": False,
            "source_of_independent_oracle": "inherited4f qualification above",
            "seconds": prefix_seconds,
            "reports": {case: file_sha(prefix / f"{case}.json") for case in CASES},
        },
    )
    # Every model is created and compared to old state0 again inside run_case.
    case_root = args.output / "cases"
    case_root.mkdir()
    completed = []
    for case in CASES:
        write(
            args.output / "progress.json",
            {
                "stage": "fresh0-main-learning",
                "case": case,
                "completed_cases": [c["case"] for c in completed],
            },
        )
        completed.append(
            run_case(
                case,
                calibrations[case],
                proof,
                args.prior_root,
                identity,
                tx,
                ty,
                vx,
                vy,
                device,
                case_root / case,
            )
        )
    write(
        args.output / "run-summary.json",
        {
            "status": "completed",
            "identity": identity,
            "cases": completed,
            "main_updates_per_case": 75000,
            "main_updates_total": 675000,
            "admission_prefix_updates_total": 72,
            "inherited_qualification_kind": proof["kind"],
            "fresh_prefix_reproduction_passed_all9": True,
            "preparation_seconds": {
                "prior_evidence_admission": admission_seconds,
                "calibration": calibration_seconds,
                "data_preparation": data_preparation_seconds,
                "fresh_prefix_reproduction": prefix_seconds,
            },
            "data_exposure": manifest["exposure"],
            "heldout_evaluated": False,
            "formal_NI": False,
            "automatic_followup": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inputs",
        "prior-root",
        "consumer-manifest",
        "external-source",
        "external-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--consumer-commit", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
