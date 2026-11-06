"""StageB proposal: fresh0 sixteen-case one-epoch NONE pilot, actual StageA proof gated."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import higgs_depth_normalization_learning as helpers
from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_depth_normalization_run as native_runtime
from . import higgs_depth_study as stock
from . import higgs_dst_confirmation_driver as common
from . import higgs_linear_atom_learning_evidence as evidence
from . import higgs_linear_atom_scaling as model_contract
from . import higgs_linear_atom_state as state
from .higgs_depth_data import DATA_SHA, MANIFEST_SHA, load_pilot
from .higgs_dst_comparison import file_sha, json_sha, read, write

COHORT = "higgs-linear-atom-learning-stageB-v1"
ENTRY = "src/cst_research/experiments/higgs_linear_atom_learning_driver.py"


def protocol():
    return {
        "version": COHORT,
        "study": "higgs-linear-atom-scaling-20261106",
        "status": "PREPARED_NOT_EXECUTED; StageA actual16pass still required",
        "stage_A_consumer_commit": evidence.PARENT_COMMIT,
        "stage_A_consumer_manifest_sha256": evidence.PARENT_MANIFEST,
        "stage_A_protocol_sha256": evidence.PARENT_PROTOCOL,
        "all114_StageA_files_unchanged": True,
        "cases": model_contract.pilot_cases(),
        "main_recipe": "NONE; unit gains; no calibration; unchanged legacy initializer; q1",
        "seed": 17,
        "lr": 0.0003,
        "batch_size": 4,
        "epochs_per_case": 1,
        "updates_per_case": 25000,
        "main_updates": 400000,
        "epoch1_order_seed": 117,
        "main_start": "freshinitial0 byteexact to actuallyqualified StageA0; no state8/replay/trainingresume",
        "native_admission": "actualStageA full16 independentCPUreport + normalouter/fullraw/all114source/runtime/data exact; inheritedidentity retained, no freshoracle claim",
        "new_qualification_or_replay_updates": 0,
        "inherited_qualification_updates": "16x8syntheticnormal plus16x4resume, NOT tasktraining",
        "selection": "only fixedepoch1 available; no HPO/earlystop/automaticfollowup",
        "selected_evaluation": "fullTRAIN100k andDEV50k, rawlogits/labels/rowIDs; twoindependent exactnativefullstate restores, trainarraybytes equal and DEVarraybytes equal cachedepoch1; actualPOSTLOAD mutationbaseline",
        "timing": "unchanged bareordinary_step completeepoch; prep/eval/diagnostics/snapshot/hash/I/O/restore/admission separate; inclusive totals nonadditive",
        "memory": "peak reset justbefore synced timedepoch, captured BEFORE checkpoint/eval; resident train/dev+model/optimizer/nativebuffers, reservedcache may persist, not processmemory",
        "diagnostics": "fixed256TRAINfeatures initial/end forwardRMS+ReLUretention, lastactualB4 gradients/acceptedm/v/clocks; outsideclock, no new backward or perstep copies",
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "runtime_pin_sha256": exact.RUNTIME_SHA,
        "new_raw_plus_StageA_qualified_raw_cap_bytes": 8 * 2**30,
        "storage": "owner freshfree minus conservativecompressed/maxscratch/primaryreserve leaves>=8GiB; preservecompressedraw and sequentialownedtmpCPUadmission; no evidence deletion",
        "heldout_evaluated": False,
        "formal_noninferiority_or_minimum_c_or_linear_runtime_claim": False,
        "conditional_normalization_or_gain_adoption": False,
        "automatic_resume_retry_or_advance": False,
    }


def verify_inputs(inputs, admission_sha, release_sha):
    from . import higgs_linear_atom_qualification as admission
    from .higgs_depth_data import validate_manifest

    pins = {
        **admission.input_pins(),
        "stage-a-admission.json": admission_sha,
        "stage-a-file-release.json": release_sha,
    }
    exact.require(
        {p.name for p in inputs.iterdir()} == {*pins, "consumer-source-manifest.json"}
        and all(p.is_file() and not p.is_symlink() for p in inputs.iterdir()),
        "exact10 regular StageB inputs; no heldout or unknown payload",
    )
    for name, digest in pins.items():
        evidence.sealed_sha(inputs / name, digest)
    validate_manifest(read(inputs / "data-manifest.json"))
    return {
        **pins,
        "consumer-source-manifest.json": file_sha(
            inputs / "consumer-source-manifest.json"
        ),
    }


def diagnostics(model, optimizer, probe):
    import torch

    record = stock.layer_diagnostics(model, probe)
    activation = record["activations"]
    record["relu"] = {
        str(i): {
            "rho": (activation[str(i + 1)]["rms"] / activation[str(i)]["rms"]) ** 2
            if activation[str(i)]["rms"]
            else None,
            "positive_fraction": 1 - activation[str(i + 1)]["zero_fraction"],
        }
        for i in model._linear_atom_configuration["internal_module_indices"]
    }
    record["accepted_moments"] = {
        name: {
            key: stock.tensor_statistics(value)
            for key, value in optimizer.state.get(point, {}).items()
            if key in ("step", "exp_avg", "exp_avg_sq")
        }
        for name, point in model.named_parameters()
    }
    exact.require(
        all(bool(torch.isfinite(p).all()) for p in model.parameters()),
        "diagnostic nonfinite parameter",
    )
    record["scope"] = (
        "NONE/unitgains fixed256TRAINfeatures; currentforward+lastactualB4gradients/acceptedmoments, no backwardsprobe/one-step-displacement inference"
    )
    return record


def current_core(model, optimizer, case, identity, epoch=1, updates=25000):
    return state.core_state(
        state.snapshot(model, optimizer, epoch, updates, case, identity)
    )


def selected_evaluation(
    case, raw, cached_dev, tx, ty, vx, vy, device, output, identity
):
    import numpy as np
    from benchmarks.training.higgs_confirmation import evaluate

    total_started = time.perf_counter()
    (model, optimizer, baseline, first), first_restore = helpers.timed(
        state.load_state, case, raw, device
    )
    (train_metrics, train_logits, train_labels), first_train = helpers.timed(
        evaluate, model, tx, ty
    )
    exact.assert_equal(
        current_core(model, optimizer, case, identity),
        baseline,
        "selected-first-evaluation-mutation",
    )
    train_arrays = {
        "logits": train_logits,
        "labels": train_labels,
        "row_ids": np.arange(1000000, 1100000, dtype=np.int64),
    }
    del model, optimizer
    (model, optimizer, baseline, second), second_restore = helpers.timed(
        state.load_state, case, raw, device
    )
    (check_train, check_logits, check_labels), second_train = helpers.timed(
        evaluate, model, tx, ty
    )
    (development, dev_logits, dev_labels), selected_dev = helpers.timed(
        evaluate, model, vx, vy
    )
    dev_arrays = {
        "logits": dev_logits,
        "labels": dev_labels,
        "row_ids": np.arange(1100000, 1150000, dtype=np.int64),
    }
    helpers.assert_prediction_arrays(
        train_arrays,
        {
            "logits": check_logits,
            "labels": check_labels,
            "row_ids": train_arrays["row_ids"],
        },
    )
    helpers.assert_prediction_arrays(dev_arrays, cached_dev)
    exact.assert_equal(train_metrics, check_train, "same-selected-train-metrics")
    exact.assert_equal(
        raw["development"], development, "same-selected-development-metrics"
    )
    exact.assert_equal(
        current_core(model, optimizer, case, identity),
        baseline,
        "selected-second-evaluation-mutation",
    )
    start = time.perf_counter()
    np.savez_compressed(output / "selected-train-predictions.npz", **train_arrays)
    # The selected DEV is exactly cached epoch1. Retain a SHA reference instead of duplicate NPZ bytes.
    io_seconds = time.perf_counter() - start
    record = {
        "case": case,
        "identity": identity,
        "epoch": 1,
        "updates": 25000,
        "training": train_metrics,
        "development": development,
        "gap_dev_minus_train": {
            k: development[k] - train_metrics[k]
            for k in ("loss", "accuracy", "roc_auc")
        },
        "native_load_proofs": [first, second],
        "arraybytes_equal_after_full_native_reload": True,
        "evaluation_did_not_mutate_actual_postload_core": True,
        "prediction_arrays": {
            "train": {k: helpers.array_record(v) for k, v in train_arrays.items()},
            "development": {k: helpers.array_record(v) for k, v in dev_arrays.items()},
        },
        "evaluation_seconds": {
            "selected_train": first_train,
            "reloaded_train": second_train,
            "reloaded_dev": selected_dev,
        },
        "restore_seconds": {"first": first_restore, "second": second_restore},
        "selected_train_prediction_IO_seconds": io_seconds,
        "selected_evaluation_total_seconds_inclusive": time.perf_counter()
        - total_started,
        "total_seconds_scope": "inclusive restores/evaluations/arraybytechecks/predictionIO/hashing; components notadditive; recordserialization follows",
        "selected_dev_predictions_file": "development-epoch-1.npz",
        "selected_train_predictions_sha256": file_sha(
            output / "selected-train-predictions.npz"
        ),
        "selection_uses_training_evaluation": False,
        "heldout_evaluated": False,
    }
    write(output / "selected-evaluation.json", record)
    del model, optimizer
    gc.collect()
    return record


def run_case(case, proof, parent_root, identity, tx, ty, vx, vy, order, device, output):
    import numpy as np
    import torch
    from benchmarks.training.higgs_confirmation import evaluate
    from benchmarks.training.higgs_verify import tensor_hash

    output.mkdir()
    begin = time.perf_counter()
    model = model_contract.make_case(case).to(device)
    optimizer, contract = state.make_optimizer(model)
    initial = evidence.load_initial(parent_root, proof, case)
    fresh = state.snapshot(model, optimizer, 0, 0, case, identity)
    exact.assert_equal(
        state.core_state(fresh),
        state.core_state(initial),
        "main-fresh0-all-core-bytes-to-qualified0",
    )
    exact.require(
        not optimizer.state and all(p.grad is None for p in model.parameters()),
        "main fresh0 contains gradient/moment history",
    )
    write(
        output / "initial-state-reference.json",
        {
            "source": proof["cases"][case["id"]],
            "inherited_identity": proof["producer_identity"],
            "main_identity": identity,
            "fresh0_core_fingerprint": exact.fingerprint(state.core_state(fresh)),
            "exact_fresh0_to_qualified0": True,
            "qualified_state_loaded_into_main_model": False,
        },
    )
    del initial, fresh
    initial_diagnostics, initial_diagnostic_seconds = helpers.timed(
        diagnostics, model, optimizer, tx[:256]
    )
    indices = order.to(device)
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    torch.cuda.synchronize()
    preparation = time.perf_counter() - begin
    print(case["id"] + " epoch1/1 starting; fresh0 admitted", flush=True)
    training = helpers.train_epoch(model, optimizer, tx, ty, indices, totals)
    start = time.perf_counter()
    raw = state.snapshot(model, optimizer, 1, 25000, case, identity)
    try:
        state.check_optimizer(model, optimizer, 25000)
        exact.require(
            not helpers.nonfinite_checkpoint_fields(raw),
            "nonfinite main model/gradient/m/v/clock",
        )
    except BaseException:
        torch.save(raw, output / "failed-epoch-1.pt")
        raise
    snapshot_seconds = time.perf_counter() - start
    (development, logits, labels), dev_seconds = helpers.timed(evaluate, model, vx, vy)
    final_diagnostics, diagnostic_seconds = helpers.timed(
        diagnostics, model, optimizer, tx[:256]
    )
    exact.assert_equal(
        current_core(model, optimizer, case, identity),
        state.core_state(raw),
        "main-epoch-eval-diagnostic-mutation",
    )
    raw["development"] = development
    online_loss = float(totals[0]) / 100000
    if not math.isfinite(online_loss) or not all(
        math.isfinite(v) for v in development.values()
    ):
        torch.save(raw, output / "failed-epoch-1.pt")
        np.savez_compressed(
            output / "failed-development-epoch-1.npz", logits=logits, labels=labels
        )
        raise FloatingPointError("nonfinite train/development metric")
    start = time.perf_counter()
    torch.save(raw, output / "epoch-1.pt")
    dev_arrays = {
        "logits": logits,
        "labels": labels,
        "row_ids": np.arange(1100000, 1150000, dtype=np.int64),
    }
    np.savez_compressed(output / "development-epoch-1.npz", **dev_arrays)
    io_seconds = time.perf_counter() - start
    start = time.perf_counter()
    checkpoint_sha = file_sha(output / "epoch-1.pt")
    prediction_sha = file_sha(output / "development-epoch-1.npz")
    hash_seconds = time.perf_counter() - start
    row = {
        "epoch": 1,
        "updates": 25000,
        "online_training_loss": online_loss,
        "online_training_accuracy": float(totals[1]) / 100000,
        "development": development,
        "order_sha256": tensor_hash(order)["sha256"],
        **training,
        "checkpoint_snapshot_admission_seconds": snapshot_seconds,
        "development_evaluation_seconds": dev_seconds,
        "final_diagnostics_seconds": diagnostic_seconds,
        "checkpoint_prediction_IO_seconds": io_seconds,
        "checkpoint_prediction_hash_seconds": hash_seconds,
        "checkpoint_file": "epoch-1.pt",
        "checkpoint_sha256": checkpoint_sha,
        "development_predictions_file": "development-epoch-1.npz",
        "development_predictions_sha256": prediction_sha,
        "diagnostics": final_diagnostics,
    }
    write(
        output / "learning.json",
        {
            "cohort": COHORT,
            "case": case,
            "identity": identity,
            "optimizer_contract": contract,
            "initial_diagnostics": initial_diagnostics,
            "preparation_seconds": preparation,
            "preparation_seconds_scope": "inclusive model/optimizer/fresh0admission/referenceproof/initialdiagnostics/ordertransfer; initial_diagnostic_seconds alreadyincluded, notadditive",
            "initial_diagnostic_seconds": initial_diagnostic_seconds,
            "epochs": [row],
            "hard_epochs": 1,
            "heldout_evaluated": False,
        },
    )
    write(
        output / "best-dev.json",
        {
            k: row[k]
            for k in (
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
    print(
        case["id"] + f" epoch1/1 complete25000 devCE={development['loss']:.9g}",
        flush=True,
    )
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    selected = selected_evaluation(
        case, raw, dev_arrays, tx, ty, vx, vy, device, output, identity
    )
    complete = {
        "status": "completed-one-fixedepoch-exploratory-case",
        "case": case,
        "identity": identity,
        "epochs": 1,
        "updates": 25000,
        "selected_epoch": 1,
        "selected_checkpoint_sha256": row["checkpoint_sha256"],
        "best_reference_sha256": file_sha(output / "best-dev.json"),
        "selected_evaluation_sha256": file_sha(output / "selected-evaluation.json"),
        "heldout_evaluated": False,
        "automatic_followup": False,
        "selected_evaluation": selected,
        "case_total_seconds_inclusive": time.perf_counter() - begin,
        "case_total_scope": "inclusive allcasepreparation/qualified0read/training/checkpoint/evaluation/diagnostics/restore/IO/hashing; components notadditive; completionserialization follows",
    }
    write(output / "complete.json", complete)
    del raw, dev_arrays
    gc.collect()
    torch.cuda.empty_cache()
    return complete


def run(args):
    from .higgs_coordinate_lr import verify_external

    science_started = time.perf_counter()
    consumer = Path(__file__).resolve().parents[3]
    manifest = common.verify_consumer(
        consumer,
        args.consumer_manifest,
        args.consumer_commit,
        consumer / common.DRIVER_PATH,
    )
    exact.require(
        manifest["files"][ENTRY] == file_sha(consumer / ENTRY),
        "executed StageB driver bytes differ",
    )
    inputs = verify_inputs(
        args.inputs, args.stage_a_admission_sha256, args.stage_a_file_release_sha256
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
    import torchcst
    from benchmarks.training.higgs_confirmation import _configure

    exact.require(
        Path(torchcst.__file__)
        .resolve()
        .is_relative_to(args.external_source.resolve()),
        "runtime outside verifiedpublic975",
    )
    _configure()
    runtime = native_runtime.runtime_fingerprint()
    exact.require(
        native_runtime.validate_runtime(runtime) == "RTX4090"
        and runtime == read(args.inputs / "runtime-pin.json"),
        "same actualStageA4090 runtime required",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    proof = evidence.verify_stage_a(
        args.stage_a_root,
        args.inputs / "stage-a-admission.json",
        args.stage_a_admission_sha256,
        args.inputs / "stage-a-file-release.json",
        args.stage_a_file_release_sha256,
        consumer,
        runtime,
    )
    admission_seconds = time.perf_counter() - begin
    identity = {
        "cohort": COHORT,
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "protocol_sha256": json_sha(protocol()),
        "inherited_native_qualification_identity": proof["producer_identity"],
        "StageA_independent_CPU_admission_sha256": args.stage_a_admission_sha256,
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "runtime_pin_sha256": exact.RUNTIME_SHA,
        "runtime_fingerprint": runtime,
        "external_source_commit": stock.SOURCE,
        "external_source_manifest_sha256": stock.SOURCE_MANIFEST,
        "verified_inputs": inputs,
    }
    write(
        args.output / "run-manifest.json",
        {"identity": identity, "protocol": protocol(), "argv": sys.argv},
    )
    write(args.output / "inherited-StageA-admission.json", proof)
    begin = time.perf_counter()
    arrays, _ = load_pilot(
        args.inputs / "train-development.npz", args.inputs / "data-manifest.json"
    )
    device = torch.device("cuda:0")
    tx, ty, vx, vy = (
        torch.from_numpy(arrays[k]).to(device)
        for k in ("x_train", "y_train", "x_development", "y_development")
    )
    order = torch.randperm(100000, generator=torch.Generator().manual_seed(117))
    torch.save(order, args.output / "epoch1-order.pt")
    data_seconds = time.perf_counter() - begin
    cases = args.output / "cases"
    cases.mkdir()
    completed = []
    for case in model_contract.pilot_cases():
        completed.append(
            run_case(
                case,
                proof,
                args.stage_a_root,
                identity,
                tx,
                ty,
                vx,
                vy,
                order,
                device,
                cases / case["id"],
            )
        )
        write(
            args.output / "progress.json",
            {
                "completed_case_ids": [r["case"]["id"] for r in completed],
                "main_updates": 25000 * len(completed),
                "fixed_total_main_updates": 400000,
            },
        )
        new_bytes = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
        qualified = read(
            exact.safe_member(args.stage_a_root, "result/science/run-summary.json")
        )["raw_bytes"]
        exact.require(
            new_bytes + qualified
            <= protocol()["new_raw_plus_StageA_qualified_raw_cap_bytes"],
            "combined8GiBrawcap exceeded; preservefailedpartial, no delete/retry",
        )
    evidence.strict_files(Path(args.stage_a_root), proof["files_verified_in_place"])
    write(
        args.output / "run-summary.json",
        {
            "status": "completed-bounded16-oneepoch-linear-Atom-exploratory-pilot",
            "identity": identity,
            "cases": completed,
            "main_updates": 400000,
            "new_qualification_or_replay_updates": 0,
            "task_training_updates_inherited": 0,
            "inherited_StageA_normal_plus_resume_synthetic_updates": 192,
            "prior_StageA_unchanged": True,
            "science_seconds_before_summary_inclusive": time.perf_counter()
            - science_started,
            "science_seconds_scope": "source/runtime/fullrawadmission/data/allcases/finalreadonlyhashcheck inclusive; no driverbootstrap/summaryserialization; components notadditive",
            "StageA_fullraw_admission_seconds": admission_seconds,
            "data_preparation_seconds": data_seconds,
            "heldout_evaluated": False,
            "automatic_followup": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-commit", required=True)
    for name in (
        "consumer-manifest",
        "external-source",
        "external-manifest",
        "inputs",
        "output",
        "stage-a-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("stage-a-admission-sha256", "stage-a-file-release-sha256"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
