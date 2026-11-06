"""Fresh A100 six-arm H8 conditioning diagnostic; fixedfinal150k task updates."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from . import higgs_depth_normalization_learning as helpers
from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_depth_normalization_run as native_runtime
from . import higgs_depth_study as stock
from . import higgs_dst_confirmation_driver as common
from . import higgs_multilayer_conditioning as study
from . import higgs_multilayer_conditioning_observation as observation
from . import higgs_multilayer_conditioning_qualification as qualification
from . import higgs_multilayer_conditioning_state as state
from .higgs_depth_data import DATA_SHA, MANIFEST_SHA, load_pilot, validate_manifest
from .higgs_dst_comparison import file_sha, json_sha, read, write

RUNTIME_SHA = "35016be129960614d829affbea959797ae8d035b5ea41197b6803ae963709f55"
ENTRY = "src/cst_research/experiments/higgs_multilayer_conditioning_driver.py"
RAW_CAP = 6 * 2**30
JSON_CAP = 64 * 2**20


def protocol():
    return {
        "version": study.COHORT,
        "study_id": "higgs-multilayer-conditioning-20261106",
        "status": "PREPARED_NOT_EXECUTED",
        "cases": list(study.CASES),
        "width": 1024,
        "hidden_layers": 8,
        "internal_biasfree_CST_maps": 7,
        "atoms_per_CST_map": 4096,
        "seed": 17,
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "runtime_pin_sha256": RUNTIME_SHA,
        "device": "exactA10080GBPCIeMIG3g.40gb/42SM/CC8.0/42144366592B",
        "normalization": "featurewise no-affineLN epsilon1e-5, orNONE; ordinaryDenseIO trainable",
        "gains": "bothCSTgainarms sequentiallymatch SAME initialDense-stockpreReLURMS onfixedfirst256TRAINfeatures; positiveFP32[1e-6,1e6], no labels/DEV; zero/nonfinite/outofbounds failwithoutclamp",
        "initialization": "unchanged public975 and explicitlegacy ordinary; pairedrawparams/IO; noamp-init intervention",
        "optimizer": "stockq1 lr.0003 betas(.9,.999) eps1e-8 ordinaryWD.01 amplitudeWD.01 rawatomWD0; FP32TF32off eagerB4",
        "epochs": 1,
        "task_updates_per_case": 25000,
        "total_task_updates": 150000,
        "orders": "actualsavedCPUtorchrandperm100000 seed117, samefor6cases",
        "milestones": list(study.MILESTONES),
        "milestone_evaluation": "FULLTRAIN100k+DEV50k atsameexactstate, outsidelearningblocks; no point0backward",
        "selection": "fixedfinal25000 ONLY; no bestDEV/earlystop/HPO/autofollowup",
        "qualification": {
            "scope": "newactual6H8 fullY/dX/all11parameterFP64manualreverse includingexactLN/gain +unchanged975chunkedallatoms256; independentquad31/scalarAdam beforetaskupdates",
            "scales": [1.0, 1e24],
            "normal_elementwise": "3e-5+3e-4*abs(ref), mandatory",
            "relative_fields_and_packed_columns": "FP64stable relativeL2 andmaxerror/refmax<=3e-4; exactzero strict; normalroundedzerotails disclosed notwaived",
            "amplified_material": "dX andallpackedcolumns anyabs(ref)>3e-5+3e-4abs(ref) AND anyactualnonzero; amplifiedcomponentfailures preserveddiagnostic",
            "amplified_optimizer_step": False,
            "updates_per_case": 20,
            "total_synthetic_updates": 120,
            "trajectory": "8ordinary vs4+native7ownerload+4, andobservedfirst+7ordinary vs8ordinary exactcorebytes; nooptimizer monkeypatch/backwardhooks",
        },
        "first_task_step": "actualfirstB4 CEbackward/rawP.grad +unchangedoptimizer.step acceptedP/m/v/clocks/poststep.grad; retainedinputs/labels/order/intermediateforward/upstreamgrads",
        "denominator": "accepted-state/source-FP32opmath reconstruction separatelyscalarqualified, NOT actualnativeintermediate; canonicalFlatTorus prepare/transportidentity; accepteddisplacement includesretraction/amplitudeWD",
        "endpoint_probe": "samefirstB4 onindependentloadedfinalcopy; onebackward,nooptimizerstep; allothercopystateexact andoriginalcheckpoint/model/optimizer/grad unchanged",
        "native_load": "exact7actualowners2/4/6/8/10/12/14 max(before+1,saved) only, everyothercorebyte literal; evalbaseline actualPOSTLOAD",
        "timing": "DIAGNOSTIC notbaselineperformancebenchmark; firstupdate/observation/eval/diagnostics/calibration/save/hash separated; baretrainingblocks includeprescribed1000step sync/logging, reportscomponents nonadditive",
        "rawretention": "qualifiedinitial0/4/8/resumed8/observer8; actualfirsttaskraw/fullfinalcheckpoint; milestonefullpredictions+typedstatefingerprints notfullintermediateoptimizercheckpoint; no independentrawmidstate replayclaim",
        "raw_cap_bytes": RAW_CAP,
        "JSON_cap_bytes": JSON_CAP,
        "retention_route": "immutablefullA100raw; compactproofs/predictionsretrieval; eithercompressedlocalactualspacefits OR separatepeerCPUauditorserver-sideCUDAoff/rootpinnedcode; no mandatorywholelocaldecompressedduplicate",
        "heldout_test_or_original_confirmation_evaluated": False,
        "formal_NI_Atomorder_capacity_or_LNnecessity_claim": False,
        "automatic_retry_resume_or_advance": False,
    }


def input_pins():
    from .higgs_depth_normalization_driver import ARCHIVE_SHA

    return {
        "original-source.tar.gz": ARCHIVE_SHA,
        "original-source-manifest.json": stock.SOURCE_MANIFEST,
        "train-development.npz": DATA_SHA,
        "data-manifest.json": MANIFEST_SHA,
        "runtime-pin.json": RUNTIME_SHA,
    }


def verify_inputs(inputs):
    pins = input_pins()
    exact.require(
        {p.name for p in inputs.iterdir()} == {*pins, "consumer-source-manifest.json"}
        and all(p.is_file() and not p.is_symlink() for p in inputs.iterdir()),
        "exact6regularinputs required; no heldout",
    )
    for name, digest in pins.items():
        exact.require(file_sha(inputs / name) == digest, "inputSHA differs: " + name)
    validate_manifest(read(inputs / "data-manifest.json"))
    return {
        **pins,
        "consumer-source-manifest.json": file_sha(
            inputs / "consumer-source-manifest.json"
        ),
    }


def runtime_gate(runtime, pin):
    exact.require(
        native_runtime.validate_runtime(runtime) == "A100-MIG3g.40gb"
        and runtime == pin,
        "exactpinnedA100MIG3g40 runtime required, no3090/4090 substitution",
    )


def budget(root):
    files = [p for p in root.rglob("*") if p.is_file()]
    raw = sum(p.stat().st_size for p in files)
    metadata = sum(p.stat().st_size for p in files if p.suffix == ".json")
    exact.require(
        raw <= RAW_CAP and metadata <= JSON_CAP,
        "retainedraw/JSON capexceeded; preservefailedpartial withoutdelete/retry",
    )
    return {"retained_raw_bytes": raw, "JSON_bytes": metadata}


def calibration_features(data):
    import numpy as np

    with np.load(data, allow_pickle=False) as archive:
        exact.require(
            set(archive.files)
            == {"x_train", "y_train", "x_development", "y_development"},
            "calibrationpayloadinventory differs",
        )
        features = archive["x_train"]
        exact.require(
            features.dtype == np.float32
            and features.shape == (100000, 28)
            and bool(np.isfinite(features[:256]).all()),
            "calibrationTRAINfeatures differ",
        )
        return features[:256].copy()


def activations(model, probe):
    import torch

    traces = {}
    with torch.no_grad():
        for batch in probe.split(4):
            for i, value in enumerate(study.trace(model, batch)):
                traces.setdefault(str(i), []).append(value.detach().cpu())
    stats = {k: stock.tensor_statistics(torch.cat(v)) for k, v in traces.items()}
    return {
        "fixed256TRAINfeatures": True,
        "activations": stats,
        "relu": {
            str(i): {
                "rho": (stats[str(i + 1)]["rms"] / stats[str(i)]["rms"]) ** 2
                if stats[str(i)]["rms"]
                else None,
                "positive_fraction": 1 - stats[str(i + 1)]["zero_fraction"],
            }
            for i in model._linear_atom_configuration["internal_module_indices"]
        },
        "backward_performed": False,
    }


def bare_block(model, optimizer, tx, ty, order, start, stop, totals, rolling, output):
    import torch
    from benchmarks.training.full_step_graph import ordinary_step

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    logging_seconds = 0.0
    previous_sum = rolling[-1]["cumulative_FP32_loss_sum"] if rolling else 0.0
    previous_correct = rolling[-1]["cumulative_correct"] if rolling else 0
    previous_update = rolling[-1]["update"] if rolling else 0
    for step in range(start, stop):
        ordinary_step(model, optimizer, tx, ty, order[4 * step : 4 * step + 4], totals)
        if (step + 1) % 1000 == 0:
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss_sum, correct = float(totals[0]), int(totals[1])
            rolling.append(
                {
                    "update": step + 1,
                    "cumulative_FP32_loss_sum": loss_sum,
                    "cumulative_correct": correct,
                    "window_loss": (loss_sum - previous_sum)
                    / (4 * (step + 1 - previous_update)),
                    "window_accuracy": (correct - previous_correct)
                    / (4 * (step + 1 - previous_update)),
                    "scope": "differenceofprescribedFP32cumulativetotals, descriptiveonlinepreupdateCE",
                }
            )
            previous_sum, previous_correct, previous_update = (
                loss_sum,
                correct,
                step + 1,
            )
            write(output / "online-progress.json", rolling)
            print(
                f"{model._conditioning_case} update{step + 1}/25000 onlineCE={rolling[-1]['window_loss']:.9g}",
                flush=True,
            )
            logging_seconds += time.perf_counter() - started
    torch.cuda.synchronize()
    return {
        "from_update": start,
        "to_update": stop,
        "bare_optimizer_updates": stop - start,
        "loop_seconds_inclusive_prescribed_1000step_logging": time.perf_counter()
        - begin,
        "logging_seconds_component": logging_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scope": "diagnostic eagerbareordinary_step interval; residenttrain/dev/model/optimizer/nativebuffers, allocatorreservation maypersist; notprocessmemory/notoldBthroughputcomparison",
    }


def prediction_arrays(kind, logits, labels):
    import numpy as np

    if kind not in ("train", "development"):
        raise ValueError("undeclared prediction split")
    first, count = (1000000, 100000) if kind == "train" else (1100000, 50000)
    exact.require(
        logits.shape == (count, 2)
        and logits.dtype == np.float32
        and labels.shape == (count,)
        and labels.dtype == np.int64,
        "full split binary-logit/label dtype/shape differs",
    )
    return {
        "logits": logits,
        "labels": labels,
        "row_ids": np.arange(first, first + count, dtype=np.int64),
    }


def evaluate_point(
    model, optimizer, update, identity, tx, ty, vx, vy, output, science_root
):
    import numpy as np
    import torch
    from benchmarks.training.higgs_confirmation import evaluate

    started = time.perf_counter()
    raw = state.snapshot(model, optimizer, update, identity, epoch=1 if update else 0)
    if update:
        state.check_optimizer(model, optimizer, update)
    exact.require(
        not helpers.nonfinite_checkpoint_fields(raw), "nonfinitepointmodel/grad/moments"
    )
    checkpoint = None
    if update == 25000:
        checkpoint = "final-25000.pt"
        torch.save(raw, output / checkpoint)
    fingerprint = state.core_digest(raw)
    write(
        output / f"state-fingerprint-{update}.json",
        {
            "update": update,
            "identity": identity,
            "complete_typedcore_sha256": fingerprint,
            "rawstate_retained": update in (0, 25000),
            "intermediate_raw_replay_claim": False,
        },
    )
    snapshot_seconds = time.perf_counter() - started
    (training, train_logits, train_labels), train_seconds = helpers.timed(
        evaluate, model, tx, ty
    )
    (development, dev_logits, dev_labels), dev_seconds = helpers.timed(
        evaluate, model, vx, vy
    )
    exact.assert_equal(
        state.core_state(
            state.snapshot(model, optimizer, update, identity, epoch=1 if update else 0)
        ),
        state.core_state(raw),
        "point-evaluation-mutatedmodel/optimizer/grad",
    )
    exact.require(
        all(
            math.isfinite(v)
            for metrics in (training, development)
            for v in metrics.values()
        ),
        "nonfinitepointmetrics",
    )
    start = time.perf_counter()
    names = {}
    for kind, logits, labels, first, rows in (
        ("train", train_logits, train_labels, 1000000, 100000),
        ("development", dev_logits, dev_labels, 1100000, 50000),
    ):
        filename = f"{kind}-step-{update}.npz"
        np.savez_compressed(
            output / filename,
            logits=logits,
            labels=labels,
            row_ids=np.arange(first, first + rows, dtype=np.int64),
        )
        names[kind] = {"file": filename, "sha256": file_sha(output / filename)}
    row = {
        "update": update,
        "fixed_final_selection": update == 25000,
        "training": training,
        "development": development,
        "gap_dev_minus_train": {
            k: development[k] - training[k] for k in ("loss", "accuracy", "roc_auc")
        },
        "predictions": names,
        "checkpoint_file": checkpoint,
        "checkpoint_sha256": file_sha(output / checkpoint) if checkpoint else None,
        "qualified_initial0_reference": str(
            (
                science_root
                / "qualification/cases"
                / model._conditioning_case
                / "initial-state.pt"
            ).relative_to(science_root)
        )
        if update == 0
        else None,
        "state_fingerprint_sha256": file_sha(
            output / f"state-fingerprint-{update}.json"
        ),
        "snapshot_fingerprint_IO_seconds": snapshot_seconds,
        "evaluation_seconds": {"fulltrain": train_seconds, "fulldev": dev_seconds},
        "prediction_IO_hash_seconds": time.perf_counter() - start,
        "evaluation_did_not_mutate_fullcore": True,
    }
    write(output / f"metrics-{update}.json", row)
    return row, raw


def selected_reload(case, raw, cached, tx, ty, vx, vy, device, output):
    import numpy as np
    from benchmarks.training.higgs_confirmation import evaluate

    exact.require(
        file_sha(output / cached["checkpoint_file"]) == cached["checkpoint_sha256"],
        "fixedfinalsavedcheckpoint bytes changed",
    )
    (model, optimizer, baseline, proof), restore_seconds = helpers.timed(
        state.load, raw, device
    )
    times = {}
    for kind, x, y, metrics in (
        ("train", tx, ty, cached["training"]),
        ("development", vx, vy, cached["development"]),
    ):
        (actual, logits, labels), times[kind] = helpers.timed(evaluate, model, x, y)
        exact.require(
            file_sha(output / cached["predictions"][kind]["file"])
            == cached["predictions"][kind]["sha256"],
            "cachedprediction bytes changed",
        )
        with np.load(
            output / cached["predictions"][kind]["file"], allow_pickle=False
        ) as archive:
            expected = {n: archive[n].copy() for n in archive.files}
        helpers.assert_prediction_arrays(
            prediction_arrays(kind, logits, labels),
            expected,
        )
        exact.assert_equal(actual, metrics, "fixedfinalreloaded-" + kind + "metrics")
    exact.assert_equal(
        state.core_state(
            state.snapshot(
                model, optimizer, 25000, raw["configuration"]["identity"], epoch=1
            )
        ),
        baseline,
        "fixedfinal-eval-mutatedPOSTLOADbaseline",
    )
    record = {
        "case": case,
        "selected_update": 25000,
        "selection": "fixedfinal, notminimumDEV",
        "native_load": proof,
        "arraybytes_and_metrics_equal_originalfinal": True,
        "evaluation_did_not_mutate_POSTLOADstate": True,
        "restore_seconds": restore_seconds,
        "evaluation_seconds": times,
        "checkpoint_sha256": cached["checkpoint_sha256"],
        "prediction_references": cached["predictions"],
    }
    write(output / "selected-final.json", record)
    return record


def run_case(
    case, gains, identity, tx, ty, vx, vy, order, device, science_root, output
):
    import torch

    output.mkdir()
    begin = time.perf_counter()
    model = study.make_model(case, gains=gains).to(device)
    optimizer, contract = state.make_optimizer(model)
    qualified_path = science_root / "qualification/cases" / case / "initial-state.pt"
    initial = torch.load(qualified_path, map_location="cpu", weights_only=True)
    exact.require(
        initial["configuration"]["case"] == case
        and initial["configuration"]["model_configuration"]
        == model._conditioning_configuration,
        "qualified initial0 case/norm/gain contract differs",
    )
    exact.assert_equal(
        state.core_state(state.snapshot(model, optimizer, 0, identity)),
        state.core_state(initial),
        "mainfresh0-qualified0-fullbytes",
    )
    write(
        output / "initial-state-reference.json",
        {
            "qualified_initial0_file": qualified_path.relative_to(
                science_root
            ).as_posix(),
            "sha256": file_sha(qualified_path),
            "exactfresh0": True,
            "qualification8notloaded": True,
        },
    )
    initial_diagnostic, initial_diag_seconds = helpers.timed(
        activations, model, tx[:256]
    )
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    rolling = []
    blocks = []
    points = []
    first_indices = order[:4]
    first_x, first_y = tx[first_indices], ty[first_indices]
    row, raw = evaluate_point(
        model, optimizer, 0, identity, tx, ty, vx, vy, output, science_root
    )
    points.append(row)
    del initial, raw
    model.train()
    (first_summary, first_raw), first_seconds = helpers.timed(
        partial(
            observation.observed_step,
            model,
            optimizer,
            first_x,
            first_y,
            first_indices,
            identity,
            step=1,
            epoch=1,
            totals=totals,
        )
    )
    torch.save(first_raw, output / "actual-first-task-step.pt")
    write(output / "actual-first-task-step.json", first_summary)
    del first_raw
    current = 1
    for stop in study.MILESTONES[1:]:
        model.train()
        blocks.append(
            bare_block(
                model, optimizer, tx, ty, order, current, stop, totals, rolling, output
            )
        )
        current = stop
        row, raw = evaluate_point(
            model, optimizer, stop, identity, tx, ty, vx, vy, output, science_root
        )
        points.append(row)
        write(
            output / "learning.json",
            {
                "case": case,
                "identity": identity,
                "optimizer_contract": contract,
                "initial_diagnostics": initial_diagnostic,
                "initial_diagnostics_seconds": initial_diag_seconds,
                "actual_first_task_observation_seconds": first_seconds,
                "training_blocks": blocks,
                "milestones": points,
                "selection": "fixedfinal25000",
                "task_updates": stop,
                "timing_scope": "DIAGNOSTIC only; blocksinclude prescribedrollinglogs; eval/firstobservation/snapshot separated, nonadditivecomponents",
            },
        )
        if stop != 25000:
            del raw
    final_diagnostic, final_diag_seconds = helpers.timed(activations, model, tx[:256])
    (endpoint_summary, endpoint_raw), endpoint_seconds = helpers.timed(
        observation.endpoint_probe, raw, first_x, first_y, first_indices, device
    )
    torch.save(endpoint_raw, output / "endpoint-sameB4-probe.pt")
    write(output / "endpoint-sameB4-probe.json", endpoint_summary)
    exact.assert_equal(
        state.core_state(state.snapshot(model, optimizer, 25000, identity, epoch=1)),
        state.core_state(raw),
        "endpointprobe-mutated-originalmodel/grad",
    )
    selected = selected_reload(case, raw, points[-1], tx, ty, vx, vy, device, output)
    complete = {
        "status": "completed-fixedfinal-oneepoch-conditioning-diagnostic",
        "case": case,
        "identity": identity,
        "task_updates": 25000,
        "selected_update": 25000,
        "first_step_is_part_of_25000": True,
        "endpoint_probe_optimizer_updates": 0,
        "initial_diagnostics": initial_diagnostic,
        "final_diagnostics": final_diagnostic,
        "final_diagnostics_seconds": final_diag_seconds,
        "endpoint_probe_seconds": endpoint_seconds,
        "selected_final": selected,
        "case_total_seconds_inclusive": time.perf_counter() - begin,
        "heldout_evaluated": False,
        "automatic_followup": False,
    }
    write(output / "complete.json", complete)
    del model, optimizer, raw, endpoint_raw
    gc.collect()
    torch.cuda.empty_cache()
    return complete


def run(args):
    from .higgs_coordinate_lr import verify_external

    science_start = time.perf_counter()
    consumer = Path(__file__).resolve().parents[3]
    manifest = common.verify_consumer(
        consumer,
        args.consumer_manifest,
        args.consumer_commit,
        consumer / common.DRIVER_PATH,
    )
    exact.require(
        manifest["files"][ENTRY] == file_sha(consumer / ENTRY),
        "executedconditioningdriver differs",
    )
    pins = verify_inputs(args.inputs)
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

    _configure()
    runtime = native_runtime.runtime_fingerprint()
    runtime_gate(runtime, read(args.inputs / "runtime-pin.json"))
    exact.require(
        Path(torchcst.__file__)
        .resolve()
        .is_relative_to(args.external_source.resolve()),
        "runtimeoutsideverified975",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "cohort": study.COHORT,
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "protocol_sha256": json_sha(protocol()),
        "external_source_commit": stock.SOURCE,
        "external_source_manifest_sha256": stock.SOURCE_MANIFEST,
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "runtime_pin_sha256": RUNTIME_SHA,
        "runtime_fingerprint": runtime,
        "verified_inputs": pins,
    }
    write(
        args.output / "run-manifest.json",
        {"identity": identity, "protocol": protocol(), "argv": sys.argv},
    )
    started = time.perf_counter()
    features = calibration_features(args.inputs / "train-development.npz")
    dense = study.make_model("dense-none")
    calibrations = {}
    for case in study.CASES:
        model = study.make_model(case)
        calibrations[case] = study.calibrate(dense, model, torch.from_numpy(features))
    calibration_seconds = time.perf_counter() - started
    write(
        args.output / "calibration.json",
        {
            "identity": identity,
            "features": helpers.array_record(features),
            "seconds": calibration_seconds,
            "cases": calibrations,
            "labels_or_DEV_used": False,
        },
    )
    del model, dense, features
    marker = qualification.qualify_all(
        calibrations, identity, args.output / "qualification", torch.device("cuda:0")
    )
    budget(args.output)
    # Labels and DEV arrays are decoded only after all six native qualifications.
    arrays, _ = load_pilot(
        args.inputs / "train-development.npz", args.inputs / "data-manifest.json"
    )
    device = torch.device("cuda:0")
    tx, ty, vx, vy = (
        torch.from_numpy(arrays[k]).to(device)
        for k in ("x_train", "y_train", "x_development", "y_development")
    )
    order_cpu = torch.randperm(100000, generator=torch.Generator().manual_seed(117))
    torch.save(order_cpu, args.output / "epoch1-order.pt")
    order = order_cpu.to(device)
    cases = args.output / "cases"
    cases.mkdir()
    completed = []
    for case in study.CASES:
        print("tasktraining " + case + " fresh0", flush=True)
        completed.append(
            run_case(
                case,
                calibrations[case]["gains"],
                identity,
                tx,
                ty,
                vx,
                vy,
                order,
                device,
                args.output,
                cases / case,
            )
        )
        write(
            args.output / "progress.json",
            {
                "completed_cases": [r["case"] for r in completed],
                "task_updates": 25000 * len(completed),
                "fixed_total_task_updates": 150000,
            },
        )
        budget(args.output)
    write(
        args.output / "run-summary.json",
        {
            "status": "completed-six-fresh-paired-A100-conditioning-diagnostic",
            "identity": identity,
            "cases": completed,
            "task_updates": 150000,
            "synthetic_qualification_updates": marker[
                "total_synthetic_qualification_updates"
            ],
            "fixedfinal25000_percase": True,
            "calibration_seconds": calibration_seconds,
            "science_seconds_before_summary_inclusive": time.perf_counter()
            - science_start,
            **budget(args.output),
            "official_test_or_heldout_evaluated": False,
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
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
