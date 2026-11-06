"""Bounded four-arm initializer experiment on one explicitly pinned GPU."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_multilayer_conditioning as contract
from . import higgs_multilayer_conditioning_observation as observation
from . import higgs_multilayer_conditioning_run as base
from . import higgs_multilayer_conditioning_state as state
from . import higgs_width_amplitude_init as study
from . import higgs_width_amplitude_qualification as qualification
from .higgs_depth_data import DATA_SHA, MANIFEST_SHA, load_pilot, validate_manifest
from .higgs_dst_comparison import file_sha, json_sha, read, write

RUNTIME_PINS = {
    "rtx4090": "1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b",
    "a100": "35016be129960614d829affbea959797ae8d035b5ea41197b6803ae963709f55",
}


def inspect_inputs(args):
    expected = {
        "data": DATA_SHA,
        "data_manifest": MANIFEST_SHA,
        "external_manifest": base.stock.SOURCE_MANIFEST,
        "runtime_pin": RUNTIME_PINS[args.device],
    }
    for field, digest in expected.items():
        path = getattr(args, field)
        exact.require(
            path.is_file() and not path.is_symlink() and file_sha(path) == digest,
            "pinned regular input differs: " + field,
        )
    validate_manifest(read(args.data_manifest))
    return expected


def probe(
    model, optimizer, raw, first_x, first_y, indices, identity, arm, output, device
):
    import torch

    start = time.perf_counter()
    summary, retained = observation.endpoint_probe(
        raw, first_x, first_y, indices, device
    )
    update = raw["updates"]
    summary.update(
        arm=arm, update=update, probe_optimizer_updates=0, loaded_copy_only=True
    )
    retained.update(identity=identity, arm=arm, update=update)
    torch.save(retained, output / f"probe-step-{update}.pt")
    write(output / f"probe-step-{update}.json", summary)
    del retained
    gc.collect()
    torch.cuda.empty_cache()
    exact.assert_equal(
        state.core_state(
            state.snapshot(model, optimizer, update, identity, epoch=1 if update else 0)
        ),
        state.core_state(raw),
        "layer-probe-mutated-original-fullcore",
    )
    return {
        "update": update,
        "seconds_including_save": time.perf_counter() - start,
        "json_sha256": file_sha(output / f"probe-step-{update}.json"),
        "raw_sha256": file_sha(output / f"probe-step-{update}.pt"),
        "original_fullcore_unchanged": True,
    }


def train_arm(
    arm,
    gains,
    identity,
    tensors,
    order,
    device,
    arm_root,
    *,
    model_factory=None,
    runtime_case=None,
    initializer_policy=None,
):
    from functools import partial

    import torch

    tx, ty, vx, vy = tensors
    output = arm_root / "task"
    output.mkdir()
    started = time.perf_counter()
    factory = study.make_model if model_factory is None else model_factory
    model = factory(arm, gains=gains).to(device)
    optimizer, optimizer_contract = state.make_optimizer(model)
    case = study.ARMS[arm] if runtime_case is None else runtime_case
    qualified = arm_root / "qualification/cases" / case / "initial-state.pt"
    initial = torch.load(qualified, map_location="cpu", weights_only=True)
    exact.assert_equal(
        state.core_state(state.snapshot(model, optimizer, 0, identity)),
        state.core_state(initial),
        "fresh-task0-must-equal-qualified0",
    )
    del initial
    first_indices = order[:4]
    first_x, first_y = tx[first_indices], ty[first_indices]
    totals = (tx.new_zeros(()), torch.zeros((), device=device, dtype=torch.int64))
    points, probes, blocks, rolling = [], [], [], []
    initial_diagnostics = base.activations(model, tx[:256])
    row, raw = base.evaluate_point(
        model, optimizer, 0, identity, tx, ty, vx, vy, output, arm_root
    )
    points.append(row)
    probes.append(
        probe(
            model,
            optimizer,
            raw,
            first_x,
            first_y,
            first_indices,
            identity,
            arm,
            output,
            device,
        )
    )
    del raw
    (first_summary, first_raw), first_seconds = base.helpers.timed(
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
    first_summary["arm"] = arm
    torch.save(first_raw, output / "actual-first-task-step.pt")
    write(output / "actual-first-task-step.json", first_summary)
    del first_raw
    current = 1
    for stop in study.MILESTONES[1:]:
        model.train()
        blocks.append(
            base.bare_block(
                model, optimizer, tx, ty, order, current, stop, totals, rolling, output
            )
        )
        current = stop
        row, raw = base.evaluate_point(
            model, optimizer, stop, identity, tx, ty, vx, vy, output, arm_root
        )
        points.append(row)
        probes.append(
            probe(
                model,
                optimizer,
                raw,
                first_x,
                first_y,
                first_indices,
                identity,
                arm,
                output,
                device,
            )
        )
        write(
            output / "learning.json",
            {
                "arm": arm,
                "identity": identity,
                "model_contract_case": case,
                "initialization": study.initialization_policy(arm)
                if initializer_policy is None
                else initializer_policy,
                "optimizer_contract": optimizer_contract,
                "initial_diagnostics": initial_diagnostics,
                "actual_first_task_observation_seconds": first_seconds,
                "training_blocks": blocks,
                "milestones": points,
                "layer_probes": probes,
                "selection": "fixedfinal25000",
                "task_updates": stop,
            },
        )
        base.budget(arm_root)
        if stop != 25000:
            del raw
    final_diagnostics = base.activations(model, tx[:256])
    selected = base.selected_reload(
        case, raw, points[-1], tx, ty, vx, vy, device, output
    )
    result = {
        "status": "completed-fixedfinal-oneepoch-amplitude-comparison",
        "arm": arm,
        "identity": identity,
        "task_updates": 25000,
        "initial_diagnostics": initial_diagnostics,
        "final_diagnostics": final_diagnostics,
        "selected_final": selected,
        "layer_probes": probes,
        "first_observed_step_included_in_25000": True,
        "probe_optimizer_updates": 0,
        "heldout_evaluated": False,
        "case_total_seconds_inclusive": time.perf_counter() - started,
    }
    write(output / "complete.json", result)
    del model, optimizer, raw
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args):
    from .higgs_coordinate_lr import verify_external
    from .higgs_depth_normalization_run import scalar_adam_gate

    started = time.perf_counter()
    consumer = Path(__file__).resolve().parents[3]
    manifest = base.common.verify_consumer(
        consumer,
        args.consumer_manifest,
        args.consumer_commit,
        consumer / base.common.DRIVER_PATH,
    )
    exact.require(
        manifest["files"]["src/cst_research/experiments/higgs_width_amplitude_run.py"]
        == file_sha(Path(__file__)),
        "executed initializer runner differs",
    )
    pins = inspect_inputs(args)
    verify_external(
        SimpleNamespace(
            external_source=args.external_source,
            external_manifest=args.external_manifest,
            identity={"external_source_manifest_sha256": base.stock.SOURCE_MANIFEST},
        )
    )
    sys.path[:0] = [str(args.external_source / "src"), str(args.external_source)]
    import torch
    import torchcst
    from benchmarks.training.higgs_confirmation import _configure

    _configure()
    runtime = base.native_runtime.runtime_fingerprint()
    label = base.native_runtime.validate_runtime(runtime)
    exact.require(
        label == ("RTX4090" if args.device == "rtx4090" else "A100-MIG3g.40gb")
        and runtime == read(args.runtime_pin),
        "exact requested hardware/runtime required",
    )
    exact.require(
        Path(torchcst.__file__)
        .resolve()
        .is_relative_to(args.external_source.resolve()),
        "torchcst import outside verified public975",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "cohort": study.COHORT,
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "external_source_commit": base.stock.SOURCE,
        "external_source_manifest_sha256": base.stock.SOURCE_MANIFEST,
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "runtime_pin_sha256": RUNTIME_PINS[args.device],
        "runtime_fingerprint": runtime,
        "protocol_sha256": json_sha(study.protocol()),
        "device": args.device,
    }
    write(
        args.output / "run-manifest.json",
        {
            "identity": identity,
            "protocol": study.protocol(),
            "verified_inputs": pins,
            "argv": sys.argv,
        },
    )
    gains = {arm: [1.0] * 7 for arm in study.ARMS}
    calibration_start = time.perf_counter()
    features = base.calibration_features(args.data)
    dense, candidate = (
        study.make_model("dense-control"),
        study.make_model("old-init-gain"),
    )
    calibrated = contract.calibrate(dense, candidate, torch.from_numpy(features))
    gains["old-init-gain"] = calibrated["gains"]
    write(
        args.output / "calibration.json",
        {
            "identity": identity,
            "old-init-gain": calibrated,
            "seconds": time.perf_counter() - calibration_start,
            "labels_or_DEV_used": False,
        },
    )
    del features, dense, candidate
    device = torch.device("cuda:0")
    scalar = args.output / "scalar-Adam"
    scalar.mkdir()
    scalar_adam_gate(scalar, device)
    arms_root = args.output / "arms"
    arms_root.mkdir()
    qualifications = []
    for arm in study.ARMS:
        arm_root = arms_root / arm
        (arm_root / "qualification/cases").mkdir(parents=True)
        quad = arm_root / "qualification/quadrature"
        quad.mkdir()
        arm_identity = {
            **identity,
            "arm": arm,
            "initialization": study.initialization_policy(arm),
        }
        model = study.make_model(arm, gains=gains[arm])
        write(
            arm_root / "initial-physical-values.json",
            {
                "identity": arm_identity,
                "gains": gains[arm],
                "layers": study.physical_initialization_record(model),
            },
        )
        del model
        print("qualification " + arm, flush=True)
        qualifications.append(
            qualification.qualify_arm(
                arm,
                gains[arm],
                arm_identity,
                arm_root / "qualification/cases" / study.ARMS[arm],
                quad,
                device,
            )
        )
        base.budget(args.output)
    write(
        args.output / "all-arms-native-qualification.json",
        {
            "passed": True,
            "arms": list(study.ARMS),
            "synthetic_updates": 80,
            "task_updates": 0,
        },
    )
    arrays, _ = load_pilot(args.data, args.data_manifest)
    tensors = tuple(
        torch.from_numpy(arrays[k]).to(device)
        for k in ("x_train", "y_train", "x_development", "y_development")
    )
    order_cpu = torch.randperm(100000, generator=torch.Generator().manual_seed(117))
    torch.save(order_cpu, args.output / "epoch1-order.pt")
    order = order_cpu.to(device)
    completed = []
    for arm in study.ARMS:
        print("tasktraining " + arm + " fresh0", flush=True)
        arm_identity = {
            **identity,
            "arm": arm,
            "initialization": study.initialization_policy(arm),
        }
        completed.append(
            train_arm(
                arm, gains[arm], arm_identity, tensors, order, device, arms_root / arm
            )
        )
        write(
            args.output / "progress.json",
            {
                "completed_arms": [r["arm"] for r in completed],
                "task_updates": 25000 * len(completed),
                "fixed_total_task_updates": 100000,
            },
        )
        base.budget(args.output)
    write(
        args.output / "run-summary.json",
        {
            "status": "completed-four-arm-width-amplitude-comparison",
            "identity": identity,
            "cases": completed,
            "task_updates": 100000,
            "synthetic_qualification_updates": 80,
            "qualifications": qualifications,
            "science_seconds_inclusive": time.perf_counter() - started,
            "heldout_or_officialtest_scored": False,
            "automatic_followup": False,
            **base.budget(args.output),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--device", choices=tuple(RUNTIME_PINS), required=True)
    for name in (
        "consumer-manifest",
        "external-source",
        "external-manifest",
        "data",
        "data-manifest",
        "runtime-pin",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
