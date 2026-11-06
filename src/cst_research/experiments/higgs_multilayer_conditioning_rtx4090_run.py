"""Fresh RTX4090 cohort orchestration; unchanged a43d numerical functions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import higgs_multilayer_conditioning_run as base
from .higgs_multilayer_conditioning_run import (
    DATA_SHA,
    MANIFEST_SHA,
    common,
    exact,
    file_sha,
    helpers,
    json_sha,
    load_pilot,
    native_runtime,
    qualification,
    read,
    stock,
    study,
    validate_manifest,
    write,
)

COHORT = "higgs-multilayer-conditioning-rtx4090-v1"
STUDY_ID = "higgs-multilayer-conditioning-rtx4090-20261106"
RUNTIME_SHA = "1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b"
ENTRY = "src/cst_research/experiments/higgs_multilayer_conditioning_rtx4090_driver.py"
NUMERICAL_SOURCE_COMMIT = "a43d6a938db7ff1bbf3522343cd35f10f3dcac1c"
NUMERICAL_MODULES = {
    "higgs_multilayer_conditioning.py": "1fc7a066bab7146f8a78060a845d5589b58d23ef40a564db1d2e6a4c5eac4596",
    "higgs_multilayer_conditioning_reference.py": "bf99a2d4b83bb27aa8cb3c738c168f724e81c3a08990d5821f28a465cfac264a",
    "higgs_multilayer_conditioning_state.py": "bc990c3631b2c38bac1bed4f19a1bb82b56ecc49a4231a689ddaafb8bf1c14d2",
    "higgs_multilayer_conditioning_observation.py": "bbab43a2a549257e7d56f9089f08ad9af169363b1201d8c23e187187d0329154",
    "higgs_multilayer_conditioning_qualification.py": "4c76f959d86b9aae33bed0f2cb59dcd070d8fad3bd3f7c7aafb05c5ede6e6835",
    "higgs_multilayer_conditioning_run.py": "401f5dc4d9cb1a686e74aea69965e473eb296e0b0246e6da362a2c3b523264d0",
    "higgs_multilayer_conditioning_driver.py": "88fbf42e4ab8e6e8d256f62c7e6bb90dd206438b457025995606debac1cf1006",
}


def verify_numerical_source(consumer):
    for name, digest in NUMERICAL_MODULES.items():
        path = consumer / "src/cst_research/experiments" / name
        exact.require(
            path.is_file() and not path.is_symlink() and file_sha(path) == digest,
            "unchanged a43d numerical module required: " + name,
        )


def protocol():
    result = base.protocol()
    result.update(
        version=COHORT,
        study_id=STUDY_ID,
        runtime_pin_sha256=RUNTIME_SHA,
        device="exactRTX4090/CC8.9/128SM/25262096384B",
        retention_route="immutable full RTX4090 raw; compact proofs/predictions plus space-admitted compressed local archive or separately pinned peer CPU auditor with CUDA off",
        numerical_source_commit=NUMERICAL_SOURCE_COMMIT,
        numerical_modules_sha256=NUMERICAL_MODULES,
        cross_device_scope="six fresh independently paired RTX4090 arms; same N1024/H8/A4096 as A100, no parent-state reuse, speed adoption or Atom-order claim",
    )
    return result


def input_pins():
    return {**base.input_pins(), "runtime-pin.json": RUNTIME_SHA}


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
        native_runtime.validate_runtime(runtime) == "RTX4090" and runtime == pin,
        "exact pinned RTX4090 runtime required; no A100/3090 substitution",
    )


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
    verify_numerical_source(consumer)
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
        "cohort": COHORT,
        "numerical_model_contract_cohort": study.COHORT,
        "numerical_source_commit": NUMERICAL_SOURCE_COMMIT,
        "numerical_modules_sha256": NUMERICAL_MODULES,
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
    features = base.calibration_features(args.inputs / "train-development.npz")
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
    base.budget(args.output)
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
            base.run_case(
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
        base.budget(args.output)
    write(
        args.output / "run-summary.json",
        {
            "status": "completed-six-fresh-paired-RTX4090-conditioning-diagnostic",
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
            **base.budget(args.output),
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
