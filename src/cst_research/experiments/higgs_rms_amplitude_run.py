"""One raw-amplitude Dense-RMS-calibrated H8 trial, no output gain or LayerNorm."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import higgs_multilayer_conditioning_run as base
from . import higgs_rms_amplitude_init as init
from . import higgs_width_amplitude_init as previous
from . import higgs_width_amplitude_qualification as qualification
from . import higgs_width_amplitude_run as shared
from .higgs_dst_comparison import file_sha, json_sha, read, write


def protocol():
    value = previous.protocol()
    value.update(
        version=init.COHORT,
        arms=[init.ARM],
        total_task_updates=25000,
        initialization={
            "method": "sequential raw Gaussian amplitude coefficient search matching initial Dense pre-ReLU RMS on first256 TRAIN features",
            "search_relative_tolerance": init.SEARCH_TOLERANCE,
            "native_relative_tolerance": init.NATIVE_TOLERANCE,
            "coarse_coefficients": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            "maximum_bisections": 32,
            "maximum_coefficient": init.MAX_COEFFICIENT,
            "formula": "k_layer/sqrt(A/N)",
            "unchanged": [
                "standardized normal draws",
                "centers",
                "ordinary input/output initialization",
                "kernel and optimizer",
                "amplitude_max1",
                "alpha_init0",
            ],
            "runtime_gain": [1.0] * 7,
            "normalization": "none",
            "calibration_before_learning_only": True,
            "labels_or_DEV_used": False,
            "radius_changes_and_symmetric_clipping_recorded": True,
        },
        native_qualification="one fresh full-network VJP/quad/scalar/native8+4load4+observer qualification;20 network synthetic updates, scalar8 separate",
        comparator="already admitted same-GPU seed17 Dense/old/width0.1/old+gain trial; no new timing adoption or NI claim",
    )
    return value


def run(args):
    from .higgs_coordinate_lr import verify_external
    from .higgs_depth_normalization_run import scalar_adam_gate

    started = time.perf_counter()
    source = Path(__file__).resolve().parents[3]
    manifest = base.common.verify_consumer(
        source,
        args.consumer_manifest,
        args.consumer_commit,
        source / base.common.DRIVER_PATH,
    )
    if manifest["files"][
        "src/cst_research/experiments/higgs_rms_amplitude_run.py"
    ] != file_sha(Path(__file__)):
        raise ValueError("executed RMS initializer runner differs")
    pins = shared.inspect_inputs(args)
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
    torch.set_num_threads(2)
    runtime = base.native_runtime.runtime_fingerprint()
    if (
        args.device != "rtx4090"
        or runtime != read(args.runtime_pin)
        or base.native_runtime.validate_runtime(runtime) != "RTX4090"
    ):
        raise ValueError("exact reserved RTX4090 required for this follow-up")
    if (
        not Path(torchcst.__file__)
        .resolve()
        .is_relative_to(args.external_source.resolve())
    ):
        raise ValueError("wrong public runtime import boundary")
    args.output.mkdir(parents=True, exist_ok=False)
    base_identity = {
        "cohort": init.COHORT,
        "arm": init.ARM,
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "external_source_commit": base.stock.SOURCE,
        "external_source_manifest_sha256": base.stock.SOURCE_MANIFEST,
        "data_sha256": base.DATA_SHA,
        "data_manifest_sha256": base.MANIFEST_SHA,
        "runtime_pin_sha256": shared.RUNTIME_PINS[args.device],
        "runtime_fingerprint": runtime,
        "protocol_sha256": json_sha(protocol()),
        "device": args.device,
    }
    features = torch.from_numpy(base.calibration_features(args.data))

    def progress(atoms, rows):
        torch.save(atoms, args.output / "partial-initial-atoms.pt")
        write(
            args.output / "calibration-progress.json",
            {"completed_layers": rows, "task_updates": 0, "labels_or_DEV_used": False},
        )
        print(
            "calibrated site"
            + str(rows[-1]["site"])
            + " coefficient="
            + str(rows[-1]["coefficient"])
            + " relerror="
            + str(rows[-1]["relative_error"]),
            flush=True,
        )

    cal_started = time.perf_counter()
    atoms, calibration, witness = init.calibrate(features, progress=progress)
    calibration["physical_layers"] = previous.physical_initialization_record(
        init.make_model(atoms)
    )
    native = init.native_rms_witness(
        atoms, features, calibration, torch.device("cuda:0")
    )
    witness.update(native)
    calibration.update(
        seconds_including_native_check=time.perf_counter() - cal_started,
        base_identity=base_identity,
        feature_record=base.helpers.array_record(features.numpy()),
    )
    torch.save(atoms, args.output / "initial-atoms.pt")
    torch.save(witness, args.output / "initial-RMS-witness.pt")
    write(args.output / "rms-calibration.json", calibration)
    identity = {
        **base_identity,
        "initialization": protocol()["initialization"],
        "initial_atoms_sha256": file_sha(args.output / "initial-atoms.pt"),
        "rms_calibration_sha256": file_sha(args.output / "rms-calibration.json"),
        "initial_RMS_witness_sha256": file_sha(args.output / "initial-RMS-witness.pt"),
    }
    write(
        args.output / "run-manifest.json",
        {
            "identity": identity,
            "protocol": protocol(),
            "verified_inputs": pins,
            "argv": sys.argv,
        },
    )
    del native, witness, features
    gc.collect()
    torch.cuda.empty_cache()

    def factory(arm, *, gains=None):
        if arm != init.ARM or gains != [1.0] * 7:
            raise ValueError("RMS raw initializer requires gain1")
        return init.make_model(atoms)

    scalar = args.output / "scalar-Adam"
    scalar.mkdir()
    scalar_adam_gate(scalar, torch.device("cuda:0"))
    arm_root = args.output / "arms" / init.ARM
    (arm_root / "qualification/cases").mkdir(parents=True)
    quad = arm_root / "qualification/quadrature"
    quad.mkdir()
    qualification_record = qualification.qualify_arm(
        init.ARM,
        [1.0] * 7,
        identity,
        arm_root / "qualification/cases/cst-none",
        quad,
        torch.device("cuda:0"),
        model_factory=factory,
        runtime_case="cst-none",
    )
    write(
        args.output / "all-arms-native-qualification.json",
        {
            "passed": True,
            "arms": [init.ARM],
            "synthetic_updates": 20,
            "task_updates": 0,
        },
    )
    arrays, _ = base.load_pilot(args.data, args.data_manifest)
    tensors = tuple(
        torch.from_numpy(arrays[k]).to("cuda:0")
        for k in ("x_train", "y_train", "x_development", "y_development")
    )
    order = torch.randperm(100000, generator=torch.Generator().manual_seed(117))
    torch.save(order, args.output / "epoch1-order.pt")
    print("tasktraining rms-init fresh0", flush=True)
    completed = shared.train_arm(
        init.ARM,
        [1.0] * 7,
        identity,
        tensors,
        order.to("cuda:0"),
        torch.device("cuda:0"),
        arm_root,
        model_factory=factory,
        runtime_case="cst-none",
        initializer_policy=identity["initialization"],
    )
    write(
        args.output / "run-summary.json",
        {
            "status": "completed-one-raw-RMS-matched-initialization-arm",
            "identity": identity,
            "cases": [completed],
            "task_updates": 25000,
            "synthetic_qualification_updates": 20,
            "qualifications": [qualification_record],
            "science_seconds_inclusive": time.perf_counter() - started,
            "heldout_or_officialtest_scored": False,
            "automatic_followup": False,
            **base.budget(args.output),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--device", choices=("rtx4090",), required=True)
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
