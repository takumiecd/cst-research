"""Fresh paired Dense/CST He or Xavier H8 trial on one pinned owner GPU."""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from . import higgs_fan_initialization as init
from . import higgs_multilayer_conditioning_run as base
from . import higgs_width_amplitude_qualification as qualification
from . import higgs_width_amplitude_run as shared
from .higgs_dst_comparison import file_sha, json_sha, read, write


def protocol(scheme, q):
    init.target_energy(scheme, 1024, 1024)
    if isinstance(q, bool) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("finite q in [0,1] required")
    return dict(
        version=init.COHORT,
        scheme=scheme,
        q_init=q,
        arms=["dense-fan", "cst-fan"],
        width=1024,
        hidden_layers=8,
        internal_maps=7,
        atoms_per_CST_map=4096,
        seed=17,
        internal_initialization="Dense normal He fan_in/ReLU or Xavier gain1; CST bounded symmetric Gaussian amplitudes, conditional sign-energy target matches same Dense ensemble moment",
        ordinary_input_output="unchanged paired legacy initialization; only seven internal maps change",
        target_energy=init.target_energy(scheme, 1024, 1024),
        q_parameter="Polar alpha_init; lowercase q/s/t; distinct from Atom density",
        calibration="no data or Dense reference; FP64 atom factor norm products of actual FP32 candidate Atoms; error<=0.005",
        symmetric_amplitude_ratio_clip=[-1 + 1e-6, 1 - 1e-6],
        maximum_scale=init.MAX_SCALE,
        distribution_scope="clipped amplitude Gaussian is not an exact Gaussian weight distribution; all clip fractions retained",
        runtime_gain=[1.0] * 7,
        normalization="none",
        epochs=1,
        task_updates_per_arm=25000,
        total_task_updates=50000,
        batch_size=4,
        learning_rate=0.0003,
        position_multiplier=1,
        eps=1e-8,
        optimizer="stockq1 AdamW/CSTAdamW; ordinary/amplitude decay.01 rawAtom decay0",
        dtype="FP32",
        TF32=False,
        execution="eager",
        milestones=[0, 1000, 5000, 10000, 15000, 20000, 25000],
        selection="fixed final25000, no HPO/bestDEV/earlystop",
        data="TRAIN100k rows[1000000,1100000), DEV50k[1100000,1150000)",
        qualification="fresh all-parameter Y/dX/dP FP64 native VJP dy1/dy1e24,31Atom quadrature,scalarAdam,8bare vs4load4 vsobservedfirst+7 equality;40 synthetic separate",
        probes="same first TRAIN B4 copied probes at all seven milestones; no probe updates",
        automatic_followup=False,
        heldout_or_officialtest_scored=False,
        adoption="exploratory only; no public default adoption or quality/scaling guarantee",
    )


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
        "src/cst_research/experiments/higgs_fan_initialization_run.py"
    ] != file_sha(Path(__file__)):
        raise ValueError("executed fan initializer differs from pinned source")
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
    if runtime != read(args.runtime_pin):
        raise ValueError("runtime pin mismatch")
    if args.device == "rtx4090":
        if base.native_runtime.validate_runtime(runtime) != "RTX4090":
            raise ValueError("RTX4090 required")
    elif "A100" not in runtime["gpu"] or runtime["multiprocessors"] != 42:
        raise ValueError("exact A100 MIG pin required")
    if (
        not Path(torchcst.__file__)
        .resolve()
        .is_relative_to(args.external_source.resolve())
    ):
        raise ValueError("wrong torchcst boundary")
    spec = protocol(args.scheme, args.q)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(
        cohort=init.COHORT,
        consumer_commit=args.consumer_commit,
        consumer_manifest_sha256=file_sha(args.consumer_manifest),
        external_source_commit=base.stock.SOURCE,
        external_source_manifest_sha256=base.stock.SOURCE_MANIFEST,
        data_sha256=base.DATA_SHA,
        data_manifest_sha256=base.MANIFEST_SHA,
        runtime_pin_sha256=shared.RUNTIME_PINS[args.device],
        runtime_fingerprint=runtime,
        protocol_sha256=json_sha(spec),
        device=args.device,
        initialization=spec,
    )

    def progress(atoms, rows):
        torch.save(atoms, args.output / "partial-initial-atoms.pt")
        write(
            args.output / "calibration-progress.json",
            dict(records=rows, task_updates=0, data_used=False),
        )
        row = rows[-1]
        print(
            "calibrated site"
            + str(row["site"])
            + " scheme="
            + args.scheme
            + " q="
            + str(args.q)
            + " scale="
            + str(row["nominal_normal_std_before_clip"])
            + " clips="
            + str(row["clipped_atoms"])
            + " error="
            + str(row["relative_error"]),
            flush=True,
        )

    calibration_start = time.perf_counter()
    atoms, records, energies = init.calibrate(
        scheme=args.scheme, q=args.q, progress=progress
    )
    torch.save(atoms, args.output / "initial-atoms.pt")
    torch.save(energies, args.output / "initial-atom-energies.pt")
    write(
        args.output / "fan-calibration.json",
        dict(
            records=records,
            seconds=time.perf_counter() - calibration_start,
            q_init=args.q,
            data_or_Dense_reference_used=False,
            runtime_gain=[1.0] * 7,
            normalization="none",
        ),
    )
    identity.update(
        initial_atoms_sha256=file_sha(args.output / "initial-atoms.pt"),
        initial_energy_sha256=file_sha(args.output / "initial-atom-energies.pt"),
        fan_calibration_sha256=file_sha(args.output / "fan-calibration.json"),
    )
    write(
        args.output / "run-manifest.json",
        dict(identity=identity, protocol=spec, verified_inputs=pins, argv=sys.argv),
    )

    def factory(arm, *, gains=None):
        if arm not in ("dense-fan", "cst-fan") or gains != [1.0] * 7:
            raise ValueError("explicit two arms and gain1 required")
        return init.make_model(
            "dense" if arm == "dense-fan" else "cst", scheme=args.scheme, atoms=atoms
        )

    scalar = args.output / "scalar-Adam"
    scalar.mkdir()
    scalar_adam_gate(scalar, torch.device("cuda:0"))
    qualified = []
    for arm in spec["arms"]:
        case = "dense-none" if arm == "dense-fan" else "cst-none"
        arm_root = args.output / "arms" / arm
        (arm_root / "qualification/cases").mkdir(parents=True)
        quad = arm_root / "qualification/quadrature"
        quad.mkdir()
        qualified.append(
            qualification.qualify_arm(
                arm,
                [1.0] * 7,
                dict(identity, arm=arm),
                arm_root / "qualification/cases" / case,
                quad,
                torch.device("cuda:0"),
                model_factory=factory,
                runtime_case=case,
            )
        )
    write(
        args.output / "all-arms-native-qualification.json",
        dict(passed=True, arms=spec["arms"], synthetic_updates=40, task_updates=0),
    )
    arrays, _ = base.load_pilot(args.data, args.data_manifest)
    tensors = tuple(
        torch.from_numpy(arrays[k]).to("cuda:0")
        for k in ("x_train", "y_train", "x_development", "y_development")
    )
    del arrays
    gc.collect()
    order = torch.randperm(100000, generator=torch.Generator().manual_seed(117))
    torch.save(order, args.output / "epoch1-order.pt")
    device_order = order.to("cuda:0")
    completed = []
    for arm in spec["arms"]:
        print("tasktraining " + arm + " fresh0", flush=True)
        case = "dense-none" if arm == "dense-fan" else "cst-none"
        completed.append(
            shared.train_arm(
                arm,
                [1.0] * 7,
                dict(identity, arm=arm),
                tensors,
                device_order,
                torch.device("cuda:0"),
                args.output / "arms" / arm,
                model_factory=factory,
                runtime_case=case,
                initializer_policy=spec,
            )
        )
    write(
        args.output / "run-summary.json",
        dict(
            status="completed-paired-fan-initialization",
            identity=identity,
            cases=completed,
            task_updates=50000,
            synthetic_qualification_updates=40,
            qualifications=qualified,
            science_seconds_inclusive=time.perf_counter() - started,
            heldout_or_officialtest_scored=False,
            automatic_followup=False,
            **base.budget(args.output),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--device", choices=("rtx4090", "a100"), required=True)
    parser.add_argument("--scheme", choices=("he", "xavier"), required=True)
    parser.add_argument("--q", type=float, required=True)
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
