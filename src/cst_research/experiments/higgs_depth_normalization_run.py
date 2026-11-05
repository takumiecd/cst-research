"""Qualified8-update task probe; no epoch, development evaluation or selection."""

from __future__ import annotations

import argparse
import contextlib
import platform
import sys
import time
from pathlib import Path

from . import higgs_depth_gain as inherited
from . import higgs_depth_normalization as study
from . import higgs_depth_normalization_steps as observe
from . import higgs_depth_study as stock
from .higgs_depth_data import load_pilot
from .higgs_dst_comparison import file_sha, json_sha, read, write


@contextlib.contextmanager
def inherited_factory(case, gains):
    """Explicit temporary scientific adapter; all parent files stay byte-identical."""
    old = inherited.qualified_model, inherited.reference_forward, inherited.COHORT
    inherited.qualified_model = lambda arm, values, device: study.make_model(
        case, gains=values
    ).to(device)
    inherited.reference_forward = study.reference_forward
    inherited.COHORT = study.COHORT
    try:
        yield
    finally:
        inherited.qualified_model, inherited.reference_forward, inherited.COHORT = old


def same_state(actual, expected):
    from benchmarks.training.higgs_verify import byte_equal, tensor_fields

    a, b = dict(tensor_fields(actual)), dict(tensor_fields(expected))
    return a.keys() == b.keys() and all(byte_equal(a[k], b[k]) for k in a)


def scalar_adam_gate(output, device):
    import torch
    from benchmarks.training.owner_window_fp32 import errors

    model = study.make_model("dense-stock").to(device)
    actual_optimizer, _ = stock.make_optimizer(model)
    options = actual_optimizer.param_groups[0]
    point = torch.nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.AdamW(
        [point],
        **{
            k: options[k]
            for k in (
                "lr",
                "betas",
                "eps",
                "weight_decay",
                "fused",
                "capturable",
                "amsgrad",
                "maximize",
            )
        },
    )
    rows, raw = [], []
    for step in range(1, 9):
        before = point.detach().cpu().clone()
        point.grad = torch.tensor([1e-10, 1e-8, 1e-6], device=device) * (
            -1 if step % 3 == 0 else 1
        )
        optimizer.step()
        state = stock.clone_cpu(optimizer.state[point])
        mhat, denom, _, contract = observe.adam_denominator(state, options)
        expected = (
            before * (1 - options["lr"] * options["weight_decay"])
            - options["lr"] * mhat / denom
        )
        actual = point.detach().cpu().clone()
        normal = errors(actual, expected)
        norm = inherited.gradient_norm_gate(actual, expected)
        raw.append(
            {
                "step": step,
                "before": before,
                "actual": actual,
                "expected": expected,
                "state": state,
                "gradient": point.grad.cpu(),
                "contract": contract,
            }
        )
        rows.append(
            {
                "step": step,
                "elementwise": normal,
                "relative_norm": norm,
                "contract": contract,
            }
        )
        if not normal["pass"] or not norm["pass"]:
            torch.save(raw, output / "adam-denominator-scalar.pt")
            write(output / "adam-denominator-scalar.json", rows)
            raise ValueError(
                "actual native fused Adam denominator reconstruction gate failed"
            )
    torch.save(raw, output / "adam-denominator-scalar.pt")
    write(output / "adam-denominator-scalar.json", rows)
    return rows


def qualify(case, gains, output, device):
    import torch
    from benchmarks.training.full_step_graph import ordinary_step

    output.mkdir()
    with inherited_factory(case, gains):
        parent = inherited.qualify(
            case, gains, output / "independent-normal-and-resume", device
        )
    generator = torch.Generator().manual_seed(1817)
    x = torch.randn((32, 28), generator=generator).to(device)
    y = torch.arange(32, device=device) % 2
    plain, watched = [study.make_model(case, gains=gains).to(device) for _ in range(2)]
    po, _ = stock.make_optimizer(plain)
    wo, _ = stock.make_optimizer(watched)
    totals = (x.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    watched_rows = []
    for step in range(1, 9):
        indices = torch.arange(4 * (step - 1), 4 * step, device=device)
        ordinary_step(plain, po, x, y, indices, totals)
        row, _ = observe.observed_step(
            watched, wo, x[indices], y[indices], step=step, retain_raw=False
        )
        watched_rows.append(row)
        if not same_state(
            stock.checkpoint(plain, po, 0, step, {}, None),
            stock.checkpoint(watched, wo, 0, step, {}, None),
        ):
            raise ValueError(
                "observation changes native parameter/moment/clock/gradient bytes"
            )
        if "_adamw_proposal" in wo.__dict__:
            raise ValueError("native proposal observation override not restored")
    torch.save(
        {
            "plain": stock.checkpoint(plain, po, 0, 8, {}, None),
            "observed": stock.checkpoint(watched, wo, 0, 8, {}, None),
        },
        output / "observer-transparency.pt",
    )
    write(output / "observer-transparency.json", watched_rows)
    record = {
        "status": "passed",
        "case": case,
        "parent_gate": parent,
        "parent_gate_adapter": "temporary factory/reference/COHORT only, restored finally; no source mutation; inherited exact normal/norm/packed-column/material/resume rules",
        "observation_native8_all_step_tensor_bytes_equal": True,
        "synthetic_qualification_not_task_updates": True,
        "public_Graph_qualification_claim": False,
    }
    write(output / "qualification.json", record)
    return record


def runtime_fingerprint():
    import subprocess

    import numpy as np
    import torch
    import triton

    props = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "numpy": np.__version__,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=15,
        )
        .strip()
        .splitlines()[0],
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "multiprocessors": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
    }


# Exact authorized hardware tuples; the root launcher separately verifies the
# requested Hub profile. Device properties alone do not attest a Hub profile ID.
APPROVED_HARDWARE = {
    "RTX3090": ("NVIDIA GeForce RTX 3090", [8, 6], 82, 25296044032),
    "RTX4090": ("NVIDIA GeForce RTX 4090", [8, 9], 128, 25262096384),
    "A100-MIG3g.40gb": ("NVIDIA A100 80GB PCIe MIG 3g.40gb", [8, 0], 42, 42144366592),
}


def validate_runtime(runtime):
    if (runtime["torch"], runtime["cuda"], runtime["triton"]) != (
        "2.11.0+cu126",
        "12.6",
        "3.6.0",
    ):
        raise ValueError(
            "this diagnostic requires reviewed Torch2.11cu126/Triton3.6 fused coefficient semantics"
        )
    actual = (
        runtime["gpu"],
        runtime["capability"],
        runtime["multiprocessors"],
        runtime["total_memory_bytes"],
    )
    for name, expected in APPROVED_HARDWARE.items():
        if actual == expected:
            return name
    raise ValueError(
        "not an exact root-authorized RTX3090/RTX4090/A100 MIG3g.40gb hardware tuple"
    )


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
    if file_sha(args.runtime_file) != args.runtime_sha256:
        raise ValueError("reviewed runtime pin bytes differ")
    actual_runtime = runtime_fingerprint()
    admitted_hardware = validate_runtime(actual_runtime)
    if actual_runtime != read(args.runtime_file):
        raise ValueError("reserved device/runtime differs from reviewed pin")
    arrays, data_manifest = load_pilot(args.data, args.data_manifest)
    # Development arrays are mechanically admitted by the existing data reader;
    # no development tensors, forward, loss, metrics or selection are constructed.
    features = torch.from_numpy(arrays["x_train"])
    labels = torch.from_numpy(arrays["y_train"])
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "external_source_commit": stock.SOURCE,
        "external_source_manifest_sha256": stock.SOURCE_MANIFEST,
        "data_sha256": file_sha(args.data),
        "data_manifest_sha256": file_sha(args.data_manifest),
        "runtime_fingerprint": actual_runtime,
        "admitted_hardware_family": admitted_hardware,
        "runtime_pin_sha256": args.runtime_sha256,
        "protocol_sha256": json_sha(study.protocol()),
    }
    write(
        args.output / "run-manifest.json",
        {
            "identity": identity,
            "protocol": study.protocol(),
            "argv": sys.argv,
            "data_exposure": data_manifest["exposure"],
            "status": "started",
        },
    )
    calibrated = {}
    dense = study.make_model("dense-stock")
    for case in study.CASES:
        model = study.make_model(case)
        calibrated[case] = study.calibrate(dense, model, features[:256])
    write(args.output / "calibration.json", calibrated)
    del dense, model
    device = torch.device("cuda:0")
    qualification = args.output / "qualification"
    qualification.mkdir()
    scalar_adam_gate(qualification, device)
    qualifications = {}
    # All9 cases qualify before the first true task update in any case.
    for case in study.CASES:
        write(
            args.output / "progress.json",
            {"stage": "qualification", "case": case, "task_updates": 0},
        )
        print("qualifying " + case, flush=True)
        qualifications[case] = qualify(
            case, calibrated[case]["gains"], qualification / case, device
        )
    order = torch.randperm(100000, generator=torch.Generator().manual_seed(117))
    indices = order[:32]
    torch.save(
        {"full_order": order, "task_indices_first32": indices},
        args.output / "data-order.pt",
    )
    tx, ty = features[indices].to(device), labels[indices].to(device)
    completed = []
    for case in study.CASES:
        root = args.output / case
        root.mkdir()
        model = study.make_model(case, gains=calibrated[case]["gains"]).to(device)
        optimizer, contract = stock.make_optimizer(model)
        torch.cuda.reset_peak_memory_stats()
        config = {
            "cohort": study.COHORT,
            "architecture": study.protocol()["architecture"],
            "inherited_factory_configuration": model._depth_configuration,
            "case": case,
            "identity": identity,
            "gains": calibrated[case]["gains"],
            "normalization": study.CASES[case][2],
            "optimizer_contract": contract,
        }
        torch.save(
            stock.checkpoint(model, optimizer, 0, 0, config, None), root / "state-0.pt"
        )
        rows = []
        for step in range(1, 9):
            begin = time.perf_counter()
            row, raw = observe.observed_step(
                model,
                optimizer,
                tx[4 * (step - 1) : 4 * step],
                ty[4 * (step - 1) : 4 * step],
                step=step,
                retain_raw=step in study.SNAPSHOTS,
            )
            torch.cuda.synchronize()
            row["wall_seconds_including_observation"] = time.perf_counter() - begin
            row["task_row_ids"] = indices[4 * (step - 1) : 4 * step].tolist()
            rows.append(row)
            print(f"task {case} step{step}/8 CE={row['task_CE']:.9g}", flush=True)
            if raw is not None:
                raw["accepted"]["configuration"] = config
                torch.save(raw, root / f"update-{step}.pt")
                # No duplicate accepted checkpoint: update-N.pt owns stateN.
            write(root / "updates.json", rows)
            write(
                args.output / "progress.json",
                {
                    "stage": "task-probe",
                    "case": case,
                    "updates": step,
                    "completed_cases": completed,
                },
            )
        completed.append(case)
        write(
            root / "complete.json",
            {
                "case": case,
                "identity": identity,
                "task_updates": 8,
                "saved_states": list(study.SNAPSHOTS),
                "state_locator": "state0 in state-0.pt; states1/2/4/8 in update-N.pt.accepted",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "wall_scope": "includes full diagnostic copies/statistics; not a training throughput comparison",
            },
        )
        del model, optimizer
    write(
        args.output / "run-summary.json",
        {
            "status": "completed",
            "identity": identity,
            "cases": completed,
            "qualifications": qualifications,
            "task_updates_per_case": 8,
            "development_evaluated": False,
            "selection_performed": False,
            "future_confirmation_evaluated": False,
            "official_test_evaluated": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data",
        "data-manifest",
        "consumer-manifest",
        "external-source",
        "external-manifest",
        "runtime-file",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
