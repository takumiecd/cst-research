"""New-cohort eager depth/layer-position pilot over verified external975 operators.

This consumer owns the architecture and training loop. It does not masquerade as
an old single-middle-layer cohort or change public torchcst/Graph dispatch.
"""

from __future__ import annotations

import argparse
import copy
import re
import time
from pathlib import Path
from types import SimpleNamespace

from .higgs_depth_data import data_spec, load_pilot
from .higgs_dst_comparison import file_sha, json_sha, read, write

COHORT = "higgs-depth-position-v1"
REPLACEMENTS = {
    "dense": (),
    "early": (0,),
    "middle": (1,),
    "late": (2,),
    "all": (0, 1, 2),
}
DIVISORS = (10000, 1000, 100)
SOURCE = "97580582ed28c4e494bf5c2656bbe94a85c544c3"
SOURCE_MANIFEST = "17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931"


def protocol():
    return {
        "version": COHORT,
        "external_source_commit": SOURCE,
        "architecture": [28, 1024, 1024, 1024, 1024, 2],
        "square_hidden_layers": 3,
        "square_bias": False,
        "relu_layers": 4,
        "dense_io": True,
        "all_layers_trainable": True,
        "positions": list(REPLACEMENTS),
        "divisors": list(DIVISORS),
        "alphas": [0.0001, 0.001, 0.01],
        "seed": 17,
        "epochs": 3,
        "batch_size": 4,
        "lr": 0.0003,
        "position_multiplier": 1.0,
        "mode": "eager",
        "dtype": "FP32",
        "TF32": False,
        "pilot_configurations": 13,
        "normalization": "new training100k only; common across all arms",
        "data_specification_sha256": json_sha(data_spec()),
        "sequence": "dense+9single-position cases, inspect learning/gradient diagnostics, then3all-hidden cases",
        "selection": "earliest minimum new-development CE; no density-specific LR grid",
        "init": "ordinary input/output seed+1/+2; dense square seed+100+j; CST atoms seed+1000+j; no gain calibration",
        "confirmation_evaluated": False,
        "official_test_evaluated": False,
        "formal_claim": False,
    }


def configuration(kind, divisor, width=1024):
    if kind not in REPLACEMENTS or type(width) is not int or width < 4:
        raise ValueError("invalid depth model configuration")
    if kind == "dense" and divisor is not None:
        raise ValueError("dense has no atom density")
    if kind != "dense" and divisor not in DIVISORS:
        raise ValueError("choose declared coarse alpha")
    count = None if divisor is None else width * width // divisor
    if count is not None and count < 1:
        raise ValueError("configuration requires at least one atom")
    return {
        "cohort": COHORT,
        "kind": kind,
        "width": width,
        "divisor": divisor,
        "alpha": None if divisor is None else 1 / divisor,
        "atoms_per_cst_layer": count,
        "replaced_square_layers": list(REPLACEMENTS[kind]),
        "architecture": [28, width, width, width, width, 2],
        "middle_indices": [2, 4, 6],
        "id": "dense" if kind == "dense" else f"{kind}-D{divisor}",
    }


def make_model(kind, divisor, seed=17, width=1024):
    import torch
    from benchmarks.training.cell_mass_mnist import make_radius_cst
    from torch import nn
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

    cfg = configuration(kind, divisor, width)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        first = nn.Linear(28, width)
        torch.manual_seed(seed + 2)
        last = nn.Linear(width, 2)
        layers = [first, nn.ReLU()]
        for j in range(3):
            if j in REPLACEMENTS[kind]:
                torch.manual_seed(seed + 1000 + j)
                layer = make_radius_cst(
                    width,
                    cfg["atoms_per_cst_layer"],
                    execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
                )
            else:
                torch.manual_seed(seed + 100 + j)
                layer = nn.Linear(width, width, bias=False)
            layers.extend([layer, nn.ReLU()])
        layers.append(last)
    model = nn.Sequential(*layers)
    model._cst_research_cohort = COHORT
    model._depth_configuration = cfg
    return model


def make_optimizer(model, *, cpu_reference=False):
    import torch
    from benchmarks.training.cell_mass_mnist import radius_optimizer_contract
    from torchcst import CSTAdamW, FixedSelector, make_algorithm_registry
    from torchcst._backends.cuda.algorithms.polar_update.plans import (
        RADIUS_FLAT_TORUS_FUSED,
    )
    from torchcst.nn import CSTModule

    contract = radius_optimizer_contract()
    contract["lr"] = 0.0003
    atoms = [site.atoms.p for site in model.modules() if isinstance(site, CSTModule)]
    atom_ids = {id(p) for p in atoms}
    ordinary = [p for p in model.parameters() if id(p) not in atom_ids]
    groups = [
        {
            "params": ordinary,
            "weight_decay": contract["ordinary_weight_decay"],
            "role": "ordinary",
        }
    ]
    if atoms:
        groups.append(
            {
                "params": atoms,
                "weight_decay": 0.0,
                "amplitude_weight_decay": contract["amplitude_weight_decay"],
                "role": "atoms",
            }
        )
    options = {
        k: contract[k]
        for k in ("lr", "fused", "capturable", "betas", "eps", "amsgrad", "maximize")
    }
    if cpu_reference:
        options["fused"] = False
    if not atoms:
        return torch.optim.AdamW(groups, **options), contract
    kwargs = (
        {}
        if cpu_reference
        else {
            "update_selector": FixedSelector(
                RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry()
            )
        }
    )
    return CSTAdamW(
        groups, model=model, check_finite=cpu_reference, **kwargs, **options
    ), contract


def clone_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_cpu(v) for v in value)
    return copy.deepcopy(value)


def checkpoint(model, optimizer, epoch, updates, configuration, development):
    return clone_cpu(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gradients": {n: p.grad for n, p in model.named_parameters()},
            "epoch": epoch,
            "updates": updates,
            "configuration": configuration,
            "development": development,
        }
    )


def tensor_statistics(value):
    import torch

    value = value.detach()
    return {
        "shape": list(value.shape),
        "rms": float(value.double().square().mean().sqrt()),
        "max_abs": float(value.abs().max()),
        "zero_fraction": float((value == 0).double().mean()),
        "finite": bool(torch.isfinite(value).all()),
    }


def layer_diagnostics(model, probe):
    import torch

    by_layer = {str(i): [] for i in range(len(model))}
    with torch.no_grad():
        for batch in probe.split(4):
            value = batch
            for i, module in enumerate(model):
                value = module(value)
                by_layer[str(i)].append(value.detach().cpu())
    return {
        "activations": {
            i: tensor_statistics(torch.cat(values)) for i, values in by_layer.items()
        },
        "parameter_gradients": {
            name: None if p.grad is None else tensor_statistics(p.grad)
            for name, p in model.named_parameters()
        },
        "probe_rows": len(probe),
        "scope": "fixed first256 new-training rows; diagnostics outside training time",
    }


def reference_forward(model, x):
    """Compose public CPU FP64 dense_weight references, without CUDA arithmetic."""
    from torch import nn
    from torchcst.nn import CSTModule

    for module in model:
        x = (
            nn.functional.linear(x, module.dense_weight())
            if isinstance(module, CSTModule)
            else module(x)
        )
    return x


def qualify(config, seed, output, device):
    import torch
    from benchmarks.training.full_step_graph import ordinary_step
    from benchmarks.training.higgs_verify import (
        byte_equal,
        finite_fields,
        tensor_fields,
    )
    from benchmarks.training.owner_window_fp32 import errors

    output.mkdir()
    model = make_model(config["kind"], config["divisor"], seed).to(device)
    generator = torch.Generator().manual_seed(seed + 800)
    x = torch.randn((4, 28), generator=generator).to(device).requires_grad_()
    reference = copy.deepcopy(model).cpu().double()
    rx = x.detach().cpu().double().requires_grad_()
    actual_y = model(x)
    expected_y = reference_forward(reference, rx)
    dy = torch.tensor(
        [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]], device=device
    )
    actual = (actual_y, *torch.autograd.grad(actual_y, (x, *model.parameters()), dy))
    expected = (
        expected_y,
        *torch.autograd.grad(
            expected_y, (rx, *reference.parameters()), dy.cpu().double()
        ),
    )
    metrics = [errors(a, r) for a, r in zip(actual, expected)]
    torch.save(
        {
            "actual": clone_cpu(actual),
            "expected": clone_cpu(expected),
            "configuration": config,
        },
        output / "full-network-vjp.pt",
    )
    if not all(r["pass"] for r in metrics):
        raise ValueError("new-depth independent Y/dX/all-parameter oracle failed")
    model = make_model(config["kind"], config["divisor"], seed).to(device)
    optimizer, _ = make_optimizer(model)
    tx = torch.randn((32, 28), generator=generator).to(device)
    ty = torch.arange(32, device=device) % 2
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    midway = None
    for step in range(8):
        ordinary_step(
            model,
            optimizer,
            tx,
            ty,
            torch.arange(step * 4, step * 4 + 4, device=device),
            totals,
        )
        if step == 3:
            midway = checkpoint(model, optimizer, 0, 4, config, None)
    final = checkpoint(model, optimizer, 0, 8, config, None)
    resumed = make_model(config["kind"], config["divisor"], seed).to(device)
    restored, _ = make_optimizer(resumed)
    resumed.load_state_dict(midway["model"])
    restored.load_state_dict(midway["optimizer"])
    for step in range(4, 8):
        ordinary_step(
            resumed,
            restored,
            tx,
            ty,
            torch.arange(step * 4, step * 4 + 4, device=device),
            totals,
        )
    after = checkpoint(resumed, restored, 0, 8, config, None)
    a, b = dict(tensor_fields(final)), dict(tensor_fields(after))
    equal = a.keys() == b.keys() and all(byte_equal(a[k], b[k]) for k in a)
    if not equal or finite_fields(a):
        raise ValueError("new-depth eager resume/finite gate failed")
    if any(float(s["step"]) != 8 for s in optimizer.state.values()):
        raise ValueError("new-depth optimizer clocks differ")
    torch.save(
        {"midway": midway, "final": final, "resumed": after}, output / "eager-resume.pt"
    )
    record = {
        "status": "passed",
        "configuration": config,
        "independent_full_network_vjp": metrics,
        "eight_steps_four_resume_tensor_equal": equal,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "future_confirmation_evaluated": False,
        "official_test_evaluated": False,
    }
    write(output / "qualification.json", record)
    return record


def run(args):
    from . import higgs_dst_confirmation_driver as common

    consumer = Path(__file__).resolve().parents[3]
    if not re.fullmatch(r"[0-9a-f]{40}", args.consumer_commit):
        raise ValueError("full committed consumer identity required")
    common.verify_consumer(
        consumer,
        args.consumer_manifest,
        args.consumer_commit,
        consumer / common.DRIVER_PATH,
    )
    # Verify source before importing either torchcst or its benchmark helpers.
    from .higgs_coordinate_lr import verify_external

    verify_external(
        SimpleNamespace(
            external_source=args.external_source,
            external_manifest=args.external_manifest,
            identity={"external_source_manifest_sha256": SOURCE_MANIFEST},
        )
    )
    import numpy as np
    import torch
    from benchmarks.training.full_step_graph import ordinary_step, snapshot
    from benchmarks.training.higgs_confirmation import _configure, evaluate
    from benchmarks.training.higgs_verify import finite_fields, tensor_hash

    cfg = configuration(args.kind, args.divisor)
    if args.epochs != 3 or args.seed != 17:
        raise ValueError("this coarse pilot fixes3epochs/seed17")
    arrays, data_manifest = load_pilot(args.data, args.data_manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    _configure()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(
            "training requires explicitly admitted CUDA; CPU tests do not imply a GPU result"
        )
    begin_preparation = time.perf_counter()
    qual = qualify(cfg, args.seed, args.output / "qualification", device)
    qualification_seconds = time.perf_counter() - begin_preparation
    import gc

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    model = make_model(args.kind, args.divisor, args.seed).to(device)
    optimizer, contract = make_optimizer(model)
    configuration_record = {
        **cfg,
        "seed": args.seed,
        "epochs": 3,
        "batch_size": 4,
        "lr": 0.0003,
        "protocol_sha256": json_sha(protocol()),
        "data_sha256": file_sha(args.data),
        "data_manifest_sha256": file_sha(args.data_manifest),
        "external_source_commit": SOURCE,
        "external_source_manifest_sha256": SOURCE_MANIFEST,
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "optimizer_contract": contract,
    }
    torch.save(
        {"model": clone_cpu(model.state_dict()), "configuration": configuration_record},
        args.output / "initialization.pt",
    )
    tx = torch.from_numpy(arrays["x_train"]).to(device)
    ty = torch.from_numpy(arrays["y_train"]).to(device)
    vx = torch.from_numpy(arrays["x_development"]).to(device)
    vy = torch.from_numpy(arrays["y_development"]).to(device)
    initial = layer_diagnostics(model, tx[:256])
    initial_dev, _, _ = evaluate(model, vx, vy)
    preparation_seconds = time.perf_counter() - begin_preparation
    rows = []
    best = None
    updates = 0
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 4):
        order = torch.randperm(
            len(tx),
            generator=torch.Generator().manual_seed(args.seed + 100 + epoch - 1),
        )
        order_sha = tensor_hash(order)["sha256"]
        indices = order.to(device)
        for total in totals:
            total.zero_()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for step in range(25000):
            ordinary_step(
                model, optimizer, tx, ty, indices[4 * step : 4 * step + 4], totals
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        updates += 25000
        state = snapshot(model, optimizer)
        gradients = {
            name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
        }
        if finite_fields(state) or any(
            not bool(torch.isfinite(g).all()) for g in gradients.values()
        ):
            raise FloatingPointError("nonfinite depth model/optimizer/gradient")
        if len(optimizer.state) != len(list(model.parameters())) or any(
            float(s["step"]) != updates for s in optimizer.state.values()
        ):
            raise ValueError("depth parameter ownership/clock gate failed")
        development, logits, labels = evaluate(model, vx, vy)
        diagnostics = layer_diagnostics(model, tx[:256])
        after_diagnostics = snapshot(model, optimizer)
        if after_diagnostics.keys() != state.keys() or any(
            not torch.equal(state[k], after_diagnostics[k]) for k in state
        ):
            raise ValueError("evaluation/diagnostics changed model/optimizer tensors")
        if any(not s["finite"] for s in diagnostics["activations"].values()):
            raise FloatingPointError("nonfinite depth activation")
        row = {
            "epoch": epoch,
            "updates": updates,
            "training_seconds": seconds,
            "train_loss": float(totals[0]) / 100000,
            "train_accuracy": float(totals[1]) / 100000,
            "development": development,
            "order_sha256": order_sha,
            "state": {k: tensor_hash(v) for k, v in state.items()},
            "gradients": {k: tensor_hash(v) for k, v in gradients.items()},
            "layer_diagnostics": diagnostics,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        raw = checkpoint(
            model, optimizer, epoch, updates, configuration_record, development
        )
        torch.save(raw, args.output / f"epoch-{epoch}.pt")
        if best is None or development["loss"] < best["development"]["loss"]:
            torch.save(raw, args.output / "best-dev.pt")
            np.savez_compressed(
                args.output / "best-dev-predictions.npz",
                logits=logits,
                labels=labels,
                row_ids=np.arange(1100000, 1150000, dtype=np.int64),
            )
            best = {
                "epoch": epoch,
                "updates": updates,
                "development": development,
                "checkpoint_sha256": file_sha(args.output / "best-dev.pt"),
                "predictions_sha256": file_sha(
                    args.output / "best-dev-predictions.npz"
                ),
            }
            write(args.output / "best-dev.json", best)
        rows.append(row)
        write(
            args.output / "learning.json",
            {
                "cohort": COHORT,
                "configuration": configuration_record,
                "initial_development": initial_dev,
                "initial_layer_diagnostics": initial,
                "epochs": rows,
                "best_development_checkpoint": best,
                "qualification_sha256": file_sha(
                    args.output / "qualification/qualification.json"
                ),
                "qualification": qual,
                "qualification_seconds": qualification_seconds,
                "preparation_seconds": preparation_seconds,
                "gpu_profile": {
                    "name": torch.cuda.get_device_name(device),
                    "capability": list(torch.cuda.get_device_capability(device)),
                    "multiprocessors": torch.cuda.get_device_properties(
                        device
                    ).multi_processor_count,
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        device
                    ).total_memory,
                },
                "data_exposure_audit": data_manifest["exposure"],
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "mode": "eager",
                "TF32": False,
                "timing_scope": "synchronized gather/forward/loss/backward/optimizer/train metrics; excludes evaluation/diagnostics/checkpoint I/O",
                "memory_scope": "resident model/data/optimizer and diagnostics; allocator allocated/reserved, not total process memory",
                "future_confirmation_evaluated": False,
                "official_test_evaluated": False,
            },
        )
    import shutil

    shutil.copy2(args.output / "epoch-3.pt", args.output / "final.pt")
    return read(args.output / "learning.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data",
        "data-manifest",
        "external-source",
        "external-manifest",
        "consumer-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--kind", choices=tuple(REPLACEMENTS), required=True)
    parser.add_argument("--divisor", type=int, choices=DIVISORS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
