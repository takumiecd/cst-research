"""Bounded forward-conditioning intervention over unchanged external975 operators.

Preparation is not a GPU result. Calibration consumes unlabelled initial training
features only; fixed output multipliers change the effective function/gradient
scale and amplitude/regularization mapping, not merely parameter initialization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import re
import struct
import time
from pathlib import Path
from types import SimpleNamespace

from . import higgs_depth_study as stock
from .higgs_depth_data import load_pilot
from .higgs_dst_comparison import file_sha, json_sha, write

COHORT = "higgs-depth-gain-v1"
ARMS = ("dense", "stock-all", "gain-all")
DIVISOR = 10000
GAIN_BOUNDS = (1e-6, 1e6)


class CalibrationInconclusive(ValueError):
    """The declared one-shot calibration is infeasible; do not clip or tune."""


def protocol():
    return {
        "version": COHORT,
        "parent_consumer_commit": "f6d08ad76001a69070b0965cefe7614f192ddf3f",
        "external_source_commit": stock.SOURCE,
        "architecture": [28, 1024, 1024, 1024, 1024, 2],
        "arms": list(ARMS),
        "divisor": DIVISOR,
        "alpha": 0.0001,
        "seed": 17,
        "epochs": 3,
        "batch_size": 4,
        "lr": 0.0003,
        "position_multiplier": 1.0,
        "mode": "eager",
        "dtype": "FP32",
        "TF32": False,
        "calibration": {
            "rows": 256,
            "partition": "first256 new-training feature rows in original order",
            "labels_used": False,
            "development_used": False,
            "target": "corresponding Dense initial square-linear-output RMS on the same features",
            "algorithm": "sequential public CPU-FP64 dense_weight composition; preceding gains applied before calibrating the next site",
            "gain_storage": "three nontrainable FP32 gains in one fixed vector buffer, fixed throughout optimization",
            "gain_bounds_inclusive": list(GAIN_BOUNDS),
            "clip_or_search": False,
            "infeasible": "inconclusive; no training or automatic replacement target",
        },
        "selection": "earliest minimum development CE within each arm",
        "qualification": "all3 arms normal elementwise Y/dX/all-grad plus field/packed-column norm and material coverage; 8step/4resume/finite/ownership/clocks before any learning",
        "amplified_componentwise_role": "diagnostic only; inherited failed stress remains failed, no arbitrary-cotangent per-component claim",
        "training_metrics": "online train CE/accuracy and separate full-train/development evaluation at the same checkpoint outside the training clock",
        "scope": "post-observation exploratory forward-conditioning intervention; changes effective amplitude bound, gradients and regularization mapping",
        "adaptive_density_choice": "Initially prepared alpha.001 without execution; changed before any gain GPU result to alpha.0001 after stock13 development curves showed all-D10000 stalled at AUC.5 while .001 and .01 learned. Exploratory targeting, not held-out confirmation.",
        "formal_claim": False,
        "future_confirmation_evaluated": False,
        "official_test_evaluated": False,
        "automatic_followup": False,
    }


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def parameter_hashes(model):
    return {n: tensor_sha(p) for n, p in model.named_parameters()}


def make_model(arm, *, seed=17, width=1024, gains=None):
    import torch
    from torch import nn

    if arm not in ARMS:
        raise ValueError("undeclared gain intervention arm")
    base = stock.make_model(
        "dense" if arm == "dense" else "all",
        None if arm == "dense" else DIVISOR,
        seed,
        width,
    )

    class FixedGainSequential(nn.Sequential):
        def forward(self, value):
            for i, layer in enumerate(self):
                value = layer(value)
                if i in (2, 4, 6):
                    value = value * self.fixed_output_gains[(i - 2) // 2]
            return value

    model = FixedGainSequential(*list(base.children()))
    model.register_buffer("fixed_output_gains", torch.ones(3, dtype=torch.float32))
    model._gain_arm = arm
    model._depth_configuration = base._depth_configuration
    model._cst_research_cohort = COHORT
    if gains is not None:
        install_gains(model, gains)
    return model


def install_gains(model, gains):
    import torch

    values = torch.as_tensor(gains, dtype=torch.float32)
    if (
        values.shape != (3,)
        or not bool(torch.isfinite(values).all())
        or not bool(((values >= GAIN_BOUNDS[0]) & (values <= GAIN_BOUNDS[1])).all())
    ):
        raise CalibrationInconclusive(
            "fixed gains must lie within declared positive bounds"
        )
    if model._gain_arm != "gain-all" and not bool((values == 1).all()):
        raise ValueError("controls require unit gains")
    with torch.no_grad():
        model.fixed_output_gains.copy_(values.to(model.fixed_output_gains.device))


def reference_forward(model, value):
    """Independent public CPU dense_weight composition with constant gain buffers."""
    from torch import nn
    from torchcst.nn import CSTModule

    for i, layer in enumerate(model):
        value = (
            nn.functional.linear(value, layer.dense_weight())
            if isinstance(layer, CSTModule)
            else layer(value)
        )
        if i in (2, 4, 6):
            value = value * model.fixed_output_gains[(i - 2) // 2]
    return value


def activation_trace(model, value):
    """Post-module activations, including the model's fixed square-output gains."""
    activations = []
    for i, layer in enumerate(model):
        value = layer(value)
        if i in (2, 4, 6):
            value = value * model.fixed_output_gains[(i - 2) // 2]
        activations.append(value)
    return activations


def layer_diagnostics(model, probe):
    import torch

    by_layer = {str(i): [] for i in range(len(model))}
    with torch.no_grad():
        for batch in probe.split(4):
            for i, value in enumerate(activation_trace(model, batch)):
                by_layer[str(i)].append(value.detach().cpu())
    return {
        "activations": {
            i: stock.tensor_statistics(torch.cat(values))
            for i, values in by_layer.items()
        },
        "parameter_gradients": {
            name: None if p.grad is None else stock.tensor_statistics(p.grad)
            for name, p in model.named_parameters()
        },
        "probe_rows": len(probe),
        "scope": "fixed first256 new-training rows; gain-aware post-module activations; diagnostics outside training time",
        "fixed_output_gains": model.fixed_output_gains.detach().cpu().tolist(),
    }


def timed_cuda_call(device, function, *args):
    """Record a separate synchronized component, outside the training clock."""
    import torch

    torch.cuda.synchronize(device)
    begin = time.perf_counter()
    result = function(*args)
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - begin


def bounded_gain(target, observed):
    if (
        not math.isfinite(target)
        or not math.isfinite(observed)
        or target <= 0
        or observed <= 0
    ):
        raise CalibrationInconclusive("zero/nonfinite target or calibration signal")
    ratio = target / observed
    if not math.isfinite(ratio) or not GAIN_BOUNDS[0] <= ratio <= GAIN_BOUNDS[1]:
        raise CalibrationInconclusive(
            "required gain lies outside fixed bounds; no clipping"
        )
    rounded = struct.unpack("f", struct.pack("f", ratio))[0]
    if not GAIN_BOUNDS[0] <= rounded <= GAIN_BOUNDS[1]:
        raise CalibrationInconclusive("FP32 gain outside declared bounds")
    return rounded


def calibrate(dense, intervention, features):
    """No label argument; untrained copies are used and trainable tensors stay exact."""
    import torch
    from torch import nn
    from torchcst.nn import CSTModule

    if (
        features.shape != (256, 28)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("calibration needs exactly256 finite CPU FP32 feature rows")
    if dense._gain_arm != "dense" or intervention._gain_arm != "gain-all":
        raise ValueError("Dense target and declared intervention required")
    if any(
        p.grad is not None for m in (dense, intervention) for p in m.parameters()
    ) or not bool((intervention.fixed_output_gains == 1).all()):
        raise ValueError(
            "calibration must precede gradients/updates and begin at unit gains"
        )
    before = parameter_hashes(intervention)
    ref = copy.deepcopy(dense).cpu().double()
    candidate = copy.deepcopy(intervention).cpu().double()
    rows = []
    with torch.no_grad():
        value = features.double()
        targets = {}
        for i, layer in enumerate(ref):
            value = layer(value)
            if i in (2, 4, 6):
                targets[i] = float(value.square().mean().sqrt())
        value = features.double()
        gains = []
        for i, layer in enumerate(candidate):
            value = (
                nn.functional.linear(value, layer.dense_weight())
                if isinstance(layer, CSTModule)
                else layer(value)
            )
            if i in (2, 4, 6):
                observed = float(value.square().mean().sqrt())
                gain = bounded_gain(targets[i], observed)
                value = value * gain
                gains.append(gain)
                rows.append(
                    {
                        "layer": i,
                        "target_dense_RMS": targets[i],
                        "sequential_pre_gain_RMS": observed,
                        "fixed_FP32_gain": gain,
                        "post_gain_reference_RMS": float(value.square().mean().sqrt()),
                    }
                )
    install_gains(intervention, gains)
    if parameter_hashes(intervention) != before:
        raise ValueError("calibration changed a trainable tensor")
    return {
        "status": "calibrated",
        "algorithm": protocol()["calibration"],
        "rows": rows,
        "gains": gains,
        "features_sha256": tensor_sha(features),
        "calibration_rows": 256,
        "trainable_tensors_unchanged": True,
        "initial_trainable_hashes": before,
        "labels_read": False,
        "development_read": False,
        "confirmation_read": False,
        "test_read": False,
    }


def gradient_norm_gate(actual, expected):
    """No epsilon: exact-zero reference permits only exact-zero actual."""
    import torch

    a, r = actual.double(), expected.double()
    if a.shape != r.shape or not bool(
        torch.isfinite(a).all() and torch.isfinite(r).all()
    ):
        return {"pass": False, "reason": "nonfinite/shape"}
    maximum = float(r.abs().max())
    if maximum == 0:
        exact = bool((a == 0).all())
        return {
            "pass": exact,
            "reference_exact_zero": True,
            "actual_exact_zero": exact,
            "relative_L2": None,
            "maxerror_over_refmax": None,
            "negative_control_scope": "zero/signflip identical to zero reference; not informative",
        }
    # Normalize before the norm: no epsilon and no false exact-zero inference
    # from underflow/overflow of an unscaled squared norm.
    normalized_reference = r / maximum
    normalized_error = (a - r) / maximum
    norm = float(torch.linalg.vector_norm(normalized_reference))
    relative = float(torch.linalg.vector_norm(normalized_error)) / norm
    maximum_error = float(normalized_error.abs().max())
    zero_error = float(torch.linalg.vector_norm(-normalized_reference)) / norm
    signflip_error = float(torch.linalg.vector_norm(-2 * normalized_reference)) / norm
    return {
        "pass": relative <= 3e-4 and maximum_error <= 3e-4,
        "reference_exact_zero": False,
        "relative_L2": relative,
        "maxerror_over_refmax": maximum_error,
        "zero_substitution_rejected": zero_error > 3e-4,
        "signflip_substitution_rejected": signflip_error > 3e-4,
        "zero_substitution_relative_L2": zero_error,
        "signflip_substitution_relative_L2": signflip_error,
    }


def qualified_model(arm, gains, device):
    return make_model(arm, gains=gains if arm == "gain-all" else None).to(device)


def qualify(arm, gains, output, device):
    import torch
    from benchmarks.training.full_step_graph import ordinary_step
    from benchmarks.training.higgs_verify import (
        byte_equal,
        finite_fields,
        tensor_fields,
    )
    from benchmarks.training.owner_window_fp32 import errors

    output.mkdir()
    model = qualified_model(arm, gains, device)
    reference = copy.deepcopy(model).cpu().double()
    generator = torch.Generator().manual_seed(817)
    probe = torch.randn(4, 28, generator=generator)
    names = ["Y", "dX"] + [n for n, _ in model.named_parameters()]
    reports = []
    # Normal elementwise remains mandatory. Amplification only improves signal
    # coverage; componentwise stress failures are recorded, never relabelled.
    for scale in (1.0, 1e12):
        x = probe.to(device).requires_grad_()
        rx = probe.double().requires_grad_()
        y = model(x)
        ry = reference_forward(reference, rx)
        dy = (
            torch.tensor(
                [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]],
                dtype=torch.float32,
                device=device,
            )
            * scale
        )
        actual = (y, *torch.autograd.grad(y, (x, *model.parameters()), dy))
        expected = (
            ry,
            *torch.autograd.grad(ry, (rx, *reference.parameters()), dy.cpu().double()),
        )
        raw = {
            "actual": stock.clone_cpu(actual),
            "expected": stock.clone_cpu(expected),
            "fields": names,
            "cotangent": dy.cpu(),
            "scale": scale,
        }
        torch.save(raw, output / f"vjp-scale-{scale:.0e}.pt")
        elementwise = {n: errors(a, r) for n, a, r in zip(names, actual, expected)}
        norms = {}
        coverage = {}
        for name, a, r in zip(names[1:], actual[1:], expected[1:]):
            norms[name] = gradient_norm_gate(a, r.to(a.device))
            if name.endswith("atoms.p"):
                norms[name]["packed_columns"] = [
                    gradient_norm_gate(a[:, i], r[:, i].to(a.device)) for i in range(4)
                ]
            if scale == 1e12 and (name == "dX" or name.endswith("atoms.p")):
                coverage[name] = bool(
                    (r.abs() > (3e-5 + 3e-4 * r.abs())).any()
                ) and bool((a != 0).any())
        record = {
            "scale": scale,
            "elementwise": elementwise,
            "norms": norms,
            "critical_material_coverage": coverage,
        }
        reports.append(record)
        write(output / "vjp-report.json", reports)
        if (
            (scale == 1.0 and not all(v["pass"] for v in elementwise.values()))
            or not all(
                v["pass"] and all(c["pass"] for c in v.get("packed_columns", []))
                for v in norms.values()
            )
            or not all(coverage.values())
        ):
            raise ValueError(
                "gain pilot full-network normal/norm/material-coverage gate failed"
            )
    # Scaling the cotangent must not change either forward result.
    one = torch.load(output / "vjp-scale-1e+00.pt", weights_only=True)
    large = torch.load(output / "vjp-scale-1e+12.pt", weights_only=True)
    if not byte_equal(one["actual"][0], large["actual"][0]) or not byte_equal(
        one["expected"][0], large["expected"][0]
    ):
        raise ValueError("cotangent changed forward")
    # Fresh identical buffers/initialization, 8 updates vs4+restore+4.
    model = qualified_model(arm, gains, device)
    optimizer, _ = stock.make_optimizer(model)
    resume_cfg = {
        "cohort": COHORT,
        "arm": arm,
        "gains": model.fixed_output_gains.cpu().tolist(),
    }
    tx = torch.randn(32, 28, generator=generator).to(device)
    ty = torch.arange(32, device=device) % 2
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    midway = None
    for step in range(8):
        ordinary_step(
            model,
            optimizer,
            tx,
            ty,
            torch.arange(4 * step, 4 * step + 4, device=device),
            totals,
        )
        if step == 3:
            midway = stock.checkpoint(model, optimizer, 0, 4, resume_cfg, None)
    final = stock.checkpoint(model, optimizer, 0, 8, resume_cfg, None)
    resumed = qualified_model(arm, gains, device)
    restored, _ = stock.make_optimizer(resumed)
    resumed.load_state_dict(midway["model"])
    restored.load_state_dict(midway["optimizer"])
    for step in range(4, 8):
        ordinary_step(
            resumed,
            restored,
            tx,
            ty,
            torch.arange(4 * step, 4 * step + 4, device=device),
            totals,
        )
    after = stock.checkpoint(resumed, restored, 0, 8, resume_cfg, None)
    a, b = dict(tensor_fields(final)), dict(tensor_fields(after))
    torch.save(
        {"midway": midway, "final": final, "resumed": after}, output / "eager-resume.pt"
    )
    if (
        a.keys() != b.keys()
        or not all(byte_equal(a[k], b[k]) for k in a)
        or finite_fields(a)
        or len(optimizer.state) != 7
        or any(float(v["step"]) != 8 for v in optimizer.state.values())
    ):
        raise ValueError("gain pilot resume/finite/ownership/clock gate failed")
    for snapshot in (midway, final, after):
        if not torch.equal(
            snapshot["model"]["fixed_output_gains"], model.fixed_output_gains.cpu()
        ):
            raise ValueError("qualification gain buffer mutated")
    record = {
        "status": "passed",
        "arm": arm,
        "gains": model.fixed_output_gains.cpu().tolist(),
        "vjp": reports,
        "eight_steps_four_resume_tensor_equal": True,
        "optimizer_states": 7,
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
    from .higgs_coordinate_lr import verify_external

    verify_external(
        SimpleNamespace(
            external_source=args.external_source,
            external_manifest=args.external_manifest,
            identity={"external_source_manifest_sha256": stock.SOURCE_MANIFEST},
        )
    )
    import numpy as np
    import torch
    from benchmarks.training.full_step_graph import ordinary_step, snapshot
    from benchmarks.training.higgs_confirmation import _configure, evaluate
    from benchmarks.training.higgs_verify import finite_fields, tensor_hash

    arrays, data_manifest = load_pilot(args.data, args.data_manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    _configure()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(
            "actual admitted CUDA required; CPU tests are not model-quality evidence"
        )
    fingerprint = {
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "sms": torch.cuda.get_device_properties(device).multi_processor_count,
        "memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    preparation = time.perf_counter()
    dense = make_model("dense")
    intervention = make_model("gain-all")
    stock_control = make_model("stock-all")
    calibration_features = torch.from_numpy(arrays["x_train"][:256].copy())
    try:
        calibration = calibrate(dense, intervention, calibration_features)
    except CalibrationInconclusive as exc:
        write(
            args.output / "inconclusive.json",
            {
                "status": "inconclusive-calibration",
                "reason": str(exc),
                "protocol": protocol(),
                "training_started": False,
            },
        )
        raise SystemExit(2)
    hashes = parameter_hashes(stock_control)
    if hashes != parameter_hashes(intervention):
        raise ValueError("stock/gain trainable initialization differs")
    for name in ("0.weight", "0.bias", "8.weight", "8.bias"):
        if parameter_hashes(dense)[name] != hashes[name]:
            raise ValueError("ordinary IO initialization differs")
    gains = calibration["gains"]
    calibration.update(
        {
            "data_sha256": file_sha(args.data),
            "data_manifest_sha256": file_sha(args.data_manifest),
            "all_trainable_initialization_equal": True,
            "raw_row_ids": [1000000, 1000256],
        }
    )
    torch.save(
        {
            "features": calibration_features,
            "dense_initial_model": stock.clone_cpu(dense.state_dict()),
            "stock_initial_model": stock.clone_cpu(stock_control.state_dict()),
            "calibrated_initial_model": stock.clone_cpu(intervention.state_dict()),
            "calibration": calibration,
        },
        args.output / "calibration.pt",
    )
    write(args.output / "calibration.json", calibration)
    del dense, intervention, stock_control
    qualifications = {}
    # Fail all arms before any training if any fresh numerical gate fails.
    for arm in ARMS:
        qualifications[arm] = qualify(
            arm, gains, args.output / ("qualification-" + arm), device
        )
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    tx = torch.from_numpy(arrays["x_train"]).to(device)
    ty = torch.from_numpy(arrays["y_train"]).to(device)
    vx = torch.from_numpy(arrays["x_development"]).to(device)
    vy = torch.from_numpy(arrays["y_development"]).to(device)
    common_preparation = time.perf_counter() - preparation
    arms = {}
    for arm in ARMS:
        directory = args.output / arm
        directory.mkdir()
        start = time.perf_counter()
        model = qualified_model(arm, gains, device)
        optimizer, contract = stock.make_optimizer(model)
        cfg = {
            "cohort": COHORT,
            "arm": arm,
            "seed": 17,
            "epochs": 3,
            "width": 1024,
            "divisor": None if arm == "dense" else DIVISOR,
            "lr": 0.0003,
            "batch_size": 4,
            "gains": model.fixed_output_gains.cpu().tolist(),
            "calibration_sha256": file_sha(args.output / "calibration.json"),
            "protocol_sha256": json_sha(protocol()),
            "consumer_commit": args.consumer_commit,
            "consumer_manifest_sha256": file_sha(args.consumer_manifest),
            "external_source_commit": stock.SOURCE,
            "external_source_manifest_sha256": stock.SOURCE_MANIFEST,
            "data_sha256": file_sha(args.data),
            "data_manifest_sha256": file_sha(args.data_manifest),
            "optimizer_contract": contract,
        }
        torch.save(
            {"model": stock.clone_cpu(model.state_dict()), "configuration": cfg},
            directory / "initialization.pt",
        )
        initial, initial_diagnostics_seconds = timed_cuda_call(
            device, layer_diagnostics, model, tx[:256]
        )
        (initial_dev, _, _), initial_development_seconds = timed_cuda_call(
            device, evaluate, model, vx, vy
        )
        arm_preparation = time.perf_counter() - start
        rows = []
        best = None
        updates = 0
        totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, 4):
            order = torch.randperm(
                100000, generator=torch.Generator().manual_seed(117 + epoch - 1)
            )
            indices = order.to(device)
            for t in totals:
                t.zero_()
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
                n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
            }
            if (
                finite_fields(state)
                or any(not bool(torch.isfinite(g).all()) for g in gradients.values())
                or len(optimizer.state) != 7
                or any(float(v["step"]) != updates for v in optimizer.state.values())
            ):
                raise ValueError("training finite/ownership/clock gate failed")
            (train_eval, train_logits, train_labels), train_evaluation_seconds = (
                timed_cuda_call(device, evaluate, model, tx, ty)
            )
            (development, logits, labels), development_evaluation_seconds = (
                timed_cuda_call(device, evaluate, model, vx, vy)
            )
            diagnostics, diagnostics_seconds = timed_cuda_call(
                device, layer_diagnostics, model, tx[:256]
            )
            after = snapshot(model, optimizer)
            if not all(
                v["finite"] for v in diagnostics["activations"].values()
            ) or not all(
                math.isfinite(metrics[key])
                for metrics in (train_eval, development)
                for key in ("loss", "accuracy", "roc_auc", "logit_rms")
            ):
                raise FloatingPointError("nonfinite gain pilot activation/evaluation")
            if after.keys() != state.keys() or any(
                not torch.equal(state[k], after[k]) for k in state
            ):
                raise ValueError(
                    "evaluation changed parameter/moment/buffer/clock state"
                )
            if not torch.equal(
                model.fixed_output_gains.cpu(), torch.tensor(cfg["gains"])
            ):
                raise ValueError("fixed gains mutated")
            row = {
                "epoch": epoch,
                "updates": updates,
                "training_seconds": seconds,
                "training_evaluation_seconds": train_evaluation_seconds,
                "development_evaluation_seconds": development_evaluation_seconds,
                "diagnostics_seconds": diagnostics_seconds,
                "train_loss": float(totals[0]) / 100000,
                "train_accuracy": float(totals[1]) / 100000,
                "training_evaluation": train_eval,
                "development": development,
                "order_sha256": tensor_hash(order)["sha256"],
                "state": {k: tensor_hash(v) for k, v in state.items()},
                "gradients": {k: tensor_hash(v) for k, v in gradients.items()},
                "layer_diagnostics": diagnostics,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            raw = stock.checkpoint(model, optimizer, epoch, updates, cfg, development)
            raw["training_evaluation"] = train_eval
            torch.save(raw, directory / f"epoch-{epoch}.pt")
            if best is None or development["loss"] < best["development"]["loss"]:
                torch.save(raw, directory / "best-dev.pt")
                np.savez_compressed(
                    directory / "best-dev-predictions.npz",
                    logits=logits,
                    labels=labels,
                    row_ids=np.arange(1100000, 1150000, dtype=np.int64),
                )
                np.savez_compressed(
                    directory / "best-train-predictions.npz",
                    logits=train_logits,
                    labels=train_labels,
                    row_ids=np.arange(1000000, 1100000, dtype=np.int64),
                )
                best = {
                    "epoch": epoch,
                    "updates": updates,
                    "development": development,
                    "training_evaluation": train_eval,
                    "checkpoint_sha256": file_sha(directory / "best-dev.pt"),
                    "predictions_sha256": file_sha(
                        directory / "best-dev-predictions.npz"
                    ),
                    "training_predictions_sha256": file_sha(
                        directory / "best-train-predictions.npz"
                    ),
                }
                write(directory / "best-dev.json", best)
            rows.append(row)
            write(
                directory / "learning.json",
                {
                    "cohort": COHORT,
                    "configuration": cfg,
                    "initial_development": initial_dev,
                    "initial_layer_diagnostics": initial,
                    "initial_diagnostics_seconds": initial_diagnostics_seconds,
                    "initial_development_evaluation_seconds": initial_development_seconds,
                    "epochs": rows,
                    "best_development_checkpoint": best,
                    "qualification": qualifications[arm],
                    "qualification_sha256": file_sha(
                        args.output / ("qualification-" + arm) / "qualification.json"
                    ),
                    "arm_preparation_seconds": arm_preparation,
                    "common_preparation_seconds_including_calibration_all3qualifications": common_preparation,
                    "runtime_device": fingerprint,
                    "mode": "eager",
                    "TF32": False,
                    "timing_scope": "synchronized gather/forward/loss/backward/stockoptimizer/online train metrics; excludes all evaluation, diagnostics, raw I/O and calibration/qualification/preparation",
                    "memory_scope": "allocator allocated/reserved per-arm cumulative peaks; includes resident data/evaluation/diagnostics, not process memory",
                    "future_confirmation_evaluated": False,
                    "official_test_evaluated": False,
                },
            )
        import shutil

        shutil.copy2(directory / "epoch-3.pt", directory / "final.pt")
        arms[arm] = {
            "learning_sha256": file_sha(directory / "learning.json"),
            "checkpoint_sha256": file_sha(directory / "best-dev.pt"),
            "best": best,
        }
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    write(
        args.output / "pilot.json",
        {
            "status": "exploratory-three-arm-completed",
            "protocol": protocol(),
            "calibration_sha256": file_sha(args.output / "calibration.json"),
            "runtime_device": fingerprint,
            "arms": arms,
            "selection_or_followup_authorized": False,
            "future_confirmation_evaluated": False,
            "official_test_evaluated": False,
        },
    )


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
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
