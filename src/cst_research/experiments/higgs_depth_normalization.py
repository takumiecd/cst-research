"""Eight-step normalization diagnostic; separate consumer, unchanged public975 runtime."""

from __future__ import annotations

import copy

from . import higgs_depth_gain as gain
from . import higgs_depth_study as stock
from .higgs_depth_data import DATA_SHA, MANIFEST_SHA
from .higgs_dst_comparison import json_sha

COHORT = "higgs-depth-normalization-diagnostic-v1"
EPS = 1e-5
CASES = {
    "dense-stock": ("dense", None, "none", False),
    "dense-rms": ("dense", None, "rms", True),
    "dense-ln": ("dense", None, "ln", True),
    "cst104-stock": ("all", 10000, "none", False),
    "cst104-gain": ("all", 10000, "none", True),
    "cst104-postgain": ("all", 10000, "none", True),
    "cst104-rms": ("all", 10000, "rms", True),
    "cst104-ln": ("all", 10000, "ln", True),
    "cst1048-stock": ("all", 1000, "none", False),
}
SITES = (2, 4, 6)
SNAPSHOTS = (0, 1, 2, 4, 8)


def protocol():
    return {
        "version": COHORT,
        "external_source_commit": stock.SOURCE,
        "architecture": [28, 1024, 1024, 1024, 1024, 2],
        "cases": list(CASES),
        "preexecution_amendment": "Ninth cst104-postgain added before any GPU/task result following the user's post-output-scale hypothesis; unexecuted eight-case prototype preserved separately",
        "calibration_target_by_case": {
            case: ("post-ReLU" if case == "cst104-postgain" else "pre-ReLU")
            for case in CASES
        },
        "square_sites": list(SITES),
        "square_bias": False,
        "all_layers_trainable": True,
        "seed": 17,
        "batch_size": 4,
        "training_updates": 8,
        "first_epoch_order_seed": 117,
        "state_steps": list(SNAPSHOTS),
        "lr": 0.0003,
        "position_multiplier": 1,
        "mode": "eager",
        "dtype": "FP32",
        "TF32": False,
        "normalization": {
            "axis": "1024 features per example",
            "eps": EPS,
            "affine": False,
            "location": "after each square before ReLU",
            "rms": "z/sqrt(mean(z^2)+eps)",
            "ln": "(z-mean(z))/sqrt(mean((z-mean(z))^2)+eps)",
        },
        "calibration": {
            "feature_rows": 256,
            "labels_used": False,
            "target": "Dense-stock initial pre-ReLU RMS except explicit cst104-postgain post-ReLU target",
            "algorithm": "sequential CPU FP64 public dense_weight, including preceding normalization/gains",
            "storage": "positive nontrainable FP32 post-normalization gains",
            "gain_bounds": list(gain.GAIN_BOUNDS),
            "zero_nonfinite_or_out_of_bounds": "fail closed; no clip, eps sweep or replacement",
        },
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "development_evaluated": False,
        "future_confirmation_evaluated": False,
        "official_test_evaluated": False,
        "HIGGS_confirmation_evaluated": False,
        "selection_or_HPO": False,
        "automatic_followup": False,
        "formal_claim": False,
        "qualification": "independent CPU FP64 full-network Y/dX/all-grad; inherited normal elementwise and relative norm/packed-column criteria; 8step and4+resume+4 exact states",
        "amplified_componentwise": "diagnostic only; prior failed componentwise stress is not reset or promoted",
        "scope": "post-observation mechanism diagnostic,8 actual task minibatches only; no epoch-quality inference",
    }


def norm(value, kind, *, reference=False):
    if kind == "none":
        return value
    if kind == "rms":
        if reference:
            return value / (value.square().mean(-1, keepdim=True) + EPS).sqrt()
        import torch.nn.functional as F

        return F.rms_norm(value, (value.shape[-1],), weight=None, eps=EPS)
    if kind != "ln":
        raise ValueError("undeclared normalization")
    if reference:
        centered = value - value.mean(-1, keepdim=True)
        return centered / (centered.square().mean(-1, keepdim=True) + EPS).sqrt()
    import torch.nn.functional as F

    return F.layer_norm(value, (value.shape[-1],), weight=None, bias=None, eps=EPS)


def make_model(case, *, width=1024, gains=None):
    import torch
    from torch import nn

    if case not in CASES:
        raise ValueError("undeclared diagnostic case")
    kind, divisor, normalization, _ = CASES[case]
    base = stock.make_model(kind, divisor, 17, width)

    class DiagnosticSequential(nn.Sequential):
        def forward(self, value):
            return trace(self, value)[-1]

    model = DiagnosticSequential(*list(base.children()))
    model.register_buffer("fixed_output_gains", torch.ones(3, dtype=torch.float32))
    model._diagnostic_case = case
    model._normalization = normalization
    model._depth_configuration = base._depth_configuration
    model._cst_research_cohort = COHORT
    if gains is not None:
        install_gains(model, gains)
    return model


def install_gains(model, values):
    import torch

    values = torch.as_tensor(values, dtype=torch.float32)
    if (
        values.shape != (3,)
        or not bool(torch.isfinite(values).all())
        or not bool(
            ((values >= gain.GAIN_BOUNDS[0]) & (values <= gain.GAIN_BOUNDS[1])).all()
        )
    ):
        raise gain.CalibrationInconclusive("invalid fixed normalization gains")
    if not CASES[model._diagnostic_case][3] and not bool((values == 1).all()):
        raise ValueError("stock controls require unit gains")
    with torch.no_grad():
        model.fixed_output_gains.copy_(values.to(model.fixed_output_gains.device))


def trace(model, value, *, reference=False, detailed=None):
    from torch import nn
    from torchcst.nn import CSTModule

    values = []
    for i, layer in enumerate(model):
        value = (
            nn.functional.linear(value, layer.dense_weight())
            if reference and isinstance(layer, CSTModule)
            else layer(value)
        )
        if i in SITES:
            if detailed is not None:
                detailed[f"{i}:raw-square"] = value
            value = norm(value, model._normalization, reference=reference)
            if detailed is not None:
                detailed[f"{i}:post-norm"] = value
            value = value * model.fixed_output_gains[SITES.index(i)]
        values.append(value)
        if detailed is not None:
            detailed[str(i)] = value
    return values


def reference_forward(model, value):
    return trace(model, value, reference=True)[-1]


def calibrate(dense, model, features):
    import torch
    from torch import nn
    from torchcst.nn import CSTModule

    if (
        features.shape != (256, 28)
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError(
            "calibration requires fixed256 finite CPU FP32 training feature rows"
        )
    if dense._diagnostic_case != "dense-stock" or any(
        p.grad is not None for m in (dense, model) for p in m.parameters()
    ):
        raise ValueError("calibration precedes any task gradient/update")
    if not bool((model.fixed_output_gains == 1).all()):
        raise ValueError("calibration may run once from unit gains only")
    before = gain.parameter_hashes(model)
    target_model = copy.deepcopy(dense).double()
    candidate = copy.deepcopy(model).double()
    rows, gains = [], []
    with torch.no_grad():
        target_trace = trace(target_model, features.double(), reference=True)
        value = features.double()
        for i, layer in enumerate(candidate):
            value = (
                nn.functional.linear(value, layer.dense_weight())
                if isinstance(layer, CSTModule)
                else layer(value)
            )
            if i in SITES:
                value = norm(value, model._normalization, reference=True)
                post_target = model._diagnostic_case == "cst104-postgain"
                signal = value.relu() if post_target else value
                observed = float(signal.square().mean().sqrt())
                target_index = i + 1 if post_target else i
                target = float(target_trace[target_index].square().mean().sqrt())
                multiplier = (
                    gain.bounded_gain(target, observed)
                    if CASES[model._diagnostic_case][3]
                    else 1.0
                )
                value = value * multiplier
                gains.append(multiplier)
                rows.append(
                    {
                        "site": i,
                        "target_dense_RMS": target,
                        "target_kind": "post-ReLU" if post_target else "pre-ReLU",
                        "target_site": target_index,
                        "matching_signal_RMS_before_gain": observed,
                        "fixed_FP32_gain": multiplier,
                        "post_gain_RMS": float(
                            (value.relu() if post_target else value)
                            .square()
                            .mean()
                            .sqrt()
                        ),
                        "post_gain_pre_relu_RMS": float(value.square().mean().sqrt()),
                        "post_gain_post_relu_RMS": float(
                            value.relu().square().mean().sqrt()
                        ),
                    }
                )
    install_gains(model, gains)
    if gain.parameter_hashes(model) != before:
        raise ValueError("calibration changed trainable tensors")
    return {
        "case": model._diagnostic_case,
        "status": "calibrated"
        if CASES[model._diagnostic_case][3]
        else "unit-stock-control",
        "rows": rows,
        "gains": gains,
        "feature_rows": 256,
        "features_sha256": gain.tensor_sha(features),
        "labels_used": False,
        "trainable_hashes": before,
        "protocol_sha256": json_sha(protocol()),
    }
