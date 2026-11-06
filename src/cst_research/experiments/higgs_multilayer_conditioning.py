"""Six paired H8 NONE/LN/fixed-gain models; public975 numerical runtime unchanged."""

from __future__ import annotations

import copy

from . import higgs_depth_gain as gain_policy
from . import higgs_depth_normalization as normalization
from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_linear_atom_scaling as factory

COHORT = "higgs-multilayer-conditioning-v1"
EPS = 1e-5
WIDTH, HIDDEN, ATOMS, SEED = 1024, 8, 4096, 17
CASES = {
    "dense-none": ("dense", "none", False),
    "dense-bare-ln": ("dense", "ln", False),
    "cst-none": ("cst", "none", False),
    "cst-fixed-gain": ("cst", "none", True),
    "cst-bare-ln": ("cst", "ln", False),
    "cst-ln-gain": ("cst", "ln", True),
}
MILESTONES = (0, 1000, 5000, 10000, 15000, 20000, 25000)


def configuration(case, *, width=WIDTH, hidden=HIDDEN):
    if case not in CASES:
        raise ValueError("undeclared six-case conditioning arm")
    method, norm, calibrated = CASES[case]
    cfg = factory.configuration(
        method=method,
        width=width,
        hidden_layers=hidden,
        c=4 if method == "cst" else None,
        normalization=norm,
        ordinary_initialization=factory.ORDINARY_INIT,
    )
    return {
        "cohort": COHORT,
        "case": case,
        "method": method,
        "normalization": norm,
        "epsilon": EPS,
        "calibrated": calibrated,
        "factory_configuration": cfg,
        "calibration_target": "same Dense-stock initial pre-ReLU RMS, sequential fixed256 TRAIN features; no labels or DEV"
        if calibrated
        else "unit gains",
        "seed": SEED,
        "trainable_affine_in_LN": False,
    }


def make_model(case, *, gains=None, width=WIDTH, hidden=HIDDEN):
    cfg = configuration(case, width=width, hidden=hidden)
    if normalization.EPS != EPS:
        raise ValueError("inherited no-affine LayerNorm epsilon changed")
    model = factory.make_model(
        method=cfg["method"],
        width=width,
        hidden_layers=hidden,
        c=4 if cfg["method"] == "cst" else None,
        normalization=cfg["normalization"],
        ordinary_initialization=factory.ORDINARY_INIT,
        seed=SEED,
    )
    model._conditioning_configuration = cfg
    model._conditioning_case = case
    model._cst_research_cohort = COHORT
    if gains is not None:
        install_gains(model, gains)
    return model


def install_gains(model, values):
    import torch

    values = torch.as_tensor(values, dtype=torch.float32)
    exact.require(
        values.shape == model.fixed_output_gains.shape
        and bool(torch.isfinite(values).all())
        and bool(((values >= 1e-6) & (values <= 1e6)).all()),
        "finite positive FP32 gains in [1e-6,1e6] required; no clamp",
    )
    exact.require(
        model._conditioning_configuration["calibrated"] or bool((values == 1).all()),
        "bare/none controls require unit gains",
    )
    with torch.no_grad():
        model.fixed_output_gains.copy_(values.to(model.fixed_output_gains.device))


def trace(model, value, *, detailed=None, retain_grad=False):
    """Same runtime expressions as the unchanged factory; no runtime/backward hooks."""
    sites = model._linear_atom_configuration["internal_module_indices"]
    result = []
    for index, layer in enumerate(model):
        value = layer(value)
        if index in sites:
            if detailed is not None:
                detailed[f"{index}:raw-square"] = value
            value = normalization.norm(
                value, model._conditioning_configuration["normalization"]
            )
            if detailed is not None:
                detailed[f"{index}:post-norm"] = value
            value = value * model.fixed_output_gains[sites.index(index)]
        result.append(value)
        if detailed is not None:
            detailed[str(index)] = value
    if retain_grad and detailed is not None:
        for value in {id(v): v for v in detailed.values()}.values():
            if value.requires_grad:
                value.retain_grad()
    return result


def calibrate(dense, candidate, features):
    """No labels/DEV: sequential FP64 public chunk reference, trainables unchanged."""
    import torch

    from . import higgs_multilayer_conditioning_reference as reference

    exact.require(
        features.dtype == torch.float32
        and features.device.type == "cpu"
        and features.shape == (256, 28)
        and bool(torch.isfinite(features).all()),
        "fixed256 finite CPU FP32 TRAIN features required",
    )
    exact.require(
        dense._conditioning_case == "dense-none"
        and dense._linear_atom_configuration["width"]
        == candidate._linear_atom_configuration["width"]
        and dense._linear_atom_configuration["hidden_layers"]
        == candidate._linear_atom_configuration["hidden_layers"],
        "paired Dense-stock target topology required",
    )
    exact.require(
        all(p.grad is None for model in (dense, candidate) for p in model.parameters())
        and bool((candidate.fixed_output_gains == 1).all()),
        "calibrate once before any backward from unit gains",
    )
    before = gain_policy.parameter_hashes(candidate)
    target_model, model = (
        copy.deepcopy(dense).double(),
        copy.deepcopy(candidate).double(),
    )
    with torch.no_grad():
        targets = reference.forward(target_model, features.double(), collect=True)[1]
        value = features.double()
        rows, gains = [], []
        sites = model._linear_atom_configuration["internal_module_indices"]
        for index, layer in enumerate(model):
            value = reference.affine_or_relu(layer, value)
            if index in sites:
                value, _ = reference.normalize(
                    value, model._conditioning_configuration["normalization"]
                )
                target = float(targets[str(index)].square().mean().sqrt())
                observed = float(value.square().mean().sqrt())
                multiplier = (
                    gain_policy.bounded_gain(target, observed)
                    if model._conditioning_configuration["calibrated"]
                    else 1.0
                )
                value = value * multiplier
                gains.append(multiplier)
                rows.append(
                    {
                        "site": index,
                        "target_Dense_stock_pre_ReLU_RMS": target,
                        "observed_post_norm_before_gain_RMS": observed,
                        "fixed_FP32_gain": multiplier,
                        "post_gain_pre_ReLU_RMS": float(value.square().mean().sqrt()),
                        "post_gain_post_ReLU_RMS": float(
                            value.relu().square().mean().sqrt()
                        ),
                    }
                )
    install_gains(candidate, gains)
    exact.require(
        before == gain_policy.parameter_hashes(candidate),
        "calibration changed raw trainable P/ordinary tensors",
    )
    return {
        "case": candidate._conditioning_case,
        "status": "calibrated"
        if candidate._conditioning_configuration["calibrated"]
        else "unit-control",
        "gains": gains,
        "rows": rows,
        "target_case": "dense-none",
        "target_kind": "pre-ReLU",
        "features_rows": [1000000, 1000256],
        "labels_or_development_used": False,
        "parameters_unchanged": True,
    }
