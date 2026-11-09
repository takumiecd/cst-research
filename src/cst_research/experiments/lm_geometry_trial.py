"""Short, fixed-horizon capacity versus Adam-epsilon diagnostic on RTX 3090."""

from dataclasses import replace

import torch
from torchcst import ChebyshevSpec, CSTLinear, FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.connected_cell_mass_coalesced.envelope import (
    CONNECTED_ORDERED_ENVELOPE,
    ConnectedOrderedEnvelopeAlgorithm,
)
from torchcst.kernels.spec import RadiusConnectionSpec

from .ffn_dst_run import config
from .ffn_model import configure_native, parameter_digest
from .ffn_selective import SITE_SETS, make_model, projections
from .ffn_selective import make_optimizer as base_optimizer

# kind, coefficient, Atom epsilon, profile. Ordinary Adam epsilon is always1e-8.
ARMS = {
    "dense": ("dense", 8, 1e-8, None),
    "dst-c8": ("dst", 8, 1e-8, None),
    "m16-c8-e8": ("cst", 8, 1e-8, "connected-m16"),
    "m16-c8-e10": ("cst", 8, 1e-10, "connected-m16"),
    "m16-c16-e8": ("cst", 16, 1e-8, "connected-m16"),
    "m16-c16-e10": ("cst", 16, 1e-10, "connected-m16"),
    "legacy-m8": ("cst", 8, 1e-8, "legacy-m8"),
    "dst-c16": ("dst", 16, 1e-8, None),
}
STEPS = 976
EVALUATIONS = (0, 200, 256, 400, 600, 800, 976)
SIGNALS = (0, 256, 976)
FIELDS = ("amplitude", "radius_fraction", "output_center", "input_center")


def model_for(arm, device="cuda"):
    kind, coefficient, _, profile = ARMS[arm]
    cfg = replace(config(), coefficient=coefficient)
    model = make_model(cfg, kind, () if kind == "dense" else SITE_SETS["all"])
    initial = parameter_digest(model)
    if profile == "connected-m16":
        for projection in projections(model).values():
            old = projection.map
            spec = replace(
                old.kernel.spec,
                revision=10,
                profiles=tuple(
                    replace(
                        p,
                        profile=ChebyshevSpec(modes=16),
                        connection=RadiusConnectionSpec(),
                    )
                    for p in old.kernel.spec.profiles
                ),
            )
            projection.map = CSTLinear(
                *old.cst_charts(),
                atoms=old.atoms.p.detach().clone(),
                kernel=spec,
                dtype=torch.float32,
                backend="factored",
            )
            projection.map.initialization_report = old.initialization_report
        assert initial == parameter_digest(model)
    model = model.to(device)
    if kind == "cst" and device == "cuda":
        if profile == "legacy-m8":
            configure_native(model)
        else:
            registry = make_algorithm_registry()
            registry.register(ConnectedOrderedEnvelopeAlgorithm())
            for p in projections(model).values():
                p.map.selector = FixedSelector(
                    CONNECTED_ORDERED_ENVELOPE, registry=registry
                )
    model.study_arm = arm
    return model


def optimizer_for(model):
    optimizer = base_optimizer(model)
    atom_ids = {id(m.atoms.p) for m in model.modules() if isinstance(m, CSTLinear)}
    for group in optimizer.param_groups:
        managed = [id(p) in atom_ids for p in group["params"]]
        assert not any(managed) or all(managed), "mixed ordinary/Atom group"
        group["eps"] = ARMS[model.study_arm][2] if any(managed) else 1e-8
    return optimizer


@torch.no_grad()
def geometry(model, optimizer):
    """True bias-corrected moment scales; descriptive, not proposed next updates."""
    from torchcst._backends.torch.parameterizations.radius_envelope import from_state

    rows = {}
    for name, projection in projections(model).items():
        if not isinstance(projection.map, CSTLinear):
            continue
        layer = projection.map
        raw = layer.atoms.p
        physical = from_state(layer.kernel, raw).double()
        active = physical[:, 0].abs() >= 0.01
        radius = physical[:, 1]
        near = ((physical[:, 2:] - physical[:, 2:].round()).abs() < 0.01).all(1)
        state = optimizer.state.get(raw, {})
        row = {
            "A": len(raw),
            "active_atoms": int(active.sum()),
            "radius_mean": float(radius.mean()),
            "radius_quantiles": torch.quantile(
                radius, radius.new_tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            ).tolist(),
            "fraction_mean": float(raw[:, 1].mean()),
            "fraction_at_zero": float((raw[:, 1] == 0).float().mean()),
            "active_sharp_share": float((radius[active] < 0.6).float().mean()),
            "active_both_near_grid_share": float(near[active].float().mean()),
            "active_amplitude_rms": float(physical[active, 0].square().mean().sqrt()),
        }
        if state:
            group = next(
                g for g in optimizer.param_groups if any(p is raw for p in g["params"])
            )
            clock = int(state["step"])
            scale = (
                state["exp_avg_sq"].double() / (1 - group["betas"][1] ** clock)
            ).sqrt()
            row.update(
                actual_atom_epsilon=group["eps"],
                adam_updates=clock,
                active_moment_scale_below_1e8_share=(scale[active] < 1e-8)
                .double()
                .mean(0)
                .tolist(),
                active_moment_scale_below_1e10_share=(scale[active] < 1e-10)
                .double()
                .mean(0)
                .tolist(),
                active_moment_scale_below_used_epsilon_share=(
                    scale[active] < group["eps"]
                )
                .double()
                .mean(0)
                .tolist(),
            )
        rows[name] = row
    return rows
