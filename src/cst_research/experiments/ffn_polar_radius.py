"""Research composition Phi(s,t,mu) -> existing physical Linear; no new CUDA math."""

from __future__ import annotations

import torch
from torch import nn
from torchcst import CSTLinear
from torchcst._backends.torch.parameterizations.polar_amp_width import _width_bounds
from torchcst._backends.torch.parameterizations.radius_cell_mass import decoded_values

from .ffn_radius import convert_layer, stats

MODES = ("polar_detached", "polar_live")


def mapped_parameters(layer, p, *, live):
    """Same forward envelope, with inward radial slopes at the feasible endpoints.

    Feasible domain is 1<=norm(s,t)<=2 (FP32 projection tolerance5e-7).
    The value clamps are exact; their surrogate derivative is explicitly the
    interior one-sided derivative, including q0/q1. Outside this domain this
    is not an unconstrained derivative contract.
    """
    z = p[:, :2].double()
    square = z.square().sum(1)
    a = float(layer.kernel.scalar("amplitude_max")) * z[:, 0] / square.sqrt()
    raw_q = (square - 1) / 3
    q_value = raw_q.clamp(0, 1)
    q = raw_q + (q_value - raw_q).detach()
    value, lower, upper = _width_bounds(
        layer.kernel,
        a,
        q_value,
        side="input",
        scalar_names=layer.kernel.spec.parameterization.width_scalar_names,
    )
    expression = lower * torch.exp(q * torch.log(upper / lower))
    radius = expression + (value - expression).detach()
    if not live:
        radius = radius.detach()
    return torch.cat((a[:, None], radius[:, None], p[:, 2:].double()), 1).to(p)


class PolarRadiusProjection(nn.Module):
    """Own original Polar atoms; execute physical declaration on mapped Tensors."""

    def __init__(self, latent, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError("unknown Polar radius experiment mode")
        self.latent = latent
        self.mode = mode
        prototype = convert_layer(latent, "learned")
        self.physical_spec = prototype.execution_declaration()
        self._dispatcher = None
        self.track_probe = False
        self._probe_physical = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_dispatcher=None, _probe_physical=None, track_probe=False)
        return state

    @property
    def atoms(self):
        return self.latent.atoms

    @property
    def in_features(self):
        return self.latent.in_features

    @property
    def out_features(self):
        return self.latent.out_features

    def physical_parameters(self, p=None):
        return mapped_parameters(
            self.latent,
            self.atoms.p if p is None else p,
            live=self.mode == "polar_live",
        )

    def _physical_factors(self, p):
        # CPU oracle only: a one-Atom public Module supplies fixed physical
        # Chart/Kernel state. Its unused Parameter is never owned by the LM.
        from torchcst import ChebyshevSpec, chart_presets, kernels

        helper = CSTLinear(
            chart_presets.regular_grid(
                self.out_features, geometry="flat_torus", spacing=1.0
            ),
            chart_presets.regular_grid(
                self.in_features, geometry="flat_torus", spacing=1.0
            ),
            atoms=1,
            dtype=p.dtype,
            backend="factored",
            kernel=kernels.product(
                kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
                coordinates=kernels.physical(
                    radius_init=(
                        0.5,
                        min(50.0, self.in_features / 2, self.out_features / 2),
                    ),
                    radius_floor=0.5,
                ),
            ),
        ).to(p.device)
        return helper._factor_atoms(p)

    def _factor_atoms(self, p):
        return self._physical_factors(self.physical_parameters(p))

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError("unexpected projection width")
        physical = self.physical_parameters()
        if self.track_probe and torch.is_grad_enabled():
            physical.retain_grad()
            self._probe_physical = physical
        flat = x.reshape(-1, self.in_features).contiguous()
        if x.is_cuda:
            from torchcst import (
                Dispatcher,
                LinearBinding,
                LinearInputs,
                make_algorithm_registry,
            )
            from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_multirow import (
                LM_PHYSICAL_LEVELS,
                RadiusLMPhysicalLevelsAlgorithm,
            )

            if self._dispatcher is None:
                registry = make_algorithm_registry()
                registry.register(RadiusLMPhysicalLevelsAlgorithm())
                self._dispatcher = Dispatcher(registry=registry)
            binding = LinearBinding(self.physical_spec, physical)
            y = self._dispatcher.run(
                binding, LinearInputs(flat), plan=LM_PHYSICAL_LEVELS
            )
        else:
            v, u = self._physical_factors(physical)
            y = (flat @ v) @ u.T
        return y.reshape(*x.shape[:-1], self.out_features)


def projections(model):
    return [
        (n, m) for n, m in model.named_modules() if isinstance(m, PolarRadiusProjection)
    ]


def convert_ffn(model):
    for block in model.blocks:
        for name in ("up", "down"):
            projection = getattr(block.ffn, name)
            projection.map = PolarRadiusProjection(
                projection.map, model.config.radius_mode
            )


class ProjectedPolarAdamW(torch.optim.AdamW):
    """Cartesian Adam moments, post-step annulus projection; no tangent removal."""

    def __init__(self, parameters, *, layers, **options):
        if options.get("weight_decay", 0) != 0:
            raise ValueError("pilot requires zero weight decay")
        super().__init__(parameters, **options)
        self.polar_layers = layers

    def step(self, closure=None):
        result = super().step(closure)
        with torch.no_grad():
            for layer in self.polar_layers:
                p = layer.atoms.p
                norm = p[:, :2].double().norm(dim=1, keepdim=True)
                if not bool((torch.isfinite(norm) & (norm > 0)).all()):
                    raise FloatingPointError(
                        "Polar annulus proposal reached singular origin/nonfinite state"
                    )
                p[:, :2].mul_((norm.clamp(1, 2) / norm).to(p))
        return result


def optimizer(model, *, atom_lr_multiplier, **options):
    layers = [m for _, m in projections(model)]
    if not layers:
        raise ValueError("projected Polar requires experiment projections")
    ids = {id(m.atoms.p) for m in layers}
    return ProjectedPolarAdamW(
        [
            {
                "params": [p for p in model.parameters() if id(p) not in ids],
                "lr_scale": 1.0,
            },
            {
                "params": [m.atoms.p for m in layers],
                "lr": options["lr"] * atom_lr_multiplier,
                "lr_scale": atom_lr_multiplier,
            },
        ],
        layers=layers,
        **options,
    )


@torch.no_grad()
def snapshot(model, *, gradients=False):
    result = {}
    for name, layer in projections(model):
        p = layer.atoms.p
        a, ri, ro = decoded_values(layer.latent.kernel, p)
        norm = p[:, :2].double().norm(dim=1)
        q = ((norm.square() - 1) / 3).clamp(0, 1)
        _, lo, hi = _width_bounds(
            layer.latent.kernel,
            a,
            q,
            side="input",
            scalar_names=layer.latent.kernel.spec.parameterization.width_scalar_names,
        )
        if not bool(((norm >= 1 - 5e-7) & (norm <= 2 + 5e-7)).all()):
            raise FloatingPointError("latent norm outside projected domain")
        row = {
            "amplitude": stats(a),
            "input_radius": stats(ri),
            "output_radius": stats(ro),
            "q": stats(q),
            "latent_norm": stats(norm),
            "lower": stats(lo),
            "upper": stats(hi),
            "q_zero_fraction": float((q <= 0).double().mean()),
            "q_one_fraction": float((q >= 1).double().mean()),
            "at_lower_fraction": float((ri <= 0.5).double().mean()),
            "at_upper_fraction": float((ri >= 50).double().mean()),
            "radius_mode": layer.mode,
        }
        if gradients:
            g = p.grad
            partial = layer._probe_physical.grad
            if not bool(torch.isfinite(g).all() & torch.isfinite(partial).all()):
                raise FloatingPointError("nonfinite mapped gradient")
            radial = (g[:, :2].double() * p[:, :2].double() / norm[:, None]).sum(1)
            row.update(
                field_gradient_rms=g.double().square().mean(0).sqrt().tolist(),
                physical_radius_partial=stats(partial[:, 1]),
                radius_task_gradient=stats(
                    partial[:, 1]
                    if layer.mode == "polar_live"
                    else torch.zeros_like(partial[:, 1])
                ),
                radial_latent_gradient=stats(radial),
            )
        result[name] = row
    return result
