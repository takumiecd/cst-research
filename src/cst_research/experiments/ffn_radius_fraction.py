"""Public radius-envelope FFN consumer; no duplicate map or Linear runtime."""

from dataclasses import asdict
from itertools import pairwise

import torch
from torchcst import (
    ChebyshevSpec,
    CSTAdamW,
    CSTLinear,
    FixedSelector,
    chart_presets,
    init,
    kernels,
    make_algorithm_registry,
)

from .ffn_contract import atom_count
from .ffn_radius import stats


def make_layer(ni, no, *, config, std, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer = CSTLinear(
            chart_presets.regular_grid(no, spacing=1.0, geometry="flat_torus"),
            chart_presets.regular_grid(ni, spacing=1.0, geometry="flat_torus"),
            atoms=atom_count(ni, no, config.coefficient),
            dtype=torch.float32,
            backend="factored",
            kernel=kernels.product(
                kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
                coordinates=kernels.radius_envelope(
                    radius_min=0.5,
                    radius_max=50.0,
                    tau=config.tau,
                    radius_fraction_init=config.radius_fraction,
                    amplitude_max=config.amplitude_max,
                ),
            ),
        )
    layer.initialization_report = asdict(
        init.variance_scaling_(
            layer,
            variance=std**2,
            radius_fraction=config.radius_fraction,
            generator=torch.Generator().manual_seed(seed + 1),
        )
    )
    return layer


def configure_native(model):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.radius_envelope import (
        RADIUS_ENVELOPE_CACHED_LM,
        RadiusEnvelopeCachedLMAlgorithm,
    )

    registry = make_algorithm_registry()
    registry.register(RadiusEnvelopeCachedLMAlgorithm())
    for layer in model.modules():
        if isinstance(layer, CSTLinear):
            layer.selector = FixedSelector(RADIUS_ENVELOPE_CACHED_LM, registry=registry)


def optimizer(model, *, atom_lr_multiplier, **options):
    layers = [m for m in model.modules() if isinstance(m, CSTLinear)]
    atom_ids = {id(m.atoms.p) for m in layers}
    groups = [
        {
            "params": [p for p in model.parameters() if id(p) not in atom_ids],
            "lr_scale": 1.0,
        },
        {
            "params": [m.atoms.p for m in layers],
            "lr": options["lr"] * atom_lr_multiplier,
            "lr_scale": atom_lr_multiplier,
            "field_lrs": {
                "radius_fraction": options["lr"]
                * model.config.radius_fraction_lr_multiplier
            },
        },
    ]
    return CSTAdamW(groups, model=model, **options)


@torch.no_grad()
def snapshot(model, *, gradients=False):
    from torchcst._backends.torch.parameterizations.radius_envelope import from_state

    from .ffn_envelope import EDGES

    result = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, CSTLinear):
            continue
        # Decode in the Parameter dtype before moving to CPU (floor branches).
        physical = from_state(layer.kernel, layer.atoms.p).cpu().double()
        a, r = physical[:, 0], physical[:, 1]
        f = layer.atoms.p[:, 1].detach().cpu().double()
        lo = 0.5 + 49.5 * torch.exp(-(a / model.config.tau).square())

        def row(mask, a=a, r=r, f=f, lo=lo):
            if not bool(mask.any()):
                return {"count": 0}
            return {
                "count": int(mask.sum()),
                "absolute_amplitude": stats(a[mask].abs()),
                "radius": stats(r[mask]),
                "radius_fraction": stats(f[mask]),
                "lower": stats(lo[mask]),
                "fraction_at_zero": float((f[mask] == 0).double().mean()),
                "fraction_at_one": float((f[mask] == 1).double().mean()),
            }

        entry = {
            "all": row(torch.ones(len(a), dtype=torch.bool)),
            "bins": [
                {"low": low, "high": high, **row((a.abs() >= low) & (a.abs() < high))}
                for low, high in pairwise(EDGES)
            ],
        }
        assert sum(b["count"] for b in entry["bins"]) == len(a)
        if gradients and layer.atoms.p.grad is not None:
            g = layer.atoms.p.grad
            if not bool(torch.isfinite(g).all()):
                raise FloatingPointError("nonfinite envelope gradient")
            entry["field_names"] = [
                "amplitude",
                "radius_fraction",
                "output_center",
                "input_center",
            ]
            entry["field_gradient_rms"] = g.double().square().mean(0).sqrt().tolist()
            entry["field_gradient_max_abs"] = g.abs().amax(0).tolist()
        result[name] = entry
    return result
