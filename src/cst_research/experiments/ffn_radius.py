"""Independent physical radius for the fixed FFN experiment; no learner loop."""

from __future__ import annotations

import copy
import hashlib

import torch
from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels
from torchcst._backends.torch.parameterizations.radius_cell_mass import decoded_values


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def freeze_radius_gradient(gradient):
    result = gradient.clone()
    result[:, 1] = 0
    return result


def convert_layer(source, mode):
    """Copy the SAME realized Polar atom to (a,r,mu_out,mu_in), rounded to FP32."""
    if mode not in ("learned", "fixed"):
        raise ValueError("physical mode must be learned or fixed")
    old = source.atoms.p.detach()
    amplitude, ri, ro = decoded_values(source.kernel, old)
    torch.testing.assert_close(ri, ro, atol=0, rtol=0)
    physical = torch.stack((amplitude, ri, old[:, 2], old[:, 3]), dim=1).to(old)
    maximum = min(50.0, source.in_features / 2, source.out_features / 2)
    result = CSTLinear(
        chart_presets.regular_grid(
            source.out_features, spacing=1.0, geometry="flat_torus"
        ),
        chart_presets.regular_grid(
            source.in_features, spacing=1.0, geometry="flat_torus"
        ),
        atoms=physical,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
            coordinates=kernels.physical(radius_init=(0.5, maximum), radius_floor=0.5),
        ),
        dtype=old.dtype,
        backend="factored",
    ).to(old.device)
    result.independent_radius_mode = mode
    result.physical_radius_maximum = maximum
    result.physical_amplitude_maximum = float(source.kernel.scalar("amplitude_max"))
    result.initialization_report = {
        **copy.deepcopy(source.initialization_report),
        "independent_radius": mode,
        "source_p_sha256": digest(old),
        "decoded_physical_p_sha256": digest(physical),
        "conversion": "Same realized Polar amplitude/radius/centers, rounded once to Parameter dtype",
    }
    if mode == "fixed":
        result.atoms.p.register_hook(freeze_radius_gradient)
    return result


def convert_ffn(model):
    for block in model.blocks:
        for name in ("up", "down"):
            projection = getattr(block.ffn, name)
            projection.map = convert_layer(projection.map, model.config.radius_mode)


class PhysicalRadiusAdamW(torch.optim.AdamW):
    """Ordinary AdamW followed by projection; forward has no radius clamp."""

    def __init__(self, parameters, *, layers, **options):
        if options.get("weight_decay", 0) != 0:
            raise ValueError("fixed radius experiment requires zero weight decay")
        super().__init__(parameters, **options)
        self.radius_layers = layers

    def step(self, closure=None):
        result = super().step(closure)
        with torch.no_grad():
            for layer in self.radius_layers:
                p = layer.atoms.p
                p[:, 0].clamp_(
                    -layer.physical_amplitude_maximum, layer.physical_amplitude_maximum
                )
                p[:, 1].clamp_(0.5, layer.physical_radius_maximum)
        return result


def physical_optimizer(model, *, atom_lr_multiplier, **options):
    layers = [m for m in model.modules() if isinstance(m, CSTLinear)]
    ids = {id(m.atoms.p) for m in layers}
    groups = [
        {
            "params": [p for p in model.parameters() if id(p) not in ids],
            "lr_scale": 1.0,
        },
        {
            "params": [m.atoms.p for m in layers],
            "lr": options["lr"] * atom_lr_multiplier,
            "lr_scale": atom_lr_multiplier,
        },
    ]
    return PhysicalRadiusAdamW(groups, layers=layers, **options)


def stats(tensor):
    value = tensor.detach().double().flatten()
    return {
        "mean": float(value.mean()),
        "rms": float(value.square().mean().sqrt()),
        "p10": float(torch.quantile(value, 0.1)),
        "p50": float(torch.quantile(value, 0.5)),
        "p90": float(torch.quantile(value, 0.9)),
        "max_abs": float(value.abs().max()),
    }


@torch.no_grad()
def radius_snapshot(model, *, gradients=False):
    result = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, CSTLinear):
            continue
        a, ri, ro = decoded_values(layer.kernel, layer.atoms.p)
        row = {
            "amplitude": stats(a),
            "input_radius": stats(ri),
            "output_radius": stats(ro),
        }
        row["at_lower_fraction"] = float((ri <= 0.5).double().mean())
        row["at_upper_fraction"] = float(
            (ri >= min(50, layer.in_features / 2, layer.out_features / 2))
            .double()
            .mean()
        )
        if gradients and layer.atoms.p.grad is not None:
            g = layer.atoms.p.grad
            if not bool(torch.isfinite(g).all()):
                raise FloatingPointError("nonfinite atom gradient")
            row["field_gradient_rms"] = g.double().square().mean(0).sqrt().tolist()
            row["field_gradient_max_abs"] = g.abs().amax(0).tolist()
            row["radius_task_gradient"] = (
                stats(g[:, 1]) if model.config.radius_mode != "polar" else None
            )
        row["radius_mode"] = model.config.radius_mode
        result[name] = row
    return result


def probe_radius(model, x, targets):
    """Fixed data task gradients, after the update and outside its timer."""
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss = model(x, targets)
    loss.backward()
    result = {
        "loss": float(loss.detach()),
        "layers": radius_snapshot(model, gradients=True),
        "scope": "Fixed probe, unscaled/unclipped loss gradient; Polar radius is detached; fixed mode masks radius gradient",
    }
    model.zero_grad(set_to_none=True)
    return result
