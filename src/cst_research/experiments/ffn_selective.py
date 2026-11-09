"""Research-only selective FFN replacement; public operator math is unchanged."""

import math

import torch
from torchcst import CSTLinear

from . import ffn_dst
from .ffn_contract import atom_count
from .ffn_model import LanguageModel, Projection, cst_optimizer, parameter_seed

# Block numbers are zero based. Keep network depth and all other sites fixed.
SITE_SETS = {
    "up": ("1.up",),
    "down": ("1.down",),
    "block": ("1.up", "1.down"),
    "two_blocks": ("1.up", "1.down", "2.up", "2.down"),
    "all": tuple(f"{i}.{d}" for i in range(4) for d in ("up", "down")),
    "scattered_up": tuple(f"{i}.up" for i in range(4)),
    "scattered_down": tuple(f"{i}.down" for i in range(4)),
    "scattered_mixed": ("0.up", "1.down", "2.up", "3.down"),
}


def validate_sites(config, kind, sites):
    sites = tuple(sites)
    allowed = {f"{i}.{d}" for i in range(config.layers) for d in ("up", "down")}
    if kind not in ("dense", "cst", "dst") or len(set(sites)) != len(sites):
        raise ValueError("explicit model kind and unique sites required")
    if set(sites) - allowed or (kind == "dense") != (not sites):
        raise ValueError("Dense uses no replacements; CST/DST require valid sites")
    if config.attention_kind != "dense" or config.radius_mode != "envelope":
        raise ValueError("this study requires Dense attention and radius-envelope FFN")
    return sites


def make_model(config, kind, sites, seed=17):
    sites = validate_sites(config, kind, sites)
    model = LanguageModel(config, "dense", seed)
    for site in sites:
        block, direction = site.split(".")
        i = int(block)
        projection = getattr(model.blocks[i].ffn, direction)
        old = projection.map
        if kind == "cst":
            projection.map = Projection(
                old.in_features,
                old.out_features,
                config=config,
                kind="cst",
                std=config.init_std
                / (math.sqrt(2 * config.layers) if direction == "down" else 1),
                seed=parameter_seed(seed, f"block{i}.{direction}"),
            ).map
        else:
            projection.map = ffn_dst.MaskedLinear(
                old.weight,
                4 * atom_count(old.in_features, old.out_features, config.coefficient),
                torch.Generator().manual_seed(
                    parameter_seed(seed, f"block{i}.{direction}.dst-mask")
                ),
            )
    model.kind = kind
    model.ffn_selected_sites = sites
    return model


def active_ffn_scalars(model):
    stored = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
    return stored - sum(
        layer.weight.numel() - int(layer.mask.sum())
        for layer in ffn_dst.layers(model).values()
    )


def make_optimizer(model):
    if model.kind == "cst":
        return cst_optimizer(
            model,
            atom_lr_multiplier=model.config.atom_lr_multiplier,
            lr=0.0003,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            foreach=False,
        )
    if model.kind == "dst":
        return ffn_dst.make_optimizer(model, 0.003)
    return torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )


def projections(model):
    return {
        f"{i}.{d}": getattr(b.ffn, d)
        for i, b in enumerate(model.blocks)
        for d in ("up", "down")
    }


def rms(x):
    return float(x.detach().double().square().mean().sqrt())


def mechanism_probe(model, x, y):
    """Fixed TRAIN probe, outside timing: activations and unscaled cotangents."""
    model.train()
    model.zero_grad(set_to_none=True)
    values, handles = {}, []
    modules = {
        **{f"block{i}": b for i, b in enumerate(model.blocks)},
        **projections(model),
    }
    try:
        for name, module in modules.items():

            def capture(_module, inputs, output, name=name):
                row = {"input_rms": rms(inputs[0]), "output_rms": rms(output)}
                values[name] = row

                def gradient(g):
                    row["output_gradient_rms"] = rms(g)

                output.register_hook(gradient)

            handles.append(module.register_forward_hook(capture))
        _, loss = model(x, y)
        loss.backward()
        fields = {}
        for name, p in model.named_parameters():
            if ".ffn." not in name or p.grad is None:
                continue
            assert torch.isfinite(p.grad).all()
            row = {"parameter_rms": rms(p), "gradient_rms": rms(p.grad)}
            if name.endswith("atoms.p"):
                row["field_parameter_rms"] = (
                    p.detach().double().square().mean(0).sqrt().tolist()
                )
                row["field_gradient_rms"] = (
                    p.grad.detach().double().square().mean(0).sqrt().tolist()
                )
            fields[name] = row
        return {
            "loss": float(loss.detach()),
            "layers": values,
            "parameters": fields,
            "scope": "Fixed TRAIN prefix, unscaled/unclipped cotangents; different parameterizations' raw gradient magnitudes are not equivalent update scales",
        }
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def before_update(model):
    """Fixed random actions compare realized function updates across encodings."""
    record = {}
    for name, p in projections(model).items():
        ni = (
            p.map.weight.shape[1]
            if isinstance(p.map, ffn_dst.MaskedLinear)
            else p.map.in_features
        )
        x = torch.randn(
            7,
            ni,
            generator=torch.Generator().manual_seed(
                parameter_seed(17, name + ".action")
            ),
        ).to(p.bias.device)
        record[name] = (
            x,
            p(x).clone(),
            p.map.atoms.p.detach().clone() if isinstance(p.map, CSTLinear) else None,
        )
    return record


@torch.no_grad()
def after_update(model, previous):
    record = {}
    for name, p in projections(model).items():
        x, y, old = previous[name]
        now = p(x)
        delta = rms(now - y)
        row = {
            "action_delta_rms": delta,
            "previous_action_rms": rms(y),
            "relative_action_delta": delta / max(rms(y), 1e-30),
        }
        if old is not None:
            change = p.map.atoms.p - old
            for col, period in ((2, p.map.out_features), (3, p.map.in_features)):
                change[:, col] = (change[:, col] + period / 2).remainder(
                    period
                ) - period / 2
            row["field_actual_delta_rms"] = (
                change.double().square().mean(0).sqrt().tolist()
            )
        record[name] = row
    return record
