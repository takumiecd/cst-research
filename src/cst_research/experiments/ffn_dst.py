"""Rectangular row-balanced RigL-style LM accuracy reference, using dense storage.

Prune/grow rules adapt torchcst f37729e6:higgs_dst_reference.py. The LM uses
its existing named normal initializer and active-fan variance correction.
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ffn_contract import atom_count
from .ffn_model import LanguageModel, parameter_seed


def tensor_sha(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


class MaskedLinear(nn.Module):
    def __init__(self, dense_weight, active_count, generator):
        super().__init__()
        no, ni = dense_weight.shape
        if not no <= active_count <= no * ni:
            raise ValueError("each row needs at least one active scalar slot")
        counts = torch.full((no,), active_count // no, dtype=torch.int64)
        counts[: active_count % no] += 1
        order = torch.rand(no, ni, generator=generator).argsort(dim=1, stable=True)
        mask = torch.zeros(no, ni, dtype=torch.bool)
        mask.scatter_(1, order, torch.arange(ni).expand(no, -1) < counts[:, None])
        values = dense_weight.detach().clone() * mask
        values *= (ni / counts.to(values)).sqrt()[:, None]
        self.weight = nn.Parameter(values)
        self.register_buffer("mask", mask)
        self.register_buffer("row_active_counts", counts)
        self.register_buffer(
            "dense_gradient", torch.zeros_like(values), persistent=False
        )

    def forward(self, x):
        # Inactive stored weights/moments are zero, but dormant dW is retained.
        return F.linear(x, self.weight)

    @torch.no_grad()
    def prepare_gradient(self):
        self.dense_gradient.copy_(self.weight.grad)
        self.weight.grad.mul_(self.mask)


def make_model(config, seed=17):
    model = LanguageModel(config, "dense", seed)
    for i, block in enumerate(model.blocks):
        for direction in ("up", "down"):
            projection = getattr(block.ffn, direction)
            old = projection.map
            projection.map = MaskedLinear(
                old.weight,
                4 * atom_count(old.in_features, old.out_features, config.coefficient),
                torch.Generator().manual_seed(
                    parameter_seed(seed, f"block{i}.{direction}.dst-mask")
                ),
            )
    model.kind = "dst"
    return model


def layers(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, MaskedLinear)}


def make_optimizer(model, sparse_lr):
    masked = [m.weight for m in layers(model).values()]
    ids = {id(p) for p in masked}
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if id(p) not in ids],
                "lr": 0.0003,
                "lr_scale": 1.0,
            },
            {"params": masked, "lr": sparse_lr, "lr_scale": sparse_lr / 0.0003},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )


def drop_fraction(step, total_steps):
    end = int(0.75 * total_steps)
    return 0.3 * (1 + math.cos(math.pi * step / end)) / 2 if step < end else 0.0


def is_topology_step(step, total_steps):
    return step > 0 and step % 250 == 0 and step < int(0.75 * total_steps)


@torch.no_grad()
def rewire(linear, optimizer, fraction):
    if not 0 < fraction < 1:
        raise ValueError("positive topology fraction below1 required")
    mask, weight = linear.mask, linear.weight
    old_mask = mask.clone()
    magnitude = weight.abs()
    threshold = (
        magnitude[old_mask].kthvalue(max(1, int(fraction * int(old_mask.sum())))).values
    )
    keep = old_mask & (magnitude > threshold)
    empty = ~keep.any(1)
    rescue = magnitude.masked_fill(~old_mask, -1).argmax(1)
    rows = empty.nonzero(as_tuple=True)[0]
    keep[rows, rescue[rows]] = True
    missing = linear.row_active_counts - keep.sum(1)
    order = (
        linear.dense_gradient.abs()
        .masked_fill(keep, -torch.inf)
        .argsort(dim=1, descending=True, stable=True)
    )
    rank = torch.empty_like(order)
    rank.scatter_(
        1, order, torch.arange(mask.shape[1], device=mask.device).expand_as(order)
    )
    grown = rank < missing[:, None]
    state = optimizer.state[weight]
    before = {k: state[k].clone() for k in ("exp_avg", "exp_avg_sq")}
    survivor_values = weight[keep].clone()
    clock = state["step"].clone()
    weight.mul_(keep)
    mask.copy_(keep | grown)
    weight.grad.mul_(keep)
    for k in before:
        state[k].mul_(keep)
    checks = {
        "row_counts": torch.equal(mask.sum(1), linear.row_active_counts),
        "global_count": int(mask.sum()) == int(old_mask.sum()),
        "inactive_zero": bool((weight[~mask] == 0).all()),
        "grown_zero": bool((weight[grown] == 0).all()),
        "reset_moments": all(bool((state[k][~keep] == 0).all()) for k in before),
        "survivors_preserved": torch.equal(weight[keep], survivor_values)
        and all(torch.equal(state[k][keep], before[k][keep]) for k in before),
        "clock_unchanged": torch.equal(clock, state["step"]),
    }
    assert all(checks.values()), checks
    return {
        "drop_fraction": fraction,
        "active_slots": int(mask.sum()),
        "pruned": int((old_mask & ~keep).sum()),
        "grown": int(grown.sum()),
        "immediate_regrowth": int((grown & old_mask).sum()),
        "mask_sha256": tensor_sha(mask),
        "checks": checks,
    }


def finish_step(model, optimizer, step, total_steps):
    for layer in layers(model).values():
        layer.prepare_gradient()
    if is_topology_step(step, total_steps):
        return {
            "topology": {
                n: rewire(m, optimizer, drop_fraction(step, total_steps))
                for n, m in layers(model).items()
            },
            "gradient_norm_before_clip": None,
        }
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return {"topology": None, "gradient_norm_before_clip": float(norm)}


@torch.no_grad()
def invariants(model, optimizer=None):
    result = {}
    for n, m in layers(model).items():
        assert torch.equal(m.mask.sum(1), m.row_active_counts)
        assert (m.weight[~m.mask] == 0).all() and torch.isfinite(m.weight).all()
        state = optimizer.state.get(m.weight, {}) if optimizer is not None else {}
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert (
                    torch.isfinite(state[key]).all()
                    and (state[key][~m.mask] == 0).all()
                )
        result[n] = {
            "active_slots": int(m.mask.sum()),
            "row_min": int(m.row_active_counts.min()),
            "row_max": int(m.row_active_counts.max()),
            "mask_sha256": tensor_sha(m.mask),
        }
    return result
