"""Small causal language model consuming the public torchcst initialization API."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from .ffn_contract import ModelConfig, atom_count


def parameter_seed(seed, name):
    return (
        seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
    ) % (2**63 - 1)


def make_cst(
    d_in,
    d_out,
    *,
    coefficient=8,
    std=0.02,
    q=0.0,
    seed=17,
    dtype=torch.float32,
    amplitude_max=1.0,
    w_c=0.3,
    radius_birth=4.0,
):
    from torchcst import (
        ChebyshevSpec,
        CSTLinear,
        RadiusBounds,
        chart_presets,
        init,
        kernels,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer = CSTLinear(
            chart_presets.regular_grid(d_out, spacing=1.0, geometry="flat_torus"),
            chart_presets.regular_grid(d_in, spacing=1.0, geometry="flat_torus"),
            atoms=atom_count(d_in, d_out, coefficient),
            dtype=dtype,
            backend="factored",
            kernel=kernels.product(
                kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
                coordinates=kernels.polar(
                    amplitude_max=amplitude_max,
                    w_c=w_c,
                    input_bounds=RadiusBounds(
                        minimum=0.5,
                        birth=min(radius_birth, min(d_in, d_out) / 2),
                        maximum=min(50.0, min(d_in, d_out) / 2),
                        upper_floor=1.0,
                    ),
                ),
            ),
        )
    report = init.variance_scaling_(
        layer,
        variance=std**2,
        q=q,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    layer.initialization_report = asdict(report)
    return layer


def configure_native(model):
    from torchcst import CSTLinear, FixedSelector, make_algorithm_registry
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

    for layer in model.modules():
        if isinstance(layer, CSTLinear):
            plan = (
                plans.GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048
                if max(layer.in_features, layer.out_features) > 1024
                else plans.GLOBAL_ACTIVE_GRAD4_NATIVE
            )
            layer.selector = FixedSelector(plan, registry=make_algorithm_registry())


def cst_optimizer(model, *, atom_lr_multiplier=1.0, **options):
    """Explicit GPU coordinate-update Plan; default radius reference is CPU-only."""
    from torchcst import CSTAdamW, CSTLinear, FixedSelector, make_algorithm_registry
    from torchcst._backends.cuda.algorithms.polar_update.plans import (
        RADIUS_FLAT_TORUS_FUSED,
    )

    parameters = model.parameters()
    if atom_lr_multiplier != 1.0:
        if "lr" not in options:
            raise ValueError("explicit base LR required for Atom multiplier")
        atom_ids = {
            id(module.atoms.p)
            for module in model.modules()
            if isinstance(module, CSTLinear)
        }
        atoms = [p for p in model.parameters() if id(p) in atom_ids]
        ordinary = [p for p in model.parameters() if id(p) not in atom_ids]
        parameters = []
        if ordinary:
            parameters.append({"params": ordinary, "lr_scale": 1.0})
        parameters.append(
            {
                "params": atoms,
                "lr": options["lr"] * atom_lr_multiplier,
                "lr_scale": atom_lr_multiplier,
            }
        )
    return CSTAdamW(
        parameters,
        model=model,
        update_selector=FixedSelector(
            RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry()
        ),
        **options,
    )


class Projection(nn.Module):
    def __init__(self, ni, no, *, config, kind, std, seed):
        super().__init__()
        if kind == "cst":
            self.map = make_cst(
                ni,
                no,
                coefficient=config.coefficient,
                std=std,
                q=config.q,
                seed=seed,
                amplitude_max=config.amplitude_max,
                w_c=config.w_c,
                radius_birth=config.radius_birth,
            )
        else:
            self.map = nn.Linear(ni, no, bias=False)
        self.bias = nn.Parameter(torch.zeros(no))

    def forward(self, x):
        return self.map(x) + self.bias


class FeedForward(nn.Module):
    def __init__(self, config, kind, seed, index):
        super().__init__()
        self.kind = kind
        self.register_buffer("hidden_order", None)
        self.register_buffer("output_order", None)
        if kind != "none":
            if config.feature_order == "permuted":
                self.hidden_order = torch.randperm(
                    config.intermediate,
                    generator=torch.Generator().manual_seed(
                        parameter_seed(seed, f"block{index}.hidden_order")
                    ),
                )
                self.output_order = torch.randperm(
                    config.width,
                    generator=torch.Generator().manual_seed(
                        parameter_seed(seed, f"block{index}.output_order")
                    ),
                )
            self.up = Projection(
                config.width,
                config.intermediate,
                config=config,
                kind=kind,
                std=config.init_std,
                seed=parameter_seed(seed, f"block{index}.up"),
            )
            self.down = Projection(
                config.intermediate,
                config.width,
                config=config,
                kind=kind,
                std=config.init_std / math.sqrt(2 * config.layers),
                seed=parameter_seed(seed, f"block{index}.down"),
            )

    def forward(self, x):
        if self.kind == "none":
            return torch.zeros_like(x)
        hidden = F.gelu(self.up(x))
        if self.hidden_order is not None:
            hidden = hidden.index_select(-1, self.hidden_order)
        output = self.down(hidden)
        return (
            output
            if self.output_order is None
            else output.index_select(-1, self.output_order)
        )


class Block(nn.Module):
    def __init__(self, config, kind, seed, index):
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, config.width * 3)
        self.attention_out = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.ffn = FeedForward(config, kind, seed, index)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = (
            a.reshape(batch, tokens, self.heads, width // self.heads).transpose(1, 2)
            for a in (q, k, v)
        )
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0
        )
        x = x + self.attention_out(
            attended.transpose(1, 2).contiguous().reshape(batch, tokens, width)
        )
        return x + self.ffn(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig, kind="cst", seed=17):
        super().__init__()
        if kind not in ("dense", "cst", "none"):
            raise ValueError("explicit dense/cst/none FFN kind required")
        self.config, self.kind, self.seed = config, kind, seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(config.vocabulary, config.width)
            self.position = nn.Embedding(config.context, config.width)
            self.blocks = nn.ModuleList(
                Block(config, kind, seed, i) for i in range(config.layers)
            )
            self.final_norm = nn.LayerNorm(config.width)
        # Named streams preserve all shared ordinary weights across all three arms.
        for name, parameter in self.named_parameters():
            if name.endswith("atoms.p"):
                continue
            with torch.no_grad():
                if name.endswith("bias"):
                    parameter.zero_()
                elif "norm" in name and name.endswith("weight"):
                    parameter.fill_(1)
                else:
                    std = config.init_std
                    if "attention_out.weight" in name or ".ffn.down.map.weight" in name:
                        std /= math.sqrt(2 * config.layers)
                    parameter.normal_(
                        std=std,
                        generator=torch.Generator().manual_seed(
                            parameter_seed(seed, name)
                        ),
                    )

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError("tokens must have shape [batch, length<=context]")
        x = self.embedding(tokens) + self.position(
            torch.arange(tokens.shape[1], device=tokens.device)
        )
        for block in self.blocks:
            x = block(x)
        logits = F.linear(self.final_norm(x), self.embedding.weight)
        if targets is None:
            return logits
        if targets.shape != tokens.shape:
            raise ValueError(
                "externally shifted next-token targets must match input shape"
            )
        return logits, F.cross_entropy(
            logits.reshape(-1, self.config.vocabulary), targets.reshape(-1)
        )

    def initialization_reports(self):
        return {
            name: layer.initialization_report
            for name, layer in self.named_modules()
            if hasattr(layer, "initialization_report")
        }
