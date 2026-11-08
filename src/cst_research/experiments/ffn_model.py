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
    radius_envelope="rational",
    closed_lower_power=4.0,
    closed_upper_power=1.0,
    zero_opening_scale=0.03,
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
                        birth=min(
                            50.0
                            if radius_envelope == "closed_polynomial"
                            else radius_birth,
                            min(d_in, d_out) / 2,
                        ),
                        maximum=min(50.0, min(d_in, d_out) / 2),
                        upper_floor=0.5
                        if radius_envelope == "closed_polynomial"
                        else 1.0,
                    ),
                    radius_envelope=radius_envelope,
                    closed_lower_power=closed_lower_power,
                    closed_upper_power=closed_upper_power,
                    zero_opening_scale=zero_opening_scale,
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


def configure_native(model, *, linear_backend=None):
    from torchcst import CSTLinear, FixedSelector, make_algorithm_registry
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

    backend = linear_backend or getattr(
        getattr(model, "config", None), "linear_backend", "native"
    )
    if backend not in ("native", "levels", "levels_stream64"):
        raise ValueError("unknown explicit Linear backend")
    registry = make_algorithm_registry()
    if backend != "native":
        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_multirow import (
            LM_LEVELS,
            LM_LEVELS_STREAM64,
            RadiusLMLevelsAlgorithm,
            RadiusLMLevelsStream64Algorithm,
        )

        registry.register(RadiusLMLevelsAlgorithm())
        registry.register(RadiusLMLevelsStream64Algorithm())
    if any(
        getattr(m, "independent_radius_mode", None) is not None for m in model.modules()
    ):
        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_multirow import (
            LM_PHYSICAL_LEVELS,
            RadiusLMPhysicalLevelsAlgorithm,
        )

        registry.register(RadiusLMPhysicalLevelsAlgorithm())
    for layer in model.modules():
        if isinstance(layer, CSTLinear):
            if getattr(layer, "independent_radius_mode", None) is not None:
                layer.selector = FixedSelector(LM_PHYSICAL_LEVELS, registry=registry)
                continue
            plan = (
                plans.GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048
                if max(layer.in_features, layer.out_features) > 1024
                else plans.GLOBAL_ACTIVE_GRAD4_NATIVE
            )
            if backend != "native":
                plan = LM_LEVELS if backend == "levels" else LM_LEVELS_STREAM64
            layer.selector = FixedSelector(plan, registry=registry)


def cst_optimizer(model, *, atom_lr_multiplier=1.0, **options):
    """Explicit GPU coordinate-update Plan; default radius reference is CPU-only."""
    if getattr(getattr(model, "config", None), "radius_mode", "polar") in (
        "polar_live",
        "polar_detached",
    ):
        from .ffn_polar_radius import optimizer

        return optimizer(model, atom_lr_multiplier=atom_lr_multiplier, **options)
    if getattr(getattr(model, "config", None), "radius_mode", "polar") != "polar":
        from .ffn_radius import physical_optimizer

        return physical_optimizer(
            model, atom_lr_multiplier=atom_lr_multiplier, **options
        )
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
                radius_envelope=config.radius_envelope,
                closed_lower_power=config.closed_lower_power,
                closed_upper_power=config.closed_upper_power,
                zero_opening_scale=config.zero_opening_scale,
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


class CSTAttentionProjection(nn.Module):
    """One explicit CST Q/K/V map with byte-matched Dense complementary rows."""

    def __init__(self, config, seed, index):
        super().__init__()
        self.slot = {"cst_q": 0, "cst_k": 1, "cst_v": 2}[config.attention_kind]
        self.dense_name = ("kv", "qv", "qk")[self.slot]
        self.cst_name = ("query", "key", "value")[self.slot]
        setattr(self, self.dense_name, nn.Linear(config.width, 2 * config.width))
        setattr(
            self,
            self.cst_name,
            Projection(
                config.width,
                config.width,
                config=config,
                kind="cst",
                std=config.init_std,
                seed=parameter_seed(
                    seed, f"block{index}.attention." + ("q", "k", "v")[self.slot]
                ),
            ),
        )

    def forward(self, x):
        ordinary = getattr(self, self.dense_name)(x)
        selected = getattr(self, self.cst_name)(x)
        if self.slot == 2:
            return torch.cat((ordinary, selected), dim=-1)
        parts = list(ordinary.chunk(2, dim=-1))
        parts.insert(self.slot, selected)
        return torch.cat(parts, dim=-1)


class CSTAllQKVProjection(nn.Module):
    """Independent square CST Q, K and V maps; the attention operator is unchanged."""

    def __init__(self, config, seed, index):
        super().__init__()
        for field, short in (("query", "q"), ("key", "k"), ("value", "v")):
            setattr(
                self,
                field,
                Projection(
                    config.width,
                    config.width,
                    config=config,
                    kind="cst",
                    std=config.init_std,
                    seed=parameter_seed(seed, f"block{index}.attention.{short}"),
                ),
            )

    def forward(self, x):
        return torch.cat((self.query(x), self.key(x), self.value(x)), dim=-1)


class Block(nn.Module):
    def __init__(self, config, kind, seed, index):
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = (
            CSTAllQKVProjection(config, seed, index)
            if config.attention_kind in ("cst_qkv", "cst_qkvo")
            else CSTAttentionProjection(config, seed, index)
            if config.attention_kind != "dense"
            else nn.Linear(config.width, config.width * 3)
        )
        self.attention_out = (
            Projection(
                config.width,
                config.width,
                config=config,
                kind="cst",
                std=config.init_std / math.sqrt(2 * config.layers),
                seed=parameter_seed(seed, f"block{index}.attention.o"),
            )
            if config.attention_kind == "cst_qkvo"
            else nn.Linear(config.width, config.width)
        )
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

        # Recover the exact historical packed Q/K initialization. Named streams
        # keep every other shared Parameter unchanged across Dense/CST-V arms.
        if config.attention_kind in ("cst_q", "cst_k", "cst_v"):
            for i, block in enumerate(self.blocks):
                packed = torch.empty(3 * config.width, config.width).normal_(
                    std=config.init_std,
                    generator=torch.Generator().manual_seed(
                        parameter_seed(seed, f"blocks.{i}.qkv.weight")
                    ),
                )
                with torch.no_grad():
                    remaining = [
                        row
                        for slot in range(3)
                        if slot != block.qkv.slot
                        for row in range(slot * config.width, (slot + 1) * config.width)
                    ]
                    getattr(block.qkv, block.qkv.dense_name).weight.copy_(
                        packed[remaining]
                    )

        if config.radius_mode in ("polar_live", "polar_detached") and kind == "cst":
            from .ffn_polar_radius import convert_ffn

            convert_ffn(self)
        elif config.radius_mode != "polar" and kind == "cst":
            from .ffn_radius import convert_ffn

            convert_ffn(self)

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
