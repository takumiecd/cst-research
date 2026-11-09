"""Data-free FFN feasibility contract; imports no Torch or GPU dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, isfinite, isqrt, pi


def atom_count(d_in: int, d_out: int, coefficient: int = 8) -> int:
    """Exact ceil(c sqrt(d_in d_out)), avoiding floating-point rounding."""
    if any(type(x) is not int or x < 1 for x in (d_in, d_out, coefficient)):
        raise ValueError("positive integer dimensions and coefficient required")
    square = coefficient**2 * d_in * d_out
    root = isqrt(square)
    return root + (root * root < square)


def learning_rate(step: int, horizon_updates: int) -> float:
    """One fixed planned horizon, independent of an observation's stop point."""
    if (
        any(type(n) is not int or n < 1 for n in (step, horizon_updates))
        or step > horizon_updates
    ):
        raise ValueError("positive step within the planned update horizon required")
    warmup = max(1, min(200, horizon_updates // 10))
    if step <= warmup:
        return 0.0003 * step / warmup
    progress = (step - warmup) / max(1, horizon_updates - warmup)
    return 0.00003 + 0.00027 * (1 + cos(pi * progress)) / 2


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    layers: int = 4
    heads: int = 4
    context: int = 128
    vocabulary: int = 50257
    expansion: int = 4
    coefficient: int = 8
    q: float = 0.0
    init_std: float = 0.02
    amplitude_max: float = 1.0
    w_c: float = 0.3
    radius_birth: float = 4.0
    feature_order: str = "contiguous"
    atom_lr_multiplier: float = 1.0
    radius_envelope: str = "rational"
    closed_lower_power: float = 4.0
    closed_upper_power: float = 1.0
    zero_opening_scale: float = 0.03
    attention_kind: str = "dense"
    linear_backend: str = "native"
    radius_mode: str = "polar"
    initialization_mode: str = "moments"
    tau: float = 0.03
    radius_fraction: float = 0.0
    radius_fraction_lr_multiplier: float = 1.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key not in (
                "q",
                "tau",
                "radius_fraction",
                "radius_fraction_lr_multiplier",
                "init_std",
                "amplitude_max",
                "w_c",
                "radius_birth",
                "feature_order",
                "attention_kind",
                "linear_backend",
                "radius_mode",
                "initialization_mode",
                "atom_lr_multiplier",
                "radius_envelope",
                "closed_lower_power",
                "closed_upper_power",
                "zero_opening_scale",
            ) and (type(value) is not int or value < 1):
                raise ValueError(f"positive integer {key} required")
        for key in (
            "tau",
            "radius_fraction",
            "radius_fraction_lr_multiplier",
            "q",
            "init_std",
            "amplitude_max",
            "w_c",
            "radius_birth",
            "atom_lr_multiplier",
            "closed_lower_power",
            "closed_upper_power",
            "zero_opening_scale",
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"finite numeric {key} required")
        if (
            self.width % self.heads
            or not 0 <= self.q <= 1
            or not 0 < self.init_std <= 1
            or self.amplitude_max <= 0
            or self.w_c <= 0
            or not 0.5 <= self.radius_birth <= 50
            or self.feature_order not in ("contiguous", "permuted")
            or self.attention_kind
            not in ("dense", "cst_v", "cst_q", "cst_k", "cst_qkv", "cst_qkvo")
            or self.linear_backend
            not in ("native", "levels", "levels_stream64", "levels_wide_amplitude")
            or self.radius_mode
            not in (
                "polar",
                "learned",
                "fixed",
                "polar_detached",
                "polar_live",
                "envelope",
            )
            or (
                self.radius_mode in ("polar_detached", "polar_live")
                and self.linear_backend != "levels"
            )
            or (self.radius_mode != "polar" and self.attention_kind != "dense")
            or self.tau <= 0
            or not 0 <= self.radius_fraction <= 1
            or self.radius_fraction_lr_multiplier <= 0
            or (
                self.radius_mode == "envelope"
                and (
                    self.width != 512
                    or self.expansion != 4
                    or self.coefficient not in (8, 16)
                    or self.linear_backend not in ("levels", "levels_wide_amplitude")
                )
            )
            or (
                self.radius_mode == "envelope"
                and (
                    self.amplitude_max > 32
                    or (
                        self.amplitude_max > 1
                        and self.linear_backend != "levels_wide_amplitude"
                    )
                )
            )
            or (
                self.linear_backend == "levels_wide_amplitude"
                and self.radius_mode != "envelope"
            )
            or self.initialization_mode not in ("moments", "probe_rms")
            or (
                self.initialization_mode == "probe_rms"
                and self.radius_mode != "envelope"
            )
            or self.atom_lr_multiplier <= 0
            or self.radius_envelope
            not in ("rational", "closed_polynomial", "rational_zero_opening")
            or self.zero_opening_scale <= 0
            or not 1 <= self.closed_upper_power <= self.closed_lower_power
        ):
            raise ValueError("invalid head divisibility, q or initialization scale")

    def envelope_settings(self):
        """Explicit kernel constructor defaults also required in admission receipts."""
        return {
            key: getattr(self, key)
            for key in (
                "radius_envelope",
                "radius_birth",
                "closed_lower_power",
                "closed_upper_power",
                "zero_opening_scale",
            )
        }

    def recipe(self):
        result = {
            key: getattr(self, key)
            for key in (
                "q",
                "amplitude_max",
                "w_c",
                "radius_birth",
                "feature_order",
                "atom_lr_multiplier",
                "coefficient",
            )
        }
        if self.radius_envelope != "rational":
            result.update(
                radius_envelope=self.radius_envelope,
                closed_lower_power=self.closed_lower_power,
                closed_upper_power=self.closed_upper_power,
            )
        if self.radius_envelope == "rational_zero_opening":
            result["zero_opening_scale"] = self.zero_opening_scale
        if self.attention_kind != "dense":
            result["attention_kind"] = self.attention_kind
        if self.linear_backend != "native":
            result["linear_backend"] = self.linear_backend
        if self.radius_mode != "polar":
            result["radius_mode"] = self.radius_mode
        if self.radius_mode == "envelope":
            result.update(
                tau=self.tau,
                radius_fraction=self.radius_fraction,
                radius_fraction_lr_multiplier=self.radius_fraction_lr_multiplier,
            )
        if self.initialization_mode != "moments":
            result["initialization_mode"] = self.initialization_mode
        return result

    @property
    def intermediate(self):
        return self.width * self.expansion

    @property
    def atoms_per_projection(self):
        return atom_count(self.width, self.intermediate, self.coefficient)

    def ffn_parameter_budget(self):
        bias = self.width + self.intermediate
        return {
            "atoms_per_projection": self.atoms_per_projection,
            "cst_parameters": self.layers * (8 * self.atoms_per_projection + bias),
            "dense_parameters": self.layers
            * (2 * self.width * self.intermediate + bias),
            "scope": "FFN only; Dense attention and vocabulary weights are excluded",
        }


def qualification_covers_model(receipt, config, *, ffn_kind=None):
    """Admit both native directions and the full model's actual recipe."""
    if (
        config.linear_backend != "native" or config.attention_kind == "cst_qkvo"
    ) and asdict(config) not in receipt.get("task_model_configs", []):
        return False
    if config.recipe() not in receipt.get("task_recipes", []):
        return False
    if (
        ffn_kind is not None
        and config.attention_kind != "dense"
        and {
            "ffn_kind": ffn_kind,
            "attention_kind": config.attention_kind,
        }
        not in receipt.get("model_kinds", [])
    ):
        return False
    scalars = {
        key: getattr(config, key)
        for key in ("q", "amplitude_max", "w_c", "radius_birth")
    }
    if config.radius_mode != "polar":
        scalars["radius_mode"] = config.radius_mode
    if config.linear_backend != "native":
        scalars["linear_backend"] = config.linear_backend
    if config.radius_envelope != "rational":
        scalars.update(
            radius_envelope=config.radius_envelope,
            closed_lower_power=config.closed_lower_power,
            closed_upper_power=config.closed_upper_power,
        )
    if config.radius_envelope == "rational_zero_opening":
        scalars["zero_opening_scale"] = config.zero_opening_scale
    if config.radius_mode == "envelope":
        scalars.update(
            tau=config.tau,
            radius_fraction=config.radius_fraction,
            radius_fraction_lr_multiplier=config.radius_fraction_lr_multiplier,
        )
    if config.initialization_mode != "moments":
        scalars["initialization_mode"] = config.initialization_mode
    dimensions = (
        [(config.width, config.width)]
        if config.attention_kind != "dense"
        else [(config.width, config.intermediate), (config.intermediate, config.width)]
    )
    if config.attention_kind != "dense" and ffn_kind == "cst":
        dimensions += [
            (config.width, config.intermediate),
            (config.intermediate, config.width),
        ]
    for ni, no in dimensions:
        if not any(
            row.get("in") == ni
            and row.get("out") == no
            and row.get("coefficient") == config.coefficient
            and row.get("atoms") == atom_count(ni, no, config.coefficient)
            and all(row.get(key) == value for key, value in scalars.items())
            for row in receipt.get("cases", [])
        ):
            return False
    return True
