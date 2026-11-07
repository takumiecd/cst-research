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

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key not in (
                "q",
                "init_std",
                "amplitude_max",
                "w_c",
                "radius_birth",
                "feature_order",
                "atom_lr_multiplier",
            ) and (type(value) is not int or value < 1):
                raise ValueError(f"positive integer {key} required")
        for key in (
            "q",
            "init_std",
            "amplitude_max",
            "w_c",
            "radius_birth",
            "atom_lr_multiplier",
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
            or self.atom_lr_multiplier <= 0
        ):
            raise ValueError("invalid head divisibility, q or initialization scale")

    def recipe(self):
        return {
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


def qualification_covers_model(receipt, config):
    """Admit both native directions and the full model's actual recipe."""
    if config.recipe() not in receipt.get("task_recipes", []):
        return False
    scalars = {
        key: getattr(config, key)
        for key in ("q", "amplitude_max", "w_c", "radius_birth")
    }
    for ni, no in (
        (config.width, config.intermediate),
        (config.intermediate, config.width),
    ):
        if not any(
            row.get("in") == ni
            and row.get("out") == no
            and row.get("coefficient") == config.coefficient
            and row.get("atoms") == config.atoms_per_projection
            and all(row.get(key) == value for key, value in scalars.items())
            for row in receipt.get("cases", [])
        ):
            return False
    return True
