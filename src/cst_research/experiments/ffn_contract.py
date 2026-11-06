"""Data-free FFN feasibility contract; imports no Torch or GPU dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isqrt


def atom_count(d_in: int, d_out: int, coefficient: int = 8) -> int:
    """Exact ceil(c sqrt(d_in d_out)), avoiding floating-point rounding."""
    if any(type(x) is not int or x < 1 for x in (d_in, d_out, coefficient)):
        raise ValueError("positive integer dimensions and coefficient required")
    square = coefficient**2 * d_in * d_out
    root = isqrt(square)
    return root + (root * root < square)


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

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key not in ("q", "init_std") and (type(value) is not int or value < 1):
                raise ValueError(f"positive integer {key} required")
        if self.width % self.heads or not 0 <= self.q <= 1 or not 0 < self.init_std <= 1:
            raise ValueError("invalid head divisibility, q or initialization scale")

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
            "dense_parameters": self.layers * (2 * self.width * self.intermediate + bias),
            "scope": "FFN only; Dense attention and vocabulary weights are excluded",
        }
