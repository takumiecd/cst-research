"""Read-only center displacement diagnostics for unit-spaced flat-torus pilots."""

from __future__ import annotations

import torch


class CenterMotion:
    """Per-update shortest torus displacement, normalized by the previous radius.

    CPU sampling runs after synchronized timing. It changes no Parameter, gradient,
    optimizer, or RNG state. Counts use current amplitude bins; totals are path
    statistics, rather than the displacement between infrequent snapshots.
    """

    def __init__(self, model):
        from torchcst import CSTLinear

        self.layers = {
            n: m for n, m in model.named_modules() if isinstance(m, CSTLinear)
        }
        self.previous = {n: self.sample(m) for n, m in self.layers.items()}
        self.square = {
            n: torch.zeros(m.atoms.count, dtype=torch.float64)
            for n, m in self.layers.items()
        }
        self.path = {n: torch.zeros_like(v) for n, v in self.square.items()}
        self.last = {n: torch.zeros_like(v) for n, v in self.square.items()}
        self.steps = 0

    @staticmethod
    def sample(layer):
        from torchcst._backends.torch.parameterizations.radius_cell_mass import (
            decoded_values,
        )

        p = layer.atoms.p.detach().to(device="cpu", dtype=torch.float64)
        a, ri, ro = decoded_values(layer.kernel, p)
        return p, a, ri, ro

    def update(self):
        for n, m in self.layers.items():
            p, a, ri, ro = self.sample(m)
            old, _, old_ri, old_ro = self.previous[n]
            d = p[:, 2:] - old[:, 2:]
            period = torch.tensor([m.out_features, m.in_features], dtype=torch.float64)
            d = (d + period / 2).remainder(period) - period / 2
            scaled = d / torch.stack((old_ro, old_ri), -1)
            distance = scaled.square().mean(-1).sqrt()
            assert torch.isfinite(distance).all()
            self.square[n] += distance.square()
            self.path[n] += distance
            self.last[n] = distance
            self.previous[n] = (p, a, ri, ro)
        self.steps += 1

    def report(self):
        from .ffn_envelope import EDGES

        result = {}
        for n, (_, a, _, _) in self.previous.items():

            def stats(mask):
                count = int(mask.sum())
                if not count:
                    return {"count": 0}
                values = {
                    "last_step_over_radius": self.last[n],
                    "rms_step_over_radius": (
                        self.square[n] / max(1, self.steps)
                    ).sqrt(),
                    "path_over_radius": self.path[n],
                }
                return {
                    "count": count,
                    **{
                        key: {
                            "mean": float(v[mask].mean()),
                            "p90": float(torch.quantile(v[mask], 0.9)),
                        }
                        for key, v in values.items()
                    },
                }

            result[n] = {
                "sampled_updates": self.steps,
                "all": stats(torch.ones(len(a), dtype=torch.bool)),
                "bins": [
                    {"low": lo, "high": hi, **stats((a.abs() >= lo) & (a.abs() < hi))}
                    for lo, hi in zip(EDGES[:-1], EDGES[1:])
                ],
            }
        return result
