"""Optional CPU amplitude-trajectory diagnostics; no optimizer/model mutation."""

from __future__ import annotations

import math

import torch


class AmplitudeMotion:
    """Distinguish net amplitude change from accumulated per-update motion.

    Inputs are named Polar [Atom,4] Parameters and their amplitude bounds.
    All retained diagnostic tensors are on CPU. Sampling happens outside the
    synchronized complete-update timer; instrumented elapsed time includes it.
    """

    def __init__(self, parameters):
        self.parameters = parameters
        self.steps = 0
        self.initial, self.previous, self.path, self.square_motion = {}, {}, {}, {}
        for name, (parameter, maximum) in parameters.items():
            if (
                parameter.ndim != 2
                or parameter.shape[1] != 4
                or not math.isfinite(maximum)
                or maximum <= 0
            ):
                raise ValueError("Polar [Atom,4] and positive amplitude bound required")
            value = self._sample(parameter, maximum)
            self.initial[name] = value
            self.previous[name] = value.clone()
            self.path[name] = torch.zeros_like(value)
            self.square_motion[name] = torch.zeros_like(value)

    @staticmethod
    def _sample(parameter, maximum):
        polar = parameter.detach()[:, :2].to(device="cpu", dtype=torch.float64)
        norm = polar.square().sum(-1).sqrt()
        if not bool(torch.isfinite(polar).all()) or bool((norm == 0).any()):
            raise FloatingPointError("nonfinite or singular Polar diagnostic sample")
        return maximum * polar[:, 0] / norm

    def update(self):
        for name, (parameter, maximum) in self.parameters.items():
            value = self._sample(parameter, maximum)
            delta = value - self.previous[name]
            self.path[name] += delta.abs()
            self.square_motion[name] += delta.square()
            self.previous[name] = value
        self.steps += 1

    def report(self):
        rms = lambda value: float(value.square().mean().sqrt())
        return {
            name: {
                "sampled_updates": self.steps,
                "initial_amplitude_rms": rms(self.initial[name]),
                "current_amplitude_rms": rms(value),
                "net_amplitude_change_rms": rms(value - self.initial[name]),
                "cumulative_absolute_amplitude_motion_rms": rms(self.path[name]),
                "per_update_amplitude_motion_rms": float(
                    (self.square_motion[name].mean() / max(1, self.steps)).sqrt()
                ),
            }
            for name, value in self.previous.items()
        }


@torch.no_grad()
def geometry_snapshot(model):
    """Read physical amplitudes/radii under the existing Polar decoding law."""
    from torchcst import CSTLinear
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )

    result = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, CSTLinear):
            continue
        p = layer.atoms.p.detach().to(device="cpu", dtype=torch.float64)
        amplitude, radius_in, radius_out = decoded_values(layer.kernel, p)
        maximum = float(layer.kernel.scalar("amplitude_max"))
        q = ((p[:, :2].square().sum(-1) - 1) / 3).clamp(0, 1)
        result[name] = {
            "amplitude_rms": float(amplitude.square().mean().sqrt()),
            "near_amplitude_bound_fraction": float(
                (amplitude.abs() >= maximum * (1 - 1e-7)).double().mean()
            ),
            "q_mean": float(q.mean()),
            "input_radius_mean": float(radius_in.mean()),
            "input_radius_p05": float(torch.quantile(radius_in, 0.05)),
            "input_radius_p95": float(torch.quantile(radius_in, 0.95)),
            "output_radius_mean": float(radius_out.mean()),
            "output_radius_p05": float(torch.quantile(radius_out, 0.05)),
            "output_radius_p95": float(torch.quantile(radius_out, 0.95)),
        }
    return result
