"""A learned, shared chart from four Atom latents to physical parameters."""

import math

import torch
from torch import nn


class PhysicalAtomDecoder(nn.Module):
    fields = ("amplitude", "radius", "output_center", "input_center")

    def __init__(self, amplitude_scale, *, hidden=32, period=1024.0):
        super().__init__()
        self.amplitude_scale = float(amplitude_scale)
        self.period = float(period)
        self.network = nn.Sequential(
            nn.Linear(6, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def base(self, latent):
        """Independent coordinate map used to match initial physical atoms."""
        amplitude, log_radius, out, inp = latent.unbind(-1)
        return torch.stack(
            (
                amplitude.clamp(-32.0, 32.0),
                log_radius.clamp(math.log(0.5), math.log(512.0))
                .exp()
                .clamp(0.5, 512.0),
                out.remainder(self.period),
                inp.remainder(self.period),
            ),
            -1,
        )

    def preclip(self, latent):
        """Return signed amplitude, log radius, and unwrapped centers."""
        a, r, out, inp = latent.unbind(-1)
        angles = latent[..., 2:].remainder(self.period) * (math.tau / self.period)
        features = torch.cat(
            (
                torch.asinh(a / self.amplitude_scale).unsqueeze(-1),
                ((r - math.log(4.0)) / 2).unsqueeze(-1),
                angles.sin(),
                angles.cos(),
            ),
            dim=-1,
        )
        da, dr, dout, din = self.network(features).unbind(-1)
        return torch.stack(
            (a + self.amplitude_scale * da, r + dr, out + 8 * dout, inp + 8 * din),
            dim=-1,
        )

    def forward(self, latent):
        return self.base(self.preclip(latent))
