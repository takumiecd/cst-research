"""One shared residual decoder for research models with four Atom fields."""

import math

import torch
from torch import nn


class SharedAtomDecoder(nn.Sequential):
    def __init__(self, hidden=16):
        super().__init__(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, 4))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)

    def decode(self, latent, *, amplitude_scale, amplitude_max, periods):
        amplitude, fraction, out, inp = latent.unbind(1)
        out_period, in_period = periods
        features = torch.stack(
            (
                torch.asinh(amplitude / amplitude_scale),
                fraction,
                torch.cos(math.tau * out / out_period),
                torch.cos(math.tau * inp / in_period),
            ),
            dim=1,
        )
        da, df, dc_out, dc_in = self(features).unbind(1)
        return torch.stack(
            (
                (amplitude + amplitude_scale * da).clamp(-amplitude_max, amplitude_max),
                (fraction + df).clamp(0, 1),
                (out + dc_out).remainder(out_period),
                (inp + dc_in).remainder(in_period),
            ),
            dim=1,
        )
