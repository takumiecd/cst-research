"""Reproducible, gradient-conditioned proposals in the physical Atom chart."""

import math
from itertools import product


def actions(seed):
    """Yield 384 actions, each described by a family, four coefficients and a seed."""
    import torch

    yield "control", (0.0,) * 4, seed
    for field, sign, scale in product(range(4), (-1, 1), (0.125, 0.25, 0.5, 1, 2, 4)):
        coefficients = [0.0] * 4
        coefficients[field] = sign * scale
        yield "axis_gradient", tuple(coefficients), seed
    engine = torch.quasirandom.SobolEngine(4, scramble=True, seed=seed)
    for family, count in (
        ("joint_gradient", 95),
        ("local_random", 96),
        ("relocate", 96),
        ("mixed", 48),
    ):
        for index, row in enumerate(engine.draw(count).tolist()):
            coefficients = tuple(
                math.copysign(2 ** (5 * abs(2 * x - 1) - 3), x - 0.5) for x in row
            )
            yield family, coefficients, seed + index


def direction(physical, gradients):
    """Normalize each gradient in units of the corresponding physical move."""
    import torch

    amplitude, radius = physical[:, :2].unbind(1)
    units = torch.stack(
        (amplitude.abs().clamp_min(0.03), 0.25 * radius, 0.5 * radius, 0.5 * radius),
        dim=1,
    )
    normalized = gradients * units
    reference = torch.quantile(normalized.abs(), 0.90, dim=0)
    denominator = torch.where(reference > 0, reference, torch.ones_like(reference))
    return -(normalized / denominator).clamp(-3, 3)


def propose(raw, physical, descent, action):
    """Apply a proposal through the rational-L chart, returning reachable raw rows."""
    import torch

    family, coefficients, seed = action
    if family == "control":
        return raw.clone()
    coefficient = raw.new_tensor(coefficients)
    if family in ("local_random", "relocate"):
        generator = torch.Generator(device=raw.device).manual_seed(seed)
        heading = (
            2
            * torch.rand(
                raw.shape, generator=generator, device=raw.device, dtype=raw.dtype
            )
            - 1
        )
    else:
        heading = descent
    delta = heading * coefficient
    amp, radius, out, inp = physical.double().unbind(1)
    a = (amp + delta[:, 0] * amp.abs().clamp_min(0.03)).clamp(-32, 32)
    result = raw.clone()
    result[:, 0] = a
    if coefficients[0] or coefficients[1]:
        a = result[:, 0].double()
        lower = 0.5 + 3.5 / (1 + 3 * (a / 0.3).square())
        target = (radius * (0.25 * delta[:, 1].double()).exp()).maximum(lower)
        target = target.clamp_max(512.0)
        result[:, 1] = (
            (target.log() - lower.log()) / (math.log(512.0) - lower.log())
        ).clamp(0, 1)
    for field, center in ((2, out), (3, inp)):
        if coefficients[field]:
            shift = 0.5 * radius * delta[:, field]
            if family == "relocate":
                # Different atoms sample different destinations across the chart.
                shift = delta[:, field].sign() * (8 * (64 ** heading[:, field].abs()))
            elif family == "mixed":
                shift = coefficient[field].sign() * (
                    8 * (64 ** descent[:, field].abs().div(3))
                )
            result[:, field] = (center + shift).remainder(1024)
    return result
