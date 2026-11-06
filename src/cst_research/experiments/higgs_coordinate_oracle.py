"""Independent scalar verification math; never called by the training adapter."""

from __future__ import annotations

import math


def adam_scalar(
    point, gradient, mean, square, step, lr, *, beta1=0.9, beta2=0.999, eps=1e-8
):
    mean = beta1 * mean + (1 - beta1) * gradient
    square = beta2 * square + (1 - beta2) * gradient * gradient
    proposal = point - lr * (mean / (1 - beta1**step)) / (
        math.sqrt(square / (1 - beta2**step)) + eps
    )
    return proposal, mean, square


def retract_scalar(
    previous,
    proposal,
    *,
    lr=0.0003,
    amplitude_decay=0.01,
    period=1024.0,
    radial_regularization=0.1,
):
    """Specified Polar finite-chord activity law plus decoupled old amplitude.

    This tiny FP64 scalar oracle is deliberately independent of torchcst's tensor
    implementation and CUDA kernel. The position LR never enters the Polar law.
    """
    s, t = previous[:2]
    q = s * s + t * t
    if q == 0:
        s, t, q = 0.0, 1.0, 1.0
    else:
        radius = min(2.0, max(1.0, math.sqrt(q)))
        factor = radius / math.sqrt(q)
        s, t, q = s * factor, t * factor, radius * radius
    ds, dt = proposal[0] - previous[0], proposal[1] - previous[1]
    radial = (ds * s + dt * t) / q
    ds, dt = ds - radial * s, dt - radial * t
    cs, ct = s + ds, t + dt
    norm = math.hypot(cs, ct)
    direction = (cs / norm, ct / norm)
    task_q = min(4.0, max(1.0, q + ds * ds + dt * dt))
    accepted_q = 1.0 / (
        1.0 - ((task_q - 1.0) / task_q) * math.exp(-4.0 * radial_regularization * lr)
    )
    new_radius = math.sqrt(accepted_q)
    old_amplitude = previous[0] / max(math.hypot(previous[0], previous[1]), 1e-150)
    target = min(1.0, max(-1.0, direction[0] - lr * amplitude_decay * old_amplitude))
    accepted_s = new_radius * target
    accepted_t = math.copysign(
        new_radius * math.sqrt(max(0.0, (1.0 - abs(target)) * (1.0 + abs(target)))),
        direction[1],
    )
    return [accepted_s, accepted_t, proposal[2] % period, proposal[3] % period]
