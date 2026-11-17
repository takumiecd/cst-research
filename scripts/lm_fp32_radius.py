"""Fused FP32 formula and VJP for the frozen rational-radius declaration."""

import math

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit(debug=True)
def _decode(
    raw,
    physical,
    jacobian,
    atoms,
    RADIUS_MIN: tl.constexpr,
    RADIUS_MAX: tl.constexpr,
    LOG_RADIUS_MAX: tl.constexpr,
    ENCODED_MIN: tl.constexpr,
    ENCODED_MAX: tl.constexpr,
    AMPLITUDE_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < atoms
    p = raw + 4 * index
    a = tl.load(p, valid, 0)
    q = tl.load(p + 1, valid, 0)
    co = tl.load(p + 2, valid, 0)
    ci = tl.load(p + 3, valid, 0)
    tl.device_assert(
        (q >= 0)
        & (q <= 1)
        & (tl.abs(a) <= AMPLITUDE_MAX)
        & (tl.abs(co) < float("inf"))
        & (tl.abs(ci) < float("inf")),
        "rational radius requires finite Atom rows, bounded amplitude and q in [0,1]",
    )
    # These constants are the formula used by the archived FFN-down run.
    scaled = a / 0.3
    denominator = 1 + 3 * scaled * scaled
    opening = (4.0 - RADIUS_MIN) / denominator
    lower = RADIUS_MIN + opening
    log_lower = tl.log(lower)
    log_ratio = LOG_RADIUS_MAX - log_lower
    raw_radius = tl.exp(tl.fma(q, log_ratio, log_lower))
    radius = tl.where(q == 0, lower, tl.where(q == 1, RADIUS_MAX, raw_radius))
    radius = tl.minimum(tl.maximum(radius, ENCODED_MIN), ENCODED_MAX)
    lower_gradient = -20.0 * scaled * opening / denominator
    tangent = (raw_radius / lower) * (1 - q) * lower_gradient
    out = physical + 4 * index
    tl.store(out, a, valid)
    tl.store(out + 1, radius, valid)
    tl.store(out + 2, co, valid)
    tl.store(out + 3, ci, valid)
    tl.store(jacobian + 2 * index, tangent, valid)
    tl.store(jacobian + 2 * index + 1, raw_radius * log_ratio, valid)


@triton.jit
def _vjp(physical_grad, jacobian, raw_grad, atoms, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < atoms
    dp = physical_grad + 4 * index
    da = tl.load(dp, valid, 0)
    dR = tl.load(dp + 1, valid, 0)
    co = tl.load(dp + 2, valid, 0)
    ci = tl.load(dp + 3, valid, 0)
    ja = tl.load(jacobian + 2 * index, valid, 0)
    jq = tl.load(jacobian + 2 * index + 1, valid, 0)
    out = raw_grad + 4 * index
    tl.store(out, tl.fma(ja, dR, da), valid)
    tl.store(out + 1, jq * dR, valid)
    tl.store(out + 2, co, valid)
    tl.store(out + 3, ci, valid)


class _RationalDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, radius_min, radius_max, amplitude_max):
        assert p.is_cuda and p.dtype == torch.float32
        assert p.ndim == 2 and p.shape[1] == 4 and p.is_contiguous()
        assert 0.5 <= radius_min <= 4 <= radius_max <= 512 and 0 < amplitude_max <= 32
        encoding_min, encoding_max = np.float32(radius_min), np.float32(radius_max)
        if float(encoding_min) < radius_min:
            encoding_min = np.nextafter(encoding_min, np.float32(math.inf))
        if float(encoding_max) > radius_max:
            encoding_max = np.nextafter(encoding_max, np.float32(0))
        physical = torch.empty_like(p)
        jacobian = torch.empty((len(p), 2), device=p.device, dtype=p.dtype)
        _decode[(triton.cdiv(len(p), 256),)](
            p,
            physical,
            jacobian,
            len(p),
            radius_min,
            radius_max,
            math.log(radius_max),
            float(encoding_min),
            float(encoding_max),
            amplitude_max,
            256,
        )
        ctx.save_for_backward(p, jacobian)
        return physical

    @staticmethod
    def backward(ctx, physical_grad):
        p, jacobian = ctx.saved_tensors
        raw_grad = torch.empty_like(p)
        _vjp[(triton.cdiv(len(p), 256),)](
            physical_grad.contiguous(), jacobian, raw_grad, len(p), 256
        )
        return raw_grad, None, None, None


def decode(p, specification):
    return _RationalDecode.apply(
        p,
        specification.radius_min,
        specification.radius_max,
        specification.amplitude_max,
    )
