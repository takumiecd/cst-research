"""Invert candidate decoders to match a common physical Atom initialization."""

import torch


@torch.no_grad()
def invert(decoder, physical_log, guess, iterations=18):
    """Solve the smooth unwrapped chart using damped Newton steps."""
    units = guess.new_tensor([decoder.amplitude_scale, 1.0, 32.0, 32.0])
    target = physical_log / units
    x = guess / units
    h = torch.eye(4, device=x.device, dtype=x.dtype) * 0.0001

    def transform(z):
        return decoder.preclip(z * units) / units

    # Use the unwrapped pre-clip outputs, so residuals have no coordinate seam.
    for _ in range(iterations):
        residual = transform(x) - target
        pending = (residual.abs().amax(-1) > 4e-10).nonzero().flatten()
        if not len(pending):
            break
        current = x[pending]
        jacobian = (
            (
                (transform(current[:, None] + h) - transform(current[:, None] - h))
                / 0.0002
            )
            .transpose(1, 2)
            .contiguous()
        )
        delta, info = torch.linalg.solve_ex(jacobian, residual[pending].unsqueeze(-1))
        delta = delta.squeeze(-1)
        delta = torch.where((info == 0)[:, None] & torch.isfinite(delta), delta, 0.0)
        delta /= delta.abs().amax(1, keepdim=True).clamp_min(1.0)
        fractions = current.new_tensor([0.0, 1.0, 0.5, 0.25, 0.125, 0.0625])
        trials = current[:, None] - delta[:, None] * fractions[None, :, None]
        errors = (transform(trials) - target[pending, None]).square().sum(-1)
        x[pending] = trials[
            torch.arange(len(trials), device=x.device), errors.argmin(1)
        ]
    return x * units


@torch.no_grad()
def bracket_amplitude(decoder, target, initial):
    """Follow a different amplitude branch while solving the other three fields."""

    def conditional(amplitude):
        trial = initial.clone()
        trial[:, 0] = amplitude
        for _ in range(12):
            trial[:, 1:] -= (decoder.preclip(trial) - target)[:, 1:]
        return trial, (decoder.preclip(trial) - target)[:, 0]

    center = initial[:, 0]
    distance = torch.full_like(center, decoder.amplitude_scale * 2)
    lower, upper = center - distance, center + distance
    _, flo = conditional(lower)
    _, fhi = conditional(upper)
    for _ in range(8):
        extend = flo * fhi > 0
        if not bool(extend.any()):
            break
        distance[extend] *= 2
        lower, upper = center - distance, center + distance
        _, flo = conditional(lower)
        _, fhi = conditional(upper)
    for _ in range(38):
        mid = (lower + upper) * 0.5
        _, fm = conditional(mid)
        same = (flo < 0) == (fm < 0)
        lower, upper = torch.where(same, mid, lower), torch.where(same, upper, mid)
        flo = torch.where(same, fm, flo)
    result, _ = conditional((lower + upper) * 0.5)
    return result


@torch.no_grad()
def matched_latents(decoder, target_log, initial, verify_decoder, expected=None):
    """Return float32 latents and physical residuals in the training decoder."""
    restored = []
    for begin in range(0, len(initial), 8192):
        end = begin + 8192
        target = target_log[begin:end]
        reference = initial[begin:end].double()
        result = invert(decoder, target, reference)
        bad = (decoder.preclip(result) - target).abs().amax(1) > 2e-7
        # Try nearby roots for the subset that encountered a nearly flat chart.
        for shift in (-0.6, 0.6, -2.0, 2.0):
            if not bool(bad.any()):
                break
            idx = bad.nonzero().flatten()
            offset = reference[idx].clone()
            offset[:, 0] += shift * decoder.amplitude_scale
            alternative = invert(decoder, target[idx], offset)
            old = (decoder.preclip(result[idx]) - target[idx]).square().sum(1)
            new = (decoder.preclip(alternative) - target[idx]).square().sum(1)
            result[idx] = torch.where((new < old)[:, None], alternative, result[idx])
            bad = (decoder.preclip(result) - target).abs().amax(1) > 2e-7
        if bool(bad.any()):
            result[bad] = bracket_amplitude(decoder, target[bad], reference[bad])
        restored.append(result.to(initial.dtype))
    candidate = torch.cat(restored)
    if expected is None:
        expected = verify_decoder.base(target_log).to(initial.dtype)
    actual = verify_decoder(candidate)
    difference = actual - expected
    difference[:, 1] = (actual[:, 1] / expected[:, 1]).log()
    period = verify_decoder.period
    difference[:, 2:] = (difference[:, 2:] + period / 2).remainder(period) - period / 2
    max_error = difference.abs().amax(0)
    limits = candidate.new_tensor([4e-7, 4e-6, 1.5e-4, 1.5e-4])
    return candidate, max_error, bool((max_error <= limits).all())
