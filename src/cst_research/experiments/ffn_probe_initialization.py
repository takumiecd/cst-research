"""One-time amplitude initialization matching actual FFN probe output RMS.

No persistent gain, normalization layer, label-dependent tuning or runtime map.
"""

import copy
import math

import torch


def output_rms(value):
    return float(value.detach().float().square().mean().sqrt())


@torch.no_grad()
def capture_rms(model, x):
    values, handles = {}, []
    for name, module in model.named_modules():
        if name.endswith((".ffn.up", ".ffn.down")):

            def hook(_module, _inputs, output, name=name):
                values[name] = output_rms(output)

            handles.append(module.register_forward_hook(hook))
    try:
        model(x)
        return values
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def match_projection_rms(projection, inputs, target, *, amplitude_max, tolerance=0.001):
    if not math.isfinite(target) or target <= 0:
        raise ValueError("positive finite probe RMS target required")
    p = projection.map.atoms.p
    if not bool((projection.bias == 0).all()):
        raise ValueError("probe initialization requires unchanged zero projection bias")
    base = p[:, 0].clone()
    history = []

    def trial(scale):
        p[:, 0].copy_((base * scale).clamp(-amplitude_max, amplitude_max))
        value = output_rms(projection(inputs))
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite calibration output RMS")
        history.append({"scale": scale, "output_rms": value})
        return value

    try:
        before = trial(1.0)
        low, high = 0.0, 1.0
        value = before
        while value < target and len(history) < 12:
            low = high
            high *= 2
            value = trial(high)
        if value < target:
            raise ValueError("probe RMS target exceeds bounded amplitude search")
        scale = 1.0
        if abs(before / target - 1) > tolerance:
            for _ in range(28):
                scale = (low + high) / 2
                value = trial(scale)
                if abs(value / target - 1) <= tolerance:
                    break
                if value < target:
                    low = scale
                else:
                    high = scale
            else:
                raise ValueError(
                    f"probe RMS bisection did not reach target {target}; tail={history[-6:]}"
                )
        else:
            value = trial(1.0)
        return {
            "target_output_rms": target,
            "before_output_rms": before,
            "after_output_rms": value,
            "relative_error": abs(value / target - 1),
            "amplitude_multiplier": scale,
            "amplitude_rms_after": output_rms(p[:, 0]),
            "amplitude_at_cap_fraction": float(
                (p[:, 0].abs() == amplitude_max).float().mean()
            ),
            "history": history,
        }
    except BaseException:
        p[:, 0].copy_(base)
        raise


@torch.no_grad()
def calibrate_model(model, x, *, second_probe=None):
    """Greedy execution order keeps earlier matched outputs unchanged."""
    from .ffn_model import LanguageModel

    dense = LanguageModel(model.config, "dense", model.seed).to(x.device)
    dense.eval()
    targets = capture_rms(dense, x)
    second_targets = (
        capture_rms(dense, second_probe) if second_probe is not None else None
    )
    del dense
    sites = {
        name: module
        for name, module in model.named_modules()
        if name.endswith((".ffn.up", ".ffn.down"))
    }
    assert len(sites) == len(targets) == 8
    saved = {
        name: (
            site.map.atoms.p.detach().clone(),
            copy.deepcopy(site.map.initialization_report),
        )
        for name, site in sites.items()
    }
    before = capture_rms(model, x)
    result = {
        "method": "train-window-output-rms",
        "tolerance": 0.001,
        "calibration_uses_labels": False,
        "runtime_gain_added": False,
        "targets": targets,
        "before": before,
        "sites": {},
    }
    try:
        for name, site in sites.items():
            captured = []

            def capture(_module, inputs, captured=captured):
                captured.append(inputs[0].detach())

            handle = site.register_forward_pre_hook(capture)
            try:
                model(x)
            finally:
                handle.remove()
            assert len(captured) == 1
            record = match_projection_rms(
                site,
                captured[0],
                targets[name],
                amplitude_max=model.config.amplitude_max,
            )
            assert torch.equal(site.map.atoms.p[:, 1:], saved[name][0][:, 1:])
            site.map.initialization_report = {
                "scheme": "train_probe_rms",
                "moment_seed_report": saved[name][1],
                "radius_fraction": model.config.radius_fraction,
                **record,
            }
            result["sites"][name] = record
        result["after"] = capture_rms(model, x)
        assert all(
            abs(result["after"][name] / targets[name] - 1) <= 0.001 for name in targets
        )
        if second_probe is not None:
            result["second_train_window_targets"] = second_targets
            result["second_train_window_observed"] = capture_rms(model, second_probe)
        return result
    except BaseException:
        for name, site in sites.items():
            site.map.atoms.p.copy_(saved[name][0])
            site.map.initialization_report = saved[name][1]
        raise
