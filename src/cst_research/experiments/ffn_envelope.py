"""Exploratory closed-envelope diagnostics and admission; no automatic HPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import torch

EDGES = (0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.6, 1.000001)


def summarize(mask, amplitude, ri, ro, q, lower, upper):
    n = int(mask.sum())
    if not n:
        return {"count": 0}

    def stats(t):
        t = t[mask]
        return {
            "mean": float(t.mean()),
            "p10": float(torch.quantile(t, 0.1)),
            "p90": float(torch.quantile(t, 0.9)),
        }

    return {
        "count": n,
        "absolute_amplitude": stats(amplitude.abs()),
        "input_radius": stats(ri),
        "output_radius": stats(ro),
        "q": stats(q),
        "lower": stats(lower),
        "upper": stats(upper),
    }


@torch.no_grad()
def envelope_snapshot(model):
    """Current-amplitude bins; membership changes over time, empty bins are explicit.

    Compute only after a complete update. There are no RNG, optimizer, gradient
    or model writes. Diagnostics are excluded from synchronized update samples.
    """
    if getattr(getattr(model, "config", None), "radius_mode", None) == "envelope":
        from .ffn_radius_fraction import snapshot

        return snapshot(model)
    from torchcst import CSTLinear
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )
    from torchcst._backends.torch.parameterizations.polar_amp_width import _width_bounds

    result = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, CSTLinear):
            continue
        p = layer.atoms.p.detach().to(device="cpu", dtype=torch.float64)
        a, ri, ro = decoded_values(layer.kernel, p)
        q = ((p[:, :2].square().sum(-1) - 1) / 3).clamp(0, 1)
        _, lo, hi = _width_bounds(
            layer.kernel,
            a,
            scalar_names=layer.kernel.spec.parameterization.width_scalar_names,
        )
        all_rows = torch.ones(len(a), dtype=torch.bool)
        bins = [
            {
                "low": low,
                "high": high,
                **summarize((a.abs() >= low) & (a.abs() < high), a, ri, ro, q, lo, hi),
            }
            for low, high in zip(EDGES[:-1], EDGES[1:])
        ]
        assert sum(b["count"] for b in bins) == len(a)
        result[name] = {"all": summarize(all_rows, a, ri, ro, q, lo, hi), "bins": bins}
    return result


def main(argv=None):
    """Qualify one exact envelope/q recipe, including all eight actual FFN sites."""
    from .ffn_contract import ModelConfig
    from .ffn_model import parameter_seed
    from .ffn_qualification import (
        independent_quadrature_check,
        qualify_case,
        optimizer_and_resume_check,
        task_step_and_resume_check,
    )

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q", type=float, choices=(0.0, 1.0), required=True)
    parser.add_argument(
        "--radius-envelope",
        choices=("rational", "closed_polynomial", "rational_zero_opening"),
        required=True,
    )
    parser.add_argument("--zero-opening-scale", type=float, default=0.03)
    parser.add_argument("--radius-birth", type=float, default=4.0)
    parser.add_argument(
        "--expected-gpu",
        choices=("NVIDIA GeForce RTX 4090", "NVIDIA A100 80GB PCIe MIG 3g.40gb"),
        default="NVIDIA GeForce RTX 4090",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == args.expected_gpu
    config = ModelConfig(
        q=args.q,
        radius_envelope=args.radius_envelope,
        atom_lr_multiplier=16,
        zero_opening_scale=args.zero_opening_scale,
        radius_birth=args.radius_birth,
    )
    envelope = dict(
        radius_envelope=args.radius_envelope,
        closed_lower_power=4.0,
        closed_upper_power=1.0,
        zero_opening_scale=args.zero_opening_scale,
        radius_birth=args.radius_birth,
    )
    result = {
        "state": "running",
        "cohort": "recipe",
        "config": asdict(config),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "cases": [],
        "task_recipes": [config.recipe()],
        "source_sha256": {
            n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
            for n in (
                "ffn_model.py",
                "ffn_contract.py",
                "ffn_qualification.py",
                "ffn_envelope.py",
            )
        },
    }

    def save():
        (args.output / "qualification.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    save()
    try:
        # Small independent quadrature does not need moment matching to exercise
        # boundary atoms; its low target avoids unattainable piecewise targets.
        result["quadrature"] = independent_quadrature_check(
            q=args.q, std=0.001, **envelope
        )
        save()
        for ni, no, std in [(256, 1024, 0.02), (1024, 256, 0.02 / (8**0.5))]:
            result["cases"].append(
                qualify_case(
                    ni,
                    no,
                    128,
                    q=args.q,
                    std=std,
                    seed=parameter_seed(17, "block0.up" if ni < no else "block0.down"),
                    **envelope,
                )
            )
            save()
            print(f"PASS all-Atom native VJP {ni}->{no}", flush=True)
        result["optimizer"] = optimizer_and_resume_check(
            q=args.q, atom_lr_multiplier=16, std=0.001, **envelope
        )
        save()
        result["task_step"] = task_step_and_resume_check(
            q=args.q, atom_lr_multiplier=16, full_task=True, **envelope
        )
        save()
        result["state"] = "passed"
        save()
        print("QUALIFICATION PASSED", flush=True)
    except Exception as e:
        result.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main()
