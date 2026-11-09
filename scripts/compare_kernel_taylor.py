"""CPU diagnostic: Gaussian cell integrals versus frozen Fourier-window profiles.

The Gaussian is an independent mathematical control, not a runtime backend or
speed candidate. No model, optimizer or training data are used.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import torch

SITES = (128, 256)
STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12)
MODES = (4, 8, 16, 32, 64)


def normal_interval(low, high):
    """Stable Phi(high)-Phi(low), avoiding positive-tail subtraction from one."""
    root = math.sqrt(2)
    positive = 0.5 * (torch.erfc(low / root) - torch.erfc(high / root))
    negative = 0.5 * (torch.erfc(-high / root) - torch.erfc(-low / root))
    crossing = 0.5 * (torch.erf(high / root) - torch.erf(low / root))
    return torch.where(low >= 0, positive, torch.where(high <= 0, negative, crossing))


def gaussian_axis(sites, sigma, center, images=4, analytic=False):
    """Wrapped Gaussian integrated over every fixed cell, no moving cutoff.

    Fixed images -4..4 suffice far below FP64 resolution for this study's
    sigma<=2.01 and sites>=128. This is not an arbitrary-width public operator.
    """
    cells = torch.arange(sites, device=sigma.device, dtype=sigma.dtype)
    wraps = torch.arange(-images, images + 1, device=sigma.device, dtype=sigma.dtype)
    offsets = (cells[:, None] + sites * wraps) - center
    low, high = (offsets - 0.5) / sigma, (offsets + 0.5) / sigma
    mass = normal_interval(low, high).sum(1)
    if not analytic:
        return mass
    factor = 1 / math.sqrt(2 * math.pi)
    pdf_low = factor * torch.exp(-0.5 * low.square())
    pdf_high = factor * torch.exp(-0.5 * high.square())
    dcenter = ((pdf_low - pdf_high) / sigma).sum(1)
    dsigma = ((low * pdf_low - high * pdf_high) / sigma).sum(1)
    return mass, dsigma, dcenter


def gaussian_weight(point):
    amplitude, sigma, mu_out, mu_in = point[0]
    out = gaussian_axis(SITES[0], sigma, mu_out)
    inp = gaussian_axis(SITES[1], sigma, mu_in)
    return amplitude * out[:, None] * inp[None, :]


def gaussian_analytic_jvp(point, vector):
    amplitude, sigma, mu_out, mu_in = point[0]
    da, ds, do, di = vector[0]
    out, os, om = gaussian_axis(SITES[0], sigma, mu_out, analytic=True)
    inp, ins, im = gaussian_axis(SITES[1], sigma, mu_in, analytic=True)
    return (
        da * out[:, None] * inp[None, :]
        + amplitude * (ds * os + do * om)[:, None] * inp[None, :]
        + amplitude * out[:, None] * (ds * ins + di * im)[None, :]
    )


def public_fourier(modes, dtype):
    from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels

    layer = CSTLinear(
        chart_presets.regular_grid(SITES[0], spacing=1.0, geometry="flat_torus"),
        chart_presets.regular_grid(SITES[1], spacing=1.0, geometry="flat_torus"),
        atoms=1,
        dtype=dtype,
        backend="factored",
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=modes), merge_factor=2),
            coordinates=kernels.physical(radius_init=(0.5, 1.0)),
        ),
    )

    def weight(point):
        vi, uo = layer.operator.factors(point)
        return uo @ vi.T

    return weight


def route(point):
    """Fixed-study declaration metadata only; never evaluates the kernel."""
    radius = float(point[0, 1])
    result = []
    tile = 1
    while radius > tile:
        tile *= 2
    count = math.ceil(2 * radius / tile) + 1
    for size, k in zip(SITES, (2, 3)):
        relative = float(point[0, k]) % size
        first = math.floor((relative + 0.5) / tile - (count - 1) / 2)
        result.extend((tile, count, first % (size // tile)))
    return result


def fixtures():
    cases = []
    for boundary in (0.5, 1.0, 2.0):
        for reverse in (False, True):
            cases.append(
                {
                    "field": 1,
                    "name": f"radius-{boundary}-{'shrink' if reverse else 'expand'}",
                    "base": [1.0, boundary, 7.3, 13.7],
                    "cross": True,
                    "reverse": reverse,
                }
            )
    for radius, boundary in (
        (0.5, 7.0),
        (0.5001, 7.5),
        (0.75, 7.5),
        (1.5, 7.5),
        (0.75, 0.0),
    ):
        for field in (2, 3):
            point = [1.0, radius, 7.3, 13.7]
            point[field] = boundary
            cases.append(
                {
                    "field": field,
                    "name": f"center-{field}-r{radius}-b{boundary}",
                    "base": point,
                    "cross": True,
                    "reverse": False,
                }
            )
    for field in (1, 2, 3):
        cases.append(
            {
                "field": field,
                "name": f"within-branch-field{field}",
                "base": [1.0, 0.75, 7.3, 13.7],
                "cross": False,
                "reverse": False,
            }
        )
    return cases


def measure(weight, case, step, dtype, gaussian=False):
    start = torch.tensor([case["base"]], dtype=dtype)
    end = start.clone()
    field = case["field"]
    start[:, field] -= step if case["cross"] else 0
    end[:, field] += step
    if case["reverse"]:
        start, end = end, start
    displacement = end - start
    initial, predicted = torch.autograd.functional.jvp(weight, start, displacement)
    actual = weight(end) - initial
    residual = actual - predicted
    actual, predicted, residual = (
        value.double() for value in (actual, predicted, residual)
    )
    actual_norm, predicted_norm, residual_norm = (
        float(value.norm()) for value in (actual, predicted, residual)
    )
    result = {
        "step": step,
        "encoded_displacement": displacement[0].double().tolist(),
        "quantized_noop": not bool(displacement.any()),
        "actual_norm": actual_norm,
        "predicted_norm": predicted_norm,
        "absolute_taylor_error": residual_norm,
        "relative_taylor_error": residual_norm / predicted_norm
        if predicted_norm
        else None,
        "prediction_finite_cosine": float((actual * predicted).sum())
        / (actual_norm * predicted_norm)
        if actual_norm and predicted_norm
        else None,
        "weight_norm": float(initial.double().norm()),
    }
    if gaussian:
        analytic = gaussian_analytic_jvp(start, displacement).double()
        result["analytic_jvp_relative_error"] = float(
            (analytic - predicted).norm()
        ) / max(predicted_norm, 1e-300)
        if dtype == torch.float64 and predicted_norm:
            assert result["analytic_jvp_relative_error"] < 1e-11
    else:
        first, last = route(start), route(end)
        result.update(start_route=first, end_route=last, route_changed=first != last)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = Path(__file__).resolve()
    import torchcst

    operator = Path(torchcst.__file__).resolve().parents[2]
    operator_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=operator, text=True
    ).strip()
    assert operator_commit == "0d277ae6dc0ae14fe515c7559f39535ed0238950"
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=operator, text=True
    ).strip()
    summary = {
        "state": "running",
        "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "consumer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source.parent.parent, text=True
        ).strip(),
        "operator_commit": operator_commit,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "CPU",
        "scope": "One-Atom mathematical Taylor diagnostic only; no optimizer, learning, public backend implementation or speed measurement",
        "sigma_radius_note": "Gaussian sigma and Chebyshev radius use the same numeric values solely to probe identical geometric boundaries; shapes/initial RMS are not matched.",
        "finite_image_control": {
            "images": [-4, 4],
            "sites": SITES,
            "max_sigma": 2.01,
            "moving_cutoff": False,
        },
        "M_note": "ChebyshevSpec M changes the mathematical density, not merely the accuracy of approximating a fixed Gaussian; CPU reference only, no new CUDA mode support.",
        "records": [],
        "training_updates": 0,
    }
    target = args.output / "summary.json"
    save = lambda: target.write_text(json.dumps(summary, indent=2) + "\n")
    save()
    try:
        for dtype in (torch.float64, torch.float32):
            profiles = [
                ("gaussian-fixed-cells", None),
                *[
                    (f"chebyshev-M{m}", m)
                    for m in (MODES if dtype == torch.float64 else (8,))
                ],
            ]
            for profile, modes in profiles:
                function = (
                    gaussian_weight if modes is None else public_fourier(modes, dtype)
                )
                for case in fixtures():
                    rows = [
                        measure(function, case, step, dtype, modes is None)
                        for step in STEPS
                    ]
                    summary["records"].append(
                        {
                            "profile": profile,
                            "modes": modes,
                            "dtype": str(dtype),
                            "case": case,
                            "rows": rows,
                        }
                    )
                save()
                print("PROFILE PASS", profile, dtype, flush=True)
        assert len(summary["records"]) == 152
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest() == summary["script_sha256"]
        )
        summary.update(state="passed", records_count=len(summary["records"]))
        save()
    except BaseException as error:
        summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
