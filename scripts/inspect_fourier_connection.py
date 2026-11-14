"""Small CPU mathematical composition of frozen public Fourier cell integrals.

This is a definition/continuity diagnostic, not a public operator or backend.
No model, optimizer, CUDA implementation or throughput claim is introduced.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import torch
from compare_kernel_taylor import STEPS
from torchcst import ChebyshevSpec
from torchcst._backends.torch.profiles.radius_fourier import (
    integrated_cells,
    support_eps,
)

DELTA = 0.1
UPPER = 0.9
KAPPA = 1.25


def smooth(t):
    left = t.pow(3) * (10 + t * (-15 + 6 * t))
    u = 1 - t
    right = 1 - u.pow(3) * (10 + u * (-15 + 6 * u))
    return torch.where(t <= 0.5, left, right)


def smooth_derivative(t):
    return 30 * t.square() * (1 - t).square()


def validate(delta, upper, kappa):
    if not 0 < delta < 0.5 or not 0 < upper <= 1 - delta or not 1 < kappa < 2:
        raise ValueError("need0<delta<.5,0<upper<=1-delta,1<kappa<2")


def scale_components(radius, upper=UPPER, kappa=KAPPA, detach=False):
    """Return at most two (tile,weight,dweight/dR) triples."""
    tile = 1
    while float(radius.detach()) > upper * tile:
        tile *= 2
    lower = upper / kappa
    if float(radius.detach()) <= lower * tile:
        return [(tile, radius * 0 + 1, radius * 0)]
    t = (torch.log(radius / (lower * tile)) / math.log(kappa)).clamp(0, 1)
    w = smooth(t)
    slope = smooth_derivative(t) / (radius * math.log(kappa))
    if detach:
        w = w.detach()
    return [(tile, 1 - w, -slope), (2 * tile, w, slope)]


def position_components(center, tile, delta=DELTA, hard=False, detach=False):
    """Fixed 3-tile windows: interpolate near the nearest-window boundary."""
    coordinate = (center + 0.5) / tile - 1.5
    first = math.floor(float(coordinate.detach()))
    t = coordinate - first
    if hard:
        chosen = first if float(t.detach()) < 0.5 else first + 1
        return [(chosen, center * 0 + 1, center * 0)]
    if float(t.detach()) <= 0.5 - delta:
        return [(first, center * 0 + 1, center * 0)]
    if float(t.detach()) >= 0.5 + delta:
        return [(first + 1, center * 0 + 1, center * 0)]
    u = ((t - (0.5 - delta)) / (2 * delta)).clamp(0, 1)
    w = smooth(u)
    slope = smooth_derivative(u) / (2 * delta * tile)
    if detach:
        w = w.detach()
    return [(first, 1 - w, -slope), (first + 1, w, slope)]


def bank_cells(radius, center, tile, first, *, size, modes):
    """Compose public integrated_cells on one immutable 3-tile window."""
    if size % tile:
        raise ValueError("study tile must divide chart")
    indices = torch.arange(first * tile, (first + 3) * tile, dtype=torch.int64)
    masses = integrated_cells(
        ChebyshevSpec(modes=modes),
        radius.reshape(1),
        center.reshape(1),
        3.0 * tile,
        indices.reshape(1, -1),
        1.0,
    )[0]
    return torch.zeros(size, dtype=radius.dtype).scatter_add(
        0, indices.remainder(size), masses
    )


def connected(
    point, *, size=128, modes=8, delta=DELTA, upper=UPPER, kappa=KAPPA, variant="full"
):
    radius, center = point
    validate(delta, upper, kappa)
    result = torch.zeros(size, dtype=point.dtype)
    for tile, weight, _ in scale_components(
        radius, upper, kappa, variant == "omit-radius-weight-gradient"
    ):
        for first, position_weight, _ in position_components(
            center,
            tile,
            delta,
            variant == "radius-only",
            variant == "omit-position-weight-gradient",
        ):
            result = result + weight * position_weight * bank_cells(
                radius, center, tile, first, size=size, modes=modes
            )
    return result


def explicit_jvp(
    point, vector, *, size=128, modes=8, delta=DELTA, upper=UPPER, kappa=KAPPA
):
    """Independent outer chain rule; branch partials consume the public reference."""
    radius, center = point
    result = torch.zeros(size, dtype=point.dtype)
    for tile, weight, wr in scale_components(radius, upper, kappa):
        for first, pos_weight, pm in position_components(center, tile, delta):
            f = lambda p, tile=tile, first=first: bank_cells(
                p[0], p[1], tile, first, size=size, modes=modes
            )
            value, partial = torch.autograd.functional.jvp(f, point, vector)
            result += (
                weight * pos_weight * partial
                + (wr * vector[0] * pos_weight + weight * pm * vector[1]) * value
            )
    return result


def geometry(point, *, delta=DELTA, upper=UPPER, kappa=KAPPA):
    radius, center = point
    scales = scale_components(radius, upper, kappa)
    branches = []
    for tile, weight, wr in scales:
        positions = position_components(center, tile, delta)
        for first, pm, _ in positions:
            if float((weight * pm).detach()) == 0:
                continue
            left, right = first * tile - 0.5, (first + 3) * tile - 0.5
            margin = min(float(center.detach()) - left, right - float(center.detach()))
            branches.append(
                {
                    "tile": tile,
                    "first": first,
                    "weight": float((weight * pm).detach()),
                    "core_margin": margin - float(radius.detach()),
                    "radius_tile_ratio": float(radius.detach()) / tile,
                }
            )
    return {
        "active_scales": sum(float(w.detach()) != 0 for _, w, _ in scales),
        "active_windows": len(branches),
        "branches": branches,
    }


def old_public(point, size=128, modes=8):
    from torchcst import CSTLinear, chart_presets, kernels

    # Cached by the runner below; helper kept explicit for source-qualified tests.
    layer = CSTLinear(
        chart_presets.regular_grid(size, spacing=1.0, geometry="flat_torus"),
        chart_presets.regular_grid(size, spacing=1.0, geometry="flat_torus"),
        atoms=1,
        dtype=point.dtype,
        backend="factored",
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=modes)),
            coordinates=kernels.physical(),
        ),
    )

    def axis(p):
        row = torch.stack((p[0] * 0 + 1, p[0], p[1], p[1])).reshape(1, 4)
        _, output = layer.operator.factors(row)
        return output[:, 0]

    return axis


def fixtures():
    lower, upper = UPPER / KAPPA, UPPER
    cases = []
    for tile in (1, 2, 4):
        for label, bound in (("start", lower * tile), ("end", upper * tile)):
            for reverse in (False, True):
                cases.append(
                    {
                        "name": f"radius-{label}-T{tile}-{reverse}",
                        "point": [bound, 7.3],
                        "direction": [1.0, 0.0],
                        "reverse": reverse,
                    }
                )
    for boundary in (0.5, 1.0, 2.0):
        cases.append(
            {
                "name": f"old-radius-{boundary}",
                "point": [boundary, 7.3],
                "direction": [1.0, 0.0],
                "reverse": False,
            }
        )
    for label, center in (("start", 7.4), ("mid", 7.5), ("end", 7.6)):
        cases.append(
            {
                "name": f"position-{label}",
                "point": [0.8, center],
                "direction": [0.0, 1.0],
                "reverse": False,
            }
        )
    cases.append(
        {
            "name": "joint-transition",
            "point": [math.sqrt(lower * upper), 7.5],
            "direction": [0.3, 1.0],
            "reverse": False,
        }
    )
    cases.append(
        {
            "name": "periodic-seam",
            "point": [1.5, 0.0],
            "direction": [0.0, 1.0],
            "reverse": False,
        }
    )
    assert len(cases) == 20
    return cases


def measure(function, case, h, modes):
    base = torch.tensor(case["point"], dtype=torch.float64)
    direction = torch.tensor(case["direction"], dtype=torch.float64)
    start, end = base - h * direction, base + h * direction
    if case["reverse"]:
        start, end = end, start
    displacement = end - start
    initial, predicted = torch.autograd.functional.jvp(function, start, displacement)
    final = function(end)
    assert all(bool(torch.isfinite(t).all()) for t in (initial, predicted, final))
    actual = final - initial
    error = float((actual - predicted).norm())
    return {
        "h": h,
        "actual_norm": float(actual.norm()),
        "predicted_norm": float(predicted.norm()),
        "absolute_taylor_error": error,
        "error_per_parameter_step": error / float(displacement.norm()),
        "prediction_finite_cosine": float((actual * predicted).sum())
        / max(float(actual.norm() * predicted.norm()), 1e-300),
        "mass_error": max(abs(float(initial.sum()) - 1), abs(float(final.sum()) - 1)),
        "minimum_mass": min(float(initial.min()), float(final.min())),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    import torchcst

    script = Path(__file__).resolve()
    operator = Path(torchcst.__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=operator, text=True
    ).strip()
    assert commit == "0d277ae6dc0ae14fe515c7559f39535ed0238950"
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=operator, text=True
    ).strip()
    summary = {
        "state": "running",
        "scope": "CPU mathematical composition of public frozen profile integrals; new definition proposal, no public backend/GPU/model/optimizer/default or performance adoption",
        "operator_commit": commit,
        "consumer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=script.parent.parent, text=True
        ).strip(),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "CPU",
        "parameters": {
            "delta": DELTA,
            "upper": UPPER,
            "kappa": KAPPA,
            "lower": UPPER / KAPPA,
            "bank_count_tiles": 3,
        },
        "records": [],
        "tail_screen": [],
        "geometry": [],
        "training_updates": 0,
    }
    output = args.output / "summary.json"
    save = lambda: output.write_text(json.dumps(summary, indent=2) + "\n")
    save()
    try:
        for modes in (8, 16, 32):
            old = old_public(torch.tensor([1.0, 7.3], dtype=torch.float64), modes=modes)
            variants = (
                (
                    "old",
                    "radius-only",
                    "full",
                    "omit-radius-weight-gradient",
                    "omit-position-weight-gradient",
                )
                if modes == 8
                else ("full",)
            )
            for variant in variants:
                function = (
                    old
                    if variant == "old"
                    else lambda p, modes=modes, variant=variant: connected(
                        p, modes=modes, variant=variant
                    )
                )
                for case in fixtures():
                    rows = [measure(function, case, h, modes) for h in STEPS]
                    summary["records"].append(
                        {"modes": modes, "variant": variant, "case": case, "rows": rows}
                    )
                save()
                print("CONNECTION PASS", modes, variant, flush=True)
        for kappa in (1.05, 1.1, 1.25, 1.4):
            for modes in (8, 16, 32):
                for i in range(41):
                    r = 0.5 * (4 / 0.5) ** (i / 40)
                    radius = torch.tensor(r, dtype=torch.float64)
                    components = scale_components(radius, UPPER, kappa)
                    leakage = sum(
                        weight
                        * support_eps(
                            ChebyshevSpec(modes=modes), radius.reshape(1), 3.0 * tile
                        )[0]
                        for tile, weight, _ in components
                    )
                    summary["tail_screen"].append(
                        {
                            "kappa": kappa,
                            "modes": modes,
                            "radius": r,
                            "tail_mass": float(leakage),
                            "max_scale_weight_slope": max(
                                abs(float(wr)) for _, _, wr in components
                            ),
                        }
                    )
        for r in (0.5, 0.72, 0.8, 0.9, 1.0, 1.5, 3.0):
            for center in (0.0, 0.4, 0.5, 0.6, 7.3, 7.5, 127.9):
                value = geometry(torch.tensor([r, center], dtype=torch.float64))
                assert value["active_scales"] <= 2 and value["active_windows"] <= 4
                assert min(b["core_margin"] for b in value["branches"]) >= -1e-12
                summary["geometry"].append({"radius": r, "center": center, **value})
        assert len(summary["records"]) == 140
        summary.update(state="passed", records_count=140, finite_displacements=1120)
        save()
    except BaseException as error:
        summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
