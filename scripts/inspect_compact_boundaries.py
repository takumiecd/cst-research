"""CPU boundary diagnostic for public compact profiles and integral controls.

Consumes the frozen public Torch profile evaluator. The Triweight cell integral
is an independent mathematical control, not a runtime implementation or Plan.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import torch
from compare_kernel_taylor import STEPS, gaussian_axis

SITES = 128
FLOOR = 1e-6


def public_profile(shape, normalized=False, dtype=torch.float64):
    from torchcst import (
        BiweightSpec,
        TriweightSpec,
        WendlandC2Spec,
        chart_presets,
        kernels,
    )
    from torchcst._backends.torch.profiles import execution
    from torchcst.charts import compile_chart
    from torchcst.kernels.state import ProfileState

    types = {
        "triweight": TriweightSpec,
        "biweight": BiweightSpec,
        "wendland_c2": WendlandC2Spec,
    }
    spec = types[shape]()
    binding = (
        kernels.l2(spec, sigma=1.0, floor=FLOOR)
        if normalized
        else kernels.point(spec, sigma=1.0)
    )
    state = ProfileState(binding).to(dtype=dtype)
    chart = compile_chart(
        chart_presets.regular_grid(SITES, spacing=1.0, geometry="flat_torus"),
        dtype=dtype,
    )

    def values(point):
        radius, center = point
        return execution.evaluate_with_precision(
            state, chart, center.reshape(1, 1), radius.reciprocal().square().reshape(1)
        )[:, 0]

    return values


def triweight_cells(point):
    """Normalized compact density integrated over fixed cells, periodic images.

    Endpoint-tail form avoids cancellation of tiny newly entered cell masses.
    No L2 normalization, adaptive period or moving integration window is used.
    """
    radius, center = point
    sites = torch.arange(SITES, dtype=point.dtype, device=point.device)[:, None]
    images = torch.arange(-2, 3, dtype=point.dtype, device=point.device)
    offset = (sites + SITES * images) - center
    low = ((offset - 0.5) / radius).clamp(-1, 1)
    high = ((offset + 0.5) / radius).clamp(-1, 1)
    constant = 35 / 32

    def tail(u):
        v = 1 - u.clamp(0, 1)
        return constant * v.pow(4) * (2 - 12 / 5 * v + v.square() - v.pow(3) / 7)

    def primitive(u):
        return constant * (u - u.pow(3) + 3 / 5 * u.pow(5) - u.pow(7) / 7)

    positive = tail(low) - tail(high)
    negative = tail(-high) - tail(-low)
    crossing = primitive(high) - primitive(low)
    return torch.where(
        low >= 0, positive, torch.where(high <= 0, negative, crossing)
    ).sum(1)


def gaussian_cells(point):
    return gaussian_axis(SITES, point[0], point[1])


def fixtures():
    cases = []

    def add(name, radius, center, field, cross=True, reverse=False):
        cases.append(
            {
                "name": name,
                "point": [radius, center],
                "field": field,
                "cross": cross,
                "reverse": reverse,
            }
        )

    for radius in (0.5, 1.0, 2.0):
        for reverse in (False, True):
            add(
                f"hierarchy-radius-{radius}-{'shrink' if reverse else 'expand'}",
                radius,
                7.3,
                0,
                reverse=reverse,
            )
    for reverse in (False, True):
        add(f"support-radius-1-{reverse}", 1.0, 7.0, 0, reverse=reverse)
        add(f"empty-birth-radius-0.5-{reverse}", 0.5, 7.5, 0, reverse=reverse)
        add(f"support-center-7-{reverse}", 1.0, 7.0, 1, reverse=reverse)
        add(f"half-center-7.5-{reverse}", 0.5, 7.5, 1, reverse=reverse)
    add("old-center-window-7.5", 0.5001, 7.5, 1)
    for field in (0, 1):
        add(f"within-field-{field}", 0.75, 7.3, field, cross=False)
        add(f"empty-interior-field-{field}", 0.4, 7.5, field, cross=False)
    add("periodic-center-0", 0.75, 0.0, 1)
    gap = FLOOR ** (1 / 3)
    for reverse in (False, True):
        add(
            f"floor-center-{reverse}", 0.4, 0.4 * math.sqrt(1 - gap), 1, reverse=reverse
        )
        add(
            f"floor-radius-{reverse}", 0.4 / math.sqrt(1 - gap), 0.4, 0, reverse=reverse
        )
    assert len(cases) == 24
    return cases


def inspect(function, raw_triweight, case, step, dtype):
    start = torch.tensor(case["point"], dtype=dtype)
    end = start.clone()
    start[case["field"]] -= step if case["cross"] else 0
    end[case["field"]] += step
    if case["reverse"]:
        start, end = end, start
    vector = end - start
    initial, predicted = torch.autograd.functional.jvp(function, start, vector)
    final = function(end)
    assert all(bool(torch.isfinite(x).all()) for x in (initial, predicted, final))
    change = (final - initial).double()
    predicted = predicted.double()
    residual = change - predicted
    actual_norm, predicted_norm, error = (
        float(x.norm()) for x in (change, predicted, residual)
    )
    return {
        "step": step,
        "start": start.double().tolist(),
        "end": end.double().tolist(),
        "quantized_noop": not bool(vector.any()),
        "initial_norm": float(initial.double().norm()),
        "final_norm": float(final.double().norm()),
        "actual_norm": actual_norm,
        "predicted_norm": predicted_norm,
        "absolute_taylor_error": error,
        "relative_taylor_error": error / predicted_norm if predicted_norm else None,
        "prediction_finite_cosine": float((change * predicted).sum())
        / (actual_norm * predicted_norm)
        if actual_norm and predicted_norm
        else None,
        "raw_triweight_initial_norm": float(raw_triweight(start).double().norm()),
        "raw_triweight_final_norm": float(raw_triweight(end).double().norm()),
        "raw_triweight_initial_support": int((raw_triweight(start) != 0).sum()),
        "raw_triweight_final_support": int((raw_triweight(end) != 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    import torchcst

    operator = Path(torchcst.__file__).resolve().parents[2]
    op_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=operator, text=True
    ).strip()
    assert op_commit == "0d277ae6dc0ae14fe515c7559f39535ed0238950"
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=operator, text=True
    ).strip()
    script = Path(__file__).resolve()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    summary = {
        "state": "running",
        "scope": "CPU profile boundary analysis only; no model, optimizer, GPU, runtime addition, speed or learning experiment",
        "operator_commit": op_commit,
        "consumer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=script.parent.parent, text=True
        ).strip(),
        "script_sha256": sha(script),
        "gaussian_helper_sha256": sha(script.with_name("compare_kernel_taylor.py")),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "CPU",
        "sites": SITES,
        "normalization_floor": FLOOR,
        "training_updates": 0,
        "notes": [
            "Point/raw, discrete L2 and unit-mass cell integrals are different contracts; widths/shapes/norms are not matched.",
            "Vanishing predicted directions at support edges make relative ratios/cosines insufficient; examine absolute errors and convergence orders.",
            "CPU profile evaluation does not establish a CUDA Plan or speed result.",
        ],
        "records": [],
    }
    target = args.output / "summary.json"
    save = lambda: target.write_text(json.dumps(summary, indent=2) + "\n")
    save()
    try:
        for dtype in (torch.float64, torch.float32):
            raw = public_profile("triweight", dtype=dtype)
            profiles = {
                "triweight-point": raw,
                "triweight-l2": public_profile("triweight", True, dtype),
                "triweight-cell-control": triweight_cells,
                "gaussian-cell-control": gaussian_cells,
            }
            if dtype == torch.float64:
                profiles.update(
                    {
                        "biweight-point": public_profile("biweight", dtype=dtype),
                        "wendland-c2-point": public_profile("wendland_c2", dtype=dtype),
                    }
                )
            for name, function in profiles.items():
                for case in fixtures():
                    rows = [inspect(function, raw, case, h, dtype) for h in STEPS]
                    summary["records"].append(
                        {
                            "profile": name,
                            "dtype": str(dtype),
                            "case": case,
                            "rows": rows,
                        }
                    )
                save()
                print("PROFILE PASS", name, dtype, flush=True)
        assert len(summary["records"]) == 240
        assert sha(script) == summary["script_sha256"]
        summary.update(state="passed", records_count=240, finite_displacements=1920)
        save()
    except BaseException as error:
        summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
