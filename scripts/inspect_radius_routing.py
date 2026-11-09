"""Deterministic CPU witness of the frozen radius kernel's routing continuity."""

import argparse
import json
from pathlib import Path

import torch
from analyze_lm_tangent_history import route_signature
from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels
from torchcst._backends.torch.parameterizations.radius_envelope import from_state


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layer = CSTLinear(
        chart_presets.regular_grid(128, spacing=1.0, geometry="flat_torus"),
        chart_presets.regular_grid(256, spacing=1.0, geometry="flat_torus"),
        atoms=1,
        dtype=torch.float64,
        backend="factored",
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
            coordinates=kernels.radius_envelope(
                radius_min=0.5,
                radius_max=50.0,
                tau=0.003,
                radius_fraction_init=0.0,
                amplitude_max=32.0,
            ),
        ),
    )
    results = []
    for dtype in (torch.float64, torch.float32):
        p = torch.tensor([[1.0, 0.0, 7.3, 13.7]], dtype=dtype)

        def weight(point):
            vi, uo = layer.operator.factors(point)
            return uo @ vi.T

        direction = torch.zeros_like(p)
        direction[:, 1] = 1
        old, derivative = torch.autograd.functional.jvp(weight, p, direction)
        rows = []
        for epsilon in (0.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
            point = p + epsilon * direction
            value = weight(point)
            delta = value - old
            rows.append(
                {
                    "epsilon": epsilon,
                    "radius": float(from_state(layer.kernel, point)[0, 1]),
                    "routing": route_signature(layer, point)[0].tolist(),
                    "weight_change_frobenius": float(delta.double().norm()),
                    "predicted_change_frobenius": epsilon
                    * float(derivative.double().norm()),
                }
            )
        results.append(
            {
                "dtype": str(dtype),
                "derivative_frobenius": float(derivative.double().norm()),
                "rows": rows,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "Single-Atom deterministic CPU public-factor continuity witness; no training or kernel modification",
                "parameters": {
                    "amplitude": 1.0,
                    "radius_fraction": 0.0,
                    "output_center": 7.3,
                    "input_center": 13.7,
                    "radius_min": 0.5,
                    "radius_max": 50.0,
                    "tau": 0.003,
                    "modes": 8,
                    "base_tile_sites": 1,
                },
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    for result in results:
        print(result["dtype"], result["rows"][-1])


if __name__ == "__main__":
    main()
