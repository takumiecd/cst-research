"""CPU audit of exact cell-mass identities and geometry-dependent sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels


def audit_relations():
    layer = CSTLinear(
        chart_presets.regular_grid(32, geometry="flat_torus", spacing=1.0),
        chart_presets.regular_grid(32, geometry="flat_torus", spacing=1.0),
        atoms=torch.tensor([[1.0, 1.1, 0.25, 0.25]], dtype=torch.float64),
        dtype=torch.float64,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
            coordinates=kernels.physical(radius_init=(0.5, 8.0), radius_floor=0.5),
        ),
        backend="factored",
    )

    def factors(p):
        vi, uo = layer._factor_atoms(p.reshape(1, 4))
        return torch.cat((vi[:, 0], uo[:, 0]))

    def inspect(amplitude, radius, phase):
        p = torch.tensor([amplitude, radius, phase, phase], dtype=torch.float64)
        factor = factors(p)
        jac = torch.autograd.functional.jacobian(factors, p, vectorize=True)
        vi, uo = factor[:32], factor[32:]
        matrix = uo[:, None] * vi[None, :]
        columns = [
            jac[32:, k, None] * vi[None, :] + uo[:, None] * jac[:32, k][None, :]
            for k in range(4)
        ]
        energy = float(matrix.square().sum())
        expected_energy = float(vi.square().sum() * uo.square().sum())
        torch.testing.assert_close(matrix.sum(), p[0], atol=2e-12, rtol=2e-12)
        assert abs(energy - expected_energy) < 2e-12
        for k in (1, 2, 3):
            torch.testing.assert_close(
                columns[k].sum(), p.new_tensor(0), atol=2e-11, rtol=0
            )
        return {
            "amplitude": amplitude,
            "radius": radius,
            "phase": phase,
            "matrix_sum": float(matrix.sum()),
            "matrix_energy": energy,
            "energy_identity_error": abs(energy - expected_energy),
            "parameter_jacobian_frobenius": [float(c.norm()) for c in columns],
            "geometry_derivative_mass_sums": [
                float(columns[k].sum()) for k in (1, 2, 3)
            ],
            "input_effective_count": float(1 / vi.square().sum())
            if amplitude
            else None,
            "output_effective_count": float(amplitude**2 / uo.square().sum())
            if amplitude
            else None,
        }

    rows = [
        inspect(1.0, r, phase)
        for r in (0.5005, 1.1, 2.3, 4.7)
        for phase in (0.0, 0.25, 0.49)
    ]
    for row in rows:
        assert 1 - 1e-10 <= row["input_effective_count"] <= 32 + 1e-10
        assert 1 - 1e-10 <= row["output_effective_count"] <= 32 + 1e-10
    zero = inspect(0.0, 1.1, 0.25)
    assert zero["parameter_jacobian_frobenius"][0] > 0
    assert zero["parameter_jacobian_frobenius"][1:] == [0.0, 0.0, 0.0]
    # A concrete rank-order reversal: compare mass amplitudes on two shapes.
    narrow = inspect(0.1, 0.5005, 0.0)
    broad = inspect(0.2, 4.7, 0.0)
    reversal = {
        "narrow": narrow,
        "broad": broad,
        "smaller_mass_has_larger_energy": narrow["matrix_energy"]
        > broad["matrix_energy"],
    }
    assert reversal["smaller_mass_has_larger_energy"]

    def unit_shape(geometry):
        p = torch.cat((geometry.new_ones(1), geometry))
        factor = factors(p)
        matrix = factor[32:, None] * factor[:32][None, :]
        return matrix / matrix.norm()

    geometry = torch.tensor([1.1, 0.25, 0.25], dtype=torch.float64)
    normalized = unit_shape(geometry)
    normalized_jac = torch.autograd.functional.jacobian(unit_shape, geometry)
    orthogonality = [
        float((normalized * normalized_jac[:, :, k]).sum()) for k in range(3)
    ]
    assert max(abs(x) for x in orthogonality) < 2e-12
    unit_energy = []
    for row in rows:
        norm = row["matrix_energy"] ** 0.5
        a = 1 / norm
        p = torch.tensor(
            [a, row["radius"], row["phase"], row["phase"]], dtype=torch.float64
        )
        v, u = layer._factor_atoms(p[None, :])
        torch.testing.assert_close(
            (u @ v.T).norm(), p.new_tensor(1.0), atol=2e-12, rtol=2e-12
        )
        unit_energy.append(
            {
                "radius": row["radius"],
                "phase": row["phase"],
                "mass_amplitude_for_unit_energy": a,
            }
        )
    return {
        "state": "passed",
        "scope": "CPU FP64 N32 physical cell-mass M8/base1/merge2. Identities and selected branch derivatives only; no LM training or optimal-radius claim.",
        "torch": torch.__version__,
        "rows": rows,
        "zero_amplitude": zero,
        "amplitude_order_example": reversal,
        "unit_energy": unit_energy,
        "unit_energy_geometry_inner_products": orthogonality,
        "maximum_mass_error": max(abs(x["matrix_sum"] - 1) for x in rows),
        "maximum_energy_identity_error": max(x["energy_identity_error"] for x in rows),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    result = audit_relations()
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "unit_energy")}
        )
    )


if __name__ == "__main__":
    main()
