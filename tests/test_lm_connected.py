"""Read-only diagnostic arithmetic, with no GPU required."""

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
    import torchcst  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional Torch/torchcst")
class ConnectedDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/analyze_lm_connected.py"
        spec = importlib.util.spec_from_file_location("connected_diagnostic", path)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_runner_import_is_gpu_free(self):
        from cst_research.experiments import lm_connected_run

        self.assertTrue(callable(lm_connected_run.make_optimizer))

    def test_connected_chain_predicts_fraction_step_across_old_radius_boundary(self):
        from dataclasses import replace

        from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels
        from torchcst._backends.torch.parameterizations.radius_envelope import (
            physical_parameters,
        )
        from torchcst.kernels.spec import RadiusConnectionSpec

        from cst_research.experiments.lm_connected import physical_reference

        kernel = kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=16)),
            coordinates=kernels.radius_envelope(
                radius_min=0.5, radius_max=50.0, tau=0.003, amplitude_max=32.0
            ),
        )
        kernel = replace(
            kernel,
            revision=10,
            profiles=tuple(
                replace(p, connection=RadiusConnectionSpec()) for p in kernel.profiles
            ),
        )
        layer = CSTLinear(
            chart_presets.regular_grid(256, spacing=1.0, geometry="flat_torus"),
            chart_presets.regular_grid(128, spacing=1.0, geometry="flat_torus"),
            atoms=1,
            kernel=kernel,
            dtype=torch.float32,
            backend="factored",
        )
        reference = physical_reference(layer.execution_declaration())
        x = torch.randn(
            7, 128, dtype=torch.float64, generator=torch.Generator().manual_seed(9)
        )

        def action(p):
            physical = physical_parameters(
                p, radius_min=0.5, radius_max=50.0, tau=0.003, amplitude_max=32.0
            ).double()
            vi, vo = reference.operator.factors(physical)
            return (x @ vi) @ vo.T

        p = torch.tensor([[1.0, 0.0, 7.3, 13.7]])
        v = torch.tensor([[0.0, 1e-4, 0.0, 0.0]])
        base, j = torch.autograd.functional.jvp(action, p, v)
        finite = action(p + v) - base
        stat = self.m.statistics(finite, j, -finite)
        self.assertGreater(stat["linear_finite_cosine"], 0.999)
        self.assertLess(stat["relative_defect_to_prediction"], 0.01)

    def test_wrapped_delta_and_exact_field_endpoint(self):
        old = torch.tensor([[0.3, 0.0, 127.9, 255.9]])
        new = torch.tensor([[0.4, 0.2, 0.1, 0.1]])
        v = self.m.displacement(old, new, 256, 128)
        self.assertTrue(torch.all(v[:, 2:].abs() < 0.3))
        for k in range(4):
            point = self.m.candidate(old, new, v, k, 1.0, 256, 128)
            self.assertTrue(torch.equal(point[:, k], new[:, k]))
            for j in range(4):
                if j != k:
                    self.assertTrue(torch.equal(point[:, j], old[:, j]))

    def test_linear_agreement_and_zero_direction(self):
        d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        s = self.m.statistics(d, d, -d)
        self.assertAlmostEqual(s["linear_finite_cosine"], 1.0)
        self.assertEqual(s["relative_defect_to_prediction"], 0.0)
        s = self.m.statistics(d, torch.zeros_like(d), -d)
        self.assertIsNone(s["linear_finite_cosine"])
        self.assertIsNone(s["finite_over_prediction_norm"])


if __name__ == "__main__":
    unittest.main()
