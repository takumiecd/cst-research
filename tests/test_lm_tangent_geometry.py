"""Independent small Jacobians and known projections for analysis geometry."""

import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class TangentGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "lm_tangent_geometry",
            Path(__file__).resolve().parents[1] / "scripts/lm_tangent_geometry.py",
        )
        cls.math = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.math)

    def test_cgls_known_projection_and_conditioned_gradient_are_distinct(self):
        import torch

        # J covers both first axes but is highly anisotropic. Raw gradient can
        # align badly even though the target is entirely in range(J).
        matrix = torch.tensor([[100.0, 0], [0, 1], [0, 0]], dtype=torch.float64)

        class Operator:
            size = 2

            def jvp(self, v):
                return matrix @ v

            def vjp(self, g):
                return matrix.T @ g

            def column_norms(self):
                return matrix.square().sum(0).sqrt()

        target = torch.tensor([1.0, 10.0, 3.0], dtype=torch.float64)
        op = Operator()
        initial = op.vjp(target)
        result, audit = self.math.tangent_projection(
            op, target, initial, max_iterations=8, tolerance=1e-9
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0, 10.0, 0.0], dtype=torch.float64)
        )
        self.assertTrue(audit["converged"])
        self.assertLess(self.math.cosine(op.jvp(initial), target), 0.2)
        self.assertGreater(self.math.cosine(result, target), 0.95)

    def test_factor_actions_match_explicit_nonlinear_jacobian(self):
        import torch

        torch.manual_seed(3)
        p = torch.randn(3, 4, dtype=torch.float64)
        x = torch.randn(7, 2, dtype=torch.float64)

        def factors(point):
            vi = torch.stack((point[:, 0].sin(), point[:, 1].exp()))
            uo = torch.stack((point[:, 2].square(), point[:, 3].cos()))
            return vi, uo

        vi, uo = factors(p)
        dis, dos = [], []
        for k in range(4):
            v = torch.zeros_like(p)
            v[:, k] = 1
            _, (a, b) = torch.autograd.functional.jvp(factors, p, v)
            dis.append(a)
            dos.append(b)
        op = self.math.FactorTangent(x, vi, uo, dis, dos)
        vector = torch.randn(op.size, dtype=torch.float64)
        explicit = torch.autograd.functional.jacobian(
            lambda point: (x @ factors(point)[0]) @ factors(point)[1].T, p
        ).reshape(14, 12)
        bias_jac = torch.eye(2, dtype=torch.float64).repeat(7, 1)
        jac = torch.cat((explicit, bias_jac), 1)
        torch.testing.assert_close(op.jvp(vector).flatten(), jac @ vector)
        goal = torch.randn(7, 2, dtype=torch.float64)
        torch.testing.assert_close(op.vjp(goal), jac.T @ goal.flatten())
        torch.testing.assert_close(op.column_norms(), jac.square().sum(0).sqrt())
        mask = torch.tensor([True, False, True])
        pieces = (
            op.atom_subset_jvp(vector, mask)
            + op.atom_subset_jvp(vector, ~mask)
            + vector[-2:]
        )
        torch.testing.assert_close(pieces, op.jvp(vector))

    def test_free_and_masked_known_subspaces(self):
        import torch

        x = torch.tensor([[-1.0, 0], [0, 1], [1, 0]], dtype=torch.float64)
        target = torch.tensor([[1.0, 0], [2, 1], [3, 0]], dtype=torch.float64)
        full, _ = self.math.free_projection(x, target)
        torch.testing.assert_close(full, target)
        mask = torch.tensor([[True, False], [True, False]])
        restricted, _ = self.math.masked_projection(x, target, mask)
        expected = torch.stack(
            (torch.tensor([1.0, 2, 3]), torch.tensor([1 / 3, 1 / 3, 1 / 3])), 1
        ).double()
        torch.testing.assert_close(restricted, expected)

    @unittest.skipUnless(
        importlib.util.find_spec("torchcst"), "optional operator source"
    )
    def test_public_radius_factor_jvps_match_direct_vjp(self):
        import torch
        from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels

        layer = CSTLinear(
            chart_presets.regular_grid(32, spacing=1.0, geometry="flat_torus"),
            chart_presets.regular_grid(64, spacing=1.0, geometry="flat_torus"),
            atoms=7,
            dtype=torch.float32,
            backend="factored",
            kernel=kernels.product(
                kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
                coordinates=kernels.radius_envelope(
                    radius_min=0.5,
                    radius_max=8.0,
                    tau=0.03,
                    radius_fraction_init=0.25,
                    amplitude_max=32.0,
                ),
            ),
        )
        p = layer.atoms.p.detach().clone()
        x = torch.randn(11, 64)
        g = torch.randn(11, 32)
        vi, uo, di, do = self.math.factor_partials(layer, p, chunk=4)
        op = self.math.FactorTangent(x, vi, uo, di, do)
        point = p.requires_grad_()
        a, b = layer.operator.factors(point)
        expected = torch.autograd.grad((x @ a) @ b.T, point, g)[0]
        torch.testing.assert_close(
            op.vjp(g)[:-32].reshape_as(p), expected, atol=1e-5, rtol=1e-4
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torchcst"), "optional operator source"
    )
    def test_field_atom_scores_sum_to_task_output_inner_product(self):
        from types import SimpleNamespace

        import torch
        from analyze_lm_tangent_history import analyze_fields, atom_score_summary
        from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels

        torch.manual_seed(73)
        layer = CSTLinear(
            chart_presets.regular_grid(16, spacing=1.0, geometry="flat_torus"),
            chart_presets.regular_grid(32, spacing=1.0, geometry="flat_torus"),
            atoms=3,
            dtype=torch.float64,
            backend="factored",
            kernel=kernels.product(
                kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
                coordinates=kernels.radius_envelope(
                    radius_min=0.5,
                    radius_max=8.0,
                    tau=0.03,
                    radius_fraction_init=0.25,
                    amplitude_max=32.0,
                ),
            ),
        )
        point = layer.atoms.p.detach().clone()
        x, g = (
            torch.randn(7, 32, dtype=torch.float64),
            torch.randn(7, 16, dtype=torch.float64),
        )
        factors = self.math.factor_partials(layer, point)
        op = self.math.FactorTangent(x, *factors)
        actual = torch.randn(op.size, dtype=torch.float64) * 1e-4
        row = {
            "x": x,
            "g": g,
            "old_p": point,
            "actual_vector": actual,
            "bias_delta": torch.zeros(16, dtype=torch.float64),
            "vectors": {"current_gradient": -op.vjp(g)},
        }
        result = analyze_fields(SimpleNamespace(map=layer), row, factors)
        self.assertEqual(len(result["fields"]), 5)
        for field in result["fields"].values():
            self.assertGreaterEqual(field["changed_route_fraction"], 0)
            self.assertEqual(
                field["history_atom_linear_scores"]["current_gradient"][
                    "negative_atom_fraction"
                ],
                0,
            )
        known = atom_score_summary(torch.tensor([2.0, -1.0, 0.0]))
        self.assertEqual(known["signed_sum"], 1.0)
        self.assertAlmostEqual(known["negative_absolute_score_share"], 1 / 3)

    def test_history_factorial_has_exact_no_history_limit(self):
        import torch

        g = torch.tensor([2.0, -3.0], dtype=torch.float64)
        v = self.math.adam_directions(
            g, g * 0.1, g.square() * 0.05, 1, torch.ones_like(g)
        )
        torch.testing.assert_close(
            v["historical_m_historical_v"], v["current_g_current_g2"]
        )
        torch.testing.assert_close(
            v["current_g_historical_v"], v["current_g_current_g2"]
        )
