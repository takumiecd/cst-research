"""Connection-law, public-integral, periodic and missing-chain-rule controls."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torchcst"), "optional frozen operator source"
)
class FourierConnectionTests(unittest.TestCase):
    def test_quintic_endpoint_derivatives_and_convexity(self):
        import torch
        from inspect_fourier_connection import smooth

        for endpoint in (0.0, 1.0):
            t = torch.tensor(endpoint, dtype=torch.float64, requires_grad=True)
            value = smooth(t)
            first = torch.autograd.grad(value, t, create_graph=True)[0]
            second = torch.autograd.grad(first, t)[0]
            self.assertEqual(float(value.detach()), endpoint)
            self.assertEqual(float(first.detach()), 0)
            self.assertEqual(float(second), 0)
        t = torch.linspace(0, 1, 1001, dtype=torch.float64)
        self.assertTrue(bool(((smooth(t) >= 0) & (smooth(t) <= 1)).all()))
        torch.testing.assert_close(smooth(t), 1 - smooth(1 - t), atol=5e-16, rtol=1e-14)

    def test_outer_chain_rule_and_periodic_normalization(self):
        import torch
        from inspect_fourier_connection import connected, explicit_jvp, geometry

        for size, radius, center in ((128, 0.8, 7.5), (8, 1.6, 7.5), (128, 3.2, -0.1)):
            point = torch.tensor([radius, center], dtype=torch.float64)
            direction = torch.tensor([0.3, 1.0], dtype=torch.float64)
            fun = lambda p, size=size: connected(p, size=size)
            value, derivative = torch.autograd.functional.jvp(fun, point, direction)
            torch.testing.assert_close(
                derivative,
                explicit_jvp(point, direction, size=size),
                atol=1e-13,
                rtol=1e-11,
            )
            self.assertAlmostEqual(float(value.sum()), 1, delta=1e-12)
            self.assertGreaterEqual(float(value.min()), -1e-13)
            shift = point + torch.tensor([0.0, size], dtype=point.dtype)
            torch.testing.assert_close(fun(shift), value, atol=2e-13, rtol=1e-11)
            meta = geometry(point)
            self.assertLessEqual(meta["active_scales"], 2)
            self.assertLessEqual(meta["active_windows"], 4)
            self.assertGreaterEqual(
                min(b["core_margin"] for b in meta["branches"]), -1e-12
            )

    def test_fixed_window_integrals_match_independent_quadrature(self):
        import math

        import numpy as np
        import torch
        from inspect_fourier_connection import bank_cells
        from torchcst import ChebyshevSpec
        from torchcst._backends.torch.profiles.radius_fourier import (
            density_coefficients,
        )

        radius = torch.tensor(1.6, dtype=torch.float64)
        center = torch.tensor(7.5, dtype=torch.float64)
        # P12 on an8-site chart exercises repeated images, not clipped indices.
        actual = bank_cells(radius, center, 4, 0, size=8, modes=8)
        coefficient = density_coefficients(
            ChebyshevSpec(modes=8), radius.reshape(1), 12.0
        )[0].numpy()
        nodes, weights = np.polynomial.legendre.leggauss(32)
        expected = np.zeros(8)
        for j in range(12):
            x = j + nodes / 2 - float(center)
            values = coefficient[8] + 2 * sum(
                coefficient[8 + k] * np.cos(2 * math.pi * k * x / 12)
                for k in range(1, 9)
            )
            expected[j % 8] += float(np.dot(weights, values) / 2)
        torch.testing.assert_close(
            actual, torch.tensor(expected), atol=2e-14, rtol=1e-11
        )

    def test_taylor_order_across_new_endpoints_and_old_boundaries(self):
        from inspect_fourier_connection import connected, fixtures, measure

        for case in fixtures():
            with self.subTest(case=case["name"]):
                coarse = measure(connected, case, 1e-3, 8)
                fine = measure(connected, case, 1e-4, 8)
                self.assertLess(
                    fine["absolute_taylor_error"],
                    coarse["absolute_taylor_error"] * 0.025 + 5e-14,
                )
                self.assertLess(fine["mass_error"], 1e-12)

    def test_omitted_radius_weight_derivative_leaves_linear_error(self):
        from inspect_fourier_connection import connected, fixtures, measure

        case = next(c for c in fixtures() if c["name"] == "joint-transition")
        function = lambda p: connected(p, variant="omit-radius-weight-gradient")
        coarse = measure(function, case, 1e-4, 8)
        fine = measure(function, case, 1e-5, 8)
        ratio = fine["absolute_taylor_error"] / coarse["absolute_taylor_error"]
        self.assertGreater(ratio, 0.08)
        self.assertLess(ratio, 0.12)
        correct = measure(connected, case, 1e-5, 8)
        self.assertLess(
            correct["absolute_taylor_error"], fine["absolute_taylor_error"] * 0.02
        )
