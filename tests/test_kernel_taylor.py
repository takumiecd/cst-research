"""Gaussian quadrature/analytic derivative/periodic and Taylor-order checks."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class GaussianTaylorTests(unittest.TestCase):
    def test_gaussian_cell_integrals_match_independent_quadrature(self):
        import math

        import numpy as np
        import torch
        from compare_kernel_taylor import gaussian_axis

        nodes, weights = np.polynomial.legendre.leggauss(64)
        sigma, center = 0.5001, 7.3
        actual = gaussian_axis(
            128,
            torch.tensor(sigma, dtype=torch.float64),
            torch.tensor(center, dtype=torch.float64),
        )
        for site in (0, 5, 7, 8, 10, 127):
            expected = 0.0
            for node, weight in zip(nodes, weights):
                x = site + node / 2
                density = sum(
                    math.exp(-0.5 * ((x - center + k * 128) / sigma) ** 2)
                    / (sigma * math.sqrt(2 * math.pi))
                    for k in range(-4, 5)
                )
                expected += weight * density / 2
            self.assertAlmostEqual(float(actual[site]), expected, delta=2e-15)

    def test_gaussian_normalization_and_periodic_centers(self):
        import torch
        from compare_kernel_taylor import gaussian_axis

        for sigma in (0.25, 0.5001, 2.01):
            s = torch.tensor(sigma, dtype=torch.float64)
            for sites in (128, 256):
                for center in (-0.01, 0.0, 7.3):
                    c = torch.tensor(center, dtype=torch.float64)
                    values = gaussian_axis(sites, s, c)
                    shifted = gaussian_axis(sites, s, c + sites)
                    self.assertAlmostEqual(float(values.sum()), 1.0, delta=3e-15)
                    self.assertTrue(bool((values >= 0).all()))
                    torch.testing.assert_close(values, shifted, atol=2e-14, rtol=1e-12)

    def test_gaussian_autograd_matches_independent_analytic_jvp(self):
        import torch
        from compare_kernel_taylor import gaussian_analytic_jvp, gaussian_weight

        for sigma in (0.25, 0.5001, 2.01):
            point = torch.tensor([[1.0, sigma, 0.0, 13.7]], dtype=torch.float64)
            direction = torch.tensor([[0.2, -0.1, 0.7, -0.3]], dtype=torch.float64)
            _, actual = torch.autograd.functional.jvp(gaussian_weight, point, direction)
            torch.testing.assert_close(
                actual, gaussian_analytic_jvp(point, direction), atol=1e-14, rtol=1e-11
            )

    def test_gaussian_taylor_remainder_is_quadratic_across_fixture_boundaries(self):
        import torch
        from compare_kernel_taylor import fixtures, gaussian_weight, measure

        torch.set_num_threads(1)
        for case in fixtures():
            with self.subTest(case=case["name"]):
                first = measure(gaussian_weight, case, 1e-3, torch.float64, True)
                second = measure(gaussian_weight, case, 1e-4, torch.float64, True)
                self.assertLess(
                    second["absolute_taylor_error"],
                    first["absolute_taylor_error"] * 0.012,
                )
                self.assertGreater(second["prediction_finite_cosine"], 0.99999)

    def test_fp32_quantized_noop_is_not_interpreted_as_continuity(self):
        import torch
        from compare_kernel_taylor import fixtures, gaussian_weight, measure

        record = measure(gaussian_weight, fixtures()[0], 1e-12, torch.float32, True)
        self.assertTrue(record["quantized_noop"])
        self.assertIsNone(record["relative_taylor_error"])
        self.assertIsNone(record["prediction_finite_cosine"])
