"""Independent compact support, clamped-normalization and exact integral checks."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torchcst"), "optional frozen operator source"
)
class CompactBoundaryTests(unittest.TestCase):
    def test_public_triweight_and_support_derivatives_match_polynomial(self):
        import torch
        from inspect_compact_boundaries import public_profile

        profile = public_profile("triweight")
        point = torch.tensor([1.0, 7.0], dtype=torch.float64, requires_grad=True)
        value = profile(point)
        self.assertEqual(float(value[8].detach()), 0)
        self.assertEqual(float(value[7].detach()), 1)
        gradient = torch.autograd.grad(value[8], point, create_graph=True)[0]
        self.assertTrue(torch.equal(gradient, torch.zeros_like(point)))
        hessian = torch.stack(
            [torch.autograd.grad(v, point, retain_graph=True)[0] for v in gradient]
        )
        self.assertTrue(torch.equal(hessian, torch.zeros_like(hessian)))
        point = torch.tensor([0.75, 7.3], dtype=torch.float64)
        site = torch.arange(128, dtype=torch.float64)
        offset = (site - point[1] + 64).remainder(128) - 64
        gap = (1 - offset.square() / point[0].square()).clamp_min(0)
        torch.testing.assert_close(profile(point), gap.pow(3))
        expected = torch.stack(
            (
                6 * offset.square() * gap.square() / point[0].pow(3),
                6 * offset * gap.square() / point[0].square(),
            ),
            1,
        )
        torch.testing.assert_close(
            torch.autograd.functional.jacobian(profile, point),
            expected,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_cell_integral_matches_piecewise_quadrature_and_has_no_empty_mass(self):
        import math

        import numpy as np
        import torch
        from inspect_compact_boundaries import public_profile, triweight_cells

        nodes, weights = np.polynomial.legendre.leggauss(8)
        for radius, center in ((0.4, 7.5), (0.75, 0.0), (1.5, 7.3)):
            point = torch.tensor([radius, center], dtype=torch.float64)
            actual = triweight_cells(point)
            self.assertAlmostEqual(float(actual.sum()), 1, delta=3e-15)
            self.assertTrue(bool((actual >= 0).all()))
            expected = []
            for site in range(128):
                mass = 0.0
                for image in range(-2, 3):
                    low = max(site - 0.5 + image * 128, center - radius)
                    high = min(site + 0.5 + image * 128, center + radius)
                    if high > low:
                        x = (low + high) / 2 + nodes * (high - low) / 2
                        mass += float(
                            np.dot(
                                weights,
                                (35 / (32 * radius))
                                * (1 - ((x - center) / radius) ** 2) ** 3,
                            )
                            * (high - low)
                            / 2
                        )
                expected.append(mass)
            torch.testing.assert_close(
                actual,
                torch.tensor(expected, dtype=torch.float64),
                atol=3e-15,
                rtol=1e-12,
            )
            shifted = point + torch.tensor([0.0, 128.0], dtype=point.dtype)
            torch.testing.assert_close(
                triweight_cells(shifted), actual, atol=3e-14, rtol=1e-11
            )
        empty = torch.tensor([0.4, 7.5], dtype=torch.float64)
        self.assertEqual(float(public_profile("triweight")(empty).norm()), 0)
        self.assertGreater(float(triweight_cells(empty).norm()), 0.7)
        self.assertTrue(math.isfinite(float(triweight_cells(empty).sum())))

    def test_l2_floor_changes_slope_but_values_are_continuous(self):
        import math

        import torch
        from inspect_compact_boundaries import FLOOR, public_profile

        function = public_profile("triweight", True)
        boundary = 0.4 * math.sqrt(1 - FLOOR ** (1 / 3))
        left = torch.tensor([0.4, boundary - 1e-7], dtype=torch.float64)
        right = torch.tensor([0.4, boundary + 1e-7], dtype=torch.float64)
        jl = torch.autograd.functional.jacobian(function, left)
        jr = torch.autograd.functional.jacobian(function, right)
        self.assertLess(abs(float(jl[0, 1])), 1e-6)
        self.assertGreater(abs(float(jr[0, 1])), 1000)
        self.assertLess(float((function(right) - function(left)).norm()), 5e-4)
        closer_left = torch.tensor([0.4, boundary - 1e-9], dtype=torch.float64)
        closer_right = torch.tensor([0.4, boundary + 1e-9], dtype=torch.float64)
        self.assertLess(
            float((function(closer_right) - function(closer_left)).norm()), 5e-6
        )
