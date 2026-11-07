"""Minimum-based L is expressed by existing Polar bounds, not a copied kernel."""

import importlib
import unittest


class MinimumRadiusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch = importlib.import_module("torch")
            importlib.import_module("torchcst")
        except ImportError as e:
            raise unittest.SkipTest("optional Torch/torchcst required") from e

    def test_independent_minimum_based_formula(self):
        from cst_research.experiments.ffn_model import make_cst
        from torchcst._backends.torch.parameterizations.polar_amp_width import (
            _width_bounds,
        )

        t = self.torch
        for epsilon in (0.001, 0.003):
            m = make_cst(
                32,
                32,
                std=0.001,
                radius_birth=0.5,
                radius_envelope="rational_zero_opening",
                zero_opening_scale=epsilon,
            )
            a = t.tensor([0.0, 0.001, 0.01, 0.03, 0.1, 1.0], dtype=t.float64)
            r, lo, hi = _width_bounds(
                m.kernel,
                a,
                t.zeros_like(a),
                scalar_names=m.kernel.spec.parameterization.width_scalar_names,
            )
            eps = float(m.kernel.scalar("zero_opening_scale"))
            expected = 0.5 + (hi - 0.5) * eps**2 / (a * a + eps**2)
            t.testing.assert_close(lo, expected, atol=1e-12, rtol=1e-12)
            t.testing.assert_close(r, lo, atol=0, rtol=0)
            self.assertEqual(float(lo[0]), float(hi[0]))
            self.assertLess(float(lo[-1]), 0.501)

    def test_shortest_torus_center_motion_and_read_only_sampling(self):
        from cst_research.experiments.ffn_model import make_cst
        from cst_research.experiments.ffn_center_motion import CenterMotion

        t = self.torch
        m = make_cst(32, 32, std=0.001)
        with t.no_grad():
            m.atoms.p[0, 2] = 31.99
        motion = CenterMotion(m)
        radius = float(motion.previous[""][3][0])
        with t.no_grad():
            m.atoms.p[0, 2] = 0.01
        before = m.atoms.p.detach().clone()
        rng = t.get_rng_state().clone()
        motion.update()
        r = motion.report()[""]
        self.assertAlmostEqual(
            float(motion.last[""][0]), 0.02 / radius / (2**0.5), places=5
        )
        self.assertEqual(sum(v["count"] for v in r["bins"]), m.atoms.count)
        t.testing.assert_close(before, m.atoms.p, atol=0, rtol=0)
        t.testing.assert_close(rng, t.get_rng_state(), atol=0, rtol=0)
