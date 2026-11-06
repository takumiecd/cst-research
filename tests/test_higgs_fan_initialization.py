"""Moment algebra, Polar inverse and independent overlapping-Atom energy oracle."""

import itertools
import os
import unittest

from cst_research.experiments import higgs_fan_initialization as init


class FanTargets(unittest.TestCase):
    def test_logical_rectangular_he_and_xavier_reduce_to_dense_targets(self):
        self.assertAlmostEqual(init.target_energy("he", 3, 7), 14.0)
        self.assertAlmostEqual(init.target_energy("he", 3, 7, mode="fan_out"), 6.0)
        self.assertAlmostEqual(init.target_energy("xavier", 3, 7), 4.2)
        for scheme, ni, no, mode, gain in [
            ("he", 3, 7, "fan_in", 2.0),
            ("he", 7, 3, "fan_out", 1.0),
            ("xavier", 3, 7, "fan_in", 1.5),
        ]:
            target = init.target_energy(scheme, ni, no, mode=mode, gain=gain)
            energy = ni * no  # One unit coordinate-basis Atom per dense entry.
            fi, fo = energy / no, energy / ni
            sigma2 = (
                gain**2 / (fi if mode == "fan_in" else fo)
                if scheme == "he"
                else gain**2 * 2 / (fi + fo)
            )
            self.assertAlmostEqual(sigma2 * energy, target)
        for kwargs in [
            dict(scheme="auto", fan_in=3, fan_out=7),
            dict(scheme="he", fan_in=True, fan_out=7),
            dict(scheme="he", fan_in=3, fan_out=7, gain=float("nan")),
        ]:
            with self.assertRaises(ValueError):
                init.target_energy(**kwargs)

    def test_solver_rejects_unreachable_and_nonfinite_targets(self):
        scale, result, history = init.solve(lambda s: (s * s, s), 2.0)
        self.assertLessEqual(abs(scale * scale / 2 - 1), init.TOLERANCE)
        self.assertEqual(scale, result)
        self.assertLessEqual(len(history), 44)
        with self.assertRaises(ValueError):
            init.solve(lambda s: (0.0, s), 1.0)
        with self.assertRaises(ValueError):
            init.solve(lambda s: (float("nan"), s), 1.0)


@unittest.skipUnless(
    os.environ.get("CST_DEPTH_SOURCE"), "verified public975 CPU fixture required"
)
class PublicFactorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_higgs_linear_atom_scaling import PublicFactoryChecks

        PublicFactoryChecks.setUpClass.__func__(cls)

    def test_polar_inverse_q_and_overlap_sign_energy_without_dense_assumptions(self):
        t = self.t
        from torchcst._backends.torch.parameterizations.radius_cell_mass import (
            decoded_values,
        )

        model = init.contract.make_model("cst-none", width=128, hidden=4)
        layer = model[2].double()
        original = model[2].atoms.p.float().detach().clone()
        amplitudes = []
        for q in (0.0, 0.25, 1.0):
            p, _ = init.encode(original, 0.2, q)
            a, ri, ro = decoded_values(layer.kernel, p.double())
            radiusq = ((p[:, :2].double().square().sum(1) - 1) / 3).clamp(0, 1)
            self.assertTrue(
                t.allclose(radiusq, t.full_like(radiusq, q), rtol=0, atol=2e-7)
            )
            self.assertTrue(t.equal(p[:, 2:], original[:, 2:]))
            self.assertTrue(bool(t.isfinite(ri).all() and t.isfinite(ro).all()))
            amplitudes.append(a)
        for a in amplitudes[1:]:
            self.assertTrue(t.allclose(a, amplitudes[0], rtol=2e-7, atol=1e-8))
        p, _ = init.encode(original, 0.2, 0.25)
        p = p[:3].double()
        # Make all centers identical so overlap/cross terms cannot be ignored
        # for an individual realization. Enumerating signs cancels them exactly.
        p[:, 2:] = p[0, 2:]
        energy = init.atom_energies(layer, p)
        x, y = layer._factor_atoms(p)
        per_atom = t.stack([t.outer(y[:, i], x[:, i]).square().sum() for i in range(3)])
        self.assertTrue(t.allclose(energy, per_atom, rtol=1e-12, atol=1e-14))
        realized = []
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            z = p.clone()
            z[:, 0] *= t.tensor(signs)
            xi, yo = layer._factor_atoms(z)
            realized.append((yo @ xi.T).square().sum())
        self.assertTrue(
            t.allclose(t.stack(realized).mean(), energy.sum(), rtol=1e-12, atol=1e-14)
        )
        self.assertGreater(float(t.stack(realized).std()), 1e-5)
        for q in (-0.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                init.encode(original, 0.2, q)

    def test_actual_layer_matches_energy_target_and_preserves_centers(self):
        model = init.contract.make_model("cst-none", width=128, hidden=4)
        old = model[2].atoms.p.detach().clone()
        for scheme in ("he", "xavier"):
            p, record, energy = init.initialize_layer(model[2], scheme=scheme, q=0.0)
            self.assertLessEqual(record["relative_error"], init.TOLERANCE)
            self.assertTrue(self.t.equal(p[:, 2:], old[:, 2:]))
            self.assertTrue(self.t.equal(model[2].atoms.p, old))
            self.assertAlmostEqual(
                float(energy.sum()), record["observed_conditional_sign_energy"]
            )
            self.assertFalse(record["data_features_or_Dense_reference_used"])


if __name__ == "__main__":
    unittest.main()
