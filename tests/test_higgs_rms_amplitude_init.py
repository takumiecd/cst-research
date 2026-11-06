"""Search failures and actual raw-Atom RMS calibration against public references."""

import os
import unittest

from cst_research.experiments import higgs_rms_amplitude_init as init


class RMSPolicy(unittest.TestCase):
    def test_search_accepts_actual_target_error_and_rejects_unreachable_target(self):
        best, history = init.search(lambda k: (k * k, {"k": k}), 0.7)
        self.assertLessEqual(best[0], init.SEARCH_TOLERANCE)
        self.assertLessEqual(len(history), 40)
        with self.assertRaises(ValueError):
            init.search(lambda k: (k * k, k), 10000.0)
        with self.assertRaises(ValueError):
            init.search(lambda k: (float("nan"), k), 1.0)


@unittest.skipUnless(
    os.environ.get("CST_DEPTH_SOURCE"), "verified public975 CPU fixture required"
)
class PublicRMSCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_higgs_linear_atom_scaling import PublicFactoryChecks

        PublicFactoryChecks.setUpClass.__func__(cls)

    def test_actual_three_layer_calibration_matches_reference_and_preserves_centers_IO(
        self,
    ):
        t = self.t
        features = t.randn(256, 28, generator=t.Generator().manual_seed(8107))
        old = init.contract.make_model("cst-none", width=128, hidden=4)
        atoms, report, witness = init.calibrate(features, width=128, hidden=4)
        model = init.make_model(atoms, width=128, hidden=4)
        self.assertTrue(t.equal(model.fixed_output_gains, t.ones(3)))
        self.assertEqual(model._conditioning_configuration["normalization"], "none")
        for index in (0, 8):
            self.assertTrue(t.equal(old[index].weight, model[index].weight))
            self.assertTrue(t.equal(old[index].bias, model[index].bias))
        traces = init.reference.forward(
            model.double(), features.double(), collect=True
        )[1]
        for row in report["records"]:
            site = row["site"]
            self.assertTrue(t.equal(old[site].atoms.p[:, 2:], atoms[site][:, 2:]))
            target = float(
                witness["Dense_FP64_pre_ReLU"][str(site)].square().mean().sqrt()
            )
            rms = float(traces[str(site)].square().mean().sqrt())
            self.assertLessEqual(abs(rms / target - 1), init.SEARCH_TOLERANCE)
            expected, clipped = init.trial_atoms(
                old[site].atoms.p, 128, row["coefficient"]
            )
            self.assertTrue(t.equal(expected, atoms[site]))
            self.assertEqual(clipped, row["clipped_atoms"])
        self.assertFalse(report["labels_or_DEV_used"])


if __name__ == "__main__":
    unittest.main()
