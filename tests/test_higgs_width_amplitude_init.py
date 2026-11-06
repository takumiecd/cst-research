"""Paired initial values, public native load, and unchanged architecture contracts."""

import os
import unittest

from cst_research.experiments import higgs_width_amplitude_init as study


class InitializerProtocol(unittest.TestCase):
    def test_fixed_bounded_comparison_and_no_norm_or_gain_for_new_initializer(self):
        p = study.protocol()
        self.assertEqual(p["total_task_updates"], 100000)
        self.assertEqual(p["milestones"], [0, 1000, 5000, 10000, 15000, 20000, 25000])
        self.assertEqual(
            study.initialization_policy("width-init")["physical_amplitude_std"], 0.05
        )
        self.assertEqual(
            study.initialization_policy("old-init")["physical_amplitude_std"], 0.0015625
        )
        self.assertEqual(study.initialization_policy("width-init")["gain"], "all ones")
        self.assertEqual(
            study.initialization_policy("width-init")["normalization"], "none"
        )
        self.assertFalse(p["automatic_followup"])


@unittest.skipUnless(
    os.environ.get("CST_DEPTH_SOURCE"), "verified public975 CPU fixture required"
)
class PublicInitializerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_higgs_linear_atom_scaling import PublicFactoryChecks

        PublicFactoryChecks.setUpClass.__func__(cls)

    def test_exact32_amplitude_draw_change_centers_and_ordinary_IO_preserved(self):
        t = self.t
        before = t.random.get_rng_state().clone()
        old = study.make_model("old-init")
        new = study.make_model("width-init")
        self.assertTrue(t.equal(before, t.random.get_rng_state()))
        self.assertEqual(
            old._conditioning_configuration, new._conditioning_configuration
        )
        for index in (0, 16):
            self.assertTrue(t.equal(old[index].weight, new[index].weight))
            self.assertTrue(t.equal(old[index].bias, new[index].bias))
        for index in (2, 4, 6, 8, 10, 12, 14):
            p, q = old[index].atoms.p, new[index].atoms.p
            self.assertTrue(t.equal(p[:, 2:], q[:, 2:]))
            self.assertTrue(t.equal(p[:, 0] * 32, q[:, 0]))
            self.assertTrue(t.equal((1 - q[:, 0].square()).sqrt(), q[:, 1]))
        self.assertTrue(t.equal(new.fixed_output_gains, t.ones(7)))
        self.assertEqual(
            sum(p.numel() for p in old.parameters()),
            sum(p.numel() for p in new.parameters()),
        )
        old_physical = study.physical_initialization_record(old)
        new_physical = study.physical_initialization_record(new)
        self.assertTrue(
            all(
                n["input_radius_min"] < o["input_radius_min"]
                for o, n in zip(old_physical, new_physical, strict=True)
            )
        )

    def test_gain_control_is_prior_public_model_and_unit_gain_is_enforced(self):
        from cst_research.experiments import (
            higgs_multilayer_conditioning_state as state,
        )

        actual = study.make_model("old-init-gain", gains=[2.0] * 3, width=128, hidden=4)
        prior = study.model_contract.make_model(
            "cst-fixed-gain", gains=[2.0] * 3, width=128, hidden=4
        )
        state.exact.assert_equal(
            actual.state_dict(), prior.state_dict(), "prior gain control exact state"
        )
        with self.assertRaises(ValueError):
            study.make_model("width-init", gains=[2.0] * 3, width=128, hidden=4)

    def test_new_initial_values_roundtrip_through_unchanged_native_shape_loader(self):
        from cst_research.experiments import (
            higgs_multilayer_conditioning_state as state,
        )

        model = study.make_model("width-init", width=128, hidden=4)
        optimizer, _ = state.make_optimizer(model, cpu_reference=True)
        identity = {
            "arm": "width-init",
            "initialization": study.initialization_policy("width-init"),
        }
        raw = state.snapshot(model, optimizer, 0, identity)
        loaded, _, core, proof = state.load(raw, "cpu", cpu_reference=True)
        state.exact.assert_equal(
            core,
            state.qualified_state.with_generations(raw, state.owners(loaded)),
            "new raw initializer native load",
        )
        self.assertEqual(
            proof["actual_postload_core_sha256"],
            proof["expected_only_native_generations_adjusted_core_sha256"],
        )
        self.assertTrue(self.t.equal(loaded[2].atoms.p, model[2].atoms.p))

    def test_new_initialization_has_finite_full_network_reference_and_larger_signal(
        self,
    ):
        t = self.t
        from cst_research.experiments import (
            higgs_multilayer_conditioning_reference as ref,
        )

        inputs = t.randn(
            4, 28, generator=t.Generator().manual_seed(817), dtype=t.float64
        )
        dy = t.tensor(
            [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]], dtype=t.float64
        )
        traces = {}
        for arm in ("old-init", "width-init"):
            model = study.make_model(arm, width=128, hidden=4).double()
            y, dx, dp = ref.full_reference(model, inputs, dy)
            self.assertTrue(
                all(bool(t.isfinite(x).all()) for x in (y, dx, *dp.values()))
            )
            self.assertEqual(list(dp), [n for n, _ in model.named_parameters()])
            traces[arm] = ref.forward(model, inputs, collect=True)[1]
        self.assertGreater(
            float(traces["width-init"]["6"].square().mean()),
            100 * float(traces["old-init"]["6"].square().mean()),
        )


if __name__ == "__main__":
    unittest.main()
