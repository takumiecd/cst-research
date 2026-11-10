"""Capacity changes preserve ordinary initialization and the M8 definition."""

import unittest
from dataclasses import replace
from unittest.mock import patch

try:
    import torch
    import torchcst  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional Torch/torchcst")
class M8CapacityTests(unittest.TestCase):
    def test_c8_and_c16_match_the_prior_named_raw_initialization_families(self):
        from cst_research.experiments import lm_geometry_trial as old
        from cst_research.experiments import lm_m8_capacity as new
        from cst_research.experiments.ffn_dst_run import config
        from cst_research.experiments.ffn_model import parameter_digest
        from cst_research.experiments.ffn_selective import projections
        from cst_research.experiments.lm_m8_capacity_run import ordinary

        cfg = replace(config(), vocabulary=32, context=8)
        with (
            patch.object(new, "config", return_value=cfg),
            patch.object(old, "config", return_value=cfg),
        ):
            a = new.model_for("m8-c8", device="cpu")
            old_a = old.model_for("legacy-m8", device="cpu")
            self.assertEqual(parameter_digest(a), parameter_digest(old_a))
            shared = ordinary(a)
            del a, old_a
            b = new.model_for("m8-c16", device="cpu")
            old_b = old.model_for("m16-c16-e8", device="cpu")
            self.assertEqual(parameter_digest(b), parameter_digest(old_b))
            self.assertEqual(ordinary(b), shared)
            for p in projections(b).values():
                self.assertEqual(tuple(p.map.atoms.p.shape), (16384, 4))
                self.assertTrue(
                    all(
                        a.profile.modes == 8 and a.connection is None
                        for a in p.map.kernel.spec.profiles
                    )
                )
            opt = new.optimizer_for(b)
            self.assertEqual([g["eps"] for g in opt.param_groups], [1e-8, 1e-8])
            self.assertEqual([g["lr"] for g in opt.param_groups], [0.0003, 0.0096])
            self.assertEqual(
                opt.param_groups[1]["field_lrs"], {"radius_fraction": 0.0096}
            )


if __name__ == "__main__":
    unittest.main()
