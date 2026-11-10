"""Capacity admission, exact initialization and optimizer epsilon isolation."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from cst_research.experiments.ffn_contract import (
    ModelConfig,
    qualification_covers_model,
)


class CapacityAdmissionTests(unittest.TestCase):
    def test_previous_c8_receipt_does_not_qualify_c16(self):
        from dataclasses import asdict

        c8 = ModelConfig(
            width=512, heads=8, radius_mode="envelope", linear_backend="levels"
        )
        c16 = replace(c8, coefficient=16)
        self.assertEqual(c16.atoms_per_projection, 16384)
        receipt = {
            "task_model_configs": [asdict(c8)],
            "task_recipes": [c8.recipe()],
            "cases": [
                {**c8.recipe(), "in": ni, "out": no, "atoms": 8192}
                for ni, no in ((512, 2048), (2048, 512))
            ],
        }
        self.assertTrue(qualification_covers_model(receipt, c8))
        self.assertFalse(qualification_covers_model(receipt, c16))
        with self.assertRaises(ValueError):
            replace(c8, coefficient=32)


try:
    import torch
    import torchcst  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional Torch/torchcst")
class EpsilonTests(unittest.TestCase):
    def test_same_rows_different_epsilon_and_serialized_group_retention(self):
        from cst_research.experiments import lm_geometry_trial as trial
        from cst_research.experiments.ffn_dst_run import config
        from cst_research.experiments.ffn_model import parameter_digest

        cfg = replace(config(), vocabulary=32, context=8)
        with patch.object(trial, "config", return_value=cfg):
            a = trial.model_for("m16-c8-e8", device="cpu")
            b = trial.model_for("m16-c8-e10", device="cpu")
            self.assertEqual(parameter_digest(a), parameter_digest(b))
            old, new = trial.optimizer_for(a), trial.optimizer_for(b)
            self.assertEqual([g["eps"] for g in old.param_groups], [1e-8, 1e-8])
            self.assertEqual([g["eps"] for g in new.param_groups], [1e-8, 1e-10])
            self.assertEqual(
                [g["lr"] for g in old.param_groups], [g["lr"] for g in new.param_groups]
            )
            self.assertEqual(
                old.param_groups[1]["field_lrs"], new.param_groups[1]["field_lrs"]
            )
            new.load_state_dict(old.state_dict())
            self.assertEqual([g["eps"] for g in new.param_groups], [1e-8, 1e-8])


if __name__ == "__main__":
    unittest.main()
