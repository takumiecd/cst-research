"""Activity-source and update-implementation identities are part of admission."""

import importlib
import unittest
from dataclasses import replace

from cst_research.experiments.ffn_contract import (
    ModelConfig,
    qualification_covers_model,
)


class CenterActivityTests(unittest.TestCase):
    def test_receipt_cannot_substitute_source_gain_or_update_backend(self):
        c = ModelConfig(
            activity_mode="center_chord",
            activity_gain=64,
            update_backend="radius_eager",
        )
        scalars = {
            k: v
            for k, v in c.recipe().items()
            if k not in ("feature_order", "atom_lr_multiplier", "coefficient")
        }
        receipt = {
            "task_recipes": [c.recipe()],
            "cases": [
                dict(scalars, **{"in": ni, "out": no, "atoms": 4096, "coefficient": 8})
                for ni, no in ((256, 1024), (1024, 256))
            ],
        }
        self.assertTrue(qualification_covers_model(receipt, c))
        self.assertFalse(
            qualification_covers_model(receipt, replace(c, activity_gain=1))
        )
        self.assertFalse(
            qualification_covers_model(
                receipt, replace(c, activity_mode="finite_chord")
            )
        )
        for case in receipt["cases"]:
            case.pop("update_backend")
        self.assertFalse(qualification_covers_model(receipt, c))

    def test_native_center_mode_rejected_and_legacy_recipe_unchanged(self):
        with self.assertRaises(ValueError):
            ModelConfig(activity_mode="center_chord")
        self.assertNotIn("activity_mode", ModelConfig().recipe())
        self.assertNotIn("update_backend", ModelConfig().recipe())
        for gain in (-1, 0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                ModelConfig(activity_gain=gain)

    def test_same_initial_tensors_and_calibration_reports(self):
        try:
            import torch

            importlib.import_module("torchcst")
        except ImportError:
            raise unittest.SkipTest("optional Torch/torchcst required")
        from cst_research.experiments.ffn_model import make_cst

        old = make_cst(16, 64, std=0.001, q=0)
        for gain in (1, 64, 4096):
            new = make_cst(
                16, 64, std=0.001, q=0, activity_mode="center_chord", activity_gain=gain
            )
            torch.testing.assert_close(old.atoms.p, new.atoms.p, atol=0, rtol=0)
            self.assertEqual(old.initialization_report, new.initialization_report)
