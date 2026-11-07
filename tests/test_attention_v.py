"""Shared initialization and explicit attention-recipe admission."""

import unittest
from dataclasses import replace
from cst_research.experiments.ffn_contract import (
    ModelConfig,
    atom_count,
    qualification_covers_model,
)


class AttentionContractTests(unittest.TestCase):
    def test_square_v_qualification_is_distinct_from_ffn(self):
        c = ModelConfig(attention_kind="cst_v", atom_lr_multiplier=16)
        receipt = {
            "task_recipes": [c.recipe()],
            "cases": [
                dict(
                    c.recipe(), **{"in": 256, "out": 256, "atoms": atom_count(256, 256)}
                )
            ],
        }
        self.assertTrue(qualification_covers_model(receipt, c))
        self.assertFalse(
            qualification_covers_model(receipt, replace(c, attention_kind="dense"))
        )
        receipt["cases"][0]["atoms"] *= 2
        self.assertFalse(qualification_covers_model(receipt, c))

    def test_query_and_key_preserve_complementary_projection_rows(self):
        try:
            import torch
            import torchcst
        except ImportError:
            raise unittest.SkipTest("optional Torch/torchcst required")
        from cst_research.experiments.attention_qualification import (
            shared_initialization,
        )

        for kind in ("cst_q", "cst_k"):
            r = shared_initialization(
                ModelConfig(attention_kind=kind, atom_lr_multiplier=16)
            )
            self.assertTrue(r["complementary_projection_weights_and_biases_exact"])
            self.assertNotIn("all_QK_weights_and_biases_exact", r)
            self.assertEqual(len(r["reports"]), 4)

    def test_initial_qk_and_shared_parameters_are_exact(self):
        try:
            import torch
            import torchcst
        except ImportError:
            raise unittest.SkipTest("optional Torch/torchcst required")
        from cst_research.experiments.attention_qualification import (
            shared_initialization,
        )

        r = shared_initialization(
            ModelConfig(attention_kind="cst_v", atom_lr_multiplier=16)
        )
        self.assertTrue(r["all_QK_weights_and_biases_exact"])
        self.assertEqual(len(r["reports"]), 4)
