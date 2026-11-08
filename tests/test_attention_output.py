"""Full attention output replacement must preserve residual scale and admission."""

import unittest
from dataclasses import asdict, replace

from cst_research.experiments.ffn_contract import (
    ModelConfig,
    qualification_covers_model,
)


class OutputAttentionTests(unittest.TestCase):
    def test_new_backend_requires_exact_architecture_and_operator(self):
        c = ModelConfig(
            width=512,
            heads=8,
            attention_kind="cst_qkvo",
            linear_backend="levels",
            atom_lr_multiplier=16,
        )
        row = dict(c.recipe(), **{"in": 512, "out": 512, "atoms": 4096})
        receipt = {
            "task_recipes": [c.recipe()],
            "task_model_configs": [asdict(c)],
            "model_kinds": [{"ffn_kind": "dense", "attention_kind": "cst_qkvo"}],
            "cases": [row],
        }
        self.assertTrue(qualification_covers_model(receipt, c, ffn_kind="dense"))
        self.assertFalse(
            qualification_covers_model(receipt, replace(c, heads=4), ffn_kind="dense")
        )
        self.assertFalse(
            qualification_covers_model(
                receipt, replace(c, linear_backend="levels_stream64"), ffn_kind="dense"
            )
        )
        del row["linear_backend"]
        self.assertFalse(qualification_covers_model(receipt, c, ffn_kind="dense"))

    def test_all_24_sites_have_residual_output_target_and_tied_vocab(self):
        try:
            import torch  # noqa: F401
            import torchcst  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("optional torch/torchcst required")
        from cst_research.experiments.attention_qualification import (
            shared_initialization,
        )
        from cst_research.experiments.ffn_model import LanguageModel

        c = ModelConfig(
            width=256, heads=4, attention_kind="cst_qkvo", atom_lr_multiplier=16
        )
        r = shared_initialization(c, "cst")
        self.assertEqual(len(r["reports"]), 24)
        for name, report in r["reports"].items():
            variance = (
                0.02**2 / (2 * c.layers)
                if (".attention_out." in name or ".ffn.down." in name)
                else 0.02**2
            )
            self.assertAlmostEqual(report["variance"], variance)
        m = LanguageModel(c, "cst", 17)
        names = dict(m.named_parameters())
        self.assertIn("embedding.weight", names)
        self.assertFalse(any("lm_head" in name for name in names))
        self.assertEqual(len([name for name in names if name.endswith("atoms.p")]), 24)
