"""Admission identity and diagnostic sampling contracts for the new envelope."""

import importlib
import unittest
from dataclasses import replace

from cst_research.experiments.ffn_contract import (
    ModelConfig,
    qualification_covers_model,
)


class ClosedEnvelopeContractTests(unittest.TestCase):
    def test_old_qualification_cannot_admit_new_envelope(self):
        c = ModelConfig(
            radius_envelope="closed_polynomial", q=1.0, atom_lr_multiplier=16
        )
        cases = [
            dict(**c.recipe(), **{"in": ni, "out": no, "atoms": c.atoms_per_projection})
            for ni, no in [(256, 1024), (1024, 256)]
        ]
        receipt = {"task_recipes": [c.recipe()], "cases": cases}
        self.assertTrue(qualification_covers_model(receipt, c))
        self.assertFalse(
            qualification_covers_model(receipt, replace(c, radius_envelope="rational"))
        )
        self.assertFalse(
            qualification_covers_model(receipt, replace(c, closed_lower_power=8))
        )

    def test_zero_opening_scale_is_part_of_admission(self):
        c = ModelConfig(
            radius_envelope="rational_zero_opening",
            zero_opening_scale=0.03,
            atom_lr_multiplier=16,
        )
        cases = [
            dict(**c.recipe(), **{"in": ni, "out": no, "atoms": c.atoms_per_projection})
            for ni, no in [(256, 1024), (1024, 256)]
        ]
        receipt = {"task_recipes": [c.recipe()], "cases": cases}
        self.assertTrue(qualification_covers_model(receipt, c))
        self.assertFalse(
            qualification_covers_model(receipt, replace(c, zero_opening_scale=0.1))
        )
        broken = {"task_recipes": [c.recipe()], "cases": [dict(row) for row in cases]}
        broken["cases"][1]["zero_opening_scale"] = 0.1
        self.assertFalse(qualification_covers_model(broken, c))

    def test_diagnostic_bins_conserve_atoms_and_do_not_mutate_rng_or_parameters(self):
        try:
            torch = importlib.import_module("torch")
            importlib.import_module("torchcst")
        except ImportError as e:
            raise unittest.SkipTest("optional Torch/torchcst required") from e
        from cst_research.experiments.ffn_model import make_cst
        from cst_research.experiments.ffn_envelope import envelope_snapshot

        layer = make_cst(16, 64, q=1.0, std=0.001, radius_envelope="closed_polynomial")
        before = layer.atoms.p.detach().clone()
        rng = torch.get_rng_state().clone()
        row = envelope_snapshot(layer)[""]
        self.assertEqual(sum(b["count"] for b in row["bins"]), layer.atoms.count)
        self.assertEqual(row["all"]["count"], layer.atoms.count)
        torch.testing.assert_close(before, layer.atoms.p, atol=0, rtol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
