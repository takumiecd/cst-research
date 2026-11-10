"""Budget staging and the descriptive near-Dense decision are fixed in advance."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    import torchcst  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional Torch/torchcst")
class AtomBudgetTests(unittest.TestCase):
    def test_near_dense_and_stage_boundaries(self):
        from cst_research.experiments.lm_atom_budget_run import near_dense, stage_arms

        dense = {"ce": 3.0, "next_token_top1_accuracy": 0.4}
        self.assertTrue(
            near_dense(dense, {"ce": 3.01, "next_token_top1_accuracy": 0.395})
        )
        self.assertFalse(
            near_dense(dense, {"ce": 3.0101, "next_token_top1_accuracy": 0.4})
        )
        self.assertFalse(
            near_dense(dense, {"ce": 3.0, "next_token_top1_accuracy": 0.3949})
        )
        self.assertEqual(stage_arms(0), ("dense",))
        self.assertEqual(
            stage_arms(32),
            ("m8-c32", "dst-c32-lr0.003", "dst-c32-lr0.01", "dst-c32-lr0.03"),
        )

    def test_private_config_does_not_widen_ordinary_admission(self):
        from dataclasses import asdict

        from cst_research.experiments.ffn_contract import ModelConfig
        from cst_research.experiments.ffn_dst_run import config
        from cst_research.experiments.lm_atom_budget import AtomBudgetConfig

        values = asdict(config()) | {"coefficient": 32}
        with self.assertRaises(ValueError):
            ModelConfig(**values)
        self.assertEqual(AtomBudgetConfig(**values).coefficient, 32)
        with self.assertRaises(ValueError):
            AtomBudgetConfig(**(values | {"coefficient": 17}))

    def test_mask_count_layout_and_init_pairing(self):
        from cst_research.experiments import ffn_dst
        from cst_research.experiments import lm_atom_budget as study
        from cst_research.experiments.ffn_dst_run import config
        from cst_research.experiments.ffn_model import parameter_digest
        from cst_research.experiments.lm_m8_capacity_run import ordinary

        cfg = replace(config(), vocabulary=32, context=8)
        with patch.object(study, "config", return_value=cfg):
            dense = study.model_for("dense", device="cpu")
            shared = ordinary(dense)
            self.assertEqual(study.active_slots(dense), {})
            for c in (16, 32, 128):
                a = study.model_for(f"dst-c{c}-lr0.003", device="cpu")
                self.assertEqual(set(study.active_slots(a).values()), {c * 1024})
                self.assertEqual(ordinary(a), shared)
                masks = {n: v.mask.clone() for n, v in ffn_dst.layers(a).items()}
                digest = parameter_digest(a)
                self.assertTrue(
                    all(
                        v.weight.layout == torch.strided and v.mask.dtype == torch.bool
                        for v in ffn_dst.layers(a).values()
                    )
                )
                del a
                b = study.model_for(f"dst-c{c}-lr0.03", device="cpu")
                self.assertEqual(parameter_digest(b), digest)
                self.assertTrue(
                    all(
                        torch.equal(v.mask, masks[n])
                        for n, v in ffn_dst.layers(b).items()
                    )
                )
                self.assertEqual(
                    [g["lr"] for g in study.optimizer_for(b).param_groups],
                    [0.0003, 0.03],
                )
                del b

    def test_selection_only_uses_target_count_and_retains_all_lrs(self):
        from cst_research.experiments.lm_atom_budget_run import select, stage_arms

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for a in stage_arms(32)[1:]:
                p = root / a
                p.mkdir()
                ce = 3.3 if a.endswith("0.003") else 3.2
                (p / "summary.json").write_text(
                    json.dumps({"state": "completed", "completed_steps": 976})
                )
                (p / "curves.jsonl").write_text(
                    json.dumps({"step": 976, "development": {"ce": ce}}) + "\n"
                )
            target = root / "selection"
            target.mkdir()
            select(root, target, 32)
            s = json.loads((target / "selection.json").read_text())
            self.assertEqual(s["selected"], {"32": "dst-c32-lr0.01"})
            self.assertEqual(len(s["candidates"]), 3)
            self.assertFalse(s["confirmation_evaluated"])
