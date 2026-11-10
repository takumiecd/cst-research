"""The primary DST budget is mask-one count A, with LR trials paired at init."""

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
class ActiveCountTests(unittest.TestCase):
    def test_selection_uses_final_dev_and_breaks_ties_with_smaller_lr(self):
        from cst_research.experiments.lm_m8_active_count import ARMS
        from cst_research.experiments.lm_m8_active_count_run import select

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, (kind, c, lr) in ARMS.items():
                if kind != "dst":
                    continue
                out = root / arm
                out.mkdir()
                ce = (
                    (3.6 if lr == 0.001 else 3.4)
                    if c == 8
                    else (3.2 if lr == 0.001 else 3.5)
                )
                (out / "summary.json").write_text(
                    json.dumps({"state": "completed", "completed_steps": 976})
                )
                (out / "curves.jsonl").write_text(
                    json.dumps({"step": 976, "development": {"ce": ce}}) + "\n"
                )
            target = root / "selection"
            target.mkdir()
            select(root, target)
            result = json.loads((target / "selection.json").read_text())
            self.assertEqual(
                result["selected"], {"8": "dst-c8-lr0.003", "16": "dst-c16-lr0.001"}
            )
            self.assertEqual(len(result["candidates"]), 6)
            self.assertFalse(result["confirmation_evaluated"])

    def test_exact_one_atom_one_active_weight_and_lr_pairing(self):
        from cst_research.experiments import ffn_dst
        from cst_research.experiments import lm_m8_active_count as study
        from cst_research.experiments.ffn_dst_run import config
        from cst_research.experiments.ffn_model import parameter_digest
        from cst_research.experiments.ffn_selective import active_ffn_scalars
        from cst_research.experiments.lm_m8_capacity_run import ordinary

        cfg = replace(config(), vocabulary=32, context=8)
        shared = None
        with patch.object(study, "config", return_value=cfg):
            for c in (8, 16):
                a = study.model_for(f"dst-c{c}-lr0.001", device="cpu")
                self.assertEqual(set(study.active_slots(a).values()), {c * 1024})
                self.assertEqual(active_ffn_scalars(a), 8 * c * 1024 + 10240)
                if shared is not None:
                    self.assertEqual(ordinary(a), shared)
                shared = ordinary(a)
                for layer in ffn_dst.layers(a).values():
                    self.assertEqual(layer.weight.layout, torch.strided)
                    self.assertEqual(layer.mask.dtype, torch.bool)
                    self.assertEqual(int(layer.mask.sum()), c * 1024)
                    self.assertTrue((layer.weight.detach()[~layer.mask] == 0).all())
                digest = parameter_digest(a)
                del a
                b = study.model_for(f"dst-c{c}-lr0.01", device="cpu")
                self.assertEqual(parameter_digest(b), digest)
                opt = study.optimizer_for(b)
                self.assertEqual([g["lr"] for g in opt.param_groups], [0.0003, 0.01])
                self.assertTrue(all(g["eps"] == 1e-8 for g in opt.param_groups))
                del b, opt


if __name__ == "__main__":
    unittest.main()
