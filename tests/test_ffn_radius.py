"""Initial physical map parity, gradient direction, and post-step projection."""

import unittest
from dataclasses import replace

from cst_research.experiments.ffn_contract import (
    ModelConfig,
    qualification_covers_model,
)


class RadiusContractTests(unittest.TestCase):
    def test_radius_mode_cannot_reuse_polar_admission(self):
        c = ModelConfig(
            width=512, heads=8, linear_backend="levels", radius_mode="learned"
        )
        self.assertEqual(c.recipe()["radius_mode"], "learned")
        self.assertFalse(qualification_covers_model({}, c, ffn_kind="cst"))
        with self.assertRaises(ValueError):
            replace(c, attention_kind="cst_qkvo")

    def test_physical_initialization_and_projected_step(self):
        try:
            import torch

            from cst_research.experiments.ffn_model import LanguageModel, cst_optimizer
            from cst_research.experiments.ffn_radius import probe_radius
        except ImportError:
            self.skipTest("optional torch/torchcst required")
        c = ModelConfig(
            width=16,
            layers=1,
            heads=2,
            context=8,
            vocabulary=32,
            radius_birth=0.5,
            radius_envelope="rational_zero_opening",
            zero_opening_scale=0.001,
            atom_lr_multiplier=16,
        )
        models = {
            mode: LanguageModel(replace(c, radius_mode=mode), "cst", 17)
            for mode in ["polar", "learned", "fixed"]
        }
        x = torch.arange(8)[None, :]
        baseline = models["polar"](x)[0]
        for mode in ["learned", "fixed"]:
            torch.testing.assert_close(
                models[mode](x)[0], baseline, atol=3e-5, rtol=3e-4
            )
        for mode in ["learned", "fixed"]:
            model = models[mode]
            before = {
                n: p.detach().clone()
                for n, p in model.named_parameters()
                if n.endswith("atoms.p")
            }
            _, loss = model(x, x)
            loss.backward()
            for n, p in model.named_parameters():
                if n in before:
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertEqual(
                        bool(p.grad[:, 1].abs().max() > 0), mode == "learned"
                    )
            opt = cst_optimizer(
                model,
                atom_lr_multiplier=16,
                lr=0.01,
                betas=(0.9, 0.95),
                weight_decay=0,
                foreach=False,
            )
            opt.step()
            for n, p in model.named_parameters():
                if n in before:
                    self.assertTrue((p[:, 1] >= 0.5).all())
                    if mode == "fixed":
                        torch.testing.assert_close(
                            p[:, 1], before[n][:, 1], atol=0, rtol=0
                        )
            saved = opt.state_dict()
            other = cst_optimizer(
                model,
                atom_lr_multiplier=16,
                lr=0.01,
                betas=(0.9, 0.95),
                weight_decay=0,
                foreach=False,
            )
            other.load_state_dict(saved)
            diagnostic = probe_radius(model, x, x)
            self.assertEqual(len(diagnostic["layers"]), 2)


if __name__ == "__main__":
    unittest.main()
