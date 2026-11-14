"""Mapped Polar derivatives, feasible endpoint slopes, storage and resume."""

import unittest
from dataclasses import replace

from cst_research.experiments.ffn_contract import (
    ModelConfig,
    qualification_covers_model,
)


class MappedPolarTests(unittest.TestCase):
    def test_explicit_modes_require_own_levels_admission(self):
        for mode in ("polar_live", "polar_detached"):
            c = ModelConfig(
                width=512, heads=8, linear_backend="levels", radius_mode=mode
            )
            self.assertFalse(qualification_covers_model({}, c, ffn_kind="cst"))
            with self.assertRaises(ValueError):
                replace(c, linear_backend="native")
            with self.assertRaises(ValueError):
                replace(c, attention_kind="cst_qkvo")

    def test_same_values_live_radial_gradient_and_projected_resume(self):
        try:
            import copy

            import torch

            from cst_research.experiments.ffn_model import (
                LanguageModel,
                cst_optimizer,
                make_cst,
            )
            from cst_research.experiments.ffn_polar_radius import (
                mapped_parameters,
                projections,
            )
            from cst_research.experiments.ffn_qualification import _assert_tree_equal
            from cst_research.experiments.ffn_radius import probe_radius
        except ImportError:
            self.skipTest("optional torch/torchcst required")
        base = make_cst(
            16,
            64,
            std=0.001,
            radius_birth=0.5,
            radius_envelope="rational_zero_opening",
            zero_opening_scale=0.001,
        ).double()
        # At q0 and q1 the declared inward dRadius/dq remains nonzero.
        for rho in [1.0, 2.0, 1.4]:
            p = torch.tensor(
                [[rho * 0.6, rho * 0.8, 3.13, 1.17]],
                dtype=torch.float64,
                requires_grad=True,
            )
            on = mapped_parameters(base, p, live=True)
            off = mapped_parameters(base, p, live=False)
            torch.testing.assert_close(on, off, atol=0, rtol=0)
            g = torch.autograd.grad(on[:, 1].sum(), p)[0]
            self.assertGreater(float((g[:, :2] * p[:, :2] / rho).sum()), 0)
        c = ModelConfig(
            width=16,
            layers=1,
            heads=2,
            context=8,
            vocabulary=32,
            linear_backend="levels",
            radius_birth=0.5,
            radius_envelope="rational_zero_opening",
            zero_opening_scale=0.001,
            atom_lr_multiplier=16,
        )
        models = {
            mode: LanguageModel(replace(c, radius_mode=mode), "cst", 17)
            for mode in ["polar_detached", "polar_live"]
        }
        self.assertEqual(
            sum(p.numel() for p in models["polar_live"].parameters()),
            sum(p.numel() for p in LanguageModel(c, "cst", 17).parameters()),
        )
        x = torch.arange(8)[None, :]
        torch.testing.assert_close(
            models["polar_live"](x)[0], models["polar_detached"](x)[0], atol=0, rtol=0
        )
        for mode, m in models.items():
            diag = probe_radius(m, x, x)
            for r in diag["layers"].values():
                self.assertGreater(r["physical_radius_partial"]["max_abs"], 0)
                self.assertEqual(
                    r["radius_task_gradient"]["max_abs"] > 0, mode == "polar_live"
                )
            opt = cst_optimizer(
                m,
                atom_lr_multiplier=16,
                lr=0.003,
                betas=(0.9, 0.95),
                weight_decay=0.0,
                foreach=False,
            )

            def step(model, optimizer):
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, x)
                loss.backward()
                optimizer.step()

            step(m, opt)
            for _, l in projections(m):
                n = l.atoms.p[:, :2].double().norm(dim=1)
                self.assertTrue(((n >= 1 - 5e-7) & (n <= 2 + 5e-7)).all())
            clone = LanguageModel(replace(c, radius_mode=mode), "cst", 17)
            clone.load_state_dict(copy.deepcopy(m.state_dict()))
            other = cst_optimizer(
                clone,
                atom_lr_multiplier=16,
                lr=0.003,
                betas=(0.9, 0.95),
                weight_decay=0.0,
                foreach=False,
            )
            other.load_state_dict(copy.deepcopy(opt.state_dict()))
            step(m, opt)
            step(clone, other)
            for a, b in zip(m.parameters(), clone.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            _assert_tree_equal(opt.state_dict(), other.state_dict())


if __name__ == "__main__":
    unittest.main()
