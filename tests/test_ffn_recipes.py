"""Recipe admission, fixed-feature-order equivalence and actual model use."""

import copy
import importlib
import unittest

from cst_research.experiments.ffn_contract import (
    ModelConfig,
    qualification_covers_model,
)


class RecipeAdmissionTests(unittest.TestCase):
    def test_each_kernel_scalar_and_order_needs_its_own_qualification(self):
        config = ModelConfig()
        physical = {
            k: getattr(config, k) for k in ("q", "amplitude_max", "w_c", "radius_birth")
        }
        receipt = {
            "task_recipes": [config.recipe()],
            "cases": [
                {"in": ni, "out": no, "coefficient": 8, "atoms": 4096, **physical}
                for ni, no in [(256, 1024), (1024, 256)]
            ],
        }
        self.assertTrue(qualification_covers_model(receipt, config))
        for kwargs in [
            {"q": 0.5},
            {"amplitude_max": 4.0},
            {"w_c": 1.2},
            {"radius_birth": 1.0},
            {"feature_order": "permuted"},
            {"coefficient": 16},
            {"atom_lr_multiplier": 16},
        ]:
            self.assertFalse(qualification_covers_model(receipt, ModelConfig(**kwargs)))
        broken = copy.deepcopy(receipt)
        broken["cases"][0]["atoms"] = 1
        self.assertFalse(qualification_covers_model(broken, config))
        for kwargs in [
            {"q": True},
            {"amplitude_max": float("inf")},
            {"w_c": 0},
            {"radius_birth": 0.1},
            {"feature_order": "unknown"},
        ]:
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)


class FeatureOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            importlib.import_module("torch")
            importlib.import_module("torchcst")
        except ImportError as error:
            raise unittest.SkipTest("optional Torch/torchcst required") from error

    def test_new_radius_scalars_pass_independent_polynomial_quadrature(self):
        from cst_research.experiments.ffn_qualification import (
            independent_quadrature_check,
        )

        for settings in [
            {"q": 0.0, "radius_birth": 1.0},
            {"q": 1.0, "amplitude_max": 4.0, "w_c": 1.2},
        ]:
            row = independent_quadrature_check(**settings)
            self.assertLess(row["max_dp_error"], 2e-9)

    def test_actual_width_and_aspect_qualification_covers_both_target_scales(self):
        import math

        from cst_research.experiments.ffn_contract import atom_count
        from cst_research.experiments.ffn_qualification import recipe_case_specs

        for width, expansion in [(128, 1), (128, 4), (256, 4), (512, 4), (128, 16)]:
            config = ModelConfig(width=width, expansion=expansion, q=0.5)
            rows = recipe_case_specs(width=width, expansion=expansion, q=0.5)
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [r[:2] for r in rows],
                [
                    (config.width, config.intermediate),
                    (config.intermediate, config.width),
                ],
            )
            self.assertEqual([r[-1] for r in rows], [0.02, 0.02 / math.sqrt(8)])
            receipt = {
                "task_recipes": [config.recipe()],
                "cases": [
                    {
                        "in": ni,
                        "out": no,
                        "atoms": atom_count(ni, no, c),
                        "coefficient": c,
                        "q": q,
                        "amplitude_max": 1.0,
                        "w_c": 0.3,
                        "radius_birth": 4.0,
                    }
                    for ni, no, _, c, q, _ in rows
                ],
            }
            self.assertTrue(qualification_covers_model(receipt, config))
            if width != 256 or expansion != 4:
                self.assertFalse(
                    qualification_covers_model(receipt, ModelConfig(q=0.5))
                )

    def test_atom_lr_group_does_not_change_ordinary_parameter_rates(self):
        for multiplier in (16.0, 32.0, 64.0):
            with self.subTest(multiplier=multiplier):
                self.check_atom_lr_group(multiplier)

    def check_atom_lr_group(self, multiplier):
        from torchcst import CSTLinear

        from cst_research.experiments.ffn_model import LanguageModel, cst_optimizer

        model = LanguageModel(
            ModelConfig(width=8, layers=1, heads=2, context=4, vocabulary=31)
        )
        optimizer = cst_optimizer(
            model, atom_lr_multiplier=multiplier, lr=0.0003, foreach=False
        )
        atoms = {
            id(module.atoms.p)
            for module in model.modules()
            if isinstance(module, CSTLinear)
        }
        seen = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                seen.append(id(parameter))
                expected = multiplier if id(parameter) in atoms else 1.0
                self.assertEqual(group["lr_scale"], expected)
                self.assertAlmostEqual(group["lr"], 0.0003 * expected)
        self.assertEqual(set(seen), {id(p) for p in model.parameters()})
        self.assertEqual(len(seen), len(set(seen)))

    def test_dense_can_absorb_fixed_orders_without_additional_parameters(self):
        import torch

        from cst_research.experiments.ffn_model import LanguageModel

        config = ModelConfig(width=8, layers=1, heads=2, context=4, vocabulary=31)
        plain = LanguageModel(config, "dense")
        ordered = LanguageModel(
            ModelConfig(**{**vars(config), "feature_order": "permuted"}), "dense"
        )
        a, b = plain.blocks[0].ffn, ordered.blocks[0].ffn
        self.assertEqual(
            sum(p.numel() for p in a.parameters()),
            sum(p.numel() for p in b.parameters()),
        )
        with torch.no_grad():
            inverse = torch.argsort(b.hidden_order)
            a.down.map.weight.copy_(b.down.map.weight[b.output_order][:, inverse])
            a.down.bias.copy_(b.down.bias[b.output_order])
        x = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(811))
        torch.testing.assert_close(a(x), b(x), atol=1e-8, rtol=2e-5)

    def test_permuted_cst_is_causal_and_serializes_its_actual_orders(self):
        import torch

        from cst_research.experiments.ffn_model import LanguageModel

        config = ModelConfig(
            width=8,
            layers=1,
            heads=2,
            context=4,
            vocabulary=31,
            feature_order="permuted",
        )
        model = LanguageModel(config, "cst")
        restored = LanguageModel(config, "cst", seed=101)
        restored.load_state_dict(model.state_dict())
        x = torch.tensor([[1, 2, 3, 4]])
        changed = x.clone()
        changed[:, 2:] = 7
        torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)
        torch.testing.assert_close(
            model(x)[:, :2], model(changed)[:, :2], atol=0, rtol=0
        )
        _, loss = model(x, x.roll(-1, -1))
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.endswith("atoms.p"):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().max()), 0)
        for ffn in [model.blocks[0].ffn]:
            torch.testing.assert_close(
                ffn.hidden_order.sort().values, torch.arange(config.intermediate)
            )
            torch.testing.assert_close(
                ffn.output_order.sort().values, torch.arange(config.width)
            )
