"""Causal leakage, arm parity and FFN use; these are CPU checks, not GPU evidence."""

import unittest


class FFNModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import torchcst
        except ImportError as error:
            raise unittest.SkipTest('optional Torch/torchcst required') from error

    def test_causal_attention_has_no_future_token_leakage(self):
        import torch
        from cst_research.experiments.ffn_contract import ModelConfig
        from cst_research.experiments.ffn_model import LanguageModel
        config = ModelConfig(width=8, layers=1, heads=2, context=6, vocabulary=31)
        model = LanguageModel(config, 'dense').eval()
        x = torch.tensor([[1, 2, 3, 4, 5, 6]])
        other = x.clone(); other[:, 3:] = 7
        with torch.no_grad():
            a, b = model(x), model(other)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[:, 3:], b[:, 3:]))

    def test_independent_quadrature_covers_declared_polar_task_vjp(self):
        from cst_research.experiments.ffn_qualification import independent_quadrature_check
        result = independent_quadrature_check()
        self.assertEqual(result['atoms_checked'], 8)
        self.assertLess(result['max_dp_error'], 2e-9)

    def test_shared_parameters_match_and_cst_ffn_changes_predictions(self):
        import torch
        from cst_research.experiments.ffn_contract import ModelConfig
        from cst_research.experiments.ffn_model import LanguageModel
        config = ModelConfig(width=8, layers=1, heads=2, context=4, vocabulary=31)
        models = [LanguageModel(config, kind) for kind in ('dense', 'cst', 'none')]
        for name, p in models[2].named_parameters():
            for model in models[:2]:
                torch.testing.assert_close(p, dict(model.named_parameters())[name], rtol=0, atol=0)
        x = torch.tensor([[1, 2, 3, 4]])
        cst, without = models[1](x), models[2](x)
        self.assertGreater(float((cst - without).detach().abs().max()), 0)
        _, loss = models[1](x, x.roll(-1, -1))
        loss.backward()
        atoms = [p for name, p in models[1].named_parameters() if name.endswith('atoms.p')]
        self.assertEqual(len(atoms), 2)
        for p in atoms:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().max()), 0)
