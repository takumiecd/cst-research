"""Validate the read-only diagnostic against ordinary task autograd."""

import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class SignalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "audit_lm_signals",
            Path(__file__).resolve().parents[1] / "scripts/audit_lm_signals.py",
        )
        cls.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.audit)

    def test_uniform_attention_and_centered_feature_rank_have_known_answers(self):
        import torch

        v = self.audit.attention_statistics(torch.zeros(1, 4, 24), 2)
        self.assertAlmostEqual(v["entropy_over_log_available_keys"], 1.0, places=6)
        self.assertAlmostEqual(
            v["maximum_probability_mean"], (1 + 0.5 + 1 / 3 + 0.25) / 4, places=6
        )
        a = self.audit.statistics(torch.eye(4), spectrum=True)
        self.assertAlmostEqual(
            a["sample_feature_spectrum"]["participation_rank"], 3.0, places=5
        )
        z = self.audit.statistics(torch.zeros(4, 8), spectrum=True)
        self.assertEqual(z["sample_feature_spectrum"]["participation_rank"], 0)
        self.assertEqual(z["exact_zero_fraction"], 1.0)

    def test_hooks_reproduce_task_gradients_without_parameter_or_rng_changes(self):
        from types import SimpleNamespace

        import torch
        from torch import nn
        from torch.nn import functional as F

        class FFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.Linear(8, 16)
                self.down = nn.Linear(16, 8)

            def forward(self, x):
                return self.down(F.gelu(self.up(x)))

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(8)
                self.norm2 = nn.LayerNorm(8)
                self.qkv = nn.Linear(8, 24)
                self.attention_out = nn.Linear(8, 8)
                self.ffn = FFN()

            def forward(self, x):
                b, t, d = x.shape
                q, k, v = [
                    a.reshape(b, t, 2, 4).transpose(1, 2)
                    for a in self.qkv(self.norm1(x)).chunk(3, -1)
                ]
                a = (
                    F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
                    .transpose(1, 2)
                    .reshape(b, t, d)
                )
                x = x + self.attention_out(a)
                return x + self.ffn(self.norm2(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(heads=2)
                self.embedding = nn.Embedding(11, 8)
                self.blocks = nn.ModuleList([Block()])
                self.final_norm = nn.LayerNorm(8)

            def forward(self, x, y):
                z = self.embedding(x)
                for b in self.blocks:
                    z = b(z)
                logits = F.linear(self.final_norm(z), self.embedding.weight)
                return logits, F.cross_entropy(logits.flatten(0, 1), y.flatten())

        torch.manual_seed(7)
        m = Model()
        x = torch.tensor([[1, 2, 3, 4]])
        y = x.roll(-1, -1)
        _, loss = m(x, y)
        loss.backward()
        expected = {
            n: float(p.grad.double().square().mean().sqrt())
            for n, p in m.named_parameters()
        }
        digest = self.audit.digest_parameters(m)
        rng = torch.get_rng_state().clone()
        observed = self.audit.probe(m, x, y)
        self.assertEqual(observed["loss"], float(loss.detach()))
        for name, value in expected.items():
            self.assertAlmostEqual(
                observed["parameter_gradients"][name]["rms"], value, places=10
            )
        self.assertIn("input_gradient", observed["layers"]["blocks.0.ffn.down"])
        self.assertIn("output_gradient", observed["layers"]["blocks.0.qkv"])
        self.assertEqual(digest, self.audit.digest_parameters(m))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_probe_calls_respect_the_qualified_native_row_limit(self):
        scopes = [v for step in (0, 976, 4882) for v in self.audit.probe_scopes(step)]
        self.assertEqual(len(scopes), 6)
        self.assertTrue(all(len(starts) * 128 <= 256 for _, starts in scopes))
        self.assertEqual(
            [starts[0] for _, starts in self.audit.probe_scopes(4882)],
            [0, 4096, 8192, 12288],
        )
