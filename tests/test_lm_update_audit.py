"""Known-answer tests for task update directions and endpoint locality."""

import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class UpdateAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "audit_lm_updates",
            Path(__file__).resolve().parents[1] / "scripts/audit_lm_updates.py",
        )
        cls.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.audit)

    def test_descent_alignment_and_zero_update_known_answers(self):
        import torch

        g = torch.tensor([3.0, 4.0])
        v = self.audit.directional_statistics([g], [-2 * g])
        self.assertAlmostEqual(v["predicted_loss_reduction"], 50)
        self.assertAlmostEqual(v["descent_cosine"], 1)
        v = self.audit.directional_statistics([g], [torch.zeros_like(g)])
        self.assertIsNone(v["descent_cosine"])
        self.assertEqual(v["predicted_loss_reduction"], 0)

    def test_endpoint_restores_complement_and_two_layer_interaction(self):
        import torch
        from torch import nn

        m = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(1)
        old = self.audit.clone_state(m)
        new = {n: p + 0.1 for n, p in old.items()}
        x = torch.ones(1, 1)

        def loss():
            return float((m(x) - 2).square().sum().detach())

        before = loss()
        self.audit.endpoint(m, old, new, ["0.weight"])
        first = loss()
        self.assertEqual(float(m[1].weight.detach()), 1)
        self.audit.endpoint(m, old, new, ["1.weight"])
        second = loss()
        self.assertEqual(float(m[0].weight.detach()), 1)
        self.audit.endpoint(m, old, new, list(old))
        joint = loss()
        self.assertAlmostEqual(before, 1)
        self.assertAlmostEqual(first, 0.81, places=6)
        self.assertAlmostEqual(second, 0.81, places=6)
        self.assertAlmostEqual(joint, 0.6241, places=6)
        self.assertLess((before - joint) - (before - first) - (before - second), 0)
        self.audit.endpoint(m, old, new, [])
        self.assertEqual(loss(), before)

    def test_packed_field_endpoint_changes_only_requested_column(self):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.atoms = nn.Module()
                self.atoms.p = nn.Parameter(torch.arange(12.0).reshape(3, 4))
                self.bias = nn.Parameter(torch.ones(1))

        m = Model()
        old = self.audit.clone_state(m)
        new = {n: p + 10 for n, p in old.items()}
        self.audit.endpoint(m, old, new, ["atoms.p"], field=1)
        expected = old["atoms.p"].clone()
        expected[:, 1] += 10
        self.assertTrue(torch.equal(m.atoms.p, expected))
        self.assertTrue(torch.equal(m.bias, old["bias"]))
