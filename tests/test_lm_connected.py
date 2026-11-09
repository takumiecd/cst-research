"""Read-only diagnostic arithmetic, with no GPU required."""

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
    import torchcst  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional Torch/torchcst")
class ConnectedDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/analyze_lm_connected.py"
        spec = importlib.util.spec_from_file_location("connected_diagnostic", path)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_wrapped_delta_and_exact_field_endpoint(self):
        old = torch.tensor([[0.3, 0.0, 127.9, 255.9]])
        new = torch.tensor([[0.4, 0.2, 0.1, 0.1]])
        v = self.m.displacement(old, new, 256, 128)
        self.assertTrue(torch.all(v[:, 2:].abs() < 0.3))
        for k in range(4):
            point = self.m.candidate(old, new, v, k, 1.0, 256, 128)
            self.assertTrue(torch.equal(point[:, k], new[:, k]))
            for j in range(4):
                if j != k:
                    self.assertTrue(torch.equal(point[:, j], old[:, j]))

    def test_linear_agreement_and_zero_direction(self):
        d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        s = self.m.statistics(d, d, -d)
        self.assertAlmostEqual(s["linear_finite_cosine"], 1.0)
        self.assertEqual(s["relative_defect_to_prediction"], 0.0)
        s = self.m.statistics(d, torch.zeros_like(d), -d)
        self.assertIsNone(s["linear_finite_cosine"])
        self.assertIsNone(s["finite_over_prediction_norm"])


if __name__ == "__main__":
    unittest.main()
