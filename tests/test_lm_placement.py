"""Placement controls and diagnostic definitions, without GPU execution."""

import unittest

try:
    import torch
    import torchcst  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional Torch/torchcst")
class PlacementTests(unittest.TestCase):
    def test_matched_family_and_roles(self):
        from cst_research.experiments.lm_placement import ARMS, MATCHED, PAIRS

        self.assertEqual(len(PAIRS), 6)
        self.assertEqual(len(set(MATCHED.values())), 8)
        for sites in MATCHED.values():
            self.assertEqual(len(set(sites)), 4)
            self.assertEqual(sum(s.endswith(".up") for s in sites), 2)
            self.assertEqual(sum(s.endswith(".down") for s in sites), 2)
        self.assertEqual(ARMS["dst-pair12"][1], MATCHED["pair12"])

    def test_diagnostics_low_rank_and_invariant_rotation(self):
        from cst_research.experiments.lm_placement_probe import cosine, features

        a = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(features(a)["stable_rank"], 1.0)
        self.assertAlmostEqual(features(a)["entropy_rank"], 1.0)
        self.assertAlmostEqual(cosine(a, -a), -1.0)
        self.assertIsNone(cosine(a, torch.zeros_like(a)))
        self.assertIsNone(features(torch.ones(3, 2))["stable_rank"])

    def test_import_runner(self):
        from cst_research.experiments import lm_placement_run

        self.assertTrue(callable(lm_placement_run.train))

    def test_cka_recognizes_rotated_features_without_cosine_agreement(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts/analyze_lm_placement.py"
        spec = importlib.util.spec_from_file_location("placement_analysis", path)
        analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analysis)
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        self.assertAlmostEqual(analysis.cka(x, x @ rotation), 1.0)
        self.assertAlmostEqual(analysis.cosine(x, x @ rotation), 0.0)
        self.assertIsNone(analysis.cka(x, torch.ones_like(x)))
        with self.assertRaises(ValueError):
            analysis.cka(x, x[:2])


if __name__ == "__main__":
    unittest.main()
