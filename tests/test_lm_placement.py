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


if __name__ == "__main__":
    unittest.main()
