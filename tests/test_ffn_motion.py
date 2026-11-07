"""Cumulative amplitude motion must detect reversals hidden by endpoints."""

import importlib
import math
import unittest


class AmplitudeMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            importlib.import_module("torch")
        except ImportError as error:
            raise unittest.SkipTest("optional Torch required") from error

    def test_reversal_has_nonzero_path_and_zero_net_motion_without_mutation(self):
        import torch

        from cst_research.experiments.ffn_motion import AmplitudeMotion

        parameter = torch.tensor([[0.1, math.sqrt(0.99), 0, 0]], dtype=torch.float64)
        original = parameter.clone()
        probe = AmplitudeMotion({"atom": (parameter, 1.0)})
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        parameter[0, :2] = torch.tensor([0.3, math.sqrt(0.91)], dtype=torch.float64)
        probe.update()
        parameter.copy_(original)
        probe.update()
        row = probe.report()["atom"]
        self.assertAlmostEqual(row["net_amplitude_change_rms"], 0)
        self.assertAlmostEqual(row["cumulative_absolute_amplitude_motion_rms"], 0.4)
        self.assertAlmostEqual(row["per_update_amplitude_motion_rms"], 0.2)
        self.assertEqual(row["sampled_updates"], 2)
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        self.assertFalse(probe.path["atom"].is_cuda)

    def test_pure_radial_motion_does_not_change_physical_amplitude(self):
        import torch

        from cst_research.experiments.ffn_motion import AmplitudeMotion

        parameter = torch.tensor([[0.6, 0.8, 0, 0]], dtype=torch.float64)
        probe = AmplitudeMotion({"atom": (parameter, 2.0)})
        parameter[:, :2] *= 1.5
        probe.update()
        row = probe.report()["atom"]
        self.assertAlmostEqual(row["current_amplitude_rms"], 1.2)
        self.assertAlmostEqual(row["cumulative_absolute_amplitude_motion_rms"], 0)

    def test_geometry_reads_physical_q_without_changing_model_values(self):
        try:
            importlib.import_module("torchcst")
        except ImportError as error:
            raise unittest.SkipTest("optional torchcst required") from error
        import torch

        from cst_research.experiments.ffn_contract import ModelConfig
        from cst_research.experiments.ffn_model import LanguageModel
        from cst_research.experiments.ffn_motion import geometry_snapshot

        model = LanguageModel(
            ModelConfig(width=8, layers=1, heads=2, context=4, vocabulary=31, q=0.5)
        )
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        rows = geometry_snapshot(model)
        self.assertEqual(len(rows), 2)
        for row in rows.values():
            self.assertAlmostEqual(row["q_mean"], 0.5, places=6)
            self.assertGreaterEqual(row["input_radius_mean"], 0.5)
            self.assertLessEqual(row["output_radius_mean"], 4.0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
