"""Field-rate scheduling and non-mutating envelope diagnostics."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torchcst"), "optional torchcst dependency"
)
class RadiusFractionTests(unittest.TestCase):
    def test_fraction_rate_tracks_group_schedule_once_and_probe_conserves_atoms(self):
        import torch
        from torch import nn

        from cst_research.experiments.ffn_radius_fraction import (
            make_layer,
            optimizer,
            snapshot,
        )
        from cst_research.experiments.ffn_radius_fraction_qualification import (
            config_for,
        )

        model = nn.Module()
        model.config = config_for(0.03)
        model.map = make_layer(128, 128, config=model.config, std=0.001, seed=17)
        model.ordinary = nn.Parameter(torch.zeros(1))
        opt = optimizer(
            model, atom_lr_multiplier=16.0, lr=0.0003, weight_decay=0.0, foreach=False
        )
        self.assertAlmostEqual(opt.get_field_lrs()[-1]["radius_fraction"], 0.0003)
        for group in opt.param_groups:
            group["lr"] *= 0.25
        self.assertAlmostEqual(
            opt.get_field_lrs()[-1]["radius_fraction"], 0.0003 * 0.25
        )
        before = model.map.atoms.p.detach().clone()
        rng = torch.get_rng_state().clone()
        result = snapshot(model)["map"]
        self.assertEqual(sum(b["count"] for b in result["bins"]), len(before))
        self.assertEqual(result["all"]["fraction_at_zero"], 1.0)
        self.assertTrue(torch.equal(before, model.map.atoms.p))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_resume_allows_only_declared_layout_invalidation(self):
        import copy

        import torch
        from torch import nn

        from cst_research.experiments.ffn_radius_fraction import make_layer
        from cst_research.experiments.ffn_radius_fraction_qualification import (
            assert_model_resume,
            config_for,
        )

        model = nn.Module()
        model.config = config_for(0.03)
        model.map = make_layer(128, 128, config=model.config, std=0.001, seed=17)
        restored = copy.deepcopy(model)
        restored.load_state_dict(model.state_dict())
        transitions = assert_model_resume(model, restored)
        self.assertEqual(list(transitions.values()), [[0, 1]])
        with torch.no_grad():
            restored.map.atoms.p[0, 0] += 0.001
        with self.assertRaises(AssertionError):
            assert_model_resume(model, restored)
