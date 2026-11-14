"""Observer preserves training state and separates radius changes algebraically."""

import copy
import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torchcst"), "optional torchcst dependency"
)
class FractionUpdateTests(unittest.TestCase):
    def test_actual_updates_path_radius_decomposition_and_read_only_report(self):
        import torch
        from torch import nn

        from cst_research.experiments.ffn_fraction_updates import FractionUpdates
        from cst_research.experiments.ffn_radius_fraction import make_layer, optimizer
        from cst_research.experiments.ffn_radius_fraction_qualification import (
            assert_nested,
            config_for,
        )

        model = nn.Module()
        model.config = config_for(
            0.003,
            radius_fraction=0.5,
            amplitude_max=32.0,
            linear_backend="levels_wide_amplitude",
            radius_fraction_lr_multiplier=4.0,
        )
        model.map = make_layer(128, 128, config=model.config, std=0.001, seed=17)
        model.bias = nn.Parameter(torch.zeros(128))
        opt = optimizer(
            model, atom_lr_multiplier=16.0, lr=0.0003, weight_decay=0.0, foreach=False
        )
        observer = FractionUpdates(model)
        initial = model.map.atoms.p[:, 1].detach().clone()
        self.assertEqual(observer.report(opt)["steps_observed"], 0)
        x = torch.randn(3, 128, generator=torch.Generator().manual_seed(91))
        for _ in range(2):
            opt.zero_grad()
            (model.map(x) + model.bias).square().mean().backward()
            opt.step()
            observer.update()
        before = model.map.atoms.p.detach().clone()
        state = copy.deepcopy(opt.state_dict())
        gradient = model.map.atoms.p.grad.clone()
        rng = torch.get_rng_state().clone()
        record = observer.report(opt)
        row = record["layers"]["map"]
        self.assertEqual(record["steps_observed"], 2)
        self.assertAlmostEqual(row["effective_fraction_lr"], 0.0012)
        self.assertAlmostEqual(
            row["net_fraction_delta"]["mean_absolute"],
            float((before[:, 1] - initial).double().abs().mean()),
        )
        self.assertGreaterEqual(
            row["fraction_path_length"]["mean_absolute"],
            row["net_fraction_delta"]["mean_absolute"],
        )
        self.assertAlmostEqual(
            row["net_radius_delta"]["mean_signed"],
            row["radius_fraction_component"]["mean_signed"]
            + row["radius_amplitude_component"]["mean_signed"],
        )
        self.assertTrue(torch.equal(before, model.map.atoms.p))
        self.assertTrue(torch.equal(gradient, model.map.atoms.p.grad))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        assert_nested(state, opt.state_dict())
