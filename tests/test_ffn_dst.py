"""Independent dense oracles and exact state semantics for the DST reference."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class DSTTests(unittest.TestCase):
    def test_rectangular_raw_and_active_derivatives(self):
        import torch

        from cst_research.experiments.ffn_dst import MaskedLinear

        for ni, no in ((11, 7), (7, 11)):
            layer = MaskedLinear(
                torch.randn(no, ni, dtype=torch.float64),
                no * 3 + 2,
                torch.Generator().manual_seed(7),
            )
            x = torch.randn(5, ni * 2, dtype=torch.float64)[:, ::2].requires_grad_()
            dy = torch.randn(5, no, dtype=torch.float64)
            y = layer(x)
            y.backward(dy)
            raw = dy.T @ x.detach()
            torch.testing.assert_close(
                x.grad, dy @ layer.weight.detach(), atol=1e-12, rtol=1e-12
            )
            torch.testing.assert_close(
                y,
                x.detach() @ (layer.weight.detach() * layer.mask).T,
                atol=1e-12,
                rtol=1e-12,
            )
            layer.prepare_gradient()
            torch.testing.assert_close(
                layer.dense_gradient, raw, atol=1e-12, rtol=1e-12
            )
            torch.testing.assert_close(
                layer.weight.grad, raw * layer.mask, atol=1e-12, rtol=1e-12
            )

    def test_topology_ties_reset_clock_and_all_parameter_skip(self):
        import torch
        from torch import nn

        from cst_research.experiments.ffn_dst import (
            MaskedLinear,
            finish_step,
            invariants,
            make_optimizer,
        )

        model = nn.Module()
        model.map = MaskedLinear(
            torch.ones(3, 8), 12, torch.Generator().manual_seed(11)
        )
        model.ordinary = nn.Parameter(torch.ones(1))
        opt = make_optimizer(model, 0.001)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        finish_step(model, opt, 1, 976)
        before = model.ordinary.clone()
        clock = opt.state[model.ordinary]["step"].clone()
        # Equal magnitudes force row rescue; tied growth must prefer low columns.
        with torch.no_grad():
            model.map.weight.copy_(model.map.mask)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        finish_step(model, opt, 250, 976)
        torch.testing.assert_close(model.ordinary, before, atol=0, rtol=0)
        torch.testing.assert_close(
            opt.state[model.ordinary]["step"], clock, atol=0, rtol=0
        )
        self.assertTrue(model.map.mask[:, :4].all())
        invariants(model, opt)

    def test_model_budget_shared_values_and_exact_resume(self):
        import io

        import torch

        from cst_research.experiments.ffn_contract import ModelConfig
        from cst_research.experiments.ffn_dst import (
            finish_step,
            invariants,
            make_model,
            make_optimizer,
        )
        from cst_research.experiments.ffn_model import LanguageModel

        config = ModelConfig(width=32, layers=1, heads=4, context=4, vocabulary=31)
        model = make_model(config)
        dense = LanguageModel(config, "dense", 17)
        for n, p in model.named_parameters():
            if ".ffn." not in n or not n.endswith("map.weight"):
                torch.testing.assert_close(
                    p, dict(dense.named_parameters())[n], atol=0, rtol=0
                )
        self.assertEqual(
            sum(v["active_slots"] for v in invariants(model).values()), 4096
        )
        opt = make_optimizer(model, 0.001)
        x = torch.tensor([[1, 2, 3, 4]])

        def step(m, o, k):
            o.zero_grad()
            m(x, x.roll(-1, -1))[1].backward()
            finish_step(m, o, k, 976)

        step(model, opt, 1)
        for k in (2, 250):
            b = io.BytesIO()
            torch.save({"m": model.state_dict(), "o": opt.state_dict()}, b)
            b.seek(0)
            s = torch.load(b, weights_only=False)
            other = make_model(config)
            other.load_state_dict(s["m"])
            oo = make_optimizer(other, 0.001)
            oo.load_state_dict(s["o"])
            step(model, opt, k)
            step(other, oo, k)
            for n, p in model.state_dict().items():
                torch.testing.assert_close(p, other.state_dict()[n], atol=0, rtol=0)
            for a, c in zip(opt.state.values(), oo.state.values()):
                for key in a:
                    torch.testing.assert_close(a[key], c[key], atol=0, rtol=0)
