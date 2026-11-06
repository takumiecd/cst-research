"""Proposal semantics, ordinary Adam rates, checkpoint identity and task resume."""

import copy
import importlib
import io
import unittest


class FieldRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch = importlib.import_module("torch")
            importlib.import_module("torchcst")
        except ImportError as error:
            raise unittest.SkipTest("optional Torch/torchcst required") from error
        cls.torch.set_num_threads(2)

    def layer(self):
        from torch import nn
        from cst_research.experiments.ffn_model import make_cst

        return nn.Sequential(
            make_cst(16, 64, dtype=self.torch.float64), nn.Linear(64, 3).double()
        )

    def assert_states(self, a, oa, b, ob, *, atol=1e-13):
        torch = self.torch
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, atol=atol, rtol=1e-12)
            self.assertEqual(set(oa.state[pa]), set(ob.state[pb]))
            for key in oa.state[pa]:
                torch.testing.assert_close(
                    oa.state[pa][key], ob.state[pb][key], atol=atol, rtol=1e-12
                )

    def test_unit_block_ratios_match_stock_cst_adamw(self):
        from torchcst import CSTAdamW
        from cst_research.experiments.ffn_field_lr import FieldRateOptimizer

        torch = self.torch
        a = self.layer()
        b = copy.deepcopy(a)
        oa = FieldRateOptimizer(a, native=False)
        ob = CSTAdamW(
            [
                {"params": [b[0].atoms.p], "lr": 0.0048},
                {"params": list(b[1].parameters()), "lr": 0.0003},
            ],
            model=b,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
        )
        rng = torch.Generator().manual_seed(501)
        for _ in range(5):
            for pa, pb in zip(a.parameters(), b.parameters()):
                pa.grad = torch.randn(pa.shape, generator=rng, dtype=pa.dtype)
                pb.grad = pa.grad.clone()
            oa.step()
            ob.step()
            self.assert_states(a, oa, b, ob, atol=0)

    def test_scaled_proposal_matches_independent_adam_equations(self):
        from cst_research.experiments.ffn_field_lr import (
            FieldRateOptimizer,
            FieldRateRecipe,
        )
        from torchcst._backends.torch.kernels import execution
        from torchcst.optim.state import OptimizerStateAdapter

        torch = self.torch
        a = self.layer()
        b = copy.deepcopy(a)
        recipe = FieldRateRecipe(amplitude=0.0003)
        opt = FieldRateOptimizer(a, recipe, native=False)
        states = {
            p: {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
            for p in b.parameters()
        }
        rng = torch.Generator().manual_seed(502)
        # Include changing gradient signs, transport, bias corrections and zero-LR schedule.
        for step, factor in enumerate((0.5, 1.0, 0.25, 0.0, 0.8), 1):
            for group in opt.param_groups:
                group["lr"] = recipe.ordinary * factor * group["lr_scale"]
            for pa, pb in zip(a.parameters(), b.parameters()):
                grad = torch.randn(pa.shape, generator=rng, dtype=pa.dtype)
                pa.grad = grad.clone()
                is_atom = pb is b[0].atoms.p
                old = pb.detach().clone()
                # These fixed Euclidean/torus charts pass raw gradients through.
                # The Polar law projects the Adam displacement during retraction.
                state = states[pb]
                state["exp_avg"].mul_(0.9).add_(grad, alpha=0.1)
                state["exp_avg_sq"].mul_(0.95).addcmul_(grad, grad, value=0.05)
                rate = (recipe.outer_clock if is_atom else recipe.ordinary) * factor
                delta = (
                    -rate
                    * (state["exp_avg"] / (1 - 0.9**step))
                    / ((state["exp_avg_sq"] / (1 - 0.95**step)).sqrt() + 1e-8)
                )
                with torch.no_grad():
                    proposal = old + delta
                    if is_atom:
                        proposal[:, :2] = (
                            old[:, :2]
                            + (proposal[:, :2] - old[:, :2])
                            * recipe.amplitude
                            / recipe.outer_clock
                        )
                    pb.copy_(proposal)
                    if is_atom and rate:
                        # CPU declared retraction is separate from proposal/Adam arithmetic.
                        new = execution.apply_parameter_update(
                            b[0].kernel,
                            *b[0].cst_charts(),
                            old,
                            pb - old,
                            step_size=rate,
                        )
                        pb.copy_(new)
                        OptimizerStateAdapter(("exp_avg",)).transport(
                            b[0], old, new, state
                        )
            opt.step()
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, atol=2e-13, rtol=1e-12)
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        opt.state[pa][key], states[pb][key], atol=2e-13, rtol=1e-12
                    )
                self.assertEqual(int(opt.state[pa]["step"]), step)

    def test_task_serialization_rebuilds_hooks_and_rejects_other_recipe(self):
        from cst_research.experiments.ffn_contract import ModelConfig
        from cst_research.experiments.ffn_model import LanguageModel
        from cst_research.experiments.ffn_field_lr import (
            FieldRateOptimizer,
            FieldRateRecipe,
        )

        torch = self.torch
        config = ModelConfig(width=8, layers=1, heads=2, context=4, vocabulary=31)
        model = LanguageModel(config)
        recipe = FieldRateRecipe(amplitude=0.0003)
        opt = FieldRateOptimizer(model, recipe, native=False)
        x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])

        def step(m, o):
            o.zero_grad()
            _, loss = m(x, x.roll(-1, -1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()
            return loss.detach()

        step(model, opt)
        buf = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, buf)
        buf.seek(0)
        payload = torch.load(buf, weights_only=True)
        restored = LanguageModel(config, seed=99)
        restored.load_state_dict(payload["model"])
        other = FieldRateOptimizer(restored, recipe, native=False)
        other.load_state_dict(payload["optimizer"])
        torch.testing.assert_close(
            step(model, opt), step(restored, other), atol=0, rtol=0
        )
        self.assert_states(model, opt, restored, other, atol=0)
        wrong = FieldRateOptimizer(copy.deepcopy(restored), native=False)
        before = copy.deepcopy(wrong.state_dict())
        with self.assertRaisesRegex(ValueError, "recipe differs"):
            wrong.load_state_dict(payload["optimizer"])
        from cst_research.experiments.ffn_qualification import _assert_tree_equal

        _assert_tree_equal(wrong.state_dict(), before)

    def test_rejects_invalid_rate_and_unsupported_scope(self):
        from cst_research.experiments.ffn_field_lr import (
            FieldRateOptimizer,
            FieldRateRecipe,
        )

        for value in (True, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FieldRateRecipe(amplitude=value)
        with self.assertRaisesRegex(ValueError, "CUDA FP32"):
            FieldRateOptimizer(self.layer())
        with self.assertRaises(ValueError):
            FieldRateOptimizer(self.torch.nn.Linear(2, 3), native=False)

    def test_controlled_fixture_copies_before_binding_different_state_contracts(self):
        from cst_research.experiments.ffn_field_lr import (
            FieldRateOptimizer,
            FieldRateRecipe,
        )
        from cst_research.experiments.ffn_field_lr_qualification import (
            controlled_models,
        )
        from torchcst import CSTAdamW

        recipe = FieldRateRecipe()
        cpu, live, stock = controlled_models(recipe, device="cpu")
        FieldRateOptimizer(cpu, recipe, native=False)
        FieldRateOptimizer(live, recipe, native=False)
        # Binding the stock optimizer must succeed independently of the wrappers.
        CSTAdamW(stock.parameters(), model=stock, weight_decay=0, foreach=False)
