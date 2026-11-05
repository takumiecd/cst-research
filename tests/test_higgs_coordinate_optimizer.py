"""Pinned-source CPU arithmetic admission. No CUDA execution or data access."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

from cst_research.experiments.higgs_coordinate_oracle import adam_scalar, retract_scalar

SOURCE = os.environ.get("CST_COORDINATE_LR_SOURCE")


class ScalarOracleTests(unittest.TestCase):
    def test_coordinate_scaling_does_not_enter_polar_law(self):
        prior = [0.2, 1.2, 1023.999, 0.0001]
        a = retract_scalar(prior, [0.21, 1.199, 1024.002, -0.0002])
        b = retract_scalar(prior, [0.21, 1.199, 1024.02, -0.002])
        self.assertEqual(a[:2], b[:2])
        self.assertGreater(a[2], 0.0)
        self.assertLess(a[2], 0.01)
        self.assertGreater(a[3], 1023.0)


@unittest.skipUnless(
    SOURCE, "set CST_COORDINATE_LR_SOURCE to an explicitly verified975 source"
)
class PositionOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(SOURCE).resolve()
        manifest_path = Path(os.environ["CST_COORDINATE_LR_MANIFEST"])
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["source_commit"] != "97580582ed28c4e494bf5c2656bbe94a85c544c3"
            or manifest["archive_sha256"]
            != "edbb585de56e304887b118069b7c7ecd9e45ae5fe2e08a5afca828ad8544eb5f"
        ):
            raise ValueError("tests require exact external975 archive inventory")
        for name, digest in manifest["files"].items():
            path = source / name
            if (
                path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("external source byte mismatch")
        sys.path[:0] = [str(source / "src"), str(source)]
        import torch
        import torchcst

        if not Path(torchcst.__file__).resolve().is_relative_to(source):
            raise ValueError("run adapter tests in a fresh Python with pinned source")
        from benchmarks.training.higgs_confirmation import make_model

        from cst_research.experiments.higgs_coordinate_optimizer import (
            PositionLRCSTAdamW,
        )

        cls.torch, cls.make_model, cls.optimizer_type = (
            torch,
            staticmethod(make_model),
            PositionLRCSTAdamW,
        )
        torch.set_num_threads(1)

    def pair(self, ratio, dtype=None, force=False, fused=True):
        torch = self.torch
        dtype = dtype or torch.float64
        model = self.make_model(1024, 1000, 17, "cst").to(dtype)
        with torch.no_grad():
            point = model[2].atoms.p
            point[:, 0] = 0.2
            point[:, 1] = 1.2
            point[:, 2] = 513.00006103515625
            point[:, 3] = 1023.99999
        groups = [
            {
                "params": [*model[0].parameters(), *model[4].parameters()],
                "weight_decay": 0.01,
            },
            {"params": [point], "weight_decay": 0.0, "amplitude_weight_decay": 0.01},
        ]
        optimizer = self.optimizer_type(
            groups,
            model=model,
            position_multiplier=ratio,
            force_recompute_for_qualification=force,
            lr=0.0003,
            fused=fused,
            check_finite=True,
        )
        return model, optimizer

    def gradients(self, model, step):
        torch = self.torch
        generator = torch.Generator().manual_seed(937 + step)
        for point in model.parameters():
            point.grad = (
                torch.randn(point.shape, generator=generator, dtype=point.dtype) * 0.05
            )

    def assert_tensor_state(self, left, right):
        torch = self.torch

        def tensors(value, prefix=""):
            if isinstance(value, torch.Tensor):
                yield prefix, value
            elif isinstance(value, dict):
                for key in value:
                    yield from tensors(value[key], prefix + "/" + str(key))
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    yield from tensors(item, prefix + "/" + str(index))

        a, b = dict(tensors(left)), dict(tensors(right))
        self.assertEqual(a.keys(), b.keys())
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_scale1_delegates_exact_native_parameters_gradients_moments_and_clocks(
        self,
    ):
        from torchcst import CSTAdamW

        model, optimizer = self.pair(1.0, self.torch.float32)
        native = self.make_model(1024, 1000, 17, "cst").to(self.torch.float32)
        with self.torch.no_grad():
            native[2].atoms.p.copy_(model[2].atoms.p)
        baseline = CSTAdamW(
            [
                {
                    "params": [*native[0].parameters(), *native[4].parameters()],
                    "weight_decay": 0.01,
                },
                {
                    "params": [native[2].atoms.p],
                    "weight_decay": 0.0,
                    "amplitude_weight_decay": 0.01,
                },
            ],
            model=native,
            lr=0.0003,
            fused=True,
            check_finite=True,
        )
        for step in range(6):
            self.gradients(model, step)
            self.gradients(native, step)
            optimizer.step()
            baseline.step()
            self.assert_tensor_state(model.state_dict(), native.state_dict())
            self.assert_tensor_state(optimizer.state_dict(), baseline.state_dict())
            self.assertTrue(
                all(
                    self.torch.equal(a.grad, b.grad)
                    for a, b in zip(model.parameters(), native.parameters())
                )
            )
        self.assertIsNone(optimizer._last_position_clock)

    def test_actual_fp32_center_ulp_uses_direct_lr_and_forced1_matches(self):
        torch = self.torch
        small, opt = self.pair(0.1, torch.float32)
        control, base = self.pair(1.0, torch.float32)
        forced, forced_opt = self.pair(1.0, torch.float32, force=True)
        prior = small[2].atoms.p[:, 2].detach().clone()
        for model in (small, control, forced):
            for p in model.parameters():
                p.grad = torch.ones_like(p)
        opt.step()
        base.step()
        forced_opt.step()
        self.assertTrue(torch.equal(small[2].atoms.p[:, 2], prior))
        self.assertFalse(torch.equal(control[2].atoms.p[:, 2], prior))
        self.assert_tensor_state(control.state_dict(), forced.state_dict())
        self.assert_tensor_state(base.state_dict(), forced_opt.state_dict())
        self.assertEqual(
            float(opt._last_position_clock), float(opt.state[small[2].atoms.p]["step"])
        )

    def test_scalar_adam_plus_independent_polar_retraction_and_torus_crossing(self):
        for ratio in (0.1, 1.0, 10.0):
            with self.subTest(ratio=ratio):
                model, opt = self.pair(ratio)
                point = model[2].atoms.p
                expected = point[0].detach().tolist()
                means = [0.0] * 4
                squares = [0.0] * 4
                for step in range(1, 9):
                    self.gradients(model, step)
                    gradient = point.grad[0].detach().tolist()
                    proposal = []
                    for col in range(4):
                        value, means[col], squares[col] = adam_scalar(
                            expected[col],
                            gradient[col],
                            means[col],
                            squares[col],
                            step,
                            0.0003 * (ratio if col >= 2 else 1.0),
                        )
                        proposal.append(value)
                    expected = retract_scalar(
                        expected,
                        proposal,
                        radial_regularization=float(
                            model[2].kernel.scalar("radial_regularization")
                        ),
                    )
                    opt.step()
                    self.assertTrue(
                        self.torch.allclose(
                            point[0], point.new_tensor(expected), rtol=1e-12, atol=2e-12
                        )
                    )
                    self.assertTrue(
                        self.torch.allclose(
                            opt.state[point]["exp_avg"][0],
                            point.new_tensor(means),
                            rtol=1e-12,
                            atol=1e-14,
                        )
                    )
                    self.assertEqual(float(opt.state[point]["step"]), step)
                    self.assertGreaterEqual(float(point[:, 3].detach().min()), 0.0)
                    self.assertLess(float(point[:, 3].detach().max()), 1024.0)

    def test_resume_identical_and_ratio_mismatch_rejected(self):
        for ratio in (0.1, 1.0, 10.0):
            model, opt = self.pair(ratio, self.torch.float32)
            for step in range(4):
                self.gradients(model, step)
                opt.step()
            saved_model = copy.deepcopy(model.state_dict())
            saved_opt = copy.deepcopy(opt.state_dict())
            resumed, reloaded = self.pair(ratio, self.torch.float32)
            resumed.load_state_dict(saved_model)
            reloaded.load_state_dict(saved_opt)
            for step in range(4, 9):
                self.gradients(model, step)
                self.gradients(resumed, step)
                opt.step()
                reloaded.step()
                self.assert_tensor_state(model.state_dict(), resumed.state_dict())
                self.assert_tensor_state(opt.state_dict(), reloaded.state_dict())
            wrong, wrongopt = self.pair(
                10.0 if ratio != 10.0 else 0.1, self.torch.float32
            )
            wrong.load_state_dict(saved_model)
            with self.assertRaisesRegex(ValueError, "checkpoint contract differs"):
                wrongopt.load_state_dict(saved_opt)

    def test_recomputation_never_changes_polar_or_ordinary_proposals(self):
        models = []
        for ratio in (0.1, 1.0, 10.0):
            models.append(self.pair(ratio, self.torch.float32))
        for step in range(5):
            for model, opt in models:
                self.gradients(model, step)
                opt.step()
            base, baseopt = models[1]
            for model, opt in (models[0], models[2]):
                self.assertTrue(
                    self.torch.equal(model[2].atoms.p[:, :2], base[2].atoms.p[:, :2])
                )
                for i in (0, 4):
                    self.assert_tensor_state(
                        model[i].state_dict(), base[i].state_dict()
                    )
                self.assert_tensor_state(
                    opt.state_dict()["state"], baseopt.state_dict()["state"]
                )

    def test_resume_restores_saved_lr_instead_of_cached_constructor_group(self):
        model, opt = self.pair(0.1, self.torch.float32)
        for step in range(3):
            self.gradients(model, step)
            opt.step()
        saved_model = copy.deepcopy(model.state_dict())
        saved_opt = copy.deepcopy(opt.state_dict())
        resumed, reloaded = self.pair(0.1, self.torch.float32)
        for group in reloaded.param_groups:
            group["lr"] = 0.2
        resumed.load_state_dict(saved_model)
        reloaded.load_state_dict(saved_opt)
        self.assertTrue(all(g["lr"] == 0.0003 for g in reloaded.param_groups))
        for step in range(3, 7):
            self.gradients(model, step)
            self.gradients(resumed, step)
            opt.step()
            reloaded.step()
            self.assert_tensor_state(model.state_dict(), resumed.state_dict())
            self.assert_tensor_state(opt.state_dict(), reloaded.state_dict())
