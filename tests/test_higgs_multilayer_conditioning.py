"""CPU policy, independent LN reverse and actual public-model calibration contracts."""

import copy
import math
import os
import unittest

from cst_research.experiments import higgs_multilayer_conditioning as study
from cst_research.experiments import (
    higgs_multilayer_conditioning_reference as reference,
)


class ConditioningPolicy(unittest.TestCase):
    def test_six_declared_H8_arms_no_hidden_affine_or_Atom_budget_change(self):
        self.assertEqual(len(study.CASES), 6)
        self.assertEqual(study.MILESTONES, (0, 1000, 5000, 10000, 15000, 20000, 25000))
        for case in study.CASES:
            cfg = study.configuration(case)
            self.assertEqual(cfg["epsilon"], 1e-5)
            self.assertFalse(cfg["trainable_affine_in_LN"])
            self.assertEqual(cfg["factory_configuration"]["hidden_layers"], 8)
            self.assertEqual(
                cfg["factory_configuration"]["internal_module_indices"],
                [2, 4, 6, 8, 10, 12, 14],
            )
            self.assertEqual(cfg["factory_configuration"]["trainable_tensor_count"], 11)
            if cfg["method"] == "cst":
                self.assertEqual(
                    cfg["factory_configuration"]["atoms_per_cst_layer"], 4096
                )

    def test_strict_A100_runtime_and_unstaged_payload_guards(self):
        import tempfile
        from pathlib import Path

        from cst_research.experiments import higgs_multilayer_conditioning_run as run

        runtime = {
            "torch": "2.11.0+cu126",
            "cuda": "12.6",
            "triton": "3.6.0",
            "gpu": "NVIDIA A100 80GB PCIe MIG 3g.40gb",
            "capability": [8, 0],
            "multiprocessors": 42,
            "total_memory_bytes": 42144366592,
        }
        run.runtime_gate(runtime, runtime)
        for key, value in (
            ("gpu", "NVIDIA A100 80GB PCIe"),
            ("multiprocessors", 108),
            ("torch", "2.11.0+cu130"),
        ):
            wrong = {**runtime, key: value}
            with self.assertRaises(ValueError):
                run.runtime_gate(wrong, wrong)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "future-confirmation.npz").write_bytes(b"mustnotdecode")
            with self.assertRaisesRegex(ValueError, "exact6"):
                run.verify_inputs(root)
        p = run.protocol()
        self.assertEqual(p["total_task_updates"], 150000)
        self.assertEqual(p["qualification"]["total_synthetic_updates"], 120)
        self.assertFalse(p["qualification"]["amplified_optimizer_step"])
        self.assertFalse(p["automatic_retry_resume_or_advance"])


@unittest.skipUnless(
    os.environ.get("CST_DEPTH_SOURCE"),
    "explicit verified public975 CPU fixture required",
)
class PublicConditioningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_higgs_linear_atom_scaling import PublicFactoryChecks

        PublicFactoryChecks.setUpClass.__func__(cls)

    def model(self, case, *, width=128, hidden=4, gains=None):
        return study.make_model(case, width=width, hidden=hidden, gains=gains)

    def relative(self, actual, expected):
        t = self.t
        scale = float(expected.detach().abs().max())
        if scale == 0:
            self.assertTrue(t.equal(actual, expected))
            return
        self.assertLess(
            float(
                t.linalg.vector_norm((actual - expected.detach()) / scale)
                / t.linalg.vector_norm(expected.detach() / scale)
            ),
            1e-8,
        )

    def test_manual_LN_reverse_matches_independent_functional_autograd_and_epsilon_floor(
        self,
    ):
        t = self.t
        for value in (
            t.randn(4, 8, dtype=t.float64),
            t.full((4, 8), 2.0, dtype=t.float64),
        ):
            x = value.requires_grad_()
            dy = t.randn_like(x)
            expected = t.nn.functional.layer_norm(
                x, (8,), weight=None, bias=None, eps=1e-5
            )
            dx = t.autograd.grad(expected, x, dy)[0]
            actual, cache = reference.normalize(x.detach(), "ln")
            self.relative(actual, expected)
            self.relative(reference.norm_reverse(dy, cache), dx)
            self.assertFalse(
                t.allclose(reference.norm_reverse(dy, cache), dy, rtol=1e-6, atol=1e-10)
            )
        tiny = t.tensor([[1e-12, -1e-12] * 4] * 4, dtype=t.float64)
        normed, _ = reference.normalize(tiny, "ln")
        self.assertLess(float(normed.square().mean()), 1e-15)
        self.assertAlmostEqual(
            float(normed[0, 0] / tiny[0, 0]), 1 / math.sqrt(1e-5), places=8
        )
        with self.assertRaises(ValueError):
            reference.normalize(tiny, "auto")

    def test_full_manual_chunk_reverse_all_fields_matches_public_full_autograd(self):
        t = self.t
        for case in study.CASES:
            width = 8 if case.startswith("dense") else 128
            model = self.model(case, width=width).double()
            if study.CASES[case][2]:
                study.install_gains(model, [0.7, 1.2, 2.0])
            x = t.randn(4, 28, dtype=t.float64, requires_grad=True)
            dy = t.randn(4, 2, dtype=t.float64)
            # Independent full autograd uses public dense exports and actual functional LN,
            # without changing the factory's intentionally CUDA-only execution selector.
            from torchcst.nn import CSTModule

            y = x
            sites = model._linear_atom_configuration["internal_module_indices"]
            for index, layer in enumerate(model):
                y = (
                    t.nn.functional.linear(y, layer.dense_weight())
                    if isinstance(layer, CSTModule)
                    else layer(y)
                )
                if index in sites:
                    y = study.normalization.norm(
                        y, model._conditioning_configuration["normalization"]
                    )
                    y = y * model.fixed_output_gains[sites.index(index)]
            expected = (y, *t.autograd.grad(y, (x, *model.parameters()), dy))
            yy, dx, dp = reference.full_reference(model, x.detach(), dy, chunk_atoms=31)
            self.assertEqual(list(dp), [n for n, _ in model.named_parameters()])
            for actual, wanted in zip((yy, dx, *dp.values()), expected, strict=True):
                self.relative(actual, wanted)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_matched_raw_trainables_and_sequential_sameDense_pre_targets_no_labels(
        self,
    ):
        t = self.t
        dense = self.model("dense-none")
        features = t.randn(256, 28, generator=t.Generator().manual_seed(719))
        candidate = self.model("cst-fixed-gain")
        ln = self.model("cst-ln-gain")
        baseline = {n: p.detach().clone() for n, p in candidate.named_parameters()}
        for name, point in ln.named_parameters():
            self.assertTrue(t.equal(point, baseline[name]))
        a = study.calibrate(dense, candidate, features)
        b = study.calibrate(dense, ln, features)
        self.assertEqual(
            [r["target_Dense_stock_pre_ReLU_RMS"] for r in a["rows"]],
            [r["target_Dense_stock_pre_ReLU_RMS"] for r in b["rows"]],
        )
        for model, record in ((candidate, a), (ln, b)):
            self.assertFalse(record["labels_or_development_used"])
            for name, point in model.named_parameters():
                self.assertTrue(t.equal(point, baseline[name]))
            for row in record["rows"]:
                self.assertLess(
                    abs(
                        row["post_gain_pre_ReLU_RMS"]
                        / row["target_Dense_stock_pre_ReLU_RMS"]
                        - 1
                    ),
                    1e-6,
                )
            with self.assertRaises(ValueError):
                study.calibrate(dense, model, features)
        with self.assertRaises(ValueError):
            study.install_gains(self.model("cst-bare-ln"), [2, 2, 2])
        bad = self.model("cst-fixed-gain")
        with t.no_grad():
            bad[0].weight.zero_()
            bad[0].bias.zero_()
        with self.assertRaises(study.gain_policy.CalibrationInconclusive):
            study.calibrate(dense, bad, features)
        self.assertTrue(t.equal(bad.fixed_output_gains, t.ones(3)))
        with self.assertRaises(ValueError):
            study.calibrate(
                dense, self.model("cst-fixed-gain"), t.full((256, 28), float("nan"))
            )

    def test_native_seven_owner_resume_and_stable_torus_delta(self):
        from cst_research.experiments import (
            higgs_multilayer_conditioning_observation as observation,
        )
        from cst_research.experiments import (
            higgs_multilayer_conditioning_state as state,
        )

        t = self.t
        model = self.model("cst-ln-gain", hidden=8, gains=[1.1] * 7)
        optimizer, _ = state.make_optimizer(model, cpu_reference=True)
        for point in model.parameters():
            point.grad = t.full_like(point, 0.001)
        optimizer.step()
        state.check_optimizer(model, optimizer, 1)
        saved = state.snapshot(model, optimizer, 1, {"fixture": "CPU-only"})
        loaded, optim, _, proof = state.load(saved, "cpu", cpu_reference=True)
        self.assertEqual(
            set(proof["loaded"]),
            {f"{i}.atom_state._extra_state" for i in range(2, 16, 2)},
        )
        for candidate in (model, loaded):
            for point in candidate.parameters():
                point.grad = t.full_like(point, -0.002)
        optimizer.step()
        optim.step()
        state.check_optimizer(loaded, optim, 2)
        a = state.snapshot(model, optimizer, 2, {})
        b = state.snapshot(loaded, optim, 2, {})
        state.exact.assert_equal(
            state.core_state(b),
            state.qualified_state.with_generations(a, proof["loaded"]),
            "actualCPUresume",
        )
        self.assertEqual(
            proof["actual_postload_core_sha256"],
            proof["expected_only_native_generations_adjusted_core_sha256"],
        )
        original_digest = state.core_digest(saved)
        for field in (
            "parameter",
            "gradient",
            "m",
            "v",
            "clock",
            "metadata",
            "row_ids",
        ):
            damaged = copy.deepcopy(saved)
            if field == "parameter":
                damaged["model"]["0.weight"].reshape(-1)[0] += 1
            elif field == "gradient":
                damaged["gradients"]["0.weight"].reshape(-1)[0] += 1
            elif field in ("m", "v", "clock"):
                actual_state = next(iter(damaged["optimizer"]["state"].values()))
                key = {"m": "exp_avg", "v": "exp_avg_sq", "clock": "step"}[field]
                actual_state[key].reshape(-1)[0] += 1
            elif field == "metadata":
                damaged["epoch"] += 1
            else:
                original_sites = damaged["optimizer"]["cst"]["sites"]
                site = list(original_sites[0])
                ids = list(site[2])
                ids[0] += 1
                site[2] = type(site[2])(ids)
                sites = list(original_sites)
                sites[0] = type(original_sites[0])(site)
                damaged["optimizer"]["cst"]["sites"] = type(original_sites)(sites)
            self.assertNotEqual(state.core_digest(damaged), original_digest, field)
        bad = copy.deepcopy(saved)
        bad["model"]["2.atom_state._extra_state"]["layout_version"] = -1
        with self.assertRaises(ValueError):
            state.load(bad, "cpu", cpu_reference=True)
        bad = copy.deepcopy(saved)
        next(iter(bad["gradients"].values())).fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            state.load(bad, "cpu", cpu_reference=True)
        tiny = 2.0**-47
        delta = observation.stable_displacement(
            t.tensor([0.0, 0.0], dtype=t.float64),
            t.tensor([tiny, 1024.0], dtype=t.float64),
            period=1024,
        )
        self.assertEqual(delta["max_abs"], tiny)
        self.assertEqual(delta["physical_nonzero_fraction"], 0.5)
        self.assertEqual(delta["encoded_changed_fraction"], 1.0)

    def test_retain_grad_trace_backward_does_not_change_plain_Dense_parameter_gradients(
        self,
    ):
        from cst_research.experiments import (
            higgs_multilayer_conditioning_observation as observation,
        )

        t = self.t
        for case in ("dense-none", "dense-bare-ln"):
            model = self.model(case, width=8, hidden=8)
            plain = copy.deepcopy(model)
            x = t.randn(4, 28)
            labels = t.tensor([0, 1, 0, 1])
            _, _, summary, raw = observation.forward_backward(model, x, labels)
            t.nn.functional.cross_entropy(plain(x), labels).backward()
            for (_, a), (_, b) in zip(
                model.named_parameters(), plain.named_parameters(), strict=True
            ):
                self.assertTrue(t.equal(a.grad, b.grad))
            self.assertIn("2:raw-square", summary["backward"])
            self.assertTrue(t.equal(raw["inputs"], x))
            with t.no_grad(), self.assertRaisesRegex(ValueError, "grad-enabled"):
                observation.forward_backward(model, x, labels)

    def test_prediction_row_binding_is_independent_of_cached_row_ids(self):
        import numpy as np

        from cst_research.experiments import higgs_multilayer_conditioning_run as run

        actual = run.prediction_arrays(
            "development",
            np.zeros((50000, 2), dtype=np.float32),
            np.zeros(50000, dtype=np.int64),
        )
        self.assertEqual(
            (actual["row_ids"][0], actual["row_ids"][-1]), (1100000, 1149999)
        )
        damaged = {k: v.copy() for k, v in actual.items()}
        damaged["row_ids"][0] += 1
        with self.assertRaises(ValueError):
            run.helpers.assert_prediction_arrays(actual, damaged)
        with self.assertRaises(ValueError):
            run.prediction_arrays(
                "development",
                np.zeros((4, 2), dtype=np.float32),
                np.zeros(4, dtype=np.int64),
            )

    def test_reference_rejects_wrong_dtype_nonfinite_and_target_shape(self):
        t = self.t
        model = self.model("dense-bare-ln", width=8).double()
        x = t.randn(4, 28, dtype=t.float64)
        dy = t.ones(4, 2, dtype=t.float64)
        for xx, dd in (
            (x.float(), dy),
            (x, t.ones(4, 3, dtype=t.float64)),
            (x, t.full_like(dy, float("nan"))),
        ):
            with self.assertRaises(ValueError):
                reference.full_reference(model, xx, dd)
        bad = copy.deepcopy(model)
        bad._conditioning_configuration["epsilon"] = 1e-8
        with self.assertRaises(ValueError):
            reference.full_reference(bad, x, dy)


if __name__ == "__main__":
    unittest.main()
