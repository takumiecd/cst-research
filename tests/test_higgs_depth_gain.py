"""CPU preparation checks; no GPU, learning-quality, or pilot-completion claim."""

import copy
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

from cst_research.experiments import higgs_depth_gain as g


class GainProtocolTests(unittest.TestCase):
    def test_adaptive_bounded_three_arm_scope(self):
        p = g.protocol()
        self.assertEqual(p["arms"], ["dense", "stock-all", "gain-all"])
        self.assertEqual(
            (p["divisor"], p["alpha"], p["epochs"], p["seed"]), (10000, 0.0001, 3, 17)
        )
        self.assertIn("Initially prepared alpha.001", p["adaptive_density_choice"])
        self.assertFalse(p["calibration"]["labels_used"])
        self.assertFalse(p["calibration"]["development_used"])
        self.assertFalse(p["calibration"]["clip_or_search"])
        self.assertFalse(p["formal_claim"])
        self.assertFalse(p["automatic_followup"])
        self.assertFalse(p["future_confirmation_evaluated"])
        self.assertFalse(p["official_test_evaluated"])

    def test_registry_matches_the_executable_preparation_protocol(self):
        manifest = (
            Path(__file__).resolve().parents[1]
            / "src/cst_research/studies/higgs-depth-gain-20261106.json"
        )
        study = json.loads(manifest.read_text())
        self.assertEqual(study["status"], "COMPLETED")
        self.assertEqual(
            study["runner_entrypoint"], "cst_research.experiments.higgs_depth_gain"
        )
        for key, value in g.protocol().items():
            self.assertEqual(study["protocol"][key], value, key)

    def test_gain_bounds_fail_instead_of_clipping_or_zero_target_replacement(self):
        self.assertEqual(g.bounded_gain(2.0, 0.25), 8.0)
        for target, observed in [
            (0.0, 1.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (float("nan"), 1.0),
            (1e8, 1.0),
            (1e-9, 1.0),
        ]:
            with self.assertRaises(g.CalibrationInconclusive):
                g.bounded_gain(target, observed)


SOURCE = os.environ.get("CST_DEPTH_SOURCE")


@unittest.skipUnless(SOURCE, "explicit pinned public975 source required")
class GainPublic975Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(SOURCE).resolve()
        manifest = json.loads(Path(os.environ["CST_DEPTH_MANIFEST"]).read_text())
        if manifest["source_commit"] != g.stock.SOURCE:
            raise ValueError("wrong external source")
        for n, h in manifest["files"].items():
            p = source / n
            if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
                raise ValueError("external source changed")
        sys.path[:0] = [str(source / "src"), str(source)]
        import torch
        import torchcst

        if not Path(torchcst.__file__).resolve().is_relative_to(source):
            raise ValueError("fresh pinned-source interpreter required")
        cls.torch = torch
        torch.set_num_threads(1)
        cls.features = torch.randn(
            (256, 28), generator=torch.Generator().manual_seed(20261106)
        )
        cls.dense = g.make_model("dense")
        cls.stock = g.make_model("stock-all")
        cls.gain = g.make_model("gain-all")
        cls.calibration = g.calibrate(cls.dense, cls.gain, cls.features)

    def test_same_trainables_nontrainable_gain_and_features_only_calibration(self):
        t = self.torch
        c = self.calibration
        self.assertEqual(g.parameter_hashes(self.stock), g.parameter_hashes(self.gain))
        self.assertEqual(len(list(self.gain.parameters())), 7)
        self.assertFalse(self.gain.fixed_output_gains.requires_grad)
        for n in ("0.weight", "0.bias", "8.weight", "8.bias"):
            self.assertEqual(
                g.parameter_hashes(self.dense)[n], g.parameter_hashes(self.gain)[n]
            )
        self.assertEqual(c["features_sha256"], g.tensor_sha(self.features))
        self.assertEqual(c["calibration_rows"], 256)
        self.assertTrue(c["trainable_tensors_unchanged"])
        self.assertFalse(c["labels_read"])
        self.assertFalse(c["development_read"])
        self.assertTrue(t.equal(self.stock.fixed_output_gains, t.ones(3)))
        for row in c["rows"]:
            self.assertLessEqual(
                abs(row["post_gain_reference_RMS"] / row["target_dense_RMS"] - 1), 1e-7
            )
        again = g.make_model("gain-all")
        self.assertEqual(g.calibrate(self.dense, again, self.features), c)
        with self.assertRaises(ValueError):
            g.calibrate(self.dense, again, self.features)
        with self.assertRaises(ValueError):
            g.calibrate(self.dense, g.make_model("gain-all"), self.features[:255])

    def test_ordinary_public_cpu_fp64_chain_and_independent_all_gradient_composition(
        self,
    ):
        t = self.torch
        m = copy.deepcopy(self.gain).double()
        x = t.randn(
            (4, 28),
            dtype=t.float64,
            generator=t.Generator().manual_seed(817),
            requires_grad=True,
        )
        # public forward on CPU vs separately composed dense_weight operators.
        # The frozen GPU-only execution plan intentionally rejects CPU. Replace
        # only its evaluation for this CPU algebra/placement test; actual CUDA
        # operators still require the fresh run-time full-network gate.
        from torchcst.nn import CSTModule

        for layer in m:
            if isinstance(layer, CSTModule):
                layer.forward = lambda value, site=layer: t.nn.functional.linear(
                    value, site.dense_weight()
                )
        y = m(x)
        reference = g.reference_forward(m, x)
        self.assertTrue(t.allclose(y, reference, rtol=1e-10, atol=1e-12))
        dy = t.tensor(
            [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]], dtype=t.float64
        )
        a = t.autograd.grad(y, (x, *m.parameters()), dy, retain_graph=True)
        b = t.autograd.grad(reference, (x, *m.parameters()), dy)
        self.assertEqual(len(a), 8)
        for actual, expected in zip(a, b):
            self.assertTrue(t.isfinite(actual).all())
            self.assertTrue(t.allclose(actual, expected, rtol=1e-8, atol=1e-11))

    def test_component_timer_synchronizes_before_and_after_without_gpu(self):
        from unittest.mock import patch

        events = []

        def fn(value):
            events.append("evaluate")
            return value + 1

        with (
            patch(
                "torch.cuda.synchronize",
                side_effect=lambda d: events.append("sync:" + str(d)),
            ),
            patch.object(g.time, "perf_counter", side_effect=[2.0, 5.0]),
        ):
            result, seconds = g.timed_cuda_call("mock-device", fn, 4)
        self.assertEqual((result, seconds), (5, 3.0))
        self.assertEqual(events, ["sync:mock-device", "evaluate", "sync:mock-device"])

    def test_gain_aware_trace_final_y_matches_model_and_reported_activations(self):
        t = self.torch
        m = copy.deepcopy(self.gain)
        from torchcst.nn import CSTModule

        for layer in m:
            if isinstance(layer, CSTModule):
                layer.forward = lambda value, site=layer: t.nn.functional.linear(
                    value, site.dense_weight()
                )
        x = self.features
        with t.no_grad():
            outputs = [m(batch) for batch in x.split(4)]
            traces = [g.activation_trace(m, batch) for batch in x.split(4)]
        for output, trace in zip(outputs, traces):
            self.assertTrue(t.equal(output, trace[-1]))
        diagnostic = g.layer_diagnostics(m, x)
        self.assertEqual(diagnostic["fixed_output_gains"], self.calibration["gains"])
        for i in range(9):
            expected = g.stock.tensor_statistics(t.cat([row[i] for row in traces]))
            self.assertEqual(diagnostic["activations"][str(i)], expected)
        self.assertEqual(
            diagnostic["activations"]["8"], g.stock.tensor_statistics(t.cat(outputs))
        )
        # Nonunit intervention must be visible at the first CST activation.
        with t.no_grad():
            pre = m[2](m[1](m[0](x[:4])))
        self.assertTrue(t.equal(traces[0][2], pre * m.fixed_output_gains[0]))
        self.assertFalse(t.equal(traces[0][2], pre))

    def test_positive_homogeneous_bias_free_branch_is_product_gain_oracle(self):
        t = self.torch
        gain = copy.deepcopy(self.gain).double()
        control = copy.deepcopy(self.stock).double()
        x = self.features[:4].double().requires_grad_()
        other = x.detach().clone().requires_grad_()
        actual = g.reference_forward(gain, x)
        from torchcst.nn import CSTModule

        hidden = other
        for layer in list(control)[:-1]:
            hidden = (
                t.nn.functional.linear(hidden, layer.dense_weight())
                if isinstance(layer, CSTModule)
                else layer(hidden)
            )
        product = gain.fixed_output_gains.prod()
        # Do not subtract a large head bias from a tiny stock output: that
        # avoidable cancellation loses the branch signal before amplification.
        expected = (
            t.nn.functional.linear(hidden, control[8].weight) * product
            + control[8].bias
        )
        self.assertTrue(t.allclose(actual, expected, rtol=1e-9, atol=1e-11))
        dy = t.tensor(
            [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]], dtype=t.float64
        )
        actual_grads = t.autograd.grad(actual, (x, *gain.parameters()), dy)
        expected_grads = t.autograd.grad(expected, (other, *control.parameters()), dy)
        for a, b in zip(actual_grads, expected_grads):
            self.assertTrue(t.allclose(a, b, rtol=1e-8, atol=1e-10))

    def test_stock_unit_gains_match_original_parameters_and_fp64_forward(self):
        t = self.torch
        m = copy.deepcopy(self.stock).double()
        original = g.stock.make_model("all", 10000).double()
        self.assertEqual(g.parameter_hashes(m), g.parameter_hashes(original))
        x = self.features[:4].double()
        self.assertTrue(
            t.equal(g.reference_forward(m, x), g.stock.reference_forward(original, x))
        )

    def test_gain_buffer_checkpoint_resume_and_cpu_optimizer_ownership(self):
        t = self.torch
        m = copy.deepcopy(self.gain)
        opt, _ = g.stock.make_optimizer(m, cpu_reference=True)

        def step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            # Controlled identical gradients exercise clocks/Polar update, not a model-quality test.
            for p in model.parameters():
                p.grad = t.full_like(p, 0.125)
            optimizer.step()

        for _ in range(4):
            step(m, opt)
        mid = {
            "model": copy.deepcopy(m.state_dict()),
            "optimizer": copy.deepcopy(opt.state_dict()),
        }
        for _ in range(4):
            step(m, opt)
        restored = g.make_model("gain-all", gains=self.calibration["gains"])
        other, _ = g.stock.make_optimizer(restored, cpu_reference=True)
        restored.load_state_dict(mid["model"])
        other.load_state_dict(mid["optimizer"])
        for _ in range(4):
            step(restored, other)

        def assert_nested(a, b, path=""):
            if isinstance(a, t.Tensor):
                self.assertTrue(t.equal(a, b), path)
            elif isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for k, v in a.items():
                    assert_nested(v, b[k], path + "/" + str(k))
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for i, (v, w) in enumerate(zip(a, b)):
                    assert_nested(v, w, path + "/" + str(i))
            elif path.endswith("atom_state._extra_state/layout_version"):
                self.assertEqual(
                    b, a + 1
                )  # documented load generation, not numerical state
            else:
                self.assertEqual(a, b, path)

        assert_nested(m.state_dict(), restored.state_dict())
        self.assertEqual(len(opt.state), 7)
        self.assertTrue(all(float(v["step"]) == 8 for v in opt.state.values()))
        for a, b in zip(opt.state.values(), other.state.values()):
            self.assertTrue(t.equal(a["exp_avg"], b["exp_avg"]))
            self.assertTrue(t.equal(a["exp_avg_sq"], b["exp_avg_sq"]))
        self.assertTrue(t.equal(m.fixed_output_gains, self.gain.fixed_output_gains))
        with self.assertRaises(ValueError):
            g.install_gains(self.stock, [2, 1, 1])
        for values in ([0, 1, 1], [1e7, 1, 1], [float("nan"), 1, 1]):
            with self.assertRaises(g.CalibrationInconclusive):
                g.install_gains(g.make_model("gain-all"), values)

    def test_no_epsilon_norm_and_small_column_negative_controls(self):
        t = self.torch
        r = t.tensor([1.0, -2.0, 3.0], dtype=t.float64)
        self.assertTrue(g.gradient_norm_gate(r, r)["pass"])
        self.assertFalse(g.gradient_norm_gate(t.zeros_like(r), r)["pass"])
        self.assertFalse(g.gradient_norm_gate(-r, r)["pass"])
        tiny = r * 1e-30
        self.assertFalse(g.gradient_norm_gate(t.zeros_like(tiny), tiny)["pass"])
        zeros = t.zeros(3, dtype=t.float64)
        self.assertTrue(g.gradient_norm_gate(zeros, zeros)["pass"])
        self.assertFalse(
            g.gradient_norm_gate(t.tensor([0, 0, 1e-30], dtype=t.float64), zeros)[
                "pass"
            ]
        )
        field = t.stack([r, r * 1e-10, r, r], 1)
        wrong = field.clone()
        wrong[:, 1] = 0
        self.assertTrue(g.gradient_norm_gate(wrong, field)["pass"])
        self.assertFalse(g.gradient_norm_gate(wrong[:, 1], field[:, 1])["pass"])
