"""Preparation tests, not GPU admission or evidence that a normalization learns."""

import copy
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

from cst_research.experiments import higgs_depth_normalization as n


class NormalizationPolicyTests(unittest.TestCase):
    def test_nine_cases_eight_task_updates_no_selection_or_heldout(self):
        p = n.protocol()
        self.assertEqual(len(p["cases"]), 9)
        self.assertEqual(p["training_updates"], 8)
        self.assertEqual(p["first_epoch_order_seed"], 117)
        self.assertEqual(p["normalization"]["eps"], 1e-5)
        self.assertFalse(p["normalization"]["affine"])
        self.assertEqual(p["state_steps"], [0, 1, 2, 4, 8])
        for field in (
            "development_evaluated",
            "future_confirmation_evaluated",
            "HIGGS_confirmation_evaluated",
            "official_test_evaluated",
            "selection_or_HPO",
            "automatic_followup",
            "formal_claim",
        ):
            self.assertFalse(p[field])

    def test_exact_authorized4090_and_unchanged_software_requirements(self):
        from cst_research.experiments.higgs_depth_normalization_run import (
            validate_runtime,
        )

        runtime = {
            "gpu": "NVIDIA GeForce RTX 4090",
            "capability": [8, 9],
            "multiprocessors": 128,
            "total_memory_bytes": 25262096384,
            "torch": "2.11.0+cu126",
            "cuda": "12.6",
            "triton": "3.6.0",
        }
        self.assertEqual(validate_runtime(runtime), "RTX4090")
        for key, value in [
            ("gpu", "NVIDIA GeForce RTX 4090 D"),
            ("capability", [8, 6]),
            ("multiprocessors", 127),
            ("total_memory_bytes", 25262096385),
            ("torch", "2.6.0+cu126"),
            ("cuda", "13.0"),
            ("triton", "3.7.0"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_runtime({**runtime, key: value})


SOURCE = os.environ.get("CST_DEPTH_SOURCE")


@unittest.skipUnless(SOURCE, "explicit source975 fixture required")
class NormalizationPublic975Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(SOURCE).resolve()
        manifest = json.loads(Path(os.environ["CST_DEPTH_MANIFEST"]).read_text())
        if manifest["source_commit"] != n.stock.SOURCE:
            raise ValueError("wrong public source")
        for name, digest in manifest["files"].items():
            path = source / name
            if (
                path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("source fixture changed")
        sys.path[:0] = [str(source / "src"), str(source)]
        import torch
        import torchcst

        if not Path(torchcst.__file__).resolve().is_relative_to(source):
            raise ValueError("fresh interpreter needs verified external975")
        cls.t = torch
        torch.set_num_threads(1)
        cls.x = torch.randn(
            (256, 28), generator=torch.Generator().manual_seed(20261106)
        )
        cls.models = {case: n.make_model(case) for case in n.CASES}
        cls.calibrations = {
            case: n.calibrate(cls.models["dense-stock"], model, cls.x)
            for case, model in cls.models.items()
        }

    def test_trainable_ownership_same_initials_and_sequential_calibration(self):
        t = self.t
        for case, model in self.models.items():
            with self.subTest(case=case):
                self.assertEqual(len(list(model.parameters())), 7)
                self.assertFalse(model.fixed_output_gains.requires_grad)
                matching = (
                    "dense-stock"
                    if case.startswith("dense")
                    else (
                        "cst1048-stock" if case == "cst1048-stock" else "cst104-stock"
                    )
                )
                self.assertEqual(
                    n.gain.parameter_hashes(model),
                    n.gain.parameter_hashes(self.models[matching]),
                )
                for row in self.calibrations[case]["rows"]:
                    if n.CASES[case][3]:
                        self.assertLess(
                            abs(row["post_gain_RMS"] / row["target_dense_RMS"] - 1),
                            1e-7,
                        )
                self.assertTrue(all(p.grad is None for p in model.parameters()))
        for m in self.models.values():
            for name in ("0.weight", "0.bias", "8.weight", "8.bias"):
                self.assertTrue(
                    t.equal(
                        m.state_dict()[name],
                        self.models["dense-stock"].state_dict()[name],
                    )
                )

    def test_normalization_epsilon_population_axes_sign_and_no_affine(self):
        t = self.t
        z = t.tensor(
            [[1.0, 2.0, -4.0, 0.0], [5.0, 5.0, 5.0, 5.0]],
            dtype=t.float64,
            requires_grad=True,
        )
        rms = n.norm(z, "rms")
        ln = n.norm(z, "ln")
        self.assertTrue(
            t.allclose(rms, z / (z.square().mean(1, keepdim=True) + 1e-5).sqrt())
        )
        self.assertTrue(
            t.allclose(ln, n.norm(z, "ln", reference=True), rtol=1e-13, atol=1e-13)
        )
        self.assertTrue(t.equal(rms.sign(), z.sign()))
        self.assertTrue(t.allclose(ln.mean(1), t.zeros(2, dtype=t.float64), atol=1e-15))
        self.assertTrue(t.equal(ln[1], t.zeros(4, dtype=t.float64)))
        self.assertTrue(
            t.isfinite(
                t.autograd.grad(ln.square().sum() + rms.square().sum(), z)[0]
            ).all()
        )

    def test_full_fp64_Y_dX_all_grad_independent_normalization_reference(self):
        t = self.t
        from torchcst.nn import CSTModule

        for case in ("dense-rms", "dense-ln", "cst104-rms", "cst104-ln"):
            model = copy.deepcopy(self.models[case]).double()
            # CPU algebra only: public dense_weight replaces GPU-only CUDA plan;
            # actual runtime operators require new GPU full-network admission.
            for site in model:
                if isinstance(site, CSTModule):
                    site.forward = lambda value, site=site: t.nn.functional.linear(
                        value, site.dense_weight()
                    )
            x = self.x[:4].double().clone().requires_grad_()
            y = model(x)
            ref = n.reference_forward(model, x)
            self.assertTrue(t.allclose(y, ref, rtol=1e-9, atol=1e-12), case)
            dy = t.tensor(
                [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]], dtype=t.float64
            )
            actual = t.autograd.grad(y, (x, *model.parameters()), dy, retain_graph=True)
            expected = t.autograd.grad(ref, (x, *model.parameters()), dy)
            for a, b in zip(actual, expected, strict=True):
                self.assertTrue(t.isfinite(a).all())
                self.assertTrue(t.allclose(a, b, rtol=1e-7, atol=1e-10), case)

    def test_post_relu_target_is_sequential_and_distinct_from_pre_target(self):
        post = self.calibrations["cst104-postgain"]
        pre = self.calibrations["cst104-gain"]
        self.assertNotEqual(post["gains"], pre["gains"])
        for row in post["rows"]:
            self.assertEqual(row["target_kind"], "post-ReLU")
            self.assertEqual(row["target_site"], row["site"] + 1)
            self.assertLess(
                abs(row["post_gain_post_relu_RMS"] / row["target_dense_RMS"] - 1), 1e-7
            )
        self.assertEqual(
            n.gain.parameter_hashes(self.models["cst104-postgain"]),
            n.gain.parameter_hashes(self.models["cst104-stock"]),
        )

    def test_recalibration_and_zero_signal_fail_closed(self):
        with self.assertRaises(ValueError):
            n.calibrate(self.models["dense-stock"], self.models["cst104-rms"], self.x)
        with self.assertRaises(ValueError):
            n.calibrate(
                self.models["dense-stock"], n.make_model("cst104-ln"), self.x[:255]
            )
        for target, observed in ((1.0, 0.0), (0.0, 1.0), (1.0, float("nan"))):
            with self.assertRaises(n.gain.CalibrationInconclusive):
                n.gain.bounded_gain(target, observed)

    def test_observation_native_cpu_equality_and_restores_seam_on_failure(self):
        from unittest.mock import patch

        from torchcst.nn import CSTModule

        from cst_research.experiments import higgs_depth_normalization_steps as recorder
        from cst_research.experiments.higgs_depth_normalization_run import same_state

        t = self.t
        for case in ("dense-ln", "cst104-ln"):
            plain, watched = [copy.deepcopy(self.models[case]) for _ in range(2)]
            for model in (plain, watched):
                for site in model:
                    if isinstance(site, CSTModule):
                        site.forward = lambda value, site=site: t.nn.functional.linear(
                            value, site.dense_weight()
                        )
            po, _ = n.stock.make_optimizer(plain, cpu_reference=True)
            wo, _ = n.stock.make_optimizer(watched, cpu_reference=True)
            # CPU-only generic CST retraction, native fused Adam CPU; no GPU claim.
            for opt in (po, wo):
                for group in opt.param_groups:
                    group["fused"] = True
            x = self.x[:4]
            labels = t.tensor([0, 1, 0, 1])
            for step in range(1, 3):
                po.zero_grad(set_to_none=True)
                t.nn.functional.cross_entropy(plain(x), labels).backward()
                po.step()
                summary, raw = recorder.observed_step(watched, wo, x, labels, step=step)
                self.assertTrue(
                    same_state(
                        n.stock.checkpoint(plain, po, 0, step, {}, None),
                        n.stock.checkpoint(watched, wo, 0, step, {}, None),
                    )
                )
                self.assertEqual(summary["step"], step)
                self.assertTrue(raw["raw_gradients"])
                self.assertNotIn("_adamw_proposal", wo.__dict__)
            if case == "cst104-ln":
                with (
                    patch.object(
                        type(wo),
                        "_adamw_proposal",
                        side_effect=RuntimeError("sentinel"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "sentinel"),
                ):
                    recorder.observed_step(watched, wo, x, labels, step=3)
                self.assertNotIn("_adamw_proposal", wo.__dict__)

    def test_denominator_scalar_uses_native_state_and_contract(self):
        import tempfile

        from cst_research.experiments.higgs_depth_normalization_run import (
            scalar_adam_gate,
        )

        # Source fixture is unchanged; ordinary CPU fused numerical plumbing only.
        with tempfile.TemporaryDirectory() as directory:
            rows = scalar_adam_gate(Path(directory), "cpu")
            self.assertEqual(len(rows), 8)
            self.assertTrue(
                all(
                    r["elementwise"]["pass"] and r["relative_norm"]["pass"]
                    for r in rows
                )
            )
            self.assertEqual(rows[0]["contract"]["eps"], 1e-8)


class DiagnosticDriverTests(unittest.TestCase):
    def test_unapproved_payload_refused_before_source_runtime_or_gpu(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch

        from cst_research.experiments import higgs_depth_normalization_driver as driver
        from cst_research.experiments import higgs_dst_confirmation_driver as common

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "research-inputs").mkdir()
            (root / "research-inputs/future-confirmation.npz").write_bytes(
                b"never opened"
            )
            with (
                patch.object(Path, "cwd", return_value=root),
                patch.object(common, "verify_consumer") as verify,
            ):
                with self.assertRaisesRegex(ValueError, "exact six"):
                    driver.run(SimpleNamespace(output=root / "fresh-output"))
                verify.assert_not_called()
            self.assertFalse((root / "fresh-output").exists())

    def test_foreground_timeout_reaps_owned_child(self):
        import subprocess
        import tempfile

        from cst_research.experiments.higgs_depth_normalization_driver import foreground

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root / "pid"
            command = [
                sys.executable,
                "-B",
                "-c",
                "import os,time,pathlib;pathlib.Path('pid').write_text(str(os.getpid()));time.sleep(30)",
            ]
            with (
                (root / "log").open("w") as log,
                self.assertRaises(subprocess.TimeoutExpired),
            ):
                foreground(
                    command,
                    cwd=root,
                    environment=os.environ.copy(),
                    log=log,
                    timeout=0.2,
                )
            pid = int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_registry_matches_measured_nine_case_protocol(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src/cst_research/studies/higgs-depth-normalization-diagnostic-20261106.json"
        )
        registered = json.loads(path.read_text())
        self.assertEqual(registered["protocol"], n.protocol())
        self.assertEqual(registered["status"], "COMPLETED")
        self.assertEqual(registered["preparation_status"], "PREPARED")

    def test_existing_cache_binding_mismatch_is_rejected_without_rewrite(self):
        import tempfile

        from cst_research.experiments.higgs_depth_normalization_driver import bind_cache

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bind_cache(root, {"runtime": {"torch": "pinned"}, "source": "frozen"})
            before = (root / "runtime-source-binding.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "binding differs"):
                bind_cache(root, {"runtime": {"torch": "changed"}, "source": "frozen"})
            self.assertEqual(
                (root / "runtime-source-binding.json").read_bytes(), before
            )
