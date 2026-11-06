"""Device-cohort boundary and unchanged numerical-function ownership; CPU only."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from cst_research.experiments import higgs_multilayer_conditioning_rtx4090_run as cohort
from cst_research.experiments import higgs_multilayer_conditioning_run as base


class RTX4090CohortPolicy(unittest.TestCase):
    def test_exact_device_and_runtime_pin_required_not_generic_cuda(self):
        runtime = {
            "python": "3.12.9",
            "torch": "2.11.0+cu126",
            "cuda": "12.6",
            "triton": "3.6.0",
            "numpy": "2.0.2",
            "driver": "570.133.20",
            "gpu": "NVIDIA GeForce RTX 4090",
            "capability": [8, 9],
            "multiprocessors": 128,
            "total_memory_bytes": 25262096384,
        }
        cohort.runtime_gate(runtime, runtime)
        for key, value in (
            ("gpu", "NVIDIA A100 80GB PCIe MIG 3g.40gb"),
            ("multiprocessors", 42),
            ("capability", [8, 0]),
            ("total_memory_bytes", 25296044032),
            ("torch", "2.11.0+cu130"),
            ("driver", "changed"),
        ):
            changed = {**runtime, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                cohort.runtime_gate(changed, runtime)

    def test_only_device_cohort_metadata_changes_in_shared_six_arm_protocol(self):
        actual, original = cohort.protocol(), base.protocol()
        replacements = {
            "version",
            "study_id",
            "runtime_pin_sha256",
            "device",
            "retention_route",
            "numerical_source_commit",
            "numerical_modules_sha256",
            "cross_device_scope",
        }
        self.assertEqual(
            {k: v for k, v in actual.items() if k not in replacements},
            {k: v for k, v in original.items() if k not in replacements},
        )
        self.assertNotEqual(
            actual["runtime_pin_sha256"], original["runtime_pin_sha256"]
        )
        self.assertNotEqual(actual["version"], original["version"])
        self.assertEqual(actual["total_task_updates"], 150000)
        self.assertEqual(actual["qualification"]["total_synthetic_updates"], 120)
        self.assertFalse(actual["automatic_retry_resume_or_advance"])

    def test_numerical_inventory_tamper_fails_before_import_or_execution(self):
        root = Path(__file__).resolve().parents[1]
        cohort.verify_numerical_source(root)
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            with self.assertRaisesRegex(ValueError, "unchanged a43d"):
                cohort.verify_numerical_source(candidate)
            for name in cohort.NUMERICAL_MODULES:
                p = candidate / "src/cst_research/experiments" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(
                    (root / "src/cst_research/experiments" / name).read_bytes()
                )
            cohort.verify_numerical_source(candidate)
            p.write_bytes(p.read_bytes() + b"\n# tamper\n")
            with self.assertRaisesRegex(ValueError, "unchanged a43d"):
                cohort.verify_numerical_source(candidate)

    def test_unstaged_payload_refused_without_opening_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "future-confirmation.npz").write_bytes(b"not a dataset")
            with self.assertRaisesRegex(ValueError, "exact6regularinputs"):
                cohort.verify_inputs(root)
        self.assertEqual(cohort.input_pins()["runtime-pin.json"], cohort.RUNTIME_SHA)

    def test_native_qualification_precedes_only_shared_case_runner(self):
        tree = ast.parse(Path(cohort.__file__).read_text())
        run = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"
        )
        calls = [
            (n.lineno, ast.unparse(n.func))
            for n in ast.walk(run)
            if isinstance(n, ast.Call)
        ]
        qualify = [line for line, name in calls if name == "qualification.qualify_all"]
        train = [line for line, name in calls if name == "base.run_case"]
        self.assertEqual(len(qualify), 1)
        self.assertEqual(len(train), 1)
        self.assertLess(qualify[0], train[0])
        self.assertFalse(any(name in ("setattr", "exec", "eval") for _, name in calls))
        self.assertFalse(
            any(
                isinstance(n, ast.FunctionDef) and n.name == "run_case"
                for n in tree.body
            )
        )

    def test_registered_protocol_and_source_boundary(self):
        root = Path(__file__).resolve().parents[1]
        record = json.loads(
            (
                root / "src/cst_research/studies" / (cohort.STUDY_ID + ".json")
            ).read_text()
        )
        self.assertEqual(record["protocol"], cohort.protocol())
        self.assertEqual(record["status"], "COMPLETED")
        self.assertEqual(record["execution"]["task_updates"], 150000)
        self.assertFalse(record["execution"]["confirmation_or_test_evaluated"])
        self.assertEqual(record["source_commit"], base.stock.SOURCE)
        self.assertEqual(
            record["runner_entrypoint"],
            "cst_research.experiments.higgs_multilayer_conditioning_rtx4090_driver",
        )
