"""Operational boundaries; these stdlib tests do not claim GPU correctness."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cst_research.experiments import higgs_gpu_comparison as experiment


class GPUComparisonTests(unittest.TestCase):
    def test_exact_reserved_mig_is_required(self):
        valid = {
            "name": experiment.GPU_NAME,
            "capability": [8, 0],
            "sms": 42,
            "memory_bytes": experiment.GPU_BYTES,
        }
        experiment.require_device(valid, 1)
        for field, value in (
            ("name", "NVIDIA A100 80GB PCIe"),
            ("sms", 108),
            ("memory_bytes", 85899345920),
            ("capability", [8, 9]),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                experiment.require_device(valid | {field: value}, 1)
        with self.assertRaises(ValueError):
            experiment.require_device(valid, 2)

    def test_wrong_single_counter_cannot_pass_byte_gate(self):
        experiment.require_trajectory({"all_byte_equal": True}, 16, 16, [16] * 5, 16)
        for left, right, clocks, equal in (
            (15, 16, [16] * 5, True),
            (16, 15, [16] * 5, True),
            (16, 16, [16] * 4, True),
            (16, 16, [15] * 5, True),
            (16, 16, [16] * 5, False),
        ):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                experiment.require_trajectory(
                    {"all_byte_equal": equal}, left, right, clocks, 16
                )

    def test_frozen_source_tamper_missing_extra_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            file = root / "runner.py"
            file.write_text("immutable source")
            manifest = Path(directory) / "manifest.json"
            record = {
                "source_commit": "a" * 40,
                "archive_sha256": "b" * 64,
                "files": {"runner.py": hashlib.sha256(file.read_bytes()).hexdigest()},
            }
            manifest.write_text(json.dumps(record))
            experiment.verify_source(root, manifest, "a" * 40)
            file.write_text("changed")
            with self.assertRaises(ValueError):
                experiment.verify_source(root, manifest, "a" * 40)
            file.write_text("immutable source")
            extra = root / "untracked.py"
            extra.write_text("unexpected")
            with self.assertRaises(ValueError):
                experiment.verify_source(root, manifest, "a" * 40)
            extra.unlink()
            file.unlink()
            with self.assertRaises(ValueError):
                experiment.verify_source(root, manifest, "a" * 40)
            record["files"] = {"../outside.py": "c" * 64}
            manifest.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                experiment.verify_source(root, manifest, "a" * 40)

    def test_budgets_and_data_partition_are_fixed(self):
        for kind in ("dense", "cst"):
            pilot = experiment.command(kind, "graph", Path("data.npz"), Path("fresh"))
            qualification = experiment.command(
                kind, "eager", Path("data.npz"), Path("qual"), True
            )
            self.assertEqual(pilot[pilot.index("--epochs") + 1], "1")
            self.assertNotIn("--steps-per-epoch", pilot)
            self.assertEqual(
                qualification[qualification.index("--development-rows") + 1], "32"
            )
            self.assertEqual(
                qualification[qualification.index("--steps-per-epoch") + 1], "16"
            )
            self.assertFalse(any("evaluate" in arg for arg in pilot + qualification))
            self.assertEqual(pilot[pilot.index("--width") + 1], "2048")
            self.assertEqual(pilot[pilot.index("--divisor") + 1], "1000")


if __name__ == "__main__":
    unittest.main()
