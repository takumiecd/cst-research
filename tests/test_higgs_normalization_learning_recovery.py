"""Pure amendment/transport-boundary checks; no fabricated passed parent or GPU."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cst_research.experiments import higgs_normalization_learning_recovery as recovery
from cst_research.experiments import (
    higgs_normalization_learning_recovery_driver as driver,
)


class RecoveryPolicy(unittest.TestCase):
    def test_300k_parent_plus_only_five_fresh_cases_fixed_original_budget(self):
        p = recovery.protocol()
        self.assertEqual(
            p["parent_main_updates"] + p["recovery_main_updates"],
            p["combined_main_updates"],
        )
        self.assertEqual(p["combined_main_updates"], 675000)
        self.assertEqual(p["fresh_remaining_cases"], list(recovery.parent.CASES[4:]))
        self.assertEqual(
            len(p["fresh_remaining_cases"]) * p["updates_per_fresh_case"], 375000
        )
        self.assertEqual(
            p["reuse_complete_selected_cases"], ["dense-stock", "dense-rms", "dense-ln"]
        )
        self.assertEqual(p["finish_parent_selected_evaluation_case"], "cst104-stock")
        self.assertEqual(p["parent_prefix_updates"], 72)
        self.assertEqual(p["new_prefix_updates"], 0)
        for key in (
            "automatic_retry_or_resume",
            "heldout_evaluated",
            "HPO",
            "additional_recipe_budget_or_selection_change",
        ):
            self.assertFalse(p[key])

    def test_missing_or_unpassed_parent_report_never_claims_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "not-ready.json"
            report.write_text(
                json.dumps({"status": "preparation-only-not-actual-audit"})
            )
            before = report.read_bytes()
            with self.assertRaisesRegex(ValueError, "report differs"):
                recovery.verify_parent(
                    root, report, hashlib.sha256(before).hexdigest(), root
                )
            with self.assertRaisesRegex(ValueError, "report differs"):
                recovery.verify_parent(root, report, "0" * 64, root)
            self.assertEqual(report.read_bytes(), before)
            self.assertEqual({p.name for p in root.iterdir()}, {"not-ready.json"})

    def test_parent_extra_file_and_alias_refused_without_mutating_owned_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "result").mkdir()
            original = root / "result/raw.pt"
            original.write_bytes(b"preserved")
            (root / "outer-receipt.json").write_text("unit-fixture")
            expected = {"result/raw.pt", "outer-receipt.json"}
            self.assertEqual(
                set(recovery.require_parent_inventory(root, expected)), expected
            )
            added = root / "unexpected-selected.npz"
            added.write_bytes(b"must reject")
            with self.assertRaisesRegex(ValueError, "unexpected failed-parent member"):
                recovery.require_parent_inventory(root, expected)
            self.assertEqual(original.read_bytes(), b"preserved")
            self.assertEqual(added.read_bytes(), b"must reject")
            alias = root / "alias"
            alias.symlink_to(original)
            with self.assertRaisesRegex(ValueError, "nonregular"):
                recovery.require_parent_inventory(
                    root, expected | {added.name, "alias"}
                )
            self.assertEqual(original.read_bytes(), b"preserved")

    def test_explicit_evaluator_seam_restored_on_success_and_exception(self):
        original = recovery.parent.evaluate_selected
        marker = object()
        calls = []

        def corrected(*args, **kwargs):
            calls.append((args, kwargs))
            return marker

        with (
            patch.object(recovery.correction, "evaluate_selected", new=corrected),
            patch.object(
                recovery.inspect, "getsource", return_value="unit-fixture-source"
            ),
        ):
            with recovery.selected_evaluator_amendment(
                {"new": "amendment"}, Path("/readonly/parent")
            ) as provenance:
                self.assertIsNot(recovery.parent.evaluate_selected, original)
                actual = recovery.parent.evaluate_selected(
                    "case", [], {}, Path("/new/case"), 1, 2, 3, 4, "cpu"
                )
                self.assertIs(actual, marker)
                self.assertTrue(provenance["restored_finally"])
            self.assertIs(recovery.parent.evaluate_selected, original)
            self.assertEqual(
                calls[0][1]["readonly_parent_root"], Path("/readonly/parent")
            )
            with (
                self.assertRaisesRegex(RuntimeError, "unit-stop"),
                recovery.selected_evaluator_amendment({}, Path("/readonly/parent")),
            ):
                raise RuntimeError("unit-stop")
            self.assertIs(recovery.parent.evaluate_selected, original)

    def test_extra_future_payload_refused_before_source_runtime_or_output(self):
        from cst_research.experiments import higgs_dst_confirmation_driver as common

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "research-inputs"
            inputs.mkdir()
            (inputs / "future-confirmation.npz").write_bytes(b"never opened")
            with (
                patch.object(Path, "cwd", return_value=root),
                patch.object(common, "verify_consumer") as verified,
            ):
                with self.assertRaisesRegex(ValueError, "exact eleven"):
                    driver.run(SimpleNamespace(output=root / "fresh"))
                verified.assert_not_called()
            self.assertFalse((root / "fresh").exists())

    def test_amendment_registry_does_not_relabel_failed_parent_or_change_recipe(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src/cst_research/studies/higgs-normalization-learning-recovery-20261106.json"
        )
        study = json.loads(path.read_text())
        self.assertEqual(study["status"], "COMPLETED")
        self.assertEqual(study["preparation_status"], "PREPARED")
        self.assertEqual(study["protocol"], recovery.protocol())
        self.assertEqual(study["parent_original_exit_code"], 1)
        self.assertFalse(study["parent_original_run_complete"])
