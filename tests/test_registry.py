"""Meaningful frozen-design, local-path and user-facing CLI checks; no GPU deps."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cst_research.cli import main
from cst_research.config import evidence_location
from cst_research.registry import STUDY_ID, load_studies, validate_study


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.study = next(study for study in load_studies() if study["id"] == STUDY_ID)

    def test_stopped_study_retains_exact_external_source(self):
        self.assertEqual(validate_study(self.study), [])
        self.assertEqual(self.study["status"], "STOPPED")
        self.assertEqual(
            self.study["source_commit"], "97580582ed28c4e494bf5c2656bbe94a85c544c3"
        )
        self.assertEqual(
            self.study["runner_entrypoint"],
            "benchmarks.training.higgs_confirmation_campaign",
        )

    def test_frozen_margins_splits_seeds_and_iut_cannot_silently_change(self):
        changes = [
            (("statistics", "accuracy_margin"), 0.02),
            (("statistics", "roc_auc_margin"), 0.02),
            (("data", "confirmation_validation_indices"), [0, 50000]),
            (("data", "official_test_evaluated"), True),
            (("training", "confirmation_seeds"), [17, 29, 43]),
            (("statistics", "primary", "configuration_count"), 16),
            (("training", "common_horizon_per_width", "extended"), 30),
        ]
        for keys, value in changes:
            with self.subTest(keys=keys):
                changed = copy.deepcopy(self.study)
                node = changed["protocol"]
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                self.assertTrue(validate_study(changed))

    def test_evidence_locator_rejects_private_absolute_and_escape_paths(self):
        for value in (
            "/private/data",
            "../escape",
            "study/../../escape",
            "C:/private",
            r"C:\private",
            ".",
        ):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.study)
                changed["evidence"]["relative_path"] = value
                self.assertTrue(validate_study(changed))

    def test_missing_or_malformed_protocol_produces_validation_errors(self):
        for value in ([], {}, {"schema_version": 1, "status": []}):
            self.assertTrue(validate_study(value))
        changed = copy.deepcopy(self.study)
        changed["protocol"]["training"] = None
        self.assertTrue(validate_study(changed))

    def test_draft_registry_rejects_filename_mismatch_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong-id.json"
            path.write_text(json.dumps(self.study))
            with self.assertRaisesRegex(ValueError, "filename"):
                load_studies(temporary)
            path.write_text("{")
            with self.assertRaisesRegex(ValueError, "cannot read"):
                load_studies(temporary)

    def test_validate_does_not_require_local_evidence(self):
        with patch.dict(os.environ, {}, clear=True):
            location = evidence_location(self.study)
        self.assertFalse(location["configured"])
        self.assertIsNone(location["path"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["validate", "--json"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["valid"])

    def test_local_toml_path_is_relative_to_config_and_env_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "research.local.toml"
            config.write_text('[evidence]\nroot = "raw-evidence"\n')
            with patch.dict(os.environ, {}, clear=True):
                location = evidence_location(self.study, config)
            self.assertEqual(
                Path(location["path"]),
                Path(temporary).resolve() / "raw-evidence" / STUDY_ID,
            )
            self.assertFalse(location["exists"])
            with patch.dict(
                os.environ, {"CST_RESEARCH_EVIDENCE_ROOT": temporary}, clear=True
            ):
                location = evidence_location(
                    self.study, Path(temporary) / "does-not-exist.toml"
                )
            self.assertEqual(
                Path(location["path"]), Path(temporary).resolve() / STUDY_ID
            )

    def test_show_has_no_resolved_private_path_unless_requested(self):
        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {"CST_RESEARCH_EVIDENCE_ROOT": "/private/example"},
                clear=True,
            ),
            contextlib.redirect_stdout(output),
        ):
            code = main(["show", STUDY_ID])
        self.assertEqual(code, 0)
        self.assertNotIn("/private/example", output.getvalue())
        self.assertNotIn("local_evidence", json.loads(output.getvalue()))

    def test_unknown_study_returns_failure_without_partial_output(self):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(["show", "missing"])
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("unknown study", error.getvalue())

    def test_bounded_completion_keeps_stopped_studies_and_measured_identity_separate(
        self,
    ):
        studies = {study["id"]: study for study in load_studies()}
        self.assertEqual(len(studies), 20)
        for identifier in (
            STUDY_ID,
            "higgs-dst-comparison-20261105",
            "higgs-linear-atom-scaling-20261106",
        ):
            self.assertEqual(studies[identifier]["status"], "STOPPED")
        for identifier in (
            "higgs-coordinate-lr-20261105",
            "higgs-depth-position-20261105",
            "higgs-depth-gain-20261106",
        ):
            study = studies[identifier]
            self.assertEqual(study["status"], "COMPLETED")
            execution = study["execution"]
            self.assertRegex(execution["measured_consumer_commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(
                execution["measured_consumer_manifest_sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertRegex(
                execution["passed_summary_audit_sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertFalse(execution["integration_is_rerun_or_revalidation"])
            self.assertFalse(execution["confirmation_or_test_evaluated"])
            self.assertEqual(
                execution["summary_path"], "docs/bounded-validation-results.md"
            )

    def test_completed_transfer_requires_actual_seal_and_once_assessment_provenance(
        self,
    ):
        study = next(
            s for s in load_studies() if s["id"] == "covertype-transfer-20261106"
        )
        self.assertEqual(study["status"], "COMPLETED")
        evidence = study["execution"]
        self.assertTrue(evidence["final_assessment_complete"])
        self.assertTrue(evidence["heldout_evaluated"])
        self.assertEqual(evidence["assessment_invocations"], 1)
        self.assertEqual(evidence["selected_models"], 9)
        self.assertEqual(evidence["training_seeds"], [29, 43, 71])
        for key in (
            "strict_training_audit_sha256",
            "heldout_freeze_sha256",
            "assessment_envelope_sha256",
            "assessment_archive_sha256",
            "independent_predictions_audit_sha256",
        ):
            self.assertRegex(evidence[key], r"^[0-9a-f]{64}$")
        self.assertFalse(evidence["formal_noninferiority"])
        self.assertFalse(evidence["official_test_evaluated"])
        self.assertFalse(evidence["integration_is_rerun_or_revalidation"])
        self.assertEqual(evidence["summary_path"], "docs/covertype-transfer-results.md")

    def test_list_reports_status_and_source_registry_is_unchanged(self):
        before = copy.deepcopy(self.study)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["list", "--json"]), 0)
        listed = {study["id"]: study for study in json.loads(output.getvalue())}
        self.assertEqual(listed[STUDY_ID]["status"], "STOPPED")
        self.assertEqual(
            next(study for study in load_studies() if study["id"] == STUDY_ID), before
        )


if __name__ == "__main__":
    unittest.main()
