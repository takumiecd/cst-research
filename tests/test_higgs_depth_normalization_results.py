"""Integrity/interpretation of the published aggregate, without raw/GPU access."""

import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NormalizationResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(
            (
                ROOT / "docs/higgs-depth-normalization-diagnostic-summary.json"
            ).read_text()
        )
        cls.study = json.loads(
            (
                ROOT
                / "src/cst_research/studies/higgs-depth-normalization-diagnostic-20261106.json"
            ).read_text()
        )

    def test_source_bound_completion_is_nine_by_eight_not_quality_selection(self):
        s = self.summary
        self.assertEqual(self.study["status"], "COMPLETED")
        self.assertEqual(
            s["identity"]["consumer_commit"], "4f777a5f1ef38cca7031cf9c637d0fa6fc0d5e7b"
        )
        self.assertEqual(
            self.study["execution"]["measured_consumer_commit"],
            s["identity"]["consumer_commit"],
        )
        self.assertEqual(
            self.study["execution"]["passed_full_CPU_admission_sha256"],
            s["evidence_sha256"]["full"],
        )
        self.assertEqual(
            (s["scope"]["cases"], s["scope"]["true_task_updates_per_case"]), (9, 8)
        )
        self.assertFalse(s["scope"]["quality_or_final_accuracy_evaluation"])
        self.assertFalse(s["scope"]["heldout_evaluated"])
        self.assertFalse(
            s["scope"][
                "minimum_alpha_or_scaling_or_throughput_or_default_adoption_claim"
            ]
        )
        self.assertEqual(
            {r["case"] for r in s["activations"]},
            set(self.study["protocol"]["cases"]),
        )

    def test_energy_ratios_recompute_from_the_displayed_task_batch(self):
        for r in self.summary["activations"]:
            with self.subTest(case=r["case"]):
                self.assertEqual(r["step"], 1)
                rho = (
                    r["last_hidden_post_ReLU_RMS"] / r["last_hidden_pre_ReLU_RMS"]
                ) ** 2
                self.assertTrue(
                    math.isclose(
                        rho,
                        r["last_hidden_ReLU_energy_retention_fraction"],
                        rel_tol=1e-12,
                    )
                )
                self.assertGreaterEqual(rho, 0)
                self.assertLessEqual(rho, 1)

    def test_nonzero_gradient_scope_and_stable_torus_erratum_are_preserved(self):
        rows = {r["case"]: r for r in self.summary["coordinate_field_ranges"]}
        self.assertEqual(len(rows), 6)
        for row in rows.values():
            self.assertEqual(row["field_blocks"], 6)
            nonzero = row["eps_dominant_fraction_nonzero_grad"]
            self.assertTrue(0 <= nonzero["min"] <= nonzero["max"] <= 1)
        self.assertEqual(
            rows["cst104-stock"]["accepted_displacement_RMS"], {"min": 0.0, "max": 0.0}
        )
        self.assertGreater(
            rows["cst104-stock"]["accepted_encoded_changed_fraction"]["max"], 0
        )
        # Same initial epsilon classification cannot by itself establish a stall.
        self.assertEqual(
            rows["cst1048-stock"]["eps_dominant_fraction_nonzero_grad"],
            {"min": 1.0, "max": 1.0},
        )
        erratum = self.summary["reporting_erratum"]
        self.assertGreater(erratum["stock_A104_step8_nonzero_coordinate_delta"], 0)
        self.assertEqual(erratum["coordinate_deltas_libm_checked"], 37632)
        self.assertEqual(erratum["legacy_rounding_differences"], 7)
        self.assertFalse(erratum["legacy_scientific_source_changed"])
        self.assertFalse(erratum["GPU_rerun"])
