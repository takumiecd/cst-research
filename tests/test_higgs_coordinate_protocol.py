"""Pure-CPU scientific gates and restored benchmark-injection boundaries."""

import math
import unittest
from types import SimpleNamespace

from cst_research.experiments.higgs_coordinate_lr import (
    Observer,
    decision,
    inject,
    protocol,
)


class CoordinateProtocolTests(unittest.TestCase):
    def test_budget_and_conditional_common_horizon_are_fixed(self):
        p = protocol()
        self.assertEqual(
            (p["width"], p["divisor"], p["epochs"], p["development_seed"]),
            (1024, 1000, 3, 17),
        )
        self.assertEqual(p["advance"]["followup_epochs"], 10)
        self.assertEqual(p["advance"]["extra_paired_seeds"], [29, 43])
        self.assertFalse(p["confirmation_evaluated"])
        self.assertFalse(p["test_evaluated"])
        self.assertEqual(
            (p["base_lr"], p["ordinary_lr"], p["amplitude_weight_decay"]),
            (0.0003, 0.0003, 0.01),
        )

    def test_advance_requires_material_ce_gain_auc_and_qualification(self):
        arms = {
            0.1: {"loss": 0.538, "roc_auc": 0.801},
            1.0: {"loss": 0.540, "roc_auc": 0.800},
            10.0: {"loss": 0.535, "roc_auc": 0.798},
        }
        self.assertEqual(decision(arms, True)["selected_position_multiplier"], 0.1)
        self.assertEqual(
            decision(arms, False)["status"], "stop-no-short-pilot-evidence"
        )
        arms[0.1]["loss"] = 0.5395
        self.assertEqual(decision(arms, True)["status"], "stop-no-short-pilot-evidence")
        arms[10.0]["roc_auc"] = 0.801
        self.assertEqual(decision(arms, True)["selected_position_multiplier"], 10.0)
        arms[0.1]["loss"] = math.nan
        with self.assertRaises(ValueError):
            decision(arms, True)

    def test_exact_control_tie_never_advances_and_candidate_tie_is_declared(self):
        arms = {
            0.1: {"loss": 0.54, "roc_auc": 0.8},
            1.0: {"loss": 0.54, "roc_auc": 0.8},
            10.0: {"loss": 0.54, "roc_auc": 0.8},
        }
        self.assertIsNone(decision(arms, True)["selected_position_multiplier"])
        arms[0.1]["loss"] = arms[10.0]["loss"] = 0.538
        self.assertEqual(decision(arms, True)["selected_position_multiplier"], 0.1)

    def test_benchmark_globals_restore_even_after_training_failure(self):
        factory = lambda *args: None
        step = lambda *args: None
        checkpoint = lambda *args: None
        runner = SimpleNamespace(
            make_optimizer=factory, ordinary_step=step, _checkpoint=checkpoint
        )
        with (
            self.assertRaisesRegex(RuntimeError, "training failed"),
            inject(runner, 0.1, Observer()),
        ):
            self.assertIsNot(runner.make_optimizer, factory)
            self.assertIsNot(runner.ordinary_step, step)
            raise RuntimeError("training failed")
        self.assertIs(runner.make_optimizer, factory)
        self.assertIs(runner.ordinary_step, step)
        self.assertIs(runner._checkpoint, checkpoint)
