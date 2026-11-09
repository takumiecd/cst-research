"""Freeze long DST cadence/counters and keep observation points horizon-independent."""

import unittest

from cst_research.experiments.ffn_training_plan import TrainingPlan


class TrainingPlanTests(unittest.TestCase):
    def test_20m_events_boundaries_and_checkpoint_clocks(self):
        p = TrainingPlan(20000000, 20000000, 20000000)
        self.assertEqual(p.steps, 19531)
        self.assertEqual(p.milestones, (256, 976, 4882, 9765, 19531))
        self.assertEqual((len(p.events), p.events[0], p.events[-1]), (58, 250, 14500))
        self.assertEqual(
            [p.adam_updates(k) for k in p.milestones], [255, 973, 4863, 9726, 19473]
        )
        self.assertNotIn(14750, p.events)

    def test_prefix_preserves_the_planned_long_rewiring_law(self):
        p = TrainingPlan(20000000, 20000000, 20000000)
        prefix = TrainingPlan(5000000, 20000000, 20000000)
        self.assertEqual(prefix.events, tuple(k for k in p.events if k <= 4882))
        self.assertEqual(prefix.topology_steps, p.topology_steps)
        self.assertEqual(prefix.schedule_steps, p.schedule_steps)
        self.assertEqual(prefix.adam_updates(4882), p.adam_updates(4882))
        self.assertEqual(TrainingPlan().events, (250, 500))
        self.assertEqual(TrainingPlan().adam_updates(976), 974)

    def test_invalid_horizons_and_out_of_range_audit_fail(self):
        for values in (
            (0, 20000000, None),
            (True, 20000000, None),
            (20000001, 20000001, None),
            (20000000, 1000000, 20000000),
            (5000000, 20000000, 999424),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingPlan(*values)
        with self.assertRaises(ValueError):
            TrainingPlan().adam_updates(977)
