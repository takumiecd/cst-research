"""Width admission, fair slot budgets, fixed clocks and complete LR selection."""

import unittest
from dataclasses import asdict

from cst_research.experiments.ffn_contract import ModelConfig
from cst_research.experiments.ffn_training_plan import TrainingPlan
from cst_research.experiments.lm_width_contract import (
    ARMS,
    CHECKPOINTS,
    LONG_STEPS,
    SHORT_STEPS,
    WidthConfig,
    choose,
    slots_per_map,
)


class WidthContractTests(unittest.TestCase):
    def test_private_shapes_and_linear_budget(self):
        for width, count in ((512, 65536), (1024, 131072), (2048, 262144)):
            cfg = WidthConfig(width=width)
            self.assertEqual(slots_per_map(width), count)
            with self.assertRaises(ValueError):
                ModelConfig(**asdict(cfg))
        for kw in ({"width": 4096}, {"coefficient": 128}, {"layers": 8}, {"heads": 16}):
            with self.assertRaises(ValueError):
                WidthConfig(**kw)

    def test_short_and_long_share_schedule_and_topology(self):
        short = TrainingPlan(SHORT_STEPS * 1024, 20000000, 20000000)
        long = TrainingPlan(LONG_STEPS * 1024, 20000000, 20000000)
        self.assertEqual(short.schedule_steps, long.schedule_steps)
        self.assertEqual(short.topology_steps, long.topology_steps)
        self.assertEqual(
            short.adam_updates(SHORT_STEPS), long.adam_updates(SHORT_STEPS)
        )
        self.assertEqual(
            short.events, tuple(s for s in long.events if s <= SHORT_STEPS)
        )
        self.assertTrue({SHORT_STEPS, LONG_STEPS}.issubset(CHECKPOINTS))

    def test_lr_selection_requires_complete_candidates_and_ties_low(self):
        candidates = {name: {"ce": 3.0} for name in ARMS}
        self.assertEqual(choose(candidates, "dense"), "dense-lr0.0001")
        self.assertEqual(choose(candidates, "dst"), "dst-lr0.0003")
        del candidates["dst-lr0.01"]
        with self.assertRaises(ValueError):
            choose(candidates, "dst")
