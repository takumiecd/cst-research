"""Geometric Atom budgets, rectangular symmetry and honest scope accounting."""

import unittest

from cst_research.experiments.ffn_contract import ModelConfig, atom_count, learning_rate


class FFNContractTests(unittest.TestCase):
    def test_early_checkpoint_retains_the_long_learning_rate_horizon(self):
        short = 1000000 // 1024
        long = 20000000 // 1024
        self.assertAlmostEqual(learning_rate(short, short), 0.00003)
        self.assertGreater(learning_rate(short, long), 0.00029)
        self.assertAlmostEqual(learning_rate(long, long), 0.00003)
        self.assertAlmostEqual(learning_rate(200, long), 0.0003)
        with self.assertRaises(ValueError):
            learning_rate(long + 1, long)

    def test_exact_rounding_symmetry_and_large_integer_arithmetic(self):
        self.assertEqual(atom_count(3, 7), 37)
        self.assertEqual(atom_count(7, 3), 37)
        n = 2**55 + 3
        self.assertEqual(atom_count(n, n), 8 * n)
        for bad in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                atom_count(bad, 16)

    def test_fixed_coefficient_preserves_linear_atom_and_ffn_parameter_budgets(self):
        previous = None
        for width in (128, 256, 512):
            config = ModelConfig(width=width)
            budget = config.ffn_parameter_budget()
            self.assertEqual(config.atoms_per_projection, 16 * width)
            if previous:
                self.assertEqual(
                    budget["cst_parameters"], previous["cst_parameters"] * 2
                )
                self.assertGreater(
                    budget["dense_parameters"], previous["dense_parameters"] * 3
                )
            previous = budget

    def test_geometry_and_invalid_model_dimensions(self):
        self.assertEqual(atom_count(128, 512), atom_count(256, 256))
        for kwargs in ({"width": 129}, {"context": 0}, {"q": 1.1}):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)
