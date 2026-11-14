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


class RadiusFractionContractTests(unittest.TestCase):
    def test_tau_and_fraction_are_part_of_exact_qualification_recipe(self):
        from dataclasses import replace

        from cst_research.experiments.ffn_contract import qualification_covers_model

        config = ModelConfig(
            width=512,
            heads=8,
            radius_mode="envelope",
            linear_backend="levels",
            atom_lr_multiplier=16.0,
        )
        recipe = config.recipe()
        self.assertEqual(recipe["tau"], 0.03)
        self.assertEqual(recipe["radius_fraction"], 0.0)
        receipt = {
            "task_model_configs": [__import__("dataclasses").asdict(config)],
            "task_recipes": [recipe],
            "cases": [
                {**recipe, "in": ni, "out": no, "atoms": 8192}
                for ni, no in [(512, 2048), (2048, 512)]
            ],
        }
        self.assertTrue(qualification_covers_model(receipt, config, ffn_kind="cst"))
        self.assertFalse(
            qualification_covers_model(
                receipt, replace(config, tau=0.1), ffn_kind="cst"
            )
        )
        self.assertFalse(
            qualification_covers_model(
                receipt,
                replace(config, radius_fraction_lr_multiplier=2.0),
                ffn_kind="cst",
            )
        )
        for kwargs in [
            {"tau": 0.0},
            {"tau": True},
            {"radius_fraction": 1.01},
            {"radius_fraction_lr_multiplier": 0.0},
            {"width": 256},
        ]:
            values = {
                "width": 512,
                "heads": 8,
                "radius_mode": "envelope",
                "linear_backend": "levels",
                **kwargs,
            }
            with self.assertRaises(ValueError):
                ModelConfig(**values)

    def test_wide_amplitude_requires_its_explicit_envelope_route(self):
        ModelConfig(
            width=512,
            heads=8,
            radius_mode="envelope",
            linear_backend="levels_wide_amplitude",
            amplitude_max=32.0,
            radius_fraction=1.0,
        )
        for values in [
            dict(radius_mode="polar", linear_backend="levels_wide_amplitude"),
            dict(radius_mode="envelope", linear_backend="levels", amplitude_max=32.0),
            dict(
                radius_mode="envelope",
                linear_backend="levels_wide_amplitude",
                amplitude_max=33.0,
            ),
        ]:
            with self.assertRaises(ValueError):
                ModelConfig(width=512, heads=8, **values)

    def test_probe_initialization_is_a_separate_qualification_recipe(self):
        from dataclasses import replace

        config = ModelConfig(
            width=512,
            heads=8,
            radius_mode="envelope",
            linear_backend="levels_wide_amplitude",
            amplitude_max=32.0,
        )
        calibrated = replace(config, initialization_mode="probe_rms")
        self.assertNotEqual(config.recipe(), calibrated.recipe())
        self.assertEqual(calibrated.recipe()["initialization_mode"], "probe_rms")
        with self.assertRaises(ValueError):
            ModelConfig(initialization_mode="probe_rms")
