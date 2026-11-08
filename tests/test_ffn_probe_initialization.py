"""Measured output RMS, geometry preservation and failed-search rollback."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torchcst"), "optional torchcst dependency"
)
class ProbeInitializationTests(unittest.TestCase):
    def test_matches_actual_output_and_preserves_fraction_centers_and_rng(self):
        import torch
        from torch import nn

        from cst_research.experiments.ffn_probe_initialization import (
            match_projection_rms,
            output_rms,
        )
        from cst_research.experiments.ffn_radius_fraction import make_layer
        from cst_research.experiments.ffn_radius_fraction_qualification import (
            config_for,
        )

        for fraction in (0.0, 0.5, 1.0):

            class Projection(nn.Module):
                def __init__(self, fraction=fraction):
                    super().__init__()
                    self.map = make_layer(
                        128,
                        128,
                        config=config_for(
                            1e-8,
                            radius_fraction=fraction,
                            amplitude_max=32.0,
                            linear_backend="levels_wide_amplitude",
                        ),
                        std=0.001,
                        seed=17,
                    )
                    self.bias = nn.Parameter(torch.zeros(128))

                def forward(self, x):
                    return self.map(x) + self.bias

            p = Projection()
            x = torch.randn(3, 128, generator=torch.Generator().manual_seed(123))
            before = p.map.atoms.p.detach().clone()
            rng = torch.get_rng_state().clone()
            with torch.no_grad():
                target = output_rms(p(x)) * 0.4
            result = match_projection_rms(p, x, target, amplitude_max=32.0)
            with torch.no_grad():
                self.assertLessEqual(abs(output_rms(p(x)) / target - 1), 0.001)
            self.assertLessEqual(result["relative_error"], 0.001)
            self.assertTrue(torch.equal(p.map.atoms.p[:, 1:], before[:, 1:]))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            previous = p.map.atoms.p.detach().clone()
            with self.assertRaises(ValueError):
                match_projection_rms(p, x, 1e20, amplitude_max=32.0)
            self.assertTrue(torch.equal(p.map.atoms.p, previous))

    def test_unreachable_uniform_scale_target_rolls_back_without_relaxing_tolerance(
        self,
    ):
        import torch
        from torch import nn

        from cst_research.experiments.ffn_probe_initialization import (
            match_projection_rms,
            output_rms,
        )
        from cst_research.experiments.ffn_radius_fraction import make_layer
        from cst_research.experiments.ffn_radius_fraction_qualification import (
            config_for,
        )

        class Projection(nn.Module):
            def __init__(self):
                super().__init__()
                self.map = make_layer(
                    128,
                    128,
                    config=config_for(
                        0.03, amplitude_max=32.0, linear_backend="levels_wide_amplitude"
                    ),
                    std=0.001,
                    seed=17,
                )
                self.bias = nn.Parameter(torch.zeros(128))

            def forward(self, x):
                return self.map(x) + self.bias

        p = Projection()
        x = torch.randn(3, 128, generator=torch.Generator().manual_seed(123))
        before = p.map.atoms.p.detach().clone()
        with torch.no_grad():
            target = 0.4 * output_rms(p(x))
        # The encoded hierarchy has a finite radius-level jump near scale.780792.
        # A strict scalar calibration rejects that bracket; it must not weaken
        # the gate or leave half-calibrated Parameters behind.
        with self.assertRaisesRegex(ValueError, "did not reach target"):
            match_projection_rms(p, x, target, amplitude_max=32.0)
        self.assertTrue(torch.equal(before, p.map.atoms.p))
