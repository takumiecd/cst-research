"""Selective replacement must preserve every complementary weight and RNG."""

import importlib
import unittest


class SelectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            importlib.import_module("torch")
            importlib.import_module("torchcst")
        except ImportError as error:
            raise unittest.SkipTest(str(error))

    def config(self):
        from dataclasses import replace

        from cst_research.experiments.ffn_dst_run import config

        return replace(config(), vocabulary=31, context=4)

    def test_all_sites_reproduce_existing_full_models_and_preserve_global_rng(self):
        import torch

        from cst_research.experiments.ffn_dst import make_model as old_dst
        from cst_research.experiments.ffn_model import LanguageModel
        from cst_research.experiments.ffn_radius_fraction_qualification import (
            assert_nested,
        )
        from cst_research.experiments.ffn_selective import SITE_SETS, make_model

        cfg = self.config()
        for kind in ("cst", "dst"):
            before = torch.get_rng_state().clone()
            m = make_model(cfg, kind, SITE_SETS["all"])
            assert_nested(torch.get_rng_state(), before)
            old = LanguageModel(cfg, "cst") if kind == "cst" else old_dst(cfg)
            assert_nested(m.state_dict(), old.state_dict())

    def test_singletons_preserve_dense_complements_and_match_active_dst_budget(self):
        import torch
        from torchcst import CSTLinear

        from cst_research.experiments.ffn_model import LanguageModel
        from cst_research.experiments.ffn_selective import (
            SITE_SETS,
            active_ffn_scalars,
            make_model,
        )

        cfg = self.config()
        dense = LanguageModel(cfg, "dense")
        ordinary = dict(dense.named_parameters())
        for key in SITE_SETS.keys() - {"all"}:
            sites = SITE_SETS[key]
            m = make_model(cfg, "cst", sites)
            d = make_model(cfg, "dst", sites)
            self.assertEqual(
                len([x for x in m.modules() if isinstance(x, CSTLinear)]), len(sites)
            )
            self.assertEqual(active_ffn_scalars(m), active_ffn_scalars(d))
            selected = {
                f"blocks.{s.split('.')[0]}.ffn.{s.split('.')[1]}.map.weight"
                for s in sites
            }
            for n, p in m.named_parameters():
                if n in ordinary:
                    torch.testing.assert_close(p, ordinary[n], atol=0, rtol=0)
            for n, p in d.named_parameters():
                if n not in selected:
                    torch.testing.assert_close(p, ordinary[n], atol=0, rtol=0)

    def test_invalid_or_duplicate_sites_are_rejected(self):
        from cst_research.experiments.ffn_selective import validate_sites

        for kind, sites in [
            ("dense", ("1.up",)),
            ("cst", ()),
            ("cst", ("4.up",)),
            ("dst", ("1.up", "1.up")),
            ("x", ()),
        ]:
            with self.subTest(kind=kind, sites=sites), self.assertRaises(ValueError):
                validate_sites(self.config(), kind, sites)

    def test_readonly_probe_and_fixed_action_delta_do_not_change_weights_or_rng(self):
        import torch

        from cst_research.experiments.ffn_selective import (
            after_update,
            before_update,
            make_model,
            mechanism_probe,
        )

        m = make_model(self.config(), "dense", ())
        before = {n: p.clone() for n, p in m.named_parameters()}
        rng = torch.get_rng_state().clone()
        action = before_update(m)
        out = after_update(m, action)
        self.assertTrue(all(v["action_delta_rms"] == 0 for v in out.values()))
        x = torch.tensor([[1, 2, 3, 4]])
        probe = mechanism_probe(m, x, x.roll(-1, -1))
        self.assertEqual(len(probe["layers"]), 12)
        self.assertTrue(
            all(v["output_gradient_rms"] > 0 for v in probe["layers"].values())
        )
        for n, p in m.named_parameters():
            torch.testing.assert_close(p, before[n], atol=0, rtol=0)
            self.assertIsNone(p.grad)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        with torch.no_grad():
            m.blocks[1].ffn.up.bias.add_(0.01)
        out = after_update(m, action)
        self.assertAlmostEqual(out["1.up"]["action_delta_rms"], 0.01, places=6)
        self.assertEqual(out["0.up"]["action_delta_rms"], 0)
