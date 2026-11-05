"""CPU model/ownership/split contracts; these do not constitute GPU admission."""

import copy
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

from cst_research.experiments.higgs_depth_data import data_spec, validate_spec
from cst_research.experiments.higgs_depth_study import configuration, protocol


class DepthProtocolTests(unittest.TestCase):
    def test_coarse_architecture_is_distinct_and_bounded(self):
        p = protocol()
        self.assertEqual(p["architecture"], [28, 1024, 1024, 1024, 1024, 2])
        self.assertEqual(
            (p["square_hidden_layers"], p["relu_layers"], p["pilot_configurations"]),
            (3, 4, 13),
        )
        self.assertEqual(
            (p["epochs"], p["seed"], p["lr"], p["position_multiplier"]),
            (3, 17, 0.0003, 1.0),
        )
        self.assertFalse(p["confirmation_evaluated"])
        self.assertFalse(p["official_test_evaluated"])
        self.assertEqual(configuration("all", 10000)["atoms_per_cst_layer"], 104)
        with self.assertRaises(ValueError):
            configuration("dense", 1000)

    def test_new_label_ranges_and_current_statistics_are_separate_from_past_exposure(
        self,
    ):
        audit = validate_spec(data_spec())
        self.assertTrue(
            audit["model_and_label_ranges_disjoint_from_documented_previous"]
        )
        self.assertTrue(all(audit["historical_feature_statistics_exposure"].values()))
        self.assertFalse(audit["formal_confirmation_ready"])
        self.assertEqual(data_spec()["normalization_fit"], [1000000, 1100000])
        bad = copy.deepcopy(data_spec())
        bad["ranges"]["development"] = [10400000, 10450000]
        with self.assertRaises(ValueError):
            validate_spec(bad)
        bad = copy.deepcopy(data_spec())
        bad["normalization_fit"] = [0, 10400000]
        with self.assertRaises(ValueError):
            validate_spec(bad)


SOURCE = os.environ.get("CST_DEPTH_SOURCE")


@unittest.skipUnless(
    SOURCE, "explicit pinned external975 source required for CPU module smoke"
)
class DepthModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(SOURCE).resolve()
        manifest = json.loads(Path(os.environ["CST_DEPTH_MANIFEST"]).read_text())
        if manifest["source_commit"] != "97580582ed28c4e494bf5c2656bbe94a85c544c3":
            raise ValueError("wrong external source")
        for name, digest in manifest["files"].items():
            path = source / name
            if (
                path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("external source changed")
        sys.path[:0] = [str(source / "src"), str(source)]
        import torch
        import torchcst

        if not Path(torchcst.__file__).resolve().is_relative_to(source):
            raise ValueError("fresh interpreter with pinned source required")
        from cst_research.experiments.higgs_depth_study import (
            make_model,
            make_optimizer,
            reference_forward,
        )

        cls.torch = torch
        cls.make_model = staticmethod(make_model)
        cls.make_optimizer = staticmethod(make_optimizer)
        cls.reference_forward = staticmethod(reference_forward)
        torch.set_num_threads(1)

    def test_five_linear_four_relu_and_shared_unreplaced_initialization(self):
        torch = self.torch
        dense = self.make_model("dense", None, width=128)
        all_cst = self.make_model("all", 1000, width=128)
        self.assertEqual(len(dense), 9)
        self.assertEqual(sum(isinstance(m, torch.nn.ReLU) for m in dense), 4)
        self.assertEqual(tuple(dense(torch.randn(4, 28)).shape), (4, 2))
        for i in (2, 4, 6):
            self.assertIsNone(dense[i].bias)
        for kind, replaced in [("early", 0), ("middle", 1), ("late", 2)]:
            model = self.make_model(kind, 1000, width=128)
            for i in (0, 8):
                self.assertTrue(torch.equal(model[i].weight, dense[i].weight))
                self.assertTrue(torch.equal(model[i].bias, dense[i].bias))
            for j, i in enumerate((2, 4, 6)):
                if j == replaced:
                    self.assertTrue(torch.equal(model[i].atoms.p, all_cst[i].atoms.p))
                else:
                    self.assertTrue(torch.equal(model[i].weight, dense[i].weight))
        self.assertTrue(all(p.requires_grad for p in all_cst.parameters()))
        self.assertEqual(all_cst._cst_research_cohort, "higgs-depth-position-v1")

    def test_three_cst_sites_have_independent_complete_optimizer_ownership(self):
        torch = self.torch
        model = self.make_model("all", 1000, width=128)
        optimizer, _ = self.make_optimizer(model, cpu_reference=True)
        before = [model[i].atoms.p.detach().clone() for i in (2, 4, 6)]
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optimizer.step()
        self.assertEqual(len(list(model.parameters())), 7)
        self.assertEqual(len(optimizer.state), 7)
        states = [optimizer.state[model[i].atoms.p] for i in (2, 4, 6)]
        self.assertEqual(len({s["exp_avg"].data_ptr() for s in states}), 3)
        self.assertTrue(all(float(s["step"]) == 1 for s in optimizer.state.values()))
        self.assertTrue(
            all(
                not torch.equal(old, model[i].atoms.p)
                for old, i in zip(before, (2, 4, 6))
            )
        )

    def test_public_fp64_reference_composes_all_three_cst_layers(self):
        torch = self.torch
        model = self.make_model("all", 10000, width=128).double()
        x = torch.randn(4, 28, dtype=torch.float64, requires_grad=True)
        y = self.reference_forward(model, x)
        self.assertEqual(tuple(y.shape), (4, 2))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
