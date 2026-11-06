"""CPU serialization contract checks, not GPU recovery or new training evidence."""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cst_research.experiments import higgs_normalization_learning_restore as correction

SOURCE = os.environ.get("CST_DEPTH_SOURCE")


class ReadonlyOutputPolicy(unittest.TestCase):
    def test_parent_output_is_refused_before_any_source_prediction_read_or_write(self):
        # Optional numeric dependencies are needed only to enter the evaluation function.
        try:
            import numpy  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("optional numeric dependencies required")
        if not SOURCE:
            self.skipTest("explicit public975 source required")
        source = Path(SOURCE).resolve()
        sys.path[:0] = [str(source / "src"), str(source)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            case = root / "cases/cst104-stock"
            case.mkdir(parents=True)
            raw = case / "immutable-marker"
            raw.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "immutable failed-parent"):
                correction.evaluate_selected(
                    "cst104-stock",
                    [1.0] * 3,
                    {},
                    case,
                    case / "selected",
                    None,
                    None,
                    None,
                    None,
                    "cpu",
                    {},
                    readonly_parent_root=root,
                )
            self.assertEqual(raw.read_bytes(), b"keep")
            self.assertFalse((case / "selected").exists())


@unittest.skipUnless(SOURCE, "explicit unchanged public975 source required")
class NativeLoadContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(SOURCE).resolve()
        manifest = json.loads(Path(os.environ["CST_DEPTH_MANIFEST"]).read_text())
        if manifest["source_commit"] != correction.parent.stock.SOURCE:
            raise ValueError("wrong public975 source")
        for name, value in manifest["files"].items():
            if hashlib.sha256((cls.source / name).read_bytes()).hexdigest() != value:
                raise ValueError("public source changed")
        sys.path[:0] = [str(cls.source / "src"), str(cls.source)]
        import torch

        torch.set_num_threads(1)
        cls.t = torch

    def cpu_fresh(self, case, gains, device):
        model = correction.parent.model_contract.make_model(case, gains=gains).to(
            device
        )
        optimizer, contract = correction.parent.stock.make_optimizer(
            model, cpu_reference=True
        )
        return model, optimizer, contract

    def test_exact_three_generation_transitions_all_other_fields_and_postload_baseline(
        self,
    ):
        t = self.t
        model, optimizer, _ = self.cpu_fresh("cst104-stock", [1.0] * 3, "cpu")
        # Synthetic supplied gradients create genuine public Adam moments without a forward/GPU.
        for parameter in model.parameters():
            parameter.grad = t.ones_like(parameter) * 0.01
        optimizer.step()
        saved = correction.parent.stock.checkpoint(model, optimizer, 1, 1, {}, None)
        before = copy.deepcopy(saved)
        with patch.object(correction.parent, "fresh", side_effect=self.cpu_fresh):
            restored, opt, baseline, report = correction.restore_selected(
                "cst104-stock", [1.0] * 3, saved, "cpu"
            )
        self.assertEqual(len(report["owner_transitions"]), 3)
        self.assertTrue(
            all(
                r["before_load"] == 0
                and r["serialized"] == 0
                and r["expected_after_native_load"] == 1
                for r in report["owner_transitions"]
            )
        )
        correction.exact.assert_equal(saved, before, "raw-was-not-written")
        with self.assertRaisesRegex(ValueError, "layout_version"):
            correction.exact.assert_equal(baseline, correction.parent.core_state(saved))
        correction.assert_after_evaluation(restored, opt, saved, baseline)
        wrong = copy.deepcopy(baseline)
        wrong["model"]["2.atom_state._extra_state"]["layout_version"] += 1
        with (
            patch.object(correction.parent.stock, "checkpoint", return_value=wrong),
            self.assertRaisesRegex(ValueError, "layout_version"),
        ):
            correction.assert_after_evaluation(restored, opt, saved, baseline)
        # A wrong moment must still fail the same exact comparator.
        wrong = copy.deepcopy(baseline)
        next(iter(wrong["optimizer"]["state"].values()))["exp_avg"].reshape(-1)[0] += 1
        with self.assertRaises(ValueError):
            correction.exact.assert_equal(wrong, baseline)

    def test_saved_higher_generation_max_rule_and_native_schema_rejection(self):
        model, optimizer, _ = self.cpu_fresh("cst104-stock", [1.0] * 3, "cpu")
        saved = correction.parent.stock.checkpoint(model, optimizer, 0, 0, {}, None)
        before = correction.owner_generations(model)
        synthetic = copy.deepcopy(saved)
        for key in before:
            synthetic["model"][key]["layout_version"] = 7
        expected, changes = correction.expected_native_loaded_core(synthetic, before)
        self.assertTrue(all(r["expected_after_native_load"] == 7 for r in changes))
        for key in before:
            self.assertEqual(expected["model"][key]["layout_version"], 7)
        for bad in (-1, True, "0"):
            wrong = copy.deepcopy(saved)
            wrong["model"]["2.atom_state._extra_state"]["layout_version"] = bad
            with self.assertRaises(ValueError):
                correction.expected_native_loaded_core(wrong, before)
        with self.assertRaisesRegex(ValueError, "owner inventory"):
            correction.expected_native_loaded_core(
                saved, {"foreign.atom_state._extra_state": 0}
            )

    def test_dense_has_no_generation_exception(self):
        model, optimizer, _ = self.cpu_fresh("dense-stock", [1.0] * 3, "cpu")
        saved = correction.parent.stock.checkpoint(model, optimizer, 0, 0, {}, None)
        with patch.object(correction.parent, "fresh", side_effect=self.cpu_fresh):
            _, _, baseline, report = correction.restore_selected(
                "dense-stock", [1.0] * 3, saved, "cpu"
            )
        self.assertEqual(report["owner_transitions"], [])
        correction.exact.assert_equal(baseline, correction.parent.core_state(saved))

    @unittest.skipUnless(
        os.environ.get("CST_NORMALIZATION_RECOVERY_PARENT_CHECKPOINT"),
        "explicit SHA-verified real failed-parent selected checkpoint required",
    )
    def test_actual_failed_parent_epoch2_load_all_other_fields_exact(self):
        path = Path(os.environ["CST_NORMALIZATION_RECOVERY_PARENT_CHECKPOINT"])
        self.assertEqual(
            correction.file_sha(path),
            "75780b8066c0543ed699f56a5bb814b9e53b3c64eb13538c5fb2f6d5356aa676",
        )
        saved = self.t.load(path, map_location="cpu", weights_only=True)
        with patch.object(correction.parent, "fresh", side_effect=self.cpu_fresh):
            model, optimizer, baseline, report = correction.restore_selected(
                "cst104-stock", saved["configuration"]["gains"], saved, "cpu"
            )
        self.assertEqual(saved["epoch"], 2)
        self.assertEqual(saved["updates"], 50000)
        self.assertEqual(len(report["owner_transitions"]), 3)
        correction.assert_after_evaluation(model, optimizer, saved, baseline)
