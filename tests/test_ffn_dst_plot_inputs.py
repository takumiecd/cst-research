"""Reject mismatched tasks/sources and unselected DST curves before plotting."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ffn_lr_report_inputs",
    Path(__file__).resolve().parents[1] / "scripts/ffn_lr_report_inputs.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DSTPlotInputTests(unittest.TestCase):
    def fixture(self, base):
        roots = [base / "cst", base / "dst"]
        cfg = {
            "width": 512,
            "layers": 4,
            "heads": 8,
            "context": 128,
            "vocabulary": 50257,
            "expansion": 4,
            "attention_kind": "dense",
            "feature_order": "contiguous",
        }
        for index, r in enumerate(roots):
            d = r / "retrieved"
            (d / "qualification").mkdir(parents=True)
            (r / "retrieval-integrity.json").write_text(json.dumps({"state": "passed"}))
            (d / "trained-audit.json").write_text(json.dumps({"state": "passed"}))
            rows = [
                {
                    "label": "arm",
                    "kind": "cst" if index == 0 else "dst",
                    "seed": 17,
                    "updates": 976,
                    "development_ce": 3.4,
                    "sparse_peak_lr": 0.001,
                }
            ]
            receipt = {
                "status": "passed",
                "source_unchanged": True,
                "gpu": "same",
                "torch": "same",
                "cuda": "same",
                "protocol": {"data_manifest_sha256": "same"},
                "results": rows,
                "selection": {"label": "arm"},
            }
            (d / "receipt.json").write_text(json.dumps(receipt))
            (d / "qualification/qualification.json").write_text(
                json.dumps(
                    {
                        "source_sha256": {
                            n: "same"
                            for n in ("ffn_model.py", "ffn_contract.py", "ffn_run.py")
                        }
                    }
                )
            )
            (d / "arm").mkdir()
            (d / "arm/summary.json").write_text(
                json.dumps(
                    {
                        "config": cfg,
                        "actual_ffn_parameters": 272384,
                        "active_ffn_scalar_slots_including_bias": 272384,
                    }
                )
            )
        return roots

    def test_matching_reference_is_selected_and_keeps_original_origin(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = self.fixture(Path(t))
            rows, dirs = module.load_inputs(a, dst_run=b)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["origin_job"], "dst")
            self.assertTrue(rows[-1]["development_selected"])
            self.assertEqual(dirs[rows[-1]["label"]], b / "retrieved/arm")

    def test_data_source_model_budget_and_selection_mismatch_fail_closed(self):
        for change in ("data", "source", "model", "budget", "selection"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as t:
                a, b = self.fixture(Path(t))
                d = b / "retrieved"
                if change in ("data", "selection"):
                    p = d / "receipt.json"
                    x = json.loads(p.read_text())
                    if change == "data":
                        x["protocol"]["data_manifest_sha256"] = "other-task"
                    else:
                        x["selection"]["label"] = "not-selected"
                elif change == "source":
                    p = d / "qualification/qualification.json"
                    x = json.loads(p.read_text())
                    x["source_sha256"]["ffn_model.py"] = "different-model"
                else:
                    p = d / "arm/summary.json"
                    x = json.loads(p.read_text())
                    if change == "model":
                        x["config"]["context"] = 256
                    else:
                        x["active_ffn_scalar_slots_including_bias"] = 1
                p.write_text(json.dumps(x))
                with self.assertRaises(AssertionError):
                    module.load_inputs(a, dst_run=b)


class LongDSTPlotInputTests(unittest.TestCase):
    def fixture(self, root):
        r, _ = DSTPlotInputTests().fixture(root)
        d = r / "retrieved"
        old = json.loads((d / "arm/summary.json").read_text())
        rows = []
        for kind in ("dense", "cst", "dst"):
            (d / kind).mkdir()
            summary = {
                **old,
                "data_order_sha256": "same-order",
                "completed_tokens": 19999744,
                "actual_adam_updates": 19473,
                "topology_events": [{}] * 58,
            }
            (d / kind / "summary.json").write_text(json.dumps(summary))
            rows.append(
                {
                    "label": kind,
                    "kind": kind,
                    "seed": 17,
                    "updates": 19531,
                    "long_horizon_comparison": True,
                }
            )
        p = d / "receipt.json"
        receipt = json.loads(p.read_text())
        receipt["results"] = rows
        p.write_text(json.dumps(receipt))
        (d / "trained-audit.json").write_text(
            json.dumps({"state": "passed", "models": [{}] * 10})
        )
        (d / "dst-audit").mkdir()
        (d / "dst-audit/audit.json").write_text(
            json.dumps({"state": "passed", "models": [{}] * 5})
        )
        return r

    def test_long_fixed_three_arm_inputs_require_equal_order_budget_and_audits(self):
        for change in (None, "order", "tokens", "clock", "audit", "model", "budget"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as t:
                r = self.fixture(Path(t))
                p = r / "retrieved/dst/summary.json"
                x = json.loads(p.read_text())
                if change == "order":
                    x["data_order_sha256"] = "other"
                elif change == "tokens":
                    x["completed_tokens"] -= 1024
                elif change == "clock":
                    x["actual_adam_updates"] = 19531
                elif change == "model":
                    x["config"]["context"] = 256
                elif change == "budget":
                    x["active_ffn_scalar_slots_including_bias"] = 32768
                elif change == "audit":
                    p = r / "retrieved/dst-audit/audit.json"
                    x = {"state": "passed", "models": [{}] * 4}
                p.write_text(json.dumps(x))
                if change is None:
                    self.assertEqual(len(module.load_inputs(r)[0]), 3)
                else:
                    with self.assertRaises(AssertionError):
                        module.load_inputs(r)
