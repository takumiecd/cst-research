"""CPU preparation checks; no new native/oracle or epoch-quality claim."""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

from cst_research.experiments import higgs_depth_normalization_learning as study
from cst_research.experiments import higgs_depth_normalization_learning_driver as driver
from cst_research.experiments import (
    higgs_depth_normalization_learning_evidence as proof,
)


class EvidenceTests(unittest.TestCase):
    def test_inventory_streams_exact_bytes_and_permits_only_declared_receipt_relocation(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "result/raw.pt"
            original.parent.mkdir()
            original.write_bytes(b"original retained bytes")
            relocated = root / "receipt-copy"
            relocated.write_bytes(b"receipt")
            entries = {
                "result/raw.pt": {
                    "bytes": original.stat().st_size,
                    "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                },
                "outer-receipt.json": {
                    "bytes": 7,
                    "sha256": hashlib.sha256(b"receipt").hexdigest(),
                },
            }
            self.assertEqual(
                proof.verify_file_inventory(
                    root, entries, {"outer-receipt.json": relocated}
                ),
                entries,
            )
            before = original.read_bytes()
            with self.assertRaisesRegex(ValueError, "undeclared"):
                proof.verify_file_inventory(root, entries, {"result/raw.pt": relocated})
            original.write_bytes(before[:-1] + b"x")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                proof.verify_file_inventory(
                    root, entries, {"outer-receipt.json": relocated}
                )
            self.assertEqual(relocated.read_bytes(), b"receipt")

    def test_unsafe_or_symlink_prior_members_rejected_without_deleting_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "owned-by-other"
            target.write_bytes(b"keep")
            (root / "alias").symlink_to(target)
            for name in (
                "../owned-by-other",
                "/absolute",
                "result/../x",
                "result//x",
                "alias",
                "result\\x",
            ):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    proof.verify_file_inventory(
                        root,
                        {
                            name: {
                                "bytes": 4,
                                "sha256": hashlib.sha256(b"keep").hexdigest(),
                            }
                        },
                    )
            self.assertEqual(target.read_bytes(), b"keep")

    @unittest.skipUnless(
        torch is not None, "optional Torch required for tensor-byte gate"
    )
    def test_recursive_metadata_and_tensor_bits_are_both_required(self):
        state = {
            "model": {
                "p": torch.tensor([0.0, -0.0]),
                "_extra_state": {"plan": "fixed", "shape": (2, 4)},
            },
            "optimizer": {0: {"step": torch.tensor(8.0)}},
        }
        proof.assert_equal(state, copy.deepcopy(state))
        mutations = []
        value = copy.deepcopy(state)
        value["model"]["_extra_state"]["plan"] = "different"
        mutations.append(value)
        value = copy.deepcopy(state)
        value["model"]["p"][1] = 0.0
        mutations.append(value)
        value = copy.deepcopy(state)
        value["optimizer"][0]["step"] = torch.tensor(8.0, dtype=torch.float64)
        mutations.append(value)
        value = copy.deepcopy(state)
        value["optimizer"] = {False: value["optimizer"][0]}
        mutations.append(value)
        for value in mutations:
            with self.assertRaises(ValueError):
                proof.assert_equal(state, value)
        with self.assertRaises(ValueError):
            proof.assert_equal(float("nan"), float("nan"))
        with self.assertRaises(ValueError):
            proof.assert_equal(0.0, -0.0)
        self.assertNotEqual(proof.fingerprint(state), proof.fingerprint(mutations[0]))


class LearningPolicyTests(unittest.TestCase):
    def test_fixed_budget_inherited_identity_and_fresh_learning_scope(self):
        p = study.protocol()
        self.assertEqual(p["cases"], list(study.model_contract.CASES))
        self.assertEqual(
            p["main_updates_all9"],
            len(p["cases"]) * p["epochs"] * p["training_rows"] // p["batch_size"],
        )
        self.assertEqual(p["main_updates_all9"], 675000)
        self.assertEqual(p["admission_prefix_updates_all9"], 72)
        self.assertEqual(p["epoch_order_seeds"], [117, 118, 119])
        self.assertEqual(p["model_contract_source"], proof.PRIOR_COMMIT)
        self.assertNotEqual(study.COHORT, study.model_contract.COHORT)
        for field in (
            "HPO",
            "heldout_evaluated",
            "official_test_evaluated",
            "formal_NI",
            "automatic_followup",
        ):
            self.assertFalse(p[field])

    def test_only_three_complete_finite_epochs_select_earliest_minimum(self):
        rows = [
            {"epoch": i, "development": {"loss": loss}}
            for i, loss in enumerate((0.6, 0.5, 0.5), 1)
        ]
        self.assertEqual(study.selected_epoch(rows)["epoch"], 2)
        for invalid in (
            rows[:2],
            rows[::-1],
            [{**rows[0], "epoch": 0}, *rows[1:]],
            [rows[0], {"epoch": 2, "development": {"loss": float("nan")}}, rows[2]],
        ):
            with self.assertRaises(ValueError):
                study.selected_epoch(invalid)

    @unittest.skipUnless(
        np is not None, "optional NumPy required for prediction-array gate"
    )
    def test_prediction_reloads_require_array_bytes_labels_rows_and_dtype(self):
        expected = {
            "logits": np.array([[1.0, -1.0]], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
            "row_ids": np.array([1100000], dtype=np.int64),
        }
        study.assert_prediction_arrays(copy.deepcopy(expected), expected)
        for key in expected:
            actual = copy.deepcopy(expected)
            actual[key] = actual[key].copy()
            actual[key].flat[0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.assert_prediction_arrays(actual, expected)
        actual = copy.deepcopy(expected)
        actual["logits"] = actual["logits"].astype(np.float64)
        with self.assertRaises(ValueError):
            study.assert_prediction_arrays(actual, expected)

    def test_unapproved_payload_stops_before_source_runtime_or_output(self):
        from cst_research.experiments import higgs_dst_confirmation_driver as common

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "research-inputs").mkdir()
            (root / "research-inputs/future-confirmation.npz").write_bytes(
                b"never read"
            )
            with (
                patch.object(Path, "cwd", return_value=root),
                patch.object(common, "verify_consumer") as verify,
            ):
                with self.assertRaisesRegex(ValueError, "exact ten"):
                    driver.run(SimpleNamespace(output=root / "fresh"))
                verify.assert_not_called()
            self.assertFalse((root / "fresh").exists())

    def test_destinations_never_write_inside_inherited_source_or_raw_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prior, source = root / "prior", root / "consumer"
            prior.mkdir()
            source.mkdir()
            output, cache = root / "fresh-result", root / "cache"
            self.assertEqual(
                driver.write_destinations(output, cache, prior, source), (output, cache)
            )
            for bad_output, bad_cache in (
                (prior / "new", cache),
                (source / "new", cache),
                (output, prior / "cache"),
                (output, source / "cache"),
                (output, output / "cache"),
            ):
                with (
                    self.subTest(output=bad_output, cache=bad_cache),
                    self.assertRaises(ValueError),
                ):
                    driver.write_destinations(bad_output, bad_cache, prior, source)
            self.assertFalse(output.exists())
            self.assertFalse(cache.exists())
            self.assertEqual(list(prior.iterdir()), [])
            self.assertEqual(list(source.iterdir()), [])

    def test_registry_is_prepared_not_a_completed_learning_result(self):
        file = (
            Path(__file__).resolve().parents[1]
            / "src/cst_research/studies/higgs-depth-normalization-learning-20261106.json"
        )
        registered = json.loads(file.read_text())
        self.assertEqual(registered["protocol"], study.protocol())
        self.assertEqual(registered["status"], "PLANNED")
        self.assertEqual(registered["preparation_status"], "PREPARED")


SOURCE = os.environ.get("CST_DEPTH_SOURCE")


@unittest.skipUnless(SOURCE, "explicit unchanged external975 fixture required")
class Public975LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(SOURCE).resolve()
        manifest = json.loads(Path(os.environ["CST_DEPTH_MANIFEST"]).read_text())
        if manifest["source_commit"] != study.stock.SOURCE:
            raise ValueError("wrong public975 fixture")
        for name, digest in manifest["files"].items():
            file = source / name
            if (
                file.is_symlink()
                or hashlib.sha256(file.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("source fixture differs")
        sys.path[:0] = [str(source / "src"), str(source)]
        import torchcst

        if not Path(torchcst.__file__).resolve().is_relative_to(source):
            raise ValueError("wrong imported torchcst source")
        torch.set_num_threads(1)

    def test_actual_public_factory_topology_matches_protocol_four_hidden_three_squares(
        self,
    ):
        from torchcst.nn import CSTModule

        for case in ("dense-stock", "cst104-stock", "cst1048-stock"):
            model = study.model_contract.make_model(case)
            maps = [
                layer
                for layer in model
                if isinstance(layer, (torch.nn.Linear, CSTModule))
            ]
            dims = [(layer.in_features, layer.out_features) for layer in maps]
            architecture = [dims[0][0], *(out for _, out in dims)]
            self.assertEqual(architecture, study.protocol()["architecture"])
            self.assertEqual(len(maps), 5)
            self.assertEqual(sum(a == b == 1024 for a, b in dims), 3)
            self.assertEqual(len(list(model.parameters())), 7)

    def test_bare_loop_real_cpu_updates_and_training_peak_capture_precedes_evaluation(
        self,
    ):
        # Two real CPU B4 steps test plumbing only, not a fake epoch/GPU admission.
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        x = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
        labels = torch.tensor([0, 1] * 4)
        totals = (torch.zeros(()), torch.zeros((), dtype=torch.int64))
        events = []
        from benchmarks.training import full_step_graph

        actual_step = full_step_graph.ordinary_step

        def step(*args):
            events.append("step")
            return actual_step(*args)

        def note(name, result=None):
            def call(*args):
                events.append(name)
                return result

            return call

        with (
            patch.object(torch.cuda, "synchronize", side_effect=note("sync")),
            patch.object(
                torch.cuda, "reset_peak_memory_stats", side_effect=note("reset")
            ),
            patch.object(
                torch.cuda, "max_memory_allocated", side_effect=note("allocated", 123)
            ),
            patch.object(
                torch.cuda, "max_memory_reserved", side_effect=note("reserved", 456)
            ),
            patch.object(full_step_graph, "ordinary_step", side_effect=step),
        ):
            record = study.train_epoch(
                model, optimizer, x, labels, torch.arange(8), totals
            )
            events.append("evaluation-boundary")
        self.assertEqual(
            events,
            [
                "sync",
                "reset",
                "step",
                "step",
                "sync",
                "allocated",
                "reserved",
                "evaluation-boundary",
            ],
        )
        self.assertEqual(record["peak_allocated_bytes"], 123)
        self.assertEqual(record["peak_reserved_bytes"], 456)
        self.assertTrue(all(float(s["step"]) == 2 for s in optimizer.state.values()))
        self.assertTrue(torch.isfinite(totals[0]))

    def test_nested_checkpoint_finiteness_covers_model_moments_and_actual_gradients(
        self,
    ):
        model = study.model_contract.make_model("dense-stock")
        optimizer, _ = study.stock.make_optimizer(model, cpu_reference=True)
        x = torch.randn(4, 28, generator=torch.Generator().manual_seed(43))
        torch.nn.functional.cross_entropy(
            model(x), torch.tensor([0, 1, 0, 1])
        ).backward()
        optimizer.step()
        raw = study.stock.checkpoint(model, optimizer, 1, 1, {}, None)
        self.assertEqual(study.nonfinite_checkpoint_fields(raw), [])
        for branch in ("model", "optimizer", "gradients"):
            changed = copy.deepcopy(raw)
            if branch == "model":
                changed["model"]["0.weight"][0, 0] = float("nan")
            elif branch == "optimizer":
                changed["optimizer"]["state"][0]["exp_avg_sq"][0, 0] = float("inf")
            else:
                changed["gradients"]["0.weight"][0, 0] = float("nan")
            with self.subTest(branch=branch):
                bad = study.nonfinite_checkpoint_fields(changed)
                self.assertEqual(len(bad), 1)
                self.assertTrue(bad[0].startswith("/" + branch + "/"))

    def test_fresh_state0_contract_and_full_selected_restore(self):
        model = study.model_contract.make_model("dense-stock")
        optimizer, contract = study.stock.make_optimizer(model, cpu_reference=True)
        prior_identity = {"consumer_commit": proof.PRIOR_COMMIT}
        config = {
            "cohort": study.model_contract.COHORT,
            "architecture": study.protocol()["architecture"],
            "inherited_factory_configuration": model._depth_configuration,
            "case": "dense-stock",
            "identity": prior_identity,
            "normalization": "none",
            "gains": [1.0, 1.0, 1.0],
            "optimizer_contract": contract,
        }
        raw = study.stock.checkpoint(model, optimizer, 0, 0, config, None)
        study.assert_initial(
            model,
            optimizer,
            raw,
            "dense-stock",
            {"gains": [1.0] * 3},
            prior_identity,
            contract,
        )
        wrong = copy.deepcopy(raw)
        wrong["configuration"]["optimizer_contract"]["lr"] = 1e-4
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            study.assert_initial(
                model,
                optimizer,
                wrong,
                "dense-stock",
                {"gains": [1.0] * 3},
                prior_identity,
                contract,
            )
        x = torch.randn(4, 28, generator=torch.Generator().manual_seed(41))
        torch.nn.functional.cross_entropy(
            model(x), torch.tensor([0, 1, 0, 1])
        ).backward()
        optimizer.step()
        raw = study.stock.checkpoint(model, optimizer, 1, 1, config, None)

        # Restore uses actual factory/optimizer; force CPU reference instead of GPU selector.
        def cpu_fresh(case, gains, device):
            m = study.model_contract.make_model(case, gains=gains).to(device)
            o, c = study.stock.make_optimizer(m, cpu_reference=True)
            return m, o, c

        with patch.object(study, "fresh", side_effect=cpu_fresh):
            restored, opt = study.restore_selected("dense-stock", [1.0] * 3, raw, "cpu")
        proof.assert_equal(
            study.core_state(study.stock.checkpoint(restored, opt, 1, 1, {}, None)),
            study.core_state(raw),
        )
        self.assertEqual(float(next(iter(opt.state.values()))["step"]), 1)
