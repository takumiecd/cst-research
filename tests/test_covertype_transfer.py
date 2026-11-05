"""CPU policy/preprocessing/metric/closed-evaluation gates; no GPU admission."""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from cst_research.experiments.covertype_campaign import (
    edge_extension,
    select,
    validate_learning,
)
from cst_research.experiments.covertype_data import partition, specification
from cst_research.experiments.covertype_transfer import configuration, metrics, protocol


class CovertypePolicyTests(unittest.TestCase):
    def test_registry_protocol_matches_executable_prepared_study(self):
        from cst_research.registry import load_studies

        study = next(
            r for r in load_studies() if r["id"] == "covertype-transfer-20261106"
        )
        self.assertEqual(study["protocol"], protocol())
        self.assertEqual(study["status"], "PLANNED")

    def test_prespecified_bounded_budget_and_ties(self):
        self.assertEqual(protocol()["architecture"], [54, 1024, 1024, 7])
        self.assertEqual(
            [configuration(m)["atoms"] for m in ("cst-1000", "cst-100")], [1048, 10485]
        )
        rows = [
            {"lr": lr, "best_development_loss": 1.0, "best_epoch": 2}
            for lr in (0.0001, 0.0003, 0.001)
        ]
        self.assertEqual(edge_extension(rows), 0.00003)
        rows[2]["best_development_loss"] = 0.9
        self.assertEqual(edge_extension(rows), 0.003)
        rows[1]["best_development_loss"] = 0.8
        self.assertIsNone(edge_extension(rows))
        rows[0].update(best_development_loss=0.8, best_epoch=1)
        self.assertEqual(select(rows)["lr"], 0.0001)
        with self.assertRaises(ValueError):
            edge_extension(rows[:2])

    def test_raw_protocol_has_no_geographic_or_formal_claim(self):
        self.assertEqual(specification()["permutation"]["seed"], 20261106)
        self.assertEqual(
            specification()["split_sizes"],
            {"train": 100000, "development": 50000, "heldout": 50000},
        )
        self.assertFalse(protocol()["formal_noninferiority"])
        self.assertEqual(protocol()["position_multiplier"], 1)

    def test_cell_validation_rejects_horizon_heldout_and_wrong_best(self):
        identity = {"protocol_sha256": "fixed"}
        cfg = {
            "id": "dense",
            "seed": 17,
            "lr": 0.0003,
            "epochs": 3,
            "batch_size": 4,
            "TF32": False,
            "mode": "eager",
            **identity,
        }
        rows = [
            {"epoch": i, "updates": 25000 * i, "development": {"loss": v}}
            for i, v in enumerate([0.9, 0.8, 0.8], 1)
        ]
        learning = {
            "configuration": cfg,
            "epochs": rows,
            "heldout_evaluated": False,
            "qualification": {"status": "passed"},
            "best": {"epoch": 2, "development": rows[1]["development"]},
        }
        validate_learning(learning, "dense", 17, 0.0003, 3, identity)
        for change in ("best", "heldout", "updates"):
            wrong = copy.deepcopy(learning)
            if change == "best":
                wrong["best"]["epoch"] = 3
            elif change == "heldout":
                wrong["heldout_evaluated"] = True
            else:
                wrong["epochs"][0]["updates"] = 1
            with self.assertRaises(ValueError):
                validate_learning(wrong, "dense", 17, 0.0003, 3, identity)


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "NumPy required for preprocessing/metrics")
class CovertypeArrayTests(unittest.TestCase):
    def test_disjoint_rows_train_only_statistics_and_untouched_binary(self):
        raw = np.zeros((42, 55), np.float32)
        raw[:, :10] = np.arange(42)[:, None] * np.arange(1, 11)[None, :]
        raw[:, 10:54] = (np.arange(42)[:, None] + np.arange(44)[None, :]) % 2
        raw[:, 54] = np.arange(42) % 7 + 1
        splits, fit = partition(raw, (14, 14, 14))
        allids = np.concatenate([s["row_ids"] for s in splits.values()])
        self.assertEqual(len(np.unique(allids)), 42)
        train = raw[splits["train"]["row_ids"], :10].astype(np.float64)
        np.testing.assert_array_equal(np.asarray(fit["mean_float64"]), train.mean(0))
        for split in splits.values():
            np.testing.assert_array_equal(
                split["x"][:, 10:], raw[split["row_ids"], 10:54]
            )
        changed = raw.copy()
        changed[splits["heldout"]["row_ids"], :10] += 100000
        _, newfit = partition(changed, (14, 14, 14))
        self.assertEqual(fit["mean_float64"], newfit["mean_float64"])
        self.assertEqual(fit["std_float64"], newfit["std_float64"])
        self.assertLess(float(np.abs(splits["train"]["x"][:, :10].mean(0)).max()), 1e-6)

    def test_multiclass_auc_uses_float64_softmax_and_macro_f1(self):
        seen = []

        def auc(scores, targets):
            self.assertEqual(scores.dtype, np.float64)
            seen.append(scores.copy())
            positives = scores[targets == 1]
            negatives = scores[targets == 0]
            return float(
                (
                    (positives[:, None] > negatives).sum()
                    + 0.5 * (positives[:, None] == negatives).sum()
                )
                / (len(positives) * len(negatives))
            )

        labels = np.tile(np.arange(7), 2)
        logits = np.full((14, 7), -2.0, np.float32)
        logits[np.arange(14), labels] = 3.0
        result = metrics(logits, labels, auc)
        self.assertEqual(
            (result["accuracy"], result["macro_f1"], result["macro_ovr_auc"]),
            (1.0, 1.0, 1.0),
        )
        self.assertTrue(np.allclose(np.stack(seen, 1).sum(1), 1.0))
        shifted = metrics(logits + np.arange(14)[:, None] * 100, labels, auc)
        self.assertAlmostEqual(shifted["loss"], result["loss"])
        with self.assertRaises(ValueError):
            metrics(logits[:2], labels[:2], auc)


SOURCE = os.environ.get("CST_COVERTYPE_SOURCE")


@unittest.skipUnless(SOURCE, "explicit pinned external975 source required")
class CovertypePublicModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True
        source = Path(SOURCE).resolve()
        manifest = json.loads(Path(os.environ["CST_COVERTYPE_MANIFEST"]).read_text())
        if manifest["source_commit"] != "97580582ed28c4e494bf5c2656bbe94a85c544c3":
            raise ValueError("wrong source")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError("source bytes differ")
        sys.path[:0] = [str(source / "src"), str(source)]
        import torch
        import torchcst

        if not Path(torchcst.__file__).resolve().is_relative_to(source):
            raise ValueError("wrong imported runtime")
        from cst_research.experiments.covertype_transfer import (
            make_model,
            make_optimizer,
            reference_forward,
        )

        cls.torch = torch
        cls.make_model = staticmethod(make_model)
        cls.make_optimizer = staticmethod(make_optimizer)
        cls.reference_forward = staticmethod(reference_forward)
        torch.set_num_threads(1)

    def test_public_dense_io_shared_and_all_five_parameters_owned(self):
        torch = self.torch
        dense = self.make_model("dense", 17, width=128)
        for method in ("cst-1000", "cst-100"):
            model = self.make_model(method, 17, width=128)
            for i in (0, 4):
                self.assertTrue(torch.equal(model[i].weight, dense[i].weight))
                self.assertTrue(torch.equal(model[i].bias, dense[i].bias))
            optimizer = self.make_optimizer(model, 0.0003, cpu_reference=True)
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            optimizer.step()
            self.assertEqual(len(optimizer.state), 5)
            self.assertTrue(
                all(float(s["step"]) == 1 for s in optimizer.state.values())
            )

    def test_full_network_fp64_public_reference_all_gradients(self):
        torch = self.torch
        for method in ("dense", "cst-1000", "cst-100"):
            model = self.make_model(method, 17, width=128).double()
            x = torch.randn(4, 54, dtype=torch.float64, requires_grad=True)
            y = self.reference_forward(model, x)
            self.assertEqual(tuple(y.shape), (4, 7))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
            )

    def test_selected_checkpoint_rejects_real_one_step_state_as_full_epoch(self):
        from cst_research.experiments.covertype_transfer import (
            checkpoint,
            numerical_contract,
            validate_checkpoint,
        )

        torch = self.torch
        model = self.make_model("cst-1000", 29)
        optimizer = self.make_optimizer(model, 0.0003, cpu_reference=True)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        identity = {"unit_test": "clock contract"}
        config = {
            **configuration("cst-1000"),
            "numerical_contract": numerical_contract(),
            **identity,
            "seed": 29,
            "lr": 0.0003,
            "epochs": 10,
        }
        development = {"loss": 0.5}
        raw = checkpoint(model, optimizer, 1, 25000, config, development)
        row = {
            "method": "cst-1000",
            "seed": 29,
            "lr": 0.0003,
            "best_epoch": 1,
            "development": development,
        }
        with self.assertRaisesRegex(ValueError, "clocks/finite"):
            validate_checkpoint(raw, row, identity)

    def test_serialized_actual_dense_and_both_cst_checkpoint_schemas(self):
        from unittest.mock import patch

        from cst_research.experiments import covertype_transfer as consumer

        torch = self.torch
        for method in ("dense", "cst-1000", "cst-100"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as name:
                model = self.make_model(method, 29)
                optimizer = self.make_optimizer(model, 0.0003, cpu_reference=True)
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                identity = {
                    "unit_test": "actual serialized one-update fixture, not model-quality evidence"
                }
                config = {
                    **configuration(method),
                    "numerical_contract": consumer.numerical_contract(),
                    **identity,
                    "seed": 29,
                    "lr": 0.0003,
                    "epochs": 10,
                }
                row = {
                    "method": method,
                    "seed": 29,
                    "lr": 0.0003,
                    "best_epoch": 1,
                    "development": {"loss": 0.5},
                }
                raw = consumer.checkpoint(
                    model, optimizer, 1, 1, config, row["development"]
                )
                file = Path(name) / "actual.pt"
                torch.save(raw, file)
                restored = torch.load(file, map_location="cpu", weights_only=True)
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    consumer.validate_checkpoint(restored, row, identity)
                # Explicit miniature unit epoch: production remains25k and is
                # checked above against this real one-update partial state.
                with patch.object(consumer, "UPDATES_PER_EPOCH", 1):
                    consumer.validate_checkpoint(restored, row, identity)
                    for corruption in (
                        "dropmoment",
                        "moment-shape",
                        "moment-dtype",
                        "gradient-shape",
                        "ownership",
                    ):
                        with self.subTest(corruption=corruption):
                            invalid = copy.deepcopy(restored)
                            groups = invalid["optimizer"]["param_groups"]
                            identifier = groups[0]["params"][0]
                            name0 = "0.weight"
                            if corruption == "dropmoment":
                                del invalid["optimizer"]["state"][identifier][
                                    "exp_avg_sq"
                                ]
                            elif corruption == "moment-shape":
                                invalid["optimizer"]["state"][identifier]["exp_avg"] = (
                                    invalid["optimizer"]["state"][identifier][
                                        "exp_avg"
                                    ][:-1]
                                )
                            elif corruption == "moment-dtype":
                                invalid["optimizer"]["state"][identifier][
                                    "exp_avg_sq"
                                ] = invalid["optimizer"]["state"][identifier][
                                    "exp_avg_sq"
                                ].double()
                            elif corruption == "gradient-shape":
                                invalid["gradients"][name0] = invalid["gradients"][
                                    name0
                                ][:-1]
                            else:
                                groups[0]["params"][0] = groups[0]["params"][1]
                            with self.assertRaises(ValueError):
                                consumer.validate_checkpoint(invalid, row, identity)
                    if method != "dense":
                        invalid = copy.deepcopy(restored)
                        invalid["model"]["2.atom_state.optimizer_state._extra_state"][
                            "fields"
                        ]["exp_avg"].add_(1)
                        with self.assertRaisesRegex(
                            ValueError, "model-side optimizer store"
                        ):
                            consumer.validate_checkpoint(invalid, row, identity)
                    key = "2.weight" if method == "dense" else "2.atom_state.atoms.p"
                    malformed = copy.deepcopy(restored)
                    malformed["model"][key] = malformed["model"][key][:-1]
                    with self.assertRaisesRegex(ValueError, "architecture differs"):
                        consumer.validate_checkpoint(malformed, row, identity)


@unittest.skipUnless(np is not None, "NumPy required")
class CovertypeSelectedPredictionTests(unittest.TestCase):
    def test_reload_equality_uses_array_bytes_and_saved_metrics(self):
        from cst_research.experiments.covertype_transfer import (
            verify_selected_development,
        )

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            logits = np.zeros((7, 7), np.float32)
            labels = np.arange(7, dtype=np.int64)
            ids = np.arange(7, dtype=np.int64)
            np.savez_compressed(
                root / "best-dev-predictions.npz",
                logits=logits,
                labels=labels,
                row_ids=ids,
            )
            best = {
                "development": {"loss": 0.5},
                "checkpoint_sha256": "unit-checkpoint",
            }
            proof = verify_selected_development(
                root, best, {"loss": 0.5}, logits, labels, ids
            )
            self.assertTrue(proof["metrics_equal"])
            wrong = logits.copy()
            wrong[0, 0] = -0.0
            self.assertTrue(np.array_equal(wrong, logits))
            with self.assertRaisesRegex(ValueError, "reload predictions"):
                verify_selected_development(
                    root, best, {"loss": 0.5}, wrong, labels, ids
                )
            with self.assertRaisesRegex(ValueError, "reload predictions"):
                verify_selected_development(
                    root, best, {"loss": 0.6}, logits, labels, ids
                )


class CovertypeFreezeTests(unittest.TestCase):
    def test_incomplete_family_or_changed_selection_refused_before_heldout(self):
        from cst_research.experiments.covertype_assessment import audit_freeze
        from cst_research.experiments.higgs_dst_comparison import file_sha, write

        with tempfile.TemporaryDirectory() as name:
            state = Path(name)
            identity = {"source": "unit-test"}
            selected = {
                "identity": identity,
                "heldout_evaluated": False,
                "selected": [
                    {"method": m, "lr": 0.0003}
                    for m in ("dense", "cst-1000", "cst-100")
                ],
            }
            write(state / "selected-configurations.json", selected)
            freeze = {
                "identity": identity,
                "heldout_evaluated": False,
                "selection_sha256": file_sha(state / "selected-configurations.json"),
                "models": [],
            }
            write(state / "freeze.json", freeze)
            with self.assertRaisesRegex(ValueError, "nine models"):
                audit_freeze(
                    state,
                    state / "freeze.json",
                    file_sha(state / "freeze.json"),
                    identity,
                )
            with self.assertRaisesRegex(ValueError, "sealed freeze SHA"):
                audit_freeze(state, state / "freeze.json", "0" * 64, identity)
            selected["selected"][0]["lr"] = 0.001
            write(state / "selected-configurations.json", selected)
            with self.assertRaisesRegex(ValueError, "configurations changed"):
                audit_freeze(
                    state,
                    state / "freeze.json",
                    file_sha(state / "freeze.json"),
                    identity,
                )

    def test_screen_rejects_missing_or_second_edge(self):
        from cst_research.experiments.covertype_campaign import validate_screen

        identity = {"unit": "test"}
        rows = [
            {
                "method": "dense",
                "seed": 17,
                "epochs": 3,
                "lr": lr,
                "best_development_loss": 1.0,
                "best_epoch": 2,
            }
            for lr in (0.0001, 0.0003, 0.001)
        ]
        record = {
            "identity": identity,
            "candidates": rows,
            "edge_lr": 0.00003,
            "selected": select(rows),
        }
        with self.assertRaisesRegex(ValueError, "missing or excess"):
            validate_screen(record, "dense", identity)
        extra = {**rows[0], "lr": 0.00003}
        rows.append(extra)
        record["selected"] = select(rows)
        validate_screen(record, "dense", identity)
        rows.append({**extra, "lr": 0.003})
        with self.assertRaisesRegex(ValueError, "missing or excess"):
            validate_screen(record, "dense", identity)


class CovertypeGlobalGateOrderTests(unittest.TestCase):
    def test_ninth_raw_checkpoint_failure_precedes_any_heldout_open(self):
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch

        from cst_research.experiments import covertype_assessment as assessor

        methods = ("dense", "cst-1000", "cst-100")
        seeds = (29, 43, 71)
        rows = [
            {"method": m, "seed": s, "directory": f"cells/{m}-{s}"}
            for m in methods
            for s in seeds
        ]
        calls = []

        def validate(raw, row, identity):
            calls.append((row["method"], row["seed"]))
            if len(calls) == 9:
                raise ValueError("ninth raw checkpoint rejected")

        fake_torch = ModuleType("torch")
        fake_torch.load = lambda *a, **k: {}
        fake_numpy = ModuleType("numpy")
        fake_numpy.load = lambda *a, **k: self.fail("heldout must not open")
        fake_benchmark = ModuleType("benchmarks.training.higgs_confirmation")
        fake_benchmark._configure = lambda: self.fail(
            "no CUDA setup before complete raw gate"
        )
        args = SimpleNamespace(
            output=Path("/nonexistent-unit-output"),
            state=Path("/unit-state"),
            frozen_index=Path("/unit-freeze"),
            frozen_index_sha256="unit",
            data_manifest=Path("/unit-manifest"),
            heldout=Path("/forbidden-heldout"),
        )
        with (
            patch.object(
                assessor.argparse.ArgumentParser, "parse_args", return_value=args
            ),
            patch.object(
                assessor.cell, "admit_sources", return_value={"unit": "identity"}
            ),
            patch.object(assessor, "audit_freeze", return_value={"models": rows}),
            patch.object(assessor.cell, "validate_checkpoint", side_effect=validate),
            patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "numpy": fake_numpy,
                    "benchmarks.training.higgs_confirmation": fake_benchmark,
                },
            ),
            patch.object(
                assessor,
                "file_sha",
                side_effect=AssertionError(
                    "must not hash heldout before full raw gate"
                ),
            ),
            self.assertRaisesRegex(ValueError, "ninth raw checkpoint"),
        ):
            assessor.main()
        self.assertEqual(len(calls), 9)


if __name__ == "__main__":
    unittest.main()
