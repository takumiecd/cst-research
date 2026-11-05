"""CPU-only global78 admission, exactly-once scheduling and IUT16 checks."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import math
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cst_research.experiments import higgs_dst_assessment as module


def fixture(root, reuse=False):
    state = root / "state"
    state.mkdir()
    data = root / "screen.npz"
    data.write_bytes(b"fixture dataset; no confirmation read")
    data_sha = module.file_sha(data)
    source_index, source_digests = {"sources": {}}, {}
    for commit in (module.DST_SOURCE, module.CONTROL_SOURCE):
        directory = root / commit
        (directory / "benchmarks/training").mkdir(parents=True)
        names = ["benchmarks/training/higgs_confirmation.py"]
        if commit == module.DST_SOURCE:
            names.append("benchmarks/training/higgs_dst_reference.py")
        hashes = {}
        for name in names:
            (directory / name).write_text("# fixture: never imported\n")
            hashes[name] = module.file_sha(directory / name)
        manifest = root / (commit + "-manifest.json")
        module.write(manifest, {"source_commit": commit, "files": hashes})
        source_digests[commit] = module.file_sha(manifest)
        source_index["sources"][commit] = {
            "directory": str(directory),
            "manifest_file": str(manifest),
        }
    index_file = root / "source-index.json"
    module.write(index_file, source_index)
    dst, controls = module.expected_keys()
    rows = []
    for width, kind, divisor, seed in sorted(dst | controls):
        model_id = f"{width}-{kind}-{divisor}-{seed}"
        directory = state / "models" / model_id
        directory.mkdir(parents=True)
        checkpoint = directory / "best-dev.pt"
        checkpoint.write_text(model_id)
        metrics = {"loss": 0.5, "accuracy": 0.7, "roc_auc": 0.8, "examples": 50000}
        best = {
            "epoch": 1,
            "updates": 25000,
            "development": metrics,
            "checkpoint_sha256": module.file_sha(checkpoint),
        }
        epochs = [
            {
                "epoch": i,
                "updates": i * 25000,
                "development": metrics
                if i == 1
                else {**metrics, "loss": 0.51 + i * 0.001},
            }
            for i in range(1, 11)
        ]
        learning = {
            "kind": kind,
            "width": width,
            "divisor": divisor,
            "seed": seed,
            "lr": 0.0003,
            "dataset_sha256": data_sha,
            "batch_size": 4,
            "training_rows": 100000,
            "development_rows": 50000,
            "confirmation_evaluated": False,
            "test_evaluated": False,
            "validation_partition": {
                "development": [0, 50000],
                "confirmation": [50000, 100000],
            },
            "epochs": epochs,
            "best_development_checkpoint": best,
        }
        module.write(directory / "best-dev.json", best)
        module.write(directory / "learning.json", learning)
        row = {
            "id": model_id,
            "role": "dst" if kind in ("static", "rigl") else "control",
            "kind": kind,
            "width": width,
            "divisor": divisor,
            "seed": seed,
            "lr": 0.0003,
            "horizon": 10,
            "best_epoch": 1,
            "checkpoint_sha256": module.file_sha(checkpoint),
            "best_metadata_sha256": module.file_sha(directory / "best-dev.json"),
            "learning_sha256": module.file_sha(directory / "learning.json"),
            "relative_directory": str(directory.relative_to(state)),
            "source_commit": module.DST_SOURCE
            if kind in ("static", "rigl")
            else module.CONTROL_SOURCE,
            "producer_identity": {"dataset_sha256": data_sha, "producer": "fixture"},
            "reuse": reuse and kind in ("dense", "cst"),
        }
        rows.append(row)
    original_rows = []
    for row in rows:
        if row["role"] == "control":
            original_rows.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "seed",
                            "lr",
                            "horizon",
                            "best_epoch",
                            "checkpoint_sha256",
                            "best_metadata_sha256",
                            "learning_sha256",
                        )
                    },
                    "configuration": {
                        key: row[key] for key in ("kind", "width", "divisor")
                    },
                }
            )
    proof = {
        "configuration_count": 30,
        "configurations": original_rows,
        "identity": {"dataset_sha256": data_sha},
        "confirmation_partition": [50000, 100000],
        "test_evaluated": False,
    }
    proof_file = state / "original-proof.json"
    module.write(proof_file, proof)
    frozen = {
        "schema_version": 1,
        "identity": {"producer": "fixture"},
        "runtime_fingerprint": {
            "torch": "fixture",
            "cuda": "fixture",
            "gpu": "fixture",
        },
        "configuration_count": 78,
        "dst_count": 48,
        "control_count": 30,
        "dataset_sha256": data_sha,
        "source_manifest_sha256": source_digests,
        "comparison_horizons": {"1024": 10, "2048": 10},
        "confirmation_partition": [50000, 100000],
        "test_evaluated": False,
        "original_confirmation_freeze_kind": "comparison-owned-proof",
        "original_confirmation_freeze_file": "original-proof.json",
        "original_confirmation_freeze_sha256": module.file_sha(proof_file),
        "original_confirmation_freeze_canonical_sha256": module.json_sha(proof),
        "configurations": rows,
    }
    frozen_file = state / "global-freeze.json"
    module.write(frozen_file, frozen)
    args = SimpleNamespace(
        frozen_index=frozen_file,
        frozen_index_sha256=module.file_sha(frozen_file),
        campaign_state=state,
        data=data,
        source_index=index_file,
        output=root / "output",
        original_assessment_job=None,
    )
    return args, frozen, data_sha


def fake_record(row, directory, admission, sources):
    assert (
        admission["verified_dst_count"] == 48
        and admission["verified_control_count"] == 30
    )
    (directory / "predictions.npz").write_bytes(
        b"fake prediction bytes; scheduler test only"
    )
    module.write(
        directory / "evaluation.json",
        {
            "model_id": row["id"],
            "predictions_sha256": module.file_sha(directory / "predictions.npz"),
        },
    )


def fake_validate(directory, row, freeze_sha, data_sha):
    return module.read(Path(directory) / "evaluation.json")


class AdmissionTests(unittest.TestCase):
    def test_complete78_gate_is_cpu_only_and_does_not_import_fixture_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, data_sha = fixture(Path(temporary))
            with patch.object(module, "DATA_SHA256", data_sha):
                _, admission, sources = module.audit_freeze(
                    args.frozen_index,
                    args.frozen_index_sha256,
                    args.campaign_state,
                    args.data,
                    args.source_index,
                )
            self.assertEqual(
                (admission["verified_dst_count"], admission["verified_control_count"]),
                (48, 30),
            )
            self.assertEqual(len(sources), 2)

    def test_worker_cannot_be_called_without_parent_once_only_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, freeze, data_sha = fixture(Path(temporary))
            args.model_id = freeze["configurations"][0]["id"]
            args.admission_file = Path(temporary) / "admission.json"
            with patch.object(module, "DATA_SHA256", data_sha):
                _, admission, _ = module.audit_freeze(
                    args.frozen_index,
                    args.frozen_index_sha256,
                    args.campaign_state,
                    args.data,
                    args.source_index,
                )
                module.write(args.admission_file, admission)
                with self.assertRaisesRegex(ValueError, "parent attempt"):
                    module.worker(args)

    def test_missing_last_model_blocks_every_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, freeze, data_sha = fixture(Path(temporary))
            (
                args.campaign_state
                / freeze["configurations"][-1]["relative_directory"]
                / "best-dev.pt"
            ).unlink()
            evaluator = Mock()
            with (
                patch.object(module, "DATA_SHA256", data_sha),
                self.assertRaises(ValueError),
            ):
                module.evaluate_all(args, evaluator)
            evaluator.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_selected_checkpoint_mutation_blocks_every_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, freeze, data_sha = fixture(Path(temporary))
            (
                args.campaign_state
                / freeze["configurations"][-1]["relative_directory"]
                / "best-dev.pt"
            ).write_text("changed selected weights")
            evaluator = Mock()
            with (
                patch.object(module, "DATA_SHA256", data_sha),
                self.assertRaisesRegex(ValueError, "artifact changed"),
            ):
                module.evaluate_all(args, evaluator)
            evaluator.assert_not_called()

    def test_missing_global_seal_and_mismatched_horizon_fail_before_scoring(self):
        for mutation in ("seal", "horizon", "duplicate", "source", "proof"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                args, freeze, data_sha = fixture(Path(temporary))
                if mutation == "seal":
                    args.frozen_index_sha256 = "0" * 64
                elif mutation == "horizon":
                    freeze["configurations"][0]["horizon"] = 20
                elif mutation == "duplicate":
                    freeze["configurations"][-1] = copy.deepcopy(
                        freeze["configurations"][0]
                    )
                elif mutation == "source":
                    freeze["configurations"][0]["source_commit"] = module.DST_SOURCE
                else:
                    (args.campaign_state / "original-proof.json").write_text("{}")
                if mutation not in ("seal", "proof"):
                    module.write(args.frozen_index, freeze)
                    args.frozen_index_sha256 = module.file_sha(args.frozen_index)
                evaluator = Mock()
                with (
                    patch.object(module, "DATA_SHA256", data_sha),
                    self.assertRaises(ValueError),
                ):
                    module.evaluate_all(args, evaluator)
                evaluator.assert_not_called()

    def test_successful_cache_never_rescores_any_of78_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, data_sha = fixture(Path(temporary))
            evaluator = Mock(side_effect=fake_record)
            with (
                patch.object(module, "DATA_SHA256", data_sha),
                patch.object(module, "validate_evaluation", side_effect=fake_validate),
            ):
                first = module.evaluate_all(args, evaluator)
                self.assertEqual(evaluator.call_count, 78)
                second = module.evaluate_all(args, evaluator)
                self.assertEqual(evaluator.call_count, 78)
            self.assertEqual(first, second)

    def test_partial_attempt_is_preserved_and_not_automatically_rescored(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, data_sha = fixture(Path(temporary))
            evaluator = Mock(
                side_effect=RuntimeError("interrupted after evaluation attempt began")
            )
            with (
                patch.object(module, "DATA_SHA256", data_sha),
                self.assertRaises(RuntimeError),
            ):
                module.evaluate_all(args, evaluator)
            self.assertEqual(evaluator.call_count, 1)
            with (
                patch.object(module, "DATA_SHA256", data_sha),
                self.assertRaisesRegex(ValueError, "attempt incomplete"),
            ):
                module.evaluate_all(args, evaluator)
            self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(
                len(list((args.campaign_state / "assessments").glob("*/started.json"))),
                1,
            )

    def test_reused_control_requires_original_assessment_before_any_dst_evaluation(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, data_sha = fixture(Path(temporary), reuse=True)
            evaluator = Mock()
            with (
                patch.object(module, "DATA_SHA256", data_sha),
                self.assertRaisesRegex(ValueError, "original975 assessment job"),
            ):
                module.evaluate_all(args, evaluator)
            evaluator.assert_not_called()

    def test_reused30_controls_are_imported_and_only48_dst_models_are_scored(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, freeze, data_sha = fixture(Path(temporary), reuse=True)
            args.original_assessment_job = Path(temporary) / "original-job"
            mapping = {
                module.row_key(row): {}
                for row in freeze["configurations"]
                if row["reuse"]
            }
            evaluator = Mock(side_effect=fake_record)

            def imported(original, directory, row, freeze_sha, frozen):
                fake_record(
                    row,
                    directory,
                    {"verified_dst_count": 48, "verified_control_count": 30},
                    {},
                )

            with (
                patch.object(module, "DATA_SHA256", data_sha),
                patch.object(module, "validate_evaluation", side_effect=fake_validate),
                patch.object(
                    module, "verify_original_assessment", return_value=mapping
                ),
                patch.object(
                    module, "import_original_prediction", side_effect=imported
                ) as importer,
            ):
                module.evaluate_all(args, evaluator)
            self.assertEqual(evaluator.call_count, 48)
            self.assertEqual(importer.call_count, 30)
            self.assertTrue(
                all(call.args[0]["role"] == "dst" for call in evaluator.call_args_list)
            )


class StatisticalTests(unittest.TestCase):
    def test_df2_quantile_and_survival_are_exact(self):
        self.assertAlmostEqual(module.t_df2_ppf(0.975), 4.3026527299, places=9)
        p = 0.05 / 16
        self.assertAlmostEqual(module.t_df2_sf(module.t_df2_ppf(1 - p)), p, places=14)

    def test_zero_variance_margin_boundary_does_not_support(self):
        boundary = module.seed_statistics([-0.002] * 3)
        self.assertEqual(boundary["one_sided_noninferiority_p"], 1)
        self.assertIsNone(boundary["noninferiority_t_statistic"])
        self.assertEqual(
            module.seed_statistics([0] * 3)["one_sided_noninferiority_p"], 0
        )

    def test_iut_requires_both_endpoints_and_can_be_inconclusive(self):
        dense = [{"accuracy": 0.7, "roc_auc": 0.8}] * 3
        worse = [{"accuracy": 0.7, "roc_auc": 0.7}] * 3
        self.assertEqual(
            module.compare_seed_metrics(dense, dense)["status"], "supported"
        )
        self.assertEqual(
            module.compare_seed_metrics(worse, dense)["status"], "inferior"
        )
        varied = [{"accuracy": 0.7 + d, "roc_auc": 0.8 + d} for d in (-0.2, 0, 0.2)]
        self.assertEqual(
            module.compare_seed_metrics(varied, dense)["status"], "inconclusive"
        )

    def test_primary_iut16_can_pass_while_diagnostic32_band_crosses(self):
        result = module.seed_statistics([0.0045, 0.0055, 0.0065])
        self.assertLess(result["one_sided_noninferiority_p"], 0.05 / 16)
        self.assertLess(result["diagnostic_joint32_seed_t_lower"], -0.002)
        self.assertGreater(result["primary_one_sided_seed_t_lower"], -0.002)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy required for saved prediction validation",
    )
    def test_tied_predictions_use_class0_and_exact_half_auc(self):
        import numpy as np

        logits = np.zeros((50000, 2), dtype=np.float32)
        labels = np.tile(np.array([0, 1], dtype=np.int64), 25000)
        actual = module.prediction_metrics(logits, labels)
        self.assertEqual(actual["accuracy"], 0.5)
        self.assertEqual(actual["roc_auc"], 0.5)
        self.assertAlmostEqual(actual["loss"], math.log(2), places=12)


@unittest.skipUnless(
    importlib.util.find_spec("numpy"), "NumPy required for saved-array integration"
)
class PredictionIntegrationTests(unittest.TestCase):
    def test_all78_saved_arrays_analyze_fixed16_family_without_gpu(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            args, _, data_sha = fixture(Path(temporary))
            logits = np.zeros((50000, 2), dtype=np.float32)
            labels = np.tile(np.array([0, 1], dtype=np.int64), 25000)
            metrics = module.prediction_metrics(logits, labels)

            def evaluate(row, directory, admission, sources):
                np.savez_compressed(
                    directory / "predictions.npz", logits=logits, labels=labels
                )
                module.write(
                    directory / "evaluation.json",
                    {
                        "model_id": row["id"],
                        "freeze_sha256": admission["freeze_sha256"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "dataset_sha256": data_sha,
                        "validation_partition": [50000, 100000],
                        "test_evaluated": False,
                        "predictions_sha256": module.file_sha(
                            directory / "predictions.npz"
                        ),
                        "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                        "confirmation": metrics,
                    },
                )

            with patch.object(module, "DATA_SHA256", data_sha):
                module.evaluate_all(args, evaluate)
                result = module.analyze(
                    args.output / "assessment-index.json", Path(temporary) / "analysis"
                )
            self.assertEqual(len(result["primary"]), 16)
            self.assertTrue(
                all(row["status"] == "supported" for row in result["primary"])
            )
            self.assertEqual(len(result["cst_dst_descriptive"]), 16)
            self.assertFalse(result["combined_joint_seed_example_confidence_claim"])
            self.assertFalse(result["test_evaluated"])
            self.assertTrue(
                all(
                    row["lower_range_endpoint_supported"]
                    for row in result["minimum_density"]
                )
            )

    def test_actual_original_bundle_is_importable_only_when_source_and_proof_match(
        self,
    ):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, freeze, data_sha = fixture(root, reuse=True)
            job = root / ("l4job-" + "a" * 32)
            results = job / "results"
            artifacts = results / "artifacts"
            artifacts.mkdir(parents=True)
            source_manifest = module.read(
                root / (module.CONTROL_SOURCE + "-manifest.json")
            )["files"]
            source_contents = {
                name: (root / module.CONTROL_SOURCE / name).read_bytes()
                for name in source_manifest
            }
            source_contents["research-inputs/screen.npz"] = args.data.read_bytes()
            source_contents["__pool_driver__.py"] = (
                b"# fixture driver, never executed\n"
            )

            def pack(path, contents):
                with tarfile.open(path, "w:gz") as bundle:
                    for name, content in contents.items():
                        member = tarfile.TarInfo(name)
                        member.size = len(content)
                        bundle.addfile(member, io.BytesIO(content))

            pack(job / "source.tar.gz", source_contents)
            source_sha = module.file_sha(job / "source.tar.gz")
            module.write(
                job / "spec.json",
                {
                    "id": job.name,
                    "source_sha256": source_sha,
                    "source_files": {
                        name: hashlib.sha256(value).hexdigest()
                        for name, value in source_contents.items()
                    },
                },
            )
            module.write(
                results / "result.json",
                {
                    "id": job.name,
                    "source_sha256": source_sha,
                    "returncode": 0,
                    "timed_out": False,
                },
            )
            module.write(
                artifacts / "bootstrap-source-verification.json",
                {"source_commit": module.CONTROL_SOURCE, "dataset_sha256": data_sha},
            )
            module.write(
                artifacts / "phase.json",
                {
                    "stage": "assess",
                    "status": "completed",
                    "confirmation_evaluated": True,
                    "test_evaluated": False,
                    "identity": {"dataset_sha256": data_sha},
                },
            )
            module.write(
                artifacts / "bootstrap-status.json", {"returncode": 0, "failure": None}
            )
            proof = module.read(args.campaign_state / "original-proof.json")
            module.write(artifacts / "confirmation-freeze.json", proof)
            logits = np.zeros((50000, 2), dtype=np.float32)
            labels = np.tile(np.array([0, 1], dtype=np.int64), 25000)
            metrics = module.prediction_metrics(logits, labels)
            entries = []
            for i, old in enumerate(proof["configurations"]):
                directory = artifacts / "evaluations" / str(i)
                directory.mkdir(parents=True)
                np.savez_compressed(
                    directory / "predictions.npz", logits=logits, labels=labels
                )
                evaluation = {
                    "checkpoint_sha256": old["checkpoint_sha256"],
                    "dataset_sha256": data_sha,
                    "validation_partition": [50000, 100000],
                    "test_evaluated": False,
                    "predictions_sha256": module.file_sha(
                        directory / "predictions.npz"
                    ),
                    "confirmation": metrics,
                    "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                    "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                }
                module.write(directory / "evaluation.json", evaluation)
                entries.append(
                    {
                        **old,
                        "evaluation_directory": str(directory.relative_to(artifacts)),
                        "evaluation_sha256": module.file_sha(
                            directory / "evaluation.json"
                        ),
                        "predictions_sha256": evaluation["predictions_sha256"],
                    }
                )
            module.write(
                artifacts / "assess-index.json",
                {
                    "confirmation_evaluated": True,
                    "test_evaluated": False,
                    "confirmation_freeze_sha256": module.file_sha(
                        artifacts / "confirmation-freeze.json"
                    ),
                    "assessments": entries,
                },
            )
            manifest = {
                str(p.relative_to(results)): module.file_sha(p)
                for p in results.rglob("*")
                if p.is_file()
            }
            module.write(results / "manifest.json", manifest)
            pack(
                job / "results.tar.gz",
                {
                    str(p.relative_to(results)): p.read_bytes()
                    for p in results.rglob("*")
                    if p.is_file()
                },
            )
            module.write(
                job / "receipt.json",
                {"archive_sha256": module.file_sha(job / "results.tar.gz")},
            )
            with patch.object(module, "DATA_SHA256", data_sha):
                _, _, sources = module.audit_freeze(
                    args.frozen_index,
                    args.frozen_index_sha256,
                    args.campaign_state,
                    args.data,
                    args.source_index,
                )
                admitted = module.verify_original_assessment(
                    job, freeze, args.campaign_state, sources
                )
                self.assertEqual(len(admitted), 30)
                recovered = root / (job.name + "-ack-recovery-20261105")
                job.rename(recovered)
                with self.assertRaisesRegex(ValueError, "supervisor rescue"):
                    module.verify_original_assessment(
                        recovered, freeze, args.campaign_state, sources
                    )
                receipt = module.read(recovered / "receipt.json")
                recovery_fields = {
                    "normal_pool_acknowledgment": False,
                    "recovery_mode": "supervisor-owned read-only result recovery",
                    "original_pool_job_status": "failed",
                }
                module.write(recovered / "receipt.json", receipt | recovery_fields)
                admitted = module.verify_original_assessment(
                    recovered, freeze, args.campaign_state, sources
                )
                self.assertEqual(len(admitted), 30)
                self.assertEqual(
                    next(iter(admitted.values()))["receipt_recovery"], recovery_fields
                )
                job = recovered
                destination = root / "imported"
                destination.mkdir()
                row = next(r for r in freeze["configurations"] if r["reuse"])
                module.import_original_prediction(
                    admitted[module.row_key(row)],
                    destination,
                    row,
                    args.frozen_index_sha256,
                    freeze,
                )
                record = module.validate_evaluation(
                    destination, row, args.frozen_index_sha256, data_sha
                )
                self.assertIn("not rescored", record["evaluation_origin"])
                (
                    admitted[module.row_key(row)]["directory"] / "predictions.npz"
                ).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    module.verify_original_assessment(
                        job, freeze, args.campaign_state, sources
                    )


if __name__ == "__main__":
    unittest.main()
