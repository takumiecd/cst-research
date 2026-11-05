"""CPU policy, evidence isolation and actual import-boundary regression tests."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cst_research.experiments import higgs_dst_confirmation as campaign
from cst_research.experiments import higgs_dst_evidence as evidence
from cst_research.experiments.higgs_dst_comparison import (
    file_sha,
    read,
    write,
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def tar(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, value in files.items():
            record = tarfile.TarInfo(name)
            record.size = len(value)
            archive.addfile(record, io.BytesIO(value))


def candidate(lr, loss=0.5, epochs=10, best_epoch=9):
    return {
        "lr": lr,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_development_loss": loss,
        "development_losses": [loss + 0.01] * (epochs - 1) + [loss],
        "checkpoint_sha256": "a" * 64,
        "best_metadata_sha256": "b" * 64,
        "learning_sha256": "c" * 64,
    }


class OuterEvidenceTests(unittest.TestCase):
    def make_job(self, root, *, rescued=False):
        job_id = "l4job-" + "1" * 32
        directory = root / (
            job_id + "-ack-recovery-20261105T0836" if rescued else job_id
        )
        directory.mkdir()
        source = {
            "tracked-driver.py": b"print(1)\n",
            "__pool_driver__.py": b"print(1)\n",
            "research-inputs/screen.npz": b"data",
        }
        trusted = root / "trusted.json"
        write(
            trusted,
            {
                "commit": "1" * 40,
                "files": {"tracked-driver.py": digest(source["tracked-driver.py"])},
            },
        )
        tar(directory / "source.tar.gz", source)
        spec = {
            "id": job_id,
            "source_sha256": file_sha(directory / "source.tar.gz"),
            "source_files": {n: digest(v) for n, v in source.items()},
        }
        write(directory / "spec.json", spec)
        result = {
            "id": job_id,
            "source_sha256": spec["source_sha256"],
            "returncode": 0,
            "timed_out": False,
        }
        phase = {
            "status": "completed",
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        status = {"returncode": 0, "failure": None}
        files = {
            "result.json": json.dumps(result).encode(),
            "artifacts/phase.json": json.dumps(phase).encode(),
            "artifacts/bootstrap-status.json": json.dumps(status).encode(),
        }
        manifest = {n: digest(v) for n, v in files.items()}
        files["manifest.json"] = json.dumps(manifest).encode()
        for name, value in files.items():
            path = directory / "results" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        tar(directory / "results.tar.gz", files)
        receipt = {"archive_sha256": file_sha(directory / "results.tar.gz")}
        if rescued:
            receipt.update(
                normal_pool_acknowledgment=False,
                original_pool_job_status="running",
                recovery_mode="supervisor-owned read-only result recovery",
            )
        write(directory / "receipt.json", receipt)
        return directory, trusted

    def test_normal_and_explicit_supervisor_rescue_preserve_job_identity(self):
        for rescued in (False, True):
            with self.subTest(rescued=rescued), tempfile.TemporaryDirectory() as tmp:
                directory, trusted = self.make_job(Path(tmp), rescued=rescued)
                job = evidence.PoolJob(directory, trusted, "tracked-driver.py")
                self.assertEqual(job.proof["job_id"], "l4job-" + "1" * 32)
                self.assertEqual(job.payload("screen.npz"), b"data")
                if rescued:
                    self.assertFalse(job.proof["receipt"]["normal_pool_acknowledgment"])

    def test_renamed_job_without_recovery_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, trusted = self.make_job(Path(tmp), rescued=True)
            receipt = read(directory / "receipt.json")
            receipt.pop("recovery_mode")
            write(directory / "receipt.json", receipt)
            with self.assertRaisesRegex(ValueError, "rescue receipt"):
                evidence.PoolJob(directory, trusted, "tracked-driver.py")

    def test_local_tamper_and_unbound_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, trusted = self.make_job(Path(tmp))
            (directory / "results/artifacts/phase.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "local pool artifact"):
                evidence.PoolJob(directory, trusted, "tracked-driver.py")
        with tempfile.TemporaryDirectory() as tmp:
            directory, trusted = self.make_job(Path(tmp))
            write(trusted, {"files": {"tracked-driver.py": "a" * 64}})
            with self.assertRaisesRegex(ValueError, "trusted producer inventory"):
                evidence.PoolJob(directory, trusted, "tracked-driver.py")

    def test_duplicate_tar_member_and_link_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                for _ in range(2):
                    member = tarfile.TarInfo("a")
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"a"))
            with self.assertRaisesRegex(ValueError, "unexpected archive"):
                evidence.archive_files(path, {"a": digest(b"a")})
            with tarfile.open(path, "w:gz") as archive:
                member = tarfile.TarInfo("a")
                member.type, member.linkname = tarfile.SYMTYPE, "../outside"
                archive.addfile(member)
            with self.assertRaises(ValueError):
                evidence.archive_files(path, {"a": digest(b"a")})


class SelectionAndTransitionTests(unittest.TestCase):
    def test_full_grid_and_exact_boundary_extension_are_recomputed(self):
        config = campaign.policy.configuration(1024, "rigl", 1000)
        rows = [
            candidate(lr, loss)
            for lr, loss in zip(campaign.policy.LRS, (0.48, 0.49, 0.50, 0.51))
        ]
        rows.append(candidate(0.00003, 0.475))
        selected = {
            "configuration": config,
            "development_seed": 17,
            "horizon": 10,
            "confirmation_evaluated": False,
            "test_evaluated": False,
            "edge_extension": 0.00003,
            "base_learning_rates": list(campaign.policy.LRS),
            "evaluated_learning_rates": sorted(r["lr"] for r in rows),
            "candidates": rows,
            "selected": rows[-1],
        }
        campaign.validate_selection(selected, rows, config)
        bad = copy.deepcopy(selected)
        bad["selected"] = rows[0]
        with self.assertRaisesRegex(ValueError, "actual|raw admitted"):
            campaign.validate_selection(bad, rows, config)
        with self.assertRaisesRegex(ValueError, "LR"):
            campaign.validate_selection(selected, rows[1:], config)

    def harness(self, root, *, original_horizon=10, trigger=True):
        obj = object.__new__(campaign.Campaign)
        obj.state, obj.output = root / "state", root / "output"
        obj.output.mkdir()
        obj.identity = {"fixture": True}
        obj.args = SimpleNamespace(width=1024, kind="rigl", divisor=1000)
        original = {"horizon": original_horizon, "configurations": []}
        for cfg in campaign.control_configurations(1024):
            base = [
                candidate(lr, 0.5 + i * 0.01)
                for i, lr in enumerate(campaign.policy.LRS)
            ]
            original["configurations"].append(
                {
                    **cfg,
                    "base_tuning_candidates": base,
                    "selected_development": candidate(0.0001, epochs=original_horizon),
                }
            )
        write(obj.state / "original-widths/1024.json", {"frozen": original})
        for cfg in campaign.policy.configurations(1024):
            selected = candidate(0.001)
            if trigger and cfg["id"] == "rigl-1000":
                selected["development_losses"][7] = 0.51
            else:
                selected["best_epoch"] = 2
            group = {
                "configuration": cfg,
                "horizon": 10,
                "selected": selected,
                "candidates": [
                    candidate(0.001, 0.49),
                    candidate(0.0003, 0.50),
                    candidate(0.003, 0.51),
                    candidate(0.0001, 0.52),
                ],
            }
            write(
                obj.state / "imported-hpo/1024" / (cfg["id"] + ".json"),
                {"selection": group},
            )
        return obj

    def test_one_dst_late_curve_requires_all_eight_fresh20_and_fresh_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self.harness(Path(tmp))
            plan = obj.plan_width()
            self.assertEqual(plan["horizon"], 20)
            self.assertTrue(plan["fresh20_controls"])
            with self.assertRaises(FileNotFoundError):
                obj.freeze_width()
            calls = []

            def cell(config, seed, lr, horizon):
                calls.append((config["id"], seed, lr, horizon))
                return candidate(lr, epochs=horizon), Path("unused")

            obj.cell, obj.record = cell, {"phase_id": "test", "cells": []}
            obj.retune()
            self.assertEqual(
                calls, [("rigl-1000", 17, 0.001, 20), ("rigl-1000", 17, 0.0003, 20)]
            )
            self.assertEqual(
                read(obj.state / "retuning/1024/rigl-1000.json")["horizon"], 20
            )
            repeated = obj.retune()
            self.assertEqual(repeated["horizon"], 20)
            self.assertEqual(len(calls), 2)
            obj.args.kind, obj.args.divisor = "dense", 100
            obj.retune(True)
            self.assertEqual(
                calls[-2:], [("dense", 17, 0.0001, 20), ("dense", 17, 0.0003, 20)]
            )

    def test_original20_always_requires_common_dst20_but_reuses_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self.harness(Path(tmp), original_horizon=20, trigger=False)
            plan = obj.plan_width()
            self.assertEqual(plan["horizon"], 20)
            self.assertFalse(plan["fresh20_controls"])
            with self.assertRaisesRegex(ValueError, "no fresh20"):
                obj.retune(True)

    def test_plan_rejects_postselection_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self.harness(Path(tmp), trigger=False)
            obj.plan_width()
            write(obj.state / "imported-hpo/1024/static-1000.json", {"modified": True})
            with self.assertRaisesRegex(ValueError, "changed after plan"):
                obj.plan(1024)

    def test_no_training_after_global_seal_and_no_undeclared_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self.harness(Path(tmp))
            write(obj.state / "confirmation-freeze.json", {"sealed": True})
            with self.assertRaisesRegex(ValueError, "immutable"):
                obj.retune()
            with self.assertRaisesRegex(ValueError, "immutable"):
                obj.confirm()


class OfflineRecoveryTests(unittest.TestCase):
    def test_offline_state_requires_pinned_auditor_receipt_and_exact_files(self):
        from cst_research.experiments.higgs_dst_confirmation_driver import (
            admit_offline_receipt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, inputs = root / "state", root / "inputs"
            state.mkdir()
            inputs.mkdir()
            binding = {
                "consumer_commit": "1" * 40,
                "consumer_manifest_sha256": "a" * 64,
            }
            write(
                state / "selected.json",
                {"producer_identity": {"old": True}, "selection": 0.001},
            )
            report = {
                "target_binding": binding,
                "auditor_consumer_commit": binding["consumer_commit"],
                "auditor_consumer_manifest_sha256": binding["consumer_manifest_sha256"],
                "verified_state_files": {
                    "selected.json": file_sha(state / "selected.json")
                },
            }
            write(state / "offline-audit-receipt.json", report)
            write(
                inputs / "recovery-state-receipt.json",
                {"offline_audit_receipt_sha256": "b" * 64},
            )
            with self.assertRaisesRegex(ValueError, "not pinned"):
                admit_offline_receipt(inputs, state, binding)
            write(
                inputs / "recovery-state-receipt.json",
                {
                    "offline_audit_receipt_sha256": file_sha(
                        state / "offline-audit-receipt.json"
                    )
                },
            )
            original = (state / "selected.json").read_bytes()
            (state / "selected.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed after"):
                admit_offline_receipt(inputs, state, binding)
            (state / "selected.json").write_bytes(original)
            result = admit_offline_receipt(inputs, state, binding)
            self.assertEqual(result["binding"], binding)
            self.assertEqual(admit_offline_receipt(inputs, state, binding), result)


if __name__ == "__main__":
    unittest.main()
