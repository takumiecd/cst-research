"""Pure policy, retained-artifact admission and bootstrap boundaries; no GPU."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from cst_research.experiments import higgs_dst_comparison as campaign
from cst_research.experiments import higgs_dst_comparison_driver as driver


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def candidate(lr, loss=0.5, late=False):
    curve = [0.7, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.55, loss]
    if late:
        curve = [0.7, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.559, 0.558]
    return {
        "lr": lr,
        "epochs": 10,
        "best_development_loss": min(curve),
        "best_epoch": curve.index(min(curve)) + 1,
        "development_losses": curve,
    }


def fixture_cell(path, config, lr, epochs, identity, curve=None):
    path.mkdir(parents=True)
    for name in (
        "best-dev.pt",
        "final.pt",
        "initialization.pt",
        "best-dev-predictions.npz",
    ):
        (path / name).write_bytes((name + str(lr)).encode())
    curve = curve or [0.7 - i * 0.01 for i in range(epochs)]
    total = epochs * 25000
    steps = list(range(250, int(0.75 * total), 250)) if config["kind"] == "rigl" else []
    events = [{"step": step} for step in steps]
    metadata = {"dtype": "torch.float32", "shape": [], "sha256": "a" * 64}
    parameters = {"0.weight", "0.bias", "2.weight", "4.weight", "4.bias"}
    state = {
        "/model/" + name: dict(metadata)
        for name in parameters | {"2.mask", "2.row_active_counts"}
    }
    state.update(
        {
            f"/optimizer/state/{index}/{name}": dict(metadata)
            for index in range(5)
            for name in ("step", "exp_avg", "exp_avg_sq")
        }
    )
    rows = []
    for epoch, loss in enumerate(curve, 1):
        updates = epoch * 25000
        count = sum(step <= updates for step in steps)
        rows.append(
            {
                "epoch": epoch,
                "updates": updates,
                "minibatch_step": updates,
                "data_examples_seen": 4 * updates,
                "actual_optimizer_updates": updates - count,
                "topology_event_count": count,
                "development": {
                    "loss": loss,
                    "accuracy": 0.7,
                    "roc_auc": 0.75,
                    "examples": 50000,
                },
                "state": state,
                "gradients": {name: dict(metadata) for name in parameters},
            }
        )
    best_epoch = curve.index(min(curve)) + 1
    best = {
        "epoch": best_epoch,
        "development": rows[best_epoch - 1]["development"],
        "checkpoint_sha256": campaign.file_sha(path / "best-dev.pt"),
        "predictions_sha256": campaign.file_sha(path / "best-dev-predictions.npz"),
    }
    learning = {
        **{key: config[key] for key in ("kind", "width", "divisor")},
        "seed": 17,
        "lr": lr,
        "mode": "graph",
        "cohort": "higgs_dst_reference_v1",
        "training_rows": 100000,
        "development_rows": 50000,
        "batch_size": 4,
        "dataset_sha256": identity["dataset_sha256"],
        "all_layers_trainable": True,
        "TF32": False,
        "total_updates": total,
        "minibatch_step": total,
        "data_examples_seen": total * 4,
        "confirmation_evaluated": False,
        "test_evaluated": False,
        "validation_partition": {
            "development": [0, 50000],
            "confirmation": [50000, 100000],
        },
        "epochs": rows,
        "best_development_checkpoint": best,
        "initialization_sha256": campaign.file_sha(path / "initialization.pt"),
        "final_checkpoint": "final.pt",
        "topology_events": events,
        "actual_optimizer_updates": total - len(events),
        "graph_replays": total - 8,
        "atom_count_matched": config["width"] ** 2 // config["divisor"],
        "active_middle_weights": 4 * (config["width"] ** 2 // config["divisor"]),
        "rewiring": {
            "algorithm": config["kind"],
            "update_frequency": 250,
            "initial_drop_fraction": 0.3,
            "end_fraction": 0.75,
            "end_step_exclusive": int(0.75 * total),
            "optimizer_update_order": "topology event replaces all-layer AdamW",
        },
        "training_seconds": 1.0,
        "adoption_seconds": 1.1,
        **identity["runtime_fingerprint"],
    }
    for name, value in (
        ("learning.json", learning),
        ("best-dev.json", best),
        ("epochs.json", rows),
        ("topology-events.json", events),
    ):
        write(path / name, value)


class PolicyTests(unittest.TestCase):
    def test_selection_ties_and_single_edge_extension(self):
        rows = [candidate(lr) for lr in campaign.LRS]
        self.assertEqual(campaign.select(rows)["lr"], 0.0001)
        self.assertEqual(campaign.extension(0.0001), 0.00003)
        self.assertEqual(campaign.extension(0.003), 0.01)
        self.assertIsNone(campaign.extension(0.00003))
        with self.assertRaises(ValueError):
            campaign.select([rows[0], rows[0]])

    def test_common_horizon_uses_all_eight_and_cannot_shorten_original(self):
        groups = [
            {"configuration": c, "horizon": 10, "selected": candidate(0.001)}
            for c in campaign.configurations(1024)
        ]
        # A best epoch at10 alone is insufficient: require >.001 late CE gain.
        for group in groups:
            group["selected"] = candidate(0.001, late=True)
            group["selected"]["development_losses"][-1] = 0.5595
        self.assertEqual(campaign.comparison_horizon(10, groups, 1024), (10, []))
        self.assertEqual(campaign.comparison_horizon(20, groups, 1024)[0], 20)
        groups[-1]["selected"] = candidate(0.001, late=True)
        self.assertEqual(campaign.comparison_horizon(10, groups, 1024)[0], 20)
        with self.assertRaises(ValueError):
            campaign.comparison_horizon(10, groups[:-1], 1024)

    def test_actual_cell_counters_and_earliest_best_are_admitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {
                "dataset_sha256": "b" * 64,
                "runtime_fingerprint": {
                    "torch": "test",
                    "cuda": "test",
                    "gpu": "NVIDIA L4",
                },
            }
            for kind in campaign.METHODS:
                config = campaign.configuration(1024, kind, 1000)
                path = root / kind
                fixture_cell(
                    path,
                    config,
                    0.001,
                    10,
                    identity,
                    curve=[0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
                )
                result = campaign.validate_cell(path, config, 17, 0.001, 10, identity)
                self.assertEqual(result["best_epoch"], 3)
                self.assertEqual(
                    result["actual_optimizer_updates"],
                    250000 - (749 if kind == "rigl" else 0),
                )
            path = root / "rigl"
            record = campaign.read(path / "learning.json")
            record["actual_optimizer_updates"] = 250000
            write(path / "learning.json", record)
            with self.assertRaises(ValueError):
                campaign.validate_cell(
                    path,
                    campaign.configuration(1024, "rigl", 1000),
                    17,
                    0.001,
                    10,
                    identity,
                )

    def test_tune_resumes_only_verified_cells_and_exports_producer_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "screen.npz"
            data.write_bytes(b"fixed data")
            data_sha = campaign.file_sha(data)
            with patch.object(campaign, "DATA_SHA256", data_sha):
                identity = {
                    "dataset_sha256": data_sha,
                    "external_source_commit": campaign.EXTERNAL_COMMIT,
                    "protocol_sha256": campaign.json_sha(campaign.protocol()),
                    "consumer_commit": "c" * 40,
                    "runtime_fingerprint": {
                        "torch": "test",
                        "cuda": "test",
                        "gpu": "NVIDIA L4",
                    },
                }
                identity_path = root / "identity.json"
                write(identity_path, identity)
                state = root / "state"
                write(
                    state / "qualification.json",
                    {"identity": identity, "status": "passed"},
                )
                calls = []

                def invoke(argv, **kwargs):
                    calls.append(argv)
                    lr = float(argv[argv.index("--lr") + 1])
                    path = Path(argv[argv.index("--output") + 1])
                    fixture_cell(
                        path,
                        campaign.configuration(1024, "static", 1000),
                        lr,
                        10,
                        identity,
                    )

                args = Namespace(
                    stage="tune",
                    width=1024,
                    kind="static",
                    divisor=1000,
                    seed=None,
                    data=data,
                    output=root / "first",
                    campaign_state=state,
                    identity_file=identity_path,
                    external_source=root,
                    original_selection_index=None,
                )
                campaign.run(args, invoke)
                self.assertEqual(
                    len(calls), 5
                )  # lower LR edge once, no recursive edge search
                export = campaign.read(args.output / "hpo-export.json")
                self.assertEqual(export["producer_identity"], identity)
                args.output = root / "second"
                campaign.run(args, invoke)
                self.assertEqual(len(calls), 5)
                cached = next((state / "retained-cells").iterdir())
                (cached / "final.pt").write_bytes(b"tampered")
                args.output = root / "third"
                with self.assertRaises(ValueError):
                    campaign.run(args, invoke)
                self.assertEqual(len(calls), 5)

    def test_original_width_requires_full_identity_and_complete_lr_inventory(self):
        identity = {
            "source_identity": "a" * 64,
            "source_code_sha256": "b" * 64,
            "dataset_sha256": campaign.DATA_SHA256,
            "protocol_sha256": "c" * 64,
        }
        descriptor = {
            "source_commit": campaign.ORIGINAL_COMMIT,
            "dataset_sha256": campaign.DATA_SHA256,
            "width": 1024,
            "expected_identity": identity,
            "source_manifest_sha256": identity["source_identity"],
            "protocol_sha256": identity["protocol_sha256"],
        }
        rows = []
        for identifier in ("dense", *(f"cst-{d}" for d in campaign.DIVISORS)):
            base = [candidate(lr) for lr in (*campaign.LRS, 0.00003)]
            for item in base:
                item.update(
                    best_epoch=3,
                    best_development_loss=0.5,
                    development_losses=[
                        0.7,
                        0.6,
                        0.5,
                        0.51,
                        0.52,
                        0.53,
                        0.54,
                        0.55,
                        0.56,
                        0.57,
                    ],
                )
            selected = campaign.select(base)
            rows.append(
                {
                    "id": identifier,
                    "kind": "dense" if identifier == "dense" else "cst",
                    "width": 1024,
                    "divisor": 100
                    if identifier == "dense"
                    else int(identifier.removeprefix("cst-")),
                    "lr": selected["lr"],
                    "horizon": 10,
                    "selected_development": selected,
                    "base_tuning_candidates": base,
                    "horizon_candidates": base,
                }
            )
        frozen = {
            "schema_version": 1,
            "identity": identity,
            "width": 1024,
            "horizon": 10,
            "configurations": rows,
            "confirmation_seeds": list(campaign.SEEDS),
            "selection_uses_development_only": True,
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        self.assertEqual(campaign.validate_original_width(frozen, descriptor, 1024), 10)
        for change in (
            {"expected_identity": {}},
            {"source_manifest_sha256": "d" * 64},
            {"protocol_sha256": "d" * 64},
        ):
            with self.assertRaises(ValueError):
                campaign.validate_original_width(frozen, descriptor | change, 1024)
        missing = copy.deepcopy(frozen)
        missing["configurations"][0]["base_tuning_candidates"].pop()
        with self.assertRaises(ValueError):
            campaign.validate_original_width(missing, descriptor, 1024)

    def test_qualification_rejects_another_runtime_and_incomplete_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {
                "runtime_fingerprint": {
                    "torch": "test",
                    "cuda": "test",
                    "gpu": "NVIDIA L4",
                }
            }
            result = {
                "status": "passed",
                "oracles": [],
                "comparisons": [],
                "TF32": False,
                "confirmation_evaluated": False,
                "test_evaluated": False,
                **identity["runtime_fingerprint"],
            }
            for i, (width, active) in enumerate(
                [
                    (17, 51),
                    *[
                        (n, 4 * (n * n // d))
                        for n in campaign.WIDTHS
                        for d in campaign.DIVISORS
                    ],
                ]
            ):
                name = f"oracle-{i}.pt"
                (root / name).write_bytes(b"retained oracle")
                result["oracles"].append(
                    {
                        "width": width,
                        "active_count": active,
                        "file": name,
                        "sha256": campaign.file_sha(root / name),
                        "pass": True,
                        "checks": {
                            key: True
                            for key in ("Y", "dX", "masked_dW", "dense_growth_dW")
                        },
                    }
                )
            for width in campaign.WIDTHS:
                for divisor in (1000, 100):
                    for kind in campaign.METHODS:
                        label = f"N{width}-D{divisor}-{kind}"
                        result["comparisons"].append(
                            {
                                "label": label,
                                "pass": True,
                                "checks": {"complete": True},
                                "epoch_equality": [{"all_byte_equal": True}] * 2,
                            }
                        )
                        for mode in ("eager", "graph"):
                            folder = root / (label + "-" + mode)
                            folder.mkdir()
                            for name in (
                                "initialization.pt",
                                "epoch-1.pt",
                                "epoch-2.pt",
                                "trajectory.json",
                            ):
                                (folder / name).write_bytes(b"retained trajectory")
            write(root / "qualification.json", result)
            campaign.validate_qualification(root, identity)
            write(root / "qualification.json", result | {"torch": "other-wheel"})
            with self.assertRaises(ValueError):
                campaign.validate_qualification(root, identity)
            result["oracles"][0] = result["oracles"][1]
            write(root / "qualification.json", result)
            with self.assertRaises(ValueError):
                campaign.validate_qualification(root, identity)

    def test_no_later_stage_can_evaluate_before_controls(self):
        obj = campaign.Campaign.__new__(campaign.Campaign)
        for stage in ("confirm", "assess"):
            obj.args = Namespace(stage=stage)
            with self.assertRaises(campaign.PhaseNotReady):
                obj.later_stage()


class BootstrapTests(unittest.TestCase):
    def test_immutable_copy_never_overwrites_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            cache = root / "cache"
            cache.mkdir()
            (source / "module.py").write_bytes(b"original")
            manifest = {"files": {"module.py": driver.sha(source / "module.py")}}
            target = driver.immutable_consumer(source, cache, manifest)
            (target / "module.py").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                driver.immutable_consumer(source, cache, manifest)
            self.assertEqual((target / "module.py").read_bytes(), b"tampered")

    def test_external_archive_rejects_tamper_and_unsafe_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.tar.gz"
            manifest = root / "manifest.json"

            def build(name):
                with tarfile.open(archive, "w:gz") as bundle:
                    member = tarfile.TarInfo(name)
                    member.size = 4
                    bundle.addfile(member, io.BytesIO(b"code"))
                write(
                    manifest,
                    {
                        "source_commit": driver.EXTERNAL_COMMIT,
                        "archive_sha256": driver.sha(archive),
                        "files": {name: hashlib.sha256(b"code").hexdigest()},
                    },
                )

            build("module.py")
            with patch.object(driver, "EXTERNAL_ARCHIVE_SHA", driver.sha(archive)):
                destination = root / "good"
                destination.mkdir()
                driver.extract_verified_external(archive, manifest, destination)
                archive.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    driver.extract_verified_external(archive, manifest, root / "tamper")
            build("../outside")
            with (
                patch.object(driver, "EXTERNAL_ARCHIVE_SHA", driver.sha(archive)),
                self.assertRaises(ValueError),
            ):
                driver.extract_verified_external(archive, manifest, root / "bad")
            self.assertFalse((root / "outside").exists())

    def test_recovery_receipt_required_before_restore_is_called(self):
        class Bootstrap:
            def __init__(self):
                self.calls = 0

            def restore_state(self, *args):
                self.calls += 1
                return {"restored": True}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bootstrap = Bootstrap()
            capsule = root / "recovery-state.tar.gz"
            capsule.write_bytes(b"capsule")
            with self.assertRaises(ValueError):
                driver.require_recovery(root, bootstrap, root / "state", {})
            receipt = {
                "file": capsule.name,
                "sha256": driver.sha(capsule),
                "bytes": capsule.stat().st_size,
            }
            write(root / "recovery-state-receipt.json", receipt)
            self.assertEqual(
                driver.require_recovery(root, bootstrap, root / "state", {}),
                {"restored": True},
            )
            capsule.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                driver.require_recovery(root, bootstrap, root / "state", {})
            self.assertEqual(bootstrap.calls, 1)


if __name__ == "__main__":
    unittest.main()
