"""Synthetic receipt fixtures test report scope; none are GPU evidence."""

import copy
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from cst_research.experiments.ffn_contract import ModelConfig, learning_rate
from cst_research.experiments.ffn_report import aggregate, digest


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def request(self, *, confirmation=False, shapes=((256, 4),)):
        seeds = [29, 43, 71] if confirmation else [17]
        request = {
            "cohort": "confirmation" if confirmation else "development",
            "seeds": seeds,
            "shapes": [{"width": n, "expansion": e} for n, e in shapes],
            "recipe": ModelConfig(atom_lr_multiplier=16).recipe(),
            "token_budget": 1000000,
            "consumer_commit": "a" * 40,
            "operator_commit": "b" * 40,
            "data_manifest_sha256": "d" * 64,
            "gpu": "synthetic GPU",
            "torch": "synthetic runtime",
            "runs": [],
        }
        for width, expansion in shapes:
            for i, seed in enumerate(seeds):
                for kind in ("dense", "none", "cst"):
                    name = f"n{width}-e{expansion}-{kind}-s{seed}"
                    folder = self.root / name
                    folder.mkdir()
                    config = ModelConfig(
                        width=width, expansion=expansion, atom_lr_multiplier=16
                    )
                    ce = {"dense": 3.0, "cst": 3.1, "none": 3.3}[kind] + 0.02 * i

                    def metric(value):
                        return {
                            "ce": value,
                            "perplexity": math.exp(value),
                            "tokens": 8192,
                        }

                    checkpoints = [0, 200, 400, 600, 800, 976]
                    points = [{"step": 0, "seen_tokens": 0, "development": metric(9.0)}]
                    for step in range(1, 977):
                        if step <= 5 or step % 20 == 0 or step == 976:
                            lr = learning_rate(step, 20000000 // 1024)
                            points.append(
                                {
                                    "step": step,
                                    "seen_tokens": step * 1024,
                                    "train_ce": ce + 0.3,
                                    "lr": lr,
                                    "optimizer_group_lrs": [lr, 16 * lr]
                                    if kind == "cst"
                                    else [lr],
                                }
                            )
                        if step in checkpoints:
                            points.append(
                                {
                                    "step": step,
                                    "seen_tokens": step * 1024,
                                    "development": metric(ce + 0.2),
                                }
                            )
                    duration = 0.003 if kind == "cst" else 0.001
                    for point in points:
                        point["elapsed_seconds"] = 10 + duration * point["step"]
                    (folder / "curves.jsonl").write_text(
                        "\n".join(json.dumps(p) for p in points)
                    )
                    summary = {
                        "state": "completed_execution_not_scientific_admission",
                        "config": asdict(config),
                        "arguments": {
                            "repeat_fixed_batches": False,
                            "kind": kind,
                            "seed": seed,
                            "tokens": 1000000,
                            "batch": 2,
                            "accumulation": 4,
                            "track_atom_motion": False,
                        },
                        "completed_updates": 976,
                        "scheduled_updates": 976,
                        "completed_tokens": 999424,
                        "scheduled_tokens": 999424,
                        "data_manifest_sha256": "d" * 64,
                        "data_order_horizon_tokens": 20000000,
                        "learning_rate_horizon_updates": 20000000 // 1024,
                        "gpu": "synthetic GPU",
                        "torch": "synthetic runtime",
                        "checkpoint_sha256": digest(folder / "curves.jsonl"),
                        "latest_development": metric(ce + 0.2),
                        "confirmation": {
                            "ce": ce,
                            "perplexity": math.exp(ce),
                            "tokens": 32768,
                        },
                        "timing_samples_seconds": [0.003 if kind == "cst" else 0.001]
                        * 966,
                        "setup_seconds": 10.0,
                        "total_elapsed_seconds": 12 + duration * 976,
                        "peak_allocated_bytes": 200 * 1048576,
                        "peak_reserved_bytes": 256 * 1048576,
                        "actual_ffn_parameters": 0
                        if kind == "none"
                        else config.ffn_parameter_budget()[kind + "_parameters"],
                        "generations": [
                            {
                                "prompt": p,
                                "text": p + " Synthetic sample.",
                                "sampling_seed": 7000 + j,
                                "temperature": 0.8,
                                "top_k": 40,
                                "ids": [1, 2],
                            }
                            for j, p in enumerate(
                                (
                                    "Once upon a time, there was a little",
                                    "Lily found a red ball in the garden.",
                                )
                            )
                        ],
                    }
                    self.write(folder / "summary.json", summary)
                    receipt_path = self.root / (name + "-receipt.json")
                    receipt = {
                        "state": "finished_execution_pending_independent_audit",
                        "consumer": "a" * 40,
                        "operator": "b" * 40,
                        "commands": [
                            {
                                "label": kind,
                                "exit_code": 0,
                                "argv": ["--output", name],
                                "summary_sha256": digest(folder / "summary.json"),
                                "curves_sha256": digest(folder / "curves.jsonl"),
                                "checkpoint_sha256": summary["checkpoint_sha256"],
                            }
                        ],
                    }
                    self.write(receipt_path, receipt)
                    audit_path = self.root / (name + "-audit.json")
                    audit = {
                        "state": "passed",
                        "consumer": "a" * 40,
                        "operator": "b" * 40,
                        "receipt_sha256": digest(receipt_path),
                        "arms": {
                            kind: {
                                "checkpoint_sha256": summary["checkpoint_sha256"],
                                "updates": 976,
                                "finite_optimizer_state": True,
                                "recomputed_ce": {
                                    "development": ce + 0.2,
                                    "confirmation": ce,
                                },
                                "trained_all_atom_vjp": {
                                    f"projection{k}": {"y": {}, "dx": {}, "dp": {}}
                                    for k in range(8 if kind == "cst" else 0)
                                },
                                "disposable_task_resume": {
                                    "optimizer_moments_counters_passed": True,
                                    **{
                                        phase: {
                                            "atom_sites_with_nonzero_gradient_and_change": 8
                                            if kind == "cst"
                                            else 0
                                        }
                                        for phase in ("first", "next", "resumed_next")
                                    },
                                },
                            }
                        },
                    }
                    self.write(audit_path, audit)
                    request["runs"].append(
                        {
                            "width": width,
                            "expansion": expansion,
                            "kind": kind,
                            "seed": seed,
                            "label": kind,
                            "artifacts": name,
                            "receipt": receipt_path.name,
                            "receipt_sha256": digest(receipt_path),
                            "audit": audit_path.name,
                            "audit_sha256": digest(audit_path),
                        }
                    )
        return request

    def rehash(self, run):
        receipt_path = self.root / run["receipt"]
        receipt = json.loads(receipt_path.read_text())
        receipt["commands"][0]["summary_sha256"] = digest(
            self.root / run["artifacts"] / "summary.json"
        )
        receipt["commands"][0]["curves_sha256"] = digest(
            self.root / run["artifacts"] / "curves.jsonl"
        )
        self.write(receipt_path, receipt)
        run["receipt_sha256"] = digest(receipt_path)
        audit_path = self.root / run["audit"]
        audit = json.loads(audit_path.read_text())
        audit["receipt_sha256"] = run["receipt_sha256"]
        self.write(audit_path, audit)
        run["audit_sha256"] = digest(audit_path)

    def test_full_confirmation_pairs_seeds_and_preserves_width_parameter_counts(self):
        result = aggregate(
            self.root,
            self.request(confirmation=True, shapes=((128, 4), (256, 4), (512, 4))),
        )
        self.assertEqual(len(result["runs"]), 27)
        cst = [g for g in result["groups"] if g["kind"] == "cst"]
        self.assertEqual([g["ffn_parameters"] for g in cst], [68096, 136192, 272384])
        for group in result["groups"]:
            self.assertAlmostEqual(
                group["metrics"]["confirmation_ce"]["sample_sd"], 0.02
            )
        for pair in result["paired_differences"]:
            self.assertEqual(sorted(pair["per_seed"]), ["29", "43", "71"])
            self.assertAlmostEqual(
                pair["mean"], 0.1 if pair["control"] == "dense" else -0.2
            )
        self.assertEqual(result["state"], "aggregated_receipts_not_gpu_revalidation")

    def test_partial_duplicate_or_mislabeled_seed_family_is_rejected(self):
        request = self.request()
        for runs in (request["runs"][:-1], request["runs"] + [request["runs"][0]]):
            with self.assertRaisesRegex(ValueError, "comparison arm"):
                aggregate(self.root, {**request, "runs": runs})
        with self.assertRaisesRegex(ValueError, "seed family"):
            aggregate(self.root, {**request, "cohort": "confirmation"})

    def test_integrity_failures_precede_aggregation(self):
        request = self.request()
        run = request["runs"][0]
        (self.root / run["artifacts"] / "summary.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed summary"):
            aggregate(self.root, request)

    def test_rehashed_protocol_drift_and_smoke_still_fail(self):
        request = self.request()
        run = request["runs"][0]
        path = self.root / run["artifacts"] / "summary.json"
        original = json.loads(path.read_text())
        for field, value, error in [
            ("completed_tokens", 998400, "budget"),
            ("data_order_horizon_tokens", 1000000, "horizon"),
            ("data_manifest_sha256", "e" * 64, "data"),
        ]:
            changed = copy.deepcopy(original)
            changed[field] = value
            self.write(path, changed)
            self.rehash(run)
            with self.assertRaisesRegex(ValueError, error):
                aggregate(self.root, request)
        for argument, error in [
            ("repeat_fixed_batches", "smoke"),
            ("track_atom_motion", "instrumented"),
        ]:
            changed = copy.deepcopy(original)
            changed["arguments"][argument] = True
            self.write(path, changed)
            self.rehash(run)
            with self.assertRaisesRegex(ValueError, error):
                aggregate(self.root, request)

    def test_rehashed_missing_curve_and_wrong_atom_group_rate_fail(self):
        request = self.request()
        run = request["runs"][-1]
        path = self.root / run["artifacts"] / "curves.jsonl"
        points = [json.loads(line) for line in path.read_text().splitlines()]
        changed = [p for p in points if not (p["step"] == 200 and "development" in p)]
        path.write_text("\n".join(json.dumps(p) for p in changed))
        self.rehash(run)
        with self.assertRaisesRegex(ValueError, "validation checkpoint"):
            aggregate(self.root, request)
        changed = copy.deepcopy(points)
        next(p for p in changed if "train_ce" in p)["optimizer_group_lrs"] = [1e-4]
        path.write_text("\n".join(json.dumps(p) for p in changed))
        self.rehash(run)
        with self.assertRaisesRegex(ValueError, "group rates"):
            aggregate(self.root, request)

    def test_incomplete_trained_vjp_audit_is_not_accepted_as_full_model(self):
        request = self.request()
        run = request["runs"][-1]
        path = self.root / run["audit"]
        audit = json.loads(path.read_text())
        del audit["arms"]["cst"]["trained_all_atom_vjp"]["projection7"]
        self.write(path, audit)
        run["audit_sha256"] = digest(path)
        with self.assertRaisesRegex(ValueError, "trained VJP"):
            aggregate(self.root, request)

    def test_rehashed_elapsed_clock_corruption_is_rejected(self):
        request = self.request()
        run = request["runs"][-1]
        path = self.root / run["artifacts"] / "curves.jsonl"
        original = path.read_text()
        points = [json.loads(line) for line in original.splitlines()]
        points[-1]["elapsed_seconds"] = 0.001
        path.write_text("\n".join(json.dumps(p) for p in points))
        self.rehash(run)
        with self.assertRaisesRegex(ValueError, "clock moved backwards"):
            aggregate(self.root, request)
        path.write_text(original)
        summary_path = self.root / run["artifacts"] / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["total_elapsed_seconds"] = 1.0
        self.write(summary_path, summary)
        self.rehash(run)
        with self.assertRaisesRegex(ValueError, "elapsed budget"):
            aggregate(self.root, request)

    def test_evidence_paths_cannot_escape_root(self):
        request = self.request()
        request["runs"][0]["receipt"] = "../outside.json"
        with self.assertRaisesRegex(ValueError, "unsafe evidence path"):
            aggregate(self.root, request)
