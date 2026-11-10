"""Qualified short/long width comparisons with source-bound resumable curves."""

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import ffn_dst
from .ffn_model import parameter_digest
from .ffn_radius_fraction_qualification import assert_model_resume, assert_nested
from .ffn_run import sha, token_batch, verify_data
from .ffn_selective import active_ffn_scalars, projections
from .ffn_selective_run import development
from .ffn_training_plan import TrainingPlan
from .lm_connected import step
from .lm_geometry_trial import geometry
from .lm_geometry_trial_run import full_oracle, signals, write
from .lm_m8_capacity_run import ordinary
from .lm_placement_probe import probe
from .lm_width import active_slots, model_for, optimizer_for
from .lm_width_contract import (
    ARMS,
    CHECKPOINTS,
    EVALUATIONS,
    LONG_STEPS,
    SHORT_STEPS,
    WIDTHS,
    choose,
)


def qualification(output, data, width, manifest):
    report = {
        "state": "running",
        "width": width,
        "data_manifest_sha256": manifest,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "arms": {},
    }
    shared, initial = None, {}
    for arm, (kind, _) in ARMS.items():
        model = model_for(arm, width)
        digest = parameter_digest(model)
        ordinary_hash = ordinary(model)
        if shared is None:
            shared = ordinary_hash
        assert ordinary_hash == shared
        if kind in initial:
            assert initial[kind] == digest
        initial[kind] = digest
        row = {
            "initial_parameters_sha256": digest,
            "config": asdict(model.config),
            "active_slots": active_slots(model),
            "active_ffn_scalars": active_ffn_scalars(model),
            "initialization": model.initialization_reports(),
        }
        report["arms"][arm] = row
        write(output / "qualification.json", report)
        if kind == "cst":
            row["oracles"] = {
                site: full_oracle(projections(model)[site].map)
                for site in ("1.up", "1.down")
            }
            write(output / "qualification.json", report)
            assert all(v["passed"] for v in row["oracles"].values())
        opt = optimizer_for(model)
        batch = [token_batch(data["train"], [0, 128], 128, "cuda")]
        step(model, opt, batch, 1)
        path = output / "resume-fixture.pt"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, path)
        saved = torch.load(path, weights_only=False, map_location="cpu")
        other = model_for(arm, width)
        other.load_state_dict(saved["model"])
        oo = optimizer_for(other)
        oo.load_state_dict(copy.deepcopy(saved["optimizer"]))
        probe(model, *token_batch(data["train"], [0], 128, "cuda"), retain=True)
        step(model, opt, batch, 2)
        step(other, oo, batch, 2)
        assert_model_resume(model, other)
        assert_nested(opt.state_dict(), oo.state_dict())
        row["serialized_resume_and_probe_next_update_exact"] = True
        write(output / "qualification.json", report)
        print("QUALIFIED", width, arm, flush=True)
        del model, opt, other, oo, saved
        path.unlink()  # Temporary round-trip fixture only; no training evidence.
        torch.cuda.empty_cache()
    report.update(
        state="passed",
        ordinary_initial_parameters_equal=True,
        LR_candidate_initial_parameters_equal=True,
    )
    write(output / "qualification.json", report)


def train(arm, width, output, data, manifest, qpath, steps, resume):
    q = json.loads(qpath.read_text())
    assert (
        q["state"] == "passed"
        and q["width"] == width
        and q["data_manifest_sha256"] == manifest
    )
    assert q["torch"] == torch.__version__ and q["gpu"] == torch.cuda.get_device_name()
    model = model_for(arm, width)
    opt = optimizer_for(model)
    digest = parameter_digest(model)
    assert digest == q["arms"][arm]["initial_parameters_sha256"]
    plan = TrainingPlan(steps * 1024, 20000000, 20000000)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    order_hash = hashlib.sha256(order.tobytes()).hexdigest()
    initial_clock = 0
    result = {
        "state": "running",
        "arm": arm,
        "width": width,
        "config": asdict(model.config),
        "initial_parameters_sha256": digest,
        "data_manifest_sha256": manifest,
        "data_order_sha256": order_hash,
        "qualification_sha256": sha(qpath),
        "seed": 17,
        "schedule_steps": plan.schedule_steps,
        "effective_tokens_per_step": 1024,
        "target_steps": steps,
        "completed_steps": 0,
        "actual_adam_updates": 0,
        "active_ffn_scalars": active_ffn_scalars(model),
        "active_slots": active_slots(model),
        "timing_seconds": [],
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "checkpoints": {},
        "topology_steps": [],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    if resume:
        old = json.loads((resume / "summary.json").read_text())
        assert (
            old["state"] == "completed"
            and old["completed_steps"] == SHORT_STEPS
            and steps == LONG_STEPS
        )
        for key in (
            "arm",
            "width",
            "config",
            "initial_parameters_sha256",
            "data_manifest_sha256",
            "data_order_sha256",
            "qualification_sha256",
            "schedule_steps",
        ):
            assert old[key] == result[key], key
        path = resume / f"checkpoint-{SHORT_STEPS}.pt"
        assert sha(path) == old["checkpoints"][path.name]
        saved = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        assert parameter_digest(model) == old["final_parameters_sha256"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        initial_clock = SHORT_STEPS
        result.update(
            completed_steps=initial_clock,
            actual_adam_updates=old["actual_adam_updates"],
            topology_steps=list(old["topology_steps"]),
            timing_seconds=list(old["timing_seconds"]),
            peak_allocated_bytes=old["peak_allocated_bytes"],
            peak_reserved_bytes=old["peak_reserved_bytes"],
            resume={
                "checkpoint_sha256": sha(path),
                "summary_sha256": sha(resume / "summary.json"),
                "curves_sha256": sha(resume / "curves.jsonl"),
                "step": initial_clock,
            },
        )
        shutil.copyfile(resume / "curves.jsonl", output / "curves.jsonl")
        del saved
    started = time.monotonic()
    with (output / "curves.jsonl").open("a" if resume else "x") as stream:

        def log(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)
            write(output / "summary.json", result)

        if not resume:
            log(
                {
                    "step": 0,
                    "seen_tokens": 0,
                    "development": development(model, data["development"]),
                }
            )
            signals(model, output, data, 0)
            write(output / "geometry-0.json", geometry(model, opt))
        torch.cuda.empty_cache()
        for clock in range(initial_clock + 1, steps + 1):
            if time.monotonic() - started > 16 * 3600:
                raise TimeoutError("16-hour per-arm bound; checkpoints preserved")
            batches = [
                token_batch(
                    data["train"],
                    order[
                        ((clock - 1) * 4 + micro) * 2 : ((clock - 1) * 4 + micro) * 2
                        + 2
                    ]
                    * 128,
                    128,
                    "cuda",
                )
                for micro in range(4)
            ]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            loss, event = step(model, opt, batches, clock)
            torch.cuda.synchronize()
            duration = time.monotonic() - began
            assert math.isfinite(loss)
            if clock > initial_clock + 10:
                result["timing_seconds"].append(duration)
                result["peak_allocated_bytes"] = max(
                    result["peak_allocated_bytes"], torch.cuda.max_memory_allocated()
                )
                result["peak_reserved_bytes"] = max(
                    result["peak_reserved_bytes"], torch.cuda.max_memory_reserved()
                )
            result["completed_steps"] = clock
            if event["topology"] is None:
                result["actual_adam_updates"] += 1
            else:
                result["topology_steps"].append(clock)
            if clock <= 5 or clock % 20 == 0 or clock in EVALUATIONS:
                row = {
                    "step": clock,
                    "seen_tokens": clock * 1024,
                    "train_ce": loss,
                    "complete_step_seconds": duration,
                    "gradient_norm_before_clip": event["gradient_norm_before_clip"],
                    "topology": event["topology"],
                    "group_lrs": [g["lr"] for g in opt.param_groups],
                }
                if clock in EVALUATIONS:
                    row["development"] = development(model, data["development"])
                log(row)
            if clock in CHECKPOINTS:
                signals(model, output, data, clock)
                write(output / f"geometry-{clock}.json", geometry(model, opt))
                path = output / f"checkpoint-{clock}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": clock,
                        "arm": arm,
                        "config": asdict(model.config),
                        "data_manifest_sha256": manifest,
                        "data_order_sha256": order_hash,
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(),
                    },
                    path,
                )
                result["checkpoints"][path.name] = sha(path)
                write(output / "summary.json", result)
        expected = plan.adam_updates(steps) if model.kind == "dst" else steps
        assert result["actual_adam_updates"] == expected
        for state in opt.state.values():
            assert int(state["step"]) == expected
            for key in ("exp_avg", "exp_avg_sq"):
                assert (
                    state[key].dtype == torch.float32
                    and torch.isfinite(state[key]).all()
                )
        result["dst_invariants"] = ffn_dst.invariants(model, opt)
        result["active_slots"] = active_slots(model)
        result.update(
            state="completed",
            completed_tokens=steps * 1024,
            final_parameters_sha256=parameter_digest(model),
            elapsed_seconds=time.monotonic() - started,
        )
        write(output / "summary.json", result)


def select(root, output, width):
    candidates = {}
    for arm in ARMS:
        summary = json.loads((root / arm / "summary.json").read_text())
        assert (
            summary["state"] == "completed"
            and summary["completed_steps"] == SHORT_STEPS
        )
        assert summary["width"] == width
        candidates[arm] = next(
            json.loads(s)["development"]
            for s in (root / arm / "curves.jsonl").read_text().splitlines()
            if json.loads(s)["step"] == SHORT_STEPS
        )
    write(
        output / "selection.json",
        {
            "state": "passed",
            "width": width,
            "criterion": "minimum short-run DEV CE, tie smaller LR; extension winners fixed at short stage",
            "confirmation_evaluated": False,
            "candidates": candidates,
            "selected": [
                choose(candidates, "dense"),
                "cst-c64",
                choose(candidates, "dst"),
            ],
        },
    )


def audit(root, output, data, width, arm):
    summary = json.loads((root / "summary.json").read_text())
    assert (
        summary["state"] == "completed"
        and summary["arm"] == arm
        and summary["width"] == width
    )
    result = {"state": "running", "checkpoints": [], "oracles": {}}
    for name, digest in summary["checkpoints"].items():
        path = root / name
        assert sha(path) == digest
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["config"] == summary["config"]
        assert saved["data_manifest_sha256"] == summary["data_manifest_sha256"]
        assert saved["data_order_sha256"] == summary["data_order_sha256"]
        model = model_for(arm, width)
        model.load_state_dict(saved["model"])
        opt = optimizer_for(model)
        opt.load_state_dict(saved["optimizer"])
        plan = TrainingPlan(summary["target_steps"] * 1024, 20000000, 20000000)
        expected = (
            plan.adam_updates(saved["step"]) if model.kind == "dst" else saved["step"]
        )
        for state in opt.state.values():
            assert int(state["step"]) == expected
            assert all(
                state[k].dtype == torch.float32 and torch.isfinite(state[k]).all()
                for k in ("exp_avg", "exp_avg_sq")
            )
        active_slots(model)
        ffn_dst.invariants(model, opt)
        if saved["step"] == summary["completed_steps"]:
            assert parameter_digest(model) == summary["final_parameters_sha256"]
            expected_dev = next(
                json.loads(s)["development"]
                for s in (root / "curves.jsonl").read_text().splitlines()
                if json.loads(s)["step"] == saved["step"]
            )
            actual = development(model, data["development"])
            assert abs(actual["ce"] - expected_dev["ce"]) <= 1e-6
            assert actual["correct_next_tokens"] == expected_dev["correct_next_tokens"]
            if model.kind == "cst":
                result["oracles"] = {
                    site: full_oracle(projections(model)[site].map)
                    for site in ("1.up", "1.down")
                }
                write(output / "audit.json", result)
                assert all(v["passed"] for v in result["oracles"].values())
        result["checkpoints"].append(
            {
                "step": saved["step"],
                "sha256": digest,
                "actual_adam_updates": expected,
                "active_slots": active_slots(model),
            }
        )
        write(output / "audit.json", result)
        del model, opt, saved
        torch.cuda.empty_cache()
    result["state"] = "passed"
    write(output / "audit.json", result)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("qualify", "train", "select", "audit"))
    p.add_argument("--width", type=int, choices=WIDTHS, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--data-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--qualification", type=Path)
    p.add_argument(
        "--steps", type=int, choices=(SHORT_STEPS, LONG_STEPS), default=SHORT_STEPS
    )
    p.add_argument("--resume", type=Path)
    p.add_argument("--run", type=Path)
    a = p.parse_args()
    assert (
        torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090"
        and torch.cuda.get_device_capability() == (8, 9)
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha(a.data / "manifest.json") == a.data_manifest_sha256
    _, data = verify_data(a.data)
    a.output.mkdir(parents=True, exist_ok=False)
    if a.mode == "qualify":
        qualification(a.output, data, a.width, a.data_manifest_sha256)
    elif a.mode == "train":
        assert (a.steps == LONG_STEPS) == bool(a.resume)
        train(
            a.arm,
            a.width,
            a.output,
            data,
            a.data_manifest_sha256,
            a.qualification,
            a.steps,
            a.resume,
        )
    elif a.mode == "select":
        select(a.run, a.output, a.width)
    else:
        audit(a.run, a.output, data, a.width, a.arm)


if __name__ == "__main__":
    main()
