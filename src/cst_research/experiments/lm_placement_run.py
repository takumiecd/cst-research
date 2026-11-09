"""Frozen matched-count placement trial, qualification, training and audit."""

import argparse
import copy
import hashlib
import io
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import ffn_dst
from .ffn_model import parameter_digest
from .ffn_radius_fraction_qualification import assert_model_resume, assert_nested
from .ffn_run import sha, token_batch, verify_data
from .ffn_selective import active_ffn_scalars, make_optimizer, projections
from .ffn_selective_run import development
from .ffn_training_plan import TrainingPlan
from .lm_connected import step
from .lm_connected_run import oracle
from .lm_placement import ARMS, EVALUATIONS, MATCHED, OFFSETS, SIGNALS, STEPS, model_for
from .lm_placement_probe import probe


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def qualify(output, data):
    report = {
        "state": "running",
        "arms": {},
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    shared, site_hashes, dense_hashes = {}, {}, {}
    for arm, (kind, sites) in ARMS.items():
        m = model_for(arm)
        o = make_optimizer(m)
        for name, p in m.named_parameters():
            # Shared ordinary weights; the selected DST maps have corrected sparse variance.
            if ".map." in name:
                continue
            h = hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            assert name not in shared or shared[name] == h, name
            shared[name] = h
        for name, projection in projections(m).items():
            p = projection.map.atoms.p if kind == "cst" and name in sites else None
            if p is not None:
                h = hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                assert site_hashes.get(name, h) == h
                site_hashes[name] = h
            elif kind == "dense":
                dense_hashes[name] = hashlib.sha256(
                    projection.map.weight.detach().cpu().numpy().tobytes()
                ).hexdigest()
            elif name not in sites:
                assert (
                    hashlib.sha256(
                        projection.map.weight.detach().cpu().numpy().tobytes()
                    ).hexdigest()
                    == dense_hashes[name]
                )
        row = {
            "initial_parameters_sha256": parameter_digest(m),
            "active_ffn_scalars": active_ffn_scalars(m),
            "sites": sites,
        }
        if arm == "pair12":
            row["oracles"] = {
                s: oracle(projections(m)[s].map) for s in ("1.up", "1.down")
            }
        batch = [token_batch(data["train"], [0, 128], 128, "cuda")]
        step(m, o, batch, 1)
        stream = io.BytesIO()
        torch.save({"model": m.state_dict(), "optimizer": o.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        other = model_for(arm)
        other.load_state_dict(saved["model"])
        oo = make_optimizer(other)
        oo.load_state_dict(copy.deepcopy(saved["optimizer"]))
        # Includes retained tensors, feature spectra and backward hooks; it must be inert.
        probe(m, *token_batch(data["train"], [0], 128, "cuda"), retain=True)
        step(m, o, batch, 2)
        step(other, oo, batch, 2)
        assert_model_resume(m, other)
        assert_nested(o.state_dict(), oo.state_dict())
        row["serialized_resume_and_probe_next_update_exact"] = True
        report["arms"][arm] = row
        write(output / "qualification.json", report)
        print("QUALIFIED", arm, flush=True)
        del m, o, other, oo, saved
        torch.cuda.empty_cache()
    counts = {
        report["arms"][arm]["active_ffn_scalars"] for arm in (*MATCHED, "dst-pair12")
    }
    assert len(counts) == 1 and len(site_hashes) == 8
    report.update(
        state="passed",
        matched_active_ffn_scalars=counts.pop(),
        cst_initial_site_sha256=site_hashes,
        dense_initial_site_sha256=dense_hashes,
    )
    write(output / "qualification.json", report)


def signals(m, output, data, clock):
    records = []
    for j, offset in enumerate(OFFSETS):
        row, tensors = probe(
            m, *token_batch(data["train"], [offset], 128, "cuda"), retain=(j == 0)
        )
        records.append({"offset": offset, **row})
        if j == 0:
            torch.save(tensors, output / f"signals-{clock}.pt")
    write(
        output / f"signals-{clock}.json",
        {
            "step": clock,
            "records": records,
            "parameters_buffers_rng_unchanged": True,
            "scope": "four fixed TRAIN contexts; unscaled unclipped single-context CE; retained features at offset0",
        },
    )


def train(arm, output, data, manifest_sha, qualification):
    m = model_for(arm)
    o = make_optimizer(m)
    assert (
        parameter_digest(m) == qualification["arms"][arm]["initial_parameters_sha256"]
    )
    plan = TrainingPlan(5000000, 20000000, 20000000)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    result = {
        "state": "running",
        "arm": arm,
        "config": asdict(m.config),
        "initial_parameters_sha256": parameter_digest(m),
        "data_manifest_sha256": manifest_sha,
        "data_order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "steps": STEPS,
        "schedule_steps": plan.schedule_steps,
        "effective_tokens_per_step": 1024,
        "seed": 17,
        "sites": ARMS[arm][1],
        "active_ffn_scalars": active_ffn_scalars(m),
        "timing_seconds": [],
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "completed_steps": 0,
        "checkpoints": {},
        "topology_steps": [],
        "actual_adam_updates": 0,
    }
    started = time.monotonic()
    with (output / "curves.jsonl").open("x") as curve:

        def log(row):
            curve.write(json.dumps(row) + "\n")
            curve.flush()
            print(json.dumps(row), flush=True)
            write(output / "summary.json", result)

        log(
            {
                "step": 0,
                "seen_tokens": 0,
                "development": development(m, data["development"]),
            }
        )
        signals(m, output, data, 0)
        for clock in range(1, STEPS + 1):
            if time.monotonic() - started > 2400:
                raise TimeoutError("40min per-arm bound")
            batches = []
            for micro in range(4):
                start = ((clock - 1) * 4 + micro) * 2
                batches.append(
                    token_batch(
                        data["train"], order[start : start + 2] * 128, 128, "cuda"
                    )
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            loss, event = step(m, o, batches, clock)
            torch.cuda.synchronize()
            duration = time.monotonic() - tick
            assert math.isfinite(loss)
            if clock > 10:
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
                    "group_lrs": [g["lr"] for g in o.param_groups],
                }
                if clock in EVALUATIONS:
                    row["development"] = development(m, data["development"])
                log(row)
            if clock in SIGNALS:
                # Outside timing and optimizer update; qualification proves hook invariance.
                signals(m, output, data, clock)
                path = output / f"checkpoint-{clock}.pt"
                torch.save(
                    {
                        "model": m.state_dict(),
                        "optimizer": o.state_dict(),
                        "step": clock,
                        "arm": arm,
                    },
                    path,
                )
                result["checkpoints"][path.name] = sha(path)
                write(output / "summary.json", result)
        expected = plan.adam_updates(STEPS) if m.kind == "dst" else STEPS
        assert result["actual_adam_updates"] == expected
        for state in o.state.values():
            assert int(state["step"]) == expected
            for key in ("exp_avg", "exp_avg_sq"):
                assert (
                    state[key].dtype == torch.float32
                    and torch.isfinite(state[key]).all()
                )
        result["dst_invariants"] = ffn_dst.invariants(m, o)
        if arm == "pair12":
            result["final_oracles"] = {
                s: oracle(projections(m)[s].map) for s in ("1.up", "1.down")
            }
        result.update(
            state="completed",
            completed_tokens=STEPS * 1024,
            final_parameters_sha256=parameter_digest(m),
            elapsed_seconds=time.monotonic() - started,
        )
        write(output / "summary.json", result)


def audit(root, output, data):
    report = {"state": "running", "arms": {}}
    plan = TrainingPlan(5000000, 20000000, 20000000)
    for arm in ARMS:
        s = json.loads((root / arm / "summary.json").read_text())
        assert s["state"] == "completed" and s["completed_steps"] == STEPS
        records = []
        for name, h in s["checkpoints"].items():
            path = root / arm / name
            assert sha(path) == h
            saved = torch.load(path, map_location="cpu", weights_only=False)
            m = model_for(arm)
            m.load_state_dict(saved["model"])
            expected = (
                plan.adam_updates(saved["step"]) if m.kind == "dst" else saved["step"]
            )
            for state in saved["optimizer"]["state"].values():
                assert int(state["step"]) == expected
                assert all(
                    state[k].dtype == torch.float32 and torch.isfinite(state[k]).all()
                    for k in ("exp_avg", "exp_avg_sq")
                )
            if saved["step"] == STEPS:
                assert parameter_digest(m) == s["final_parameters_sha256"]
                expected_ce = next(
                    json.loads(v)["development"]["ce"]
                    for v in (root / arm / "curves.jsonl").read_text().splitlines()
                    if json.loads(v)["step"] == STEPS
                )
                actual = development(m, data["development"])["ce"]
                assert abs(actual - expected_ce) <= 1e-6
            records.append(
                {
                    "step": saved["step"],
                    "checkpoint_sha256": h,
                    "actual_adam_updates": expected,
                }
            )
            del m, saved
            torch.cuda.empty_cache()
        report["arms"][arm] = records
        write(output / "audit.json", report)
        print("AUDITED", arm, flush=True)
    report["state"] = "passed"
    write(output / "audit.json", report)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("qualify", "train", "audit"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--data-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--run", type=Path)
    a = p.parse_args()
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha(a.data / "manifest.json") == a.data_manifest_sha256
    _, data = verify_data(a.data)
    a.output.mkdir(parents=True, exist_ok=False)
    if a.mode == "qualify":
        qualify(a.output, data)
    elif a.mode == "audit":
        audit(a.run, a.output, data)
    else:
        q = json.loads(a.qualification.read_text())
        assert q["state"] == "passed" and q["torch"] == torch.__version__
        train(a.arm, a.output, data, a.data_manifest_sha256, q)


if __name__ == "__main__":
    main()
