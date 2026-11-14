"""Source-bound TinyStories DST qualification, learning and checkpoint audit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import ffn_dst as dst
from .ffn_contract import ModelConfig, learning_rate
from .ffn_model import LanguageModel, parameter_digest
from .ffn_run import evaluate, probe, sha, token_batch, verify_data
from .ffn_training_plan import TrainingPlan

RATES = (0.0003, 0.001, 0.003, 0.01)
SOURCES = (
    "ffn_dst.py",
    "ffn_dst_run.py",
    "ffn_training_plan.py",
    "ffn_model.py",
    "ffn_contract.py",
    "ffn_run.py",
)


def config():
    return ModelConfig(
        width=512,
        heads=8,
        radius_mode="envelope",
        linear_backend="levels_wide_amplitude",
        amplitude_max=32.0,
        tau=0.003,
        radius_fraction=0.0,
        atom_lr_multiplier=32.0,
        radius_fraction_lr_multiplier=32.0,
    )


def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def oracle(layer, seed):
    g = torch.Generator().manual_seed(seed)
    no, ni = layer.weight.shape
    x = torch.randn(7, ni * 2, generator=g)[:, ::2].cuda().requires_grad_()
    dy = torch.randn(7, no, generator=g).cuda()
    y = layer(x)
    dx, dw = torch.autograd.grad(y, (x, layer.weight), dy)
    cx = x.detach().cpu().double()
    cd = dy.cpu().double()
    cw = layer.weight.detach().cpu().double()
    expected = (cx @ cw.T, cd @ cw, cd.T @ cx)
    errors = []
    for a, b in zip((y, dx, dw), expected):
        torch.testing.assert_close(a.detach().cpu().double(), b, atol=1e-4, rtol=4e-3)
        errors.append(float((a.detach().cpu().double() - b).abs().max()))
    torch.testing.assert_close(
        (dw * layer.mask).cpu().double(),
        expected[2] * layer.mask.cpu(),
        atol=1e-4,
        rtol=4e-3,
    )
    return {
        "max_abs_y_dx_raw_dw": errors,
        "active_dw_checked": True,
        "rows": 7,
        "noncontiguous": True,
        "atol": 1e-4,
        "rtol": 4e-3,
    }


def batch_step(m, o, x, y, k, topology_steps=976):
    o.zero_grad(set_to_none=True)
    _, loss = m(x, y)
    loss.backward()
    event = dst.finish_step(m, o, k, topology_steps)
    assert math.isfinite(float(loss))
    return event


def qualify(args, data):
    plan = TrainingPlan(args.tokens, args.schedule_tokens, args.topology_tokens)
    target = args.output / "qualification.json"
    result = {
        "state": "running",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": {n: sha(Path(__file__).parent / n) for n in SOURCES},
        "data_manifest_sha256": args.data_manifest_sha256,
        "models": [],
        "training_plan": asdict(plan),
    }

    def save():
        target.write_text(json.dumps(result, indent=2) + "\n")

    save()
    x, y = token_batch(data["train"], [0, 128], 128, "cuda")
    try:
        dense = LanguageModel(config(), "dense", 17)
        for lr in (args.lr,) if args.lr is not None else RATES:
            m = dst.make_model(config(), 17)
            for n, p in m.named_parameters():
                if ".ffn." not in n or not n.endswith("map.weight"):
                    exact(p, dict(dense.named_parameters())[n])
            m = m.cuda()
            o = dst.make_optimizer(m, lr)
            row = {
                "sparse_lr": lr,
                "initial_parameter_sha256": parameter_digest(m),
                "initial_masks": dst.invariants(m),
                "oracles": [
                    oracle(m.blocks[0].ffn.up.map, 91),
                    oracle(m.blocks[0].ffn.down.map, 92),
                ],
            }
            batch_step(m, o, x, y, 1, plan.topology_steps)
            branches = sorted(
                {
                    2,
                    *(plan.planned_events[:1]),
                    *(plan.planned_events[-1:]),
                    ((int(0.75 * plan.topology_steps) // 250) + 1) * 250,
                }
            )
            row["resume_branch_steps"] = branches
            for k in branches:
                b = io.BytesIO()
                torch.save({"m": m.state_dict(), "o": o.state_dict()}, b)
                b.seek(0)
                saved = torch.load(b, weights_only=False)
                other = dst.make_model(config(), 17).cuda()
                other.load_state_dict(saved["m"])
                oo = dst.make_optimizer(other, lr)
                oo.load_state_dict(saved["o"])
                event = batch_step(m, o, x, y, k, plan.topology_steps)
                replay = batch_step(other, oo, x, y, k, plan.topology_steps)
                assert (event["topology"] is not None) == dst.is_topology_step(
                    k, plan.topology_steps
                )
                assert (replay["topology"] is not None) == (
                    event["topology"] is not None
                )
                exact(m.state_dict(), other.state_dict())
                exact(o.state_dict(), oo.state_dict())
                dst.invariants(m, o)
                del other, oo, saved, b
            row["serialized_regular_and_topology_next_step_exact"] = True
            row["shared_ordinary_initialization_exact"] = True
            result["models"].append(row)
            save()
            print("QUALIFY PASS", lr, flush=True)
            del m, o
            torch.cuda.empty_cache()
        assert len({r["initial_parameter_sha256"] for r in result["models"]}) == 1
        assert all(
            r["initial_masks"] == result["models"][0]["initial_masks"]
            for r in result["models"]
        )
        result["state"] = "passed"
        save()
    except BaseException as e:
        result.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


def train(args, data):
    plan = TrainingPlan(args.tokens, args.schedule_tokens, args.topology_tokens)
    q = json.loads(args.qualification.read_text())
    assert q["state"] == "passed" and args.lr in RATES
    assert q["training_plan"] == asdict(plan)
    assert any(r["sparse_lr"] == args.lr for r in q["models"])
    assert args.schedule_tokens + 1 <= len(data["train"])
    assert sha(args.qualification) == args.qualification_sha256
    assert q["data_manifest_sha256"] == args.data_manifest_sha256
    assert q["source_sha256"] == {n: sha(Path(__file__).parent / n) for n in SOURCES}
    assert q["gpu"] == torch.cuda.get_device_name(0) and q["torch"] == torch.__version__
    began = time.monotonic()
    m = dst.make_model(config(), 17).cuda()
    o = dst.make_optimizer(m, args.lr)
    digest = parameter_digest(m)
    assert digest == q["models"][0]["initial_parameter_sha256"]
    initial_masks = dst.invariants(m)
    assert initial_masks == q["models"][0]["initial_masks"]
    active = sum(r["active_slots"] for r in initial_masks.values()) + sum(
        p.numel()
        for n, p in m.named_parameters()
        if ".ffn." in n and not n.endswith("map.weight")
    )
    assert active == 272384
    result = {
        "state": "running",
        "config": asdict(config()),
        "kind": "dst",
        "training_plan": asdict(plan),
        "scheduled_updates": plan.steps,
        "scheduled_tokens": plan.steps * 1024,
        "learning_rate_horizon_updates": plan.schedule_steps,
        "sparse_peak_lr": args.lr,
        "initial_parameter_sha256": digest,
        "initial_masks": initial_masks,
        "qualification_sha256": args.qualification_sha256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "actual_ffn_parameters": sum(
            p.numel() for b in m.blocks for p in b.ffn.parameters()
        ),
        "active_ffn_scalar_slots_including_bias": active,
        "diagnostics": {},
        "completed_updates": 0,
        "actual_adam_updates": 0,
        "completed_tokens": 0,
        "timing_samples_seconds": [],
        "regular_timing_seconds": [],
        "topology_events": [],
        "milestone_checkpoint_sha256": {},
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "measurement": "training iterations incl topology events; dense-storage masked accuracy reference, no sparse efficiency claim",
    }
    target = args.output / "summary.json"
    curves = (args.output / "curves.jsonl").open("x")

    def save():
        target.write_text(json.dumps(result, indent=2) + "\n")

    def log(row):
        curves.write(json.dumps(row) + "\n")
        curves.flush()
        print(json.dumps(row), flush=True)

    def checkpoint(step):
        p = args.output / f"checkpoint-{step * 1024}-tokens.pt"
        tmp = args.output / "checkpoint.tmp"
        torch.save(
            {
                "model": m.state_dict(),
                "optimizer": o.state_dict(),
                "updates": step,
                "actual_adam_updates": result["actual_adam_updates"],
                "completed_tokens": step * 1024,
                "config": asdict(config()),
                "sparse_lr": args.lr,
                "data_manifest_sha256": args.data_manifest_sha256,
                "kind": "dst",
                "seed": 17,
                "training_plan": asdict(plan),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
            },
            tmp,
        )
        tmp.replace(p)
        result["milestone_checkpoint_sha256"][p.name] = sha(p)
        if step == plan.steps:
            os.link(p, args.output / "checkpoint.pt")
            result["checkpoint_sha256"] = sha(p)

    px, py = token_batch(data["train"], [0], 128, "cuda")
    try:
        result["diagnostics"]["initial"] = probe(m, px, py)
        log(
            {
                "step": 0,
                "seen_tokens": 0,
                "elapsed_seconds": time.monotonic() - began,
                "development": evaluate(m, data["development"], 2),
            }
        )
        torch.cuda.empty_cache()
        result["setup_seconds"] = time.monotonic() - began
        order = np.random.default_rng(17 + 888).permutation(plan.schedule_tokens // 128)
        result["data_order_sha256"] = hashlib.sha256(order.tobytes()).hexdigest()
        for step in range(1, plan.steps + 1):
            if time.monotonic() - began > args.max_wall_seconds:
                raise TimeoutError(
                    "frozen long-run wall budget reached; evidence preserved"
                )
            m.train()
            o.zero_grad(set_to_none=True)
            for group in o.param_groups:
                group["lr"] = (
                    learning_rate(step, plan.schedule_steps) * group["lr_scale"]
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            total = torch.zeros((), device="cuda")
            for micro in range(4):
                start = ((step - 1) * 4 + micro) * 2
                x, y = token_batch(
                    data["train"], order[start : start + 2] * 128, 128, "cuda"
                )
                _, loss = m(x, y)
                (loss / 4).backward()
                total += loss.detach()
            event = dst.finish_step(m, o, step, plan.topology_steps)
            torch.cuda.synchronize()
            duration = time.monotonic() - tick
            value = float(total / 4)
            assert math.isfinite(value)
            result["completed_updates"] = step
            result["completed_tokens"] = step * 1024
            result["peak_allocated_bytes"] = max(
                result["peak_allocated_bytes"], torch.cuda.max_memory_allocated()
            )
            result["peak_reserved_bytes"] = max(
                result["peak_reserved_bytes"], torch.cuda.max_memory_reserved()
            )
            if event["topology"] is None:
                result["actual_adam_updates"] += 1
                if step > 10:
                    result["regular_timing_seconds"].append(duration)
            else:
                result["topology_events"].append(
                    {"step": step, "seconds": duration, **event}
                )
            if step > 10:
                result["timing_samples_seconds"].append(duration)
            if (
                step <= 5
                or step % 20 == 0
                or step in plan.milestones
                or event["topology"] is not None
            ):
                log(
                    {
                        "step": step,
                        "seen_tokens": step * 1024,
                        "elapsed_seconds": time.monotonic() - began,
                        "train_ce": value,
                        "complete_update_seconds": duration,
                        "actual_adam_updates": result["actual_adam_updates"],
                        "gradient_norm_before_clip": event["gradient_norm_before_clip"],
                        "topology_step": event["topology"] is not None,
                    }
                )
            if step in (10, 100):
                result["diagnostics"][str(step)] = probe(m, px, py)
                torch.cuda.empty_cache()
            if step % 200 == 0 or step in plan.milestones:
                dev = evaluate(m, data["development"], 2)
                result["latest_development"] = dev
                log(
                    {
                        "step": step,
                        "seen_tokens": step * 1024,
                        "elapsed_seconds": time.monotonic() - began,
                        "development": dev,
                    }
                )
                save()
            if step in plan.milestones:
                dst.invariants(m, o)
                checkpoint(step)
                save()
        assert result["actual_adam_updates"] == plan.adam_updates(plan.steps) and [
            r["step"] for r in result["topology_events"]
        ] == list(plan.events)
        result["diagnostics"]["final"] = probe(m, px, py)
        result["final_masks"] = dst.invariants(m, o)
        result.update(
            state="completed_execution_not_scientific_admission",
            total_elapsed_seconds=time.monotonic() - began,
        )
        save()
    except BaseException as e:
        result.update(
            state="failed",
            error=f"{type(e).__name__}: {e}",
            total_elapsed_seconds=time.monotonic() - began,
        )
        save()
        raise
    finally:
        curves.close()


def audit(args, data):
    summary = json.loads((args.run / "summary.json").read_text())
    assert summary["state"] == "completed_execution_not_scientific_admission"
    records = []
    plan = (
        TrainingPlan(**summary["training_plan"])
        if "training_plan" in summary
        else TrainingPlan(
            summary["scheduled_tokens"],
            summary.get("data_order_horizon_tokens", 20000000),
        )
    )
    for step in plan.milestones:
        p = args.run / f"checkpoint-{step * 1024}-tokens.pt"
        assert sha(p) == summary["milestone_checkpoint_sha256"][p.name]
        saved = torch.load(p, map_location="cpu", weights_only=False)
        dense = saved["kind"] == "dense"
        cfg = ModelConfig(**saved["config"])
        m = (
            LanguageModel(cfg, "dense", 17) if dense else dst.make_model(cfg, 17)
        ).cuda()
        m.load_state_dict(saved["model"])
        o = (
            torch.optim.AdamW(
                m.parameters(),
                lr=0.0003,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.0,
                foreach=False,
            )
            if dense
            else dst.make_optimizer(m, saved["sparse_lr"])
        )
        o.load_state_dict(saved["optimizer"])
        expected = step if dense else plan.adam_updates(step)
        assert saved["completed_tokens"] == step * 1024
        if not dense:
            assert saved["training_plan"] == asdict(plan)
        assert saved.get("actual_adam_updates", step) == expected
        for state in o.state.values():
            assert int(state["step"]) == expected
            for key in ("exp_avg", "exp_avg_sq"):
                assert (
                    state[key].dtype == torch.float32
                    and torch.isfinite(state[key]).all()
                )
        curves = [
            json.loads(l) for l in (args.run / "curves.jsonl").read_text().splitlines()
        ]
        ce = next(
            r["development"]["ce"]
            for r in curves
            if r["step"] == step and "development" in r
        )
        got = evaluate(m, data["development"], 2)["ce"]
        assert abs(got - ce) <= 1e-6
        records.append(
            {
                "step": step,
                "actual_adam_updates": expected,
                "checkpoint_sha256": sha(p),
                "expected_ce": ce,
                "reproduced_ce": got,
                "absolute_difference": abs(got - ce),
                "masks": dst.invariants(m, o),
                "oracles": [
                    oracle(m.blocks[0].ffn.up.map, 91),
                    oracle(m.blocks[0].ffn.down.map, 92),
                ]
                if not dense
                else [],
            }
        )
        print("AUDIT PASS", args.run.name, step, flush=True)
        del m, o, saved
        torch.cuda.empty_cache()
    (args.output / "audit.json").write_text(
        json.dumps(
            {
                "state": "passed",
                "models": records,
                "source_sha256": {n: sha(Path(__file__).parent / n) for n in SOURCES},
            },
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("qualify", "train", "audit"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--data-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--qualification-sha256")
    p.add_argument("--lr", type=float, choices=RATES)
    p.add_argument("--run", type=Path)
    p.add_argument("--tokens", type=int, default=999424)
    p.add_argument("--schedule-tokens", type=int, default=20000000)
    p.add_argument("--topology-tokens", type=int)
    p.add_argument("--max-wall-seconds", type=float, default=7200)
    p.add_argument(
        "--expected-gpu",
        choices=("NVIDIA RTX 6000 Ada Generation", "NVIDIA GeForce RTX 4090"),
        default="NVIDIA RTX 6000 Ada Generation",
    )
    args = p.parse_args()
    assert torch.cuda.get_device_name(0) == args.expected_gpu
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha(args.data / "manifest.json") == args.data_manifest_sha256
    _, data = verify_data(args.data)
    args.output.mkdir(exist_ok=False, parents=True)
    {"qualify": qualify, "train": train, "audit": audit}[args.mode](args, data)


if __name__ == "__main__":
    main()
