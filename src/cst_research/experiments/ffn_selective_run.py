"""Fixed FFN site-count qualification/training/audit; no automatic HPO."""

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
from torchcst import CSTLinear, kernels
from torchcst._backends.torch.parameterizations.radius_envelope import from_state

from . import ffn_dst
from .ffn_contract import learning_rate
from .ffn_dst_run import config
from .ffn_dst_run import oracle as dst_oracle
from .ffn_model import configure_native, parameter_digest
from .ffn_radius_fraction import snapshot
from .ffn_radius_fraction_qualification import assert_model_resume, assert_nested
from .ffn_run import sha, token_batch, verify_data
from .ffn_selective import (
    SITE_SETS,
    active_ffn_scalars,
    after_update,
    before_update,
    make_model,
    make_optimizer,
    mechanism_probe,
    projections,
)
from .ffn_training_plan import TrainingPlan

ARMS = {"dense": ("dense", ())} | {
    f"{kind}-{label}": (kind, sites)
    for label, sites in SITE_SETS.items()
    for kind in ("cst", "dst")
}
PLAN = TrainingPlan(5000000, 20000000, 20000000)
SOURCES = (
    "ffn_selective.py",
    "ffn_selective_run.py",
    "ffn_model.py",
    "ffn_contract.py",
    "ffn_dst.py",
    "ffn_dst_run.py",
    "ffn_radius_fraction.py",
    "ffn_training_plan.py",
    "ffn_run.py",
)


def identities():
    return {n: sha(Path(__file__).parent / n) for n in SOURCES}


def model_for(arm):
    kind, sites = ARMS[arm]
    model = make_model(config(), kind, sites).cuda()
    if kind == "cst":
        configure_native(model)
    return model


@torch.no_grad()
def development(model, tokens):
    model.eval()
    total = 0.0
    count = 0
    correct = 0
    for j in range(0, (len(tokens) - 1) // 128, 2):
        x, y = token_batch(tokens, [j * 128, (j + 1) * 128], 128, "cuda")
        logits, loss = model(x, y)
        n = y.numel()
        total += float(loss) * n
        count += n
        correct += int((logits.argmax(-1) == y).sum())
    assert count == 8192
    ce = total / count
    return {
        "ce": ce,
        "perplexity": math.exp(ce),
        "tokens": count,
        "correct_next_tokens": correct,
        "next_token_top1_accuracy": correct / count,
    }


def cst_oracle(layer):
    cpu = copy.deepcopy(layer).cpu()
    op = cpu.execution_declaration()
    ref = CSTLinear(
        *op.charts,
        atoms=1,
        dtype=torch.float64,
        backend="factored",
        kernel=kernels.product(
            *op.kernel.profiles,
            coordinates=kernels.physical(radius_init=(0.5, 50.0), radius_floor=0.5),
        ),
    )
    p = cpu.atoms.p
    vi, uo = ref.operator.factors(from_state(cpu.kernel, p).double())
    gen = torch.Generator().manual_seed(657)
    cx = (
        torch.randn(7, layer.in_features * 2, generator=gen)[:, ::2]
        .double()
        .requires_grad_()
    )
    dy = torch.randn(7, layer.out_features, generator=gen).double()
    cy = (cx @ vi) @ uo.T
    cg = torch.autograd.grad(cy, (cx, p), dy)
    gx = cx.detach().float().cuda().requires_grad_()
    gy = layer(gx)
    gg = torch.autograd.grad(gy, (gx, layer.atoms.p), dy.float().cuda())
    errors = []
    for a, b in zip((gy, *gg), (cy, *cg)):
        a = a.detach().cpu().double()
        b = b.detach().double()
        torch.testing.assert_close(a, b, atol=1e-4, rtol=4e-3)
        errors.append(float((a - b).abs().max()))
    return {
        "max_abs_y_dx_dp": errors,
        "atol": 1e-4,
        "rtol": 4e-3,
        "rows": 7,
        "reference": "FP64 factored physical kernels with full FP32 radius-envelope map VJP",
    }


def oracles(model):
    result = {}
    for name, p in projections(model).items():
        if isinstance(p.map, CSTLinear):
            result[name] = cst_oracle(p.map)
        elif isinstance(p.map, ffn_dst.MaskedLinear):
            result[name] = dst_oracle(p.map, 657)
    return result


def update(model, opt, x, y, step):
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    if model.kind == "dst":
        ffn_dst.finish_step(model, opt, step, PLAN.topology_steps)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
    assert math.isfinite(float(loss.detach()))


def qualify(args, data):
    result = {
        "state": "running",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "source_sha256": identities(),
        "data_manifest_sha256": args.data_manifest_sha256,
        "training_plan": asdict(PLAN),
        "models": {},
    }
    target = args.output / "qualification.json"

    def save():
        target.write_text(json.dumps(result, indent=2) + "\n")

    save()
    x, y = token_batch(data["train"], [0, 128], 128, "cuda")
    try:
        for arm in ARMS:
            m = model_for(arm)
            opt = make_optimizer(m)
            digest = parameter_digest(m)
            row = {
                "initial_parameter_sha256": digest,
                "active_ffn_scalars": active_ffn_scalars(m),
                "oracles": oracles(m),
            }
            update(m, opt, x, y, 1)
            for k in (2, 250) if m.kind == "dst" else (2,):
                b = io.BytesIO()
                torch.save({"model": m.state_dict(), "optimizer": opt.state_dict()}, b)
                b.seek(0)
                saved = torch.load(b, weights_only=False)
                other = model_for(arm)
                other.load_state_dict(saved["model"])
                oo = make_optimizer(other)
                oo.load_state_dict(saved["optimizer"])
                previous = before_update(m)
                update(m, opt, x, y, k)
                action = after_update(m, previous)
                mechanism_probe(m, x[:1], y[:1])
                assert all(
                    math.isfinite(v["relative_action_delta"]) for v in action.values()
                )
                update(other, oo, x, y, k)
                assert_model_resume(m, other)
                assert_nested(opt.state_dict(), oo.state_dict())
                del other, oo, saved, b
            row["serialized_regular_topology_next_step_exact"] = True
            row["instrumentation_does_not_change_update"] = True
            result["models"][arm] = row
            save()
            print("QUALIFY PASS", arm, flush=True)
            del m, opt
            torch.cuda.empty_cache()
        for key in SITE_SETS.keys() - {"all"}:
            assert (
                result["models"]["cst-" + key]["active_ffn_scalars"]
                == result["models"]["dst-" + key]["active_ffn_scalars"]
            )
        result["state"] = "passed"
        save()
    except BaseException as e:
        result.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


def train(args, data):
    q = json.loads(args.qualification.read_text())
    assert sha(args.qualification) == args.qualification_sha256
    assert (
        q["state"] == "passed"
        and q["source_sha256"] == identities()
        and q["training_plan"] == asdict(PLAN)
    )
    assert (
        q["gpu"] == torch.cuda.get_device_name(0)
        and q["torch"] == torch.__version__
        and q["data_manifest_sha256"] == args.data_manifest_sha256
    )
    m = model_for(args.arm)
    opt = make_optimizer(m)
    digest = parameter_digest(m)
    assert digest == q["models"][args.arm]["initial_parameter_sha256"]
    kind, sites = ARMS[args.arm]
    began = time.monotonic()
    order = np.random.default_rng(905).permutation(20000000 // 128)
    result = {
        "state": "running",
        "arm": args.arm,
        "kind": kind,
        "sites": sites,
        "config": asdict(config()),
        "training_plan": asdict(PLAN),
        "seed": 17,
        "source_sha256": identities(),
        "data_manifest_sha256": args.data_manifest_sha256,
        "qualification_sha256": args.qualification_sha256,
        "initial_parameter_sha256": digest,
        "data_order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "active_ffn_scalars": active_ffn_scalars(m),
        "actual_ffn_parameters": sum(
            p.numel() for b in m.blocks for p in b.ffn.parameters()
        ),
        "completed_updates": 0,
        "actual_adam_updates": 0,
        "completed_tokens": 0,
        "timing_samples_seconds": [],
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "milestone_checkpoint_sha256": {},
        "diagnostics": {},
        "topology_events": [],
    }
    target = args.output / "summary.json"

    def save():
        target.write_text(json.dumps(result, indent=2) + "\n")

    curves = (args.output / "curves.jsonl").open("x")
    diagnostics = (args.output / "mechanisms.jsonl").open("x")

    def log(v):
        curves.write(json.dumps(v) + "\n")
        curves.flush()
        print(json.dumps(v), flush=True)

    px, py = token_batch(data["train"], [0], 128, "cuda")

    def diagnostic(k, actions=None):
        row = {
            "step": k,
            "seen_tokens": k * 1024,
            "probe": mechanism_probe(m, px, py),
            "actual_update_actions": actions,
        }
        if kind == "cst":
            row["geometry"] = snapshot(m)
        diagnostics.write(json.dumps(row) + "\n")
        diagnostics.flush()

    try:
        diagnostic(0)
        log(
            {
                "step": 0,
                "seen_tokens": 0,
                "elapsed_seconds": time.monotonic() - began,
                "development": development(m, data["development"]),
            }
        )
        torch.cuda.empty_cache()
        result["setup_seconds"] = time.monotonic() - began
        save()
        for step in range(1, PLAN.steps + 1):
            if time.monotonic() - began > 3600:
                raise TimeoutError("fixed selective arm wall limit")
            observed = step in PLAN.milestones or step in (10, 100) or step % 200 == 0
            m.train()
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group["lr"] = learning_rate(step, PLAN.schedule_steps) * group.get(
                    "lr_scale", 1.0
                )
            previous = before_update(m) if observed else None
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
            if kind == "dst":
                event = ffn_dst.finish_step(m, opt, step, PLAN.topology_steps)
            else:
                norm = torch.nn.utils.clip_grad_norm_(
                    m.parameters(), 1.0, error_if_nonfinite=True
                )
                opt.step()
                event = {"topology": None, "gradient_norm_before_clip": float(norm)}
            torch.cuda.synchronize()
            duration = time.monotonic() - tick
            if step > 10 and not observed:
                result["timing_samples_seconds"].append(duration)
                result["peak_allocated_bytes"] = max(
                    result["peak_allocated_bytes"], torch.cuda.max_memory_allocated()
                )
                result["peak_reserved_bytes"] = max(
                    result["peak_reserved_bytes"], torch.cuda.max_memory_reserved()
                )
            result.update(completed_updates=step, completed_tokens=step * 1024)
            if event["topology"] is None:
                result["actual_adam_updates"] += 1
            else:
                result["topology_events"].append(dict(step=step, **event))
            value = float(total / 4)
            assert math.isfinite(value)
            if step <= 5 or step % 20 == 0 or step in PLAN.milestones:
                log(
                    {
                        "step": step,
                        "seen_tokens": step * 1024,
                        "elapsed_seconds": time.monotonic() - began,
                        "train_ce": value,
                        "complete_update_seconds": duration,
                        "instrumented_step": observed,
                        "gradient_norm_before_clip": event["gradient_norm_before_clip"],
                        "topology": event["topology"] is not None,
                        "group_lrs": [g["lr"] for g in opt.param_groups],
                        "effective_field_lrs": opt.get_field_lrs()
                        if kind == "cst"
                        else None,
                    }
                )
            if observed:
                actions = after_update(m, previous)
                del previous
                diagnostic(step, actions)
                torch.cuda.empty_cache()
                if step not in (10, 100):
                    result["latest_development"] = development(m, data["development"])
                    log(
                        {
                            "step": step,
                            "seen_tokens": step * 1024,
                            "elapsed_seconds": time.monotonic() - began,
                            "development": result["latest_development"],
                        }
                    )
                    save()
            if step in PLAN.milestones:
                p = args.output / f"checkpoint-{step * 1024}-tokens.pt"
                torch.save(
                    {
                        "model": m.state_dict(),
                        "optimizer": opt.state_dict(),
                        "updates": step,
                        "completed_tokens": step * 1024,
                        "actual_adam_updates": result["actual_adam_updates"],
                        "arm": args.arm,
                        "kind": kind,
                        "sites": sites,
                        "config": asdict(config()),
                        "training_plan": asdict(PLAN),
                        "data_manifest_sha256": args.data_manifest_sha256,
                        "seed": 17,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state(),
                    },
                    p,
                )
                result["milestone_checkpoint_sha256"][p.name] = sha(p)
                save()
        expected = PLAN.adam_updates(PLAN.steps) if kind == "dst" else PLAN.steps
        assert result["actual_adam_updates"] == expected
        result.update(
            state="completed_execution_not_scientific_admission",
            total_elapsed_seconds=time.monotonic() - began,
        )
        save()
    except BaseException as e:
        result.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise
    finally:
        curves.close()
        diagnostics.close()


def audit(args, data):
    s = json.loads((args.run / "summary.json").read_text())
    assert (
        s["state"] == "completed_execution_not_scientific_admission"
        and s["source_sha256"] == identities()
    )
    curves = [
        json.loads(v) for v in (args.run / "curves.jsonl").read_text().splitlines()
    ]
    records = []
    for name, digest in s["milestone_checkpoint_sha256"].items():
        p = args.run / name
        assert sha(p) == digest
        saved = torch.load(p, map_location="cpu", weights_only=False)
        m = model_for(saved["arm"])
        m.load_state_dict(saved["model"])
        o = make_optimizer(m)
        o.load_state_dict(saved["optimizer"])
        step = saved["updates"]
        expected = PLAN.adam_updates(step) if m.kind == "dst" else step
        assert (
            saved["actual_adam_updates"] == expected
            and saved["data_manifest_sha256"] == args.data_manifest_sha256
        )
        for state in o.state.values():
            assert int(state["step"]) == expected
            for key in ("exp_avg", "exp_avg_sq"):
                assert (
                    state[key].dtype == torch.float32
                    and torch.isfinite(state[key]).all()
                )
        if m.kind == "dst":
            ffn_dst.invariants(m, o)
        metric = development(m, data["development"])
        old = next(
            v["development"] for v in curves if v["step"] == step and "development" in v
        )
        assert (
            abs(metric["ce"] - old["ce"]) <= 1e-6
            and metric["correct_next_tokens"] == old["correct_next_tokens"]
        )
        record = {
            "arm": saved["arm"],
            "step": step,
            "actual_adam_updates": expected,
            "checkpoint_sha256": digest,
            "development": metric,
            "ce_absolute_difference": abs(metric["ce"] - old["ce"]),
            "oracles": oracles(m),
        }
        records.append(record)
        print("AUDIT PASS", saved["arm"], step, flush=True)
        del m, o, saved
        torch.cuda.empty_cache()
    assert len(records) == 3
    (args.output / "audit.json").write_text(
        json.dumps(
            {"state": "passed", "models": records, "source_sha256": identities()},
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
    p.add_argument("--arm", choices=tuple(ARMS))
    p.add_argument("--qualification", type=Path)
    p.add_argument("--qualification-sha256")
    p.add_argument("--run", type=Path)
    args = p.parse_args()
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha(args.data / "manifest.json") == args.data_manifest_sha256
    _, data = verify_data(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    {"qualify": qualify, "train": train, "audit": audit}[args.mode](args, data)


if __name__ == "__main__":
    main()
