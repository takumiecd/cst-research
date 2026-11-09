"""Fixed short LM trial; no HPO, teacher reconstruction or default adoption."""

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

from .ffn_model import parameter_digest
from .ffn_radius_fraction_qualification import assert_model_resume, assert_nested
from .ffn_run import sha, token_batch, verify_data
from .ffn_selective import active_ffn_scalars, make_optimizer, projections
from .ffn_selective_run import development
from .lm_connected import (
    ARMS,
    EVALUATIONS,
    OBSERVATIONS,
    SCHEDULE_STEPS,
    SITES,
    STEPS,
    Observation,
    model_for,
    physical_reference,
    step,
)


def oracle(layer):
    """Full actual A, chunked public FP64 factor reference, raw envelope chain."""
    cpu = copy.deepcopy(layer).cpu()
    from torchcst._backends.torch.parameterizations.radius_envelope import from_state

    reference = physical_reference(cpu.execution_declaration())
    gen = torch.Generator().manual_seed(447)
    x = torch.randn(3, layer.in_features, generator=gen)
    dy = torch.randn(3, layer.out_features, generator=gen).double()
    value = torch.zeros(3, layer.out_features, dtype=torch.float64)
    gx = torch.zeros_like(x, dtype=torch.float64)
    gradients = []
    for start in range(0, len(cpu.atoms.p), 128):
        raw = cpu.atoms.p[start : start + 128].detach().clone().requires_grad_()
        xx = x.double().requires_grad_()
        vi, vo = reference.operator.factors(from_state(cpu.kernel, raw).double())
        yy = (xx @ vi.double()) @ vo.double().T
        dx, dp = torch.autograd.grad(yy, (xx, raw), dy)
        value += yy.detach()
        gx += dx
        gradients.append(dp)
    gp = torch.cat(gradients)
    xx = x.cuda().requires_grad_()
    yy = layer(xx)
    dx, dp = torch.autograd.grad(yy, (xx, layer.atoms.p), dy.float().cuda())
    result = []
    for actual, expected in zip((yy, dx, dp), (value, gx, gp)):
        a = actual.detach().cpu().double()
        b = expected.double()
        torch.testing.assert_close(a, b, atol=1e-4, rtol=4e-3)
        result.append(
            {
                "max_abs": float((a - b).abs().max()),
                "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-30)),
            }
        )
    return {
        "Y_dX_raw_dP": result,
        "A": len(cpu.atoms.p),
        "B": 3,
        "atol": 1e-4,
        "rtol": 4e-3,
        "reference": "public CPU FP64 cell factors, raw FP32 envelope chain; independent primitive quadrature is separate",
    }


def qualify(output, data):
    report = {
        "state": "running",
        "arms": {},
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    digests = {}
    shared = []
    for arm in ARMS:
        m = model_for(arm)
        o = make_optimizer(m)
        digests[arm] = parameter_digest(m)
        digest = hashlib.sha256()
        for name, parameter in m.named_parameters():
            if ".map." not in name:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        shared.append(digest.hexdigest())
        initial = {
            "shared_ordinary_parameters_sha256": shared[-1],
            "initial_parameters_sha256": digests[arm],
            "active_ffn_scalars": active_ffn_scalars(m),
        }
        batches = [token_batch(data["train"], [0, 128], 128, "cuda")]
        if m.kind == "cst":
            initial["oracles"] = {s: oracle(projections(m)[s].map) for s in SITES}
        step(m, o, batches, 1)
        saved = {
            "model": copy.deepcopy(m.state_dict()),
            "optimizer": copy.deepcopy(o.state_dict()),
        }
        # Actual serialization round trip, including kernel declaration and optimizer metadata.
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        other = model_for(arm)
        other.load_state_dict(saved["model"])
        oo = make_optimizer(other)
        oo.load_state_dict(copy.deepcopy(saved["optimizer"]))
        observed = Observation(m) if m.kind == "cst" else None
        step(m, o, batches, 2)
        if observed:
            observed.finish(m)
        step(other, oo, batches, 2)
        assert_model_resume(m, other)
        assert_nested(o.state_dict(), oo.state_dict())
        initial["serialized_resume_and_observation_next_update_exact"] = True
        report["arms"][arm] = initial
        (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        del m, o, other, oo, saved
        torch.cuda.empty_cache()
        print("QUALIFIED", arm, flush=True)
    assert len(set(shared)) == 1
    assert digests["legacy-m8"] == digests["connected-m16"]
    assert (
        report["arms"]["legacy-m8"]["active_ffn_scalars"]
        == report["arms"]["dst-all"]["active_ffn_scalars"]
    )
    report["state"] = "passed"
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")


def train(arm, output, data, manifest_sha, qualification):
    m = model_for(arm)
    o = make_optimizer(m)
    initial = parameter_digest(m)
    assert initial == qualification["arms"][arm]["initial_parameters_sha256"]
    order = np.random.default_rng(905).permutation(20000000 // 128)
    result = {
        "state": "running",
        "arm": arm,
        "config": asdict(m.config),
        "initial_parameters_sha256": initial,
        "data_manifest_sha256": manifest_sha,
        "data_order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "seed": 17,
        "target_steps": STEPS,
        "schedule_steps": SCHEDULE_STEPS,
        "effective_tokens_per_step": 1024,
        "active_ffn_scalars": active_ffn_scalars(m),
        "completed_steps": 0,
        "timing_seconds": [],
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "checkpoints": {},
    }
    begin = time.monotonic()
    curve = (output / "curves.jsonl").open("x")

    def save():
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    def log(v):
        curve.write(json.dumps(v) + "\n")
        curve.flush()
        print(json.dumps(v), flush=True)

    log(
        {
            "step": 0,
            "seen_tokens": 0,
            "development": development(m, data["development"]),
        }
    )
    save()
    for clock in range(1, STEPS + 1):
        if time.monotonic() - begin > 2400:
            raise TimeoutError("40 minute fixed per-arm bound")
        batches = []
        for micro in range(4):
            start = ((clock - 1) * 4 + micro) * 2
            batches.append(
                token_batch(data["train"], order[start : start + 2] * 128, 128, "cuda")
            )
        obs = Observation(m) if m.kind == "cst" and clock in OBSERVATIONS else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        loss, event = step(m, o, batches, clock)
        torch.cuda.synchronize()
        duration = time.monotonic() - tick
        assert math.isfinite(loss)
        if obs:
            torch.save(
                {"step": clock, "arm": arm, "records": obs.finish(m)},
                output / f"probe-{clock}.pt",
            )
        elif clock > 10:
            result["timing_seconds"].append(duration)
            result["peak_allocated_bytes"] = max(
                result["peak_allocated_bytes"], torch.cuda.max_memory_allocated()
            )
            result["peak_reserved_bytes"] = max(
                result["peak_reserved_bytes"], torch.cuda.max_memory_reserved()
            )
        result["completed_steps"] = clock
        if clock <= 5 or clock % 20 == 0 or clock in EVALUATIONS or obs:
            row = {
                "step": clock,
                "seen_tokens": clock * 1024,
                "train_ce": loss,
                "complete_step_seconds": duration,
                "instrumented": obs is not None,
                "topology_event": event["topology"] is not None,
                "gradient_norm_before_clip": event["gradient_norm_before_clip"],
                "group_lrs": [g["lr"] for g in o.param_groups],
            }
            if clock in EVALUATIONS:
                row["development"] = development(m, data["development"])
            log(row)
            save()
        if clock in (256, STEPS):
            path = output / f"checkpoint-{clock}.pt"
            torch.save(
                {
                    "model": m.state_dict(),
                    "optimizer": o.state_dict(),
                    "step": clock,
                    "config": asdict(m.config),
                },
                path,
            )
            result["checkpoints"][path.name] = sha(path)
            save()
    result["final_oracles"] = (
        {s: oracle(projections(m)[s].map) for s in SITES} if m.kind == "cst" else {}
    )
    result.update(
        state="completed",
        completed_tokens=STEPS * 1024,
        final_parameters_sha256=parameter_digest(m),
        elapsed_seconds=time.monotonic() - begin,
    )
    save()
    curve.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("qualify", "train"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--data-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--qualification", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    assert sha(a.data / "manifest.json") == a.data_manifest_sha256
    _, data = verify_data(a.data)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090"
    if a.mode == "qualify":
        qualify(a.output, data)
    else:
        q = json.loads(a.qualification.read_text())
        assert (
            q["state"] == "passed"
            and q["gpu"] == torch.cuda.get_device_name(0)
            and q["torch"] == torch.__version__
        )
        train(a.arm, a.output, data, a.data_manifest_sha256, q)


if __name__ == "__main__":
    main()
