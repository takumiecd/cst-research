"""Train matched TinyStories projection replacements on the frozen width-study sources."""

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import cst_research
import torchcst
from cst_research.experiments.ffn_model import (
    CSTAllQKVProjection,
    Projection,
    parameter_digest,
    parameter_seed,
)
from cst_research.experiments.ffn_run import evaluate, sha, token_batch, verify_data
from cst_research.experiments.ffn_selective import make_model
from cst_research.experiments.ffn_selective_run import development
from cst_research.experiments.lm_connected import step
from cst_research.experiments.lm_placement_oracle import full_oracle
from cst_research.experiments.lm_width import optimizer_for
from cst_research.experiments.lm_width_contract import (
    EVALUATIONS,
    LONG_STEPS,
    SCHEDULE_STEPS,
    WidthConfig,
)
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
    RADIUS_ENVELOPE_WIDE_AMPLITUDE_WIDTH_LM,
    RadiusEnvelopeWideAmplitudeWidthLMAlgorithm,
)
from torchcst._backends.torch.parameterizations import radius_envelope

WIDTH = 1024
SEED = 17
CAP = 512
TAU_HIGH, TAU_LOW = 0.03, 0.003
ANNEAL_END = 1536
ARMS = ("dense", "down", "o", "qkv")
SITE_COUNTS = {
    "dense": (0, 0),
    "down": (4, 196608),
    "o": (4, 49152),
    "qkv": (12, 65536),
}
EVAL_STEPS = {*EVALUATIONS, 128, 512, 768, ANNEAL_END, 2441, 3600}
SAVE_STEPS = {512, 976, ANNEAL_END, 2441, 3600, LONG_STEPS}
PROTOCOL_SHA = "583a73e2f105ff07fb7d926f618874a874c56d45de81ab8ed3cb706fcb724d88"
DATA_SHA = "5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01"
ORDER_SHA = "927ba5aad87c2f72f89d416c4ee33c7f7d8ec183e3419827258b1b868089a324"
QKV_INITIAL_SHA = "94448e4cf64bfa446b92976c3727ade6d5d6ecab31656d0ce883a8e568f37247"
DENSE_INITIAL_SHA = "4787f408c1f5eee45fe02f37e4a199aebf02a4995f2ea808ba018057201af8cd"
REFERENCE = "lm-c64-qkv-lower-ablation-20261114-01"
CUDA_PATH = "torchcst/src/torchcst/_backends/cuda/algorithms/linear/radius_cell_mass"
ADMISSION = {
    f"{CUDA_PATH}/lm_width.py": (
        "190b3a49eb5ef4c36d6490ffb9ffefd423e8dc7a65468f75233a2cd551b310a0",
        "72a1778c65506392703b8c9a332cc3e49419eea2cdb89a1c7e9c2ed7c87879b4",
    ),
    f"{CUDA_PATH}/packed_rows.cuh": (
        "621a22ef9fa1264f8aebc07dbe6e025e0a4a5f6a6d244015a467d8fc47a14b37",
        "659089f307897df1f5fd94ba3ff1f81946ace70350c1dc70eba163df91f45038",
    ),
}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_sources(archive, source):
    protocol_file = archive / "lm-width-scale-4090-20261110-01" / "protocol.json"
    assert sha(protocol_file) == PROTOCOL_SHA
    protocol = read(protocol_file)
    expected = dict(protocol["source_files"])
    for parent in (
        "lm-c64-radius-cap-20261114-01",
        "lm-c64-cap512-qkv-only-20261114-01",
    ):
        manifest = read(archive / parent / "source-changes.json")["changes"]
        for name, changes in manifest.items():
            assert expected[name] == changes.get("before", changes.get("frozen")), name
            expected[name] = changes.get("after", changes.get("experiment"))
    patch = read(archive / REFERENCE / "source-change.json")
    assert expected[patch["path"]] == patch["before"]
    expected[patch["path"]] = patch["after"]
    for name, (before, after) in ADMISSION.items():
        assert expected[name] == before, name
        expected[name] = after
    for name, digest in expected.items():
        assert sha(source / name) == digest, name
    for name, module in (("torchcst", torchcst), ("cst-research", cst_research)):
        assert Path(module.__file__).resolve().is_relative_to((source / name).resolve())
    return protocol, expected


def tau_at(clock):
    progress = (1 - math.cos(math.pi * min(clock, ANNEAL_END) / ANNEAL_END)) / 2
    return (
        TAU_HIGH * (TAU_LOW / TAU_HIGH) ** progress if clock < ANNEAL_END else TAU_LOW
    )


def set_tau(sites, clock):
    for layer in sites.values():
        layer.kernel.scalar("tau").fill_(tau_at(clock))


def model_for(dense, arm):
    model = copy.deepcopy(dense)
    config = model.config
    sites = {}
    # The archived QKV run applied variance scaling with the exponential map,
    # then switched to the rational map for both training and evaluation.
    radius_envelope.USE_RATIONAL_LOWER = False
    try:
        for i, block in enumerate(model.blocks):
            if arm == "qkv":
                block.qkv = CSTAllQKVProjection(config, seed=SEED, index=i)
                for part in ("query", "key", "value"):
                    sites[f"{i}.{part}"] = getattr(block.qkv, part).map
            elif arm in ("down", "o"):
                coefficient = 96 if arm == "down" else 48
                projection = Projection(
                    4 * WIDTH if arm == "down" else WIDTH,
                    WIDTH,
                    config=SimpleNamespace(
                        **(asdict(config) | {"coefficient": coefficient})
                    ),
                    kind="cst",
                    std=config.init_std / math.sqrt(2 * config.layers),
                    seed=parameter_seed(
                        SEED,
                        f"block{i}.down" if arm == "down" else f"block{i}.attention.o",
                    ),
                )
                if arm == "down":
                    block.ffn.down.map = projection.map
                else:
                    projection.bias = block.attention_out.bias
                    block.attention_out = projection
                sites[f"{i}.{arm}"] = projection.map
    finally:
        radius_envelope.USE_RATIONAL_LOWER = True
    count, per_site = SITE_COUNTS[arm]
    assert len(sites) == count
    registry = make_algorithm_registry()
    if sites:
        registry.register(RadiusEnvelopeWideAmplitudeWidthLMAlgorithm())
    for layer in sites.values():
        assert len(layer.atoms.p) == per_site
        assert not torch.count_nonzero(layer.atoms.p[:, 1])
        assert layer.execution_declaration().kernel.parameterization.radius_max == CAP
        layer.selector = FixedSelector(
            RADIUS_ENVELOPE_WIDE_AMPLITUDE_WIDTH_LM, registry=registry
        )
    dense_parameters = dict(dense.named_parameters())
    for name, p in model.named_parameters():
        if name in dense_parameters:
            assert torch.equal(p, dense_parameters[name]), name
    model.kind = "cst" if sites else "dense"
    model.study_arm = "cst-c64" if sites else "dense-lr0.0003"
    return model, sites


def batches_at(data, order, clock):
    starts = order[(clock - 1) * 8 : clock * 8] * 128
    return [
        token_batch(data["train"], starts[j : j + 2], 128, "cuda")
        for j in range(0, 8, 2)
    ]


def audit_q(sites, optimizer, clock):
    rates = [
        (group["field_lrs"]["radius_fraction"], fields["radius_fraction"], group["lr"])
        for group, fields in zip(
            optimizer.param_groups, optimizer.get_field_lrs(), strict=True
        )
        if "radius_fraction" in fields
    ]
    assert len(rates) == 1
    reference, effective, group_rate = rates[0]
    assert reference == 0.0096 and math.isclose(effective, group_rate, rel_tol=1e-12)
    result = {"effective_lr": effective, "sites": {}}
    for name, layer in sites.items():
        p = layer.atoms.p
        q = p[:, 1].detach()
        state = optimizer.state[p]
        changed = int(torch.count_nonzero(q))
        moments = int(torch.count_nonzero(state["exp_avg_sq"][:, 1]))
        assert int(state["step"]) == clock and bool(((q >= 0) & (q <= 1)).all())
        assert changed > 0 and moments > 0, (name, changed, moments)
        result["sites"][name] = {
            "atoms": len(q),
            "positive_q": changed,
            "nonzero_q_second_moments": moments,
            "maximum_q": float(q.max()),
        }
    return result


def reset_rng():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def timed_step(model, optimizer, batches, clock):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    loss, event = step(model, optimizer, batches, clock)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - began
    assert math.isfinite(loss) and math.isfinite(event["gradient_norm_before_clip"])
    return {
        "step": clock,
        "train_ce": loss,
        "train_step_seconds": elapsed,
        "gradient_norm_before_clip": event["gradient_norm_before_clip"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def qualify(model, sites, data, order):
    # Attaching CSTAdamW changes the AtomState checkpoint contract. Qualify
    # an independent copy so the training model still has no attached optimizer.
    model, sites = copy.deepcopy((model, sites))
    model.cuda()
    layer = next(iter(sites.values()))
    original = layer.atoms.p.detach()[:8].clone()
    rng = torch.Generator(device="cuda").manual_seed(123)
    x = torch.randn(8, layer.in_features, device="cuda", generator=rng)
    with torch.no_grad():
        output_rms = float(layer(x).square().mean().sqrt())
        # Include both endpoints and broad supports in the full-size backward oracle.
        rows = layer.atoms.p[:8]
        rows[:, 0].copy_(
            rows.new_tensor([0.0, 0.001, -0.01, 0.03, 0.1, -0.25, 1.0, 2.0])
        )
        rows[:, 1].copy_(rows.new_tensor([0.0, 0.001, 0.1, 0.2, 0.5, 0.0, 0.9, 1.0]))
    oracle = full_oracle(layer)
    print("FORWARD_BACKWARD", json.dumps(oracle), flush=True)
    assert oracle["passed"], oracle
    with torch.no_grad():
        layer.atoms.p[:8].copy_(original)
    optimizer = optimizer_for(model)
    reset_rng()
    set_tau(sites, 1)
    metrics = timed_step(model, optimizer, batches_at(data, order, 1), 1)
    q_update = audit_q(sites, optimizer, 1)
    print("FIRST_Q_UPDATE", json.dumps(q_update), flush=True)
    return {
        "forward_backward": oracle,
        "initial_projection_output_rms": output_rms,
        "first_step": metrics,
        "first_q_update": q_update,
    }


def restore_checkpoint(path, model, optimizer, clock, digest):
    assert sha(path) == digest, path
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    assert saved["step"] == clock
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])


def train(arm, model, sites, data, order, out, report):
    state = report["arms"][arm]
    curve = out / f"{arm}.jsonl"
    if state.get("state") == "completed":
        assert sha(curve) == state["curve_sha256"]
        assert (
            sha(out / f"{arm}-{LONG_STEPS}.pt") == state["checkpoints"][str(LONG_STEPS)]
        )
        return
    state["state"] = "running"
    checkpoints = state.setdefault("checkpoints", {})
    start = max(map(int, checkpoints), default=0)
    model.cuda()
    optimizer = optimizer_for(model)
    reset_rng()
    rows = []
    if start:
        restore_checkpoint(
            out / f"{arm}-{start}.pt", model, optimizer, start, checkpoints[str(start)]
        )
        rows = [json.loads(line) for line in curve.read_text().splitlines()[:start]]
        assert [row["step"] for row in rows] == list(range(1, start + 1))
    set_tau(sites, start)
    state.update(
        completed_steps=start, initial_development=state.get("initial_development")
    )
    if start == 0:
        state["initial_development"] = development(model, data["development"])
    save(out / "result.json", report)
    print("ARM_STARTED", arm, start, state["initial_development"]["ce"], flush=True)
    with curve.open("w") as log:
        for row in rows:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        for clock in range(start + 1, LONG_STEPS + 1):
            if clock <= ANNEAL_END:
                set_tau(sites, clock)
            row = timed_step(model, optimizer, batches_at(data, order, clock), clock)
            row.update(seen_tokens=1024 * clock, tau=tau_at(clock))
            if clock == 1 and sites:
                state["first_q_update"] = audit_q(sites, optimizer, clock)
            if clock in EVAL_STEPS:
                row["development"] = development(model, data["development"])
                print("DEV", arm, clock, row["development"]["ce"], flush=True)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            state["completed_steps"] = clock
            if clock in SAVE_STEPS:
                log.flush()
                path = out / f"{arm}-{clock}.pt"
                tmp = path.with_suffix(".tmp")
                torch.save(
                    {
                        "step": clock,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(),
                    },
                    tmp,
                )
                tmp.replace(path)
                checkpoints[str(clock)] = sha(path)
            if clock % 50 == 0 or clock in EVAL_STEPS or clock in SAVE_STEPS:
                log.flush()
                save(out / "result.json", report)
    digest = parameter_digest(model)
    restore_checkpoint(
        out / f"{arm}-{LONG_STEPS}.pt",
        model,
        optimizer,
        LONG_STEPS,
        checkpoints[str(LONG_STEPS)],
    )
    assert parameter_digest(model) == digest
    assert all(
        int(parameter["step"]) == LONG_STEPS
        and all(
            torch.isfinite(parameter[field]).all()
            for field in ("exp_avg", "exp_avg_sq")
        )
        for parameter in optimizer.state.values()
    )
    if sites:
        state["terminal_q"] = audit_q(sites, optimizer, LONG_STEPS)
    final = development(model, data["development"])
    assert abs(final["ce"] - rows[-1]["development"]["ce"]) <= 1e-6
    assert (
        final["correct_next_tokens"] == rows[-1]["development"]["correct_next_tokens"]
    )
    confirmation = evaluate(model, data["confirmation"], batch=2)
    assert final["tokens"] == 8192 and confirmation["tokens"] == 32768
    measured = rows[10:]
    state.update(
        state="completed",
        checkpoint_audit="passed",
        final_development=final,
        confirmation=confirmation,
        final_parameters_sha256=digest,
        curve_sha256=sha(curve),
        median_train_step_ms=1000
        * statistics.median(row["train_step_seconds"] for row in measured),
        peak_allocated_bytes=max(row["peak_allocated_bytes"] for row in measured),
        peak_reserved_bytes=max(row["peak_reserved_bytes"] for row in measured),
        train_step_seconds=sum(row["train_step_seconds"] for row in rows),
    )
    save(out / "result.json", report)
    print("ARM_COMPLETE", arm, final["ce"], confirmation["ce"], flush=True)
    model.cpu()


def run(args):
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090"
    assert torch.cuda.get_device_capability() == (8, 9)
    assert torch.__version__ == "2.9.1+cu128" and torch.version.cuda == "12.8"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CST_RESEARCH_RADIUS_CAP"] = str(CAP)
    assert LONG_STEPS == 4882 and LONG_STEPS in EVAL_STEPS
    sources, files = verify_sources(args.archive_root, args.source)
    data_root = Path(sources["data"])
    assert sha(data_root / "manifest.json") == DATA_SHA
    _, data = verify_data(data_root)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    assert hashlib.sha256(order.tobytes()).hexdigest() == ORDER_SHA
    config = replace(WidthConfig(width=WIDTH), tau=TAU_HIGH)
    identity = {
        "config": asdict(config),
        "seed": SEED,
        "updates": LONG_STEPS,
        "targets_per_update": 1024,
        "lr_horizon_updates": SCHEDULE_STEPS,
        "tau_schedule": [TAU_HIGH, TAU_LOW, ANNEAL_END],
        "radius_min": 0.5,
        "radius_max": CAP,
        "lower": "0.5 + 3.5 / (1 + 3 * (a / 0.3)^2)",
        "q_initial": 0,
        "q_optimizer": "CSTAdamW radius_fraction field",
        "runner_sha256": sha(Path(__file__)),
        "source_protocol_sha256": PROTOCOL_SHA,
        "source_files_verified": len(files),
        "admission_file_sha256": {name: files[name] for name in ADMISSION},
        "data_manifest_sha256": DATA_SHA,
        "data_order_sha256": ORDER_SHA,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    result_path = args.output / "result.json"
    if args.resume:
        report = read(result_path)
        assert report["identity"] == identity
    else:
        assert not result_path.exists(), "use a clean output directory or --resume"
        args.output.mkdir(parents=True, exist_ok=True)
        report = {
            "state": "qualifying",
            "identity": identity,
            "qualification": {},
            "arms": {},
        }
    save(result_path, report)
    try:
        dense = make_model(config, "dense", (), seed=SEED)
        assert parameter_digest(dense) == DENSE_INITIAL_SHA
        prepared = {}
        for arm in ("down", "o", "qkv", "dense"):
            if report["arms"].get(arm, {}).get("state") == "completed":
                continue
            started = time.monotonic()
            model, sites = model_for(dense, arm)
            digest = parameter_digest(model)
            if arm == "qkv":
                assert digest == QKV_INITIAL_SHA
            per_site = {
                name: {
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "atoms": len(layer.atoms.p),
                    "atom_density": len(layer.atoms.p)
                    / (layer.in_features * layer.out_features),
                    "initialization": layer.initialization_report,
                }
                for name, layer in sites.items()
            }
            state = report["arms"].setdefault(
                arm,
                {
                    "state": "initialized",
                    "initial_parameters_sha256": digest,
                    "total_atoms": sum(row["atoms"] for row in per_site.values()),
                    "sites": per_site,
                    "initialization_seconds": time.monotonic() - started,
                },
            )
            assert state["initial_parameters_sha256"] == digest
            prepared[arm] = (model, sites)
            print("INITIALIZED", arm, state["total_atoms"], digest, flush=True)
            if arm in ("down", "o") and arm not in report["qualification"]:
                report["qualification"][arm] = qualify(model, sites, data, order)
                assert parameter_digest(model) == digest
                torch.cuda.empty_cache()
            save(result_path, report)
        del dense
        assert all(
            report["qualification"][arm]["forward_backward"]["passed"]
            for arm in ("down", "o")
        )
        report["state"] = "training"
        report.pop("error", None)
        save(result_path, report)
        for arm in ARMS:
            if arm in prepared:
                model, sites = prepared.pop(arm)
                train(arm, model, sites, data, order, args.output, report)
                del model, sites
                torch.cuda.empty_cache()
        assert all(report["arms"][arm]["state"] == "completed" for arm in ARMS)
        report["state"] = "completed"
        save(result_path, report)
    except BaseException as error:
        report.update(state="interrupted", error=f"{type(error).__name__}: {error}")
        save(result_path, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
