"""Pair width-2048 QKV and FFN down replacements on an RTX 4090."""

import argparse
import copy
import json
import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_lm_fp32_seed_replication as fp32
import run_lm_qkv_2048 as base
import run_lm_qkv_local_sweep as sweep
import run_lm_qkv_role_budget as roles
import torch
from torchcst import FixedSelector, init, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
    RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM,
    RadiusEnvelopeDenseFourierWeightLMAlgorithm,
)

TORCHCST_COMMIT = "936b25bc99a1bf968c8cef34a392be563969d9a6"
MANIFEST_SHA = "0e1a0adff05b4d2b3e26530ea9e32a0a14fa8ae03ba6b61e5c66b153dce6a54f"
QKV_ANCHOR_SHA = "8dcee587925476f5c03ad3b696d87f971f272b0c5ddaa7c3ae820f1ea4144224"
QKV_COEFFICIENTS = (48, 48, 64)
DOWN_COEFFICIENT = 96
DOWN_ATOMS = 393216
ARMS = ("dense", "qkv", "down", "qkv-down")
SCHEDULES = {
    "pilot": ((17, ARMS),),
    "replication": (
        (23, ("down", "dense", "qkv-down", "qkv")),
        (42, ("qkv-down", "qkv", "dense", "down")),
    ),
}
PROBE_COUNTS = {"qkv": 98304, "down": DOWN_ATOMS, "qkv-down": 98304}
INITIALIZATION_COUNTS = PROBE_COUNTS | {"value": 64 * sweep.WIDTH}


def build_model(reference, dense, arm, seed):
    model, sites = roles.build_model(
        reference,
        dense,
        arm if arm in ("qkv", "qkv-down") else "dense",
        seed,
        allocations={"qkv": QKV_COEFFICIENTS, "qkv-down": QKV_COEFFICIENTS},
    )
    if arm in ("down", "qkv-down"):
        config = model.config
        projection_config = SimpleNamespace(
            **(asdict(config) | {"coefficient": DOWN_COEFFICIENT})
        )
        registry = make_algorithm_registry()
        registry.register(RadiusEnvelopeDenseFourierWeightLMAlgorithm())
        with patch.object(base.radius_envelope, "USE_RATIONAL_LOWER", False):
            for i, block in enumerate(model.blocks):
                projection = reference.Projection(
                    4 * config.width,
                    config.width,
                    config=projection_config,
                    kind="cst",
                    std=config.init_std / math.sqrt(2 * config.layers),
                    seed=reference.parameter_seed(seed, f"block{i}.down"),
                )
                layer = projection.map
                assert len(layer.atoms.p) == DOWN_ATOMS
                assert not torch.count_nonzero(layer.atoms.p[:, 1])
                assert (
                    layer.execution_declaration().kernel.parameterization.radius_max
                    == 512
                )
                layer.selector = FixedSelector(
                    RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM, registry=registry
                )
                block.ffn.down.map = layer
                sites[f"{i}.down"] = layer
        model.kind = "cst"

    expected_sites = {"dense": 0, "qkv": 12, "down": 4, "qkv-down": 16}
    assert len(sites) == expected_sites[arm]
    dense_params = dict(dense.named_parameters())
    for name, param in model.named_parameters():
        if name in dense_params:
            assert torch.equal(param, dense_params[name]), name
    return model, sites


def make_preparer(dense_anchor, qkv_anchor):
    prepare = base.prepare
    cpu_scale = init.variance_scaling_
    dense_checks = set()
    site_digests = {}

    def prepare_gpu(reference, dense, arm, seed):
        if seed not in dense_checks:
            prior = dense_anchor["seeds"][str(seed)]["dense"]
            assert (
                reference.parameter_digest(dense) == prior["initial_parameters_sha256"]
            )
            dense_checks.add(seed)
        with patch.object(init, "variance_scaling_", sweep.gpu_variance_scaling):
            model, sites, metadata = prepare(reference, dense, arm, seed)
        if arm == "qkv":
            prior = qkv_anchor["seeds"][str(seed)]["q48-k48-v64"]
            assert (
                metadata["initial_parameters_sha256"]
                == prior["initial_parameters_sha256"]
            )
        for name, layer in sites.items():
            digest = reference.parameter_digest(layer)
            assert site_digests.setdefault((seed, name), digest) == digest
            metadata["sites"][name]["initial_parameters_sha256"] = digest

        if seed == 17 and arm == "down":
            site = "0.down"
            gpu = sites[site]
            cpu = copy.deepcopy(gpu)
            parameter_seed = reference.parameter_seed(seed, "block0.down")
            std = model.config.init_std / math.sqrt(2 * model.config.layers)
            started = time.monotonic()
            with patch.object(base.radius_envelope, "USE_RATIONAL_LOWER", False):
                expected = cpu_scale(
                    cpu,
                    std**2,
                    generator=torch.Generator().manual_seed(parameter_seed + 1),
                    radius_fraction=0,
                )
            cpu_seconds = time.monotonic() - started
            assert torch.equal(cpu.atoms.p[:, 1:], gpu.atoms.p[:, 1:])
            torch.testing.assert_close(
                cpu.atoms.p[:, 0], gpu.atoms.p[:, 0], rtol=1e-6, atol=1e-7
            )
            metadata["down_initializer_check"] = {
                "site": site,
                "bitwise_equal": torch.equal(cpu.atoms.p, gpu.atoms.p),
                "max_abs": float(
                    (cpu.atoms.p.detach() - gpu.atoms.p.detach()).abs().max()
                ),
                "relative_energy_difference": abs(
                    expected.expected_energy
                    / gpu.initialization_report["expected_energy"]
                    - 1
                ),
                "cpu_seconds": cpu_seconds,
            }
            print(
                "DOWN_GPU_INITIALIZER",
                json.dumps(metadata["down_initializer_check"]),
                flush=True,
            )
        return model, sites, metadata

    return prepare_gpu


def benchmark_preflight(reference, native, config, data, order, preflight):
    if preflight is None:
        dense = reference.make_model(config, "dense", (), seed=17)
        prepared = {arm: base.prepare(reference, dense, arm, 17) for arm in ARMS}
    else:
        _, prepared = preflight
    warmup, measured = 10, 24
    timing = {}
    for arm in ARMS:
        base.release_cuda()
        idle_allocated = torch.cuda.memory_allocated()
        model, sites = copy.deepcopy(prepared[arm][:2])
        model.cuda()
        optimizer = reference.optimizer_for(model)
        step = partial(fp32.shared_step, native, sites, decode=fp32.decode)
        scope = patch.object(reference, "step", step) if sites else nullcontext()
        samples = []
        with patch.object(reference, "SEED", 17), scope:
            reference.reset_rng()
            for clock in range(1, warmup + measured + 1):
                reference.set_tau(sites, clock)
                sample = reference.timed_step(
                    model, optimizer, reference.batches_at(data, order, clock), clock
                )
                if clock > warmup:
                    samples.append(sample)
        timing[arm] = {
            "pre_arm_allocated_bytes": idle_allocated,
            "median_train_step_ms": 1000
            * statistics.median(row["train_step_seconds"] for row in samples),
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in samples),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in samples),
            "samples": samples,
        }
        print("PREFLIGHT_TIMING", arm, timing[arm]["median_train_step_ms"], flush=True)
        del optimizer, model, sites, step, scope
    base.release_cuda()
    return {
        "seed": 17,
        "warmup_updates": warmup,
        "measured_updates": measured,
        "arms": timing,
    }


def main(args):
    assert fp32.sha(args.paired_anchor) == sweep.PAIRED_ANCHOR_SHA
    assert fp32.sha(args.qkv_anchor) == QKV_ANCHOR_SHA
    dense_anchor = json.loads(args.paired_anchor.read_text())
    qkv_anchor = json.loads(args.qkv_anchor.read_text())
    schedule = SCHEDULES[args.phase]
    identity = {
        "study": "lm-qkv-down-2048-20261118",
        "phase": args.phase,
        "projection_driver_sha256": fp32.sha(Path(__file__)),
        "allocation_builder_sha256": fp32.sha(Path(roles.__file__)),
        "initialization_adapter_sha256": fp32.sha(Path(sweep.__file__)),
        "paired_anchor_sha256": sweep.PAIRED_ANCHOR_SHA,
        "qkv_anchor_sha256": QKV_ANCHOR_SHA,
        "qkv_coefficients": list(QKV_COEFFICIENTS),
        "down_coefficient": DOWN_COEFFICIENT,
        "atoms_per_down_projection": DOWN_ATOMS,
        "oracle_scope": "The first projection of each replacement arm, including FFN down in the down-only arm.",
    }
    with (
        patch.multiple(
            base,
            TORCHCST_COMMIT=TORCHCST_COMMIT,
            MANIFEST_SHA=MANIFEST_SHA,
            COUNTS=PROBE_COUNTS,
            SCHEDULE=schedule,
            model_for=build_model,
            prepare=make_preparer(dense_anchor, qkv_anchor),
        ),
        patch.object(sweep, "COUNTS", INITIALIZATION_COUNTS),
    ):
        base.main(
            args, extra_identity=identity, benchmark_preflight=benchmark_preflight
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--paired-anchor", type=Path, required=True)
    parser.add_argument("--qkv-anchor", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=SCHEDULES, default="pilot")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
