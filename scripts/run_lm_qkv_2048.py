"""Run paired TinyStories QKV-only comparisons at width 2048 on one RTX 4090."""

import argparse
import copy
import gc
import hashlib
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lm_fp32_radius as radius
import numpy as np
import run_lm_fp32_seed_replication as fp32
import torch
from cst_research.experiments.ffn_selective import make_optimizer
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
    RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM,
    RadiusEnvelopeDenseFourierWeightLMAlgorithm,
)
from torchcst._backends.torch.parameterizations import radius_envelope

TORCHCST_COMMIT = "9c0cf9979dde545999b290f6e0ec13ff93ecf2b5"
MANIFEST_SHA = "f1589eb84b71b5176baa02f13a87e84e45910021cd8fa84defcbbf0e0fa2354e"
WIDTH = 2048
DENSE_LR = 0.0001
COUNTS = {"qkv-c64": 131072, "qkv-c128": 262144}
SCHEDULE = (
    (17, ("dense", "qkv-c64", "qkv-c128")),
    (23, ("qkv-c64", "qkv-c128", "dense")),
    (42, ("qkv-c128", "dense", "qkv-c64")),
)
SAVE_STEPS = {976, 2441, 4882}


def model_for(reference, dense, arm, seed):
    model = copy.deepcopy(dense)
    sites = {}
    if arm == "dense":
        return model, sites
    config = SimpleNamespace(
        **(asdict(model.config) | {"coefficient": COUNTS[arm] // WIDTH})
    )
    registry = make_algorithm_registry()
    registry.register(RadiusEnvelopeDenseFourierWeightLMAlgorithm())
    with patch.object(radius_envelope, "USE_RATIONAL_LOWER", False):
        for i, block in enumerate(model.blocks):
            block.qkv = reference.CSTAllQKVProjection(config, seed=seed, index=i)
            for name in ("query", "key", "value"):
                layer = getattr(block.qkv, name).map
                assert len(layer.atoms.p) == COUNTS[arm]
                assert not torch.count_nonzero(layer.atoms.p[:, 1])
                assert (
                    layer.execution_declaration().kernel.parameterization.radius_max
                    == 512
                )
                layer.selector = FixedSelector(
                    RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM, registry=registry
                )
                sites[f"{i}.{name}"] = layer
    assert len(sites) == 12
    for name, p in model.named_parameters():
        if ".qkv." not in name:
            assert torch.equal(p, dense.get_parameter(name)), name
    model.kind = "cst"
    return model, sites


def optimizer_for(model):
    optimizer = make_optimizer(model)
    for group in optimizer.param_groups:
        if "field_lrs" not in group:
            group["lr_scale"] = DENSE_LR / 0.0003
            group["lr"] = DENSE_LR
    return optimizer


def prepare(reference, dense, arm, seed):
    print("INITIALIZING", seed, arm, flush=True)
    started = time.monotonic()
    model, sites = model_for(reference, dense, arm, seed)
    metadata = {
        "state": "initialized",
        "initial_parameters_sha256": reference.parameter_digest(model),
        "sites": {
            name: {
                "atoms": len(layer.atoms.p),
                "initialization": layer.initialization_report,
            }
            for name, layer in sites.items()
        },
        "initialization_seconds": time.monotonic() - started,
    }
    print("INITIALIZED", seed, arm, metadata["initial_parameters_sha256"], flush=True)
    return model, sites, metadata


def release_cuda():
    # Optimizer hooks can retain state until cyclic garbage collection.
    gc.collect()
    torch.cuda.empty_cache()


def qualify_all(reference, native, config, data, order, summary, result_path):
    seed = 17
    dense = reference.make_model(config, "dense", (), seed=seed)
    prepared = {"dense": prepare(reference, dense, "dense", seed)}
    with patch.object(reference, "SEED", seed):
        for arm in COUNTS:
            if arm in summary["qualification"]:
                continue
            model, sites, metadata = prepared[arm] = prepare(
                reference, dense, arm, seed
            )
            started = time.monotonic()
            audit = fp32.qualify(reference, native, model, sites, data, order)
            assert audit["forward_backward"]["A"] == COUNTS[arm]
            assert len(audit["first_q_update"]["sites"]) == 12
            assert (
                reference.parameter_digest(model)
                == metadata["initial_parameters_sha256"]
            )
            summary["qualification"][arm] = audit | {
                "initial_parameters_sha256": metadata["initial_parameters_sha256"],
                "seconds": time.monotonic() - started,
            }
            release_cuda()
            reference.save(result_path, summary)
            print("QUALIFIED", arm, flush=True)
    return dense, prepared


def run_seed(
    args, reference, native, config, data, order, identity, seed, arms, preflight
):
    out = args.output / f"seed-{seed}"
    out.mkdir(exist_ok=True)
    result_path = out / "result.json"
    run_identity = identity | {"seed": seed, "arms": list(arms)}
    if result_path.exists():
        assert args.resume
        report = reference.read(result_path)
        assert report["identity"] == run_identity
    else:
        report = {"state": "training", "identity": run_identity, "arms": {}}
        reference.save(result_path, report)

    if preflight is None:
        dense = reference.make_model(config, "dense", (), seed=seed)
        prepared = {}
    else:
        dense, prepared = preflight
    dense_digest = reference.parameter_digest(dense)
    for arm in arms:
        state = report["arms"].get(arm, {})
        if state.get("state") == "completed":
            assert fp32.sha(out / f"{arm}.jsonl") == state["curve_sha256"]
            assert fp32.sha(out / f"{arm}-4882.pt") == state["checkpoints"]["4882"]
            continue
        release_cuda()
        resident = torch.cuda.memory_allocated()
        model, sites, metadata = (
            prepared.pop(arm)
            if arm in prepared
            else prepare(reference, dense, arm, seed)
        )
        state = report["arms"].setdefault(
            arm,
            metadata
            | {
                "common_dense_initial_sha256": dense_digest,
                "pre_arm_allocated_bytes": resident,
            },
        )
        assert (
            state["initial_parameters_sha256"] == metadata["initial_parameters_sha256"]
        )
        reference.save(result_path, report)
        scope = (
            patch.multiple(
                reference,
                step=partial(fp32.shared_step, native, sites, decode=radius.decode),
                development=partial(
                    fp32.shared_eval, reference.development, native, sites
                ),
                evaluate=partial(fp32.shared_eval, reference.evaluate, native, sites),
            )
            if sites
            else nullcontext()
        )
        with patch.object(reference, "SEED", seed), scope:
            reference.train(arm, model, sites, data, order, out, report)
        del model, sites, scope
    report["state"] = "completed"
    reference.save(result_path, report)
    return report


def main(args):
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090"
    assert torch.cuda.get_device_capability() == (8, 9)
    assert torch.__version__ == "2.9.1+cu128" and torch.version.cuda == "12.8"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference, anchor = fp32.import_reference(args.reference_run)
    assert reference.LONG_STEPS == 4882 and reference.SCHEDULE_STEPS == 19531
    assert fp32.sha(args.merged_manifest) == MANIFEST_SHA
    with patch.object(fp32, "MERGED_TORCHCST", TORCHCST_COMMIT):
        protocol, file_count = fp32.check_sources(reference, args)
    os.environ["CST_RESEARCH_RADIUS_CAP"] = "512"
    radius_envelope.USE_RATIONAL_LOWER = True
    native, extension_sha = fp32.load_native()
    data_root = Path(protocol["data"])
    assert fp32.sha(data_root / "manifest.json") == reference.DATA_SHA
    _, data = reference.verify_data(data_root)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    assert hashlib.sha256(order.tobytes()).hexdigest() == reference.ORDER_SHA
    config = reference.WidthConfig(width=WIDTH, tau=reference.TAU_HIGH)
    identity = {
        "config": asdict(config),
        "study_schedule": [[seed, list(arms)] for seed, arms in SCHEDULE],
        "atoms_per_qkv_projection": COUNTS,
        "ordinary_reference_lr": DENSE_LR,
        "atom_reference_lr": 0.0096,
        "torchcst_commit": TORCHCST_COMMIT,
        "source_manifest_sha256": MANIFEST_SHA,
        "source_files_verified": file_count,
        "driver_sha256": fp32.sha(Path(__file__)),
        "fp32_helpers_sha256": {
            name: fp32.sha(Path(fp32.__file__).with_name(name))
            for name in (
                "run_lm_fp32_seed_replication.py",
                "lm_fp32_radius.py",
                "lm_fp32_shared_step.py",
            )
        },
        "reference_runner_sha256": anchor["identity"]["runner_sha256"],
        "reference_result_sha256": fp32.REFERENCE_RESULT_SHA,
        "data_manifest_sha256": reference.DATA_SHA,
        "data_order_sha256": reference.ORDER_SHA,
        "native_sha256": extension_sha,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": version("triton"),
        "tf32": False,
        "steps": reference.LONG_STEPS,
        "lr_horizon": reference.SCHEDULE_STEPS,
        "targets_per_update": 1024,
        "save_steps": sorted(SAVE_STEPS),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "summary.json"
    if path.exists():
        assert args.resume
        summary = reference.read(path)
        assert summary["identity"] == identity
    else:
        summary = {"identity": identity, "qualification": {}, "seeds": {}}
    summary.pop("error", None)
    summary["state"] = "qualifying"
    reference.save(path, summary)
    try:
        with patch.multiple(
            reference, optimizer_for=optimizer_for, SAVE_STEPS=SAVE_STEPS
        ):
            preflight = (
                qualify_all(reference, native, config, data, order, summary, path)
                if set(summary["qualification"]) != set(COUNTS)
                else None
            )
            assert set(summary["qualification"]) == set(COUNTS)
            summary["state"] = "training"
            reference.save(path, summary)
            for seed, arms in SCHEDULE:
                release_cuda()
                print("SEED", seed, arms, flush=True)
                report = run_seed(
                    args,
                    reference,
                    native,
                    config,
                    data,
                    order,
                    identity,
                    seed,
                    arms,
                    preflight if seed == 17 else None,
                )
                preflight = None
                summary["seeds"][str(seed)] = report["arms"]
                reference.save(path, summary)
        summary["state"] = "completed"
    except BaseException as failure:
        summary.update(
            state="interrupted", error=f"{type(failure).__name__}: {failure}"
        )
        raise
    finally:
        reference.save(path, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
