"""Replay the frozen TinyStories FFN-down protocol with paired initial seeds."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, replace
from functools import partial
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from lm_fp32_radius import decode
from lm_fp32_shared_step import shared_step
from torch.nn import functional as F
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import lm_width
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)

MERGED_TORCHCST = "6703a18a55cd949564be8a0ceca1ca1b813f7fc5"
REFERENCE_RESULT_SHA = (
    "c41e3e624e6aeb19fb7d08748d24b40dea0a5d5387602d639ed79f0f65f0e2d4"
)
SCHEDULE = ((23, ("dense", "down")), (42, ("down", "dense")), (17, ("down",)))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_reference(run):
    path = run / "run_lm_rational_projection.py"
    assert sha(run / "result.json") == REFERENCE_RESULT_SHA
    receipt = json.loads((run / "result.json").read_text())
    assert sha(path) == receipt["identity"]["runner_sha256"]
    spec = importlib.util.spec_from_file_location("_projection_anchor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, receipt


def check_sources(reference, args):
    archived = args.reference_run / "source"
    archive_package = SimpleNamespace(
        __file__=str(archived / "torchcst/src/torchcst/__init__.py")
    )
    with patch.object(reference, "torchcst", archive_package):
        protocol, frozen = reference.verify_sources(args.reference_run.parent, archived)
    manifest = reference.read(args.merged_manifest)
    assert manifest["commit"] == MERGED_TORCHCST
    for name, expected in (frozen | manifest["files"]).items():
        assert sha(args.source / name) == expected, name
    assert (
        Path(reference.torchcst.__file__)
        .resolve()
        .is_relative_to(args.source.resolve())
    )
    return protocol, len(frozen | manifest["files"])


def load_native():
    directory = Path(lm_width.__file__).parent
    inputs = [
        directory / "pipeline.cu",
        directory / "host_autograd.cpp",
        *sorted(directory.glob("*.cuh")),
    ]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest()[:12]
    name = f"torchcst_radius_lm_width_{digest}"
    path = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / name / (name + ".so")
    spec = importlib.util.spec_from_file_location(name, path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    lm_width.width_extension = lambda: native
    return native, sha(path)


def weight(native, layer):
    operator = layer.execution_declaration()
    specification = operator.kernel.parameterization
    layout = hierarchy_from_operator(lm_width.physical_operator(operator))
    return native.packed_dense_fourier_weight_from_atoms(
        decode(layer.atoms.p.contiguous(), specification),
        layout.input.native_config(),
        layout.output.native_config(),
    )


def shared_eval(evaluate, native, sites, model, *args, **kwargs):
    with torch.no_grad(), ExitStack() as stack:
        for layer in sites.values():
            forward = partial(F.linear, weight=weight(native, layer))
            stack.enter_context(patch.object(layer, "forward", forward))
        return evaluate(model, *args, **kwargs)


def qualified_oracle(oracle, native, layer):
    def forward(x):
        return F.linear(x, weight(native, layer))

    with patch.object(layer, "forward", forward):
        return oracle(layer)


def qualify(reference, native, model, sites, data, order):
    # Qualification uses a separate copy and cannot mutate the learner.
    model, sites = copy.deepcopy((model, sites))
    model.cuda()
    layer = next(iter(sites.values()))
    with torch.no_grad():
        first = layer.atoms.p[:8]
        initial = first.clone()
        first[:, 0].copy_(
            first.new_tensor([0.0, 0.001, -0.01, 0.03, 0.1, -0.25, 1.0, 2.0])
        )
        first[:, 1].copy_(first.new_tensor([0.0, 0.001, 0.1, 0.2, 0.5, 0.0, 0.9, 1.0]))
    oracle = qualified_oracle(reference.full_oracle, native, layer)
    print("FORWARD_BACKWARD_FP32", json.dumps(oracle), flush=True)
    assert oracle["passed"], oracle
    with torch.no_grad():
        first.copy_(initial)
    optimizer = reference.optimizer_for(model)
    reference.reset_rng()
    reference.set_tau(sites, 1)
    with patch.object(
        reference, "step", partial(shared_step, native, sites, decode=decode)
    ):
        metrics = reference.timed_step(
            model, optimizer, reference.batches_at(data, order, 1), 1
        )
    q_update = reference.audit_q(sites, optimizer, 1)
    print("QUALIFICATION_FIRST_Q", json.dumps(q_update), flush=True)
    return {
        "forward_backward": oracle,
        "first_step": metrics,
        "first_q_update": q_update,
    }


def run_seed(args, reference, anchor, native, data, order, identity, seed, arms):
    out = args.output / f"seed-{seed}"
    out.mkdir(exist_ok=True)
    result_path = out / "result.json"
    run_identity = identity | {"seed": seed, "arms": list(arms)}
    if result_path.exists():
        assert args.resume
        report = reference.read(result_path)
        assert report["identity"] == run_identity
    else:
        report = {"state": "initializing", "identity": run_identity, "arms": {}}
        reference.save(result_path, report)

    config = replace(
        reference.WidthConfig(width=reference.WIDTH), tau=reference.TAU_HIGH
    )
    with patch.object(reference, "SEED", seed):
        dense = reference.make_model(config, "dense", (), seed=seed)
        dense_digest = reference.parameter_digest(dense)
        if seed == 17:
            assert dense_digest == reference.DENSE_INITIAL_SHA
        for arm in arms:
            previous = report["arms"].get(arm, {})
            if previous.get("state") == "completed":
                assert sha(out / f"{arm}.jsonl") == previous["curve_sha256"]
                assert (
                    sha(out / f"{arm}-{reference.LONG_STEPS}.pt")
                    == previous["checkpoints"][str(reference.LONG_STEPS)]
                )
                continue
            start = time.monotonic()
            model, sites = reference.model_for(dense, arm)
            initial_hash = reference.parameter_digest(model)
            if seed == 17:
                assert initial_hash == anchor["arms"][arm]["initial_parameters_sha256"]
            per_site = {
                name: {
                    "atoms": len(layer.atoms.p),
                    "initialization": layer.initialization_report,
                }
                for name, layer in sites.items()
            }
            state = report["arms"].setdefault(
                arm,
                {
                    "state": "initialized",
                    "initial_parameters_sha256": initial_hash,
                    "common_dense_initial_sha256": dense_digest,
                    "sites": per_site,
                    "initialization_seconds": time.monotonic() - start,
                },
            )
            assert state["initial_parameters_sha256"] == initial_hash
            reference.save(result_path, report)
            print("INITIALIZED", seed, arm, initial_hash, flush=True)
            if sites:
                scope = patch.multiple(
                    reference,
                    step=partial(shared_step, native, sites, decode=decode),
                    development=partial(
                        shared_eval, reference.development, native, sites
                    ),
                    evaluate=partial(shared_eval, reference.evaluate, native, sites),
                )
            else:
                scope = nullcontext()
            with scope:
                if seed == 23 and arm == "down" and "qualification" not in report:
                    report["qualification"] = qualify(
                        reference, native, model, sites, data, order
                    )
                    assert reference.parameter_digest(model) == initial_hash
                    reference.save(result_path, report)
                report["state"] = "training"
                reference.train(arm, model, sites, data, order, out, report)
            del model, sites
            torch.cuda.empty_cache()
        assert all(report["arms"][arm]["state"] == "completed" for arm in arms)
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
    reference, anchor = import_reference(args.reference_run)
    assert reference.WIDTH == 1024 and reference.LONG_STEPS == 4882
    os.environ["CST_RESEARCH_RADIUS_CAP"] = str(reference.CAP)
    protocol, file_count = check_sources(reference, args)
    native, extension_sha = load_native()
    data_root = Path(protocol["data"])
    assert sha(data_root / "manifest.json") == reference.DATA_SHA
    _, data = reference.verify_data(data_root)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    assert hashlib.sha256(order.tobytes()).hexdigest() == reference.ORDER_SHA
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config": asdict(
            replace(
                reference.WidthConfig(width=reference.WIDTH), tau=reference.TAU_HIGH
            )
        ),
        "study_schedule": [[seed, list(arms)] for seed, arms in SCHEDULE],
        "torchcst_merge_commit": MERGED_TORCHCST,
        "source_manifest_sha256": sha(args.merged_manifest),
        "source_files_verified": file_count,
        "driver_sha256": sha(Path(__file__)),
        "decoder_sha256": sha(Path(__file__).with_name("lm_fp32_radius.py")),
        "shared_step_sha256": sha(Path(__file__).with_name("lm_fp32_shared_step.py")),
        "reference_runner_sha256": anchor["identity"]["runner_sha256"],
        "reference_result_sha256": REFERENCE_RESULT_SHA,
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
    }
    result = args.output / "summary.json"
    if result.exists():
        assert args.resume
        summary = reference.read(result)
        assert summary["identity"] == identity
    else:
        summary = {"state": "training", "identity": identity, "seeds": {}}
    summary.pop("error", None)
    summary["state"] = "training"
    reference.save(result, summary)
    try:
        for seed, arms in SCHEDULE:
            print("SEED", seed, arms, flush=True)
            report = run_seed(
                args, reference, anchor, native, data, order, identity, seed, arms
            )
            summary["seeds"][str(seed)] = report["arms"]
            reference.save(result, summary)
        summary["state"] = "completed"
    except BaseException as failure:
        summary.update(
            state="interrupted", error=f"{type(failure).__name__}: {failure}"
        )
        raise
    finally:
        reference.save(result, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
