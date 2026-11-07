"""Bounded native proposal checks and actual four-block task/resume preflight."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .ffn_contract import ModelConfig, learning_rate
from .ffn_field_lr import FieldRateOptimizer, recipe_for
from .ffn_model import LanguageModel, configure_native, cst_optimizer, make_cst
from .ffn_qualification import _assert_tree_equal, compare
from .ffn_run import token_batch, verify_data


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controlled_models(recipe, *, device="cuda"):
    """Create pristine copies before optimizer-specific Atom state is bound."""
    cpu = torch.nn.Module()
    cpu.layer = make_cst(16, 64, seed=151)
    cpu.ordinary = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
    live = copy.deepcopy(cpu).to(device)
    stock_model = (
        copy.deepcopy(live) if recipe.amplitude == recipe.outer_clock else None
    )
    return cpu, live, stock_model


def controlled_check(recipe):
    cpu, live, stock_model = controlled_models(recipe)
    reference = FieldRateOptimizer(cpu, recipe, native=False)
    optimizer = FieldRateOptimizer(live, recipe)
    stock = (
        cst_optimizer(
            stock_model,
            atom_lr_multiplier=16.0,
            lr=0.0003,
            betas=(0.9, 0.95),
            weight_decay=0,
            foreach=False,
        )
        if stock_model is not None
        else None
    )
    max_parameter_error = 0.0
    for step, factor in enumerate((0.5, 1.0, 0.25, 0.0, 0.8)):
        opts = (
            (reference, optimizer, stock)
            if stock is not None
            else (reference, optimizer)
        )
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = recipe.ordinary * factor * group["lr_scale"]
        generator = torch.Generator().manual_seed(450 + step)
        for p_cpu, p_live in zip(cpu.parameters(), live.parameters()):
            gradient = torch.randn(p_cpu.shape, generator=generator) * 0.001
            p_cpu.grad = gradient.clone()
            p_live.grad = gradient.cuda()
        if stock_model is not None:
            for p, q in zip(live.parameters(), stock_model.parameters()):
                q.grad = p.grad.clone()
        for opt in opts:
            opt.step()
        for p_cpu, p_live in zip(cpu.parameters(), live.parameters()):
            row = compare(p_live, p_cpu, atol=2e-6, rtol=2e-5)
            max_parameter_error = max(max_parameter_error, row["max_abs_error"])
            for key in ("exp_avg", "exp_avg_sq", "step"):
                compare(
                    optimizer.state[p_live][key],
                    reference.state[p_cpu][key],
                    atol=2e-8,
                    rtol=2e-5,
                )
        if stock_model is not None:
            for p, q in zip(live.parameters(), stock_model.parameters()):
                compare(p, q, atol=2e-6, rtol=2e-5)
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    compare(
                        optimizer.state[p][key],
                        stock.state[q][key],
                        atol=2e-8,
                        rtol=2e-5,
                    )
    return {
        "controlled_updates": 5,
        "schedule_factors": [0.5, 1.0, 0.25, 0.0, 0.8],
        "all_parameters_moments_counters_passed": True,
        "cpu_native_max_parameter_error": max_parameter_error,
        "stock_all16_comparison_passed": stock is not None,
    }


def task_check(config, recipe, data):
    model = LanguageModel(config, "cst", 17).cuda()
    configure_native(model)
    optimizer = FieldRateOptimizer(model, recipe)
    order = np.random.default_rng(17 + 888).permutation(20000000 // config.context)

    def step(m, opt, number):
        m.train()
        opt.zero_grad(set_to_none=True)
        lr = learning_rate(number, 20000000 // 1024)
        for group in opt.param_groups:
            group["lr"] = lr * group["lr_scale"]
        before = {
            n: p.detach().clone()
            for n, p in m.named_parameters()
            if n.endswith("atoms.p")
        }
        loss_sum = 0.0
        for micro in range(4):
            index = ((number - 1) * 4 + micro) * 2
            starts = order[index : index + 2] * config.context
            x, y = token_batch(data["train"], starts, config.context, "cuda")
            _, loss = m(x, y)
            (loss / 4).backward()
            loss_sum += float(loss.detach()) / 4
        for n, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
            if n in before:
                assert p.grad.abs().max() > 0, n
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        assert len(before) == 8 and all(
            not torch.equal(before[n], p)
            for n, p in m.named_parameters()
            if n in before
        )
        return loss_sum

    first = step(model, optimizer, 1)
    storage = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, storage
    )
    storage.seek(0)
    checkpoint = torch.load(storage, weights_only=False)
    resumed = LanguageModel(config, "cst", 91).cuda()
    resumed.load_state_dict(checkpoint["model"])
    configure_native(resumed)
    other = FieldRateOptimizer(resumed, recipe)
    other.load_state_dict(checkpoint["optimizer"])
    _assert_tree_equal(optimizer.state_dict(), other.state_dict())
    left, right = step(model, optimizer, 2), step(resumed, other, 2)
    assert abs(left - right) < 3e-5
    errors = []
    for p, q in zip(model.parameters(), resumed.parameters()):
        errors.append(compare(p, q, atol=3e-6, rtol=3e-5)["max_abs_error"])
        for key in ("step", "exp_avg", "exp_avg_sq"):
            compare(optimizer.state[p][key], other.state[q][key], atol=3e-7, rtol=3e-5)
    return {
        "cst_projections": 8,
        "actual_batch": [2, 128],
        "accumulation": 4,
        "all_sites_finite_nonzero_gradients_and_changed": True,
        "serialized_optimizer_exact_before_next_step": True,
        "resume_parameters_moments_counters_passed": True,
        "first_ce": first,
        "next_ce": left,
        "resumed_next_ce": right,
        "max_parameter_error": max(errors),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=(
            "all16",
            "amplitude1-centers16",
            "amplitude8-centers16",
            "amplitude32-centers16",
            "amplitude64-centers16",
        ),
        default=["all16", "amplitude1-centers16"],
    )
    args = parser.parse_args(argv)
    assert digest(args.data / "manifest.json") == args.data_manifest_sha256
    assert torch.__version__ == "2.11.0+cu126"
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090"
    assert torch.cuda.get_device_capability(0) == (8, 9)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, data = verify_data(args.data)
    config = ModelConfig(atom_lr_multiplier=16.0)
    args.output.mkdir(exist_ok=False)
    result = {
        "state": "running",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "config": asdict(config),
        "data_manifest_sha256": args.data_manifest_sha256,
        "source_sha256": {
            name: digest(Path(__file__).parent / name)
            for name in (
                "ffn_field_lr.py",
                "ffn_field_lr_qualification.py",
                "ffn_run.py",
                "ffn_model.py",
                "ffn_contract.py",
            )
        },
        "scope": "Proposal optimizer and actual eight-projection task preflight; operator Y/dX/all-Atom VJP qualification is separate and required by runner",
        "arms": {},
    }

    def save():
        (args.output / "qualification.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    save()
    try:
        for arm in args.arms:
            recipe = recipe_for(arm)
            result["arms"][arm] = {
                "recipe": recipe.manifest(),
                "controlled": controlled_check(recipe),
            }
            result["arms"][arm]["task_resume"] = task_check(config, recipe, data)
            save()
            print("PASSED field LR qualification " + arm, flush=True)
            torch.cuda.empty_cache()
        result["state"] = "passed"
        save()
    except BaseException as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
