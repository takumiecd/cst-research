"""Admission for a causal LM with Dense FFN and CST-only V projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from .ffn_contract import ModelConfig
from .ffn_model import LanguageModel, configure_native, parameter_seed
from .ffn_qualification import (
    compare,
    independent_quadrature_check,
    optimizer_and_resume_check,
    qualify_case,
    task_step_and_resume_check,
)


def shared_initialization(config):
    dense = LanguageModel(
        ModelConfig(**{**asdict(config), "attention_kind": "dense"}), "dense", 17
    )
    value = LanguageModel(config, "dense", 17)
    target = dict(value.named_parameters())
    checked = 0
    for name, parameter in dense.named_parameters():
        if ".qkv." in name:
            continue
        assert torch.equal(parameter, target[name]), name
        checked += 1
    for a, b in zip(dense.blocks, value.blocks):
        assert torch.equal(a.qkv.weight[: 2 * config.width], b.qkv.qk.weight)
        assert torch.equal(a.qkv.bias[: 2 * config.width], b.qkv.qk.bias)
    reports = value.initialization_reports()
    assert len(reports) == config.layers and all(
        v["relative_error"] <= 0.005 for v in reports.values()
    )
    return {
        "shared_parameters_exact": checked,
        "all_QK_weights_and_biases_exact": True,
        "reports": reports,
    }


def causal_and_manual_attention_check(config):
    """Explicit masked softmax checks integration and future-token separation."""
    model = LanguageModel(config, "dense", 17).cuda()
    configure_native(model)
    block = model.blocks[0]
    x = torch.randn(2, 7, config.width, device="cuda", requires_grad=True)
    live = block(x)
    z = block.norm1(x)
    q, k, v = block.qkv(z).chunk(3, -1)

    def heads(t):
        return t.reshape(2, 7, config.heads, config.width // config.heads).transpose(
            1, 2
        )

    q, k, v = map(heads, (q, k, v))
    scores = (q @ k.transpose(-2, -1)) / ((config.width // config.heads) ** 0.5)
    mask = torch.ones(7, 7, device="cuda", dtype=torch.bool).triu(1)
    attention = (
        (scores.masked_fill(mask, float("-inf")).softmax(-1) @ v)
        .transpose(1, 2)
        .reshape(2, 7, config.width)
    )
    intermediate = x + block.attention_out(attention)
    reference = intermediate + block.ffn(block.norm2(intermediate))
    cotangent = torch.randn_like(live)
    inputs = (x, block.qkv.value.map.atoms.p)
    ga = torch.autograd.grad(live, inputs, cotangent, retain_graph=True)
    gb = torch.autograd.grad(reference, inputs, cotangent, retain_graph=True)
    result = {
        "Y": compare(live, reference),
        "dX": compare(ga[0], gb[0]),
        "dAtoms": compare(ga[1], gb[1]),
    }
    future = torch.autograd.grad(live[:, :3].sum(), x)[0][:, 3:]
    assert torch.count_nonzero(future) == 0
    result["future_token_gradient_exactly_zero"] = True
    return result


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090"
    dense = ModelConfig(atom_lr_multiplier=16)
    config = ModelConfig(atom_lr_multiplier=16, attention_kind="cst_v")
    r = {
        "state": "running",
        "cohort": "recipe",
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "cases": [],
        "task_recipes": [dense.recipe(), config.recipe()],
        "source_sha256": {
            n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
            for n in (
                "ffn_model.py",
                "ffn_contract.py",
                "ffn_qualification.py",
                "attention_qualification.py",
            )
        },
    }

    def save():
        (args.output / "qualification.json").write_text(json.dumps(r, indent=2) + "\n")

    save()
    try:
        r["shared_initialization"] = shared_initialization(config)
        save()
        r["quadrature"] = independent_quadrature_check(q=0.0, std=0.001)
        save()
        for ni, no, std, seed_name in [
            (256, 1024, 0.02, "block0.up"),
            (1024, 256, 0.02 / (8**0.5), "block0.down"),
            (256, 256, 0.02, "block0.attention.v"),
        ]:
            r["cases"].append(
                qualify_case(
                    ni, no, 128, q=0.0, std=std, seed=parameter_seed(17, seed_name)
                )
            )
            save()
        r["optimizer"] = optimizer_and_resume_check(
            q=0.0, atom_lr_multiplier=16, std=0.001
        )
        save()
        r["attention"] = causal_and_manual_attention_check(config)
        save()
        r["task_step"] = task_step_and_resume_check(
            full_task=True,
            ffn_kind="dense",
            attention_kind="cst_v",
            atom_lr_multiplier=16,
        )
        save()
        assert r["task_step"]["cst_projections"] == 4
        r["state"] = "passed"
        save()
        print("ATTENTION QUALIFICATION PASSED", flush=True)
    except BaseException as e:
        r.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main()
