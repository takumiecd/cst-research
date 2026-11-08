"""Admission for a causal LM with Dense FFN and one CST Q/K/V projection type."""

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


def shared_initialization(config, ffn_kind="dense"):
    dense = LanguageModel(
        ModelConfig(**{**asdict(config), "attention_kind": "dense"}), ffn_kind, 17
    )
    value = LanguageModel(config, ffn_kind, 17)
    target = dict(value.named_parameters())
    checked = 0
    for name, parameter in dense.named_parameters():
        if ".qkv." in name or (
            config.attention_kind == "cst_qkvo" and ".attention_out." in name
        ):
            continue
        assert torch.equal(parameter, target[name]), name
        checked += 1
    for a, b in zip(dense.blocks, value.blocks):
        if config.attention_kind in ("dense", "cst_qkv", "cst_qkvo"):
            continue
        rows = [
            row
            for slot in range(3)
            if slot != b.qkv.slot
            for row in range(slot * config.width, (slot + 1) * config.width)
        ]
        assert torch.equal(a.qkv.weight[rows], getattr(b.qkv, b.qkv.dense_name).weight)
        assert torch.equal(a.qkv.bias[rows], getattr(b.qkv, b.qkv.dense_name).bias)
    reports = value.initialization_reports()
    expected = config.layers * (
        (
            4
            if config.attention_kind == "cst_qkvo"
            else 3
            if config.attention_kind == "cst_qkv"
            else 0
            if config.attention_kind == "dense"
            else 1
        )
        + (2 if ffn_kind == "cst" else 0)
    )
    assert len(reports) == expected and all(
        v["relative_error"] <= 0.005 for v in reports.values()
    )
    return {
        "shared_parameters_exact": checked,
        "complementary_projection_weights_and_biases_exact": True,
        "complementary_dense_maps": 0
        if config.attention_kind in ("cst_qkv", "cst_qkvo")
        else 3 * config.layers
        if config.attention_kind == "dense"
        else 2 * config.layers,
        **(
            {"all_QK_weights_and_biases_exact": True}
            if config.attention_kind == "cst_v"
            else {}
        ),
        "reports": reports,
    }


def causal_and_manual_attention_check(config, ffn_kind="dense"):
    """Explicit masked softmax checks integration and future-token separation."""
    model = LanguageModel(config, ffn_kind, 17).cuda()
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
    fields = (
        ("query", "key", "value")
        if config.attention_kind in ("cst_qkv", "cst_qkvo")
        else ()
        if config.attention_kind == "dense"
        else (block.qkv.cst_name,)
    )
    parameters = [getattr(block.qkv, field).map.atoms.p for field in fields]
    if config.attention_kind == "cst_qkvo":
        fields = (*fields, "output")
        parameters.append(block.attention_out.map.atoms.p)
    inputs = (x, *parameters)
    ga = torch.autograd.grad(live, inputs, cotangent, retain_graph=True)
    gb = torch.autograd.grad(reference, inputs, cotangent, retain_graph=True)
    result = {
        "Y": compare(live, reference),
        "dX": compare(ga[0], gb[0]),
        "dAtoms": {field: compare(a, b) for field, a, b in zip(fields, ga[1:], gb[1:])},
    }
    future = torch.autograd.grad(live[:, :3].sum(), x)[0][:, 3:]
    assert torch.count_nonzero(future) == 0
    result["future_token_gradient_exactly_zero"] = True
    return result


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--attention-kind",
        choices=("dense", "cst_v", "cst_q", "cst_k", "cst_qkv", "cst_qkvo"),
        default="cst_v",
    )
    p.add_argument("--ffn-kind", choices=("dense", "cst"), default="dense")
    p.add_argument("--width", type=int, choices=(256, 512), default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument(
        "--linear-backend",
        choices=("native", "levels", "levels_stream64"),
        default="native",
    )
    p.add_argument(
        "--radius-envelope",
        choices=("rational", "rational_zero_opening"),
        default="rational",
    )
    p.add_argument("--radius-birth", type=float, default=4.0)
    p.add_argument("--zero-opening-scale", type=float, default=0.03)
    p.add_argument(
        "--expected-gpu",
        choices=(
            "NVIDIA GeForce RTX 4090",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA A100 80GB PCIe MIG 3g.40gb",
            "NVIDIA L4",
        ),
        default="NVIDIA GeForce RTX 4090",
    )
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == args.expected_gpu
    settings = {
        "width": args.width,
        "heads": args.heads,
        "atom_lr_multiplier": 16,
        "linear_backend": args.linear_backend,
        "radius_envelope": args.radius_envelope,
        "radius_birth": args.radius_birth,
        "zero_opening_scale": args.zero_opening_scale,
    }
    dense = ModelConfig(**settings)
    config = ModelConfig(**settings, attention_kind=args.attention_kind)
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
        "task_model_configs": [asdict(dense), asdict(config)],
        "model_kinds": [
            {"ffn_kind": "dense", "attention_kind": "dense"},
            {"ffn_kind": args.ffn_kind, "attention_kind": args.attention_kind},
        ],
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
        r["shared_initialization"] = shared_initialization(config, args.ffn_kind)
        save()
        envelope = config.envelope_settings()
        r["quadrature"] = independent_quadrature_check(q=0.0, std=0.001, **envelope)
        save()
        cases = [
            (config.width, config.intermediate, config.init_std, "block0.up"),
            (
                config.intermediate,
                config.width,
                config.init_std / (8**0.5),
                "block0.down",
            ),
        ]
        roles = (
            ("q", "k", "v")
            if args.attention_kind in ("cst_qkv", "cst_qkvo")
            else ()
            if args.attention_kind == "dense"
            else ({"cst_v": "v", "cst_q": "q", "cst_k": "k"}[args.attention_kind],)
        )
        cases += [
            (config.width, config.width, config.init_std, "block0.attention." + role)
            for role in roles
        ]
        if config.attention_kind == "cst_qkvo":
            cases.append(
                (
                    config.width,
                    config.width,
                    config.init_std / (8**0.5),
                    "block0.attention.o",
                )
            )
        for ni, no, std, seed_name in cases:
            r["cases"].append(
                qualify_case(
                    ni,
                    no,
                    128,
                    q=0.0,
                    std=std,
                    seed=parameter_seed(17, seed_name),
                    linear_backend=config.linear_backend,
                    **envelope,
                )
            )
            save()
        r["optimizer"] = optimizer_and_resume_check(
            q=0.0, atom_lr_multiplier=16, std=0.001, **envelope
        )
        save()
        r["attention"] = causal_and_manual_attention_check(config, args.ffn_kind)
        save()
        r["task_step"] = task_step_and_resume_check(
            full_task=True,
            ffn_kind=args.ffn_kind,
            task_config=config,
            atom_lr_multiplier=16,
        )
        save()
        assert r["task_step"]["cst_projections"] == config.layers * (
            (
                4
                if args.attention_kind == "cst_qkvo"
                else 3
                if args.attention_kind == "cst_qkv"
                else 0
                if args.attention_kind == "dense"
                else 1
            )
            + (2 if args.ffn_kind == "cst" else 0)
        )
        r["state"] = "passed"
        save()
        print("ATTENTION QUALIFICATION PASSED", flush=True)
    except BaseException as e:
        r.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main()
