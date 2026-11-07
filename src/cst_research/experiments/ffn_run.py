"""Bounded, foreground CST-FFN learning run; provenance and raw curves retained."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .ffn_contract import ModelConfig, learning_rate, qualification_covers_model
from .ffn_model import LanguageModel, configure_native, cst_optimizer


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token_batch(tokens, starts, context, device):
    """Shift targets once; no circular wrap and no cross-split window reads."""
    rows = np.stack([tokens[int(i) : int(i) + context + 1] for i in starts]).astype(
        np.int64
    )
    if rows.shape[1] != context + 1:
        raise ValueError("next-token window exceeds its split")
    data = torch.from_numpy(rows).to(device)
    return data[:, :-1], data[:, 1:]


def verify_data(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["state"] != "prepared" or not manifest["story_hash_sets_disjoint"]:
        raise ValueError("prepared story-disjoint token data required")
    result = {}
    for name, entry in manifest["files"].items():
        path = root / entry["filename"]
        if sha(path) != entry["sha256"] or path.stat().st_size != 4 * entry["tokens"]:
            raise ValueError("token file changed: " + name)
        result[name] = np.memmap(path, mode="r", dtype="<u4")
        if int(result[name].max()) >= manifest["vocabulary"]:
            raise ValueError("token outside vocabulary")
    return manifest, result


@torch.no_grad()
def evaluate(model, tokens, batch):
    model.eval()
    length = model.config.context
    windows = (len(tokens) - 1) // length
    total, count = 0.0, 0
    for first in range(0, windows, batch):
        starts = np.arange(first, min(windows, first + batch)) * length
        x, y = token_batch(tokens, starts, length, "cuda")
        _, loss = model(x, y)
        total += float(loss) * x.numel()
        count += x.numel()
    return {"ce": total / count, "perplexity": math.exp(total / count), "tokens": count}


def probe(model, x, y):
    """Separate diagnostic backward, without optimizer or data-order changes."""
    model.train()
    model.zero_grad(set_to_none=True)
    activations, handles = {}, []
    for name, module in model.named_modules():
        if name.endswith((".ffn.up", ".ffn.down")):

            def capture(_module, _inputs, output, name=name):
                activations[name] = float(
                    output.detach().float().square().mean().sqrt()
                )

            handles.append(module.register_forward_hook(capture))
    try:
        _, loss = model(x, y)
        loss.backward()
        gradients = {}
        for name, p in model.named_parameters():
            if ".ffn." in name and p.grad is not None:
                if not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError("nonfinite FFN gradient: " + name)
                gradients[name] = {
                    "rms": float(p.grad.float().square().mean().sqrt()),
                    "max_abs": float(p.grad.abs().max()),
                }
        return {
            "loss": float(loss.detach()),
            "activation_rms": activations,
            "gradients": gradients,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def generate(model):
    import tiktoken

    encoding = tiktoken.get_encoding("gpt2")
    results = []
    model.eval()
    for i, prompt in enumerate(
        ("Once upon a time, there was a little", "Lily found a red ball in the garden.")
    ):
        ids = encoding.encode_ordinary(prompt)
        rng = torch.Generator(device="cuda").manual_seed(7000 + i)
        for _ in range(64):
            x = torch.tensor([ids[-model.config.context :]], device="cuda")
            logits = model(x)[0, -1] / 0.8
            threshold = torch.topk(logits, 40).values[-1]
            logits = logits.masked_fill(logits < threshold, -float("inf"))
            token = int(torch.multinomial(logits.softmax(-1), 1, generator=rng))
            ids.append(token)
            if token == encoding.eot_token:
                break
        results.append(
            {
                "prompt": prompt,
                "text": encoding.decode(ids),
                "ids": ids,
                "sampling_seed": 7000 + i,
                "temperature": 0.8,
                "top_k": 40,
            }
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--qualification-sha256", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("dense", "cst", "none"), required=True)
    parser.add_argument("--width", type=int, choices=(128, 256, 512), default=256)
    parser.add_argument("--expansion", type=int, choices=(1, 4, 16), default=4)
    parser.add_argument("--coefficient", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument(
        "--expected-gpu",
        choices=("NVIDIA GeForce RTX 4090", "NVIDIA A100 80GB PCIe MIG 3g.40gb"),
        default="NVIDIA GeForce RTX 4090",
    )
    parser.add_argument(
        "--q", type=float, choices=(0.0, 0.25, 0.5, 0.75, 1.0), default=0.0
    )
    parser.add_argument(
        "--amplitude-max", type=float, choices=(1.0, 2.0, 4.0), default=1.0
    )
    parser.add_argument("--w-c", type=float, choices=(0.3, 0.6, 1.2), default=0.3)
    parser.add_argument(
        "--radius-birth", type=float, choices=(1.0, 2.0, 4.0, 16.0), default=4.0
    )
    parser.add_argument(
        "--feature-order", choices=("contiguous", "permuted"), default="contiguous"
    )
    parser.add_argument(
        "--atom-lr-multiplier", type=float, choices=(1.0, 16.0), default=1.0
    )
    parser.add_argument(
        "--field-lr-arm",
        choices=(
            "all16",
            "amplitude1-centers16",
            "amplitude8-centers16",
            "amplitude32-centers16",
            "amplitude64-centers16",
        ),
    )
    parser.add_argument("--field-qualification", type=Path)
    parser.add_argument("--field-qualification-sha256")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tokens", type=int, default=1000000)
    parser.add_argument("--schedule-tokens", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    parser.add_argument("--repeat-fixed-batches", action="store_true")
    parser.add_argument("--track-atom-motion", action="store_true")
    args = parser.parse_args(argv)
    args.schedule_tokens = (
        args.tokens if args.schedule_tokens is None else args.schedule_tokens
    )
    if (
        not 1024 <= args.tokens <= 20000000
        or args.batch < 1
        or args.accumulation < 1
        or args.max_wall_seconds < 1
        or not args.tokens <= args.schedule_tokens <= 20000000
    ):
        parser.error("bounded positive training budget required")
    if args.repeat_fixed_batches and args.tokens > 102400:
        parser.error("fixed-batch diagnostic is capped at100 standard updates")
    if (
        sha(args.qualification) != args.qualification_sha256
        or sha(args.data / "manifest.json") != args.data_manifest_sha256
    ):
        parser.error(
            "qualification or data manifest identity differs from the frozen request"
        )
    qualification = json.loads(args.qualification.read_text())
    if qualification["state"] != "passed" or qualification["cohort"] not in (
        "target",
        "expanded",
        "capacity",
        "radius",
        "recipe",
    ):
        parser.error("passed target rectangular/multi-token qualification required")
    for name in ("ffn_model.py", "ffn_contract.py"):
        if sha(Path(__file__).parent / name) != qualification["source_sha256"][name]:
            parser.error("model source differs from qualified source")
    requested = ModelConfig(
        width=args.width,
        expansion=args.expansion,
        coefficient=args.coefficient,
        q=args.q,
        amplitude_max=args.amplitude_max,
        w_c=args.w_c,
        radius_birth=args.radius_birth,
        feature_order=args.feature_order,
        atom_lr_multiplier=args.atom_lr_multiplier,
    )
    if not qualification_covers_model(qualification, requested):
        parser.error(
            "qualification does not cover both dimensions, Atom count, kernel scalars and model feature ordering"
        )
    field_recipe = None
    if args.field_lr_arm is not None:
        from .ffn_field_lr import recipe_for

        if args.kind != "cst" or requested.atom_lr_multiplier != 16.0:
            parser.error("field LR arm requires CST with outer Atom multiplier16")
        if (
            args.field_qualification is None
            or sha(args.field_qualification) != args.field_qualification_sha256
        ):
            parser.error("field LR arm requires its exact separate qualification")
        field_recipe = recipe_for(args.field_lr_arm)
        field_qualification = json.loads(args.field_qualification.read_text())
        if (
            field_qualification["state"] != "passed"
            or field_qualification["gpu"] != args.expected_gpu
            or field_qualification["torch"] != torch.__version__
            or field_qualification["data_manifest_sha256"] != args.data_manifest_sha256
            or field_qualification["config"] != asdict(requested)
            or field_qualification["arms"][args.field_lr_arm]["recipe"]
            != field_recipe.manifest()
            or any(
                sha(Path(__file__).parent / name) != digest
                for name, digest in field_qualification["source_sha256"].items()
            )
        ):
            parser.error("field LR recipe/source/runtime differs from qualification")
    elif (
        args.field_qualification is not None
        or args.field_qualification_sha256 is not None
    ):
        parser.error("field qualification only belongs to an explicit field LR arm")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == args.expected_gpu
    if (
        qualification["gpu"] != args.expected_gpu
        or qualification["torch"] != torch.__version__
    ):
        parser.error("actual GPU/runtime differs from its qualification")
    manifest, data = verify_data(args.data)
    config = replace(requested, vocabulary=manifest["vocabulary"])
    if args.batch * config.context > 256:
        parser.error("initial qualified microbatch has at most256 token rows")
    if args.schedule_tokens + 1 > len(data["train"]):
        parser.error("train prefix does not cover token budget")
    model = LanguageModel(config, args.kind, args.seed).cuda()
    ffn_parameters = [p for name, p in model.named_parameters() if ".ffn." in name]
    actual_ffn_parameters = sum(p.numel() for p in ffn_parameters)
    expected_ffn_parameters = (
        0
        if args.kind == "none"
        else config.ffn_parameter_budget()[args.kind + "_parameters"]
    )
    assert actual_ffn_parameters == expected_ffn_parameters
    if args.kind == "cst":
        configure_native(model)
        if field_recipe is not None:
            from .ffn_field_lr import FieldRateOptimizer

            optimizer = FieldRateOptimizer(model, field_recipe)
        else:
            optimizer = cst_optimizer(
                model,
                atom_lr_multiplier=config.atom_lr_multiplier,
                lr=0.0003,
                betas=(0.9, 0.95),
                weight_decay=0.0,
                foreach=False,
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0003,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            foreach=False,
        )
    motion = None
    if args.track_atom_motion and args.kind == "cst":
        from torchcst import CSTLinear

        from .ffn_motion import AmplitudeMotion, geometry_snapshot

        motion = AmplitudeMotion(
            {
                name: (module.atoms.p, float(module.kernel.scalar("amplitude_max")))
                for name, module in model.named_modules()
                if isinstance(module, CSTLinear)
            }
        )
    tokens_per_update = args.batch * config.context * args.accumulation
    updates = args.tokens // tokens_per_update
    horizon_updates = args.schedule_tokens // tokens_per_update
    windows = args.schedule_tokens // config.context
    order = np.random.default_rng(args.seed + 888).permutation(windows)
    total_microbatches = updates * args.accumulation
    assert total_microbatches * args.batch <= len(order)
    milestones = {
        n // tokens_per_update for n in (1000000, 5000000, 10000000, 20000000)
    }
    px, py = token_batch(data["train"], [0], config.context, "cuda")
    summary = {
        "state": "running",
        "config": asdict(config),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "qualification_sha256": sha(args.qualification),
        "data_manifest_sha256": sha(args.data / "manifest.json"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "initialization_reports": model.initialization_reports(),
        "ffn_budget": config.ffn_parameter_budget(),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "actual_ffn_parameters": actual_ffn_parameters,
        "ffn_parameter_tensor_bytes": sum(
            p.numel() * p.element_size() for p in ffn_parameters
        ),
        "scheduled_updates": updates,
        "learning_rate_horizon_updates": horizon_updates,
        "data_order_horizon_tokens": args.schedule_tokens,
        "milestone_checkpoint_sha256": {},
        "scheduled_tokens": updates * tokens_per_update,
        "completed_updates": 0,
        "completed_tokens": 0,
        "optimizer": "AdamW baseLR3e-4 betas(.9,.95) eps1e-8 decay0, global clip1; CST coordinate policy, Atom group multiplier from config",
        "measurement": "synchronized full optimizer update including accumulation/data transfer; no probes/evaluation in timed samples; eager FP32 TF32off",
        "atom_motion_diagnostics": "CPU copy after each update; instrumented wall time includes sampling. Use separate uninstrumented runs for primary time/memory claims."
        if motion is not None
        else "disabled",
        "diagnostics": {},
        "timing_samples_seconds": [],
    }
    summary["peak_allocated_bytes"] = 0
    summary["peak_reserved_bytes"] = 0
    if field_recipe is not None:
        summary["optimizer_variant"] = field_recipe.manifest()
    curves = (args.output / "curves.jsonl").open("x")

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def log(entry):
        curves.write(json.dumps(entry) + "\n")
        curves.flush()
        print(json.dumps(entry), flush=True)

    def checkpoint(*, milestone=False):
        target = args.output / "checkpoint.pt"
        temporary = args.output / "checkpoint.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(config),
                "kind": args.kind,
                "seed": args.seed,
                "updates": summary["completed_updates"],
                "planned_horizon_tokens": args.schedule_tokens,
                "completed_tokens": summary["completed_tokens"],
                "data_manifest_sha256": summary["data_manifest_sha256"],
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
                **(
                    {"optimizer_variant": field_recipe.manifest()}
                    if field_recipe is not None
                    else {}
                ),
            },
            temporary,
        )
        temporary.replace(target)
        summary["checkpoint_sha256"] = sha(target)
        if milestone:
            name = f"checkpoint-{summary['completed_tokens']}-tokens.pt"
            destination = args.output / name
            if not destination.exists():
                os.link(target, destination)
            assert sha(destination) == summary["checkpoint_sha256"]
            summary["milestone_checkpoint_sha256"][name] = summary["checkpoint_sha256"]

    try:
        summary["diagnostics"]["initial"] = probe(model, px, py)
        initial = evaluate(model, data["development"], args.batch)
        log(
            {
                "step": 0,
                "seen_tokens": 0,
                "elapsed_seconds": time.monotonic() - began,
                "development": initial,
            }
        )
        summary["initial_development"] = initial
        if motion is not None:
            summary["atom_geometry_history"] = {"0": geometry_snapshot(model)}
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        setup_end = time.monotonic()
        summary["setup_seconds"] = setup_end - began
        for step in range(1, updates + 1):
            if time.monotonic() - began > args.max_wall_seconds:
                raise TimeoutError(
                    "bounded run wall budget reached; partial state preserved"
                )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, horizon_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            loss_sum = torch.zeros((), device="cuda")
            for micro in range(args.accumulation):
                index = (
                    micro
                    if args.repeat_fixed_batches
                    else ((step - 1) * args.accumulation + micro)
                ) * args.batch
                starts = order[index : index + args.batch] * config.context
                x, y = token_batch(data["train"], starts, config.context, "cuda")
                _, loss = model(x, y)
                (loss / args.accumulation).backward()
                loss_sum += loss.detach()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.monotonic() - tick
            summary["peak_allocated_bytes"] = max(
                summary["peak_allocated_bytes"], torch.cuda.max_memory_allocated()
            )
            summary["peak_reserved_bytes"] = max(
                summary["peak_reserved_bytes"], torch.cuda.max_memory_reserved()
            )
            if motion is not None:
                motion.update()
            value = float(loss_sum / args.accumulation)
            if not math.isfinite(value):
                raise FloatingPointError("nonfinite training loss")
            summary["completed_updates"] = step
            summary["completed_tokens"] = step * tokens_per_update
            if step > 10:
                summary["timing_samples_seconds"].append(duration)
            if step <= 5 or step % 20 == 0 or step == updates:
                log(
                    {
                        "step": step,
                        "seen_tokens": step * tokens_per_update,
                        "elapsed_seconds": time.monotonic() - began,
                        "train_ce": value,
                        "lr": lr,
                        "optimizer_group_lrs": [
                            float(group["lr"]) for group in optimizer.param_groups
                        ],
                        "gradient_norm_before_clip": float(norm),
                        "complete_update_seconds": duration,
                        **(
                            {"amplitude_motion": motion.report()}
                            if motion is not None
                            else {}
                        ),
                    }
                )
            if step in (10, 100):
                summary["diagnostics"][str(step)] = probe(model, px, py)
                torch.cuda.empty_cache()
            if step % 200 == 0 or step == updates or step in milestones:
                dev = evaluate(model, data["development"], args.batch)
                log(
                    {
                        "step": step,
                        "seen_tokens": step * tokens_per_update,
                        "elapsed_seconds": time.monotonic() - began,
                        "development": dev,
                    }
                )
                summary["latest_development"] = dev
                if motion is not None:
                    summary["atom_geometry_history"][str(step)] = geometry_snapshot(
                        model
                    )
                checkpoint(milestone=step in milestones)
                save()
        summary["diagnostics"]["final"] = probe(model, px, py)
        if motion is not None:
            summary["amplitude_motion"] = motion.report()
        if args.repeat_fixed_batches:
            summary["scope"] = (
                "repeated fixed-batch diagnostic; confirmation and generations not evaluated"
            )
        else:
            summary["confirmation"] = evaluate(model, data["confirmation"], args.batch)
            summary["generations"] = generate(model)
        summary["state"] = "completed_execution_not_scientific_admission"
        checkpoint()
        save()
    except Exception as error:
        summary["state"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        checkpoint()
        save()
        raise
    finally:
        summary["total_elapsed_seconds"] = time.monotonic() - began
        save()
        curves.close()


if __name__ == "__main__":
    main()
