"""Bounded field-rate continuation screen from the same audited task state."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

RECIPES = {
    "control32": (32, 32, 32, 32),
    "all8": (8, 8, 8, 8),
    "all16": (16, 16, 16, 16),
    "all64": (64, 64, 64, 64),
    "amplitude8": (8, 32, 32, 32),
    "geometry8": (32, 8, 8, 8),
    "fraction8": (32, 8, 32, 32),
    "centers8": (32, 32, 8, 8),
}
FIELDS = ("amplitude", "radius_fraction", "output_center", "input_center")


def read(path):
    return json.loads(path.read_text())


def configure_rates(optimizer, recipe):
    if recipe is None:
        return
    for group in optimizer.param_groups:
        if group.get("field_lrs"):
            group["field_lrs"] = {n: 0.0003 * r for n, r in zip(FIELDS, recipe)}
    expected = [{}, dict(zip(FIELDS, (0.0003 * r for r in recipe)))]
    # Check actual scheduled rates below; this also validates all declared names.
    observed = optimizer.get_field_lrs()
    assert len(observed) == len(expected)


def train_step(model, optimizer, tokens, order, step, plan):
    from cst_research.experiments import ffn_dst
    from cst_research.experiments.ffn_contract import learning_rate
    from cst_research.experiments.ffn_run import token_batch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step, plan.schedule_steps) * group.get(
            "lr_scale", 1
        )
    total = 0.0
    for micro in range(4):
        start = ((step - 1) * 4 + micro) * 2
        x, y = token_batch(tokens, order[start : start + 2] * 128, 128, "cuda")
        _, loss = model(x, y)
        (loss / 4).backward()
        total += float(loss.detach())
    if model.kind == "dst":
        event = ffn_dst.finish_step(model, optimizer, step, plan.topology_steps)
    else:
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        event = {"topology": None, "gradient_norm_before_clip": float(norm)}
    assert math.isfinite(total)
    return total / 4, event


def main():
    from cst_research.experiments.ffn_contract import ModelConfig, learning_rate
    from cst_research.experiments.ffn_model import configure_native
    from cst_research.experiments.ffn_radius_fraction import snapshot
    from cst_research.experiments.ffn_radius_fraction_qualification import (
        assert_model_resume,
        assert_nested,
    )
    from cst_research.experiments.ffn_run import sha, verify_data
    from cst_research.experiments.ffn_selective import make_model, make_optimizer
    from cst_research.experiments.ffn_selective_run import development, oracles
    from cst_research.experiments.ffn_training_plan import TrainingPlan

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = read(args.base_run / "receipt.json")
    assert parent["state"] == "passed" and parent["source_unchanged"]
    assert torch.cuda.get_device_name(0) == parent["gpu"] == "NVIDIA GeForce RTX 4090"
    assert torch.__version__ == parent["torch"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_root = Path(parent["protocol"]["data"])
    assert (
        sha(data_root / "manifest.json") == parent["protocol"]["data_manifest_sha256"]
    )
    _, data = verify_data(data_root)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    plan = TrainingPlan(5000000, 20000000, 20000000)
    family = [("dense-control", "dense", None), ("dst-control", "dst-all", None)] + [
        ("cst-" + name, "cst-all", recipe) for name, recipe in RECIPES.items()
    ]
    result = {
        "state": "running",
        "script_sha256": sha(Path(__file__)),
        "parent_source_archive_sha256": parent["protocol"]["source_archive_sha256"],
        "data_manifest_sha256": parent["protocol"]["data_manifest_sha256"],
        "data_order_sha256": __import__("hashlib").sha256(order.tobytes()).hexdigest(),
        "start_step": 976,
        "end_step": 1488,
        "results": [],
        "scope": "One-state512-update DEV screening; all candidates retained; no confirmation or from-scratch optimum claim",
    }
    target = args.output / "summary.json"

    def save():
        target.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for label, arm, recipe in family:
            directory = args.output / label
            directory.mkdir()
            s = read(args.base_run / arm / "summary.json")
            checkpoint = args.base_run / arm / "checkpoint-999424-tokens.pt"
            assert sha(checkpoint) == s["milestone_checkpoint_sha256"][checkpoint.name]
            saved = torch.load(
                checkpoint, weights_only=False, map_location="cpu", mmap=True
            )
            config = ModelConfig(**s["config"])

            def restored(
                state=saved,
                config=config,
                kind=s["kind"],
                sites=tuple(s["sites"]),
                recipe=recipe,
            ):
                m = make_model(config, kind, sites, 17).cuda()
                m.load_state_dict(state["model"])
                if m.kind == "cst":
                    configure_native(m)
                o = make_optimizer(m)
                o.load_state_dict(state["optimizer"])
                configure_rates(o, recipe)
                return m, o

            # One resumed update must survive serialization with the new public recipe.
            model, optimizer = restored()
            train_step(model, optimizer, data["train"], order, 977, plan)
            state = {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
            other, oo = restored(state)
            train_step(model, optimizer, data["train"], order, 978, plan)
            train_step(other, oo, data["train"], order, 978, plan)
            assert_model_resume(model, other)
            assert_nested(optimizer.state_dict(), oo.state_dict())
            del model, optimizer, other, oo, state
            model, optimizer = restored()
            torch.set_rng_state(saved["torch_rng_state"])
            torch.cuda.set_rng_state(saved["cuda_rng_state"])
            baseline = development(model, data["development"])
            old_curves = [
                json.loads(v)
                for v in (args.base_run / arm / "curves.jsonl").read_text().splitlines()
            ]
            old_metrics = {
                v["step"]: v["development"] for v in old_curves if "development" in v
            }
            assert baseline == old_metrics[976]
            row = {
                "label": label,
                "parent_arm": arm,
                "field_multipliers": dict(zip(FIELDS, recipe)) if recipe else None,
                "checkpoint_sha256": s["milestone_checkpoint_sha256"][checkpoint.name],
                "serialized_update_exact": True,
                "baseline_development": baseline,
                "development": [],
                "geometry": [],
                "actual_adam_updates": saved["actual_adam_updates"],
            }
            began = time.monotonic()
            with (directory / "curves.jsonl").open("x") as log:
                for step in range(977, 1489):
                    if time.monotonic() - began > 1500:
                        raise TimeoutError("bounded continuation arm expired")
                    ce, event = train_step(
                        model, optimizer, data["train"], order, step, plan
                    )
                    if event["topology"] is None:
                        row["actual_adam_updates"] += 1
                    if step % 20 == 0 or step == 1488:
                        point = {
                            "step": step,
                            "train_ce": ce,
                            "gradient_norm_before_clip": event[
                                "gradient_norm_before_clip"
                            ],
                            "topology": event["topology"] is not None,
                        }
                        if recipe:
                            observed = optimizer.get_field_lrs()[1]
                            for name, multiplier in zip(FIELDS, recipe):
                                assert math.isclose(
                                    observed[name],
                                    learning_rate(step, plan.schedule_steps)
                                    * multiplier,
                                    rel_tol=1e-12,
                                )
                            point["effective_field_lrs"] = observed
                        log.write(json.dumps(point) + "\n")
                        log.flush()
                    if step in (1200, 1400, 1488):
                        metrics = development(model, data["development"])
                        if (
                            label in ("dense-control", "dst-control", "cst-control32")
                            and step in old_metrics
                        ):
                            assert metrics == old_metrics[step], (
                                label,
                                step,
                                metrics,
                                old_metrics[step],
                            )
                        row["development"].append({"step": step, **metrics})
                        if recipe:
                            row["geometry"].append(
                                {"step": step, "layers": snapshot(model)}
                            )
            expected = plan.adam_updates(1488) if model.kind == "dst" else 1488
            assert row["actual_adam_updates"] == expected
            for optstate in optimizer.state.values():
                assert int(optstate["step"]) == expected
                for key in ("exp_avg", "exp_avg_sq"):
                    assert (
                        optstate[key].dtype == torch.float32
                        and torch.isfinite(optstate[key]).all()
                    )
            path = directory / "checkpoint-1523712-tokens.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": 1488,
                    "field_multipliers": row["field_multipliers"],
                    "data_manifest_sha256": s["data_manifest_sha256"],
                    "config": s["config"],
                    "parent_checkpoint_sha256": row["checkpoint_sha256"],
                },
                path,
            )
            final = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
            other, oo = restored(final)
            assert_model_resume(model, other)
            assert_nested(optimizer.state_dict(), oo.state_dict())
            audited = development(other, data["development"])
            assert audited == {
                k: v for k, v in row["development"][-1].items() if k != "step"
            }
            row["oracles"] = oracles(other)
            del other, oo, final
            row.update(
                state="passed",
                final_checkpoint_sha256=sha(path),
                elapsed_seconds=time.monotonic() - began,
                optimizer_audit_passed=True,
            )
            (directory / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
            result["results"].append(row)
            save()
            print("CONTINUATION PASS", label, row["development"][-1]["ce"], flush=True)
            del model, optimizer, saved, restored
            torch.cuda.empty_cache()
        assert len(result["results"]) == 10
        result["state"] = "passed"
        save()
    except BaseException as e:
        result.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main()
