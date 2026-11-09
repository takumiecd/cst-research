"""Task-gradient alignment and finite-step interaction at immutable checkpoints.

Only temporary model/optimizer copies are updated. No Dense teacher is used.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ARMS = (
    "dense",
    "cst-up",
    "cst-two_blocks",
    "cst-scattered_mixed",
    "cst-all",
    "dst-all",
)
STEPS = (976, 4882)
OFFSETS = (0, 32, 64)
FIELDS = ("amplitude", "radius_fraction", "output_center", "input_center")


def read(path):
    return json.loads(path.read_text())


def clone_state(model):
    return {n: x.detach().clone() for n, x in model.named_parameters()}


def load_optimizer_snapshot(optimizer, state):
    """load_state_dict may alias CPU counters and same-device moment tensors."""
    optimizer.load_state_dict(copy.deepcopy(state))


def validate_regular_buffers(model, before, raw_gradients):
    """DST's declared nonpersistent gradient scratch changes; masks must not."""
    from cst_research.experiments.ffn_dst import MaskedLinear
    from cst_research.experiments.ffn_radius_fraction_qualification import assert_nested

    scratch = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskedLinear):
            assert "dense_gradient" in module._non_persistent_buffers_set
            scratch[name + ".dense_gradient"] = raw_gradients[name + ".weight"]
    current = dict(model.named_buffers())
    assert_nested(
        {n: b for n, b in current.items() if n not in scratch},
        {n: b for n, b in before.items() if n not in scratch},
    )
    for name, gradient in scratch.items():
        assert_nested(current[name], gradient)


def partition(model):
    groups = {f"{i}.{d}": [] for i in range(len(model.blocks)) for d in ("up", "down")}
    groups["remainder"] = []
    for name, _ in model.named_parameters():
        matched = False
        for key, names in groups.items():
            if key == "remainder":
                continue
            i, direction = key.split(".")
            if name.startswith(f"blocks.{i}.ffn.{direction}."):
                names.append(name)
                matched = True
                break
        if not matched:
            groups["remainder"].append(name)
    assert sum(map(len, groups.values())) == len(list(model.parameters()))
    return groups


@torch.no_grad()
def loss_on(model, batches):
    return sum(float(model(x, y)[1]) for x, y in batches) / len(batches)


@torch.no_grad()
def endpoint(model, old, new, names, *, field=None):
    """Restore the same pre-update state, then install one named endpoint subset."""
    params = dict(model.named_parameters())
    for name, p in params.items():
        p.copy_(old[name])
    for name in names:
        if field is None:
            params[name].copy_(new[name])
        elif name.endswith("atoms.p"):
            params[name][:, field].copy_(new[name][:, field])


def directional_statistics(gradients, deltas):
    dot = sum(float((g.double() * d.double()).sum()) for g, d in zip(gradients, deltas))
    gg = sum(float(g.double().square().sum()) for g in gradients)
    dd = sum(float(d.double().square().sum()) for d in deltas)
    return {
        "predicted_loss_reduction": -dot,
        "descent_cosine": -dot / math.sqrt(gg * dd) if gg > 0 and dd > 0 else None,
        "gradient_norm": math.sqrt(gg),
        "delta_norm": math.sqrt(dd),
    }


def accumulate(model, optimizer, batches, *, capture=False):
    from cst_research.experiments.ffn_selective import projections

    optimizer.zero_grad(set_to_none=True)
    records = {n: [] for n in projections(model)}
    handles = []
    if capture:
        for name, layer in projections(model).items():

            def hook(_module, inputs, output, name=name):
                row = {
                    "input": inputs[0].detach().clone(),
                    "output": output.detach().clone(),
                }
                records[name].append(row)

                def grad(g):
                    row["gradient"] = g.detach().clone()

                output.register_hook(grad)

            handles.append(layer.register_forward_hook(hook))
    try:
        total = 0.0
        for x, y in batches:
            _, loss = model(x, y)
            (loss / len(batches)).backward()
            total += float(loss.detach())
        return total / len(batches), records
    finally:
        for h in handles:
            h.remove()


def finish(model, optimizer, step):
    from cst_research.experiments import ffn_dst
    from cst_research.experiments.ffn_training_plan import TrainingPlan

    plan = TrainingPlan(5000000, 20000000, 20000000)
    assert not ffn_dst.is_topology_step(step, plan.topology_steps)
    if model.kind == "dst":
        event = ffn_dst.finish_step(model, optimizer, step, plan.topology_steps)
        assert event["topology"] is None
        return event["gradient_norm_before_clip"]
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return float(norm)


def diagnose(model, optimizer, batches, step, *, replay_check=False):
    from cst_research.experiments.ffn_radius_fraction_qualification import assert_nested
    from cst_research.experiments.ffn_selective import projections

    model.train()
    old = clone_state(model)
    buffers = {n: x.detach().clone() for n, x in model.named_buffers()}
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    cpu_rng, cuda_rng = (
        torch.get_rng_state().clone(),
        torch.cuda.get_rng_state().clone(),
    )
    groups = partition(model)
    before, records = accumulate(model, optimizer, batches, capture=True)
    raw = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    norm = finish(model, optimizer, step)
    new = clone_state(model)
    new_optimizer = copy.deepcopy(optimizer.state_dict())
    # Persistent masks stay fixed; declared DST scratch retains its raw gradient.
    params = dict(model.named_parameters())
    validate_regular_buffers(model, buffers, raw)
    result = {"before_ce": before, "gradient_norm_before_clip": norm, "maps": {}}
    for name, layer in projections(model).items():
        deltas, gradients = [], []
        with torch.no_grad():
            for row in records[name]:
                deltas.append(layer(row["input"]) - row["output"])
                gradients.append(row["gradient"])
        result["maps"][name] = directional_statistics(gradients, deltas)
        result["maps"][name]["relative_output_delta"] = math.sqrt(
            sum(float(d.double().square().sum()) for d in deltas)
            / max(
                sum(float(r["output"].double().square().sum()) for r in records[name]),
                1e-30,
            )
        )
    names = list(params)
    sets = {
        **groups,
        "all_ffn": [n for k, ns in groups.items() if k != "remainder" for n in ns],
        "all": names,
    }
    sets.update(
        {
            "all_" + d: [
                n for k, ns in groups.items() if k.endswith("." + d) for n in ns
            ]
            for d in ("up", "down")
        }
    )
    result["endpoint_ce"] = {}
    try:
        endpoint(model, old, new, [])
        assert abs(loss_on(model, batches) - before) <= 1e-6
        for label, selected in sets.items():
            endpoint(model, old, new, selected)
            result["endpoint_ce"][label] = loss_on(model, batches)
        if model.kind == "cst":
            result["fields"] = {}
            atom_names = [n for n in params if n.endswith("atoms.p")]
            for column, field in enumerate(FIELDS):
                endpoint(model, old, new, atom_names, field=column)
                per_map = {}
                for name in atom_names:
                    p, q, g = (
                        old[name][:, column],
                        new[name][:, column],
                        raw[name][:, column],
                    )
                    delta = q - p
                    if column >= 2:
                        # Flat-torus chart; use local wrapped displacement for the tangent.
                        i = int(name.split(".")[1])
                        d = name.split(".")[3]
                        layer = getattr(model.blocks[i].ffn, d).map
                        period = (
                            layer.out_features if column == 2 else layer.in_features
                        )
                        delta = (delta + period / 2).remainder(period) - period / 2
                    per_map[name] = directional_statistics([g], [delta]) | {
                        "unchanged_fraction": float((q == p).float().mean()),
                        "fraction_at_zero": float((p == 0).float().mean())
                        if column == 1
                        else None,
                        "fraction_at_one": float((p == 1).float().mean())
                        if column == 1
                        else None,
                        "gradient_rms": float(g.double().square().mean().sqrt()),
                    }
                result["fields"][field] = {
                    "only_field_ce": loss_on(model, batches),
                    "parameters": per_map,
                }
        gains = {n: before - ce for n, ce in result["endpoint_ce"].items()}
        result["loss_reductions"] = gains
        result["ffn_joint_minus_sum_individual_gain"] = gains["all_ffn"] - sum(
            gains[n] for n in groups if n != "remainder"
        )
        result["all_joint_minus_sum_partition_gain"] = gains["all"] - sum(
            gains[n] for n in groups
        )
        # A finite-difference check on the independent Dense remainder partition.
        remainder = groups["remainder"]
        pred = directional_statistics(
            [raw[n] for n in remainder], [new[n] - old[n] for n in remainder]
        )
        epsilon = 0.01
        endpoint(model, old, new, [])
        with torch.no_grad():
            for n in remainder:
                params[n].copy_(old[n] + epsilon * (new[n] - old[n]))
        fd = (before - loss_on(model, batches)) / epsilon
        result["remainder_directional_check"] = {
            **pred,
            "finite_difference_reduction": fd,
            "epsilon": epsilon,
        }
        assert abs(fd - pred["predicted_loss_reduction"]) <= max(
            2e-4, 0.12 * abs(pred["predicted_loss_reduction"])
        )
        if replay_check:
            endpoint(model, old, new, [])
            load_optimizer_snapshot(optimizer, old_optimizer)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            unhooked, _ = accumulate(model, optimizer, batches)
            assert abs(unhooked - before) <= 1e-7
            finish(model, optimizer, step)
            assert_nested(dict(model.named_parameters()), new)
            assert_nested(optimizer.state_dict(), new_optimizer)
            result["instrumented_step_equals_unhooked_step"] = True
    finally:
        endpoint(model, old, new, [])
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
        load_optimizer_snapshot(optimizer, old_optimizer)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    assert_nested(dict(model.named_parameters()), old)
    assert_nested(dict(model.named_buffers()), buffers)
    assert_nested(optimizer.state_dict(), old_optimizer)
    result["model_optimizer_rng_restored"] = True
    return result


def main():
    from cst_research.experiments.ffn_contract import ModelConfig, learning_rate
    from cst_research.experiments.ffn_model import configure_native
    from cst_research.experiments.ffn_run import sha, token_batch, verify_data
    from cst_research.experiments.ffn_selective import make_model, make_optimizer
    from cst_research.experiments.ffn_training_plan import TrainingPlan

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base-run", type=Path, required=True)
    p.add_argument("--placement-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = [read(r / "receipt.json") for r in (args.base_run, args.placement_run)]
    assert all(r["state"] == "passed" and r["source_unchanged"] for r in receipts)
    assert all(receipts[0][k] == receipts[1][k] for k in ("gpu", "torch", "cuda"))
    assert (
        torch.cuda.get_device_name(0) == receipts[0]["gpu"] == "NVIDIA GeForce RTX 4090"
    )
    assert torch.__version__ == receipts[0]["torch"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_root = Path(receipts[0]["protocol"]["data"])
    assert (
        sha(data_root / "manifest.json")
        == receipts[0]["protocol"]["data_manifest_sha256"]
    )
    _, data = verify_data(data_root)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    plan = TrainingPlan(5000000, 20000000, 20000000)
    result = {
        "state": "running",
        "evaluator_sha256": sha(Path(__file__)),
        "arms": list(ARMS),
        "checkpoint_steps": list(STEPS),
        "batch_offsets": list(OFFSETS),
        "records": [],
        "scope": "Restored optimizer one-step counterfactuals on original TRAIN targets; no new training trajectory, teacher or confirmation",
    }
    target = args.output / "summary.json"

    def save():
        target.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for arm in ARMS:
            root = args.placement_run if "scattered" in arm else args.base_run
            s = read(root / arm / "summary.json")
            assert s["data_order_sha256"] == hashlib.sha256(order.tobytes()).hexdigest()
            for step in STEPS:
                path = root / arm / f"checkpoint-{step * 1024}-tokens.pt"
                assert sha(path) == s["milestone_checkpoint_sha256"][path.name]
                saved = torch.load(
                    path, weights_only=False, map_location="cpu", mmap=True
                )
                for offset in OFFSETS:
                    model = make_model(
                        ModelConfig(**s["config"]), s["kind"], tuple(s["sites"]), 17
                    ).cuda()
                    model.load_state_dict(saved["model"])
                    if s["kind"] == "cst":
                        configure_native(model)
                    optimizer = make_optimizer(model)
                    load_optimizer_snapshot(optimizer, saved["optimizer"])
                    assert all(
                        int(s["step"]) == saved["actual_adam_updates"]
                        for s in optimizer.state.values()
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate(
                            step + 1, plan.schedule_steps
                        ) * group.get("lr_scale", 1)
                    batches = []
                    for micro in range(4):
                        start = ((step + offset) * 4 + micro) * 2
                        batches.append(
                            token_batch(
                                data["train"],
                                order[start : start + 2] * 128,
                                128,
                                "cuda",
                            )
                        )
                    row = diagnose(
                        model, optimizer, batches, step + 1, replay_check=offset == 0
                    )
                    row.update(
                        arm=arm,
                        checkpoint_step=step,
                        batch_offset_steps=offset,
                        checkpoint_sha256=s["milestone_checkpoint_sha256"][path.name],
                        optimizer_clock_after_restore=sorted(
                            {int(s["step"]) for s in optimizer.state.values()}
                        ),
                    )
                    assert row["optimizer_clock_after_restore"] == [
                        saved["actual_adam_updates"]
                    ]
                    result["records"].append(row)
                    save()
                    print("UPDATE AUDIT PASS", arm, step, offset, flush=True)
                    del model, optimizer, batches
                    torch.cuda.empty_cache()
                del saved
        assert len(result["records"]) == len(ARMS) * len(STEPS) * len(OFFSETS)
        result["state"] = "passed"
        save()
    except BaseException as e:
        result.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main()
