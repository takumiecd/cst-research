"""Train a QKV physical decoder with paired 32-update reward evaluations."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

if __package__:
    from .qkv_shared_decoder_run import (
        attach,
        batches_for,
        load_reference,
        make_optimizer,
        sha,
    )
else:
    from qkv_shared_decoder_run import (
        attach,
        batches_for,
        load_reference,
        make_optimizer,
        sha,
    )


@torch.no_grad()
def ce(model, batches):
    model.eval()
    value = sum(float(model(*batch)[1]) for batch in batches) / len(batches)
    if not math.isfinite(value):
        raise ValueError("Non-finite evaluation CE")
    return value


def reward_gradient(plus_ce, minus_ce, noise, sigma):
    """Estimate ascent on expected negative CE under Gaussian perturbations."""
    return ((minus_ce - plus_ce)[:, None] * noise).mean(0) / (2 * sigma)


class Branches:
    def __init__(self, model, projections, optimizer, core, shared, checkpoint):
        self.model = model
        self.projections = projections
        self.optimizer = optimizer
        self.core = core
        self.shared = shared
        self.checkpoint = checkpoint

    @torch.no_grad()
    def reset(self, weights):
        from copy import deepcopy

        self.optimizer.zero_grad(set_to_none=True)
        self.model.load_state_dict(self.checkpoint["model"])
        self.optimizer.load_state_dict(deepcopy(self.checkpoint["optimizer"]))
        self.put(weights)

    @torch.no_grad()
    def put(self, weights):
        # vector_to_parameters aliases its input storage; protect the candidate.
        vector_to_parameters(weights.clone(), self.shared)

    def weights(self):
        return parameters_to_vector(self.shared).detach()

    def update(self, batches, clocks, *, core, shared):
        from cst_research.experiments.ffn_contract import learning_rate

        clock, schedule = clocks
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.core:
            p.requires_grad_(core)
        for p in self.shared:
            p.requires_grad_(shared)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(clock, schedule) * group["lr_scale"]
        total = 0.0
        for batch in batches:
            loss = self.model(*batch)[1] / len(batches)
            loss.backward()
            total += float(loss.detach())
        if core:
            torch.nn.utils.clip_grad_norm_(self.core, 1.0, error_if_nonfinite=True)
        if shared:
            torch.nn.utils.clip_grad_norm_(self.shared, 0.1, error_if_nonfinite=True)
        self.optimizer.step()
        if core:
            with torch.no_grad():
                for projection in self.projections.values():
                    latent = projection.map.layer.atoms.p
                    latent[:, 0].clamp_(-32.0, 32.0)
                    latent[:, 1].clamp_(math.log(0.5), math.log(512.0))
        return total

    def rollout(
        self, weights, updates, validation, start, schedule, *, joint_adam=False
    ):
        self.reset(weights)
        scores = {"0": ce(self.model, validation)}
        for offset, batches in enumerate(updates, start=1):
            self.update(
                batches, (start + offset, schedule), core=True, shared=joint_adam
            )
            if offset in (8, 32):
                scores[str(offset)] = ce(self.model, validation)
        final = self.weights()
        if not joint_adam:
            assert torch.equal(final, weights)
        return scores, final


def evaluation_batches(train, tokens, start, length):
    return [
        train.token_batch(tokens, [i, i + 128], 128, "cuda")
        for i in range(start, start + length, 256)
    ]


def run(args, train, record, protocol, report):
    from torchcst._backends.torch.parameterizations import radius_envelope

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name() == record["gpu"]
    assert torch.__version__ == record["torch"] and torch.version.cuda == record["cuda"]
    data_dir = Path(protocol["data"])
    assert sha(data_dir / "manifest.json") == record["data_manifest_sha256"]
    _, data = train.verify_data(data_dir)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    import hashlib

    assert hashlib.sha256(order.tobytes()).hexdigest() == record["data_order_sha256"]
    anchor_report = json.loads((args.checkpoint_dir / "result.json").read_text())
    path = args.checkpoint_dir / f"decoder-{args.step}.pt"
    digest = sha(path)
    assert anchor_report["transform"] == "direct physical"
    assert anchor_report["checkpoints"][str(args.step)] == digest
    assert (
        anchor_report["runner_sha256"] != ""
        and anchor_report["target_steps"] >= args.step
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert checkpoint["step"] == args.step
    for key in ("data_manifest_sha256", "data_order_sha256"):
        assert checkpoint[key] == anchor_report[key] == record[key]
    assert all(
        int(s["step"]) == args.step for s in checkpoint["optimizer"]["state"].values()
    )
    report.update(
        checkpoint_sha256=digest,
        data_manifest_sha256=record["data_manifest_sha256"],
        data_order_sha256=record["data_order_sha256"],
        gpu=record["gpu"],
    )
    train.write(args.output / "result.json", report)
    print("CHECKPOINT_VERIFIED", args.step, digest, flush=True)

    os.environ["CST_RESEARCH_RADIUS_CAP"] = "512"
    model = train.model_for()
    assert train.parameter_digest(model) == record["initial_parameters_sha256"]
    radius_envelope.USE_RATIONAL_LOWER = True
    model.cuda()
    projections = train.projections(model)
    scale = attach(model, projections, record["protocol"]["seed"], True)
    assert checkpoint["amplitude_scale"] == scale
    optimizer, core, shared = make_optimizer(model, projections)
    branches = Branches(model, projections, optimizer, core, shared, checkpoint)
    model.load_state_dict(checkpoint["model"])
    base = branches.weights()
    branches.reset(base)
    rng = torch.Generator().manual_seed(217)
    samples = {
        site: projection.map.layer.atoms.p.detach().cpu()[
            torch.randperm(len(projection.map.layer.atoms.p), generator=rng)[:256]
        ]
        for site, projection in projections.items()
    }
    schedule = train.SCHEDULE_STEPS
    head = model.decoder.network[-1]
    head_size = head.weight.numel() + head.bias.numel()
    assert tuple(shared[-1].shape) == tuple(head.bias.shape) and shared[-1] is head.bias
    center = torch.nn.Parameter(base[-head_size:].clone())
    outer = torch.optim.Adam([center], lr=args.reward_lr)
    random = torch.Generator(device=base.device).manual_seed(6517)
    report["state"] = "running"
    report["head_parameters"] = head_size
    report["base_parameters"] = base.numel()
    report["horizons"] = [0, 8, 32]
    report["validation_tokens"] = {"reward": 1024, "confirmation": 4096}
    train.write(args.output / "result.json", report)

    def save_function(name, weights, score):
        branches.put(weights)
        target = args.output / f"function-{name}.pt"
        tmp = target.with_suffix(".tmp")
        torch.save(
            {
                "step": args.step,
                "decoder": {
                    key: value.detach().cpu()
                    for key, value in model.decoder.state_dict().items()
                },
                "latents": samples,
                "amplitude_scale": scale,
                "period": model.decoder.period,
                "development": score,
                "label": name,
            },
            tmp,
        )
        tmp.replace(target)
        report["function_snapshots"][name] = sha(target)

    def write_branch(collection, name, scores):
        collection[name] = scores
        train.write(args.output / "result.json", report)
        print("CE", name, scores, flush=True)

    train_source = [
        [
            batches_for(train, data, order, args.step + offset + j * 64)
            for offset in range(1, 33)
        ]
        for j in range(args.rounds + 1)
    ]
    with (args.checkpoint_dir / "curve.jsonl").open() as curve:
        expected = next(
            row for line in curve if (row := json.loads(line))["step"] == args.step + 1
        )
    branches.reset(base)
    replay = branches.update(
        train_source[0][0], (args.step + 1, schedule), core=True, shared=True
    )
    report["anchor_replay_ce_difference"] = replay - expected["train_ce"]
    assert abs(report["anchor_replay_ce_difference"]) <= 2e-5
    pure_adam = branches.weights()
    print(
        "ANCHOR_REPLAY",
        args.step + 1,
        report["anchor_replay_ce_difference"],
        flush=True,
    )
    original_weights = (base, pure_adam)
    for index in range(args.rounds):
        batches = train_source[index]
        selection = evaluation_batches(
            train, data["development"], 512 + index * 1536, 1024
        )
        current = torch.cat((base[:-head_size], center.detach()))
        outcomes = {}
        round_report = {"index": index, "candidates": outcomes}
        report["rounds"].append(round_report)
        scores, _ = branches.rollout(current, batches, selection, args.step, schedule)
        if index == 0:
            save_function("anchor", base, scores["0"])
        write_branch(outcomes, "control", scores)
        if index == 0:
            scores, _ = branches.rollout(
                pure_adam, batches, selection, args.step, schedule
            )
            save_function("candidate_adam_1", pure_adam, scores["32"])
            write_branch(outcomes, "ce_adam_1", scores)
        eps = torch.randn(
            args.directions, head_size, device=base.device, generator=random
        )
        plus, minus = [], []
        for j, epsilon in enumerate(eps):
            for sign, collection in ((1, plus), (-1, minus)):
                shifted = torch.cat(
                    (base[:-head_size], center.detach() + sign * args.sigma * epsilon)
                )
                scores, _ = branches.rollout(
                    shifted, batches, selection, args.step, schedule
                )
                collection.append(scores["32"])
                write_branch(
                    outcomes, f"pair_{j}_{'plus' if sign > 0 else 'minus'}", scores
                )
        gradient = reward_gradient(
            base.new_tensor(plus), base.new_tensor(minus), eps, args.sigma
        )
        outer.zero_grad(set_to_none=True)
        center.grad = -gradient
        outer.step()
        round_report["reward_gradient_norm"] = float(gradient.norm())
        round_report["head_distance_from_anchor"] = float(
            (center.detach() - base[-head_size:]).norm()
        )
        weights = torch.cat((base[:-head_size], center.detach()))
        scores, _ = branches.rollout(weights, batches, selection, args.step, schedule)
        save_function(f"es_round_{index + 1}", weights, scores["32"])
        write_branch(outcomes, "updated_es", scores)
        torch.save(
            {
                "center": center.detach().cpu(),
                "outer": outer.state_dict(),
                "round": index,
            },
            args.output / f"reward-state-{index}.pt",
        )

    final = report["confirmation"] = {}
    hold = evaluation_batches(train, data["confirmation"], 1536, 4096)
    final_weights = dict(zip(("control", "ce_adam_1"), original_weights, strict=True))
    final_weights["trained_es"] = torch.cat((base[:-head_size], center.detach()))
    for name, weights in final_weights.items():
        scores, _ = branches.rollout(
            weights, train_source[-1], hold, args.step, schedule
        )
        write_branch(final, name, scores)
    scores, ordinary = branches.rollout(
        base, train_source[-1], hold, args.step, schedule, joint_adam=True
    )
    write_branch(final, "joint_adam_32", scores)
    final_weights["joint_adam_32"] = ordinary
    for name, weights in final_weights.items():
        save_function(name, weights, final[name]["32"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-changes", type=Path, required=True, action="append")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, choices=(2, 3), default=2)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--reward-lr", type=float, default=0.001)
    args = parser.parse_args()
    if not (1 <= args.directions <= 16 and args.sigma > 0 and args.reward_lr > 0):
        parser.error(
            "require a positive perturbation scale, learning rate and 1–16 directions"
        )
    train, record, protocol = load_reference(args)
    args.output.mkdir()
    report = {
        "state": "initializing",
        "step": args.step,
        "sigma": args.sigma,
        "directions": args.directions,
        "reward_lr": args.reward_lr,
        "rounds": [],
        "function_snapshots": {},
        "runner_sha256": sha(Path(__file__)),
    }
    target = args.output / "result.json"
    train.write(target, report)
    try:
        run(args, train, record, protocol, report)
        report["state"] = "completed"
    except BaseException as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        train.write(target, report)


if __name__ == "__main__":
    main()
