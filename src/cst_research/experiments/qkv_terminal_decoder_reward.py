"""Use terminal CE to improve a frozen physical decoder from matched step-zero runs."""

import argparse
import hashlib
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .match_physical_atom_init import matched_latents
    from .qkv_decoder_reward_run import Branches
    from .qkv_shared_decoder_run import (
        attach,
        batches_for,
        load_reference,
        make_optimizer,
        read,
        sha,
    )
else:
    from match_physical_atom_init import matched_latents
    from qkv_decoder_reward_run import Branches
    from qkv_shared_decoder_run import (
        attach,
        batches_for,
        load_reference,
        make_optimizer,
        read,
        sha,
    )


CHECKPOINT_STEPS = (512, 1536, 2441, 3600, 4882)
EVAL_STEPS = (128, 256, *CHECKPOINT_STEPS)
PHYSICAL_LIMITS = (4e-7, 4e-6, 1.5e-4, 1.5e-4)


class InitialGeometryMismatch(ValueError):
    pass


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    state = model.state_dict()
    for name, value in state.items():
        state[name] = (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else deepcopy(value)
        )
    return state


@torch.no_grad()
def head_direction(decoder, seed):
    layer = decoder.network[-1]
    noise = torch.randn(
        (layer.out_features, layer.in_features + 1),
        generator=torch.Generator().manual_seed(seed),
    )
    return noise[:, :-1].contiguous(), noise[:, -1].contiguous()


@torch.no_grad()
def with_head(decoder, base, direction, amount):
    decoder.load_state_dict(base, strict=True)
    layer = decoder.network[-1]
    for p, d in zip((layer.weight, layer.bias), direction, strict=True):
        p.add_(d.to(p), alpha=amount)


@torch.no_grad()
def physical_error(decoder, latent, expected):
    actual = decoder(latent)
    diff = actual - expected
    diff[:, 1] = (actual[:, 1] / expected[:, 1]).log()
    diff[:, 2:] = (diff[:, 2:] + decoder.period / 2).remainder(
        decoder.period
    ) - decoder.period / 2
    return diff.abs().amax(0)


@torch.no_grad()
def prepare_candidate(decoder, initial, targets, targets_log, coefficient, direction):
    with_head(decoder, initial["decoder"], direction, coefficient)
    double = deepcopy(decoder).double()
    solutions, geometry_error, maximum_jump = {}, {}, {}
    for site, latent in initial["latents"].items():
        z = latent.to(next(decoder.parameters()).device)
        target = targets[site].to(z)
        smooth = targets_log[site].to(device=z.device)
        solution, _, okay = matched_latents(double, smooth, z, decoder, target)
        geometry_error[site] = physical_error(decoder, solution, target).tolist()
        maximum_jump[site] = (solution - z).abs().amax(0).tolist()
        if not okay:
            raise InitialGeometryMismatch(
                f"initial geometry mismatch at {site}: {geometry_error[site]}"
            )
        solutions[site] = solution.cpu()
    return {
        "latents": solutions,
        "decoder": cpu_state(decoder),
        "coefficient": coefficient,
        "maximum_geometry_error": geometry_error,
        "maximum_latent_jump": maximum_jump,
    }


def shape_snapshot(path, decoder, warm, score, step):
    atomic_save(
        {
            "decoder": cpu_state(decoder),
            "latents": warm["latents"],
            "amplitude_scale": warm["amplitude_scale"],
            "period": warm["period"],
            "development": score,
            "step": step,
        },
        path,
    )
    return sha(path)


def train_arm(
    args,
    train,
    record,
    model,
    projections,
    data,
    order,
    common,
    candidate,
    name,
    joint,
    warm,
):
    from cst_research.experiments.ffn_run import evaluate

    root = args.output / name
    root.mkdir(exist_ok=True)
    result_file = root / "result.json"
    arm = (
        read(result_file)
        if result_file.is_file()
        else {
            "completed_steps": 0,
            "state": "initializing",
            "checkpoints": {},
            "evaluations": {},
        }
    )
    if arm["state"] == "completed":
        final = root / f"decoder-{args.steps}.pt"
        assert sha(final) == arm["checkpoints"][str(args.steps)]
        return arm

    model.load_state_dict(common["model"])
    model.decoder.load_state_dict(candidate["decoder"])
    with torch.no_grad():
        for site, projection in projections.items():
            p = projection.map.layer.atoms.p
            p.copy_(candidate["latents"][site])
            err = physical_error(
                model.decoder, p, common["targets"][site].to(p)
            ).tolist()
            assert all(e <= lim for e, lim in zip(err, PHYSICAL_LIMITS, strict=True)), (
                site,
                err,
            )
    live_initial = train.development(model, data["development"])["ce"]
    assert abs(live_initial - common["initial_ce"]) <= 3e-6, (
        name,
        live_initial,
        common["initial_ce"],
    )
    optimizer, core, shared = make_optimizer(model, projections)
    branches = Branches(model, projections, optimizer, core, shared, {})
    start = 0
    if arm["checkpoints"]:
        start = max(map(int, arm["checkpoints"]))
        path = root / f"decoder-{start}.pt"
        assert sha(path) == arm["checkpoints"][str(start)]
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        assert saved["step"] == start
        assert saved["candidate_sha256"] == candidate["sha256"]
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        arm["evaluations"] = {
            clock: value
            for clock, value in arm["evaluations"].items()
            if int(clock) <= start
        }
    else:
        torch.manual_seed(record["protocol"]["seed"])
    assert (
        len(optimizer.state) == 0
        if start == 0
        else all(int(state["step"]) == start for state in optimizer.state.values())
    )
    weights_before = branches.weights()
    arm.update(
        state="running",
        completed_steps=start,
        joint_adam=joint,
        initial_ce=live_initial,
        candidate_sha256=candidate["sha256"],
    )
    train.write(result_file, arm)
    print("ARM_STARTED", name, start, live_initial, flush=True)
    for clock in range(start + 1, args.steps + 1):
        loss = branches.update(
            batches_for(train, data, order, clock),
            (clock, train.SCHEDULE_STEPS),
            core=True,
            shared=joint,
        )
        assert math.isfinite(loss)
        arm["completed_steps"] = clock
        if clock in EVAL_STEPS or clock == args.steps:
            development = train.development(model, data["development"])
            arm["evaluations"][str(clock)] = {
                "train_ce": loss,
                "development": development,
            }
            print("EVAL", name, clock, development["ce"], flush=True)
        if clock in CHECKPOINT_STEPS or clock == args.steps:
            checkpoint = root / f"decoder-{clock}.pt"
            atomic_save(
                {
                    "step": clock,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "candidate_sha256": candidate["sha256"],
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                checkpoint,
            )
            arm["checkpoints"][str(clock)] = sha(checkpoint)
        if clock % 50 == 0 or clock in EVAL_STEPS or clock == args.steps:
            train.write(result_file, arm)
    assert joint or torch.equal(weights_before, branches.weights())
    final = root / f"decoder-{args.steps}.pt"
    assert sha(final) == arm["checkpoints"][str(args.steps)]
    saved = torch.load(final, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    assert all(int(s["step"]) == args.steps for s in optimizer.state.values())
    restored = train.development(model, data["development"])
    assert (
        abs(restored["ce"] - arm["evaluations"][str(args.steps)]["development"]["ce"])
        <= 1e-6
    )
    confirmation = evaluate(model, data["confirmation"], batch=2)
    snapshot = args.output / f"function-{name}.pt"
    arm.update(
        state="completed",
        final_development=restored,
        confirmation=confirmation,
        function_sha256=shape_snapshot(
            snapshot, model.decoder, warm, restored["ce"], args.steps
        ),
    )
    train.write(result_file, arm)
    print("ARM_COMPLETE", name, restored["ce"], confirmation["ce"], flush=True)
    return arm


def run(args, train, record, protocol, report):
    from torchcst._backends.torch.parameterizations import radius_envelope

    assert torch.cuda.get_device_name() == record["gpu"]
    assert torch.__version__ == record["torch"] and torch.version.cuda == record["cuda"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_dir = Path(protocol["data"])
    assert sha(data_dir / "manifest.json") == record["data_manifest_sha256"]
    _, data = train.verify_data(data_dir)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    assert hashlib.sha256(order.tobytes()).hexdigest() == record["data_order_sha256"]
    warm_report = read(args.warm_result)
    assert warm_report["checkpoint_audit"] == "passed"
    warm = torch.load(args.warm_function, map_location="cpu", weights_only=True)
    assert (
        sha(args.warm_function)
        == warm_report["function_checkpoints"][str(warm["step"])]
    )
    os.environ["CST_RESEARCH_RADIUS_CAP"] = "512"
    model = train.model_for()
    assert train.parameter_digest(model) == record["initial_parameters_sha256"]
    radius_envelope.USE_RATIONAL_LOWER = True
    model.cuda()
    projections = train.projections(model)
    scale = attach(model, projections, record["protocol"]["seed"], True)
    assert scale == warm["amplitude_scale"]
    model.decoder.load_state_dict(warm["decoder"])
    current = next(model.parameters()).device
    report["gpu"] = record["gpu"]
    report["state"] = "matching_initial_geometry"
    train.write(args.output / "result.json", report)

    common_file = args.output / "matched-base.pt"
    if "matched_base_sha256" in report:
        assert sha(common_file) == report["matched_base_sha256"]
        common = torch.load(
            common_file, map_location="cpu", weights_only=True, mmap=True
        )
    else:
        with torch.no_grad():
            initial = {
                "decoder": cpu_state(model.decoder),
                "latents": {
                    site: p.map.layer.atoms.p.detach().cpu().clone()
                    for site, p in projections.items()
                },
                "coefficient": 0.0,
            }
            base = deepcopy(model.decoder).double()
            common = {
                "model": cpu_state(model),
                "initial": initial,
                "targets": {
                    site: model.decoder(z.to(current)).cpu()
                    for site, z in initial["latents"].items()
                },
                "targets_log": {
                    site: base.preclip(z.to(device=current, dtype=torch.float64)).cpu()
                    for site, z in initial["latents"].items()
                },
                "initial_ce": train.development(model, data["development"])["ce"],
            }
        atomic_save(common, common_file)
        report["matched_base_sha256"] = sha(common_file)
        train.write(args.output / "result.json", report)
    initial = common["initial"]
    direction = head_direction(model.decoder, args.seed)

    def prepare(amount):
        return prepare_candidate(
            model.decoder,
            initial,
            common["targets"],
            common["targets_log"],
            amount,
            direction,
        )

    pair = {}
    if "effective_sigma" not in report:
        for factor in (1.0, 0.5, 0.25):
            sigma = args.sigma * factor
            try:
                pair = {"paired_plus": prepare(sigma), "paired_minus": prepare(-sigma)}
            except InitialGeometryMismatch as exc:
                print("PAIR_GEOMETRY_RETRY", sigma, str(exc), flush=True)
                report.setdefault("geometry_retries", []).append(
                    {"sigma": sigma, "error": str(exc)}
                )
                train.write(args.output / "result.json", report)
            else:
                report["effective_sigma"] = sigma
                train.write(args.output / "result.json", report)
                break
        else:
            raise InitialGeometryMismatch(
                "could not match both signs at any tested sigma"
            )

    sigma = report["effective_sigma"]

    def candidate_for(name, amount):
        file = args.output / f"initial-{name}.pt"
        if name in report["candidate_sha256"]:
            assert sha(file) == report["candidate_sha256"][name]
            candidate = torch.load(
                file, map_location="cpu", weights_only=True, mmap=True
            )
        else:
            if amount == 0:
                candidate = initial
            else:
                candidate = pair[name] if name in pair else prepare(amount)
            atomic_save(candidate, file)
            report["candidate_sha256"][name] = sha(file)
            report["initial_geometry"][name] = candidate.get(
                "maximum_geometry_error", "baseline"
            )
            if amount != 0:
                report.setdefault("initial_latent_jump", {})[name] = candidate[
                    "maximum_latent_jump"
                ]
            train.write(args.output / "result.json", report)
        assert candidate["coefficient"] == amount
        print("MATCHED_INITIAL_GEOMETRY", name, amount, flush=True)
        return dict(candidate, sha256=report["candidate_sha256"][name])

    matched = {
        "incumbent_fixed": candidate_for("incumbent_fixed", 0.0),
        "ordinary_adam": candidate_for("ordinary_adam", 0.0),
        "paired_plus": candidate_for("paired_plus", sigma),
        "paired_minus": candidate_for("paired_minus", -sigma),
    }
    report["state"] = "running"
    train.write(args.output / "result.json", report)
    for name in ("incumbent_fixed", "paired_plus", "paired_minus", "ordinary_adam"):
        report["arms"][name] = train_arm(
            args,
            train,
            record,
            model,
            projections,
            data,
            order,
            common,
            matched[name],
            name,
            name == "ordinary_adam",
            warm,
        )
        report["function_snapshots"][name] = report["arms"][name]["function_sha256"]
        train.write(args.output / "result.json", report)
    terminal = {
        name: arm["final_development"]["ce"] for name, arm in report["arms"].items()
    }
    gradient = (terminal["paired_minus"] - terminal["paired_plus"]) / (2 * sigma)
    coefficient = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    outer = torch.optim.Adam([coefficient], lr=sigma / 2)
    coefficient.grad = torch.tensor(-gradient, dtype=coefficient.dtype)
    outer.step()
    amount = float(coefficient.detach())
    report["reward_update"] = {
        "ce_rewards": {
            name: terminal["incumbent_fixed"] - terminal[name]
            for name in ("paired_plus", "paired_minus")
        },
        "gradient": gradient,
        "coefficient": amount,
    }
    train.write(args.output / "result.json", report)
    print("TERMINAL_REWARD_UPDATE", report["reward_update"], flush=True)
    new = candidate_for("reward_updated_fixed", amount)
    report["arms"]["reward_updated_fixed"] = train_arm(
        args,
        train,
        record,
        model,
        projections,
        data,
        order,
        common,
        new,
        "reward_updated_fixed",
        False,
        warm,
    )
    report["function_snapshots"]["reward_updated_fixed"] = report["arms"][
        "reward_updated_fixed"
    ]["function_sha256"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-changes", type=Path, required=True, action="append")
    parser.add_argument("--warm-result", type=Path, required=True)
    parser.add_argument("--warm-function", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4882)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=6517)
    args = parser.parse_args()
    train, record, protocol = load_reference(args)
    if not (1 <= args.steps <= train.LONG_STEPS and args.sigma > 0):
        parser.error("require a positive perturbation and a valid step budget")
    args.output.mkdir(exist_ok=True)
    target = args.output / "result.json"
    fingerprint = {
        "steps": args.steps,
        "sigma": args.sigma,
        "seed": args.seed,
        "warm_sha256": sha(args.warm_function),
        "runner_sha256": sha(Path(__file__)),
    }
    report = (
        read(target)
        if target.is_file()
        else {
            "protocol": fingerprint,
            "arms": {},
            "candidate_sha256": {},
            "initial_geometry": {},
            "function_snapshots": {},
        }
    )
    assert report["protocol"] == fingerprint
    if report.get("state") == "completed":
        return
    started = time.monotonic()
    report.update(state="initializing", error=None)
    train.write(target, report)
    try:
        run(args, train, record, protocol, report)
        report["state"] = "completed"
    except BaseException as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        train.write(target, report)


if __name__ == "__main__":
    main()
