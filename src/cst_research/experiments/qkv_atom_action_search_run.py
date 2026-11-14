"""Search QKV Atom updates and confirm selected actions on disjoint batches."""

import argparse
import gc
import json
import math
import time
from contextlib import contextmanager
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

import torch

if __package__:
    from . import atom_action_search as search
    from .qkv_atom_intervention_run import load_reference, transform_physical
else:
    import atom_action_search as search
    from qkv_atom_intervention_run import load_reference, transform_physical


def context_groups(values, gradients, offsets, batch_seed):
    """Sample by three amplitude ranks and two gradient-magnitude ranks."""
    generator = torch.Generator().manual_seed(batch_seed)
    random_order = torch.randperm(len(values), generator=generator)
    amplitude_order = random_order[values[random_order, 0].abs().argsort(stable=True)]
    for band, section in enumerate(amplitude_order.tensor_split(3)):
        grad_norm = (
            search.direction(values[section], gradients[section]).square().sum(1)
        )
        gradient_order = section[grad_norm.argsort(stable=True)]
        for strength, pool in enumerate(gradient_order.tensor_split(2)):
            indices = (
                pool[torch.randperm(len(pool), generator=generator)[:1024]]
                .sort()
                .values
            )
            boundaries = torch.searchsorted(indices, offsets).tolist()
            by_site = list(pairwise(boundaries))
            yield band, strength, indices, by_site


@contextmanager
def replace_rows(parameters, masks, raw, candidate):
    with torch.no_grad():
        for parameter, indices, rows in zip(parameters, masks, candidate, strict=True):
            parameter[indices] = rows
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, indices, rows in zip(parameters, masks, raw, strict=True):
                parameter[indices] = rows


def features(model, batch, sites, parameters):
    captured = {}

    def capture(site, rows):
        assert site not in captured
        captured[site] = rows
        return rows

    model.eval()
    with transform_physical(sites, capture):
        _, loss = model(*batch)
    physical = tuple(captured[name] for name in sites)
    gradients = torch.autograd.grad(loss, (*parameters, *physical))
    assert all(bool(torch.isfinite(tensor).all()) for tensor in (*physical, *gradients))
    raw_grad, physical_grad = (
        gradients[: len(parameters)],
        gradients[len(parameters) :],
    )
    return (
        torch.cat([p.detach().cpu() for p in physical]),
        torch.cat([g.cpu() for g in physical_grad]),
        torch.cat([g.cpu() for g in raw_grad]),
        float(loss.detach()),
    )


@torch.no_grad()
def evaluate(model, batch):
    model.eval()
    result = float(model(*batch)[1])
    if not math.isfinite(result):
        raise ValueError("non-finite CE")
    return result


def snapshot_initial_state(model):
    """Freeze tensors and preserve PyTorch and CST checkpoint metadata."""
    state = model.state_dict()
    for name, value in state.items():
        state[name] = (
            value.detach().to(device="cpu", copy=True)
            if isinstance(value, torch.Tensor)
            else deepcopy(value)
        )
    return state


def rollout(
    model,
    train,
    state,
    optimizer,
    optimizer_state,
    parameters,
    changes,
    batches,
    hold,
    step,
):
    """Branch from the same weights and Adam state, then run matched updates."""
    model.load_state_dict(state)
    train.set_tau(model, train.tau_at(step))
    optimizer.load_state_dict(deepcopy(optimizer_state))
    if changes is not None:
        masks, candidate = changes
        with torch.no_grad():
            for parameter, mask, rows in zip(parameters, masks, candidate, strict=True):
                parameter[mask] = rows
    torch.manual_seed(17001 + step)
    try:
        for offset, batch in enumerate(batches, start=1):
            train.set_tau(model, train.tau_at(step + offset))
            train.step(model, optimizer, batch, step + offset)
        return evaluate(model, hold)
    finally:
        model.zero_grad(set_to_none=True)
        model.load_state_dict(state)
        train.set_tau(model, train.tau_at(step))
        model.eval()


def collect(
    model,
    train,
    data,
    state,
    optimizer,
    optimizer_state,
    step,
    pair,
    budget,
    horizon,
    save_immediate,
):
    sites = train.projections(model)
    parameters = tuple(p.map.atoms.p for p in sites.values())
    offsets = torch.tensor(
        [0, *torch.tensor([len(p) for p in parameters]).cumsum(0).tolist()]
    )
    starts = [pair * 258, pair * 258 + 129]
    gradient_batch, exploration, hold = (
        train.token_batch(data[split], starts, 128, "cuda")
        for split in ("train", "development", "confirmation")
    )
    values, gradients, raw_gradients, gradient_loss = features(
        model, gradient_batch, sites, parameters
    )
    baseline = evaluate(model, exploration)
    confirmation_baseline = evaluate(model, hold)
    seed = 1703 + 101 * step + pair
    groups = []
    selected = {}
    began = time.monotonic()
    for band, strength, ids, by_site in context_groups(
        values, gradients, offsets, seed
    ):
        masks = [
            (ids[left:right] - start).to(parameters[0].device)
            for (left, right), start in zip(by_site, offsets.tolist())
        ]
        with torch.no_grad():
            raw = tuple(p[idx] for p, idx in zip(parameters, masks, strict=True))
            packed = torch.cat(raw)
            physical = values[ids].to(packed.device)
            descent = search.direction(physical, gradients[ids].to(packed.device))
            count = [len(r) for r in raw]
            observations = []
            candidates = []
            best = {}
            for index, action in enumerate(search.actions(seed + 31 * len(groups))):
                if index >= budget:
                    break
                candidate = search.propose(packed, physical, descent, action)
                changed = int((candidate != packed).any(1).sum())
                rows = candidate.split(count)
                with replace_rows(parameters, masks, raw, rows):
                    loss = evaluate(model, exploration)
                family = action[0]
                delta = loss - baseline
                observations.append((action, delta, changed))
                candidates.append(candidate)
                if changed and (family not in best or delta < best[family][0]):
                    best[family] = (delta, index, candidate)
                if index == 0:
                    assert abs(delta) <= 1e-7
            assert abs(evaluate(model, exploration) - baseline) <= 1e-7
            assert all(
                torch.equal(p[idx], original)
                for p, idx, original in zip(parameters, masks, raw, strict=True)
            )
            groups.append(
                {
                    "amplitude_rank": band,
                    "gradient_rank": strength,
                    "indices": ids,
                    "raw": packed.cpu(),
                    "physical": values[ids],
                    "physical_gradients": gradients[ids],
                    "raw_gradients": raw_gradients[ids],
                    "proposed_raw": torch.stack(candidates).cpu(),
                    "observations": observations,
                }
            )
            for family, (delta, index, candidate) in best.items():
                if family not in selected or delta < selected[family][0]:
                    selected[family] = (
                        delta,
                        len(groups) - 1,
                        index,
                        masks,
                        raw,
                        candidate.split(count),
                    )
        print(
            step,
            pair,
            band,
            strength,
            len(observations),
            "candidates measured",
            flush=True,
        )

    # Rank on exploration, then score one winner per family on separate tokens.
    # Also keep a distant move in the multistep comparison.
    ranking = sorted(selected, key=lambda family: selected[family][0])
    rollout_families = set(ranking[:2])
    distant = next((f for f in ranking if f in ("relocate", "mixed")), None)
    if distant is not None:
        rollout_families.add(distant)
    branches = []
    for family in ranking:
        delta, group, index, masks, raw, rows = selected[family]
        with replace_rows(parameters, masks, raw, rows):
            confirmation = evaluate(model, hold) - confirmation_baseline
        branches.append(
            {
                "family": family,
                "group": group,
                "index": index,
                "delta_exploration": delta,
                "delta_confirmation": confirmation,
                "confirmation_after_updates": None,
                "delta_after_updates": None,
                "proposed_raw": torch.cat(rows).cpu(),
            }
        )
    save_immediate(branches)
    rollout_batches = []
    for update in range(horizon):
        start = 2_000_000 + (pair * horizon + update) * 1024
        rollout_batches.append(
            [
                train.token_batch(
                    data["train"], [start + offset, start + offset + 128], 128, "cuda"
                )
                for offset in range(0, 1024, 256)
            ]
        )
    after_control = (
        rollout(
            model,
            train,
            state,
            optimizer,
            optimizer_state,
            parameters,
            None,
            rollout_batches,
            hold,
            step,
        )
        if horizon
        else None
    )
    for branch in branches:
        family = branch["family"]
        if after_control is None or family not in rollout_families:
            continue
        _, _, _, masks, _, rows = selected[family]
        after = rollout(
            model,
            train,
            state,
            optimizer,
            optimizer_state,
            parameters,
            (masks, rows),
            rollout_batches,
            hold,
            step,
        )
        branch["confirmation_after_updates"] = after
        branch["delta_after_updates"] = after - after_control
    return {
        "step": step,
        "pair": pair,
        "sites": tuple(sites),
        "offsets": offsets,
        "starts": starts,
        "groups": groups,
        "shortlist": branches,
        "seconds": time.monotonic() - began,
        "gradient_loss": gradient_loss,
        "exploration_baseline": baseline,
        "confirmation_baseline": confirmation_baseline,
        "confirmation_after_control": after_control,
        "rollout_steps": horizon,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=(0, 976, 1536))
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--actions", type=int, default=384)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 31 or not 2 <= args.actions <= 384 or args.horizon < 0:
        parser.error("require 1–31 pairs, 2–384 actions, and a nonnegative horizon")
    args.output.mkdir()
    path = args.output / "manifest.json"
    report = {"state": "initializing", "parts": []}

    def write():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    write()
    try:
        probe, train, record, data, model = load_reference(args.reference_probe)
        arm = record["arms"][probe.ARM]
        report.update(
            state="running",
            gpu=record["gpu"],
            torch=torch.__version__,
            cuda=torch.version.cuda,
            data_manifest_sha256=record["data_manifest_sha256"],
            source_protocol_sha256=record["source_protocol_sha256"],
            initial_parameters_sha256=record["initial_parameters_sha256"],
            checkpoint_hashes=arm["checkpoints"],
            steps=args.steps,
            pairs=args.pairs,
            actions_per_group=args.actions,
            rollout_steps=args.horizon,
            files={
                name: train.sha(Path(__file__).with_name(name))
                for name in (
                    "atom_action_search.py",
                    "qkv_atom_action_search_run.py",
                    "qkv_atom_intervention_run.py",
                )
            },
        )
        write()
        for step in args.steps:
            if step:
                saved = probe.load_checkpoint(record, step)
                state, optimizer_state = saved["model"], saved["optimizer"]
                model.load_state_dict(state)
                expected = arm["evaluations"][str(step)]["development"]
                assert saved["tau"] == train.tau_at(step)
            else:
                assert (
                    train.parameter_digest(model) == record["initial_parameters_sha256"]
                )
                expected = record["initial_development"]
            train.set_tau(model, train.tau_at(step))
            optimizer = train.optimizer_for(model)
            if not step:
                # Binding Adam gives AtomState its permanent checkpoint schema.
                state = snapshot_initial_state(model)
                optimizer_state = deepcopy(optimizer.state_dict())
            observed = train.development(model, data["development"])
            assert abs(observed["ce"] - expected["ce"]) <= 1e-6
            assert observed["correct_next_tokens"] == expected["correct_next_tokens"]
            print("CHECKPOINT_VERIFIED", step, observed["ce"], flush=True)
            for pair in range(args.pairs):
                immediate = args.output / f"step-{step}-pair-{pair}-immediate.json"

                def save_immediate(branches, *, path=immediate, step=step, pair=pair):
                    rows = [
                        {k: v for k, v in branch.items() if k != "proposed_raw"}
                        for branch in branches
                    ]
                    train.write(path, {"step": step, "pair": pair, "shortlist": rows})

                payload = collect(
                    model,
                    train,
                    data,
                    state,
                    optimizer,
                    optimizer_state,
                    step,
                    pair,
                    args.actions,
                    args.horizon,
                    save_immediate,
                )
                target = args.output / f"step-{step}-pair-{pair}.pt"
                tmp = target.with_suffix(".tmp")
                torch.save(payload, tmp)
                tmp.replace(target)
                record_part = {
                    "filename": target.name,
                    "sha256": train.sha(target),
                    "step": step,
                    "pair": pair,
                    "rows": sum(len(g["observations"]) for g in payload["groups"]),
                    "seconds": payload["seconds"],
                    "shortlist": [
                        {
                            key: value
                            for key, value in b.items()
                            if key != "proposed_raw"
                        }
                        for b in payload["shortlist"]
                    ],
                }
                report["parts"].append(record_part)
                write()
                print(json.dumps(record_part, allow_nan=False), flush=True)
            del state, optimizer_state, optimizer
            gc.collect()  # CST optimizer hooks can retain the previous updater.
        report["state"] = "completed"
    except BaseException as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write()


if __name__ == "__main__":
    main()
