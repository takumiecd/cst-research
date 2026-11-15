"""Measure how middle-stage QKV atoms respond to matched interventions."""

import argparse
import gc
import hashlib
import json
import math
import time
from copy import deepcopy
from itertools import product
from pathlib import Path

import torch

if __package__:
    from .atom_action_search import direction
    from .fit_qkv_atom_loss import features as pooled_features
    from .fit_qkv_atom_loss import fit, predict
    from .qkv_atom_action_search_run import (
        context_groups,
        evaluate,
        features,
        replace_rows,
    )
    from .qkv_atom_intervention_run import load_reference
else:
    from atom_action_search import direction
    from fit_qkv_atom_loss import features as pooled_features
    from fit_qkv_atom_loss import fit, predict
    from qkv_atom_action_search_run import (
        context_groups,
        evaluate,
        features,
        replace_rows,
    )
    from qkv_atom_intervention_run import load_reference


def proposals(raw, physical, gradient, seed):
    """Return matched, in-chart combinations on a common set of atoms."""
    descent = direction(physical, gradient).double()
    a, r, o, i = physical.double().unbind(1)
    center = torch.stack((o, i), 1)
    zeros = torch.zeros_like(center)
    generator = torch.Generator(device=raw.device).manual_seed(seed)
    signs = (
        torch.randint(0, 2, center.shape, device=raw.device, generator=generator) * 2
        - 1
    )

    def encode(amp, radius, center_shift):
        candidate = raw.clone()
        candidate[:, 0] = amp.clamp(-32, 32)
        if amp is not a or radius is not r:
            amp = candidate[:, 0].double()
            low = 0.5 + 3.5 / (1 + 3 * (amp / 0.3).square())
            intended = radius.clamp(0.5, 512.0)
            candidate[:, 1] = (
                (intended.maximum(low).log() - low.log())
                / (math.log(512.0) - low.log())
            ).clamp(0, 1)
            clipped = int((intended < low - 2e-6).sum())
        else:
            clipped = 0
        candidate[:, 2:] = (center + center_shift).remainder(1024)
        return candidate, clipped

    yield "control", "control", raw.clone(), 0
    for gain, scaling in product((0.8, 1.0, 1.25), (0.67, 1.0, 1.5)):
        if gain == scaling == 1.0:
            continue
        yield (
            f"geometry_{gain:g}_{scaling:g}",
            "geometry",
            *encode(
                a if gain == 1.0 else gain * a,
                r if scaling == 1.0 else scaling * r,
                zeros,
            ),
        )
    rms = a.abs().clamp_min(0.03)
    for scale in (-1.0, 0.25, 1.0, 2.0):
        yield (
            f"gradient_amp_{scale:g}",
            "geometry",
            *encode(a + 0.2 * scale * rms * descent[:, 0], r, zeros),
        )
        yield (
            f"gradient_radius_{scale:g}",
            "geometry",
            *encode(a, r * (0.2 * scale * descent[:, 1]).exp(), zeros),
        )
    for scale, axes in product((-0.5, 0.25, 1.0), ((1, 0), (0, 1), (1, 1))):
        axes_tensor = center.new_tensor(axes)
        yield (
            f"gradient_center_{scale:g}_{axes[0]}{axes[1]}",
            "local",
            *encode(a, r, 0.5 * scale * r[:, None] * descent[:, 2:] * axes_tensor),
        )
    for distance, axes in product((4, 32, 128), ((1, 0), (0, 1), (1, 1))):
        shift = distance * signs * center.new_tensor(axes)
        yield (
            f"random_center_{distance}_{axes[0]}{axes[1]}",
            "local" if distance == 4 else "relocate",
            *encode(a, r, shift),
        )
    for distance, gain, scaling in product(
        (32, 128), (0.25, 1.0, 1.25), (0.67, 1.0, 1.5)
    ):
        if gain == scaling == 1.0:
            continue
        yield (
            f"relocate_{distance}_{gain:g}_{scaling:g}",
            "relocate",
            *encode(
                a if gain == 1.0 else gain * a,
                r if scaling == 1.0 else scaling * r,
                distance * signs,
            ),
        )


def trajectory(
    model, train, state, optimizer, opt_state, parameters, change, batches, hold, step
):
    """Evaluate all requested horizons on the same branch, restoring the exact start."""
    model.load_state_dict(state)
    train.set_tau(model, train.tau_at(step))
    optimizer.load_state_dict(deepcopy(opt_state))
    if change is not None:
        masks, candidate = change
        with torch.no_grad():
            for parameter, mask, rows in zip(parameters, masks, candidate, strict=True):
                parameter[mask] = rows
    torch.manual_seed(21611 + step)
    output = {}
    try:
        for offset, batch in enumerate(batches, start=1):
            train.set_tau(model, train.tau_at(step + offset))
            train.step(model, optimizer, batch, step + offset)
            if offset == 8 or offset == len(batches):
                output[str(offset)] = [evaluate(model, sample) for sample in hold]
        return output
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
    opt_state,
    step,
    pair,
    write_immediate,
    horizon,
):
    sites = train.projections(model)
    parameters = tuple(p.map.atoms.p for p in sites.values())
    device = parameters[0].device
    offsets = torch.tensor(
        [0, *torch.tensor([len(p) for p in parameters]).cumsum(0).tolist()]
    )
    base = 1536 + pair * 320
    dev = train.token_batch(data["development"], [base, base + 129], 128, "cuda")
    hold = [
        train.token_batch(data["confirmation"], [start, start + 129], 128, "cuda")
        for start in (base, base + 1536)
    ]
    start = 3_500_000 + pair * 4096
    gradient_batch = train.token_batch(data["train"], [start, start + 129], 128, "cuda")
    physical, gradients, _, gradient_ce = features(
        model, gradient_batch, sites, parameters
    )
    samples = [dev, *hold]
    baseline = [evaluate(model, sample) for sample in samples]
    best = {}
    groups = []
    pool_features, pool_grad, pool_dev, pool_hold = [], [], [], []
    groups_seed = 2311 + pair + step * 1009
    for amp_rank, grad_rank, ids, spans in context_groups(
        physical, gradients, offsets, groups_seed
    ):
        masks = [
            (ids[lo:hi] - offset).to(device)
            for (lo, hi), offset in zip(spans, offsets.tolist())
        ]
        raw = tuple(p.detach()[mask] for p, mask in zip(parameters, masks, strict=True))
        raw_pack = torch.cat(raw)
        size = [len(v) for v in raw]
        actual = physical[ids]
        grad = gradients[ids]
        entries = []
        action_rows = []
        for name, family, candidate, clipped in proposals(
            raw_pack, actual.to(device), grad.to(device), groups_seed + len(groups)
        ):
            rows = candidate.split(size)
            with replace_rows(parameters, masks, raw, rows):
                delta = [
                    evaluate(model, b) - ref
                    for b, ref in zip(samples, baseline, strict=True)
                ]
            if name == "control":
                assert all(abs(d) <= 1e-7 for d in delta)
            changed = int((candidate != raw_pack).any(1).sum())
            entries.append(
                {
                    "name": name,
                    "family": family,
                    "delta_ce": delta,
                    "lower_clips": clipped,
                    "changed_atoms": changed,
                }
            )
            action_rows.append(candidate.cpu())
            if changed and (family not in best or delta[0] < best[family][0]):
                best[family] = (delta[0], len(groups), len(entries) - 1, masks, rows)
        assert abs(evaluate(model, dev) - baseline[0]) <= 1e-7
        x, dot = pooled_features(
            {
                "physical": actual,
                "physical_gradients": grad,
                "proposed_raw": torch.stack(action_rows),
            }
        )
        pool_features.append(x)
        pool_grad.append(dot)
        pool_dev.append(torch.tensor([entry["delta_ce"][0] for entry in entries]))
        pool_hold.append(torch.tensor([entry["delta_ce"][1:] for entry in entries]))
        groups.append(
            {
                "amplitude_rank": amp_rank,
                "gradient_rank": grad_rank,
                "atoms": len(ids),
                "amplitude_median": float(actual[:, 0].abs().median()),
                "radius_median": float(actual[:, 1].median()),
                "actions": entries,
            }
        )
    report = {
        "step": step,
        "pair": pair,
        "baseline": baseline,
        "gradient_ce": gradient_ce,
        "dev_starts": [base, base + 129],
        "confirmation_starts": [[base, base + 129], [base + 1536, base + 1665]],
        "groups": groups,
        "trajectories": {},
    }
    write_immediate(report)
    matrices = {
        "step": step,
        "pair": pair,
        "x": torch.stack(pool_features),
        "dot": torch.stack(pool_grad),
        "y": torch.stack(pool_dev),
        "hold": torch.stack(pool_hold),
    }
    if not horizon:
        return report, matrices
    batches = []
    for offset in range(horizon):
        start = 4_000_000 + pair * 65536 + offset * 1024
        batches.append(
            [
                train.token_batch(
                    data["train"], [start + j, start + j + 128], 128, "cuda"
                )
                for j in range(0, 1024, 256)
            ]
        )
    control = trajectory(
        model, train, state, optimizer, opt_state, parameters, None, batches, hold, step
    )
    for family, (_, group, action, masks, candidate) in best.items():
        # At 32 updates keep the prespecified relocation comparison and its control.
        length = horizon if family == "relocate" else 8
        result = trajectory(
            model,
            train,
            state,
            optimizer,
            opt_state,
            parameters,
            (masks, candidate),
            batches[:length],
            hold,
            step,
        )
        report["trajectories"][family] = {
            "group": group,
            "action": action,
            "delta_ce": {
                offset: [a - b for a, b in zip(value, control[offset], strict=True)]
                for offset, value in result.items()
            },
        }
    return report, matrices


def cross_validate(matrices):
    selections = []
    for pair in sorted({r["pair"] for r in matrices}):
        fitted = fit([r for r in matrices if r["pair"] != pair])
        for row in matrices:
            if row["pair"] != pair:
                continue
            scores = predict(row, fitted)
            methods = {}
            for name, score in scores.items():
                group, action = divmod(int(score.argmin()), score.shape[1])
                methods[name] = {
                    "group": group,
                    "action": action,
                    "forecast": float(score[group, action]),
                    "confirmation_delta_ce": row["hold"][group, action].tolist(),
                }
            selections.append({"step": row["step"], "pair": pair, "methods": methods})
    return selections


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(1536, 2441, 3600))
    parser.add_argument("--pairs", type=int, default=4, choices=range(2, 5))
    parser.add_argument("--no-rollout", action="store_true")
    args = parser.parse_args()
    args.output.mkdir()
    report = {
        "state": "initializing",
        "parts": [],
        "steps": args.steps,
        "pairs": args.pairs,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    target = args.output / "manifest.json"

    def write(path, payload):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)

    write(target, report)
    started = time.monotonic()
    try:
        probe, train, reference, data, model = load_reference(args.reference_probe)
        report.update(
            state="running",
            gpu=reference["gpu"],
            torch=torch.__version__,
            data_manifest_sha256=reference["data_manifest_sha256"],
            reference_checkpoints=reference["arms"][probe.ARM]["checkpoints"],
        )
        write(target, report)
        matrices = []
        for step in args.steps:
            saved = probe.load_checkpoint(reference, step)
            state, opt_state = saved["model"], saved["optimizer"]
            model.load_state_dict(state)
            train.set_tau(model, train.tau_at(step))
            assert saved["tau"] == train.tau_at(step)
            optimizer = train.optimizer_for(model)
            optimizer.load_state_dict(deepcopy(opt_state))
            print("checkpoint", step, "ready", flush=True)
            for pair in range(args.pairs):
                file = args.output / f"step-{step}-pair-{pair}.json"

                def write_immediate(payload, *, file=file):
                    print(
                        payload["step"],
                        payload["pair"],
                        "immediate measurements ready",
                        flush=True,
                    )
                    write(file, payload)

                result, values = collect(
                    model,
                    train,
                    data,
                    state,
                    optimizer,
                    opt_state,
                    step,
                    pair,
                    write_immediate,
                    0 if args.no_rollout else (32 if pair % 2 == 0 else 8),
                )
                write(file, result)
                matrices.append(values)
                report["parts"].append(
                    {
                        "file": file.name,
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                        "step": step,
                        "pair": pair,
                        "candidates": sum(len(g["actions"]) for g in result["groups"]),
                    }
                )
                write(target, report)
                print(step, pair, "part complete", flush=True)
            del optimizer, saved, state, opt_state
            gc.collect()
        torch.save(matrices, args.output / "mlp-features.pt")
        write(args.output / "mlp-cross-validation.json", cross_validate(matrices))
        report["state"] = "completed"
    except BaseException as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        write(target, report)


if __name__ == "__main__":
    main()
