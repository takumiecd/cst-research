"""Evaluate frozen selectors and four-field interventions on fresh TinyStories batches."""

import argparse
import hashlib
import json
import math
import time
from itertools import pairwise, product
from pathlib import Path

import torch

if __package__:
    from .fit_qkv_atom_loss import FIELDS, differences, physical_rows, read_part
    from .qkv_atom_action_search_run import evaluate, replace_rows
    from .qkv_atom_intervention_run import load_reference, transform_physical
else:
    from fit_qkv_atom_loss import FIELDS, differences, physical_rows, read_part
    from qkv_atom_action_search_run import evaluate, replace_rows
    from qkv_atom_intervention_run import load_reference, transform_physical


def group_rows(payload, group_id, action_id, device):
    group = payload["groups"][group_id]
    ids = group["indices"]
    offsets = payload["offsets"]
    parts = torch.searchsorted(ids, offsets).tolist()
    slices = [slice(lo, hi) for lo, hi in pairwise(parts)]
    masks = [
        (ids[sl] - start).to(device)
        for sl, start in zip(slices, offsets.tolist()[:-1], strict=True)
    ]
    raw = group["raw"].to(device)
    candidate = group["proposed_raw"][action_id].to(device)
    original = group["physical"].to(device)
    return masks, slices, raw, candidate, original


def experiment(model, parameters, group, batches, baseline):
    masks, slices, raw, candidate, original = group
    physical = physical_rows(candidate)
    move = differences(original, physical)
    with replace_rows(
        parameters, masks, [raw[sl] for sl in slices], [candidate[sl] for sl in slices]
    ):
        full = [
            evaluate(model, batch) - base
            for batch, base in zip(batches, baseline, strict=True)
        ]

    # Pair toggles at fixed physical values. Retain atoms for which all combinations
    # of the two amplitudes and two radii satisfy the existing chart bounds.
    a0, a1 = original[:, 0], physical[:, 0]
    common_lower = 0.5 + 3.5 / (1 + 3 * (a0.abs().minimum(a1.abs()) / 0.3).square())
    active = original[:, 1].minimum(physical[:, 1]) >= common_lower - 2e-6
    active &= move.abs().amax(dim=1) > 0
    return full, (masks, slices, original, physical, active), move


def factorial(model, projections, prepared, batches, baseline):
    masks, slices, original, physical, active = prepared
    actual_masks = {
        site: mask[active[sl]]
        for site, mask, sl in zip(projections, masks, slices, strict=True)
    }
    payload = {}
    for flags in product((False, True), repeat=4):
        fields = torch.tensor(flags, device=original.device)
        changed = torch.where(fields, physical, original)
        site_rows = {
            site: changed[sl][active[sl]]
            for site, sl in zip(projections, slices, strict=True)
        }

        def use_rows(site, rows, *, site_rows=site_rows):
            out = rows.clone()
            out[actual_masks[site]] = site_rows[site]
            return out

        with transform_physical(projections, use_rows):
            delta = [
                evaluate(model, batch) - base
                for batch, base in zip(batches, baseline, strict=True)
            ]
        if not any(flags):
            assert all(abs(value) <= 1e-7 for value in delta), delta
        payload["".join(str(int(flag)) for flag in flags)] = delta
    pair_interactions = {}
    active_count = int(active.sum())
    for i in range(4):
        for j in range(i + 1, 4):
            effects = []
            for b in range(len(batches)):
                contrast = (
                    sum(
                        (1 if bits[i] == bits[j] else -1) * loss[b]
                        for bits, loss in payload.items()
                    )
                    / 4
                )
                effects.append(contrast)
            pair_interactions[f"{FIELDS[i]}+{FIELDS[j]}"] = effects
    return {
        "reachable_changed_atoms": active_count,
        "factorial": payload,
        "pair_interactions": pair_interactions,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference-probe", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_bytes = (args.dataset / "manifest.json").read_bytes()
    fit_bytes = args.fit.read_bytes()
    fit = json.loads(fit_bytes)
    assert fit["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    results = {
        "state": "initializing",
        "fit_sha256": hashlib.sha256(fit_bytes).hexdigest(),
        "manifest_sha256": fit["manifest_sha256"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contexts": [],
    }

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    assert not args.output.exists()
    save()
    began = time.monotonic()
    try:
        probe, train, record, data, model = load_reference(args.reference_probe)
        results["state"] = "running"
        results["gpu"] = record["gpu"]
        results["torch"] = torch.__version__
        save()
        projections = train.projections(model)
        params = tuple(p.map.atoms.p for p in projections.values())
        device = params[0].device
        part_lookup = {p["filename"]: p for p in json.loads(manifest_bytes)["parts"]}
        previous_step = None
        for row in sorted(fit["contexts"], key=lambda x: (x["step"], x["pair"])):
            step, pair = row["step"], row["pair"]
            if step != previous_step:
                if step:
                    model.load_state_dict(probe.load_checkpoint(record, step)["model"])
                train.set_tau(model, train.tau_at(step))
                previous_step = step
                print("CHECKPOINT", step, flush=True)
            payload = read_part(args.dataset, part_lookup[row["file"]])
            windows = [
                [base + pair * 258, base + pair * 258 + 129] for base in (4096, 6144)
            ]
            batches = [
                train.token_batch(data["confirmation"], starts, 128, "cuda")
                for starts in windows
            ]
            baseline = [evaluate(model, batch) for batch in batches]
            context = {
                "step": step,
                "pair": pair,
                "starts": windows,
                "baseline": baseline,
                "selectors": {},
            }
            for method, choice in row["selectors"].items():
                group = group_rows(payload, choice["group"], choice["action"], device)
                masks, slices, raw, _, _ = group
                assert all(
                    torch.equal(parameter[mask], raw[sl])
                    for parameter, mask, sl in zip(params, masks, slices, strict=True)
                )
                full, prepared, move = experiment(
                    model, params, group, batches, baseline
                )
                obs = payload["groups"][choice["group"]]["observations"][
                    choice["action"]
                ]
                context["selectors"][method] = {
                    "family": obs[0][0],
                    "group": choice["group"],
                    "action": choice["action"],
                    "delta_ce": full,
                    "changed_atoms": obs[2],
                    "median_abs_move": move.abs().median(dim=0).values.tolist(),
                }
                if method == "mlp":
                    context["mlp_factorial"] = factorial(
                        model, projections, prepared, batches, baseline
                    )
            assert all(
                math.isclose(evaluate(model, batch), base, abs_tol=1e-7, rel_tol=0)
                for batch, base in zip(batches, baseline, strict=True)
            )
            results["contexts"].append(context)
            save()
            print(
                step,
                pair,
                {k: v["delta_ce"] for k, v in context["selectors"].items()},
                flush=True,
            )
        results["state"] = "completed"
    except BaseException as exc:
        results.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        results["seconds"] = time.monotonic() - began
        save()


if __name__ == "__main__":
    main()
