"""Audit middle-stage QKV matched losses and report evidence by checkpoint."""

import hashlib
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from statistics import mean

FAMILIES = ("geometry", "local", "relocate")


def measure_by_pair(buckets):
    values = [mean(v) for _, v in sorted(buckets.items())]
    return {
        "mean_delta_ce": mean(values),
        "pairs_improved": sum(x < -1e-7 for x in values),
        "pairs": len(values),
        "by_pair": values,
    }


def difference(moves, names, signs):
    return mean(
        sum(
            sign * moves[name]["delta_ce"][batch]
            for name, sign in zip(names, signs, strict=True)
        )
        for batch in (1, 2)
    )


def summarize(dataset):
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["state"] != "completed":
        raise ValueError("expected a completed intervention run")
    by_step = defaultdict(list)
    for part in manifest["parts"]:
        raw = (dataset / part["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != part["sha256"]:
            raise ValueError(f"hash mismatch for {part['file']}")
        row = json.loads(raw)
        assert (row["step"], row["pair"]) == (part["step"], part["pair"])
        assert (
            len(row["groups"]) == 6
            and sum(len(g["actions"]) for g in row["groups"]) == part["candidates"]
        )
        by_step[row["step"]].append(row)
    assert set(by_step) == set(manifest["steps"])
    assert all(
        sorted(row["pair"] for row in rows) == list(range(manifest["pairs"]))
        for rows in by_step.values()
    )
    predictions = defaultdict(list)
    for row in json.loads((dataset / "mlp-cross-validation.json").read_text()):
        predictions[row["step"]].append(row)

    results = []
    for step, rows in sorted(by_step.items()):
        selections = defaultdict(lambda: defaultdict(list))
        factorial = defaultdict(lambda: defaultdict(list))
        per_action = defaultdict(lambda: defaultdict(list))
        clip_count = defaultdict(int)
        atom_count = defaultdict(int)
        for row in rows:
            pair = row["pair"]
            for group in row["groups"]:
                amp, grad = group["amplitude_rank"], group["gradient_rank"]
                moves = {a["name"]: a for a in group["actions"]}
                assert all(abs(z) <= 1e-7 for z in moves["control"]["delta_ce"])
                for name, move in moves.items():
                    per_action[(amp, grad, name)][pair].extend(move["delta_ce"][1:])
                    clip_count[(amp, grad, name)] += move["lower_clips"]
                    atom_count[(amp, grad, name)] += group["atoms"]
                factorial["stronger_amp_and_smaller_radius"][pair].append(
                    difference(
                        moves,
                        ("geometry_1.25_0.67", "geometry_1.25_1", "geometry_1_0.67"),
                        (1, -1, -1),
                    )
                )
                factorial["stronger_amp_and_larger_radius"][pair].append(
                    difference(
                        moves,
                        ("geometry_1.25_1.5", "geometry_1.25_1", "geometry_1_1.5"),
                        (1, -1, -1),
                    )
                )
                for distance in (4, 32, 128):
                    both = f"random_center_{distance}_11"
                    x = f"random_center_{distance}_10"
                    y = f"random_center_{distance}_01"
                    factorial[f"move_{distance}_both_vs_sum"][pair].append(
                        difference(moves, (both, x, y), (1, -1, -1))
                    )
                    if distance != 4:
                        factorial[f"move_{distance}_both_vs_4"][pair].append(
                            difference(moves, (both, "random_center_4_11"), (1, -1))
                        )
                        for factor in ("0.25_1", "1.25_1", "1_0.67", "1_1.5"):
                            factorial[
                                f"move_{distance}_with_{factor}_vs_position_only"
                            ][pair].append(
                                difference(
                                    moves,
                                    (f"relocate_{distance}_{factor}", both),
                                    (1, -1),
                                )
                            )
            for family in FAMILIES:
                options = [
                    action
                    for group in row["groups"]
                    for action in group["actions"]
                    if action["family"] == family and action["changed_atoms"]
                ]
                best = min(options, key=lambda action: action["delta_ce"][0])
                selections[f"{family}_immediate"][pair].extend(best["delta_ce"][1:])
                series = row["trajectories"].get(family)
                if series is not None:
                    selected = row["groups"][series["group"]]["actions"][
                        series["action"]
                    ]
                    assert selected["delta_ce"][0] == best["delta_ce"][0]
                    for horizon, v in series["delta_ce"].items():
                        selections[f"{family}_{horizon}"][pair].extend(v)
        for row in predictions[step]:
            for name, m in row["methods"].items():
                selections[f"selector_{name}"][row["pair"]].extend(
                    m["confirmation_delta_ce"]
                )
        results.append(
            {
                "step": step,
                "selection": {
                    key: measure_by_pair(value)
                    for key, value in sorted(selections.items())
                },
                "paired_contrasts": {
                    key: measure_by_pair(value)
                    for key, value in sorted(factorial.items())
                },
                "by_group_and_action": [
                    {
                        "amp_rank": key[0],
                        "gradient_rank": key[1],
                        "action": key[2],
                        **measure_by_pair(value),
                        "lower_clip_share": clip_count[key] / atom_count[key],
                    }
                    for key, value in sorted(per_action.items())
                ],
            }
        )
    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "contexts": len(manifest["parts"]),
        "candidates": sum(row["candidates"] for row in manifest["parts"]),
        "results": results,
    }


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = summarize(args.dataset)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
