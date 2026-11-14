"""Verify all QKV search shards and summarize held-out candidate performance."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(root):
    import torch

    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["state"] != "completed":
        raise ValueError(f"expected completed collection, got {manifest['state']}")
    keys = {
        (step, pair) for step in manifest["steps"] for pair in range(manifest["pairs"])
    }
    results = {}
    total = 0
    for part in manifest["parts"]:
        file = root / part["filename"]
        with file.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != part["sha256"]:
                raise ValueError(f"shard hash mismatch: {file.name}")
        payload = torch.load(file, map_location="cpu", weights_only=True)
        key = (payload["step"], payload["pair"])
        if key != (part["step"], part["pair"]) or key not in keys:
            raise ValueError(f"unexpected or duplicate shard: {file.name}")
        keys.remove(key)
        groups = payload["groups"]
        if len(groups) != 6:
            raise ValueError(f"missing groups: {file.name}")
        for group in groups:
            observations = group["observations"]
            count = len(observations)
            if count != manifest["actions_per_group"]:
                raise ValueError(f"wrong proposal count: {file.name}")
            if not torch.equal(group["proposed_raw"][0], group["raw"]):
                raise ValueError(f"control changed Atom rows: {file.name}")
            if observations[0][2] != 0 or abs(observations[0][1]) > 1e-7:
                raise ValueError(f"control changed loss: {file.name}")
            total += count
        rows = [
            {name: value for name, value in branch.items() if name != "proposed_raw"}
            for branch in payload["shortlist"]
        ]
        if rows != part["shortlist"]:
            raise ValueError(f"shortlist metadata mismatch: {file.name}")
        for row in rows:
            family = row["family"]
            action, delta, changed = groups[row["group"]]["observations"][row["index"]]
            if action[0] != family or delta != row["delta_exploration"] or changed == 0:
                raise ValueError(f"shortlist action mismatch: {file.name}")
            results.setdefault((key[0], family), []).append(row)
    if keys:
        raise ValueError(f"missing shards: {sorted(keys)}")

    def metrics(rows, key):
        values = [row[key] for row in rows if row[key] is not None]
        return {
            "count": len(values),
            "improved": sum(x < 0 for x in values),
            "mean_delta_ce": statistics.mean(values) if values else None,
            "median_delta_ce": statistics.median(values) if values else None,
        }

    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "verified_parts": len(manifest["parts"]),
        "candidate_evaluations": total,
        "rollout_steps": manifest["rollout_steps"],
        "results": [
            {
                "step": step,
                "family": family,
                "immediate": metrics(rows, "delta_confirmation"),
                "after_updates": metrics(rows, "delta_after_updates"),
            }
            for (step, family), rows in sorted(results.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.dumps(summarize(args.dataset), indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as handle:
        handle.write(result)
    print(f"Verified and summarized {args.dataset}")


if __name__ == "__main__":
    main()
