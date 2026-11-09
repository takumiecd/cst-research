"""Read-only same-token representation comparisons; Dense is not a teacher."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    den = a.norm() * b.norm()
    return float(a @ b / den) if den > 0 else None


def cka(a, b):
    """Centered linear CKA in token space, invariant to feature-basis rotation."""
    a, b = a.reshape(-1, a.shape[-1]).double(), b.reshape(-1, b.shape[-1]).double()
    if len(a) != len(b):
        raise ValueError("same token rows required")
    a, b = a - a.mean(0), b - b.mean(0)
    ga, gb = a @ a.T, b @ b.T
    den = ga.norm() * gb.norm()
    return float((ga * gb).sum() / den) if den > 0 else None


def spectrum(x):
    x = x.reshape(-1, x.shape[-1]).double()
    x = x - x.mean(0)
    eigen = torch.linalg.eigvalsh(x @ x.T).clamp_min(0)
    total = eigen.sum()
    if total <= 0:
        return {"stable_rank": None, "entropy_rank": None}
    p = eigen / total
    return {
        "stable_rank": float(total / eigen[-1]),
        "entropy_rank": float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum())),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    receipt = json.loads((a.run / "receipt.json").read_text())
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    protocol = receipt["protocol"]
    result = {
        "state": "running",
        "script_sha256": sha(Path(__file__)),
        "source_commits": protocol["source_commits"],
        "scope": "fixed TRAIN context offset0; same-token descriptive similarity, no Dense teacher or causal intervention",
        "records": [],
        "tensor_hashes": {},
    }
    for clock in protocol["signals"]:
        base = torch.load(a.run / "dense" / f"signals-{clock}.pt", weights_only=True)
        for arm in protocol["arms"]:
            path = a.run / arm / f"signals-{clock}.pt"
            result["tensor_hashes"][str(path.relative_to(a.run))] = sha(path)
            current = torch.load(path, weights_only=True)
            for site, data in current.items():
                metrics = {}
                for key, value in data.items():
                    metrics[key] = {
                        "cosine_to_dense": cosine(value, base[site][key]),
                        "cka_to_dense": cka(value, base[site][key]),
                    }
                    if key in ("x", "y"):
                        metrics[key]["spectrum"] = spectrum(value)
                result["records"].append(
                    {"step": clock, "arm": arm, "site": site, "metrics": metrics}
                )
            del current
            print("ANALYZED", arm, clock, flush=True)
    summaries = {}
    for arm in protocol["arms"]:
        s = json.loads((a.run / arm / "summary.json").read_text())
        rows = [
            json.loads(line)
            for line in (a.run / arm / "curves.jsonl").read_text().splitlines()
        ]
        final = next(
            row["development"] for row in rows if row["step"] == protocol["steps"]
        )
        summaries[arm] = {
            "development": final,
            "median_complete_step_ms": 1000 * statistics.median(s["timing_seconds"]),
            "peak_allocated_mib": s["peak_allocated_bytes"] / 2**20,
            "peak_reserved_mib": s["peak_reserved_bytes"] / 2**20,
            "active_ffn_scalars": s["active_ffn_scalars"],
            "sites": s["sites"],
        }
    result.update(state="passed", endpoints=summaries)
    (a.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
