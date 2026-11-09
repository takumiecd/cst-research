"""Matched completed pilot curves and diagnostics; refuses incompatible controls."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = (
    "dense",
    "dst-c8",
    "m16-c8-e8",
    "m16-c8-e10",
    "m16-c16-e8",
    "m16-c16-e10",
    "legacy-m8",
    "dst-c16",
)
LABELS = (
    "Dense",
    "DST c8",
    "M16 c8 eps1e-8",
    "M16 c8 eps1e-10",
    "M16 c16 eps1e-8",
    "M16 c16 eps1e-10",
    "Legacy M8 c8",
    "DST c16",
)
COLORS = (
    "#252525",
    "#a77416",
    "#2f77b5",
    "#239b56",
    "#9b4db5",
    "#d34b38",
    "#727ca5",
    "#dba53b",
)


def read(path):
    return json.loads(path.read_text())


def admit(run, dst_runs):
    if {p.resolve() for p in dst_runs} != {
        (run / a).resolve() for a in ("dst-c8", "dst-c16")
    }:
        raise ValueError("explicit colocated c8 and c16 DST controls required")
    manifest = read(run / "manifest.json")
    for name, h in manifest["files"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == h, name
    receipt = read(run / "receipt.json")
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    assert read(run / "audit/audit.json")["state"] == "passed"
    summaries = {a: read(run / a / "summary.json") for a in ARMS}
    base = summaries["dense"]
    for arm, s in summaries.items():
        assert s["state"] == "completed" and s["completed_steps"] == 976
        for key in (
            "data_manifest_sha256",
            "data_order_sha256",
            "steps",
            "schedule_steps",
            "seed",
            "effective_tokens_per_step",
            "completed_tokens",
            "gpu",
        ):
            assert s[key] == base[key], (arm, key)
        for key in base["config"]:
            if key != "coefficient":
                assert s["config"][key] == base["config"][key], (arm, key)
        assert s["actual_adam_updates"] == (973 if arm.startswith("dst") else 976)
    for c in (8, 16):
        assert (
            summaries[f"dst-c{c}"]["active_ffn_scalars"]
            == summaries[f"m16-c{c}-e8"]["active_ffn_scalars"]
            == summaries[f"m16-c{c}-e10"]["active_ffn_scalars"]
        )
    return summaries, receipt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dst-run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summaries, receipt = admit(a.run, a.dst_run)
    curves = {
        arm: [
            json.loads(line)
            for line in (a.run / arm / "curves.jsonl").read_text().splitlines()
        ]
        for arm in ARMS
    }
    a.output.mkdir(parents=True, exist_ok=False)
    table = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        dev = [v for v in curves[arm] if "development" in v]
        for ax, key in zip(axes, ("ce", "perplexity", "next_token_top1_accuracy")):
            ax.plot(
                [v["seen_tokens"] / 1e6 for v in dev],
                [v["development"][key] for v in dev],
                label=label,
                color=color,
                linestyle="--" if "c16" in arm else "-",
            )
            ax.set(xlabel="TRAIN targets (million)", ylabel=key, title="DEV " + key)
            ax.grid(alpha=0.2)
        s = summaries[arm]
        table.append(
            {
                "arm": arm,
                "active_ffn_scalars": s["active_ffn_scalars"],
                "final_development": dev[-1]["development"],
                "complete_step_median_seconds": statistics.median(s["timing_seconds"]),
                "peak_allocated_bytes": s["peak_allocated_bytes"],
                "peak_reserved_bytes": s["peak_reserved_bytes"],
            }
        )
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "RTX3090, all eight FFN maps; seed17, fixed20M schedule; short DEV pilot"
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(a.output / ("development." + suffix), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        rows = [v for v in curves[arm] if "train_ce" in v]
        ax.plot(
            [v["seen_tokens"] / 1e6 for v in rows],
            [v["train_ce"] for v in rows],
            label=label,
            color=color,
            alpha=0.8,
            linestyle="--" if "c16" in arm else "-",
        )
    ax.set(
        xlabel="TRAIN targets (million)",
        ylabel="TRAIN CE",
        title="Logged complete-update losses; raw, unsmoothed",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(a.output / ("training." + suffix), dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for index, metric in enumerate(
        ("complete_step_median_seconds", "peak_allocated_bytes", "peak_reserved_bytes")
    ):
        vals = [r[metric] / (2**20 if "bytes" in metric else 0.001) for r in table]
        axes[index].bar(range(len(ARMS)), vals, color=COLORS)
        axes[index].set_xticks(
            range(len(ARMS)), LABELS, rotation=70, ha="right", fontsize=8
        )
        axes[index].set_title(metric.replace("_", " "))
        axes[index].set_ylabel("MiB" if "bytes" in metric else "ms")
    fig.suptitle(
        "Synchronized complete training steps; first10 excluded; DST uses dense storage"
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(a.output / ("time-memory." + suffix), dpi=150)
    plt.close(fig)
    cst = [arm for arm in ARMS if "m16" in arm or arm == "legacy-m8"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    geometry = {}
    for arm in cst:
        series = [
            read(a.run / arm / f"geometry-{clock}.json") for clock in (0, 256, 976)
        ]
        geometry[arm] = series
        i = ARMS.index(arm)
        for ax, key in zip(
            axes, ("radius_mean", "active_sharp_share", "active_both_near_grid_share")
        ):
            values = []
            for sites in series:
                if key == "radius_mean":
                    values.append(
                        sum(r[key] * r["A"] for r in sites.values())
                        / sum(r["A"] for r in sites.values())
                    )
                else:
                    values.append(
                        sum(r[key] * r["active_atoms"] for r in sites.values())
                        / sum(r["active_atoms"] for r in sites.values())
                    )
            ax.plot(
                [0, 256, 976],
                values,
                marker="o",
                color=COLORS[i],
                label=LABELS[i],
                linestyle="--" if "c16" in arm else "-",
            )
            ax.set(xlabel="updates", ylabel=key)
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("CST-specific geometry; active means absolute amplitude≥.01")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(a.output / ("geometry." + suffix), dpi=150)
    plt.close(fig)
    (a.output / "results.json").write_text(
        json.dumps(
            {
                "state": "passed",
                "scope": "single-seed short DEV pilot, not convergence/equivalence",
                "source_commits": receipt["protocol"]["source_commits"],
                "results": table,
                "geometry": geometry,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
