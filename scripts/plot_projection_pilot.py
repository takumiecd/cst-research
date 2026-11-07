"""Render verified Dense/CST projection learning and amplitude-conditioned geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--historical", type=Path, action="append", default=[])
    args = p.parse_args()
    b = args.run.resolve()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    assert read(b / "retrieval-integrity.json")["state"] == "passed"
    receipt = read(b / "retrieved/receipt.json")
    assert receipt["state"] == "COMPLETED_AUDITED"
    labels = {
        "rational-q0": "Rational q0",
        "minimum-001": "Minimum L, epsilon=.001",
        "minimum-003": "Minimum L, epsilon=.003",
        "dense": "Dense",
        "cst-v": "CST V",
        "cst-q": "CST Q",
        "cst-k": "CST K",
        "cst-qkv": "CST QKV",
        "cst-qkv-ffn": "CST QKV + FFN",
    }
    colors = {
        "rational-q0": "#777777",
        "minimum-001": "#0072B2",
        "minimum-003": "#D55E00",
        "dense": "#222222",
        "cst-v": "#009E73",
        "cst-q": "#0072B2",
        "cst-k": "#D55E00",
        "cst-qkv": "#CC79A7",
        "cst-qkv-ffn": "#8B5A2B",
    }
    paths = {arm: b / "retrieved" / arm for arm in receipt["arms"]}
    historical = set()
    for path in args.historical:
        path = path.resolve()
        if not path.is_relative_to(b.parent):
            p.error("historical evidence must be inside the common evidence root")
        assert (
            read(path.parent.parent / "retrieval-integrity.json")["state"] == "passed"
        )
        assert path.name not in paths
        paths[path.name] = path
        historical.add(path.name)
    summaries = {arm: read(path / "summary.json") for arm, path in paths.items()}
    curves = {arm: rows(path / "curves.jsonl") for arm, path in paths.items()}
    updates = next(iter(summaries.values()))["completed_updates"]
    assert updates in (400, 976)
    for s in summaries.values():
        assert (
            s["completed_updates"] == updates
            and s["completed_tokens"] == updates * 1024
        )
        assert (
            s["data_manifest_sha256"]
            == "5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01"
        )
    geo = {
        arm: rows(path / "geometry.jsonl")
        for arm, path in paths.items()
        if arm not in historical and arm != "dense"
    }
    plt.rcParams["svg.hashsalt"] = b.name

    def save(fig, name):
        for ext in ("png", "svg"):
            path = out / (name + "." + ext)
            fig.savefig(
                path,
                dpi=160,
                bbox_inches="tight",
                metadata={"Date": None} if ext == "svg" else None,
            )
            if ext == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n"
                )
        plt.close(fig)

    for tail, name in ((False, "learning-curves"), (True, "learning-curves-tail")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for arm, history in curves.items():
            label = labels[arm] + (" (prior pilot)" if arm in historical else "")
            for col, key in enumerate(("train_ce", "development")):
                selected = [
                    r for r in history if key in r and (not tail or r["step"] >= 200)
                ]
                yy = [
                    r[key]["ce"] if key == "development" else r[key] for r in selected
                ]
                axes[col].plot(
                    [r["step"] for r in selected],
                    yy,
                    ":" if arm in historical else "--" if arm == "dense" else "-",
                    color=colors[arm],
                    label=label,
                    lw=1.7,
                    marker="o" if col else None,
                )
        for ax in axes:
            ax.set(
                xlabel="Optimizer updates",
                ylabel="Cross entropy (lower is better)",
                xlim=(190 if tail else 0, updates * 1.025),
            )
            ax.grid(alpha=0.2)
        axes[0].set_title("Training: no smoothing")
        axes[1].set_title("Development: fixed 8192 tokens")
        axes[1].legend(frameon=False, fontsize=9)
        fig.suptitle(
            "N256 / 4 blocks / c8 / seed17"
            + (
                f" — updates200–{updates}"
                if tail
                else f" — {updates} updates / {updates * 1024} tokens"
            )
        )
        save(fig, name)

    groups = {
        "query": ".qkv.query.map",
        "key": ".qkv.key.map",
        "value": ".qkv.value.map",
        "ffn-up": ".ffn.up.map",
        "ffn-down": ".ffn.down.map",
    }
    bins = [
        (0.0, 0.01, "|a| < .01", "-"),
        (0.01, 0.1, ".01 <= |a| < .1", "--"),
        (0.1, 0.6, ".1 <= |a| < .6", ":"),
        (0.6, 1.000001, ".6 <= |a| <= 1", "-."),
    ]
    for group, suffix in groups.items():
        eligible = {
            arm: history
            for arm, history in geo.items()
            if any(n.endswith(suffix) for n in history[0]["layers"])
        }
        if not eligible:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for arm, history in eligible.items():
            for lo, hi, _, style in bins:
                for col, metric in enumerate(("input_radius", "q")):
                    values = []
                    for row in history:
                        es = [
                            e
                            for n, g in row["layers"].items()
                            if n.endswith(suffix)
                            for e in g["bins"]
                            if e["low"] >= lo and e["high"] <= hi + 1e-8 and e["count"]
                        ]
                        count = sum(e["count"] for e in es)
                        values.append(
                            sum(e["count"] * e[metric]["mean"] for e in es) / count
                            if count
                            else float("nan")
                        )
                    axes[col].plot(
                        [r["step"] for r in history],
                        values,
                        style,
                        color=colors[arm],
                        lw=1.6,
                    )
        axes[0].set(ylabel="Mean support radius", yscale="log", ylim=(0.45, 55))
        axes[1].set(ylabel="Mean q")
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            ax.set_xlabel("Optimizer updates")
            ax.grid(alpha=0.2)
        handles = [
            Line2D([], [], color=colors[a], label=labels[a]) for a in eligible
        ] + [
            Line2D([], [], color="#444444", ls=style, label=label)
            for _, _, label, style in bins
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9
        )
        fig.suptitle(
            group + " — amplitude-conditioned means; current-bin membership changes"
        )
        fig.tight_layout(rect=(0, 0.16, 1, 0.93))
        save(fig, "geometry-" + group)

    if "rational-q0" in paths:
        histories = {
            arm: [row for row in history if "center_motion" in row]
            for arm, history in curves.items()
        }
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for col, side in enumerate(("up", "down")):
            for arm, history in histories.items():
                for low, high, _, style in bins:
                    for rowidx, metric in enumerate(
                        ("rms_step_over_radius", "path_over_radius")
                    ):
                        yy = []
                        for row in history:
                            es = [
                                e
                                for n, g in row["center_motion"].items()
                                if n.endswith(".ffn." + side + ".map")
                                for e in g["bins"]
                                if e["low"] >= low
                                and e["high"] <= high + 1e-8
                                and e["count"]
                            ]
                            count = sum(e["count"] for e in es)
                            yy.append(
                                sum(e["count"] * e[metric]["mean"] for e in es) / count
                                if count
                                else float("nan")
                            )
                        axes[rowidx, col].plot(
                            [r["step"] for r in history], yy, style, color=colors[arm]
                        )
            axes[0, col].set(
                title="FFN " + side, ylabel="Mean Atom RMS step / radius", yscale="log"
            )
            axes[1, col].set(ylabel="Mean cumulative path / radius", yscale="log")
        for ax in axes.flat:
            ax.set_xlabel("Optimizer updates")
            ax.grid(alpha=0.2)
        handles = [
            Line2D([], [], color=colors[a], label=labels[a]) for a in histories
        ] + [
            Line2D([], [], color="#444444", ls=style, label=label)
            for _, _, label, style in bins
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9
        )
        fig.suptitle(
            "Center motion: shortest torus displacement / previous support radius"
        )
        fig.tight_layout(rect=(0, 0.16, 1, 0.94))
        save(fig, "center-motion")
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        a = np.geomspace(1e-5, 1, 1000)
        z = (a / 0.3) ** 2
        old = 0.5 + 3.5 / (1 + 3 * z)
        upper = np.maximum(np.maximum(0.5 + 148.5 / (3 + z), 1), old)
        ax.plot(a, upper, color="#222222", label="Unchanged U0")
        ax.plot(a, old, color=colors["rational-q0"], ls="--", label="Rational L0")
        for epsilon, arm in ((0.001, "minimum-001"), (0.003, "minimum-003")):
            ax.plot(
                a,
                0.5 + (upper - 0.5) * epsilon**2 / (a * a + epsilon**2),
                color=colors[arm],
                label=labels[arm],
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Absolute amplitude |a|",
            ylabel="Support radius",
            ylim=(0.45, 55),
            title="Declared lower bounds; q=0 selects L",
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        save(fig, "envelope-shape")

    result = {
        arm: {
            "label": labels[arm],
            "historical": arm in historical,
            "DEV_CE": s["latest_development"]["ce"],
            "DEV_PPL": math.exp(s["latest_development"]["ce"]),
            "median_update_seconds": statistics.median(s["timing_samples_seconds"]),
            "peak_allocated_bytes": s["peak_allocated_bytes"],
            "peak_reserved_bytes": s["peak_reserved_bytes"],
            "total_elapsed_seconds": s["total_elapsed_seconds"],
            "cst_sites": len(s["initialization_reports"]),
            "cst_atoms": sum(
                v["atom_count"] for v in s["initialization_reports"].values()
            ),
        }
        for arm, s in summaries.items()
    }
    (b / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "source_commits": receipt["source_commits"],
        "data_manifest_sha256": next(iter(summaries.values()))["data_manifest_sha256"],
        "archive_sha256": read(b / "retrieval-integrity.json")["archive_sha256"],
        "plot_script_sha256": sha(Path(__file__)),
        "matplotlib": matplotlib.__version__,
        "source_files": {
            str(f.relative_to(b / "retrieved")): sha(f)
            for f in (b / "retrieved").rglob("*")
            if f.is_file()
            and f.name
            in ("summary.json", "curves.jsonl", "geometry.jsonl", "qualification.json")
        },
        "historical_files": {
            str(f.relative_to(b.parent)): sha(f)
            for arm, path in paths.items()
            if arm in historical
            for f in [path / "summary.json", path / "curves.jsonl"]
        },
        "figures": {
            f.name: sha(f) for f in out.iterdir() if f.suffix in (".png", ".svg")
        },
    }
    (out / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
