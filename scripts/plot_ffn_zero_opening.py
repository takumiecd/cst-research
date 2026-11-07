"""Overlay audited new pilots and explicitly reused historical controls."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--run",
    type=Path,
    required=True,
    help="Verified zero-opening evidence directory; historical controls must be siblings",
)
parser.add_argument(
    "--output", type=Path, help="Figure directory (default RUN/figures)"
)
args = parser.parse_args()
b = args.run.resolve()
old = b.parent
new = b / "retrieved"
assert json.loads((b / "retrieval-integrity.json").read_text())["state"] == "passed"
figures = args.output or b / "figures"
figures.mkdir(parents=True, exist_ok=True)
read = lambda p: json.loads(p.read_text())
lines = lambda p: [json.loads(l) for l in p.read_text().splitlines()]
colors = {
    "Dense (reused)": "#333333",
    "Rational q0 (reused)": "#888888",
    "tau=.01": "#0072B2",
    "tau=.03": "#D55E00",
    "tau=.1": "#009E73",
}
paths = {
    "Dense (reused)": old / "retrieved-field-lr-r12-02/ffn-r12-02-dense-pilot",
    "Rational q0 (reused)": old / "retrieved-field-lr-r12-02/ffn-r12-02-all16-pilot",
    "tau=.01": new / "tau-001",
    "tau=.03": new / "tau-003",
    "tau=.1": new / "tau-010",
}
summaries = {k: read(p / "summary.json") for k, p in paths.items()}
hist = {k: lines(p / "curves.jsonl") for k, p in paths.items()}
geo = {k: lines(paths[k] / "geometry.jsonl") for k in ("tau=.01", "tau=.03", "tau=.1")}
for k, samples in geo.items():
    assert len(samples) == 55 and samples[0]["step"] == 0 and samples[-1]["step"] == 976
    assert (
        read(
            new
            / (
                {"tau=.01": "tau-001", "tau=.03": "tau-003", "tau=.1": "tau-010"}[k]
                + "-audit-v2.json"
            )
        )["state"]
        == "passed"
    )
for s in summaries.values():
    assert s["completed_tokens"] == 999424 and s["completed_updates"] == 976
    assert (
        s["data_manifest_sha256"]
        == "5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01"
    )


def save(fig, name):
    for ext in ("png", "svg"):
        path = figures / (name + "." + ext)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for name, rows in hist.items():
    style = "--" if "(reused)" in name else "-"
    t = [r for r in rows if "train_ce" in r]
    d = [r for r in rows if "development" in r]
    axes[0].plot(
        [r["step"] for r in t],
        [r["train_ce"] for r in t],
        style,
        color=colors[name],
        label=name,
        lw=1.7,
    )
    axes[1].plot(
        [r["step"] for r in d],
        [r["development"]["ce"] for r in d],
        style,
        color=colors[name],
        marker="o",
        label=name,
        lw=1.7,
    )
for ax in axes:
    ax.set(
        xlabel="Optimizer updates (1024 tokens/update)",
        ylabel="Cross entropy (lower is better)",
        xlim=(0, 1000),
    )
    ax.grid(alpha=0.2)
axes[0].set_title("Training: recorded samples, no smoothing")
axes[1].set_title("Development: fixed 8192 tokens")
axes[1].legend(frameon=False)
fig.suptitle(
    "FFN short pilot — N256 / 4 blocks / c8 / seed17; Dense and rational-q0 reused"
)
save(fig, "learning-curves")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for name, rows in hist.items():
    style = "--" if "(reused)" in name else "-"
    for col, key in enumerate(("train_ce", "development")):
        selected = [r for r in rows if r["step"] >= 200 and key in r]
        values = [r[key]["ce"] if key == "development" else r[key] for r in selected]
        axes[col].plot(
            [r["step"] for r in selected],
            values,
            style,
            color=colors[name],
            label=name,
            lw=1.8,
            marker="o" if col else None,
        )
for ax in axes:
    ax.set(
        xlim=(190, 1000),
        xlabel="Optimizer updates",
        ylabel="Cross entropy (lower is better)",
    )
    ax.grid(alpha=0.2)
axes[0].set_title("Training, after warmup")
axes[1].set_title("Development, after warmup")
axes[1].legend(frameon=False)
fig.suptitle("Same pilot curves, updates200–976 — no smoothing or additional runs")
save(fig, "learning-curves-tail")


def average(row, side, metric, interval=None):
    total = value = 0
    for name, layer in row["layers"].items():
        if not name.endswith(".ffn." + side + ".map"):
            continue
        entries = (
            [layer["all"]]
            if interval is None
            else [
                g
                for g in layer["bins"]
                if g["low"] >= interval[0] and g["high"] <= interval[1] + 1e-8
            ]
        )
        for e in entries:
            if not e["count"]:
                continue
            total += e["count"]
            value += e["count"] * e[metric]["mean"]
    return value / total if total else float("nan")


past = read(old / "radius-review-20261107-01/analysis.json")["R7_recorded_history"]
fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
for col, side in enumerate(("up", "down")):
    for name, rows in geo.items():
        for ridx, metric in enumerate(("input_radius", "q")):
            axes[ridx, col].plot(
                [r["step"] for r in rows],
                [average(r, side, metric) for r in rows],
                color=colors[name],
                label=name,
            )
    for ridx, metric in enumerate(("input_radius_mean", "q_mean")):
        x = []
        y = []
        for step, groups in sorted(past.items(), key=lambda item: int(item[0])):
            vals = [
                v[metric]
                for n, v in groups.items()
                if n.endswith(".ffn." + side + ".map")
            ]
            assert len(vals) == 4
            x.append(int(step))
            y.append(np.mean(vals))
        axes[ridx, col].plot(
            x,
            y,
            "o--",
            color=colors["Rational q0 (reused)"],
            label="Rational q0 (saved history)",
        )
    axes[0, col].set(
        title="FFN " + side,
        xlabel="Optimizer updates",
        ylabel="Mean input support radius",
        yscale="log",
        ylim=(0.45, 55),
    )
    axes[1, col].set(
        xlabel="Optimizer updates", ylabel="Mean q (symlog)", ylim=(0, 1.05)
    )
    axes[1, col].set_yscale("symlog", linthresh=1e-4)
for ax in axes.flat:
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 1000)
axes[0, 1].legend(frameon=False)
fig.suptitle("Global radius/q trajectories — historical q0 only at stored samples")
for ax in axes[1]:
    values = np.concatenate([line.get_ydata() for line in ax.lines])
    ax.set_ylim(0, max(1e-4, np.nanmax(values) * 1.15))
save(fig, "global-radius-q")

groups = [
    (0.0, 0.01, "|a| < .01", "-"),
    (0.01, 0.1, ".01 <= |a| < .1", "--"),
    (0.1, 0.6, ".1 <= |a| < .6", ":"),
    (0.6, 1.000001, ".6 <= |a| <= 1", "-."),
]
weak = read(old / "radius-review-20261107-01/weak-analysis.json")["groups"][0]
assert weak["label"] == "all16" and weak["tokens"] == 999424
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for col, side in enumerate(("up", "down")):
    for name, rows in geo.items():
        for low, high, label, style in groups:
            for ridx, metric in enumerate(("input_radius", "q")):
                axes[ridx, col].plot(
                    [r["step"] for r in rows],
                    [average(r, side, metric, (low, high)) for r in rows],
                    style,
                    color=colors[name],
                    lw=1.6,
                )
    for low, high, label, style in groups:
        selected = [
            r
            for r in weak["sides"][side]["bins"]
            if r["interval"][0] >= low and r["interval"][1] <= high + 1e-8
        ]
        n = sum(r["count"] for r in selected)
        if n:
            axes[0, col].plot(
                976,
                sum(r["count"] * r["radius_mean"] for r in selected) / n,
                "x",
                color="#888888",
                ms=7,
                mew=1.5,
            )
    axes[0, col].set(
        title="FFN " + side, ylabel="Mean radius", yscale="log", ylim=(0.45, 55)
    )
    axes[1, col].set(
        ylabel="Mean q (symlog)", xlabel="Optimizer updates", ylim=(0, 1.05)
    )
    axes[1, col].set_yscale("symlog", linthresh=1e-4)
for ax in axes.flat:
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 985)
handles = (
    [Line2D([], [], color=colors[n], label=n) for n in geo]
    + [Line2D([], [], color="#333333", linestyle=s, label=l) for _, _, l, s in groups]
    + [
        Line2D(
            [],
            [],
            color="#888888",
            marker="x",
            ls="",
            label="Rational q0: radius endpoint only",
        )
    ]
)
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
fig.suptitle(
    "Amplitude-conditioned trajectories — current-bin membership changes over time"
)
fig.tight_layout(rect=(0, 0.14, 1, 0.94))
for ax in axes[1]:
    values = np.concatenate([line.get_ydata() for line in ax.lines])
    ax.set_ylim(0, max(1e-4, np.nanmax(values) * 1.15))
save(fig, "amplitude-radius-q")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for col, side in enumerate(("up", "down")):
    xs = np.arange(7)
    labels = ["0–.001", ".001–.01", ".01–.03", ".03–.1", ".1–.3", ".3–.6", ".6–1"]
    old_counts = [r["count"] for r in weak["sides"][side]["bins"]]
    assert len(old_counts) == 7
    axes[col].bar(
        xs - 0.27,
        old_counts,
        width=0.18,
        color="#888888",
        label="Rational q0 (saved endpoint)",
    )
    for i, (name, rows) in enumerate(geo.items()):
        values = [
            sum(
                v["bins"][k]["count"]
                for n, v in rows[-1]["layers"].items()
                if n.endswith(".ffn." + side + ".map")
            )
            for k in range(7)
        ]
        assert sum(values) == 16384
        axes[col].bar(
            xs - 0.09 + i * 0.18, values, width=0.18, color=colors[name], label=name
        )
    axes[col].set_xticks(xs, labels, rotation=35, ha="right")
    axes[col].set(
        title="FFN " + side,
        xlabel="Absolute amplitude at update976",
        ylabel="Atom count (4 blocks pooled)",
    )
    axes[col].grid(axis="y", alpha=0.2)
axes[1].legend(frameon=False)
fig.suptitle(
    "Final amplitude populations — sparse bins are not assigned a zero mean radius"
)
save(fig, "amplitude-populations")

# Declared envelope curves, independent of training observations.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for ax, upper_x in zip(axes, (0.15, 1.0)):
    a = np.linspace(0, upper_x, 2001)
    z = (a / 0.3) ** 2
    lo = 0.5 + 3.5 / (1 + 3 * z)
    hi = np.maximum(np.maximum(0.5 + 148.5 / (3 + z), 1.0), lo)
    ax.plot(a, hi, color="#333333", label="Unchanged U0")
    ax.plot(a, lo, "--", color="#888888", label="Old L0")
    for tau, label in [(0.01, "tau=.01"), (0.03, "tau=.03"), (0.1, "tau=.1")]:
        g = a * a / (a * a + tau * tau)
        low = hi - g * (hi - lo)
        ax.plot(a, low, color=colors[label], label=label + " lower")
    ax.set(
        xlabel="Absolute amplitude |a|",
        ylabel="Support radius",
        yscale="log",
        ylim=(0.45, 55),
    )
    ax.grid(alpha=0.2)
axes[0].set_title("Small amplitudes")
axes[1].set_title("Full amplitude domain")
axes[1].legend(frameon=False)
fig.suptitle("Declared envelopes — g=a²/(a²+tau²), L=U0-g(U0-L0), U=U0")
save(fig, "envelope-shape")

result = {}
for name, s in summaries.items():
    result[name] = {
        "DEV_CE": s["latest_development"]["ce"],
        "DEV_PPL": math.exp(s["latest_development"]["ce"]),
        "median_update_seconds": statistics.median(s["timing_samples_seconds"]),
        "peak_allocated_bytes": s["peak_allocated_bytes"],
        "peak_reserved_bytes": s["peak_reserved_bytes"],
        "total_elapsed_seconds": s["total_elapsed_seconds"],
        "historical_reuse": "reused" in name,
    }
result["scope"] = (
    "Exploratory one-seed/short-budget; new versus historical resource measurements are not a fresh paired performance comparison."
)
result["figures_sha256"] = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in figures.iterdir()
    if p.suffix in (".png", ".svg")
}
(b / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
