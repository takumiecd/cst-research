"""Plot verified radius-fraction tau evidence; no training or result selection."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    integrity = json.loads((root / "retrieval-integrity.json").read_text())
    assert integrity["state"] == "passed"
    data = root / "retrieved"
    receipt = json.loads((data / "receipt.json").read_text())
    assert receipt["status"] == "passed" and receipt["source_unchanged"]
    out = args.output or root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = receipt["results"]
    read = lambda p: json.loads(p.read_text())
    lines = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
    summaries = {r["label"]: read(data / r["label"] / "summary.json") for r in rows}
    curves = {r["label"]: lines(data / r["label"] / "curves.jsonl") for r in rows}
    colors = {
        tau: plt.get_cmap("tab10")(i)
        for i, tau in enumerate(receipt["protocol"]["taus"])
    }

    def name(row):
        return "Dense" if row["kind"] == "dense" else f"tau={row['tau']:g}"

    def color(row):
        return "#333333" if row["kind"] == "dense" else colors[row["tau"]]

    def save(fig, stem):
        for ext in ("png", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    for stage in ("short", "long"):
        selected = [r for r in rows if r["stage"] == stage]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for row in selected:
            data_rows = curves[row["label"]]
            train = [x for x in data_rows if "train_ce" in x]
            dev = [x for x in data_rows if "development" in x]
            label = f"{name(row)}, seed{row['seed']}"
            style = {17: "-", 29: "--", 43: ":"}[row["seed"]]
            for ax, points, key, scale in [
                (axes[0, 0], train, "step", 1),
                (axes[0, 1], dev, "step", 1),
                (axes[1, 0], train, "elapsed_seconds", 60),
                (axes[1, 1], dev, "elapsed_seconds", 60),
            ]:
                vals = [
                    p["development"]["ce"] if "development" in p else p["train_ce"]
                    for p in points
                ]
                ax.plot(
                    [p[key] / scale for p in points],
                    vals,
                    style,
                    color=color(row),
                    label=label,
                    alpha=0.85,
                    lw=1.4,
                )
        for j, title in enumerate(("Train cross entropy", "Development cross entropy")):
            axes[0, j].set_title(title)
            axes[0, j].set_xlabel("Optimizer updates")
            axes[1, j].set_xlabel("Elapsed wall minutes (includes probes/evaluation)")
            for ax in axes[:, j]:
                ax.set_ylabel("Cross entropy")
                ax.grid(alpha=0.2)
        axes[0, 1].legend(fontsize=8, ncol=2)
        fig.suptitle(
            f"{stage.capitalize()} FFN comparison; development-selected family, not convergence"
        )
        save(fig, f"{stage}-learning-curves")
    # Every actual FFN site, and current-amplitude conditional bins. Seed17 only
    # avoids conflating seed spread with amplitude-bin membership changes.
    for stage in ("short", "long"):
        selected = [
            r
            for r in rows
            if r["stage"] == stage and r["kind"] == "cst" and r["seed"] == 17
        ]
        geo = {
            r["label"]: lines(data / r["label"] / "geometry.jsonl") for r in selected
        }
        layer_names = list(next(iter(geo.values()))[0]["layers"])
        for key, title in [
            ("radius", "Physical radius"),
            ("radius_fraction", "Radius fraction"),
        ]:
            fig, axes = plt.subplots(4, 2, figsize=(12, 12), layout="constrained")
            for ax, layer in zip(axes.flat, layer_names):
                for row in selected:
                    entries = geo[row["label"]]
                    ax.plot(
                        [x["step"] for x in entries],
                        [x["layers"][layer]["all"][key]["mean"] for x in entries],
                        color=color(row),
                        label=name(row),
                    )
                ax.set_title(layer)
                ax.set_xlabel("Optimizer updates")
                ax.grid(alpha=0.2)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"{stage.capitalize()} seed17: {title}, all eight sites")
            save(fig, f"{stage}-{key}-layers")
        # Current-amplitude bins, equal-Atom weighting over all eight sites.
        for metric, title in [
            ("radius", "Mean radius"),
            ("radius_fraction", "Mean radius fraction"),
        ]:
            fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
            for i, ax in enumerate(axes.flat):
                if i >= 7:
                    ax.axis("off")
                    continue
                for row in selected:
                    entries = geo[row["label"]]
                    xs, ys = [], []
                    for entry in entries:
                        bins = [v["bins"][i] for v in entry["layers"].values()]
                        count = sum(b["count"] for b in bins)
                        xs.append(entry["step"])
                        ys.append(
                            sum(
                                b["count"] * b[metric]["mean"]
                                for b in bins
                                if b["count"]
                            )
                            / count
                            if count
                            else np.nan
                        )
                    ax.plot(xs, ys, color=color(row), label=name(row))
                b = next(iter(geo.values()))[0]["layers"][layer_names[0]]["bins"][i]
                ax.set_title(f"{b['low']:g} <= |a| < {b['high']:g}")
                ax.set_xlabel("Updates")
                ax.set_ylabel(title)
                ax.grid(alpha=0.2)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(
                f"{stage.capitalize()} seed17: amplitude-conditioned {metric} (current membership)"
            )
            save(fig, f"{stage}-amplitude-{metric}")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for row in selected:
            entries = lines(data / row["label"] / "radius.jsonl")
            for k, ax in enumerate(axes.flat):
                ax.plot(
                    [e["step"] for e in entries],
                    [
                        np.sqrt(
                            np.mean(
                                [
                                    layer["field_gradient_rms"][k] ** 2
                                    for layer in e["layers"].values()
                                ]
                            )
                        )
                        for e in entries
                    ],
                    color=color(row),
                    label=name(row),
                )
                ax.set_title(
                    ["Amplitude", "Radius fraction", "Output center", "Input center"][k]
                )
                ax.set_yscale("log")
                ax.set_xlabel("Updates")
                ax.set_ylabel("Task gradient RMS")
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"{stage.capitalize()} seed17: fixed-probe latent gradients (unclipped)"
        )
        save(fig, f"{stage}-field-gradients")
    table = []
    for label in sorted({name(r) for r in rows if r["stage"] == "long"}):
        group = [r for r in rows if r["stage"] == "long" and name(r) == label]
        ces = [r["development_ce"] for r in group]
        times = [
            statistics.median(summaries[r["label"]]["timing_samples_seconds"]) * 1000
            for r in group
        ]
        table.append(
            {
                "arm": label,
                "seeds": [r["seed"] for r in group],
                "development_ce": ces,
                "mean_ce": statistics.mean(ces),
                "sample_std_ce": statistics.stdev(ces),
                "median_update_ms_by_seed": times,
                "peak_allocated_mib_by_seed": [
                    r["peak_allocated_bytes"] / 2**20 for r in group
                ],
                "peak_reserved_mib_by_seed": [
                    r["peak_reserved_bytes"] / 2**20 for r in group
                ],
            }
        )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "long": table,
                "scope": "Three development seeds; selected after short screening; no universal optimum, convergence, noninferiority or confirmation-set claim",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
