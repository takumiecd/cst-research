"""Plot verified lower-tau/broad-start evidence; selection is fixed by protocol."""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ffn_lr_report_inputs import load_inputs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--recovery-run", type=Path)
    parser.add_argument("--dst-run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    assert (
        json.loads((root / "retrieval-integrity.json").read_text())["state"] == "passed"
    )
    read = lambda p: json.loads(p.read_text())
    lines = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
    rows, directories = load_inputs(root, args.recovery_run, args.dst_run)
    if any(r.get("tau_comparison") for r in rows) and args.dst_run is None:
        parser.error(
            "This tau comparison requires the source-qualified --dst-run baseline"
        )
    out = root / "figures"
    out.mkdir(exist_ok=True)
    summaries = {
        r["label"]: read(directories[r["label"]] / "summary.json") for r in rows
    }
    curves = {r["label"]: lines(directories[r["label"]] / "curves.jsonl") for r in rows}

    def name(row):
        if row["kind"] == "dense":
            return (
                "Dense (recovery)" if row["label"].startswith("recovery-") else "Dense"
            )
        if row["kind"] == "dst":
            return f"DST (RigL-style, LR{row['sparse_peak_lr']:g}; DEV selected)"
        if row.get("tau_comparison"):
            return f"CST tau={row['tau']:g}, f0=0, LR32x"
        if row.get("fraction_start_comparison"):
            return (
                f"fraction init{row['fraction']:g}, CST LR{row['cst_lr_multiplier']:g}x"
            )
        if "cst_lr_multiplier" in row:
            return f"all CST fields LR{row['cst_lr_multiplier']:g}x"
        if "fraction_lr_multiplier" in row:
            return f"fraction LR{row['fraction_lr_multiplier']:g}x"
        if row["amplitude_max"] == 1:
            return "cap1 bridge: tau=.003, f0=0"
        return f"tau={row['tau']:g}, f0={row['fraction']:g}"

    names = list(dict.fromkeys(name(r) for r in rows))
    colors = {n: plt.get_cmap("tab10")(i) for i, n in enumerate(names)}
    for r in rows:
        if r["kind"] == "dst":
            colors[name(r)] = "#2ca02c"
    for tau, color in zip(
        (0.0003, 0.001, 0.003, 0.01), ("#1f77b4", "#ff7f0e", "#d62728", "#9467bd")
    ):
        colors[f"CST tau={tau:g}, f0=0, LR32x"] = color
    colors["Dense"] = "#222222"
    colors["Dense (recovery)"] = "#777777"
    colors["cap1 bridge: tau=.003, f0=0"] = "#999999"
    colors["fraction init0, CST LR32x"] = "#1f77b4"
    colors["fraction init0.5, CST LR32x"] = "#ff7f0e"
    for rate, color in zip(
        (16, 32, 64, 128), ("#1f77b4", "#ff7f0e", "#d62728", "#9467bd")
    ):
        colors[f"all CST fields LR{rate}x"] = color

    def save(fig, stem):
        for ext in ("png", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    for rate, color in [(1.0, "#1f77b4"), (4.0, "#ff7f0e"), (16.0, "#d62728")]:
        colors[f"fraction LR{rate:g}x"] = color
    for stage in dict.fromkeys(r["stage"] for r in rows):
        selected = [r for r in rows if r["stage"] == stage]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        for row in selected:
            label = name(row)
            hist = curves[row["label"]]
            train = [p for p in hist if "train_ce" in p]
            dev = [p for p in hist if "development" in p]
            style = (
                "--"
                if row["amplitude_max"] == 1
                or (row["kind"] == "dense" and row["label"].startswith("recovery-"))
                else "-"
            )
            for ax, points, key, scale in [
                (axes[0, 0], train, "step", 1),
                (axes[0, 1], dev, "step", 1),
                (axes[1, 0], train, "elapsed_seconds", 60),
                (axes[1, 1], dev, "elapsed_seconds", 60),
            ]:
                ax.plot(
                    [p[key] / scale for p in points],
                    [
                        p["train_ce"] if "train_ce" in p else p["development"]["ce"]
                        for p in points
                    ],
                    style,
                    color=colors[label],
                    label=label,
                    lw=1.5,
                )
        for j, title in enumerate(("Train CE", "Development CE")):
            axes[0, j].set_title(title)
            axes[0, j].set_xlabel(
                "Training steps (DST topology events included)"
                if args.dst_run
                else "Optimizer updates"
            )
            axes[1, j].set_xlabel(
                "Elapsed wall minutes, includes setup/probes/evaluation"
            )
            for ax in axes[:, j]:
                ax.set_ylabel("Cross entropy")
                ax.grid(alpha=0.2)
        axes[0, 1].legend(fontsize=8, ncol=2)
        fig.suptitle(
            "Zero-start tau comparison with DEV-selected DST, common CST LR32x, seed17"
            if any(r.get("tau_comparison") for r in rows)
            else "Initial fraction0 versus.5, common CST LR32x, seed17 only"
            if any(r.get("fraction_start_comparison") for r in rows)
            else "All CST field LR comparison, seed17 only"
            if any("cst_lr_multiplier" in r for r in rows)
            else "Direct fraction LR comparison, seed17 only"
            if any("fraction_lr_multiplier" in r for r in rows)
            else f"{stage.capitalize()} lower-tau and broad-start exploration, seed17 only"
        )
        save(fig, f"{stage}-learning-curves")
        if args.dst_run:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
            for ax, start in zip(axes, (200, 600)):
                for row in selected:
                    points = [
                        r
                        for r in curves[row["label"]]
                        if "development" in r and r["step"] >= start
                    ]
                    ax.plot(
                        [r["step"] for r in points],
                        [r["development"]["ce"] for r in points],
                        color=colors[name(row)],
                        label=name(row),
                        lw=1.5,
                    )
                ax.set_xlabel("Training steps (DST includes topology events)")
                ax.set_ylabel("Development cross entropy")
                ax.set_title(f"Development CE, from step{start}")
                ax.grid(alpha=0.2)
            axes[0].legend(fontsize=7)
            fig.suptitle("Tail zoom, seed17; DST selected on DEV, no confirmation")
            save(fig, f"{stage}-development-tail")
        cst = [r for r in selected if r["kind"] == "cst"]
        geo = {
            r["label"]: lines(directories[r["label"]] / "geometry.jsonl") for r in cst
        }
        sites = list(next(iter(geo.values()))[0]["layers"])
        for metric in ("radius", "radius_fraction", "fraction_change"):
            fig, axes = plt.subplots(4, 2, figsize=(12, 12), layout="constrained")
            for ax, site in zip(axes.flat, sites):
                for row in cst:
                    entries = geo[row["label"]]
                    key = "radius_fraction" if metric == "fraction_change" else metric
                    offset = row["fraction"] if metric == "fraction_change" else 0
                    ax.plot(
                        [e["step"] for e in entries],
                        [
                            e["layers"][site]["all"][key]["mean"] - offset
                            for e in entries
                        ],
                        color=colors[name(row)],
                        label=name(row),
                    )
                if metric == "radius":
                    ax.set_yscale("log")
                ax.set_title(site)
                ax.set_xlabel("Updates")
                ax.grid(alpha=0.2)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(f"{stage.capitalize()} {metric}; every FFN site")
            save(fig, f"{stage}-{metric}-layers")
        common = [r for r in cst if r["amplitude_max"] == 32]
        count = len(geo[common[0]["label"]][0]["layers"][sites[0]]["bins"])
        for metric in ("radius", "radius_fraction"):
            fig, axes = plt.subplots(
                math.ceil(count / 4),
                4,
                figsize=(15, 3.4 * math.ceil(count / 4)),
                layout="constrained",
            )
            for i, ax in enumerate(axes.flat):
                if i >= count:
                    ax.axis("off")
                    continue
                for row in common:
                    xs, ys = [], []
                    for entry in geo[row["label"]]:
                        bins = [v["bins"][i] for v in entry["layers"].values()]
                        n = sum(b["count"] for b in bins)
                        xs.append(entry["step"])
                        ys.append(
                            sum(
                                b["count"] * b[metric]["mean"]
                                for b in bins
                                if b["count"]
                            )
                            / n
                            if n
                            else np.nan
                        )
                    ax.plot(xs, ys, color=colors[name(row)], label=name(row), lw=1.3)
                if not any(np.isfinite(line.get_ydata()).any() for line in ax.lines):
                    ax.text(
                        0.5,
                        0.5,
                        "No Atoms in this bin",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                b = geo[common[0]["label"]][0]["layers"][sites[0]]["bins"][i]
                ax.set_title(f"{b['low']:g} <= |a| < {b['high']:g}")
                ax.set_xlabel("Updates")
                ax.set_ylabel(metric)
                if metric == "radius":
                    ax.set_yscale("log")
                ax.grid(alpha=0.2)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(
                f"{stage.capitalize()} current-amplitude {metric}, common cap32"
            )
            save(fig, f"{stage}-amplitude-{metric}")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for row in cst:
            entries = lines(directories[row["label"]] / "radius.jsonl")
            for k, ax in enumerate(axes.flat):
                ax.plot(
                    [e["step"] for e in entries],
                    [
                        np.sqrt(
                            np.mean(
                                [
                                    v["field_gradient_rms"][k] ** 2
                                    for v in e["layers"].values()
                                ]
                            )
                        )
                        for e in entries
                    ],
                    color=colors[name(row)],
                    label=name(row),
                )
                ax.set_title(
                    ["Amplitude", "Radius fraction", "Output center", "Input center"][k]
                )
                ax.set_xlabel("Updates")
                ax.set_yscale("log")
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            f"{stage.capitalize()} fixed-probe task gradients, unscaled/unclipped"
        )
        save(fig, f"{stage}-field-gradients")
        dense = next(r for r in selected if r["kind"] == "dense")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
        for ax, phase in zip(axes, ("initial", "final")):
            sites = list(
                summaries[dense["label"]]["diagnostics"][phase]["activation_rms"]
            )
            baseline = summaries[dense["label"]]["diagnostics"][phase]["activation_rms"]
            values = np.array(
                [
                    [
                        summaries[r["label"]]["diagnostics"][phase]["activation_rms"][s]
                        / baseline[s]
                        for s in sites
                    ]
                    for r in selected
                ]
            )
            im = ax.imshow(
                np.log2(values), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto"
            )
            for i in range(len(selected)):
                for j in range(len(sites)):
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            ax.set_yticks(range(len(selected)), [name(r) for r in selected], fontsize=8)
            ax.set_xticks(
                range(len(sites)),
                [s.replace("blocks.", "b").replace(".ffn.", " ") for s in sites],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            ax.set_title(f"{phase.capitalize()} actual activation RMS / Dense")
        fig.colorbar(
            im,
            ax=axes,
            label="log2 RMS ratio (moment targets match; activations need not)",
        )
        save(fig, f"{stage}-activation-rms")
    table = []
    for row in rows:
        s = summaries[row["label"]]
        record = {
            **row,
            "median_update_ms": statistics.median(s["timing_samples_seconds"]) * 1000,
            "setup_seconds": s["setup_seconds"],
            "total_wall_seconds": s["total_elapsed_seconds"],
            "ffn_parameters": s["actual_ffn_parameters"],
            "initial_activation_rms": s["diagnostics"]["initial"]["activation_rms"],
            "final_activation_rms": s["diagnostics"]["final"]["activation_rms"],
        }
        if row["kind"] == "cst":
            hist = lines(directories[row["label"]] / "geometry.jsonl")
            record["initial_radius_mean"] = statistics.mean(
                v["all"]["radius"]["mean"] for v in hist[0]["layers"].values()
            )
            last = [v["all"] for v in hist[-1]["layers"].values()]
            record["final_radius_mean"] = statistics.mean(
                v["radius"]["mean"] for v in last
            )
            record["final_fraction_mean"] = statistics.mean(
                v["radius_fraction"]["mean"] for v in last
            )
            record["fraction_at_zero"] = statistics.mean(
                v["fraction_at_zero"] for v in last
            )
            record["fraction_at_one"] = statistics.mean(
                v["fraction_at_one"] for v in last
            )
        table.append(record)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "runs": table,
                "scope": (
                    "One development seed, zero-start tau family and DEV-selected DST; no optimality or convergence claim"
                    if any(r.get("tau_comparison") for r in rows)
                    else "One development seed, initial fraction0/.5 at common32x; no optimality or convergence claim"
                    if any(r.get("fraction_start_comparison") for r in rows)
                    else "One development seed, fixed all-CST-field LR family; no optimality or convergence claim"
                    if any("cst_lr_multiplier" in r for r in rows)
                    else "One development seed, fixed fraction LR family; no optimality or convergence claim"
                    if any("fraction_lr_multiplier" in r for r in rows)
                    else "One development seed, selected after short screening; no optimality, convergence or causal initial-radius claim"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
