"""Selected witness medians: degree and connection, not population inference."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    v = json.loads(a.input.read_text())
    assert v["state"] == "passed" and len(v["records"]) == 36
    a.output.mkdir(parents=True, exist_ok=False)
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    colors = ("#627eaa", "#185f9e", "#b87248", "#8c2d16")
    for row, arm in enumerate(("all-cst", "pair12")):
        for col, kind in enumerate(
            ("sharp_both_near_grid", "sharp_both_away_grid", "wide")
        ):
            records = [r for r in v["records"] if r["arm"] == arm and r["kind"] == kind]
            results = {}
            ax = axes[row, col]
            for index, (m, c) in enumerate(
                ((8, False), (8, True), (16, False), (16, True))
            ):
                grad = [
                    g
                    for r in records
                    for cell in r["cells"]
                    if cell["modes"] == m and cell["connected"] == c
                    for g in cell["raw_gradients"]
                ]
                median = [statistics.median(abs(g[k]) for g in grad) for k in range(4)]
                label = f"M{m} {'connected' if c else 'unconnected'}"
                results[label] = {
                    "witnesses": len(grad),
                    "median_abs_raw_gradient": median,
                }
                ax.bar(
                    np.arange(4) + (index - 1.5) * 0.19,
                    median,
                    width=0.19,
                    label=label,
                    color=colors[index],
                )
            summary[f"{arm}/{kind}"] = results
            ax.set_yscale("log")
            ax.set_xticks(
                range(4),
                ("amplitude", "fraction", "out center", "in center"),
                rotation=20,
            )
            ax.set_title(
                f"{arm}: {kind.replace('_', ' ')}\n{len(grad)} selected witnesses",
                fontsize=10,
            )
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("median absolute raw task VJP")
    axes[1, 0].set_ylabel("median absolute raw task VJP")
    fig.suptitle(
        "Identical raw atoms, TRAIN inputs and cotangents; CPU FP64 factors\nSelected final M16 witnesses, not a population or retrained-M8 estimate",
        fontsize=12,
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        target = a.output / ("profile-sensitivity." + suffix)
        fig.savefig(target, dpi=160)
        if suffix == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)
    (a.output / "summary.json").write_text(
        json.dumps(
            {"scope": v["scope"], "sampling": v["sampling"], "summary": summary},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
