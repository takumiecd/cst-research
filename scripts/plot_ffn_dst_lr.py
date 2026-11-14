"""Plot every verified DST LR candidate, retaining development-selection scope."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert (
        json.loads((a.run / "retrieval-integrity.json").read_text())["state"]
        == "passed"
    )
    data = a.run / "retrieved"
    receipt = json.loads((data / "receipt.json").read_text())
    assert receipt["status"] == "passed"
    a.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for i, row in enumerate(receipt["results"]):
        chosen = row["label"] == receipt["selection"]["label"]
        label = (
            "Dense"
            if row["kind"] == "dense"
            else f"DST LR{row['sparse_peak_lr']:g}"
            + (" (DEV selected)" if chosen else "")
        )
        hist = [
            json.loads(x)
            for x in (data / row["label"] / "curves.jsonl").read_text().splitlines()
        ]
        color = "#222222" if row["kind"] == "dense" else plt.get_cmap("tab10")(i)
        for j, key in enumerate(("train_ce", "development")):
            points = [r for r in hist if key in r]
            ys = [r[key] if key == "train_ce" else r[key]["ce"] for r in points]
            for ax, xs in (
                (axes[0, j], [r["step"] for r in points]),
                (axes[1, j], [r["elapsed_seconds"] / 60 for r in points]),
            ):
                ax.plot(xs, ys, label=label, color=color, lw=2.2 if chosen else 1.4)
                ax.grid(alpha=0.2)
                ax.set_ylabel("Cross entropy")
    for j, t in enumerate(("Train CE", "Development CE")):
        axes[0, j].set_title(t)
        axes[0, j].set_xlabel("Training steps (DST Adam clock skips250/500)")
        axes[1, j].set_xlabel("Elapsed wall minutes, includes setup/probes/evaluation")
        for k in (250, 500):
            axes[0, j].axvline(k, color="#999999", lw=0.7, ls=":")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(
        "Row-balanced RigL-style LR tuning, K=4A, seed17; DEV selection, not confirmation"
    )
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"dst-lr-curves.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
