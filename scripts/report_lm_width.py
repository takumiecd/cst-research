"""Read-only live/final overlays; retains every short-run LR candidate."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cst_research.experiments.lm_width_contract import ARMS, WIDTHS


def read(path):
    return json.loads(path.read_text())


def render(root, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    gapfig, gapaxes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    records = {}
    for column, width in enumerate(WIDTHS):
        base = root / f"width{width}"
        selection = base / "selection/selection.json"
        selected = set(read(selection)["selected"]) if selection.exists() else set()
        rows_by_arm = {}
        for arm, (kind, _) in ARMS.items():
            directory = base / "long" / arm
            if not (directory / "curves.jsonl").exists():
                directory = base / "short" / arm
            path = directory / "curves.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            summary = read(directory / "summary.json")
            assert summary["width"] == width and summary["arm"] == arm
            assert summary["config"]["coefficient"] == 64
            assert summary["effective_tokens_per_step"] == 1024
            assert all(r["seen_tokens"] == 1024 * r["step"] for r in rows)
            assert all(b["step"] > a["step"] for a, b in pairwise(rows))
            if kind != "dense":
                assert set(summary["active_slots"].values()) == {128 * width}
            records[f"{width}/{arm}"] = {
                "state": summary["state"],
                "completed_steps": summary["completed_steps"],
                "peak_allocated_bytes": summary["peak_allocated_bytes"],
                "peak_reserved_bytes": summary["peak_reserved_bytes"],
                "development": [r for r in rows if "development" in r],
            }
            rows_by_arm[arm] = {
                r["step"]: r["development"] for r in rows if "development" in r
            }
            color = {"dense": "#333333", "cst": "#d3522e", "dst": "#2878b5"}[kind]
            prominent = arm in selected or kind == "cst"
            alpha = 1 if prominent else 0.35
            line = "-" if prominent else ":"
            dev = [r for r in rows if "development" in r]
            axes[0, column].plot(
                [r["seen_tokens"] / 1e6 for r in dev],
                [r["development"]["ce"] for r in dev],
                line,
                color=color,
                alpha=alpha,
                label=arm,
                lw=2 if prominent else 1,
            )
            training = [r for r in rows if "train_ce" in r]
            axes[1, column].plot(
                [r["seen_tokens"] / 1e6 for r in training],
                [r["train_ce"] for r in training],
                line,
                color=color,
                alpha=alpha,
                lw=1,
            )
        for name, ax in zip(
            ("Validation CE (story-disjoint)", "TRAIN minibatch CE"), axes[:, column]
        ):
            ax.set_title(f"Width {width}: {name}")
            ax.set_xlabel("Training targets (millions)")
            ax.set_ylabel("Cross-entropy")
            ax.grid(alpha=0.2)
            if column < 3:
                ax.set_xlim(0, 5.05)
        if rows_by_arm:
            axes[0, column].legend(fontsize=7)
        dense = next((a for a in selected if ARMS[a][0] == "dense"), None)
        ax = gapaxes[column]
        if dense in rows_by_arm:
            for arm in sorted(selected - {dense}):
                if arm not in rows_by_arm:
                    continue
                steps = sorted(set(rows_by_arm[arm]) & set(rows_by_arm[dense]) - {0})
                ax.plot(
                    [s * 1024 / 1e6 for s in steps],
                    [
                        rows_by_arm[arm][s]["ce"] - rows_by_arm[dense][s]["ce"]
                        for s in steps
                    ],
                    marker=".",
                    label=arm,
                )
        ax.axhline(0, color="black", lw=1)
        ax.axhline(0.01, color="gray", ls="--", label="Descriptive +0.01 guide")
        ax.set(
            title=f"Width {width}: gap to DEV-selected Dense",
            xlabel="Training targets (millions)",
            ylabel="Validation CE minus Dense",
            xlim=(0, 5.05),
        )
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.suptitle("Fixed c64 M8 FFN width study — live pilot; seed17; DEV-selected LRs")
    gapfig.suptitle(
        "Matched-token validation gaps; no convergence or equivalence claim"
    )
    for f, name in ((fig, "learning"), (gapfig, "gaps")):
        for suffix in ("png", "svg"):
            f.savefig(output / f"{name}.{suffix}", dpi=160)
        plt.close(f)
    (output / "progress.json").write_text(
        json.dumps(
            {
                "scope": "live pilot; all short LR candidates retained; long winners chosen at976 steps; confirmation untouched",
                "arms": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render(a.run, a.output)
