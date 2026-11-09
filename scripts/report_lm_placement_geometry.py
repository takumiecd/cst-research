"""Derived-only geometry summaries and a figure, excluding raw witness rows."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.evidence / "geometry-manifest.json").read_text())
    assert all(
        hashlib.sha256((a.evidence / n).read_bytes()).hexdigest() == h
        for n, h in manifest["files"].items()
    )
    g = json.loads((a.evidence / "geometry/geometry.json").read_text())
    assert g["state"] == "passed" and len(g["records"]) == 40
    arms = (
        "pair01",
        "pair02",
        "pair03",
        "pair12",
        "pair13",
        "pair23",
        "up-even",
        "down-even",
        "all-cst",
    )
    result = {
        "source_commits": g["source_commits"],
        "diagnostic_script_sha256": g["script_sha256"],
        "scope": "post-hoc original offset0 TRAIN VJP; weighted subset shares; FP64 witnesses are selected examples, not population samples",
        "arms": {},
    }
    for arm in arms:
        rows = [r for r in g["records"] if r["arm"] == arm]
        groups = {}
        for group in rows[0]["subsets"]:
            entries = [
                r["subsets"][group] for r in rows if r["subsets"][group]["count"]
            ]
            count = sum(e["count"] for e in entries)
            row = {"count": count}
            if count:
                row["fields"] = {
                    f: {
                        k: sum(e["count"] * e["fields"][f][k] for e in entries) / count
                        for k in (
                            "task_gradient_below_1e-8_share",
                            "task_gradient_below_1e-10_share",
                            "adam_scale_below_epsilon_share",
                        )
                    }
                    for f in (
                        "amplitude",
                        "radius_fraction",
                        "output_center",
                        "input_center",
                    )
                }
            groups[group] = row
        samples = {}
        for kind in ("sharp_both_near_grid", "sharp_both_away_grid", "wide"):
            values = [
                v
                for r in rows
                for s in r["samples"]
                if s["kind"] == kind
                for v in s["FP64_factor_task_raw_gradients"]
            ]
            samples[kind] = {
                "count": len(values),
                "median_absolute_raw_gradients": {
                    field: statistics.median(abs(v[k]) for v in values)
                    if values
                    else None
                    for k, field in enumerate(
                        (
                            "amplitude",
                            "radius_fraction",
                            "output_center",
                            "input_center",
                        )
                    )
                },
            }
        result["arms"][arm] = {"subsets": groups, "selected_FP64_witnesses": samples}
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    groups = (
        ("active_sharp_output_center_near_grid", "r<0.6, center within0.01 of grid"),
        ("active_sharp_output_center_far_grid", "r<0.6, center >0.1 from grid"),
        ("active_nonsharp", "r>=0.6"),
    )
    x = np.arange(len(arms))
    for i, (group, label) in enumerate(groups):
        values = [
            100
            * result["arms"][arm]["subsets"][group]["fields"]["output_center"][
                "task_gradient_below_1e-8_share"
            ]
            for arm in arms
        ]
        axes[0].bar(x + (i - 1) * 0.25, values, width=0.25, label=label)
    axes[0].set_xticks(x, arms, rotation=35, ha="right")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Output-center task |gradient| <1e-8 (%)")
    axes[0].set_title(
        "Small coordinate gradients concentrate in sharp, grid-aligned atoms"
    )
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.25))
    kinds = ("sharp_both_near_grid", "sharp_both_away_grid", "wide")
    for field in ("amplitude", "radius_fraction", "output_center", "input_center"):
        values = [
            result["arms"]["all-cst"]["selected_FP64_witnesses"][k][
                "median_absolute_raw_gradients"
            ][field]
            for k in kinds
        ]
        axes[1].plot(range(3), values, "o-", label=field)
    axes[1].axhline(1e-8, color="gray", ls="--", label="Adam epsilon reference")
    axes[1].set_yscale("log")
    axes[1].set_xticks(
        range(3),
        ("Sharp, both centers near grid", "Sharp, both away from grid", "Radius>=16"),
        rotation=15,
        ha="right",
    )
    axes[1].set_ylabel("Median |raw gradient| of selected FP64 witnesses")
    axes[1].set_title("All-CST: amplitude gradients remain while geometry becomes flat")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Final saved task, active |amplitude|>=0.01; raw field units differ; no optimizer updates"
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            a.output / f"geometry-gradients.{ext}", dpi=160, bbox_inches="tight"
        )
    plt.close(fig)
    (a.output / "geometry-summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
