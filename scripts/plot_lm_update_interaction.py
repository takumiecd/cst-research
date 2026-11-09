"""Render only the completed and hash-verified task-update study."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    backup = read(args.run / "backup-state.json")
    assert backup["state"] == "completed"
    last = backup["snapshots"][-1]
    assert last["state"] == "passed" and sha(args.run / last["name"]) == last["sha256"]
    data = args.run / last["retrieved"]
    manifest = read(data / "manifest.json")
    assert all(sha(data / n) == h for n, h in manifest["files"].items())
    assert sha(args.run / "source.tar.gz") == manifest["source_archive_sha256"]
    with tarfile.open(args.run / "source.tar.gz") as t:
        assert {
            m.name: hashlib.sha256(t.extractfile(m).read()).hexdigest()
            for m in t
            if m.isfile()
        } == manifest["source_files"]
    receipt = read(data / "receipt.json")
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    assert receipt["previous30records_reproduced_exactly"]
    diagnostic = read(data / "diagnostic/summary.json")
    assert diagnostic["state"] == "passed"
    assert len(diagnostic["records"]) == 36
    assert all(v["model_optimizer_rng_restored"] for v in diagnostic["records"])
    assert all(
        v["optimizer_clock_after_restore"]
        == [
            v["checkpoint_step"]
            - (v["checkpoint_step"] // 250 if v["arm"] == "dst-all" else 0)
        ]
        for v in diagnostic["records"]
    )
    assert all(
        v["instrumented_step_equals_unhooked_step"]
        for v in diagnostic["records"]
        if v["batch_offset_steps"] == 0
    )
    rows = diagnostic["records"]
    arms = diagnostic["arms"]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for step in (976, 4882):
            points = [
                v for v in rows if v["arm"] == arm and v["checkpoint_step"] == step
            ]
            assert len(points) == 3
            metrics[arm][str(step)] = {
                "mean_all_parameter_gain": float(
                    np.mean([v["loss_reductions"]["all"] for v in points])
                ),
                "mean_ffn_gain": float(
                    np.mean([v["loss_reductions"]["all_ffn"] for v in points])
                ),
                "mean_ffn_joint_minus_sum_individual_gain": float(
                    np.mean([v["ffn_joint_minus_sum_individual_gain"] for v in points])
                ),
                "map_statistics": {
                    site: {
                        key: float(np.mean([v["maps"][site][key] for v in points]))
                        for key in (
                            "descent_cosine",
                            "predicted_loss_reduction",
                            "relative_output_delta",
                        )
                    }
                    for site in points[0]["maps"]
                },
                "mean_field_only_gain": {
                    field: float(
                        np.mean(
                            [
                                v["before_ce"] - v["fields"][field]["only_field_ce"]
                                for v in points
                            ]
                        )
                    )
                    for field in points[0].get("fields", {})
                },
            }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    for ax, metric, title in zip(
        axes,
        (
            "mean_all_parameter_gain",
            "mean_ffn_gain",
            "mean_ffn_joint_minus_sum_individual_gain",
        ),
        (
            "Same-batch CE reduction: all Parameters",
            "Same-batch CE reduction: FFN only",
            "FFN joint gain minus summed individual gains",
        ),
    ):
        for step, offset, color in ((976, -0.16, "#377eb8"), (4882, 0.16, "#ff7f00")):
            ax.bar(
                np.arange(len(arms)) + offset,
                [metrics[arm][str(step)][metric] for arm in arms],
                width=0.32,
                color=color,
                label=f"checkpoint {step}",
            )
        ax.set_xticks(range(len(arms)), arms, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.7)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Mean of three restored TRAIN updates; finite-step interaction is not a long-term causal test"
    )
    save(fig, "update-task-gains")
    sites = list(rows[0]["maps"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("descent_cosine", "relative_output_delta"),
        ("Actual task descent cosine", "Relative output change on actual saved inputs"),
    ):
        matrix = [
            [metrics[arm]["4882"]["map_statistics"][site][metric] for site in sites]
            for arm in arms
        ]
        extent = float(np.max(np.abs(matrix)))
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm" if metric == "descent_cosine" else "viridis",
            vmin=-extent if metric == "descent_cosine" else 0,
            vmax=extent,
        )
        ax.set_yticks(range(len(arms)), arms)
        ax.set_xticks(range(len(sites)), sites, rotation=30)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(
        "Checkpoint4882; per-map mean of three TRAIN batches; different models have different task states"
    )
    save(fig, "update-alignment")
    fields = ("amplitude", "radius_fraction", "output_center", "input_center")
    cst_arms = [arm for arm in arms if arm.startswith("cst-")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for arm in cst_arms:
        points = [v for v in rows if v["arm"] == arm and v["checkpoint_step"] == 4882]
        unchanged = {
            field: float(
                np.mean(
                    [
                        z["unchanged_fraction"]
                        for point in points
                        for z in point["fields"][field]["parameters"].values()
                    ]
                )
            )
            for field in fields
        }
        metrics[arm]["4882"]["mean_field_unchanged_fraction"] = unchanged
        axes[0].plot(
            fields,
            [metrics[arm]["4882"]["mean_field_only_gain"][field] for field in fields],
            marker="o",
            label=arm,
        )
        axes[1].plot(
            fields, [unchanged[field] * 100 for field in fields], marker="o", label=arm
        )
    axes[0].set_ylabel("Same-batch CE reduction: only this field updated")
    axes[1].set_ylabel("Unchanged entries in the realized update (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Field-only endpoints; CST selected map counts 1/4/4/8; mean of three TRAIN updates at step4882"
    )
    save(fig, "update-field-behavior")
    protocol = {
        k: v
        for k, v in receipt["protocol"].items()
        if k
        not in (
            "remote",
            "base_remote",
            "placement_remote",
            "previous_attempt_remote",
            "python",
        )
    }
    result = {
        "scope": "Analysis of observed CST task-update behavior only; one seed,36 restored TRAIN probes; no optimizer/default recommendation",
        "protocol": protocol,
        "compact_sha256": last["sha256"],
        "source_files_verified": len(manifest["source_files"]),
        "files_verified": len(manifest["files"]),
        "diagnostic_records": 36,
        "diagnostic": metrics,
        "previous36functional_records_reproduced_exactly": receipt[
            "previous36functional_records_reproduced_exactly"
        ],
        "optimization_screen_scope": "Ten short continuations had already completed when owner clarified analysis-only intent. Preserved separately; omitted from behavior analysis and no recipe selected/adopted.",
        "plotter_sha256": sha(Path(__file__)),
        "figures": {p.name: sha(p) for p in sorted(out.glob("*.png"))},
    }
    (out / "behavior-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "diagnostic_records": len(rows),
                "figures": len(result["figures"]),
                "scope": "analysis only",
            }
        )
    )


if __name__ == "__main__":
    main()
