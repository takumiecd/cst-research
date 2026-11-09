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
    screen = read(data / "screen/summary.json")
    assert diagnostic["state"] == screen["state"] == "passed"
    assert len(diagnostic["records"]) == 36 and len(screen["results"]) == 10
    assert all(v["model_optimizer_rng_restored"] for v in diagnostic["records"])
    assert all(
        v["instrumented_step_equals_unhooked_step"]
        for v in diagnostic["records"]
        if v["batch_offset_steps"] == 0
    )
    for row in screen["results"]:
        assert (
            row["state"] == "passed"
            and row["optimizer_audit_passed"]
            and row["serialized_update_exact"]
        )
        assert (
            row["final_checkpoint_sha256"]
            == manifest["checkpoints"][
                "screen/" + row["label"] + "/checkpoint-1523712-tokens.pt"
            ]
        )
        assert [v["step"] for v in row["development"]] == [1200, 1400, 1488]
        assert all(
            o["atol"] == 1e-4 and o["rtol"] == 4e-3 for o in row["oracles"].values()
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
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
        ax.set_yticks(range(len(arms)), arms)
        ax.set_xticks(range(len(sites)), sites, rotation=30)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(
        "Checkpoint4882; per-map mean of three TRAIN batches; different models have different task states"
    )
    save(fig, "update-alignment")
    conditions = screen["results"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for row in conditions:
        points = [{"step": 976, **row["baseline_development"]}, *row["development"]]
        style = (
            {"color": "#222222", "linewidth": 2.3}
            if row["label"] == "dense-control"
            else {"color": "#2ca02c", "linewidth": 2.3}
            if row["label"] == "dst-control"
            else {}
        )
        for ax, field, scale in (
            (axes[0], "ce", 1),
            (axes[1], "next_token_top1_accuracy", 100),
        ):
            ax.plot(
                [v["step"] * 1024 / 1e6 for v in points],
                [v[field] * scale for v in points],
                marker="o",
                label=row["label"],
                **style,
            )
    axes[0].set_ylabel("DEV cross entropy")
    axes[1].set_ylabel("DEV next-token top1 (%)")
    for ax in axes:
        ax.set_xlabel("Total training tokens (millions)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Same-state512-update continuation screen; original20M horizons; all candidates retained"
    )
    save(fig, "continuation-dev-curves")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    reference = next(v for v in conditions if v["label"] == "cst-control32")[
        "development"
    ][-1]["ce"]
    labels = [v["label"] for v in conditions]
    ax.barh(
        labels,
        [v["development"][-1]["ce"] - reference for v in conditions],
        color=[
            "#222222"
            if x.startswith("dense")
            else "#2ca02c"
            if x.startswith("dst")
            else "#377eb8"
            for x in labels
        ],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Final DEV CE minus CST control32; lower is better")
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Exploratory single-state screening; no independent confirmation")
    save(fig, "continuation-final-gaps")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for row in conditions:
        if not row["geometry"]:
            continue
        for ax, field in zip(axes, ("radius", "radius_fraction")):
            ax.plot(
                [v["step"] * 1024 / 1e6 for v in row["geometry"]],
                [
                    np.mean(
                        [site["all"][field]["mean"] for site in v["layers"].values()]
                    )
                    for v in row["geometry"]
                ],
                marker="o",
                label=row["label"],
            )
            ax.set(
                xlabel="Total training tokens (millions)",
                ylabel="Selected-map mean " + field,
            )
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Geometry under the current amplitudes; CST only")
    save(fig, "continuation-geometry")
    result = {
        "scope": "One-seed restored-state task update diagnosis and512-update DEV screening, no teacher/confirmation/default adoption",
        "protocol": receipt["protocol"],
        "compact_sha256": last["sha256"],
        "source_files_verified": len(manifest["source_files"]),
        "files_verified": len(manifest["files"]),
        "previous30records_reproduced_exactly": True,
        "diagnostic": metrics,
        "continuations": conditions,
        "plotter_sha256": sha(Path(__file__)),
        "figures": {p.name: sha(p) for p in sorted(out.glob("*.png"))},
    }
    # Public provenance uses external evidence locators, never private host paths.
    result["protocol"] = {
        k: v
        for k, v in result["protocol"].items()
        if k
        not in (
            "remote",
            "base_remote",
            "placement_remote",
            "previous_attempt_remote",
            "python",
        )
    }
    (out / "update-interaction-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "conditions": len(conditions),
                "diagnostic_records": len(rows),
                "figures": len(result["figures"]),
            }
        )
    )


if __name__ == "__main__":
    main()
