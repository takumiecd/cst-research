"""Verified matched task curves and read-only update-fidelity plots."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dst-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = read(a.run / "export-manifest.json")
    assert all(sha(a.run / n) == h for n, h in manifest["files"].items())
    receipt = read(a.run / "receipt.json")
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    labels = {
        "dense": "Dense",
        "dst-all": "DST",
        "legacy-m8": "Legacy M8",
        "connected-m16": "Connected M16",
    }
    colors = {
        "dense": "#555555",
        "dst-all": "#d78c20",
        "legacy-m8": "#277bb8",
        "connected-m16": "#248148",
    }
    summaries = {}
    curves = {}
    for arm in labels:
        directory = a.dst_run if arm == "dst-all" else a.run / arm
        s = read(directory / "summary.json")
        assert s["state"] == "completed" and s["completed_steps"] == 976
        rows = [
            json.loads(v) for v in (directory / "curves.jsonl").read_text().splitlines()
        ]
        summaries[arm] = s
        curves[arm] = rows
    keys = (
        "config",
        "data_manifest_sha256",
        "data_order_sha256",
        "seed",
        "target_steps",
        "schedule_steps",
        "effective_tokens_per_step",
        "completed_tokens",
    )
    for s in summaries.values():
        for key in keys:
            assert s[key] == summaries["dense"][key], key
    assert (
        summaries["dst-all"]["active_ffn_scalars"]
        == summaries["legacy-m8"]["active_ffn_scalars"]
        == summaries["connected-m16"]["active_ffn_scalars"]
    )
    assert (
        summaries["legacy-m8"]["initial_parameters_sha256"]
        == summaries["connected-m16"]["initial_parameters_sha256"]
    )
    qualification = read(a.run / "qualification/qualification.json")
    assert qualification["state"] == "passed"
    assert (
        len(
            {
                v["shared_ordinary_parameters_sha256"]
                for v in qualification["arms"].values()
            }
        )
        == 1
    )
    results = {}
    for arm, s in summaries.items():
        final = next(row["development"] for row in curves[arm] if row["step"] == 976)
        initial = next(row["development"] for row in curves[arm] if row["step"] == 0)
        results[arm] = {
            "initial_development": initial,
            "final_development": final,
            "median_complete_step_ms": statistics.median(s["timing_seconds"]) * 1000,
            "peak_allocated_MiB": s["peak_allocated_bytes"] / 2**20,
            "peak_reserved_MiB": s["peak_reserved_bytes"] / 2**20,
            "curve_sha256": sha(
                (a.dst_run if arm == "dst-all" else a.run / arm) / "curves.jsonl"
            ),
        }

    def curve_plot(detail):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
        for arm, label in labels.items():
            rows = curves[arm]
            train = [
                r for r in rows if "train_ce" in r and (not detail or r["step"] >= 200)
            ]
            axes[0].plot(
                [r["seen_tokens"] / 1e6 for r in train],
                [r["train_ce"] for r in train],
                color=colors[arm],
                alpha=0.25,
                lw=0.8,
            )
            averages = [
                statistics.mean([r["train_ce"] for r in train[max(0, j - 2) : j + 1]])
                for j in range(len(train))
            ]
            axes[0].plot(
                [r["seen_tokens"] / 1e6 for r in train],
                averages,
                color=colors[arm],
                label=label,
                lw=1.6,
            )
            dev = [
                r
                for r in rows
                if "development" in r and (not detail or r["step"] >= 200)
            ]
            for ax, key in zip(axes[1:], ("ce", "next_token_top1_accuracy")):
                ax.plot(
                    [r["seen_tokens"] / 1e6 for r in dev],
                    [r["development"][key] for r in dev],
                    "-o",
                    ms=3,
                    color=colors[arm],
                    label=label,
                )
        for ax, title in zip(
            axes, ("TRAIN CE (3 logged-point mean)", "DEV CE", "DEV next-token top1")
        ):
            ax.set_title(title)
            ax.set_xlabel("Training targets (million)")
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            "TinyStories | width512 / 4 blocks / all FFN maps | seed17 | matched raw CST initialization",
            fontsize=10,
        )
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(
                a.output / f"learning-{'detail' if detail else 'full'}.{ext}", dpi=180
            )
        plt.close(fig)

    curve_plot(False)
    curve_plot(True)
    diagnostic = read(a.run / "diagnostics/summary.json")
    assert diagnostic["state"] == "passed" and len(diagnostic["records"]) == 12
    fields = ("radius_fraction", "output_center", "input_center")
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.6), sharex=True)
    for col, field in enumerate(fields):
        for arm in ("legacy-m8", "connected-m16"):
            for site, style in (("1.up", "-"), ("1.down", "--")):
                rows = sorted(
                    (
                        r
                        for r in diagnostic["records"]
                        if r["arm"] == arm and r["site"] == site
                    ),
                    key=lambda r: r["step"],
                )
                v = [r["fields"][field][0] for r in rows]
                x = [r["step"] for r in rows]
                for ax, key in zip(
                    axes[:, col],
                    (
                        "linear_finite_cosine",
                        "relative_defect_to_prediction",
                        "finite_over_prediction_norm",
                    ),
                ):
                    yy = [r[key] for r in v]
                    ax.plot(
                        x,
                        [float("nan") if y is None else y for y in yy],
                        style + "o",
                        color=colors[arm],
                        ms=4,
                        label=labels[arm] + " " + site,
                    )
                    ax.set_xscale("log")
                    ax.grid(alpha=0.2)
        axes[0, col].set_title(field.replace("_", " "))
        axes[0, col].set_ylim(-1.05, 1.05)
        axes[0, col].axhline(1, color="black", alpha=0.2, lw=0.7)
        axes[1, col].set_yscale("log")
        axes[2, col].set_yscale("log")
        axes[2, col].axhline(1, color="black", alpha=0.2, lw=0.7)
        axes[2, col].set_xlabel("Actual optimizer update")
        axes[2, col].set_xticks([1, 256, 976], ["1", "256", "976"])
    for ax, title in zip(
        axes[:, 0],
        ("cosine(Jv, finite)", "||finite-Jv|| / ||Jv||", "||finite|| / ||Jv||"),
    ):
        ax.set_ylabel(title)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Observed task updates | raw-field isolation | actual FP32 endpoints | FP64 factors | scale=1",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"update-fidelity.{ext}", dpi=180)
    plt.close(fig)
    out = {
        "state": "verified",
        "protocol": receipt["protocol"]
        | {"source_files_count": len(receipt["protocol"]["source_files"])},
        "arms": results,
        "diagnostics": diagnostic,
        "qualification": qualification,
        "export_manifest_sha256": sha(a.run / "export-manifest.json"),
    }
    out["protocol"].pop("source_files")
    # Machine-specific evidence paths belong in raw receipts, not public notes.
    for key in ("remote_root", "data"):
        out["protocol"].pop(key, None)
    for r in out["diagnostics"]["records"]:
        r.pop("probe_path", None)
    (a.output / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
