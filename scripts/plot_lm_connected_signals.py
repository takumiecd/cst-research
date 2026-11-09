"""Plot context-matched block RMS/cotangent ratios; no model updates."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    v = json.loads(a.summary.read_text())
    assert (
        v["state"] == "passed"
        and v["parameters_buffers_rng_unchanged"]
        and len(v["records"]) == 32
    )
    index = {(r["arm"], r["step"], r["train_offset"]): r for r in v["records"]}
    assert len(index) == 32
    arms = ("dst-all", "legacy-m8", "connected-m16")
    colors = ("#d78c20", "#277bb8", "#248148")
    labels = ("DST", "Legacy M8", "Connected M16")
    table = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), sharey="col")
    for row, step in enumerate((0, 976)):
        for arm, color, label in zip(arms, colors, labels):
            vals = {k: [] for k in ("output_rms", "output_gradient_rms", "product")}
            for block in range(4):
                samples = {k: [] for k in vals}
                for offset in (0, 4096, 8192, 12288):
                    left = index[(arm, step, offset)]["probe"]["layers"][
                        f"block{block}"
                    ]
                    right = index[("dense", step, offset)]["probe"]["layers"][
                        f"block{block}"
                    ]
                    x = left["output_rms"] / right["output_rms"]
                    g = left["output_gradient_rms"] / right["output_gradient_rms"]
                    for k, z in (
                        ("output_rms", x),
                        ("output_gradient_rms", g),
                        ("product", x * g),
                    ):
                        samples[k].append(z)
                result = {
                    "arm": arm,
                    "step": step,
                    "block": block,
                    **{
                        k: {
                            "mean": statistics.mean(z),
                            "min": min(z),
                            "max": max(z),
                            "context_values": z,
                        }
                        for k, z in samples.items()
                    },
                }
                table.append(result)
                for k, values in vals.items():
                    values.append(result[k])
            for col, k in enumerate(vals):
                z = vals[k]
                means = [r["mean"] for r in z]
                axes[row, col].errorbar(
                    range(4),
                    means,
                    yerr=[
                        [r["mean"] - r["min"] for r in z],
                        [r["max"] - r["mean"] for r in z],
                    ],
                    label=label,
                    color=color,
                    fmt="-o",
                    capsize=3,
                    ms=4,
                )
        for col, title in enumerate(
            (
                "Block output RMS / Dense",
                "Output cotangent RMS / Dense",
                "RMS(output) x RMS(cotangent), ratio",
            )
        ):
            axes[row, col].axhline(1, color="black", alpha=0.35, lw=0.8)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].set_xticks(range(4))
            axes[row, col].set_xlabel("Block")
            axes[row, col].grid(alpha=0.2)
        axes[row, 0].set_ylabel("Initial" if step == 0 else "After 976 updates")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Ratios within the same TRAIN context | 4 contexts x 128 tokens | bars: context range, not CI",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"block-signals.{ext}", dpi=180)
    plt.close(fig)
    output = {
        "state": "verified",
        "raw_summary_sha256": hashlib.sha256(a.summary.read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "mean of four within-context ratios, not pooled RMS; context range is not a confidence interval; output cotangent is not parameter gradient",
        "runtime": {k: v[k] for k in ("gpu", "torch", "cuda")},
        "parent_source_commits": v["parent_source_commits"],
        "records": table,
    }
    (a.output / "signals.json").write_text(json.dumps(output, indent=2) + "\n")
    for arm in arms:
        for step in (0, 976):
            rows = [r for r in table if r["arm"] == arm and r["step"] == step]
            print(
                arm,
                step,
                {
                    k: [round(r[k]["mean"], 3) for r in rows]
                    for k in ("output_rms", "output_gradient_rms", "product")
                },
            )


if __name__ == "__main__":
    main()
