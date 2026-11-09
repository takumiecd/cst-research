"""Render only hash-verified, audited site-count pairs; partial plots are explicit."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    root = a.run.resolve()
    state = read(root / "backup-state.json")
    assert state["snapshots"]
    if not a.allow_partial:
        assert state["state"] == "completed"
    last = state["snapshots"][-1]
    archive = root / last["name"]
    assert sha(archive) == last["sha256"]
    data = root / last["retrieved"]
    manifest = read(data / "manifest.json")
    assert all(sha(data / n) == h for n, h in manifest["files"].items())
    receipt = read(data / "receipt.json")
    rows = receipt["results"]
    summary = {v["arm"]: read(data / v["arm"] / "summary.json") for v in rows}
    if not a.allow_partial:
        assert (
            receipt["state"] == "passed"
            and receipt["source_unchanged"]
            and len(rows) == 11
        )
    for v in rows:
        audit = read(data / ("audit-" + v["arm"]) / "audit.json")
        assert audit["state"] == "passed" and len(audit["models"]) == 3
    assert len({v["data_order_sha256"] for v in rows}) == 1
    assert "dense" in summary
    keys = ("up", "down", "block", "two_blocks", "all")
    complete = [k for k in keys if all(f"{t}-{k}" in summary for t in ("cst", "dst"))]
    if not complete:
        raise SystemExit("No audited CST/DST pair is complete yet")
    arms = ["dense", *[f"{t}-{k}" for k in complete for t in ("cst", "dst")]]
    for k in complete:
        assert (
            summary["cst-" + k]["active_ffn_scalars"]
            == summary["dst-" + k]["active_ffn_scalars"]
        )
    curves = {arm: lines(data / arm / "curves.jsonl") for arm in arms}
    probes = {arm: lines(data / arm / "mechanisms.jsonl") for arm in arms}
    out = root / "figures"
    out.mkdir(exist_ok=True)
    names = {
        "up": "1 map: block1 up",
        "down": "1 map: block1 down",
        "block": "2 maps: block1 up+down",
        "two_blocks": "4 maps: blocks1+2",
        "all": "8 maps: all4 blocks",
    }
    colors = {"dense": "#222222", "cst": "#1f77b4", "dst": "#2ca02c"}
    qualifier = (
        "PROVISIONAL completed pairs only"
        if len(rows) < 11
        else "Complete fixed pilot, one DEV seed"
    )

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    for metric, label in [
        ("ce", "Development cross entropy"),
        ("next_token_top1_accuracy", "Next-token top1 accuracy (%)"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, key in zip(axes.flat, keys):
            ax.set_title(names[key])
            ax.grid(alpha=0.2)
            ax.set_xlabel("Training tokens (millions)")
            ax.set_ylabel(label)
            if key not in complete:
                ax.text(
                    0.5,
                    0.5,
                    "Pair not yet complete",
                    ha="center",
                    transform=ax.transAxes,
                )
                continue
            for kind, arm in [
                ("dense", "dense"),
                ("cst", "cst-" + key),
                ("dst", "dst-" + key),
            ]:
                pts = [v for v in curves[arm] if "development" in v]
                ax.plot(
                    [v["seen_tokens"] / 1e6 for v in pts],
                    [
                        v["development"][metric] * (100 if metric != "ce" else 1)
                        for v in pts
                    ],
                    color=colors[kind],
                    label=kind.upper() if kind != "dense" else "Dense",
                )
            ax.legend(fontsize=8)
        axes.flat[-1].axis("off")
        fig.suptitle(f"{qualifier}; same4-block LM, selected maps only")
        save(fig, "site-count-" + metric)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, key in zip(axes.flat, keys):
        ax.set_title(names[key])
        ax.grid(alpha=0.2)
        ax.set_xlabel("Training tokens (millions)")
        ax.set_ylabel("Train CE, rolling mean of5 logged records")
        if key not in complete:
            ax.text(
                0.5, 0.5, "Pair not yet complete", ha="center", transform=ax.transAxes
            )
            continue
        for kind, arm in [
            ("dense", "dense"),
            ("cst", "cst-" + key),
            ("dst", "dst-" + key),
        ]:
            pts = [v for v in curves[arm] if "train_ce" in v]
            ys = np.array([v["train_ce"] for v in pts])
            smooth = np.convolve(ys, np.ones(5) / 5, mode="valid")
            ax.plot(
                [v["seen_tokens"] / 1e6 for v in pts[4:]],
                smooth,
                color=colors[kind],
                label=kind.upper() if kind != "dense" else "Dense",
            )
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle(qualifier + "; raw train records remain in source curves")
    save(fig, "site-count-train-ce")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    x = np.arange(len(complete))
    dense = summary["dense"]["latest_development"]["ce"]
    for kind, offset in [("cst", -0.12), ("dst", 0.12)]:
        ax.bar(
            x + offset,
            [
                summary[kind + "-" + k]["latest_development"]["ce"] - dense
                for k in complete
            ],
            width=0.24,
            color=colors[kind],
            label=kind.upper(),
        )
    ax.set_xticks(x, [names[k] for k in complete], rotation=15, ha="right")
    ax.axhline(0, color="#222222", lw=1)
    ax.set_ylabel("Final DEV CE minus fresh Dense")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle(qualifier + "; compression budget changes with count")
    save(fig, "site-count-final-gaps")
    sites = list(probes["dense"][-1]["probe"]["layers"])
    fig, axes = plt.subplots(
        2, 1, figsize=(14, max(7, len(arms) * 0.7)), layout="constrained"
    )
    for ax, metric, title in [
        (axes[0], "output_rms", "Fixed TRAIN output RMS / Dense"),
        (axes[1], "output_gradient_rms", "Fixed TRAIN output-cotangent RMS / Dense"),
    ]:
        base = probes["dense"][-1]["probe"]["layers"]
        values = np.array(
            [
                [
                    probes[arm][-1]["probe"]["layers"][s][metric]
                    / max(base[s][metric], 1e-30)
                    for s in sites
                ]
                for arm in arms
            ]
        )
        image = ax.imshow(
            np.log2(np.maximum(values, 1e-30)),
            vmin=-3,
            vmax=3,
            cmap="coolwarm",
            aspect="auto",
        )
        ax.set_yticks(range(len(arms)), arms)
        ax.set_xticks(range(len(sites)), sites, rotation=45, ha="right")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="log2 ratio, color clipped at +/-3")
    fig.suptitle(
        qualifier + "; final step4882, descriptive gradients are not causal proof"
    )
    save(fig, "site-count-layer-signals")
    action_sites = list(probes["dense"][-1]["actual_update_actions"])

    def action_average(arm, site):
        return np.mean(
            [
                v["actual_update_actions"][site]["relative_action_delta"]
                for v in probes[arm][-5:]
            ]
        )

    values = np.array(
        [
            [
                action_average(arm, s) / max(action_average("dense", s), 1e-30)
                for s in action_sites
            ]
            for arm in arms
        ]
    )
    fig, ax = plt.subplots(figsize=(13, max(4, len(arms) * 0.45)), layout="constrained")
    image = ax.imshow(
        np.log2(np.maximum(values, 1e-30)),
        vmin=-4,
        vmax=4,
        cmap="coolwarm",
        aspect="auto",
    )
    ax.set_yticks(range(len(arms)), arms)
    ax.set_xticks(range(len(action_sites)), action_sites, rotation=45, ha="right")
    fig.colorbar(image, ax=ax, label="log2 sampled relative action-update / Dense")
    fig.suptitle(qualifier + "; last5 observed updates, same7 Gaussian inputs/map")
    save(fig, "site-count-realized-updates")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for key in complete:
        values = probes["cst-" + key]
        for ax, metric in zip(axes, ("radius", "radius_fraction")):
            ax.plot(
                [v["seen_tokens"] / 1e6 for v in values],
                [
                    np.mean(
                        [site["all"][metric]["mean"] for site in v["geometry"].values()]
                    )
                    for v in values
                ],
                label=names[key],
            )
            ax.set_xlabel("Training tokens (millions)")
            ax.set_ylabel("Selected-map mean " + metric)
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(qualifier + "; current-amplitude geometry, CST only")
    save(fig, "site-count-geometry")
    records = []
    for arm in arms:
        s = summary[arm]
        records.append(
            {
                "arm": arm,
                "sites": s["sites"],
                "active_ffn_scalars": s["active_ffn_scalars"],
                "final_development": s["latest_development"],
                "median_uninstrumented_step_ms": float(
                    np.median(s["timing_samples_seconds"]) * 1000
                ),
                "total_wall_seconds": s["total_elapsed_seconds"],
                "peak_allocated_mib": s["peak_allocated_bytes"] / 2**20,
                "peak_reserved_mib": s["peak_reserved_bytes"] / 2**20,
            }
        )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "state": "provisional" if len(rows) < 11 else "complete_verified_pilot",
                "snapshot": last,
                "runs": records,
                "limitations": [
                    "One seed,5M prefix, no new tuning or confirmation",
                    "Number/placement and total compression budget vary; matched CST/DST at each set",
                    "Raw cotangents/function actions guide hypotheses, not causal identification",
                    "Fixed arm order; excluded instrumented steps, historical timing not paired",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
