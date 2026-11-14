"""Actual post-projection motion and analytic Adam estimates for the LR sweep."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ffn_lr_report_inputs import load_inputs


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--recovery-run", type=Path)
    args = p.parse_args()
    root = args.run.resolve()
    assert (
        json.loads((root / "retrieval-integrity.json").read_text())["state"] == "passed"
    )
    all_rows, directories = load_inputs(root, args.recovery_run)
    rows = [r for r in all_rows if r["kind"] == "cst"]
    out = root / "figures"
    out.mkdir(exist_ok=True)
    tau_comparison = any(r.get("tau_comparison") for r in rows)
    start_comparison = any(r.get("fraction_start_comparison") for r in rows)
    key = lambda r: (
        r["tau"]
        if tau_comparison
        else r["fraction"]
        if start_comparison
        else r["fraction_lr_multiplier"]
    )
    keyed_rows = {key(r): r for r in rows}
    assert len(keyed_rows) == len(rows)
    labels = {
        k: f"tau={r['tau']:g}"
        if tau_comparison
        else f"f0={r['fraction']:g}, LR32x"
        if start_comparison
        else f"{k:g}x"
        for k, r in keyed_rows.items()
    }
    histories = {
        key(r): [
            json.loads(x)
            for x in (directories[r["label"]] / "fraction-updates.jsonl")
            .read_text()
            .splitlines()
        ]
        for r in rows
    }
    rates = sorted(histories)
    colors = {
        rate: color
        for rate, color in zip(rates, ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"])
    }
    assert len(rates) <= 4

    def aggregate(entry):
        layers = list(entry["layers"].values())
        result = {}
        for key in [
            "last_fraction_delta",
            "net_fraction_delta",
            "fraction_path_length",
            "net_radius_delta",
            "radius_fraction_component",
            "radius_amplitude_component",
            "analytic_adam_fraction_delta",
        ]:
            if layers[0][key] is None:
                result[key] = None
                continue
            result[key] = {
                "mean_signed": float(np.mean([l[key]["mean_signed"] for l in layers])),
                "mean_absolute": float(
                    np.mean([l[key]["mean_absolute"] for l in layers])
                ),
                "rms": float(np.sqrt(np.mean([l[key]["rms"] ** 2 for l in layers]))),
            }
        for key in [
            "fraction_increased",
            "fraction_decreased",
            "fraction_unchanged",
            "at_zero",
            "at_one",
            "effective_fraction_lr",
        ]:
            result[key] = float(np.mean([l[key] for l in layers]))
        result["net_to_path_ratio"] = (
            result["net_fraction_delta"]["mean_absolute"]
            / result["fraction_path_length"]["mean_absolute"]
            if entry["step"]
            else 0.0
        )
        result["outward_momentum_at_boundary"] = float(
            np.mean([l.get("outward_momentum_at_boundary", 0.0) for l in layers])
        )
        return result

    agg = {rate: [aggregate(e) for e in entries] for rate, entries in histories.items()}

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    specs = [
        ("Net fraction signed mean", lambda a: a["net_fraction_delta"]["mean_signed"]),
        (
            "Net fraction mean absolute",
            lambda a: a["net_fraction_delta"]["mean_absolute"],
        ),
        (
            "Cumulative absolute path, every update",
            lambda a: a["fraction_path_length"]["mean_absolute"],
        ),
        ("Net L1 / cumulative path L1", lambda a: a["net_to_path_ratio"]),
        (
            "Last actual fraction update, mean absolute",
            lambda a: a["last_fraction_delta"]["mean_absolute"],
        ),
        ("Fraction at either endpoint", lambda a: a["at_zero"] + a["at_one"]),
    ]
    for ax, (title, getter) in zip(axes.flat, specs):
        for rate, entries in histories.items():
            ax.plot(
                [e["step"] for e in entries],
                [getter(a) for a in agg[rate]],
                label=labels[rate],
                color=colors[rate],
            )
        ax.set_title(title)
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(
        "Zero-start tau comparison, common CST LR32x: per-Atom fraction motion, seed17"
        if tau_comparison
        else "Initial fraction0 versus.5, common CST LR32x: per-Atom fraction motion, seed17"
        if start_comparison
        else "All CST fields LR: same initial fraction .5, measured per-Atom motion, seed17"
        if "cst_lr_multiplier" in rows[0]
        else "Same initial fraction .5: measured per-Atom motion, seed17"
    )
    save(fig, "fraction-movement")
    fig, axes = plt.subplots(
        len(rates),
        2,
        figsize=(12, 3.4 * len(rates)),
        squeeze=False,
        layout="constrained",
    )
    for i, rate in enumerate(sorted(histories)):
        xs = [e["step"] for e in histories[rate]]
        for j, metric in enumerate(["mean_signed", "mean_absolute"]):
            for key, label, color in [
                ("net_radius_delta", "Total", "#222222"),
                ("radius_fraction_component", "Fraction component", "#d95f02"),
                ("radius_amplitude_component", "Amplitude-bound component", "#1f77b4"),
            ]:
                axes[i, j].plot(
                    xs, [a[key][metric] for a in agg[rate]], label=label, color=color
                )
            axes[i, j].set_title(f"{labels[rate]}: radius delta {metric}")
            axes[i, j].set_xlabel("Updates")
            axes[i, j].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Exact radius decomposition at current amplitude / initial fraction; absolute means do not add"
    )
    save(fig, "radius-update-components")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for rate, entries in histories.items():
        xs = [e["step"] for e in entries if e["step"]]
        a = [v for e, v in zip(entries, agg[rate]) if e["step"]]
        for ax, metric in zip(axes[:2], ["mean_absolute", "rms"]):
            ax.plot(
                xs,
                [v["last_fraction_delta"][metric] for v in a],
                color=colors[rate],
                label=f"{labels[rate]} actual",
            )
            ax.plot(
                xs,
                [v["analytic_adam_fraction_delta"][metric] for v in a],
                "--",
                color=colors[rate],
                label=f"{labels[rate]} Adam estimate",
            )
            ax.set_yscale("log")
            ax.set_title(f"Fraction step {metric}")
            ax.set_xlabel("Updates")
            ax.grid(alpha=0.2)
        axes[2].plot(
            xs,
            [v["outward_momentum_at_boundary"] for v in a],
            color=colors[rate],
            label=labels[rate],
        )
    axes[0].legend(fontsize=7)
    axes[2].set_title("Outward momentum at boundary (not exact clip rate)")
    axes[2].set_xlabel("Updates")
    axes[2].grid(alpha=0.2)
    fig.suptitle(
        "Actual post-projection updates versus FP64 analytic estimates from stored FP32 Adam moments"
    )
    save(fig, "adam-fraction-updates")
    records = []
    for rate, entries in histories.items():
        assert all(e["step"] == e["steps_observed"] for e in entries)
        records.append(
            {
                "fraction_lr_multiplier": keyed_rows[rate]["fraction_lr_multiplier"],
                "initial_fraction": keyed_rows[rate]["fraction"],
                "tau": keyed_rows[rate]["tau"],
                **(
                    {"cst_lr_multiplier": keyed_rows[rate]["cst_lr_multiplier"]}
                    if "cst_lr_multiplier" in rows[0]
                    else {}
                ),
                "frames": len(entries),
                "at256": agg[rate][
                    next(i for i, e in enumerate(entries) if e["step"] == 256)
                ],
                "final": agg[rate][-1],
            }
        )
    (out / "fraction-update-summary.json").write_text(
        json.dumps(
            {
                "arms": records,
                "scope": "Equal Atom counts per site. Means/RMS aggregate all Atoms; path statistics reduce every per-Atom update, full individual trajectories are not saved. Algebraic radius components do not identify causal training effects.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
