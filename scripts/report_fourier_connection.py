"""Render a proposed Fourier connection law and its bounded CPU diagnostics."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.run / "summary.json"
    raw = json.loads(raw_path.read_text())
    assert raw["state"] == "passed" and raw["records_count"] == 140
    assert sum(len(r["rows"]) for r in raw["records"]) == 1120
    assert len(raw["tail_screen"]) == 492 and len(raw["geometry"]) == 49
    result = {
        k: raw[k]
        for k in (
            "scope",
            "operator_commit",
            "consumer_commit",
            "script_sha256",
            "torch",
            "python",
            "device",
            "parameters",
            "training_updates",
        )
    }
    result.update(
        state="passed",
        records=140,
        finite_displacements=1120,
        raw_summary_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    )

    def get(variant, case, modes=8):
        return next(
            r
            for r in raw["records"]
            if r["modes"] == modes
            and r["variant"] == variant
            and r["case"]["name"] == case
        )

    cases = (
        "old-radius-1.0",
        "radius-start-T1-False",
        "radius-end-T1-False",
        "position-mid",
        "joint-transition",
    )
    result["selected_records"] = [
        r for r in raw["records"] if r["case"]["name"] in cases
    ]
    result["full_conformance"] = {}
    for modes in (8, 16, 32):
        rows = [
            v
            for r in raw["records"]
            if r["modes"] == modes and r["variant"] == "full"
            for v in r["rows"]
        ]
        selected = [v for v in rows if v["h"] == 1e-6]
        result["full_conformance"][str(modes)] = {
            "max_mass_error": max(v["mass_error"] for v in rows),
            "min_mass": min(v["minimum_mass"] for v in rows),
            "max_taylor_error_h1e6": max(v["absolute_taylor_error"] for v in selected),
            "max_error_per_step_h1e6": max(
                v["error_per_parameter_step"] for v in selected
            ),
        }
    result["tail_screen_maxima"] = [
        {
            "kappa": kappa,
            "modes": modes,
            "max_sampled_tail_mass": max(
                x["tail_mass"]
                for x in raw["tail_screen"]
                if x["kappa"] == kappa and x["modes"] == modes
            ),
        }
        for kappa in (1.05, 1.1, 1.25, 1.4)
        for modes in (8, 16, 32)
    ]
    result["geometry"] = {
        "samples": 49,
        "max_active_banks": max(x["active_scales"] for x in raw["geometry"]),
        "max_active_windows": max(x["active_windows"] for x in raw["geometry"]),
        "minimum_core_margin": min(
            b["core_margin"] for x in raw["geometry"] for b in x["branches"]
        ),
    }
    result["limits"] = [
        "New definition, not a checkpoint-compatible or adopted runtime replacement",
        "Ratio[.5,1] is relaxed for helper banks to(.36,.9] in the illustrative schedule",
        "Total tail samples are not certified peak-sidelobe or global tail bounds",
        "No public backend, GPU, model quality, initialization or complete-step performance qualification",
    ]
    docs = args.repository / "docs"
    assets = docs / "assets/fourier-scale-connection-20261109"
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, case, title in zip(
        axes,
        ("old-radius-1.0", "position-mid", "joint-transition"),
        (
            "Radius crosses old boundary1",
            "Position crosses7.5",
            "Joint coordinate/radius change",
        ),
    ):
        for name in ("old", "radius-only", "full", "omit-radius-weight-gradient"):
            rows = get(name, case)["rows"]
            ax.loglog(
                [x["h"] for x in rows],
                [max(x["absolute_taylor_error"], 1e-18) for x in rows],
                "o-",
                label=name,
            )
        ax.set_title(title)
        ax.set_xlabel("Displacement scale h")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Absolute Taylor remainder (one-axis masses)")
    axes[0].legend(fontsize=7)
    fig.suptitle("M8 FP64: connect both variables and differentiate connection weights")
    fig.savefig(assets / "connection-taylor.png", dpi=160)
    plt.close(fig)

    lower, upper, delta = (raw["parameters"][k] for k in ("lower", "upper", "delta"))
    radius = np.linspace(0.5, 1.2, 401)
    t = np.clip(np.log(radius / lower) / math.log(upper / lower), 0, 1)
    weight = t**3 * (10 - 15 * t + 6 * t**2)
    derivative = 30 * t**2 * (1 - t) ** 2 / (radius * math.log(upper / lower))
    center = np.linspace(7.0, 8.0, 401)
    mu_u = np.clip((center - (7.5 - delta)) / (2 * delta), 0, 1)
    mu_w = mu_u**3 * (10 - 15 * mu_u + 6 * mu_u**2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    axes[0].plot(radius, 1 - weight, label="T=1 bank")
    axes[0].plot(radius, weight, label="T=2 bank")
    axes[0].axvspan(lower, upper, alpha=0.1, color="gray")
    axes[0].set_xlabel("Physical radius R")
    axes[0].set_ylabel("Bank weight")
    axes[0].legend()
    axes[1].plot(radius, derivative)
    axes[1].set_xlabel("Physical radius R")
    axes[1].set_ylabel("d(weight)/dR; included in backward")
    axes[2].plot(center, 1 - mu_w, label="Left fixed window")
    axes[2].plot(center, mu_w, label="Right fixed window")
    axes[2].set_xlabel("Center mu, T=1")
    axes[2].set_ylabel("Window weight")
    axes[2].legend(fontsize=8)
    fig.suptitle("Illustrative connection parameters, not adopted defaults")
    fig.savefig(assets / "connection-weights.png", dpi=160)
    plt.close(fig)

    kappas = (1.05, 1.1, 1.25, 1.4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for modes in (8, 16, 32):
        values = [
            next(
                x["max_sampled_tail_mass"]
                for x in result["tail_screen_maxima"]
                if x["kappa"] == k and x["modes"] == modes
            )
            for k in kappas
        ]
        axes[0].semilogy(kappas, values, "o-", label=f"M{modes}")
    axes[0].set_xlabel("Radius-band ratio kappa=b/a")
    axes[0].set_ylabel("Maximum sampled outside-core mass")
    axes[0].legend()
    axes[1].plot(kappas, [1.875 / math.log(k) for k in kappas], "o-")
    axes[1].set_xlabel("Radius-band ratio kappa=b/a")
    axes[1].set_ylabel("Bound on R * |d(weight)/dR|")
    fig.suptitle(
        "Narrower bands reduce helper leakage but increase connection steepness"
    )
    fig.savefig(assets / "connection-tail-tradeoff.png", dpi=160)
    plt.close(fig)
    (docs / "fourier-scale-connection-20261109-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("selected_records", "limits")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
