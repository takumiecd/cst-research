"""Render a bounded mathematical diagnostic; no performance/learning claims."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    path = args.run / "summary.json"
    raw = json.loads(path.read_text())
    assert raw["state"] == "passed" and raw["records_count"] == 152
    assert sum(len(x["rows"]) for x in raw["records"]) == 1216
    assert raw["training_updates"] == 0
    records = [r for r in raw["records"] if r["dtype"] == "torch.float64"]
    profiles = list(dict.fromkeys(r["profile"] for r in records))

    def get(profile, case):
        return next(
            r for r in records if r["profile"] == profile and r["case"]["name"] == case
        )

    def at(record, h):
        return next(r for r in record["rows"] if r["step"] == h)

    selected = (
        "radius-0.5-expand",
        "radius-1.0-expand",
        "radius-2.0-expand",
        "center-2-r0.5001-b7.5",
        "center-2-r0.75-b0.0",
        "within-branch-field1",
    )
    gaussian = [r for r in records if r["modes"] is None]
    slopes = [
        math.log10(
            at(r, 1e-3)["absolute_taylor_error"] / at(r, 1e-4)["absolute_taylor_error"]
        )
        for r in gaussian
    ]
    result = {
        "state": "passed",
        "scope": raw["scope"],
        "consumer_commit": raw["consumer_commit"],
        "operator_commit": raw["operator_commit"],
        "script_sha256": raw["script_sha256"],
        "raw_summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runtime": {k: raw[k] for k in ("torch", "python", "device")},
        "records": 152,
        "finite_displacements": 1216,
        "training_updates": 0,
        "gaussian": {
            "fixtures": len(gaussian),
            "max_analytic_jvp_relative_error": max(
                v["analytic_jvp_relative_error"] for r in gaussian for v in r["rows"]
            ),
            "max_relative_taylor_error_h1e6": max(
                at(r, 1e-6)["relative_taylor_error"] for r in gaussian
            ),
            "min_prediction_finite_cosine_h1e6": min(
                at(r, 1e-6)["prediction_finite_cosine"] for r in gaussian
            ),
            "absolute_remainder_order_h1e3_to_h1e4": [min(slopes), max(slopes)],
        },
        "notes": [
            raw["sigma_radius_note"],
            raw["M_note"],
            "Tiny-step and near-zero-derivative relative errors are roundoff-sensitive; no speed or quality inference.",
        ],
        "selected_fp64_records": [
            get(profile, case) for profile in profiles for case in selected
        ],
    }
    docs = args.repository / "docs"
    assets = docs / "assets/kernel-taylor-continuity-20261109"
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    cases = ("radius-0.5-expand", "center-2-r0.5001-b7.5", "within-branch-field1")
    titles = (
        "Width crosses 0.5",
        "Center crosses 7.5 (width 0.5001)",
        "Width within one branch",
    )
    for ax, case, title in zip(axes, cases, titles):
        for profile in ("gaussian-fixed-cells", "chebyshev-M8"):
            rows = get(profile, case)["rows"]
            ax.loglog(
                [r["step"] for r in rows],
                [max(r["absolute_taylor_error"], 1e-18) for r in rows],
                "o-",
                label=profile,
            )
        ax.set_title(title)
        ax.set_xlabel("Displacement scale h")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Absolute Taylor remainder ||delta W - J delta theta||")
    axes[0].legend(fontsize=8)
    fig.suptitle("FP64, one Atom, no optimizer; Gaussian width/shape is not matched")
    fig.savefig(assets / "gaussian-versus-current.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    modes = [4, 8, 16, 32, 64]
    for width in (0.5, 1.0, 2.0):
        values = [
            at(get(f"chebyshev-M{m}", f"radius-{width}-expand"), 1e-12)[
                "absolute_taylor_error"
            ]
            for m in modes
        ]
        axes[0].semilogy(modes, values, "o-", label=f"Radius boundary {width}")
    values = [
        at(get(f"chebyshev-M{m}", "center-2-r0.5001-b7.5"), 1e-12)[
            "absolute_taylor_error"
        ]
        for m in modes
    ]
    axes[1].semilogy(modes, values, "o-", label="Center boundary 7.5")
    axes[1].axhline(
        1e-14, color="gray", linestyle="--", label="Indicative FP64 numerical floor"
    )
    for ax in axes:
        ax.set_xlabel("Chebyshev M (changes the density)")
        ax.set_xticks(modes)
        ax.set_ylabel("Absolute Taylor remainder at h=1e-12")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Higher M suppresses sampled center jumps; hierarchy-radius jumps remain"
    )
    fig.savefig(assets / "mode-count-boundaries.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for dtype in ("torch.float64", "torch.float32"):
        record = next(
            r
            for r in raw["records"]
            if r["profile"] == "gaussian-fixed-cells"
            and r["case"]["name"] == "radius-0.5-expand"
            and r["dtype"] == dtype
        )
        rows = [
            r
            for r in record["rows"]
            if not r["quantized_noop"] and r["relative_taylor_error"] is not None
        ]
        ax.loglog(
            [r["step"] for r in rows],
            [r["relative_taylor_error"] for r in rows],
            "o-",
            label=dtype,
        )
    ax.set_xlabel("Displacement scale h")
    ax.set_ylabel("Relative Taylor remainder")
    ax.set_title("Gaussian width crosses 0.5; encoded no-ops omitted")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(assets / "gaussian-precision.png", dpi=160)
    plt.close(fig)
    (docs / "kernel-taylor-continuity-20261109-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("selected_fp64_records", "notes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
