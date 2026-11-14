"""Render compact boundary diagnostics without inferring task quality or speed."""

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
    assert raw["state"] == "passed" and raw["records_count"] == 240
    assert sum(len(r["rows"]) for r in raw["records"]) == 1920
    records = [r for r in raw["records"] if r["dtype"] == "torch.float64"]

    def get(profile, case):
        return next(
            r for r in records if r["profile"] == profile and r["case"]["name"] == case
        )

    def at(record, h):
        return next(r for r in record["rows"] if r["step"] == h)

    selected = (
        "hierarchy-radius-2.0-expand",
        "support-radius-1-False",
        "half-center-7.5-False",
        "empty-birth-radius-0.5-False",
        "empty-interior-field-0",
        "floor-center-False",
        "floor-radius-False",
    )
    profiles = list(dict.fromkeys(r["profile"] for r in records))
    result = {
        k: raw[k]
        for k in (
            "scope",
            "operator_commit",
            "consumer_commit",
            "script_sha256",
            "gaussian_helper_sha256",
            "torch",
            "python",
            "device",
            "notes",
        )
    }
    result.update(
        state="passed",
        records=240,
        finite_displacements=1920,
        raw_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        training_updates=0,
    )
    result["selected_fp64_records"] = [get(p, c) for p in profiles for c in selected]
    result["support_entry_remainder_orders"] = {
        profile: math.log10(
            at(get(profile, "support-radius-1-False"), 1e-3)["absolute_taylor_error"]
            / at(get(profile, "support-radius-1-False"), 1e-4)["absolute_taylor_error"]
        )
        for profile in ("biweight-point", "triweight-point", "wendland-c2-point")
    }
    result["l2_floor_center_remainder_order"] = math.log10(
        at(get("triweight-l2", "floor-center-False"), 1e-6)["absolute_taylor_error"]
        / at(get("triweight-l2", "floor-center-False"), 1e-8)["absolute_taylor_error"]
    )
    result["l2_floor_center_remainder_order"] /= 2  # h changes by100, not10.
    docs = args.repository / "docs"
    assets = docs / "assets/compact-boundary-20261109"
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    configurations = [
        (
            "hierarchy-radius-2.0-expand",
            "Radius crosses 2",
            (
                "triweight-point",
                "triweight-l2",
                "triweight-cell-control",
                "gaussian-cell-control",
            ),
        ),
        (
            "support-radius-1-False",
            "A point enters compact support",
            ("biweight-point", "triweight-point", "wendland-c2-point"),
        ),
        (
            "floor-center-False",
            "Center crosses L2 norm floor",
            ("triweight-point", "triweight-l2", "triweight-cell-control"),
        ),
    ]
    for ax, (case, title, names) in zip(axes, configurations):
        for name in names:
            rows = get(name, case)["rows"]
            ax.loglog(
                [r["step"] for r in rows],
                [max(r["absolute_taylor_error"], 1e-50) for r in rows],
                "o-",
                label=name,
            )
        ax.set_title(title)
        ax.set_xlabel("Displacement scale h")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Absolute Taylor remainder (one axis factor)")
    fig.suptitle(
        "FP64 compact profiles: smooth support edges; a separate L2-floor kink"
    )
    fig.savefig(assets / "compact-taylor-boundaries.png", dpi=160)
    plt.close(fig)
    (docs / "compact-boundary-20261109-results.json").write_text(
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
