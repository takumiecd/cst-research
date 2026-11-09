"""Render analysis-only evidence; inputs are verified compact result directories."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--extended-job", type=Path, required=True)
    parser.add_argument("--field-job", type=Path, required=True)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    data = {}
    evidence = []
    for name, root in (("extended", args.extended_job), ("fields", args.field_job)):
        backup = read(root / "backup-state.json")
        assert backup["state"] == "completed"
        final = backup["snapshots"][-1]
        archive = root / final["name"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == final["sha256"]
        compact = root / final["retrieved"]
        manifest = read(compact / "manifest.json")
        for file, expected in manifest["files"].items():
            assert hashlib.sha256((compact / file).read_bytes()).hexdigest() == expected
        receipt = read(compact / "receipt.json")
        data[name] = read(compact / "results/summary.json")
        assert data[name]["state"] == receipt["state"] == "passed"
        assert data[name]["training_updates"] == 0
        evidence.append(
            {
                "job": root.name,
                "source_commits": receipt["protocol"]["source_commits"],
                "runtime_import_consumer_commit": receipt["protocol"][
                    "runtime_import_consumer_commit"
                ],
                "source_archive_sha256": receipt["protocol"]["source_archive_sha256"],
                "compact_sha256": final["sha256"],
                "files_verified": final["files_verified"],
                "source_files_verified": final["source_files_verified"],
                "records": len(data[name]["records"]),
                "training_updates": 0,
            }
        )
    data["witness"] = read(args.witness)
    data["witness_sha256"] = hashlib.sha256(args.witness.read_bytes()).hexdigest()
    data["witness_script_sha256"] = hashlib.sha256(
        Path(__file__).with_name("inspect_radius_routing.py").read_bytes()
    ).hexdigest()
    data["evidence"] = evidence
    docs = args.repository / "docs"
    assets = docs / "assets/lm-tangent-history-20261109"
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    primary = [x for x in data["extended"]["records"] if x["batch_offset_steps"] == 0]
    labels = [x["arm"] + " " + x["site"] for x in primary]
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    pos = np.arange(len(primary))
    for offset, label, values in [
        (-0.25, "Actual finite update", [x["actual_finite_cosine"] for x in primary]),
        (
            0,
            "Tangent of observed displacement",
            [x["actual_linearization"]["tangent_cosine"] for x in primary],
        ),
        (
            0.25,
            "Current gradient / historical denominator",
            [x["directions"]["current_g_historical_v"]["cosine"] for x in primary],
        ),
    ]:
        ax.bar(pos + offset, values, width=0.25, label=label)
    ax.set_xticks(pos, labels, rotation=20, ha="right")
    ax.set_ylabel("Cosine with task descent cotangent")
    ax.set_title(
        "One unchanged checkpoint; next TRAIN batch; no optimizer intervention"
    )
    ax.legend(fontsize=8)
    fig.savefig(assets / "history-directions.png", dpi=160)
    plt.close(fig)

    cst = [x for x in primary if x["arm"].startswith("cst")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    for ax, record in zip(axes, cst):
        rows = record["finite_response"]
        ax.loglog(
            [x["epsilon"] for x in rows],
            [x["finite_norm"] for x in rows],
            "o-",
            label="Finite change",
        )
        ax.loglog(
            [x["epsilon"] for x in rows],
            [x["tangent_norm"] for x in rows],
            "o--",
            label="Derivative prediction",
        )
        ax.set_title(record["arm"] + " " + record["site"])
        ax.set_xlabel("Fraction of observed displacement")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Output change Frobenius norm")
    axes[0].legend(fontsize=8)
    fig.savefig(assets / "routing-finite-response.png", dpi=160)
    plt.close(fig)

    field_names = ["amplitude", "radius_fraction", "output_center", "input_center"]
    colors = ["C0", "C1"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained", sharey=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), layout="constrained", sharey=True)
    for ax, ax2, reference in zip(axes, axes2, cst):
        rows = [
            x
            for x in data["fields"]["records"]
            if (x["arm"], x["site"]) == (reference["arm"], reference["site"])
        ]
        assert len(rows) == 3
        for j, name in enumerate(("tangent_cosine", "finite_cosine")):
            matrix = np.array(
                [
                    [
                        x["field_attribution"]["fields"][field][name]
                        for field in field_names
                    ]
                    for x in rows
                ]
            )
            positions = np.arange(4) + (j - 0.5) * 0.3
            ax.bar(
                positions,
                matrix.mean(0),
                width=0.3,
                label=name.replace("_", " "),
                color=colors[j],
                alpha=0.7,
            )
            for vector in matrix:
                ax.scatter(positions, vector, color=colors[j], s=13)
        for j, key in enumerate(("atom_linear_scores", "atom_finite_scores")):
            matrix = np.array(
                [
                    [
                        x["field_attribution"]["fields"][field][key][
                            "negative_absolute_score_share"
                        ]
                        for field in field_names
                    ]
                    for x in rows
                ]
            )
            positions = np.arange(4) + (j - 0.5) * 0.3
            ax2.bar(
                positions,
                100 * matrix.mean(0),
                width=0.3,
                label=key.replace("atom_", "").replace("_scores", ""),
                color=colors[j],
                alpha=0.7,
            )
            for vector in matrix:
                ax2.scatter(positions, 100 * vector, color=colors[j], s=13)
        for plot in (ax, ax2):
            plot.set_xticks(
                np.arange(4),
                ["amp*", "fraction", "out center", "in center"],
                rotation=25,
            )
            plot.set_title(reference["arm"] + " " + reference["site"])
            plot.axhline(0, color="gray", linewidth=0.5)
    axes[0].set_ylabel("Task descent cosine (dots: three TRAIN batches)")
    axes[0].legend(fontsize=8)
    axes2[0].set_ylabel("Adverse Atom share of absolute task score (%)")
    axes2[0].legend(fontsize=8)
    fig.suptitle(
        "Observed field displacements; bias fixed. *Amplitude also changes radius"
    )
    fig2.suptitle("Additive Atom task inner products, not per-Atom loss changes")
    fig.savefig(assets / "field-directions.png", dpi=160)
    fig2.savefig(assets / "atom-task-scores.png", dpi=160)
    plt.close(fig)
    plt.close(fig2)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for result in data["witness"]["results"]:
        rows = [r for r in result["rows"] if r["epsilon"] > 0]
        ax.loglog(
            [r["epsilon"] for r in rows],
            [max(r["weight_change_frobenius"], 1e-16) for r in rows],
            "o-",
            label=result["dtype"] + " actual",
        )
    rows = data["witness"]["results"][0]["rows"][1:]
    ax.loglog(
        [r["epsilon"] for r in rows],
        [r["predicted_change_frobenius"] for r in rows],
        "--",
        label="FP64 derivative prediction",
    )
    ax.set_xlabel("Fraction displacement from zero")
    ax.set_ylabel("Weight change Frobenius norm (zeros floored at 1e-16)")
    ax.set_title("One Atom, no optimizer: discontinuity at radius 0.5")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(assets / "single-atom-witness.png", dpi=160)
    plt.close(fig)
    (docs / "lm-tangent-history-20261109-results.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
