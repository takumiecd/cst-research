"""Plot verified placement curves, signal ratios and same-token similarities."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRIMARY = (
    "pair01",
    "pair02",
    "pair03",
    "pair12",
    "pair13",
    "pair23",
    "up-even",
    "down-even",
)
COLORS = {"dense": "#333333", "dst-pair12": "#9a7919", "all-cst": "#b12337"}
COLORS.update(
    zip(
        PRIMARY,
        (
            "#277bb8",
            "#8054a0",
            "#ed9224",
            "#248148",
            "#37a6a6",
            "#d471a5",
            "#8a6c4f",
            "#727f9e",
        ),
    )
)


def read(p):
    return json.loads(p.read_text())


def save(fig, root, name):
    fig.savefig(root / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(root / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def heat(
    fig, axis, values, arms, sites, title, *, limit=None, cmap="coolwarm", log=False
):
    values = np.array(values, dtype=float)
    shown = np.log2(values) if log else values
    kwargs = {} if limit is None else {"vmin": limit[0], "vmax": limit[1]}
    im = axis.imshow(shown, aspect="auto", cmap=cmap, **kwargs)
    axis.set_xticks(range(len(sites)), sites, rotation=45, ha="right")
    axis.set_yticks(range(len(arms)), arms)
    axis.set_title(title)
    for i in range(len(arms)):
        for j in range(len(sites)):
            red, green, blue, _ = im.cmap(im.norm(shown[i, j]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if luminance > 0.55 else "white",
            )
    fig.colorbar(im, ax=axis, shrink=0.8, label="log2 ratio" if log else None)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dst-run", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--atom-state", type=Path, required=True)
    p.add_argument("--ablation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.dst_run.resolve() != (a.run / "dst-pair12").resolve():
        raise ValueError(
            "this fixed study requires its freshly trained, verified colocated DST-pair12 control"
        )
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = read(a.run / "manifest.json")
    assert all(
        hashlib.sha256((a.run / n).read_bytes()).hexdigest() == h
        for n, h in manifest["files"].items()
    )
    receipt = read(a.run / "receipt.json")
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    audit = read(a.run / "audit/audit.json")
    assert audit["state"] == "passed"
    analysis = read(a.analysis)
    atoms = read(a.atom_state)
    ablation = read(a.ablation)
    assert (
        ablation["state"] == "passed"
        and ablation["source_commits"] == analysis["source_commits"]
    )
    assert (
        atoms["state"] == "passed"
        and atoms["source_commits"] == analysis["source_commits"]
    )
    assert (
        analysis["state"] == "passed"
        and analysis["source_commits"] == receipt["protocol"]["source_commits"]
    )
    arms = ["dense", "dst-pair12", *PRIMARY, "all-cst"]
    summaries = {arm: read(a.run / arm / "summary.json") for arm in arms}
    keys = (
        "config",
        "data_manifest_sha256",
        "data_order_sha256",
        "steps",
        "schedule_steps",
        "seed",
        "effective_tokens_per_step",
        "completed_tokens",
    )
    for s in summaries.values():
        assert s["state"] == "completed"
        assert all(s[k] == summaries["dense"][k] for k in keys)
    assert (
        len({summaries[arm]["active_ffn_scalars"] for arm in (*PRIMARY, "dst-pair12")})
        == 1
    )
    curves = {
        arm: [
            json.loads(line)
            for line in (a.run / arm / "curves.jsonl").read_text().splitlines()
        ]
        for arm in arms
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dense_dev = {
        r["step"]: r["development"]["ce"] for r in curves["dense"] if "development" in r
    }
    for arm in arms:
        tr = [r for r in curves[arm] if "train_ce" in r]
        de = [r for r in curves[arm] if "development" in r]
        x = np.array([r["seen_tokens"] for r in tr]) / 1e6
        yy = np.array([r["train_ce"] for r in tr])
        # Trailing five logged observations (~100 updates); raw rows remain available.
        ys = [np.mean(yy[max(0, i - 4) : i + 1]) for i in range(len(yy))]
        style = {
            "color": COLORS[arm],
            "label": arm,
            "lw": 2 if arm in ("dense", "all-cst", "pair12") else 1.2,
            "ls": "--" if arm == "all-cst" else "-",
        }
        axes[0].plot(x, ys, **style)
        axes[1].plot(
            [r["seen_tokens"] / 1e6 for r in de],
            [r["development"]["ce"] for r in de],
            **style,
        )
        axes[2].plot(
            [r["seen_tokens"] / 1e6 for r in de],
            [r["development"]["ce"] - dense_dev[r["step"]] for r in de],
            **style,
        )
    axes[0].set(title="TRAIN CE (trailing 5 logs)")
    axes[1].set(title="DEV CE, same8192 targets")
    axes[2].set(title="DEV CE minus Dense")
    axes[2].axhline(0, color="gray", lw=0.7)
    for ax in axes:
        ax.set_xlabel("Training targets (millions)")
        ax.grid(alpha=0.2)
    axes[2].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Connected M16: eight matched CST4 placements + three controls; seed17"
    )
    for extension in ("png", "svg"):
        fig.savefig(
            a.output / f"learning-full.{extension}", dpi=160, bbox_inches="tight"
        )
    axes[0].set_ylim(2.35, 4.5)
    axes[1].set_ylim(2.45, 4.5)
    save(fig, a.output, "learning")
    # Four context means for own-state signal metrics; these are not confidence intervals.
    final = receipt["protocol"]["steps"]
    signals = {
        arm: read(a.run / arm / f"signals-{final}.json")["records"] for arm in arms
    }
    sites = [f"{i}.{d}" for i in range(4) for d in ("up", "down")]

    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    layout = [
        [
            (2 if arm == "dst-pair12" else 1) if site in summaries[arm]["sites"] else 0
            for site in sites
        ]
        for arm in arms
    ]
    axes[0].imshow(
        layout,
        cmap=ListedColormap(["#e2e6ec", "#3678a0", "#ae8c36"]),
        vmin=0,
        vmax=2,
        aspect="auto",
    )
    axes[0].set_xticks(range(8), sites, rotation=45, ha="right")
    axes[0].set_yticks(range(len(arms)), arms)
    for i, row in enumerate(layout):
        for j, value in enumerate(row):
            axes[0].text(
                j,
                i,
                ("D", "C", "S")[value],
                ha="center",
                va="center",
                color="black" if value == 0 else "white",
            )
    base_ce = analysis["endpoints"]["dense"]["development"]["ce"]
    gaps = [analysis["endpoints"][arm]["development"]["ce"] - base_ce for arm in arms]
    axes[1].barh(range(len(arms)), gaps, color=[COLORS[arm] for arm in arms])
    axes[1].set_yticks(range(len(arms)), arms)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].set_xlim(min(-0.005, min(gaps) - 0.01), max(gaps) + 0.03)
    for i, arm in enumerate(arms):
        ce = analysis["endpoints"][arm]["development"]["ce"]
        axes[1].text(
            max(0, gaps[i]) + 0.001, i, f"CE {ce:.4f}", va="center", fontsize=8
        )
    axes[0].set_title("FFN layout: D=Dense, C=CST, S=DST")
    axes[1].set_title("Final DEV CE minus Dense (lower is better)")
    axes[1].set_xlabel("CE gap at 4,999,168 training targets")
    fig.suptitle(
        "Eight CST4 placements match up2/down2 and active scalar count; controls differ"
    )
    fig.tight_layout()
    save(fig, a.output, "placement")

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for ax, key, title in zip(
        axes.flat,
        ("output_rms", "input_gradient_rms", "output_gradient_rms", "entropy_rank"),
        (
            "Output RMS / Dense",
            "Input cotangent RMS / Dense",
            "Output cotangent RMS / Dense",
            "Centered output entropy rank / Dense",
        ),
    ):
        values = []
        for arm in arms:
            row = []
            for site in sites:
                if key == "entropy_rank":
                    ratios = [
                        r["layers"][site]["output_features"][key]
                        / d["layers"][site]["output_features"][key]
                        for r, d in zip(signals[arm], signals["dense"])
                    ]
                else:
                    ratios = [
                        r["layers"][site][key] / d["layers"][site][key]
                        for r, d in zip(signals[arm], signals["dense"])
                    ]
                row.append(statistics.mean(ratios))
            values.append(row)
        heat(fig, ax, values, arms, sites, title, limit=(-1.5, 1.5), log=True)
    fig.suptitle(
        "Final fixed TRAIN contexts: mean within-context ratios (four contexts)"
    )
    fig.tight_layout()
    save(fig, a.output, "layer-signals")
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    blocks = [f"block{i}" for i in range(4)]
    for ax, key, title in zip(
        axes[:2],
        ("output_rms", "output_gradient_rms"),
        ("Block output RMS / Dense", "Block output cotangent RMS / Dense"),
    ):
        values = [
            [
                statistics.mean(
                    r["layers"][site][key] / d["layers"][site][key]
                    for r, d in zip(signals[arm], signals["dense"])
                )
                for site in blocks
            ]
            for arm in arms
        ]
        heat(fig, ax, values, arms, blocks, title, limit=(-1.5, 1.5), log=True)
    values = [
        [signals[arm][0]["residual"][str(i)]["ffn_over_residual_rms"] for i in range(4)]
        for arm in arms
    ]
    heat(
        fig,
        axes[2],
        values,
        arms,
        blocks,
        "FFN output / pre-FFN residual RMS (offset0)",
        cmap="viridis",
    )
    fig.tight_layout()
    save(fig, a.output, "block-residual")
    index = {
        (r["arm"], r["site"], r["step"]): r["metrics"] for r in analysis["records"]
    }
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for ax, key, title in zip(
        axes,
        ("cka_to_dense", "cosine_to_dense", "gradient"),
        (
            "Block representation CKA to Dense",
            "Block representation cosine to Dense",
            "Block cotangent cosine to Dense",
        ),
    ):
        values = [
            [
                index[(arm, site, final)]["dy" if key == "gradient" else "y"][
                    "cosine_to_dense" if key == "gradient" else key
                ]
                for site in blocks
            ]
            for arm in arms
        ]
        heat(
            fig,
            ax,
            values,
            arms,
            blocks,
            title,
            limit=((0, 1) if key == "cka_to_dense" else (-1, 1)),
            cmap="viridis" if key == "cka_to_dense" else "coolwarm",
        )
    fig.tight_layout()
    save(fig, a.output, "representations")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, field, key, title in zip(
        axes,
        ("y", "y", "dy"),
        ("row_centered_rms", "feature_dc_energy_share", "row_centered_rms"),
        (
            "Block row-centered output RMS / Dense",
            "Block output feature-mean energy (%)",
            "Block row-centered cotangent RMS / Dense",
        ),
    ):
        values = []
        for arm in arms:
            row = []
            for site in blocks:
                value = index[(arm, site, final)][field]["mean_components"][key]
                if key == "row_centered_rms":
                    value /= index[("dense", site, final)][field]["mean_components"][
                        key
                    ]
                else:
                    value *= 100
                row.append(value)
            values.append(row)
        heat(
            fig,
            ax,
            values,
            arms,
            blocks,
            title,
            limit=(-1.5, 1.5) if key == "row_centered_rms" else (0, 0.2),
            cmap="coolwarm" if key == "row_centered_rms" else "viridis",
            log=key == "row_centered_rms",
        )
    fig.suptitle(
        "Offset0: separate the feature-constant component removed by LayerNorm"
    )
    fig.tight_layout()
    save(fig, a.output, "centered-signals")
    fig, ax = plt.subplots(figsize=(9, 7))
    values = [
        [row["delta_ce"] for row in ablation["arms"][arm]["zeroed_ffn_branches"]]
        for arm in arms
    ]
    heat(
        fig,
        ax,
        values,
        arms,
        blocks,
        "Task DEV CE increase when one trained FFN branch is zeroed",
        cmap="magma",
    )
    ax.set_xlabel("Zeroed FFN branch; all other trained weights fixed")
    fig.suptitle(
        "Post-hoc conditional dependence; no training updates, non-additive interventions"
    )
    fig.tight_layout()
    save(fig, a.output, "branch-dependence")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, field, metric, title in zip(
        axes.flat,
        ("y", "y", "dy", "dy"),
        ("lag1", "low_frequency_share", "lag1", "high_frequency_share"),
        (
            "Block output: adjacent-feature cosine",
            "Block output: low-frequency energy share",
            "Block cotangent: adjacent-feature cosine",
            "Block cotangent: high-frequency energy share",
        ),
    ):
        values = []
        for arm in arms:
            row = []
            for site in blocks:
                spatial = index[(arm, site, final)][field]["spatial"]
                row.append(
                    spatial["lag_cosines"]["1"] if metric == "lag1" else spatial[metric]
                )
            values.append(row)
        heat(
            fig,
            ax,
            values,
            arms,
            blocks,
            title,
            limit=(-1, 1) if metric == "lag1" else (0, 1),
            cmap="coolwarm" if metric == "lag1" else "viridis",
        )
    fig.suptitle(
        "Post-hoc read-only feature-order analysis: token and row means removed; offset0"
    )
    fig.tight_layout()
    save(fig, a.output, "feature-order")
    # Prespecified observations over time for each block, with unpooled context ranges.
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for arm in (
        "dst-pair12",
        "pair12",
        "all-cst",
        "pair01",
        "pair23",
        "up-even",
        "down-even",
    ):
        for i in range(4):
            for row, key in enumerate(("output_rms", "output_gradient_rms")):
                means = []
                for clock in receipt["protocol"]["signals"]:
                    sig = read(a.run / arm / f"signals-{clock}.json")["records"]
                    den = read(a.run / "dense" / f"signals-{clock}.json")["records"]
                    means.append(
                        statistics.mean(
                            r["layers"][f"block{i}"][key]
                            / d["layers"][f"block{i}"][key]
                            for r, d in zip(sig, den)
                        )
                    )
                axes[row, i].plot(
                    np.array(receipt["protocol"]["signals"]) * 1024 / 1e6,
                    means,
                    "o-",
                    color=COLORS[arm],
                    label=arm,
                    ms=3,
                )
                axes[row, i].axhline(1, color="gray", lw=0.6)
                axes[row, i].set_title(
                    f"Block{i}: {'output' if row == 0 else 'cotangent'} RMS / Dense"
                )
                axes[row, i].grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("Training targets (millions)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, a.output, "signal-trajectories")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for arm in (*PRIMARY, "all-cst"):
        clocks = sorted({r["step"] for r in atoms["records"] if r["arm"] == arm})
        values = [[] for _ in range(6)]
        for clock in clocks:
            rows = [
                r["subsets"]["active_amp_ge_0.01"]
                for r in atoms["records"]
                if r["arm"] == arm and r["step"] == clock
            ]
            total = sum(r["count"] for r in rows)

            def average(fn, rows=rows, total=total):
                return sum(r["count"] * fn(r) for r in rows if r["count"]) / total

            values[0].append(
                average(lambda r: r["absolute_amplitude"]["rms"] ** 2) ** 0.5
            )
            values[1].append(average(lambda r: r["radius"]["mean"]))
            values[2].append(average(lambda r: r["fraction_at_zero_share"]))
            values[3].append(
                average(
                    lambda r: (
                        sum(
                            r["fields"][f]["adam_scale_below_eps_share"]
                            for f in ("output_center", "input_center")
                        )
                        / 2
                    )
                )
            )
            values[4].append(
                average(
                    lambda r: (
                        sum(
                            r["fields"][f]["sub_half_ulp_proposal_share"]
                            for f in ("output_center", "input_center")
                        )
                        / 2
                    )
                )
            )
            values[5].append(average(lambda r: r["centers_both_within_0.01_share"]))
        for ax, v in zip(axes.flat, values):
            ax.plot(
                np.array(clocks) * 1024 / 1e6,
                v,
                "o-",
                label=arm,
                color=COLORS[arm],
                ms=4,
            )
    for ax, title in zip(
        axes.flat,
        (
            "Pooled amplitude RMS",
            "Mean physical radius",
            "Fraction at lower boundary0",
            "Center Adam scale < epsilon share",
            "Center moment proposal < half ULP share",
            "Both centers within0.01 of grid sites share",
        ),
    ):
        ax.set_title(title)
        ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("Training targets (millions)")
    axes[0, 0].legend(fontsize=7, ncol=3)
    fig.suptitle(
        "Post-hoc checkpoint diagnostics, active |amplitude|>=0.01; moments are not actual next updates"
    )
    fig.tight_layout()
    save(fig, a.output, "atom-dynamics")
    (a.output / "results.json").write_text(
        json.dumps(
            {
                "source_commits": analysis["source_commits"],
                "endpoints": analysis["endpoints"],
                "primary_family": PRIMARY,
                "signal_records": analysis["records"],
                "atom_records": atoms["records"],
                "branch_dependence": ablation["arms"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
