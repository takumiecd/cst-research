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
COLORS.update(zip(PRIMARY, plt.get_cmap("tab10").colors))


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
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
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
        de = [r for r in curves[arm] if "development" in r and r["step"]]
        x = np.array([r["seen_tokens"] for r in tr]) / 1e6
        yy = np.array([r["train_ce"] for r in tr])
        # Trailing five logged observations (~100 updates); raw rows remain available.
        ys = [np.mean(yy[max(0, i - 4) : i + 1]) for i in range(len(yy))]
        style = {
            "color": COLORS[arm],
            "label": arm,
            "lw": 2 if arm in ("dense", "all-cst", "pair12") else 1.2,
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
    axes[0].set(title="TRAIN CE (trailing 5 logs)", ylim=(2.35, 4.5))
    axes[1].set(title="DEV CE, same8192 targets", ylim=(2.45, 4.5))
    axes[2].set(title="DEV CE minus Dense")
    axes[2].axhline(0, color="gray", lw=0.7)
    for ax in axes:
        ax.set_xlabel("Training targets (millions)")
        ax.grid(alpha=0.2)
    axes[2].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Connected M16: eight matched CST4 placements + three controls; seed17"
    )
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
    (a.output / "results.json").write_text(
        json.dumps(
            {
                "source_commits": analysis["source_commits"],
                "endpoints": analysis["endpoints"],
                "primary_family": PRIMARY,
                "signal_records": analysis["records"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
