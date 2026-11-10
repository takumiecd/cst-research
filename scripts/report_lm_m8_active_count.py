"""Primary A=mask-one comparison; dense-mask DST, all LR candidates retained."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def verify(root):
    manifest = read(root / "manifest.json")
    for name, h in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == h, name
    receipt = read(root / "receipt.json")
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    assert read(root / "audit/audit.json")["state"] == "passed"
    return receipt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--control-run", type=Path, required=True)
    p.add_argument("--dst-run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    new, old = verify(a.run), verify(a.control_run)
    assert new["ordinary_cpp_guard_unchanged_without_c16_macro"]
    assert new["default_packed_execution_ast_equivalent"]
    assert new["train_audit_bodies_identical_after_explicit_factory_binding"]
    q = read(a.run / "qualification/qualification.json")
    assert (
        q["state"] == "passed"
        and q["m8_c8_initial_parameters_match"]
        and q["ordinary_parameters_match"]
    )
    assert (
        q["prior_manifest_sha256"]
        == hashlib.sha256((a.control_run / "manifest.json").read_bytes()).hexdigest()
    )
    counts = read(a.run / "audit/active-count-audit.json")
    assert counts["state"] == "passed"
    assert len(q["arms"]) == len(counts["arms"]) == 7
    assert all(
        row["serialized_resume_and_probe_next_update_exact"]
        for row in q["arms"].values()
    )
    for records in counts["arms"].values():
        assert {r["step"] for r in records} == {256, 976}
    selection = read(a.run / "selection/selection.json")
    assert selection["state"] == "passed" and selection["selection_on_development"]
    for c in (8, 16):
        winner = min(
            (r for r in selection["candidates"] if r["coefficient"] == c),
            key=lambda r: (r["development"]["ce"], r["sparse_lr"]),
        )
        assert winner["arm"] == selection["selected"][str(c)]
    chosen = [selection["selected"][str(c)] for c in (8, 16)]
    assert {p.resolve() for p in a.dst_run} == {
        (a.run / arm).resolve() for arm in chosen
    }
    directories = {
        "dense": a.control_run / "dense",
        "m8-c8": a.control_run / "legacy-m8",
        "m8-c16": a.run / "m8-c16",
    } | {arm: a.run / arm for arm in chosen}
    arms = ["dense", "m8-c8", "m8-c16", *chosen]
    labels = {
        "dense": "Dense (prior)",
        "m8-c8": "M8 A8192 (prior)",
        "m8-c16": "M8 A16384 (new)",
    } | {arm: f"DST mask1={c * 1024}, DEV-selected" for c, arm in zip((8, 16), chosen)}
    colors = ("#252525", "#55729b", "#1e70b4", "#b5822c", "#d29529")
    summaries = {arm: read(d / "summary.json") for arm, d in directories.items()}
    base = summaries["dense"]
    all_dst = [r["arm"] for r in selection["candidates"]]
    for arm in all_dst:
        s = read(a.run / arm / "summary.json")
        expected = int(s["config"]["coefficient"]) * 1024
        assert len(s["dst_invariants"]) == 8
        assert all(v["active_slots"] == expected for v in s["dst_invariants"].values())
        assert s["active_ffn_scalars"] == 8 * expected + 10240
        for record in counts["arms"][arm]:
            assert set(record["slots_per_map"].values()) == {expected}
    for arm, s in summaries.items():
        assert s["state"] == "completed" and s["completed_steps"] == 976
        for key in (
            "data_manifest_sha256",
            "data_order_sha256",
            "steps",
            "schedule_steps",
            "seed",
            "effective_tokens_per_step",
            "completed_tokens",
            "gpu",
        ):
            assert s[key] == base[key], (arm, key)
        for key in base["config"]:
            if key != "coefficient":
                assert s["config"][key] == base["config"][key], (arm, key)
        assert s["actual_adam_updates"] == (973 if arm.startswith("dst") else 976)
    for arm, c in (("m8-c16", 16),):
        for record in counts["arms"][arm]:
            assert set(record["slots_per_map"].values()) == {c * 1024}
    curves = {
        arm: [
            json.loads(l)
            for l in (directories[arm] / "curves.jsonl").read_text().splitlines()
        ]
        for arm in arms
    }
    dense = {
        r["step"]: r["development"]["ce"] for r in curves["dense"] if "development" in r
    }
    a.output.mkdir(parents=True, exist_ok=False)

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "svg"):
            target = a.output / f"{name}.{suffix}"
            fig.savefig(target, dpi=150)
            if suffix == "svg":
                target.write_text(
                    "\n".join(l.rstrip() for l in target.read_text().splitlines())
                    + "\n"
                )
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    table = []
    for arm, color in zip(arms, colors):
        dev = [r for r in curves[arm] if "development" in r and r["step"] >= 200]
        x = [r["seen_tokens"] / 1e6 for r in dev]
        style = "--" if "c16" in arm else "-"
        axes[0].plot(
            x,
            [r["development"]["ce"] for r in dev],
            label=labels[arm],
            color=color,
            marker="o",
            linestyle=style,
        )
        axes[1].plot(
            x,
            [r["development"]["ce"] - dense[r["step"]] for r in dev],
            label=labels[arm],
            color=color,
            marker="o",
            linestyle=style,
        )
        s = summaries[arm]
        table.append(
            {
                "arm": arm,
                "fresh": arm not in ("dense", "m8-c8"),
                "active_units_per_map": None
                if arm == "dense"
                else int(s["config"]["coefficient"]) * 1024,
                "active_ffn_scalars": s["active_ffn_scalars"],
                "development": dev[-1]["development"],
                "step_median_seconds": statistics.median(s["timing_seconds"]),
                "peak_allocated_bytes": s["peak_allocated_bytes"],
                "peak_reserved_bytes": s["peak_reserved_bytes"],
            }
        )
    axes[0].set(ylabel="DEV CE", title="After warmup")
    axes[1].set(
        ylabel="DEV CE minus Dense",
        title="Primary criterion: Atom count = mask-one count",
    )
    for ax in axes:
        ax.set_xlabel("TRAIN targets (million)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "M8 versus dense-mask DST; seed17, short DEV pilot; prior controls labeled"
    )
    save(fig, "learning")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for c, ax in zip((8, 16), axes):
        for row in (r for r in selection["candidates"] if r["coefficient"] == c):
            cc = [
                json.loads(l)
                for l in (a.run / row["arm"] / "curves.jsonl").read_text().splitlines()
            ]
            dd = [r for r in cc if "development" in r and r["step"] >= 200]
            ax.plot(
                [r["seen_tokens"] / 1e6 for r in dd],
                [r["development"]["ce"] for r in dd],
                marker="o",
                label=f"LR{row['sparse_lr']:g}"
                + (" selected" if row["arm"] == selection["selected"][str(c)] else ""),
            )
        ax.set(
            title=f"DST mask1={c * 1024}: all LR candidates",
            xlabel="TRAIN targets (million)",
            ylabel="DEV CE",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("DEV selection, one seed; no confirmation evaluation")
    save(fig, "dst-lr")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for arm, color in zip(arms, colors):
        cc = [r for r in curves[arm] if "train_ce" in r]
        ax.plot(
            [r["seen_tokens"] / 1e6 for r in cc],
            [r["train_ce"] for r in cc],
            label=labels[arm],
            color=color,
            linestyle="--" if "c16" in arm else "-",
        )
    ax.set(
        xlabel="TRAIN targets (million)",
        ylabel="TRAIN CE",
        title="Raw logged losses; identical data order",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "training")
    geometry = {
        arm: [read(directories[arm] / f"geometry-{s}.json") for s in (0, 256, 976)]
        for arm in ("m8-c8", "m8-c16")
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for arm, snapshots in geometry.items():
        for ax, key in zip(
            axes, ("radius_mean", "active_sharp_share", "active_both_near_grid_share")
        ):
            weight = "A" if key == "radius_mean" else "active_atoms"
            y = [
                sum(r[key] * r[weight] for r in g.values())
                / sum(r[weight] for r in g.values())
                for g in snapshots
            ]
            ax.plot([0, 256, 976], y, label=labels[arm], marker="o")
            ax.set(xlabel="updates", ylabel=key)
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("CST-only geometry; same M8/radius policy/LR/epsilon")
    save(fig, "geometry")
    sites = [f"{i}.{direction}" for i in range(4) for direction in ("up", "down")]
    signals = {}
    for arm in arms:
        signals[arm] = {}
        for clock in (0, 256, 976):
            records = read(directories[arm] / f"signals-{clock}.json")["records"]
            assert [r["offset"] for r in records] == [0, 4096, 8192, 12288]
            signals[arm][clock] = {
                site: {
                    "output_rms": statistics.mean(
                        r["layers"][site]["output_rms"] for r in records
                    ),
                    "output_gradient_rms": statistics.mean(
                        r["layers"][site]["output_gradient_rms"] for r in records
                    ),
                    "stable_rank": statistics.mean(
                        r["layers"][site]["output_features"]["stable_rank"]
                        for r in records
                    ),
                }
                for site in sites
            }
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for row, clock in zip(axes, (256, 976)):
        for ax, key in zip(row, ("output_rms", "output_gradient_rms", "stable_rank")):
            for arm, color in zip(arms, colors):
                values = [
                    signals[arm][clock][site][key] / signals["dense"][clock][site][key]
                    for site in sites
                ]
                ax.plot(sites, values, marker="o", color=color, label=labels[arm])
            if key != "stable_rank":
                ax.set_yscale("log")
            ax.set(title=f"{clock} updates: {key} / Dense", xlabel="FFN map")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Mean across four fixed TRAIN contexts; each model's own inputs/gradients"
    )
    save(fig, "layers")
    write = {
        "state": "passed",
        "primary_matching": "CST Atom count equals DST mask-one count per Linear",
        "dst_storage": "dense weight+boolean mask; no COO/CSR",
        "scope": "one-seed short DEV-selected comparison; cross-job timing exploratory; not a storage-format claim",
        "fresh_source_commits": new["protocol"]["source_commits"],
        "control_source_commits": old["protocol"]["source_commits"],
        "results": table,
        "dst_selection": selection,
        "geometry": geometry,
        "layer_signal_means": signals,
        "layer_signal_scope": "four fixed TRAIN contexts; each model's own inputs/gradients, not a causal or shared-input map comparison",
    }
    (a.output / "results.json").write_text(json.dumps(write, indent=2) + "\n")


if __name__ == "__main__":
    main()
