"""Verify the executed adaptive pilot and overlay all matched DST controls."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def terminal(root, arm):
    return next(r["development"] for r in curves(root, arm) if r["step"] == 976)


def curves(root, arm):
    return [
        json.loads(s) for s in (root / arm / "curves.jsonl").read_text().splitlines()
    ]


def numerical_checks(root, coefficients):
    errors = []
    for c in coefficients:
        initial = read(root / f"qualification-c{c}/qualification.json")["arms"][
            f"m8-c{c}"
        ]["oracles"]
        final = read(root / f"m8-c{c}/final-oracles.json")
        for stage in (initial, final):
            assert set(stage) == {"1.up", "1.down"}
            for row in stage.values():
                assert row["passed"] and row["A"] == c * 1024 and row["B"] == 3
                assert set(row["errors"]) == {
                    "Y",
                    "dX",
                    "amplitude",
                    "radius_fraction",
                    "output_center",
                    "input_center",
                }
                for error in row["errors"].values():
                    assert error["max_abs"] <= 3e-4 and error["relative_l2"] <= 1e-4
                    errors.append(error)
    return {
        "tensor_field_checks": len(errors),
        "max_abs": max(x["max_abs"] for x in errors),
        "max_relative_l2": max(x["relative_l2"] for x in errors),
        "original_component_failures": sum(
            x["original_component_gate_failures"] for x in errors
        ),
        "tolerances_changed": False,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dst-run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = read(a.run / "manifest.json")
    for name, h in manifest["files"].items():
        assert hashlib.sha256((a.run / name).read_bytes()).hexdigest() == h, name
    receipt, protocol = read(a.run / "receipt.json"), read(a.run / "protocol.json")
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    assert all(x["returncode"] == 0 for x in receipt["commands"])
    assert receipt["ordinary_cpp_guard_unchanged_without_c16_macro"]
    assert receipt["default_packed_execution_ast_equivalent"]
    assert receipt["train_audit_bodies_identical_after_explicit_factory_binding"]
    decisions = read(a.run / "decisions.json")
    assert decisions["state"] == "completed" and not decisions["confirmation_evaluated"]
    coarse, refinement = decisions["coarse"], decisions["refinement"]
    assert [r["coefficient"] for r in coarse] == [16, 32, 64, 128][: len(coarse)]
    assert not any(r["near_dense"] for r in coarse[:-1])
    if coarse[-1]["near_dense"] and len(coarse) > 1:
        assert [r["coefficient"] for r in refinement] == [
            (coarse[-1]["coefficient"] + coarse[-2]["coefficient"]) // 2
        ]
    else:
        assert not refinement
        assert coarse[-1]["near_dense"] or len(coarse) == 4
    coefficients = sorted(r["coefficient"] for r in coarse + refinement)
    baseline = terminal(a.run, "dense")
    assert baseline == decisions["fresh_dense"]
    passed = []
    for r in coarse + refinement:
        dev = terminal(a.run, f"m8-c{r['coefficient']}")
        ok = (
            dev["ce"] <= baseline["ce"] + 0.01
            and dev["next_token_top1_accuracy"]
            >= baseline["next_token_top1_accuracy"] - 0.005
        )
        assert dev == r["development"] and ok == r["near_dense"]
        if ok:
            passed.append(r["coefficient"])
    assert decisions["selected_coefficient"] == (min(passed) if passed else None)
    selected = {}
    all_dst = []
    for c in coefficients:
        s = read(a.run / f"selection-c{c}/selection.json")
        assert (
            len(s["candidates"]) == 3
            and s["selection_on_development"]
            and not s["confirmation_evaluated"]
        )
        winner = min(
            s["candidates"], key=lambda r: (r["development"]["ce"], r["sparse_lr"])
        )
        assert s["selected"] == {str(c): winner["arm"]}
        selected[c] = winner["arm"]
        all_dst.extend(r["arm"] for r in s["candidates"])
    assert {d.resolve() for d in a.dst_run} == {
        (a.run / v).resolve() for v in selected.values()
    }
    all_arms = ["dense", *[f"m8-c{c}" for c in coefficients], *all_dst]
    assert {
        d.name for d in a.run.iterdir() if d.is_dir() and (d / "summary.json").exists()
    } == set(all_arms)
    summaries = {arm: read(a.run / arm / "summary.json") for arm in all_arms}
    base = summaries["dense"]
    for c in [0, *coefficients]:
        q = read(a.run / f"qualification-c{c}/qualification.json")
        audit = read(a.run / f"audit-c{c}/audit.json")
        counts = read(a.run / f"audit-c{c}/active-count-audit.json")
        expected = (
            {"dense"}
            if c == 0
            else {f"m8-c{c}", *[f"dst-c{c}-lr{lr:g}" for lr in (0.003, 0.01, 0.03)]}
        )
        assert q["state"] == audit["state"] == counts["state"] == "passed"
        assert set(q["arms"]) == set(audit["arms"]) == expected
        assert all(
            r["serialized_resume_and_probe_next_update_exact"]
            for r in q["arms"].values()
        )
        for records in audit["arms"].values():
            assert {r["step"] for r in records} == {256, 976}
        assert set(counts["arms"]) == (expected if c else set())
        for records in counts["arms"].values():
            assert {r["step"] for r in records} == {256, 976}
            assert all(set(r["slots_per_map"].values()) == {c * 1024} for r in records)
    for arm, s in summaries.items():
        assert s["state"] == "completed" and s["completed_steps"] == 976
        assert s["gpu"] == "NVIDIA GeForce RTX 4090"
        for key in (
            "data_manifest_sha256",
            "data_order_sha256",
            "steps",
            "schedule_steps",
            "seed",
            "effective_tokens_per_step",
            "completed_tokens",
        ):
            assert s[key] == base[key], (arm, key)
        for key in base["config"]:
            if key != "coefficient":
                assert s["config"][key] == base["config"][key], (arm, key)
        if arm != "dense":
            count = s["config"]["coefficient"] * 1024
            expected_scalars = 8 * count * (4 if arm.startswith("m8") else 1) + 10240
            assert s["active_ffn_scalars"] == expected_scalars
            if arm.startswith("dst"):
                assert len(s["dst_invariants"]) == 8
                assert all(
                    r["active_slots"] == count for r in s["dst_invariants"].values()
                )
        assert s["actual_adam_updates"] == (973 if arm.startswith("dst") else 976)
    arms = [
        "dense",
        *[name for c in coefficients for name in (f"m8-c{c}", selected[c])],
    ]
    table = []
    for arm in arms:
        s, d = summaries[arm], terminal(a.run, arm)
        table.append(
            {
                "arm": arm,
                "coefficient": None if arm == "dense" else s["config"]["coefficient"],
                "active_ffn_scalars": s["active_ffn_scalars"],
                "development": d,
                "ce_gap": d["ce"] - baseline["ce"],
                "top1_gap_pp": (
                    d["next_token_top1_accuracy"] - baseline["next_token_top1_accuracy"]
                )
                * 100,
                "step_median_seconds": statistics.median(s["timing_seconds"]),
                "peak_allocated_bytes": s["peak_allocated_bytes"],
                "peak_reserved_bytes": s["peak_reserved_bytes"],
            }
        )
    a.output.mkdir(parents=True, exist_ok=False)

    def save(fig, name, rect=None):
        fig.tight_layout(rect=rect)
        for suffix in ("png", "svg"):
            out = a.output / f"{name}.{suffix}"
            fig.savefig(out, dpi=150)
            if suffix == "svg":
                out.write_text(
                    "\n".join(r.rstrip() for r in out.read_text().splitlines()) + "\n"
                )
        plt.close(fig)

    colors = dict(zip(coefficients, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    dense_ce = {
        r["step"]: r["development"]["ce"]
        for r in curves(a.run, "dense")
        if "development" in r
    }
    for arm in arms:
        c = summaries[arm]["config"]["coefficient"]
        color, style = (
            ("black", "-")
            if arm == "dense"
            else (colors[c], "--" if arm.startswith("dst") else "-")
        )
        label = (
            "Dense"
            if arm == "dense"
            else (
                f"M8 A{c * 1024}"
                if arm.startswith("m8")
                else f"DST mask1={c * 1024}, DEV selected"
            )
        )
        dev = [r for r in curves(a.run, arm) if "development" in r and r["step"] >= 200]
        for ax, gap in zip(axes, (False, True)):
            ax.plot(
                [r["seen_tokens"] / 1e6 for r in dev],
                [
                    r["development"]["ce"] - (dense_ce[r["step"]] if gap else 0)
                    for r in dev
                ],
                color=color,
                linestyle=style,
                marker="o",
                label=label,
            )
            ax.set(
                xlabel="TRAIN targets (million)",
                ylabel="Validation CE minus Dense" if gap else "Validation CE",
            )
            ax.grid(alpha=0.2)
    axes[1].axhline(0.01, color="green", linestyle=":", label="pilot CE margin .01")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle("RTX4090, all8 FFN maps; unconnected M8; seed17 validation curves")
    save(fig, "learning", rect=(0, 0.13, 1, 0.95))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for kind, color in (("m8", "#2368b8"), ("dst", "#bf821f")):
        rr = [r for r in table if r["arm"].startswith(kind)]
        for ax, key in zip(axes, ("ce_gap", "top1_gap_pp")):
            ax.plot(
                [r["coefficient"] * 1024 for r in rr],
                [r[key] for r in rr],
                marker="o",
                color=color,
                label="CST M8" if kind == "m8" else "dense-mask DST, DEV selected",
            )
            ax.set_xscale("log", base=2)
            ax.set_xticks(
                [c * 1024 for c in coefficients],
                [f"{c * 1024:,}" for c in coefficients],
                rotation=25,
            )
            ax.set(
                xlabel="Atom count = mask-one count per Linear",
                ylabel="Validation CE gap"
                if key == "ce_gap"
                else "Next-token accuracy gap (pp)",
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(alpha=0.2)
    axes[0].axhline(0.01, color="green", linestyle=":", label="pilot CE margin .01")
    axes[1].axhline(
        -0.5, color="green", linestyle=":", label="pilot Top1 margin -.5 pp"
    )
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Final validation: descriptive pilot margins, seed17")
    save(fig, "budget")
    rows = math.ceil(len(coefficients) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
    for ax, c in zip(axes.flat, coefficients):
        for lr in (0.003, 0.01, 0.03):
            arm = f"dst-c{c}-lr{lr:g}"
            dev = [
                r for r in curves(a.run, arm) if "development" in r and r["step"] >= 200
            ]
            ax.plot(
                [r["seen_tokens"] / 1e6 for r in dev],
                [r["development"]["ce"] for r in dev],
                marker="o",
                label=f"LR{lr:g}" + (" selected" if selected[c] == arm else ""),
            )
        ax.set(
            title=f"DST mask1={c * 1024}",
            xlabel="TRAIN targets (million)",
            ylabel="Validation CE",
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(coefficients) :]:
        ax.axis("off")
    fig.suptitle("All DST learning-rate candidates; selection on validation")
    save(fig, "dst-lr")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, scale in zip(
        axes,
        ("step_median_seconds", "peak_allocated_bytes", "peak_reserved_bytes"),
        (1000, 1 / 2**20, 1 / 2**20),
    ):
        for kind in ("m8", "dst"):
            rr = [r for r in table if r["arm"].startswith(kind)]
            ax.plot(
                [r["coefficient"] * 1024 for r in rr],
                [r[key] * scale for r in rr],
                marker="o",
                label="CST M8" if kind == "m8" else "Dense-mask DST",
            )
        ax.axhline(table[0][key] * scale, color="black", linestyle="--", label="Dense")
        ax.set(
            xlabel="Atom count = mask-one count",
            ylabel="ms" if key == "step_median_seconds" else "MiB",
            title={
                "step_median_seconds": "Complete-step median",
                "peak_allocated_bytes": "Peak allocated",
                "peak_reserved_bytes": "Peak reserved",
            }[key],
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle("Actual complete-step cost on RTX4090; dense-mask DST")
    save(fig, "cost")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for arm in arms:
        c = summaries[arm]["config"]["coefficient"]
        color, style = (
            ("black", "-")
            if arm == "dense"
            else (colors[c], "--" if arm.startswith("dst") else "-")
        )
        rr = [r for r in curves(a.run, arm) if "train_ce" in r]
        ax.plot(
            [r["seen_tokens"] / 1e6 for r in rr],
            [r["train_ce"] for r in rr],
            color=color,
            linestyle=style,
            label=arm,
        )
    ax.set(
        xlabel="TRAIN targets (million)",
        ylabel="Raw TRAIN CE",
        title="Identical data order; raw losses, no smoothing",
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    save(fig, "training")
    geometry = {
        str(c): {
            str(k): read(a.run / f"m8-c{c}/geometry-{k}.json") for k in (0, 256, 976)
        }
        for c in coefficients
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def weighted(c, clock, key):
        g = geometry[str(c)][str(clock)]
        return sum(r[key] * r["A"] for r in g.values()) / sum(
            r["A"] for r in g.values()
        )

    for clock in (0, 976):
        axes[0].plot(
            [c * 1024 for c in coefficients],
            [weighted(c, clock, "radius_mean") for c in coefficients],
            marker="o",
            label=f"step{clock}",
        )
        axes[1].plot(
            [c * 1024 for c in coefficients],
            [weighted(c, clock, "fraction_mean") for c in coefficients],
            marker="o",
            label=f"step{clock}",
        )
    for field, label in ((2, "output center"), (3, "input center")):
        vals = []
        for c in coefficients:
            g = geometry[str(c)]["976"]
            vals.append(
                100
                * sum(
                    r["active_moment_scale_below_1e8_share"][field] * r["active_atoms"]
                    for r in g.values()
                )
                / sum(r["active_atoms"] for r in g.values())
            )
        axes[2].plot([c * 1024 for c in coefficients], vals, marker="o", label=label)
    for ax, title in zip(
        axes,
        (
            "Mean radius: fixed policy, different distributions",
            "Mean radius fraction",
            "Active coordinate moment scale <1e-8 (%)",
        ),
    ):
        ax.set(xlabel="CST Atom count per Linear", title=title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle("Unconnected M8 geometry with amplitude-dependent radius bounds")
    save(fig, "geometry")
    result = {
        "state": "passed",
        "numerical_validation": numerical_checks(a.run, coefficients),
        "qualified_arms": 1 + 4 * len(coefficients),
        "checkpoint_audits": 2 + 8 * len(coefficients),
        "active_count_audits": 8 * len(coefficients),
        "command_seconds": {
            row["label"]: row["finished"] - row["started"]
            for row in receipt["commands"]
        },
        "decisions": decisions,
        "geometry": geometry,
        "results": table,
        "all_dst_selections": {
            str(c): read(a.run / f"selection-c{c}/selection.json") for c in coefficients
        },
        "source_commits": protocol["source_commits"],
        "scope": "one seed short adaptive DEV pilot; same-job fresh4090 Dense; smallest tested pass,not Atom lower bound/convergence/independent confirmation",
        "dst_storage": "dense weight plus Boolean mask; no COO/CSR",
    }
    (a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
