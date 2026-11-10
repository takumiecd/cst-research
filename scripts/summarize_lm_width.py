"""Summarize the stopped width study using archived, hash-verified records only."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cst_research.experiments.ffn_training_plan import TrainingPlan
from cst_research.experiments.lm_width_contract import (
    ARMS,
    LONG_STEPS,
    SHORT_STEPS,
    WIDTHS,
    choose,
)

COLORS = {"dense": "#303030", "cst": "#c84b31", "dst": "#236eb1"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curves(path):
    result = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    assert all(b["step"] > a["step"] for a, b in pairwise(result))
    assert all(s["seen_tokens"] == 1024 * s["step"] for s in result)
    for row in result:
        if "development" not in row:
            continue
        d = row["development"]
        assert d["tokens"] == 8192 and math.isfinite(d["ce"])
        assert d["next_token_top1_accuracy"] == d["correct_next_tokens"] / 8192
        assert abs(d["perplexity"] - math.exp(d["ce"])) < 1e-9
    return result


def terminal(rows, steps):
    return next(s["development"] for s in rows if s["step"] == steps)


def validate(run, source, stop):
    manifest, protocol, receipt = (
        read(run / n) for n in ("manifest.json", "protocol.json", "receipt.json")
    )
    assert all(sha(run / n) == h for n, h in manifest["files"].items())
    assert sha(source) == manifest["source_archive_sha256"] == protocol["source_sha256"]
    assert manifest["source_files"] == protocol["source_files"]
    with tarfile.open(source) as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        assert set(members) == set(manifest["source_files"])
        for name, expected in manifest["source_files"].items():
            assert (
                hashlib.sha256(archive.extractfile(members[name]).read()).hexdigest()
                == expected
            )
    assert read(stop)["state"] == "stopped_by_user"
    assert not read(stop)["remaining_owned_targets"]
    assert all(r["returncode"] == 0 for r in receipt["commands"])
    assert receipt["ordinary_cpp_guard_unchanged_without_width_macro"]
    assert "148 passed" in (run / "native-tests.log").read_text()
    result = {
        "status": "STOPPED",
        "scope": "24 short and8 long arms completed; final width2048 DST incomplete; no new training or scoring",
        "source_commits": protocol["source_commits"],
        "source_archive_sha256": protocol["source_sha256"],
        "source_files": len(manifest["source_files"]),
        "payloads_verified": len(manifest["files"]),
        "stop_receipt_sha256": sha(stop),
        "qualification_arms": 0,
        "checkpoint_audits": 0,
        "short": {},
        "long": {},
        "selection": {},
        "geometry": {},
        "curves": {},
        "missing_long": [],
        "confirmation_evaluated": False,
    }
    order = None
    numerical = []
    for width in WIDTHS:
        w = str(width)
        base = run / f"width{width}"
        q = read(base / "qualification/qualification.json")
        assert q["state"] == "passed" and set(q["arms"]) == set(ARMS)
        assert all(
            v["serialized_resume_and_probe_next_update_exact"]
            for v in q["arms"].values()
        )
        result["qualification_arms"] += len(q["arms"])
        selection = read(base / "selection/selection.json")
        candidate_metrics = {}
        result["short"][w], result["long"][w], result["curves"][w] = {}, {}, {}
        for arm, (kind, _) in ARMS.items():
            directory = base / "short" / arm
            s = read(directory / "summary.json")
            assert s["state"] == "completed" and s["completed_steps"] == SHORT_STEPS
            assert s["config"] == q["arms"][arm]["config"]
            assert (
                s["initial_parameters_sha256"]
                == q["arms"][arm]["initial_parameters_sha256"]
            )
            assert s["data_manifest_sha256"] == protocol["data_manifest_sha256"]
            assert s["config"]["coefficient"] == 64 and s["width"] == width
            assert s["effective_tokens_per_step"] == 1024 and s["seed"] == 17
            assert order in (None, s["data_order_sha256"])
            order = s["data_order_sha256"]
            rows = curves(directory / "curves.jsonl")
            metric = terminal(rows, SHORT_STEPS)
            candidate_metrics[arm] = metric
            result["curves"][w][arm] = {"short": rows}
            result["short"][w][arm] = describe(s, metric, kind)
            audit = read(base / f"audit-short-{arm}/audit.json")
            check_audit(s, audit, SHORT_STEPS, kind)
            result["checkpoint_audits"] += len(audit["checkpoints"])
            if kind == "cst":
                numerical.extend(oracle_errors(q["arms"][arm]["oracles"], 128 * width))
                numerical.extend(oracle_errors(audit["oracles"], 128 * width))
        selected = [
            choose(candidate_metrics, "dense"),
            "cst-c64",
            choose(candidate_metrics, "dst"),
        ]
        assert selection["state"] == "passed" and selection["selected"] == selected
        assert selection["candidates"] == candidate_metrics
        result["selection"][w] = selected
        for arm in selected:
            kind = ARMS[arm][0]
            directory = base / "long" / arm
            if not directory.exists():
                result["missing_long"].append({"width": width, "arm": arm})
                continue
            s = read(directory / "summary.json")
            assert s["state"] == "completed" and s["completed_steps"] == LONG_STEPS
            short = base / "short" / arm
            ss = read(short / "summary.json")
            for key in (
                "config",
                "initial_parameters_sha256",
                "data_manifest_sha256",
                "data_order_sha256",
                "qualification_sha256",
                "schedule_steps",
            ):
                assert s[key] == ss[key], key
            assert (
                s["resume"]["checkpoint_sha256"]
                == ss["checkpoints"]["checkpoint-976.pt"]
            )
            assert s["resume"]["summary_sha256"] == sha(short / "summary.json")
            assert s["resume"]["curves_sha256"] == sha(short / "curves.jsonl")
            rows = curves(directory / "curves.jsonl")
            assert [r for r in rows if r["step"] <= SHORT_STEPS] == result["curves"][w][
                arm
            ]["short"]
            result["curves"][w][arm]["long"] = rows
            result["long"][w][arm] = describe(s, terminal(rows, LONG_STEPS), kind)
            audit = read(base / f"audit-long-{arm}/audit.json")
            check_audit(s, audit, LONG_STEPS, kind)
            result["checkpoint_audits"] += len(audit["checkpoints"])
            if kind == "cst":
                numerical.extend(oracle_errors(audit["oracles"], 128 * width))
        result["geometry"][w] = []
        for clock in (0, 256, 976, 2441, 4882):
            stage = "short" if clock <= 976 else "long"
            g = read(base / stage / "cst-c64" / f"geometry-{clock}.json")
            total = sum(v["A"] for v in g.values())
            active = sum(v["active_atoms"] for v in g.values())
            result["geometry"][w].append(
                {
                    "step": clock,
                    "radius_mean": sum(v["A"] * v["radius_mean"] for v in g.values())
                    / total,
                    "fraction_mean": sum(
                        v["A"] * v["fraction_mean"] for v in g.values()
                    )
                    / total,
                    "fraction_at_zero": sum(
                        v["A"] * v["fraction_at_zero"] for v in g.values()
                    )
                    / total,
                    "active_coord_moment_below_eps": [
                        sum(
                            v["active_atoms"]
                            * v["active_moment_scale_below_used_epsilon_share"][k]
                            for v in g.values()
                        )
                        / active
                        for k in (2, 3)
                    ]
                    if clock
                    else None,
                }
            )
        for arm in selected:
            if arm not in result["long"][w]:
                continue
            probe = read(base / "long" / arm / "signals-4882.json")
            assert probe["parameters_buffers_rng_unchanged"]
            result["long"][w][arm]["four_fixed_TRAIN_probe_CE_mean"] = statistics.mean(
                v["ce"] for v in probe["records"]
            )
    assert result["missing_long"] == [{"width": 2048, "arm": "dst-lr0.01"}]
    assert len(numerical) == 108 and result["checkpoint_audits"] == 64
    result["numerical"] = {
        "tensor_field_checks": len(numerical),
        "max_abs": max(v["max_abs"] for v in numerical),
        "max_relative_l2": max(v["relative_l2"] for v in numerical),
        "original_component_failures": sum(
            v["original_component_gate_failures"] for v in numerical
        ),
        "gate_changed": False,
    }
    result["data_order_sha256"] = order
    return result


def oracle_errors(oracles, atoms):
    assert set(oracles) == {"1.up", "1.down"}
    errors = []
    for row in oracles.values():
        assert row["passed"] and row["A"] == atoms and row["B"] == 3
        assert set(row["errors"]) == {
            "Y",
            "dX",
            "amplitude",
            "radius_fraction",
            "output_center",
            "input_center",
        }
        for value in row["errors"].values():
            assert value["max_abs"] <= 3e-4 and value["relative_l2"] <= 1e-4
            errors.append(value)
    return errors


def check_audit(summary, audit, steps, kind):
    assert audit["state"] == "passed" and len(audit["checkpoints"]) == 2
    plan = TrainingPlan(steps * 1024, 20000000, 20000000)
    expected = plan.adam_updates(steps) if kind == "dst" else steps
    assert summary["actual_adam_updates"] == expected
    for row in audit["checkpoints"]:
        assert summary["checkpoints"][f"checkpoint-{row['step']}.pt"] == row["sha256"]
        clock = plan.adam_updates(row["step"]) if kind == "dst" else row["step"]
        assert row["actual_adam_updates"] == clock
        if kind != "dense":
            assert set(row["active_slots"].values()) == {128 * summary["width"]}


def describe(summary, metric, kind):
    if kind != "dense":
        assert len(summary["active_slots"]) == 8 and set(
            summary["active_slots"].values()
        ) == {128 * summary["width"]}
    samples = summary["timing_seconds"]
    assert samples and all(math.isfinite(x) and x > 0 for x in samples)
    return {
        "kind": kind,
        "validation": metric,
        "active_ffn_scalars": summary["active_ffn_scalars"],
        "complete_step_median_ms": statistics.median(samples) * 1000,
        "peak_allocated_MiB": summary["peak_allocated_bytes"] / 2**20,
        "peak_reserved_MiB": summary["peak_reserved_bytes"] / 2**20,
        "actual_adam_updates": summary["actual_adam_updates"],
    }


def figures(result, output):
    output.mkdir(parents=True, exist_ok=True)
    assets = {}

    def save(fig, name):
        for ext in ("png", "svg"):
            path = output / f"{name}.{ext}"
            fig.savefig(path, dpi=150)
            assets[path.name] = sha(path)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    gapfig, gaps = plt.subplots(
        1, 2, figsize=(12, 4), sharey=True, constrained_layout=True
    )
    for j, width in enumerate(WIDTHS):
        w = str(width)
        for arm in result["selection"][w]:
            kind = ARMS[arm][0]
            entry = result["curves"][w][arm]
            rows = entry.get("long", entry["short"])
            dev = [r for r in rows if "development" in r and r["step"] >= 200]
            label = kind.upper() + (" (long incomplete)" if "long" not in entry else "")
            for ax, key, scale in zip(
                axes[:, j],
                ("ce", "perplexity", "next_token_top1_accuracy"),
                (1, 1, 100),
            ):
                ax.plot(
                    [r["seen_tokens"] / 1e6 for r in dev],
                    [r["development"][key] * scale for r in dev],
                    color=COLORS[kind],
                    label=label,
                    lw=1.8,
                )
            if kind != "dense":
                dense = result["curves"][w][result["selection"][w][0]]["long"]
                base = {
                    r["step"]: r["development"]["ce"]
                    for r in dense
                    if "development" in r
                }
                ax = gaps[0 if kind == "cst" else 1]
                ax.plot(
                    [r["seen_tokens"] / 1e6 for r in dev],
                    [r["development"]["ce"] - base[r["step"]] for r in dev],
                    marker=".",
                    label=f"Width {width}"
                    + (" (short only)" if "long" not in entry else ""),
                )
        for i, label in enumerate(
            ("Validation CE", "Validation PPL", "Next-token Top1 (%)")
        ):
            ax = axes[i, j]
            ax.set(
                title=f"Width {width}",
                xlabel="Training targets (millions)",
                ylabel=label,
                xlim=(0, 5.05),
            )
            ax.grid(alpha=0.2)
        axes[0, j].legend(fontsize=8)
    fig.suptitle(
        "Stopped M8 FFN pilot: matched Atom/mask counts, seed17; DEV-selected control LRs"
    )
    save(fig, "validation")
    for ax, name in zip(gaps, ("CST", "DST")):
        ax.axhline(0, color="black", lw=1)
        ax.axhline(0.01, color="gray", ls="--", lw=1)
        ax.set(
            title=f"{name}: validation CE minus selected Dense",
            xlabel="Training targets (millions)",
            ylabel="CE gap",
            xlim=(0, 5.05),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save(gapfig, "gaps")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for j, width in enumerate(WIDTHS):
        w = str(width)
        for arm in result["selection"][w]:
            entry = result["curves"][w][arm]
            rows = [
                r
                for r in entry.get("long", entry["short"])
                if "train_ce" in r and r["step"] >= 200
            ]
            kind = ARMS[arm][0]
            axes[j].plot(
                [r["seen_tokens"] / 1e6 for r in rows],
                [r["train_ce"] for r in rows],
                color=COLORS[kind],
                alpha=0.65,
                lw=0.8,
                label=kind.upper(),
            )
        axes[j].set(
            title=f"Width {width}",
            xlabel="Training targets (millions)",
            ylabel="TRAIN minibatch CE (pre-update)",
            xlim=(0, 5.05),
        )
        axes[j].grid(alpha=0.2)
        axes[j].legend(fontsize=8)
    fig.suptitle(
        "Recorded minibatch losses; not full-TRAIN evaluation; no additional scoring"
    )
    save(fig, "training")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for kind in ("dense", "cst", "dst"):
        values = [
            result["short"][str(w)][
                next(a for a in result["selection"][str(w)] if ARMS[a][0] == kind)
            ]
            for w in WIDTHS
        ]
        for ax, key, factor in zip(
            axes,
            ("complete_step_median_ms", "peak_allocated_MiB", "active_ffn_scalars"),
            (1, 1 / 1024, 1 / 1e6),
        ):
            ax.plot(
                WIDTHS,
                [v[key] * factor for v in values],
                marker="o",
                label=kind.upper(),
                color=COLORS[kind],
            )
    for ax, label in zip(
        axes,
        (
            "Complete-step median (ms)",
            "Peak allocated (GiB)",
            "Active FFN scalar slots (millions)",
        ),
    ):
        ax.set(xlabel="Model width", ylabel=label, xticks=WIDTHS)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    fig.suptitle(
        "Matched short-stage complete-step costs; DST uses dense storage; scalar count is not GPU memory"
    )
    save(fig, "resources")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for width in WIDTHS:
        rows = result["geometry"][str(width)]
        for ax, key in zip(axes, ("radius_mean", "fraction_mean", "fraction_at_zero")):
            ax.plot(
                [r["step"] * 1024 / 1e6 for r in rows],
                [r[key] for r in rows],
                marker="o",
                label=f"Width {width}",
            )
    for ax, label in zip(
        axes,
        (
            "Atom-weighted mean radius (grid units)",
            "Atom-weighted mean radius fraction",
            "Fraction exactly at zero",
        ),
    ):
        ax.set(xlabel="Training targets (millions)", ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Saved CST geometry only; pooled layers, not a persistent-Atom trajectory or causal diagnosis"
    )
    save(fig, "geometry")
    return assets


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--stop-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = validate(a.run, a.source_archive, a.stop_receipt)
    result["assets_sha256"] = figures(result, a.output / "assets")
    # Publish curated curve values; raw topology/timing/gradient logs stay external.
    for arms in result["curves"].values():
        for stages in arms.values():
            for stage, rows in stages.items():
                stages[stage] = [
                    {
                        k: row[k]
                        for k in ("step", "seen_tokens", "train_ce", "development")
                        if k in row
                    }
                    for row in rows
                ]
    result["postprocessor_sha256"] = sha(Path(__file__))
    (a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "payloads_verified",
                    "qualification_arms",
                    "checkpoint_audits",
                    "numerical",
                    "missing_long",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
