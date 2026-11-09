"""Render the completed, source-qualified FFN placement and signal studies."""

import argparse
import ast
import hashlib
import json
import tarfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(root):
    state = read(root / "backup-state.json")
    assert state["state"] == "completed"
    last = state["snapshots"][-1]
    assert last["state"] == "passed"
    assert sha(root / last["name"]) == last["sha256"]
    data = root / last["retrieved"]
    manifest = read(data / "manifest.json")
    assert sha(root / "source.tar.gz") == manifest["source_archive_sha256"]
    with tarfile.open(root / "source.tar.gz") as archive:
        source = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    assert manifest["source_files"] == {
        name: hashlib.sha256(value).hexdigest() for name, value in source.items()
    }
    assert all(sha(data / n) == h for n, h in manifest["files"].items())
    receipt = read(data / "receipt.json")
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    return data, receipt, source, last


def load_cohorts(base_root, placement_root):
    base, br, bs, bi = verified(base_root)
    placement, pr, ps, pi = verified(placement_root)
    assert len(br["results"]) == 11 and len(pr["results"]) == 6
    assert all(br[k] == pr[k] for k in ("gpu", "torch", "cuda"))
    assert br["protocol"]["source_commits"] == {
        "torchcst": pr["protocol"]["source_commits"]["torchcst"],
        "cst-research": "7b89c7db7105b4212fa9aee3efbc2870f1b848e3",
    }
    assert {n: v for n, v in bs.items() if n.startswith("torchcst/")} == {
        n: v for n, v in ps.items() if n.startswith("torchcst/")
    }
    prefix = "cst-research/src/cst_research/experiments/"
    for name in ("ffn_contract.py", "ffn_model.py", "ffn_dst.py", "ffn_run.py"):
        assert bs[prefix + name] == ps[prefix + name], name
    for name in ("ffn_selective.py", "ffn_selective_run.py"):
        functions = []
        for source in (bs, ps):
            functions.append(
                {
                    node.name: ast.dump(node)
                    for node in ast.parse(source[prefix + name]).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name not in ("main", "qualify")
                }
            )
        assert functions[0] == functions[1], name
    dependency = read(placement / "original-controller-receipt.json")["dependency"]
    assert (
        dependency["parent_source_archive_sha256"]
        == br["protocol"]["source_archive_sha256"]
    )
    assert all(
        dependency[k]
        for k in (
            "all_operator_source_bytes_match",
            "training_and_probe_function_asts_match",
            "source_core_bytes_match",
        )
    )
    summaries, curves, mechanisms, audits = {}, {}, {}, {}
    for directory, receipt in ((base, br), (placement, pr)):
        for result in receipt["results"]:
            arm = result["arm"]
            assert arm not in summaries
            s = read(directory / arm / "summary.json")
            audit = read(directory / ("audit-" + arm) / "audit.json")
            assert s["state"] == "completed_execution_not_scientific_admission"
            assert s["completed_updates"] == 4882
            assert audit["state"] == "passed" and len(audit["models"]) == 3
            assert {v["checkpoint_sha256"] for v in audit["models"]} == set(
                s["milestone_checkpoint_sha256"].values()
            )
            assert result["development"] == s["latest_development"]
            summaries[arm], audits[arm] = s, audit
            curves[arm] = lines(directory / arm / "curves.jsonl")
            mechanisms[arm] = lines(directory / arm / "mechanisms.jsonl")
    dense = summaries["dense"]
    for s in summaries.values():
        for key in (
            "config",
            "training_plan",
            "seed",
            "data_manifest_sha256",
            "data_order_sha256",
            "completed_tokens",
        ):
            assert s[key] == dense[key], key
    for key in (
        "up",
        "down",
        "block",
        "two_blocks",
        "all",
        "scattered_up",
        "scattered_down",
        "scattered_mixed",
    ):
        assert (
            summaries["cst-" + key]["active_ffn_scalars"]
            == summaries["dst-" + key]["active_ffn_scalars"]
        )
    assert (
        summaries["cst-two_blocks"]["active_ffn_scalars"]
        == summaries["cst-scattered_mixed"]["active_ffn_scalars"]
    )
    provenance = {
        "base": {
            "job_id": base_root.name,
            "source": br["protocol"]["source_commits"],
            "source_archive_sha256": br["protocol"]["source_archive_sha256"],
            "compact_sha256": bi["sha256"],
        },
        "placement": {
            "job_id": placement_root.name,
            "source": pr["protocol"]["source_commits"],
            "source_archive_sha256": pr["protocol"]["source_archive_sha256"],
            "compact_sha256": pi["sha256"],
            "recovery": pr["recovery"],
        },
    }
    return summaries, curves, mechanisms, audits, provenance


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base-run", type=Path, required=True)
    p.add_argument("--placement-run", type=Path, required=True)
    p.add_argument("--signals-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summaries, curves, _mechanisms, audits, provenance = load_cohorts(
        args.base_run, args.placement_run
    )
    data, receipt, _, snapshot = verified(args.signals_run)
    observed = lines(data / "results/signals.jsonl")
    audit = read(data / "results/summary.json")
    assert audit["state"] == "passed" and len(audit["models"]) == 51
    assert len(observed) == receipt["probe_records"] == 102
    assert set(audit["sources"]) == {
        v["source_archive_sha256"] for v in provenance.values()
    }
    assert all(
        v["parameter_and_rng_unchanged"]
        and v["original_probe_ce_and_ffn_gradient_rms_reproduced"]
        for v in audit["models"]
    )
    signals = {(v["arm"], v["step"], v["probe_scope"]): v for v in observed}
    assert len(signals) == 102 and {v["arm"] for v in observed} == set(summaries)
    scopes = (
        "original128",
        "additional128_offset4096",
        "additional128_offset8192",
        "additional128_offset12288",
    )
    for arm, s in summaries.items():
        for step in (0, 976, 4882):
            for scope in scopes if step == 4882 else scopes[:1]:
                row = signals[arm, step, scope]
                assert row["tokens"] == 128 and row["sites"] == s["sites"]
                assert row["checkpoint_sha256"] == (
                    s["milestone_checkpoint_sha256"][
                        f"checkpoint-{step * 1024}-tokens.pt"
                    ]
                    if step
                    else None
                )
    provenance["signals"] = {
        "job_id": args.signals_run.name,
        "source": receipt["protocol"]["source_commits"],
        "source_archive_sha256": receipt["protocol"]["source_archive_sha256"],
        "evaluator_sha256": audit["evaluator_sha256"],
        "compact_sha256": snapshot["sha256"],
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    keys = ("two_blocks", "scattered_up", "scattered_down", "scattered_mixed", "all")
    names = {
        "two_blocks": "4 maps: paired blocks 1+2",
        "scattered_up": "4 maps: up in every block",
        "scattered_down": "4 maps: down in every block",
        "scattered_mixed": "4 maps: alternating up/down",
        "all": "8 maps: both in every block",
    }
    colors = {"dense": "#222222", "cst": "#1f77b4", "dst": "#2ca02c"}
    for metric, label in (
        ("ce", "DEV cross entropy"),
        ("ce_late", "DEV cross entropy, after 2M tokens"),
        ("next_token_top1_accuracy", "DEV next-token top1 (%)"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, key in zip(axes.flat, keys):
            for kind, arm in (
                ("dense", "dense"),
                ("cst", "cst-" + key),
                ("dst", "dst-" + key),
            ):
                pts = [
                    v
                    for v in curves[arm]
                    if "development" in v
                    and (metric != "ce_late" or v["seen_tokens"] >= 2_000_000)
                ]
                field = "ce" if metric == "ce_late" else metric
                ax.plot(
                    [v["seen_tokens"] / 1e6 for v in pts],
                    [
                        v["development"][field] * (100 if field != "ce" else 1)
                        for v in pts
                    ],
                    color=colors[kind],
                    label=kind.upper(),
                )
            ax.set(title=names[key], xlabel="Training tokens (millions)", ylabel=label)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        axes.flat[-1].axis("off")
        fig.suptitle(
            "Same task/data/seed/budget; two source-qualified cohorts; one DEV seed"
        )
        save(fig, "placement-" + metric)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for kind, offset in (("cst", -0.15), ("dst", 0.15)):
        ax.bar(
            np.arange(len(keys)) + offset,
            [
                summaries[kind + "-" + k]["latest_development"]["ce"]
                - summaries["dense"]["latest_development"]["ce"]
                for k in keys
            ],
            width=0.3,
            label=kind.upper(),
            color=colors[kind],
        )
    ax.set_xticks(
        np.arange(len(keys)), [names[k] for k in keys], rotation=15, ha="right"
    )
    ax.set_ylabel("Final DEV CE minus Dense")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.suptitle(
        "Placement matters in this seed; positions and local pairing change together"
    )
    save(fig, "placement-final-gaps")

    arms = [
        "dense",
        *[
            f"{kind}-{key}"
            for key in ("up", "down", "block", *keys)
            for kind in ("cst", "dst")
        ],
    ]
    layer_names = [
        f"blocks.{i}.{suffix}" for i in range(4) for suffix in ("ffn.up", "ffn.down")
    ]
    layer_names += [f"blocks.{i}" for i in range(4)] + ["final_norm"]

    def values(arm, name, group, field, step=4882, relative=False):
        selected = scopes if step == 4882 else scopes[:1]
        result = []
        for scope in selected:
            value = signals[arm, step, scope]["layers"][name][group]
            baseline = signals["dense", step, scope]["layers"][name][group]
            for k in field.split("."):
                value, baseline = value[k], baseline[k]
            result.append(value / max(baseline, 1e-30) if relative else value)
        return result

    for group, field, title, stem in (
        ("output", "rms", "Output RMS", "output"),
        ("output_gradient", "rms", "Output-cotangent RMS", "gradient"),
        (
            "output",
            "sample_feature_spectrum.participation_rank",
            "Centered sample covariance participation rank",
            "participation",
        ),
    ):
        matrix = [
            [
                np.mean(values(arm, layer, group, field, relative=True))
                for layer in layer_names
            ]
            for arm in arms
        ]
        fig, ax = plt.subplots(figsize=(15, 8), layout="constrained")
        im = ax.imshow(
            np.log2(np.maximum(matrix, 1e-30)),
            cmap="coolwarm",
            vmin=-2,
            vmax=2,
            aspect="auto",
        )
        ax.set_yticks(range(len(arms)), arms)
        ax.set_xticks(range(len(layer_names)), layer_names, rotation=40, ha="right")
        fig.colorbar(
            im, ax=ax, label="log2 mean per-context ratio / Dense; color clipped +/-2"
        )
        fig.suptitle(
            title + "; final step; four fixed TRAIN contexts (128 targets each)"
        )
        save(fig, "signals-" + stem)

    focus = (
        "cst-two_blocks",
        "cst-scattered_up",
        "cst-scattered_down",
        "cst-scattered_mixed",
        "cst-all",
        "dst-all",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    block_metrics = {}
    for arm in focus:
        block_metrics[arm] = {}
        for column, metric in enumerate(("output", "output_gradient", "rms_product")):
            means, low, high = [], [], []
            for i in range(4):
                name = f"blocks.{i}"
                x = np.array(values(arm, name, "output", "rms", relative=True))
                g = np.array(values(arm, name, "output_gradient", "rms", relative=True))
                v = x * g if metric == "rms_product" else x if metric == "output" else g
                means.append(float(v.mean()))
                low.append(float(v.min()))
                high.append(float(v.max()))
            block_metrics[arm][metric] = {
                "mean_context_ratio": means,
                "min_context_ratio": low,
                "max_context_ratio": high,
            }
            axes[column].plot(range(4), means, marker="o", label=arm)
            axes[column].fill_between(range(4), low, high, alpha=0.1)
    for ax, label in zip(
        axes,
        (
            "Residual output RMS / Dense",
            "Residual output cotangent RMS / Dense",
            "RMS(output) x RMS(cotangent) / Dense",
        ),
    ):
        ax.set(xlabel="Block (zero-based)", ylabel=label, xticks=range(4))
        ax.axhline(1, color="black", linestyle="--", linewidth=0.7)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Final signals; bands are context range, not confidence intervals; RMS product is not directional alignment"
    )
    save(fig, "signals-block-scale")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, arm in zip(axes.flat, focus):
        for step, style in ((0, ":"), (976, "--"), (4882, "-")):
            x = [
                np.mean(values(arm, f"blocks.{i}", "output", "rms", step, True))
                for i in range(4)
            ]
            ax.plot(range(4), x, linestyle=style, marker="o", label=f"step {step}")
        ax.set(
            title=arm,
            xlabel="Block",
            ylabel="Output RMS / same-state Dense",
            xticks=range(4),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Signal scale evolves during task training; initial/intermediate: one TRAIN context, final: mean of four"
    )
    save(fig, "signals-scale-evolution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for arm in ("dense", *focus):
        entropy = [
            np.mean(
                [
                    signals[arm, 4882, s]["attention"][f"blocks.{i}.qkv"][
                        "entropy_over_log_available_keys"
                    ]
                    for s in scopes
                ]
            )
            for i in range(4)
        ]
        zero = [
            100
            * np.mean(
                [
                    signals[arm, 4882, s]["layers"][f"blocks.{i}.ffn.down"]["input"][
                        "near_zero_relative_fraction"
                    ]
                    for s in scopes
                ]
            )
            for i in range(4)
        ]
        axes[0].plot(range(4), entropy, marker="o", label=arm)
        axes[1].plot(range(4), zero, marker="o", label=arm)
    axes[0].set_ylabel("Mean entropy / log(available keys)")
    axes[1].set_ylabel("GELU input-to-down: |x| <= 0.001 RMS (%)")
    for ax in axes:
        ax.set(xlabel="Block", xticks=range(4))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Four fixed TRAIN contexts; post-hoc descriptions, not dead-unit or causal tests"
    )
    save(fig, "signals-attention-and-gelu")

    result = {
        "plotter_sha256": sha(Path(__file__)),
        "scope": "17 models, one DEV seed; task training only; 51 read-only model states and102 TRAIN probes; descriptive post-hoc analysis, no causal claim",
        "provenance": provenance,
        "data_manifest_sha256": summaries["dense"]["data_manifest_sha256"],
        "data_order_sha256": summaries["dense"]["data_order_sha256"],
        "conditions": {
            arm: {
                k: s[k]
                for k in (
                    "kind",
                    "sites",
                    "active_ffn_scalars",
                    "latest_development",
                    "completed_tokens",
                    "completed_updates",
                    "milestone_checkpoint_sha256",
                    "setup_seconds",
                    "total_elapsed_seconds",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
            }
            | {"median_step_ms": float(np.median(s["timing_samples_seconds"]) * 1000)}
            for arm, s in summaries.items()
        },
        "trained_checkpoint_audits": sum(len(v["models"]) for v in audits.values()),
        "block_signals": block_metrics,
        "signal_statistics": {
            arm: {
                layer: {
                    group + "." + field: {
                        "mean_context_ratio": float(
                            np.mean(values(arm, layer, group, field, relative=True))
                        ),
                        "context_ratios": values(
                            arm, layer, group, field, relative=True
                        ),
                    }
                    for group, field in (
                        ("output", "rms"),
                        ("output_gradient", "rms"),
                        ("output", "sample_feature_spectrum.participation_rank"),
                    )
                }
                for layer in layer_names
            }
            for arm in arms
        },
        "figures": {path.name: sha(path) for path in sorted(out.glob("*.png"))},
    }
    (out / "composition-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "arms": len(summaries),
                "trained_checkpoint_audits": result["trained_checkpoint_audits"],
                "signal_probes": len(observed),
                "output": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
