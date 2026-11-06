"""Aggregate hash-linked FFN observations; this does not rerun GPU validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from .ffn_contract import ModelConfig, learning_rate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, positive=False):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (not positive or value > 0),
        "invalid " + label,
    )
    return value


def located(root, relative):
    require(isinstance(relative, str), "relative evidence path required")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe evidence path")
    result = (root / path).resolve()
    require(result.is_relative_to(root.resolve()), "evidence path escapes root")
    return result


def checked_json(root, relative, expected):
    path = located(root, relative)
    require(digest(path) == expected, "changed evidence: " + relative)
    return json.loads(path.read_text())


def metric(value, tokens):
    ce = finite(value["ce"], "CE", positive=True)
    require(value["tokens"] == tokens, "different evaluation token count")
    require(
        math.isclose(value["perplexity"], math.exp(ce), rel_tol=1e-10),
        "CE/perplexity disagreement",
    )
    return ce


def read_run(root, request, run):
    receipt = checked_json(root, run["receipt"], run["receipt_sha256"])
    audit = checked_json(root, run["audit"], run["audit_sha256"])
    require(
        receipt["state"] == "finished_execution_pending_independent_audit"
        and all(c.get("exit_code") == 0 for c in receipt["commands"]),
        "run sequence is not terminal successful",
    )
    require(
        audit["state"] == "passed" and audit["receipt_sha256"] == run["receipt_sha256"],
        "audit is not passed or refers to another receipt",
    )
    for key in ("consumer", "operator"):
        require(
            receipt[key] == audit[key] == request[key + "_commit"],
            "different " + key + " source",
        )
    matches = [c for c in receipt["commands"] if c["label"] == run["label"]]
    require(len(matches) == 1, "missing or ambiguous run label")
    command = matches[0]
    artifacts = located(root, run["artifacts"])
    argv = command["argv"]
    require(
        Path(argv[argv.index("--output") + 1]).name == artifacts.name,
        "artifacts do not belong to the recorded command",
    )
    for name, key in (
        ("summary.json", "summary_sha256"),
        ("curves.jsonl", "curves_sha256"),
    ):
        require(digest(artifacts / name) == command[key], "changed " + name)
    summary = json.loads((artifacts / "summary.json").read_text())
    require(
        summary["state"] == "completed_execution_not_scientific_admission"
        and not summary["arguments"]["repeat_fixed_batches"],
        "partial execution or fixed-batch smoke cannot enter learning reports",
    )
    config = ModelConfig(**summary["config"])
    for key in ("width", "expansion"):
        require(getattr(config, key) == run[key], "different " + key)
    require(config.recipe() == request["recipe"], "different selected recipe")
    require(
        (
            config.layers,
            config.heads,
            config.context,
            config.vocabulary,
            config.init_std,
        )
        == (4, 4, 128, 50257, 0.02),
        "different model protocol",
    )
    args = summary["arguments"]
    for key in ("kind", "seed"):
        require(args[key] == run[key], "different " + key)
    updates = request["token_budget"] // 1024
    require(updates > 10, "too few updates for timed learning observations")
    require(
        summary["completed_updates"] == summary["scheduled_updates"] == updates
        and summary["completed_tokens"] == summary["scheduled_tokens"] == updates * 1024
        and args["tokens"] == request["token_budget"],
        "different or incomplete training budget",
    )
    require(
        summary["data_manifest_sha256"] == request["data_manifest_sha256"]
        and summary["data_order_horizon_tokens"] == 20000000
        and summary["learning_rate_horizon_updates"] == 20000000 // 1024
        and args["batch"] == 2
        and args["accumulation"] == 4,
        "different data, order, LR horizon or batch protocol",
    )
    require(not args["track_atom_motion"], "instrumented run cannot enter this report")
    require(
        summary["gpu"] == request["gpu"] and summary["torch"] == request["torch"],
        "different GPU/runtime",
    )
    audit_arm = audit["arms"][run["label"]]
    require(
        audit_arm["checkpoint_sha256"]
        == command["checkpoint_sha256"]
        == summary["checkpoint_sha256"]
        and audit_arm["updates"] == updates
        and audit_arm["finite_optimizer_state"],
        "different checkpoint or incomplete optimizer audit",
    )
    projections = 8 if run["kind"] == "cst" else 0
    require(
        len(audit_arm["trained_all_atom_vjp"]) == projections,
        "incomplete trained VJP audit",
    )
    for checks in audit_arm["trained_all_atom_vjp"].values():
        require(set(checks) == {"y", "dx", "dp"}, "incomplete projection VJP")
    resume = audit_arm["disposable_task_resume"]
    require(resume["optimizer_moments_counters_passed"], "optimizer resume not checked")
    for phase in ("first", "next", "resumed_next"):
        require(
            resume[phase]["atom_sites_with_nonzero_gradient_and_change"] == projections,
            "incomplete full-model Atom update check",
        )
    dev, confirmation = (
        metric(summary["latest_development"], 8192),
        metric(summary["confirmation"], 32768),
    )
    for key, value in (("development", dev), ("confirmation", confirmation)):
        require(
            abs(audit_arm["recomputed_ce"][key] - value) < 3e-5,
            "recomputed CE disagrees",
        )
    curves = [
        json.loads(line)
        for line in (artifacts / "curves.jsonl").read_text().splitlines()
    ]
    previous, previous_time, validation, train_steps = -1, -1.0, [], []
    for point in curves:
        step = point["step"]
        require(
            previous <= step <= updates and point["seen_tokens"] == 1024 * step,
            "invalid curve grid",
        )
        previous = step
        elapsed = finite(
            point["elapsed_seconds"], "recorded elapsed time", positive=True
        )
        require(elapsed >= previous_time, "elapsed clock moved backwards")
        previous_time = elapsed
        if "train_ce" in point:
            train_steps.append(step)
            finite(point["train_ce"], "training CE", positive=True)
            lr = learning_rate(step, 20000000 // 1024)
            expected = (
                [lr, lr * config.atom_lr_multiplier]
                if projections and config.atom_lr_multiplier != 1
                else [lr]
            )
            require(
                abs(point["lr"] - lr) < 1e-12
                and len(point["optimizer_group_lrs"]) == len(expected)
                and all(
                    abs(a - b) < 1e-12
                    for a, b in zip(point["optimizer_group_lrs"], expected)
                ),
                "optimizer group rates differ from selected protocol",
            )
        if "development" in point:
            validation.append([step, metric(point["development"], 8192)])
    require(
        validation and validation[0][0] == 0 and validation[-1] == [updates, dev],
        "incomplete validation curve",
    )
    milestones = {n // 1024 for n in (1000000, 5000000, 10000000, 20000000)}
    require(
        [p[0] for p in validation]
        == [0]
        + [
            s
            for s in range(1, updates + 1)
            if s % 200 == 0 or s == updates or s in milestones
        ],
        "missing or extra validation checkpoint",
    )
    require(
        train_steps
        == [s for s in range(1, updates + 1) if s <= 5 or s % 20 == 0 or s == updates],
        "missing or extra training curve point",
    )
    timing = summary["timing_samples_seconds"]
    require(len(timing) == updates - 10, "incomplete complete-update timings")
    for value in timing:
        finite(value, "update time", positive=True)
    setup = finite(summary["setup_seconds"], "setup time", positive=True)
    total = finite(
        summary["total_elapsed_seconds"], "total recorded elapsed time", positive=True
    )
    require(
        total >= previous_time and total >= setup and total >= sum(timing),
        "elapsed budget disagrees with measurements",
    )
    allocated, reserved = (
        summary["peak_allocated_bytes"],
        summary["peak_reserved_bytes"],
    )
    require(
        type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved,
        "invalid memory peaks",
    )
    budget = config.ffn_parameter_budget()
    parameters = 0 if run["kind"] == "none" else budget[run["kind"] + "_parameters"]
    require(
        summary["actual_ffn_parameters"] == parameters,
        "FFN parameter accounting differs",
    )
    prompts = (
        "Once upon a time, there was a little",
        "Lily found a red ball in the garden.",
    )
    require(len(summary["generations"]) == 2, "missing fixed-prompt generation")
    for i, generation in enumerate(summary["generations"]):
        require(
            generation["prompt"] == prompts[i]
            and generation["sampling_seed"] == 7000 + i
            and generation["temperature"] == 0.8
            and generation["top_k"] == 40
            and generation["text"].startswith(prompts[i]),
            "different generation protocol",
        )
    return {
        **{k: run[k] for k in ("width", "expansion", "seed", "kind")},
        "development_ce": dev,
        "confirmation_ce": confirmation,
        "development_perplexity": math.exp(dev),
        "confirmation_perplexity": math.exp(confirmation),
        "median_update_ms": statistics.median(timing) * 1000,
        "setup_seconds": setup,
        "total_elapsed_seconds": total,
        "peak_allocated_mib": allocated / 1048576,
        "peak_reserved_mib": reserved / 1048576,
        "ffn_parameters": parameters,
        "atoms_per_projection": config.atoms_per_projection if projections else 0,
        "validation": validation,
        "curves": curves,
        "generations": summary["generations"],
        "receipt_sha256": run["receipt_sha256"],
        "audit_sha256": run["audit_sha256"],
        "summary_sha256": command["summary_sha256"],
        "curves_sha256": command["curves_sha256"],
        "checkpoint_sha256": command["checkpoint_sha256"],
    }


def aggregate(root, request):
    require(request["cohort"] in ("development", "confirmation"), "unknown cohort")
    seeds = request["seeds"]
    require(all(type(s) is int for s in seeds), "integer seeds required")
    require(
        type(request["token_budget"]) is int
        and 1024 <= request["token_budget"] <= 20000000,
        "invalid token budget",
    )
    for shape in request["shapes"]:
        require(
            type(shape["width"]) is int and type(shape["expansion"]) is int,
            "integer shapes required",
        )
    for key, length in (
        ("consumer_commit", 40),
        ("operator_commit", 40),
        ("data_manifest_sha256", 64),
    ):
        require(
            re.fullmatch("[0-9a-f]{" + str(length) + "}", request[key]) is not None,
            "invalid " + key,
        )
    require(
        seeds == [17]
        if request["cohort"] == "development"
        else sorted(seeds) == [29, 43, 71],
        "cohort seed family differs",
    )
    expected = {
        (s["width"], s["expansion"], seed, kind)
        for s in request["shapes"]
        for seed in seeds
        for kind in ("dense", "none", "cst")
    }
    runs = [read_run(root, request, run) for run in request["runs"]]
    identities = [(r["width"], r["expansion"], r["seed"], r["kind"]) for r in runs]
    require(
        len(set(identities)) == len(identities) and set(identities) == expected,
        "missing, duplicate or extra comparison arm",
    )
    require(
        len(expected) == len(request["shapes"]) * len(seeds) * 3,
        "duplicate or empty shapes",
    )
    require(runs, "empty comparison family")
    require(
        all(
            [p[0] for p in r["validation"]] == [p[0] for p in runs[0]["validation"]]
            for r in runs
        ),
        "different validation grids",
    )
    groups = defaultdict(list)
    for run in runs:
        groups[(run["width"], run["expansion"], run["kind"])].append(run)
    statistics_rows = []
    for (width, expansion, kind), members in sorted(groups.items()):
        grid = [p[0] for p in members[0]["validation"]]
        require(
            all([p[0] for p in m["validation"]] == grid for m in members),
            "different validation grids",
        )
        values = {
            key: [m[key] for m in members]
            for key in (
                "development_ce",
                "confirmation_ce",
                "development_perplexity",
                "confirmation_perplexity",
                "median_update_ms",
                "peak_allocated_mib",
                "peak_reserved_mib",
            )
        }
        statistics_rows.append(
            {
                "width": width,
                "expansion": expansion,
                "kind": kind,
                "seeds": sorted(m["seed"] for m in members),
                "ffn_parameters": members[0]["ffn_parameters"],
                "atoms_per_projection": members[0]["atoms_per_projection"],
                "metrics": {
                    k: {
                        "mean": statistics.mean(v),
                        "sample_sd": statistics.stdev(v) if len(v) > 1 else None,
                    }
                    for k, v in values.items()
                },
            }
        )
    paired = []
    by_id = {identity: run for identity, run in zip(identities, runs)}
    for shape in request["shapes"]:
        width, expansion = shape["width"], shape["expansion"]
        for control in ("dense", "none"):
            for key in ("development_ce", "confirmation_ce"):
                differences = {
                    str(seed): by_id[(width, expansion, seed, "cst")][key]
                    - by_id[(width, expansion, seed, control)][key]
                    for seed in seeds
                }
                paired.append(
                    {
                        "width": width,
                        "expansion": expansion,
                        "control": control,
                        "metric": key,
                        "per_seed": differences,
                        "mean": statistics.mean(differences.values()),
                        "sample_sd": statistics.stdev(differences.values())
                        if len(seeds) > 1
                        else None,
                    }
                )
    return {
        "state": "aggregated_receipts_not_gpu_revalidation",
        "cohort": request["cohort"],
        "protocol": {k: v for k, v in request.items() if k != "runs"},
        "runs": runs,
        "groups": statistics_rows,
        "paired_differences": paired,
    }


def write_markdown(result, output):
    """Equivalent English/Japanese tables; SD is not a confidence interval."""
    labels = {"dense": "Dense", "cst": "CST", "none": "No FFN"}
    for japanese in (False, True):
        filename = "report.ja.md" if japanese else "report.md"
        intro = (
            "# FFNの観測集計\n\n[English](report.md) | 日本語\n\n"
            "hashを照合したreceiptの集計です。GPU検証を再実行したものではありません。"
            "平均±seed間の標本SDを示し、1 seedのSDは省略します。信頼区間・非劣性判定・"
            "一般的なスケーリング則ではありません。\n\n"
            if japanese
            else "# FFN observation summary\n\nEnglish | [日本語](report.ja.md)\n\n"
            "This aggregates hash-linked receipts; it does not rerun GPU validation. "
            "Values are mean ± sample SD across seeds, with SD omitted for one seed. "
            "These are not confidence intervals, noninferiority decisions or a general scaling law.\n\n"
        )
        intro += f"Cohort: {result['cohort']}; seeds: {result['protocol']['seeds']}; requested tokens/arm: {result['protocol']['token_budget']}.\n\n"
        text = intro
        for columns, headers, parameters in (
            (
                (
                    "development_ce",
                    "development_perplexity",
                    "confirmation_ce",
                    "confirmation_perplexity",
                ),
                (
                    "DEV CE",
                    "DEV perplexity",
                    "Confirmation CE",
                    "Confirmation perplexity",
                ),
                False,
            ),
            (
                ("median_update_ms", "peak_allocated_mib", "peak_reserved_mib"),
                ("Update ms", "Allocated MiB", "Reserved MiB"),
                True,
            ),
        ):
            table = (
                "| Width | Expansion | FFN | "
                + " | ".join(headers)
                + (" | FFN parameters" if parameters else "")
                + " |\n"
            )
            table += (
                "| ---: | ---: | --- | "
                + " | ".join("---:" for _ in range(len(headers) + int(parameters)))
                + " |\n"
            )
            for group in result["groups"]:
                values = []
                for key in columns:
                    value = group["metrics"][key]
                    values.append(
                        f"{value['mean']:.5f}"
                        + (
                            f" ± {value['sample_sd']:.5f}"
                            if value["sample_sd"] is not None
                            else ""
                        )
                    )
                table += (
                    f"| {group['width']} | {group['expansion']} | {labels[group['kind']]} | "
                    + " | ".join(values)
                    + (f" | {group['ffn_parameters']}" if parameters else "")
                    + " |\n"
                )
            text += table + "\n"
        text += (
            "Perplexityは各seedで計算してから集計します。\n\n"
            if japanese
            else "Perplexity is computed for each seed before aggregation.\n\n"
        )
        for stem in (
            "development-curves",
            "training-curves",
            "development-versus-time",
            "resource-observations",
        ):
            if (output / (stem + ".png")).exists():
                text += f"![{stem}]({stem}.png)\n\n"
        text += (
            "## seedを揃えたCE差（CST−対照）\n\n"
            if japanese
            else "## Paired CE differences (CST minus control)\n\n"
        )
        text += "| Width | Expansion | Control | Metric | Mean difference | Sample SD |\n| ---: | ---: | --- | --- | ---: | ---: |\n"
        for row in result["paired_differences"]:
            sd = "—" if row["sample_sd"] is None else f"{row['sample_sd']:.5f}"
            text += f"| {row['width']} | {row['expansion']} | {labels[row['control']]} | {row['metric']} | {row['mean']:.5f} | {sd} |\n"
        text += (
            "\n## 固定promptの例\n\n最初のseed・最初の固定promptを各構成で示します。良い例の選別ではなく、独立した言語品質評価でもありません。全prompt・seedの出力はreport.jsonに保持します。\n\n"
            if japanese
            else "\n## Fixed-prompt examples\n\nThe first seed and first predefined prompt are shown for every architecture/arm, without selecting the best generation. This is not an independent language-quality evaluation. All prompts/seeds remain in report.json.\n\n"
        )
        for run in sorted(
            result["runs"],
            key=lambda r: (r["width"], r["expansion"], r["seed"], r["kind"]),
        ):
            if run["seed"] != min(result["protocol"]["seeds"]):
                continue
            sample = run["generations"][0]["text"].rstrip()
            fence = "`" * max(
                3, 1 + max((len(s) for s in re.findall("`+", sample)), default=0)
            )
            text += f"### N{run['width']}, expansion{run['expansion']}, {labels[run['kind']]}\n\n{fence}text\n{sample}\n{fence}\n\n"
        text += "## 来歴\n\n" if japanese else "## Provenance\n\n"
        for key in (
            "consumer_commit",
            "operator_commit",
            "data_manifest_sha256",
            "recipe",
        ):
            text += (
                f"- {key}: `{json.dumps(result['protocol'][key], sort_keys=True)}`\n"
            )
        text += f"- request_sha256: `{result['request_sha256']}`\n"
        (output / filename).write_text(text)


def render_plots(result, output):
    """Scientific figures, with all seed trajectories and explicit SD bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result["plotting_runtime"] = {
        "matplotlib": matplotlib.__version__,
        "python": sys.version.split()[0],
    }
    matplotlib.rcParams["svg.hashsalt"] = result.get("request_sha256", "ffn-report")
    protocol = result["protocol"]
    identity = f"{protocol['cohort']}; seeds {protocol['seeds']}; c{protocol['recipe']['coefficient']}\n{protocol['gpu']}"

    shapes = result["protocol"]["shapes"]
    colors = {"dense": "#2463a6", "cst": "#d44d64", "none": "#777777"}
    labels = {"dense": "Dense", "cst": "CST", "none": "No FFN"}
    for metric_name, stem in (
        ("development", "development-curves"),
        ("train", "training-curves"),
        ("elapsed", "development-versus-time"),
    ):
        cols = min(3, len(shapes))
        rows = math.ceil(len(shapes) / cols)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(5 * cols, 4.3 * rows),
            squeeze=False,
            layout="constrained",
        )
        for ax, shape in zip(axes.flat, shapes):
            for kind in ("dense", "cst", "none"):
                members = [
                    r
                    for r in result["runs"]
                    if (r["width"], r["expansion"], r["kind"])
                    == (shape["width"], shape["expansion"], kind)
                ]
                if metric_name == "elapsed":
                    for i, run in enumerate(members):
                        points = [
                            p
                            for p in run["curves"]
                            if "development" in p and p["step"] > 0
                        ]
                        ax.plot(
                            [p["elapsed_seconds"] for p in points],
                            [p["development"]["ce"] for p in points],
                            color=colors[kind],
                            alpha=0.7,
                            lw=1.3,
                            label=labels[kind] if i == 0 else None,
                        )
                    continue
                curves = [
                    [
                        (
                            p["step"] * 1024 / 1e6,
                            p["development"]["ce"]
                            if metric_name == "development"
                            else p["train_ce"],
                        )
                        for p in r["curves"]
                        if (
                            "development"
                            if metric_name == "development"
                            else "train_ce"
                        )
                        in p
                        and p["step"] > 0
                    ]
                    for r in members
                ]
                require(
                    curves
                    and all(
                        [x for x, _ in c] == [x for x, _ in curves[0]] for c in curves
                    ),
                    "different plotted grids",
                )
                x = [x for x, _ in curves[0]]
                mean = [statistics.mean(c[i][1] for c in curves) for i in range(len(x))]
                for curve in curves:
                    ax.plot(
                        [x for x, _ in curve],
                        [y for _, y in curve],
                        color=colors[kind],
                        alpha=0.22,
                        lw=0.8,
                    )
                ax.plot(x, mean, color=colors[kind], label=labels[kind], lw=1.6)
                if len(curves) > 1:
                    sd = [
                        statistics.stdev(c[i][1] for c in curves) for i in range(len(x))
                    ]
                    ax.fill_between(
                        x,
                        [m - s for m, s in zip(mean, sd)],
                        [m + s for m, s in zip(mean, sd)],
                        color=colors[kind],
                        alpha=0.10,
                    )
            ax.set(
                title=f"N{shape['width']} / expansion{shape['expansion']}",
                xlabel="Recorded elapsed seconds"
                if metric_name == "elapsed"
                else "Seen tokens (millions)",
                ylabel="Cross-entropy (lower is better)",
            )
            if metric_name == "elapsed":
                ax.set_xscale("log")
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in list(axes.flat)[len(shapes) :]:
            ax.set_visible(False)
        axes.flat[0].legend(frameon=False)
        fig.suptitle(
            (
                "FFN validation versus time: individual seeds"
                if metric_name == "elapsed"
                else "FFN " + metric_name + " loss: all seeds and mean"
            )
            + "\n"
            + identity,
            fontsize=12,
        )
        fig.supxlabel(
            "Recorded setup/evaluation/probes included; qualification excluded.\nIndividual trajectories, without interpolated mean or SD bands."
            if metric_name == "elapsed"
            else "Bands: ±1 sample SD (n>1), not confidence intervals.\nCommon20M-token LR/data horizon.",
            fontsize=9,
        )
        for ext in ("png", "svg"):
            fig.savefig(output / (stem + "." + ext), dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    groups = result["groups"]
    for kind in ("dense", "cst", "none"):
        for expansion in sorted({g["expansion"] for g in groups}):
            selected = sorted(
                [
                    g
                    for g in groups
                    if g["kind"] == kind and g["expansion"] == expansion
                ],
                key=lambda g: g["width"],
            )
            for ax, key in zip(
                axes, ("median_update_ms", "peak_allocated_mib", "peak_reserved_mib")
            ):
                ax.errorbar(
                    [g["width"] for g in selected],
                    [g["metrics"][key]["mean"] for g in selected],
                    yerr=[g["metrics"][key]["sample_sd"] or 0 for g in selected],
                    color=colors[kind],
                    marker="o",
                    lw=1.3,
                    label=labels[kind] + f" exp{expansion}",
                    linestyle="-" if expansion == 4 else "--",
                )
    for ax, title, ylabel in zip(
        axes,
        ("Complete update time", "Peak allocated memory", "Peak reserved memory"),
        ("Milliseconds", "MiB", "MiB"),
    ):
        ax.set(title=title, xlabel="Model width", ylabel=ylabel)
        ax.set_xticks(sorted({g["width"] for g in groups}))
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Observed complete-step resources\n" + identity, fontsize=12)
    fig.supxlabel(
        "Observed eager FP32 runs; error bars are sample SD across seeds. These points do not establish asymptotic runtime/memory complexity.",
        fontsize=9,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / ("resource-observations." + ext), dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plots", action="store_true", help="requires optional matplotlib"
    )
    args = parser.parse_args(argv)
    result = aggregate(args.evidence_root, json.loads(args.request.read_text()))
    result["request_sha256"] = digest(args.request)
    args.output.mkdir(parents=True, exist_ok=False)
    result["reporter_source_sha256"] = {
        name: digest(Path(__file__).parent / name)
        for name in ("ffn_report.py", "ffn_contract.py")
    }
    if args.plots:
        render_plots(result, args.output)
    write_markdown(result, args.output)
    result["derived_figures_sha256"] = {
        path.name: digest(path)
        for path in sorted(args.output.iterdir())
        if path.suffix in (".png", ".svg")
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "state": result["state"],
                "runs": len(result["runs"]),
                "request_sha256": result["request_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
