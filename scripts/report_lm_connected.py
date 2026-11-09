"""Verified matched task curves and read-only update-fidelity plots."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def geometry_updates(root):
    """Decode stored actual raw endpoints, without changing a model or optimizer."""
    import torch
    from torchcst._backends.torch.parameterizations.radius_envelope import (
        physical_parameters,
    )

    result = []
    for arm in ("legacy-m8", "connected-m16"):
        for path in sorted((root / arm).glob("probe-*.pt")):
            probe = torch.load(path, map_location="cpu", weights_only=False)
            for site, record in probe["records"].items():
                old, new = record["old_p"], record["new_p"]
                spec = record["operator"].kernel.parameterization
                settings = {
                    k: getattr(spec, k)
                    for k in ("radius_min", "radius_max", "tau", "amplitude_max")
                }
                before = physical_parameters(old, **settings)[:, 1].double()
                after = physical_parameters(new, **settings)[:, 1].double()
                delta = after - before
                result.append(
                    {
                        "arm": arm,
                        "step": probe["step"],
                        "site": site,
                        "probe_sha256": sha(path),
                        "fraction_update_rms": float(
                            (new[:, 1].double() - old[:, 1].double())
                            .square()
                            .mean()
                            .sqrt()
                        ),
                        "fraction_changed_share": float(
                            (new[:, 1] != old[:, 1]).double().mean()
                        ),
                        "radius_update_rms": float(delta.square().mean().sqrt()),
                        "radius_changed_share": float((delta != 0).double().mean()),
                        "scope": "CPU decoding of saved raw FP32 endpoints with the public map; not a causal loss attribution",
                    }
                )
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dst-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--curves-only", action="store_true")
    p.add_argument("--checkpoint-audit", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = read(a.run / "export-manifest.json")
    assert all(sha(a.run / n) == h for n, h in manifest["files"].items())
    receipt = read(a.run / "receipt.json")
    if not a.curves_only:
        assert receipt["state"] == "passed" and receipt["source_unchanged"]
    else:
        assert all(
            next(c for c in receipt["commands"] if c["label"] == "train-" + arm)[
                "returncode"
            ]
            == 0
            for arm in ("dense", "dst-all", "legacy-m8", "connected-m16")
        )
    labels = {
        "dense": "Dense",
        "dst-all": "DST",
        "legacy-m8": "Legacy M8",
        "connected-m16": "Connected M16",
    }
    colors = {
        "dense": "#555555",
        "dst-all": "#d78c20",
        "legacy-m8": "#277bb8",
        "connected-m16": "#248148",
    }
    summaries = {}
    curves = {}
    for arm in labels:
        directory = a.dst_run if arm == "dst-all" else a.run / arm
        s = read(directory / "summary.json")
        assert s["state"] == "completed" and s["completed_steps"] == 976
        rows = [
            json.loads(v) for v in (directory / "curves.jsonl").read_text().splitlines()
        ]
        summaries[arm] = s
        curves[arm] = rows
    keys = (
        "config",
        "data_manifest_sha256",
        "data_order_sha256",
        "seed",
        "target_steps",
        "schedule_steps",
        "effective_tokens_per_step",
        "completed_tokens",
    )
    for s in summaries.values():
        for key in keys:
            assert s[key] == summaries["dense"][key], key
    assert (
        summaries["dst-all"]["active_ffn_scalars"]
        == summaries["legacy-m8"]["active_ffn_scalars"]
        == summaries["connected-m16"]["active_ffn_scalars"]
    )
    assert (
        summaries["legacy-m8"]["initial_parameters_sha256"]
        == summaries["connected-m16"]["initial_parameters_sha256"]
    )
    qualification = read(a.run / "qualification/qualification.json")
    assert qualification["state"] == "passed"
    assert (
        len(
            {
                v["shared_ordinary_parameters_sha256"]
                for v in qualification["arms"].values()
            }
        )
        == 1
    )
    results = {}
    for arm, s in summaries.items():
        final = next(row["development"] for row in curves[arm] if row["step"] == 976)
        initial = next(row["development"] for row in curves[arm] if row["step"] == 0)
        results[arm] = {
            "initial_development": initial,
            "final_development": final,
            "median_complete_step_ms": statistics.median(s["timing_seconds"]) * 1000,
            "peak_allocated_MiB": s["peak_allocated_bytes"] / 2**20,
            "peak_reserved_MiB": s["peak_reserved_bytes"] / 2**20,
            "checkpoint_sha256": s["checkpoints"],
            "final_parameters_sha256": s["final_parameters_sha256"],
            "final_oracles": s["final_oracles"],
            "curve_sha256": sha(
                (a.dst_run if arm == "dst-all" else a.run / arm) / "curves.jsonl"
            ),
        }

    def curve_plot(detail):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
        for arm, label in labels.items():
            rows = curves[arm]
            train = [
                r for r in rows if "train_ce" in r and (not detail or r["step"] >= 200)
            ]
            axes[0].plot(
                [r["seen_tokens"] / 1e6 for r in train],
                [r["train_ce"] for r in train],
                color=colors[arm],
                alpha=0.25,
                lw=0.8,
            )
            averages = [
                statistics.mean([r["train_ce"] for r in train[max(0, j - 2) : j + 1]])
                for j in range(len(train))
            ]
            axes[0].plot(
                [r["seen_tokens"] / 1e6 for r in train],
                averages,
                color=colors[arm],
                label=label,
                lw=1.6,
            )
            dev = [
                r
                for r in rows
                if "development" in r and (not detail or r["step"] >= 200)
            ]
            for ax, key in zip(axes[1:], ("ce", "next_token_top1_accuracy")):
                ax.plot(
                    [r["seen_tokens"] / 1e6 for r in dev],
                    [r["development"][key] for r in dev],
                    "-o",
                    ms=3,
                    color=colors[arm],
                    label=label,
                )
        for ax, title in zip(
            axes, ("TRAIN CE (3 logged-point mean)", "DEV CE", "DEV next-token top1")
        ):
            ax.set_title(title)
            ax.set_xlabel("Training targets (million)")
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            "TinyStories | width512 / 4 blocks / all FFN maps | seed17 | matched raw CST initialization",
            fontsize=10,
        )
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(
                a.output / f"learning-{'detail' if detail else 'full'}.{ext}", dpi=180
            )
        plt.close(fig)

    curve_plot(False)
    curve_plot(True)
    if a.curves_only:
        (a.output / "learning-summary.json").write_text(
            json.dumps(
                {"state": "training_complete_diagnostics_pending", "arms": results},
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(results, indent=2))
        return
    diagnostic = read(a.run / "diagnostics/summary.json")
    assert diagnostic["state"] == "passed" and len(diagnostic["records"]) == 12
    fields = ("radius_fraction", "output_center", "input_center")
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.6), sharex=True)
    for col, field in enumerate(fields):
        for arm in ("legacy-m8", "connected-m16"):
            for site, style in (("1.up", "-"), ("1.down", "--")):
                rows = sorted(
                    (
                        r
                        for r in diagnostic["records"]
                        if r["arm"] == arm and r["site"] == site
                    ),
                    key=lambda r: r["step"],
                )
                v = [r["fields"][field][0]["native_actual"] for r in rows]
                x = [r["step"] for r in rows]
                for ax, key in zip(
                    axes[:, col],
                    (
                        "linear_finite_cosine",
                        "relative_defect_to_prediction",
                        "finite_over_prediction_norm",
                    ),
                ):
                    yy = [r[key] for r in v]
                    ax.plot(
                        x,
                        [float("nan") if y is None else y for y in yy],
                        style + "o",
                        color=colors[arm],
                        ms=4,
                        label=labels[arm] + " " + site,
                    )
                    ax.set_xscale("log")
                    ax.grid(alpha=0.2)
        axes[0, col].set_title(field.replace("_", " "))
        axes[0, col].set_ylim(-1.05, 1.05)
        axes[0, col].axhline(1, color="black", alpha=0.2, lw=0.7)
        axes[1, col].set_yscale("log")
        axes[2, col].set_yscale("log")
        axes[2, col].axhline(1, color="black", alpha=0.2, lw=0.7)
        if col > 0:
            axes[2, col].set_yscale("linear")
            axes[2, col].set_ylim(0.95, 1.05)
        axes[2, col].set_xlabel("Actual optimizer update")
        axes[2, col].set_xticks([1, 256, 976], ["1", "256", "976"])
    for ax, title in zip(
        axes[:, 0],
        ("cosine(Jv, finite)", "||finite-Jv|| / ||Jv||", "||finite|| / ||Jv||"),
    ):
        ax.set_ylabel(title)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Observed task updates | native finite change vs declared Jv | raw FP32 endpoints | scale=1",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"update-fidelity.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for arm in ("legacy-m8", "connected-m16"):
        for site, style in (("1.up", "-"), ("1.down", "--")):
            rows = sorted(
                (
                    r
                    for r in diagnostic["records"]
                    if r["arm"] == arm and r["site"] == site
                ),
                key=lambda r: r["step"],
            )
            values = [r["fields"]["radius_fraction"][0]["native_actual"] for r in rows]
            for ax, key in zip(
                axes, ("linear_finite_cosine", "finite_over_prediction_norm")
            ):
                ax.plot(
                    [r["step"] for r in rows],
                    [v[key] for v in values],
                    style + "o",
                    color=colors[arm],
                    label=labels[arm] + " " + site,
                    ms=4,
                )
                ax.set_xscale("log")
                ax.set_xticks([1, 256, 976], ["1", "256", "976"])
                ax.set_xlabel("Actual optimizer update")
                ax.grid(alpha=0.2)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Direction: cosine(Jv, actual change)")
    axes[1].set_yscale("log")
    axes[1].set_title("Magnitude: actual norm / predicted norm")
    axes[1].axhline(1, color="black", alpha=0.3, lw=0.7)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Radius-fraction updates | native finite change | same short task protocol",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"radius-fidelity.{ext}", dpi=180)
    plt.close(fig)
    native_log = re.sub(r"\x1b\[[0-9;]*m", "", (a.run / "native-tests.log").read_text())
    native_passed = int(re.search(r"(\d+) passed", native_log).group(1))
    identity = json.loads((a.run / "identity.log").read_text().strip())
    audit = None
    if a.checkpoint_audit is not None:
        audit = read(a.checkpoint_audit)
        assert audit["state"] == "passed"
        for arm, row in audit["arms"].items():
            assert row["parameter_sha256"] == summaries[arm]["final_parameters_sha256"]
    out = {
        "checkpoint_audit": audit,
        "runtime": identity,
        "execution": {
            "native_tests_passed": native_passed,
            "source_unchanged": receipt["source_unchanged"],
            "stages": [
                {"label": c["label"], "exit_code": c["returncode"]}
                for c in receipt["commands"]
            ],
        },
        "state": "verified",
        "protocol": receipt["protocol"]
        | {"source_files_count": len(receipt["protocol"]["source_files"])},
        "arms": results,
        "diagnostics": diagnostic,
        "qualification": qualification,
        "geometry_updates": geometry_updates(a.run),
        "export_manifest_sha256": sha(a.run / "export-manifest.json"),
        "renderer_sha256": sha(Path(__file__)),
    }
    out["protocol"].pop("source_files")
    # Machine-specific evidence paths belong in raw receipts, not public notes.
    for key in ("remote_root", "data"):
        out["protocol"].pop(key, None)
    for r in out["diagnostics"]["records"]:
        r.pop("probe_path", None)
    (a.output / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
