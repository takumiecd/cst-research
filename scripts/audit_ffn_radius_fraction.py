"""Read-only trained checkpoint audit against encoded-row FP64 references.

Run with the frozen experiment source on PYTHONPATH, outside that source tree.
"""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torchcst import CSTLinear, kernels
from torchcst._backends.torch.parameterizations.radius_envelope import from_state

from cst_research.experiments.ffn_contract import ModelConfig
from cst_research.experiments.ffn_model import LanguageModel, configure_native
from cst_research.experiments.ffn_run import evaluate, sha, verify_data


def layer_oracle(layer, seed):
    cpu = copy.deepcopy(layer).cpu()
    p = cpu.atoms.p
    physical = from_state(cpu.kernel, p)
    op = cpu.execution_declaration()
    reference = CSTLinear(
        *op.charts,
        atoms=1,
        dtype=torch.float64,
        backend="factored",
        kernel=kernels.product(
            *op.kernel.profiles,
            coordinates=kernels.physical(radius_init=(0.5, 50.0), radius_floor=0.5),
        ),
    )
    vi, uo = reference.operator.factors(physical.double())
    data = torch.Generator().manual_seed(seed)
    x = torch.randn(7, layer.in_features * 2, generator=data)
    dy = torch.randn(7, layer.out_features, generator=data)
    cx = x.double()[:, ::2].requires_grad_()
    cy = (cx @ vi) @ uo.T
    cg = torch.autograd.grad(cy, (cx, p), dy.double())
    gx = x.cuda()[:, ::2].requires_grad_()
    gy = layer(gx)
    gg = torch.autograd.grad(gy, (gx, layer.atoms.p), dy.cuda())
    errors = []
    for actual, expected in zip((gy, *gg), (cy, *cg)):
        actual = actual.detach().cpu().double()
        expected = expected.detach().double()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=4e-3)
        errors.append(float((actual - expected).abs().max()))
    return {
        "rows": 7,
        "noncontiguous_input": True,
        "max_abs_y_dx_dp": errors,
        "atol": 1e-4,
        "rtol": 4e-3,
        "reference": "FP64 factored physical kernels, retaining original FP32 physical-row encoding and complete map VJP",
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-long-models", type=int, default=9)
    parser.add_argument("--include-short", action="store_true")
    parser.add_argument("--kinds", nargs="+", default=["dense", "cst"])
    parser.add_argument("--include-milestones", action="store_true")
    parser.add_argument("--allow-cli-rejection", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.run.resolve()
    receipt = json.loads((root / "receipt.json").read_text())
    rejected = None
    if args.allow_cli_rejection:
        assert receipt["status"] == "failed"
        last = receipt["commands"][-1]
        assert last["label"] == "cst-lr128" and last["returncode"] == 2
        message = (root / "cst-lr128.log").read_text()
        assert "invalid choice: '128.0'" in message
        assert not (root / "cst-lr128" / "summary.json").exists()
        assert len(receipt["results"]) == 4
        source_files = json.loads((root / "source-files.json").read_text())
        assert all(sha(root / "source" / n) == h for n, h in source_files.items())
        rejected = {
            "label": last["label"],
            "returncode": 2,
            "training_started": False,
            "log_sha256": sha(root / "cst-lr128.log"),
            "original_receipt_sha256": sha(root / "receipt.json"),
            "source_unchanged_reverified": True,
        }
    else:
        assert receipt["status"] == "passed" and receipt["source_unchanged"]
    protocol = receipt["protocol"]
    assert (
        torch.cuda.get_device_name(0) == receipt["gpu"]
        and torch.__version__ == receipt["torch"]
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert (
        sha(Path(protocol["data"]) / "manifest.json")
        == protocol["data_manifest_sha256"]
    )
    _, data = verify_data(Path(protocol["data"]))
    result = {
        "state": "running",
        "audit_source_sha256": sha(Path(__file__)),
        "models": [],
        "confirmation_evaluated": False,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "excluded_cli_rejection": rejected,
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    jobs = []
    for row in receipt["results"]:
        if row["kind"] not in args.kinds:
            continue
        if row["stage"] != "long" and not args.include_short:
            continue
        directory = root / row["label"]
        summary = json.loads((directory / "summary.json").read_text())
        jobs.append(
            (
                row,
                "checkpoint.pt",
                summary["checkpoint_sha256"],
                summary["latest_development"]["ce"],
                False,
            )
        )
        if args.include_milestones:
            curves = [
                json.loads(line)
                for line in (directory / "curves.jsonl").read_text().splitlines()
            ]
            for filename, digest in summary.get(
                "milestone_checkpoint_sha256", {}
            ).items():
                updates = int(filename.split("-")[1]) // 1024
                if updates >= row["updates"]:
                    continue
                expected = next(
                    x["development"]["ce"]
                    for x in curves
                    if x["step"] == updates and "development" in x
                )
                jobs.append(
                    ({**row, "updates": updates}, filename, digest, expected, True)
                )
    try:
        for row, filename, digest, expected, is_milestone in jobs:
            directory = root / row["label"]
            summary = json.loads((directory / "summary.json").read_text())
            checkpoint = directory / filename
            assert sha(checkpoint) == digest
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = ModelConfig(**saved["config"])
            assert saved["updates"] == row["updates"]
            assert saved["completed_tokens"] == row["updates"] * 1024
            assert saved["seed"] == row["seed"] and saved["kind"] == row["kind"]
            assert asdict(config) == summary["config"]
            assert saved["data_manifest_sha256"] == protocol["data_manifest_sha256"]
            model = LanguageModel(config, row["kind"], row["seed"]).cuda()
            if "initial_parameter_sha256" in summary:
                from cst_research.experiments.ffn_model import parameter_digest

                assert parameter_digest(model) == summary["initial_parameter_sha256"]
            initial_radius = {}
            initial_fraction = {}
            if (directory / "fraction-updates.jsonl").exists():
                with torch.no_grad():
                    for name, layer in model.named_modules():
                        if isinstance(layer, CSTLinear):
                            initial_radius[name] = (
                                from_state(layer.kernel, layer.atoms.p)[:, 1]
                                .cpu()
                                .double()
                            )
                            initial_fraction[name] = (
                                layer.atoms.p[:, 1].detach().cpu().double().clone()
                            )
            model.load_state_dict(saved["model"])
            if row["kind"] == "cst":
                configure_native(model)
            states = saved["optimizer"]["state"]
            for state in states.values():
                assert int(state["step"]) == row["updates"]
                for name in ("exp_avg", "exp_avg_sq"):
                    assert (
                        state[name].dtype == torch.float32
                        and torch.isfinite(state[name]).all()
                    )
            observed = evaluate(model, data["development"], 2)["ce"]
            assert abs(observed - expected) <= 1e-6, (observed, expected)
            record = {
                **row,
                "checkpoint_sha256": digest,
                "checkpoint_filename": filename,
                "milestone": is_milestone,
                "reproduced_development_ce": observed,
                "ce_absolute_difference": abs(observed - expected),
                "optimizer_states_checked": len(states),
                "linear_oracles": [],
            }
            if row["kind"] == "cst":
                layers = [m for m in model.modules() if isinstance(m, CSTLinear)]
                assert len(layers) == 8
                record["fraction_changes"] = []
                for layer in layers:
                    p = layer.atoms.p
                    assert (
                        torch.isfinite(p).all()
                        and (p[:, 0].abs() <= config.amplitude_max).all()
                    )
                    assert ((p[:, 1] >= 0) & (p[:, 1] <= 1)).all()
                    delta = p[:, 1].detach().double() - config.radius_fraction
                    record["fraction_changes"].append(
                        {
                            "mean_signed": float(delta.mean()),
                            "mean_absolute": float(delta.abs().mean()),
                            "p90_absolute": float(torch.quantile(delta.abs(), 0.9)),
                            "maximum_absolute": float(delta.abs().max()),
                            "changed_fraction": float((delta != 0).double().mean()),
                        }
                    )
                for i, layer in enumerate(
                    (model.blocks[0].ffn.up.map, model.blocks[0].ffn.down.map)
                ):
                    record["linear_oracles"].append(
                        layer_oracle(layer, row["seed"] + i + 640)
                    )
            if initial_radius:
                entries = [
                    json.loads(line)
                    for line in (directory / "fraction-updates.jsonl")
                    .read_text()
                    .splitlines()
                ]
                frame = next(e for e in entries if e["step"] == row["updates"])
                assert frame["steps_observed"] == row["updates"]
                named = {
                    n: m for n, m in model.named_modules() if isinstance(m, CSTLinear)
                }
                assert named.keys() == frame["layers"].keys()
                with torch.no_grad():
                    for name, layer in named.items():
                        q = layer.atoms.p[:, 1].detach().cpu().double()
                        r = from_state(layer.kernel, layer.atoms.p)[:, 1].cpu().double()
                        counter = layer.atoms.p.detach().clone()
                        counter[:, 1].copy_(initial_fraction[name].to(counter))
                        ar = from_state(layer.kernel, counter)[:, 1].cpu().double()
                        for key, value in [
                            ("net_fraction_delta", q - initial_fraction[name]),
                            ("net_radius_delta", r - initial_radius[name]),
                            ("radius_fraction_component", r - ar),
                            ("radius_amplitude_component", ar - initial_radius[name]),
                        ]:
                            x = value.numpy()
                            expected_stats = {
                                "mean_signed": float(np.mean(x)),
                                "mean_absolute": float(np.mean(np.abs(x))),
                                "rms": float(np.sqrt(np.mean(x * x))),
                            }
                            for metric, number in expected_stats.items():
                                assert np.isclose(
                                    number,
                                    frame["layers"][name][key][metric],
                                    rtol=1e-10,
                                    atol=1e-12,
                                ), (name, key, metric)
                        assert (
                            frame["layers"][name]["fraction_path_length"][
                                "mean_absolute"
                            ]
                            + 1e-12
                            >= frame["layers"][name]["net_fraction_delta"][
                                "mean_absolute"
                            ]
                        )
                record["observer_net_statistics_reproduced"] = True
            result["models"].append(record)
            save()
            print("AUDIT PASS", row["label"], flush=True)
            del model, saved, states
            torch.cuda.empty_cache()
        assert (
            sum(r["stage"] == "long" and not r["milestone"] for r in result["models"])
            == args.expected_long_models
        )
        assert len(result["models"]) == len(jobs)
        result["state"] = "passed"
        save()
    except BaseException as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
