"""Post-hoc saved-task VJP/geometry witness; no training or optimizer updates."""

import argparse
import json
from pathlib import Path

import torch
from torchcst._backends.torch.parameterizations.radius_envelope import from_state

from cst_research.experiments.ffn_model import parameter_digest
from cst_research.experiments.ffn_run import sha
from cst_research.experiments.ffn_selective import projections
from cst_research.experiments.lm_connected import physical_reference
from cst_research.experiments.lm_placement import ARMS, STEPS, model_for

FIELDS = ("amplitude", "radius_fraction", "output_center", "input_center")


def summarize(raw, physical, grad, scale):
    active = raw[:, 0].abs() >= 0.01
    phase = raw[:, 2:].double().remainder(1)
    distance = torch.minimum(phase, 1 - phase)
    masks = {
        "active": active,
        "active_sharp": active & (physical[:, 1] < 0.6),
        "active_nonsharp": active & (physical[:, 1] >= 0.6),
        "active_wide_ge16": active & (physical[:, 1] >= 16),
    }
    for k, field in ((0, "output_center"), (1, "input_center")):
        masks["active_sharp_" + field + "_near_grid"] = (
            active & (physical[:, 1] < 0.6) & (distance[:, k] < 0.01)
        )
        masks["active_sharp_" + field + "_far_grid"] = (
            active & (physical[:, 1] < 0.6) & (distance[:, k] > 0.1)
        )
    result = {}
    for name, mask in masks.items():
        row = {"count": int(mask.sum())}
        if bool(mask.any()):
            row["fields"] = {
                field: {
                    "task_abs_gradient_median": float(grad[mask, k].abs().median()),
                    "task_gradient_below_1e-8_share": float(
                        (grad[mask, k].abs() < 1e-8).double().mean()
                    ),
                    "task_gradient_below_1e-10_share": float(
                        (grad[mask, k].abs() < 1e-10).double().mean()
                    ),
                    "adam_scale_below_epsilon_share": float(
                        (scale[mask, k] < 1e-8).double().mean()
                    ),
                }
                for k, field in enumerate(FIELDS)
            }
        result[name] = row
    return result, masks


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((a.run / "receipt.json").read_text())
    manifest = json.loads((a.run / "manifest.json").read_text())
    feature_analysis = json.loads((a.run / "analysis/analysis.json").read_text())
    assert feature_analysis["state"] == "passed"
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    assert json.loads((a.run / "audit/audit.json").read_text())["state"] == "passed"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090"
    result = {
        "state": "running",
        "script_sha256": sha(Path(__file__)),
        "source_commits": receipt["protocol"]["source_commits"],
        "scope": "post-hoc final checkpoints, original offset0 TRAIN X/cotangent, unscaled/unclipped local VJP; Adam scales are accumulated clipped-training history, no optimizer changes",
        "reference_sample_rule": "first up to4 active sharp atoms with both centers within0.01 of grid,4 sharp away from grid,4 radius>=16; deterministic mechanism witnesses, not population sampling",
        "records": [],
    }
    for arm, (kind, sites) in ARMS.items():
        if kind != "cst":
            continue
        m = model_for(arm)
        summary = json.loads((a.run / arm / "summary.json").read_text())
        assert (
            sha(a.run / arm / "summary.json")
            == manifest["files"][f"{arm}/summary.json"]
        )
        path = a.run / arm / f"checkpoint-{STEPS}.pt"
        assert sha(path) == summary["checkpoints"][path.name]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(saved["model"])
        expected = summary["final_parameters_sha256"]
        assert parameter_digest(m) == expected
        context_path = a.run / arm / f"signals-{STEPS}.pt"
        assert (
            sha(context_path)
            == feature_analysis["tensor_hashes"][f"{arm}/signals-{STEPS}.pt"]
        )
        context = torch.load(context_path, weights_only=True)
        assert (
            sha(a.run / arm / f"signals-{STEPS}.json")
            == manifest["files"][f"{arm}/signals-{STEPS}.json"]
        )
        probe = json.loads((a.run / arm / f"signals-{STEPS}.json").read_text())[
            "records"
        ][0]
        opt = saved["optimizer"]
        moments = {}
        for group, names in zip(opt["param_groups"], opt["cst_fields"]["groups"]):
            if names is None:
                continue
            for pid, (name, fields) in zip(group["params"], names):
                assert [f[0] for f in fields] == list(FIELDS)
                moments[name] = (
                    opt["state"][pid]["exp_avg_sq"].double()
                    / (1 - group["betas"][1] ** STEPS)
                ).sqrt()
        for site in sites:
            layer = projections(m)[site].map
            name = f"blocks.{site.split('.')[0]}.ffn.{site.split('.')[1]}.map.atom_state.atoms.p"
            x = context[site]["x"].reshape(-1, layer.in_features)
            g = context[site]["dy"].reshape(-1, layer.out_features)
            yy = layer(x.cuda())
            grad = (
                torch.autograd.grad(yy, layer.atoms.p, g.cuda())[0]
                .detach()
                .cpu()
                .double()
            )
            expected_rms = torch.tensor(
                probe["parameters"][name]["field_gradient_rms"], dtype=torch.float64
            )
            actual_rms = grad.square().mean(0).sqrt()
            torch.testing.assert_close(actual_rms, expected_rms, atol=1e-10, rtol=1e-5)
            raw = layer.atoms.p.detach().cpu()
            physical = from_state(layer.kernel, layer.atoms.p).detach().cpu()
            groups, masks = summarize(raw, physical, grad, moments[name])
            phase = raw[:, 2:].double().remainder(1)
            distance = torch.minimum(phase, 1 - phase)
            near = masks["active_sharp"] & (distance < 0.01).all(-1)
            far = masks["active_sharp"] & (distance > 0.1).all(-1)
            choices = {
                "sharp_both_near_grid": near,
                "sharp_both_away_grid": far,
                "wide": masks["active_wide_ge16"],
            }
            reference = physical_reference(layer.execution_declaration())
            samples = []
            for label, mask in choices.items():
                ids = mask.nonzero(as_tuple=True)[0][:4]
                if not len(ids):
                    continue
                rows = raw[ids].detach().clone().requires_grad_()
                ph = from_state(layer.kernel, rows).double()
                vi, vo = reference.operator.factors(ph)
                ry = (x.double() @ vi) @ vo.T
                rg = torch.autograd.grad(ry, rows, g.double())[0].double()
                samples.append(
                    {
                        "kind": label,
                        "atom_ids": ids.tolist(),
                        "raw_rows": raw[ids].tolist(),
                        "native_task_raw_gradients": grad[ids].tolist(),
                        "FP64_factor_task_raw_gradients": rg.tolist(),
                        "scope": "reference mechanism witnesses; near-zero relative errors are not normalized or called a gate failure",
                    }
                )
            result["records"].append(
                {
                    "arm": arm,
                    "site": site,
                    "checkpoint_sha256": summary["checkpoints"][path.name],
                    "local_task_gradient_rms_matches_saved_probe": True,
                    "subsets": groups,
                    "samples": samples,
                }
            )
            print("TASK-GEOMETRY", arm, site, "PASS", flush=True)
        assert parameter_digest(m) == expected
        del m, saved, context
        torch.cuda.empty_cache()
        (a.output / "geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    result["state"] = "passed"
    (a.output / "geometry.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
