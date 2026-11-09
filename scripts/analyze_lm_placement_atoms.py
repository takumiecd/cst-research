"""Read checkpoint Atom geometry and Adam moments; never execute an optimizer."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torchcst._backends.torch.parameterizations.radius_envelope import (
    physical_parameters,
)

FIELDS = ("amplitude", "radius_fraction", "output_center", "input_center")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(value):
    x = value.double()
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "rms": float(x.square().mean().sqrt()),
        "p10": float(x.quantile(0.1)),
        "p90": float(x.quantile(0.9)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    receipt = json.loads((a.run / "receipt.json").read_text())
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    assert json.loads((a.run / "audit/audit.json").read_text())["state"] == "passed"
    result = {
        "state": "running",
        "script_sha256": sha(Path(__file__)),
        "source_commits": receipt["protocol"]["source_commits"],
        "scope": "post-hoc read-only saved Adam moments, not actual next updates or an optimizer intervention",
        "records": [],
    }
    for arm in receipt["protocol"]["arms"]:
        if arm in ("dense", "dst-pair12"):
            continue
        summary = json.loads((a.run / arm / "summary.json").read_text())
        cfg = summary["config"]
        for filename, h in summary["checkpoints"].items():
            path = a.run / arm / filename
            assert sha(path) == h
            saved = torch.load(path, map_location="cpu", weights_only=False)
            opt = saved["optimizer"]
            manifest = opt["cst_fields"]
            assert manifest["version"] == 1
            for group, declared in zip(opt["param_groups"], manifest["groups"]):
                if declared is None:
                    continue
                assert len(group["params"]) == len(declared)
                beta1, beta2 = group["betas"]
                for pid, (name, fields) in zip(group["params"], declared):
                    assert [f[0] for f in fields] == list(FIELDS)
                    assert [(f[1], f[2]) for f in fields] == [
                        (i, i + 1) for i in range(4)
                    ]
                    raw = saved["model"][name]
                    state = opt["state"][pid]
                    assert (
                        raw.shape == state["exp_avg"].shape == state["exp_avg_sq"].shape
                    )
                    assert raw.dtype == torch.float32 and raw.shape[1] == 4
                    clock = int(state["step"])
                    assert clock == saved["step"]
                    physical = physical_parameters(
                        raw,
                        radius_min=0.5,
                        radius_max=50.0,
                        tau=cfg["tau"],
                        amplitude_max=cfg["amplitude_max"],
                    )
                    m = state["exp_avg"].double() / (1 - beta1**clock)
                    scale = (state["exp_avg_sq"].double() / (1 - beta2**clock)).sqrt()
                    unit = -m / (scale + group["eps"])
                    rates = {field: group["lr"] for field in FIELDS}
                    for field, rate in group.get("field_lrs", {}).items():
                        rates[field] = (
                            rate * group["lr"] / group["_cst_field_lr_reference"]
                        )
                    proposal = unit * torch.tensor(
                        [rates[field] for field in FIELDS], dtype=torch.float64
                    )
                    direction = torch.where(
                        proposal >= 0,
                        torch.full_like(raw, float("inf")),
                        torch.full_like(raw, float("-inf")),
                    )
                    ulp = (
                        torch.nextafter(raw, direction).double() - raw.double()
                    ).abs()
                    masks = {
                        "all": torch.ones(len(raw), dtype=torch.bool),
                        "active_amp_ge_0.01": raw[:, 0].abs() >= 0.01,
                        "near_zero_amp_lt_0.003": raw[:, 0].abs() < 0.003,
                    }
                    summaries = {}
                    for label, mask in masks.items():
                        if not bool(mask.any()):
                            summaries[label] = {"count": 0}
                            continue
                        phase = raw[mask, 2:].double().remainder(1)
                        distance = torch.minimum(phase, 1 - phase)
                        field_stats = {}
                        for i, field in enumerate(FIELDS):
                            field_stats[field] = {
                                "adam_scale": stats(scale[mask, i]),
                                "normalized_moment": stats(unit[mask, i]),
                                "raw_proposal": stats(proposal[mask, i]),
                                "adam_scale_below_eps_share": float(
                                    (scale[mask, i] < group["eps"]).double().mean()
                                ),
                                "adam_scale_below_10eps_share": float(
                                    (scale[mask, i] < 10 * group["eps"]).double().mean()
                                ),
                                "sub_half_ulp_proposal_share": float(
                                    (proposal[mask, i].abs() < 0.5 * ulp[mask, i])
                                    .double()
                                    .mean()
                                ),
                            }
                        q = raw[mask, 1]
                        outward = ((q == 0) & (proposal[mask, 1] < 0)) | (
                            (q == 1) & (proposal[mask, 1] > 0)
                        )
                        summaries[label] = {
                            "count": int(mask.sum()),
                            "absolute_amplitude": stats(raw[mask, 0].abs()),
                            "signed_amplitude_mean": float(
                                raw[mask, 0].double().mean()
                            ),
                            "radius": stats(physical[mask, 1]),
                            "radius_fraction": stats(q),
                            "radius_below_0.6_share": float(
                                (physical[mask, 1] < 0.6).double().mean()
                            ),
                            "fraction_at_zero_share": float((q == 0).double().mean()),
                            "fraction_at_one_share": float((q == 1).double().mean()),
                            "fraction_outward_moment_at_boundary_share": float(
                                outward.double().mean()
                            ),
                            "output_center_distance_to_integer": stats(distance[:, 0]),
                            "input_center_distance_to_integer": stats(distance[:, 1]),
                            "centers_both_within_0.01_share": float(
                                (distance < 0.01).all(-1).double().mean()
                            ),
                            "fields": field_stats,
                        }
                    result["records"].append(
                        {
                            "arm": arm,
                            "step": clock,
                            "parameter": name,
                            "checkpoint_sha256": h,
                            "effective_field_lrs": rates,
                            "subsets": summaries,
                        }
                    )
            del saved
            print("ATOM-STATE", arm, filename, "PASS", flush=True)
    result["state"] = "passed"
    (a.output / "atom-state.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
