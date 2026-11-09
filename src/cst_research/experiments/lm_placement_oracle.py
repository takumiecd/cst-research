"""Explicit post-failure FP32 numerical admission; original failure is retained."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torchcst._backends.torch.parameterizations.radius_envelope import from_state

from .ffn_selective import projections
from .lm_connected import physical_reference
from .lm_placement import ARMS, STEPS, model_for


def error(actual, expected):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    d = a - b
    return {
        "max_abs": float(d.abs().max()),
        "relative_l2": float(d.norm() / b.norm().clamp_min(1e-30)),
        "original_component_gate_failures": int(
            (d.abs() > 1e-4 + 4e-3 * b.abs()).sum()
        ),
    }


def full_oracle(layer):
    """Same full-A reference and seed as original, with declared per-field metrics."""
    cpu = copy.deepcopy(layer).cpu()
    reference = physical_reference(cpu.execution_declaration())
    generator = torch.Generator().manual_seed(447)
    x = torch.randn(3, layer.in_features, generator=generator)
    dy = torch.randn(3, layer.out_features, generator=generator).double()
    value = torch.zeros(3, layer.out_features, dtype=torch.float64)
    gx = torch.zeros_like(x, dtype=torch.float64)
    gradients = []
    for start in range(0, len(cpu.atoms.p), 128):
        raw = cpu.atoms.p[start : start + 128].detach().clone().requires_grad_()
        xx = x.double().requires_grad_()
        vi, vo = reference.operator.factors(from_state(cpu.kernel, raw).double())
        yy = (xx @ vi.double()) @ vo.double().T
        dx, dp = torch.autograd.grad(yy, (xx, raw), dy)
        value += yy.detach()
        gx += dx
        gradients.append(dp)
    gp = torch.cat(gradients)
    xx = x.cuda().requires_grad_()
    yy = layer(xx)
    dx, dp = torch.autograd.grad(yy, (xx, layer.atoms.p), dy.float().cuda())
    reports = {"Y": error(yy, value), "dX": error(dx, gx)}
    for k, name in enumerate(
        ("amplitude", "radius_fraction", "output_center", "input_center")
    ):
        reports[name] = error(dp[:, k], gp[:, k])
    passed = all(
        r["max_abs"] <= 3e-4 and r["relative_l2"] <= 1e-4 for r in reports.values()
    )
    return {
        "passed": passed,
        "A": len(cpu.atoms.p),
        "B": 3,
        "errors": reports,
        "gate": {
            "max_abs_each_tensor_or_raw_field": 3e-4,
            "relative_l2_each_tensor_or_raw_field": 1e-4,
        },
        "scope": "post-failure numerical admission, not passage of the original componentwise gate",
    }


def witness(layer):
    """Independent64-point Gauss-Legendre integral versus Fourier factors/native."""
    from benchmarks.cuda.linear.connected_cell_mass import scalar_axis

    atom = 5334
    raw = layer.atoms.p[atom : atom + 1].detach().cpu().requires_grad_()
    physical = from_state(copy.deepcopy(layer).cpu().kernel, raw).double()
    gen = torch.Generator().manual_seed(447)
    x = torch.randn(3, layer.in_features, generator=gen).double()
    dy = torch.randn(3, layer.out_features, generator=gen).double()
    vi = scalar_axis(physical[0], layer.in_features, 16, True, 3)
    vo = scalar_axis(physical[0], layer.out_features, 16, True, 2)
    yy = (x @ vi[:, None] * physical[:, 0]) @ vo[None, :]
    quad = torch.autograd.grad(yy, raw, dy)[0]
    reference = physical_reference(layer.execution_declaration())
    rp = raw.detach().clone().requires_grad_()
    ph = from_state(copy.deepcopy(layer).cpu().kernel, rp).double()
    fi, fo = reference.operator.factors(ph)
    fy = (x @ fi) @ fo.T
    factor = torch.autograd.grad(fy, rp, dy)[0]
    torch.testing.assert_close(quad, factor, atol=1e-8, rtol=1e-5)
    return {
        "atom": atom,
        "raw": raw.detach().tolist(),
        "physical": physical.detach().tolist(),
        "quadrature_raw_gradient": quad.tolist(),
        "factor_raw_gradient": factor.tolist(),
        "max_abs": float((quad - factor).abs().max()),
        "passed": True,
        "scope": "independent polynomial-integral check, no native CUDA coefficient reuse",
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--witness", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {"state": "running", "arms": {}}
    for arm, (kind, sites) in ARMS.items():
        if kind != "cst" or (a.arm and arm != a.arm):
            continue
        m = model_for(arm)
        saved = torch.load(
            a.run / arm / f"checkpoint-{STEPS}.pt",
            map_location="cpu",
            weights_only=False,
        )
        m.load_state_dict(saved["model"])
        del saved
        selected = [
            next(s for s in sites if s.endswith("." + role)) for role in ("up", "down")
        ]
        result["arms"][arm] = {
            site: full_oracle(projections(m)[site].map) for site in selected
        }
        if a.witness and arm == "pair12":
            result["independent_witness"] = witness(projections(m)["1.down"].map)
        (a.output / "oracle.json").write_text(json.dumps(result, indent=2) + "\n")
        assert all(r["passed"] for r in result["arms"][arm].values())
        print("FULL-ORACLE", arm, "PASS", flush=True)
        del m
        torch.cuda.empty_cache()
    result["state"] = "passed"
    (a.output / "oracle.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
