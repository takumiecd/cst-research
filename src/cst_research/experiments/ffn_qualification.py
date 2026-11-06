"""Rectangular/multi-token native qualification with bounded CPU FP64 oracles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import platform
import time
from pathlib import Path

import torch

from .ffn_model import configure_native, cst_optimizer, make_cst


def reference_vjp(layer, x, cotangent, chunk=128):
    """Independent Torch factor computation; no CUDA hierarchy or native backward.

    All atom coordinates and all input rows participate. Only one chunk's
    differentiable graph lives at a time. Shared factor mathematics is augmented
    by the independent quadrature check below, rather than called independent
    kernel-mathematics validation on its own.
    """
    tx = x.detach().double().cpu().requires_grad_()
    dy = cotangent.detach().double().cpu()
    y = torch.zeros(*x.shape[:-1], layer.out_features, dtype=torch.float64)
    dx = torch.zeros_like(tx)
    gradients = []
    for first in range(0, layer.atoms.count, chunk):
        p = layer.atoms.p[first : first + chunk].detach().double().requires_grad_()
        vi, uo = layer._factor_atoms(p)
        current = (tx @ vi) @ uo.T
        gx, gp = torch.autograd.grad(current, (tx, p), dy)
        y.add_(current.detach())
        dx.add_(gx.detach())
        gradients.append(gp.detach())
    return y, dx, torch.cat(gradients)


def quadrature_factors(layer, p):
    """Raw Chebyshev-polynomial Gauss quadrature; no Fourier coefficient code.

    Retains amplitude/center derivatives and piecewise routing. Polar task
    radius derivatives are detached, matching the declared public contract;
    this is not a claim about a different fully differentiable radius operator.
    This bounded fixture uses shared input/output radius declarations.
    """
    import math

    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes, weights = p.new_tensor(nodes), p.new_tensor(weights)
    polar = p[:, :2]
    rho = polar.square().sum(-1).sqrt()
    amplitude = float(layer.kernel.scalar("amplitude_max")) * polar[:, 0] / rho
    alpha = ((rho.square() - 1) / 3).clamp(0, 1)
    z = (amplitude / float(layer.kernel.scalar("w_c"))).square()
    # The fixture is explicitly constructed with birth4, min.5 and max8.
    lower = 0.5 + 3.5 / (1 + float(layer.kernel.scalar("lower_kappa")) * z)
    upper = (0.5 + 7.5 * 3 / (3 + z)).clamp_min(1)
    upper = torch.maximum(upper, lower)
    radius = torch.exp((1 - alpha) * lower.log() + alpha * upper.log()).detach()
    factors = []
    for count_sites, column in ((layer.in_features, 3), (layer.out_features, 2)):
        rows = []
        for r, center in zip(radius, p[:, column]):
            tile = 1
            while float(r.detach()) > tile:
                tile *= 2
            count = math.ceil(2 * float(r.detach()) / tile) + 1
            period = count * tile
            center = center.remainder(count_sites)
            first = (
                math.floor((float(center.detach()) + 0.5) / tile - (count - 1) / 2)
                * tile
            )
            sites = torch.arange(first, first + count * tile, dtype=p.dtype)

            def integrate(left, right, r=r, period=period):
                u = (right - left)[..., None] / 2 * nodes + (right + left)[
                    ..., None
                ] / 2
                beta = 1 / torch.cos(torch.pi * r / period)
                v = beta * torch.cos(torch.pi * u / period)
                a, b = torch.ones_like(v), v
                for _ in range(2, 9):
                    a, b = b, 2 * v * b - a
                value = b / torch.cosh(8 * torch.acosh(beta))
                return (right - left) / 2 * (value.square() * weights).sum(-1)

            left = sites - 0.5 - center
            raw = integrate(left, left + 1)
            norm = integrate(p.new_tensor(-period / 2), p.new_tensor(period / 2))
            rows.append(
                p.new_zeros(count_sites).scatter_add(
                    0, sites.long().remainder(count_sites), raw / norm
                )
            )
        factors.append(torch.stack(rows).T)
    return factors[0], factors[1] * amplitude[None]


def independent_quadrature_check():
    layer = make_cst(16, 64, std=0.02, q=0.25, dtype=torch.float64)
    # All eight stratified atoms in this bounded fixture; the full native cases
    # separately cover every atom in their larger declared budgets.
    p = layer.atoms.p.detach()[::32].clone().requires_grad_()
    x = torch.randn(
        3, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(351)
    )
    dy = torch.randn(
        3, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(352)
    )
    vi, uo = quadrature_factors(layer, p)
    actual = (x @ vi) @ uo.T
    grad = torch.autograd.grad(actual, p, dy)[0]
    vi, uo = layer._factor_atoms(p)
    expected = (x @ vi) @ uo.T
    gp = torch.autograd.grad(expected, p, dy)[0]
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-8)
    torch.testing.assert_close(grad, gp, atol=2e-9, rtol=2e-7)
    return {
        "atoms_checked": len(p),
        "max_y_error": float((actual - expected).detach().abs().max()),
        "max_dp_error": float((grad - gp).abs().max()),
    }


def compare(got, expected, *, atol=3e-5, rtol=3e-4):
    got, expected = got.detach().cpu().double(), expected.detach().cpu().double()
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
    return {
        "max_abs_error": float((got - expected).abs().max()),
        "reference_rms": float(expected.square().mean().sqrt()),
        "atol": atol,
        "rtol": rtol,
    }


def qualify_case(ni, no, rows, *, seed=17, coefficient=8, q=0.0, std=0.02):
    began = time.monotonic()
    layer = make_cst(ni, no, seed=seed, coefficient=coefficient, q=q, std=std)
    report = layer.initialization_report
    base = torch.randn(
        2, rows, ni * 2, generator=torch.Generator().manual_seed(seed + 1)
    )
    x = base[..., ::2]
    dy = torch.randn(2, rows, no, generator=torch.Generator().manual_seed(seed + 2))
    expected = reference_vjp(layer, x, dy)
    live = copy.deepcopy(layer).cuda()
    configure_native(live)
    xx = base.cuda()[..., ::2].requires_grad_()
    torch.cuda.reset_peak_memory_stats()
    y = live(xx)
    gx, gp = torch.autograd.grad(y, (xx, live.atoms.p), dy.cuda())
    metrics = {
        key: compare(a, b)
        for key, a, b in zip(("y", "dx", "dp"), (y, gx, gp), expected)
    }
    # The leading dimensions must have no token-mixing semantics of their own.
    flat = live(xx.reshape(-1, ni)).reshape(y.shape)
    metrics["flatten"] = compare(y, flat, atol=0, rtol=0)
    return {
        "in": ni,
        "out": no,
        "batch": 2,
        "tokens": rows,
        "atoms": live.atoms.count,
        "coefficient": coefficient,
        "q": q,
        "std": std,
        "input_noncontiguous": not xx.is_contiguous(),
        "initialization": report,
        "checks": metrics,
        "all_atoms_checked": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.monotonic() - began,
    }


def _assert_tree_equal(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_tree_equal(a[key], b[key], path + "." + str(key))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for index, (left, right) in enumerate(zip(a, b)):
            _assert_tree_equal(left, right, path + "." + str(index))
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def assert_model_restore(saved, restored, before_load):
    """Check all payload bytes and the declared load-time cache invalidation.

    AtomState.set_extra_state advances layout_version on load; equal logical
    weights do not imply equal cache generations. Validate that one precise
    transformation rather than dropping metadata or weakening tensor checks.
    """
    assert saved.keys() == restored.keys() == before_load.keys()
    left, right = copy.deepcopy(saved), copy.deepcopy(restored)
    versions = {}
    for key in left:
        if key.endswith("atom_state._extra_state"):
            a, b, c = left[key], right[key], before_load[key]
            assert (
                set(a)
                == set(b)
                == set(c)
                == {"format_version", "shape", "layout_version"}
            )
            expected = max(c["layout_version"] + 1, a["layout_version"])
            assert b["layout_version"] == expected
            versions[key] = {
                "saved": a["layout_version"],
                "before_load": c["layout_version"],
                "restored": b["layout_version"],
            }
            del a["layout_version"]
            del b["layout_version"]
    _assert_tree_equal(left, right)
    return versions


def optimizer_and_resume_check():
    from torchcst import CSTOptimizer

    cpu = make_cst(16, 64, seed=151)
    live = copy.deepcopy(cpu).cuda()
    configure_native(live)
    reference = CSTOptimizer(
        torch.optim.AdamW(cpu.parameters(), lr=0.0003, weight_decay=0.0, foreach=False),
        model=cpu,
    )
    optimizer = cst_optimizer(live, lr=0.0003, weight_decay=0.0, foreach=False)
    for step in range(2):
        gradient = (
            torch.randn(
                cpu.atoms.p.shape, generator=torch.Generator().manual_seed(450 + step)
            )
            * 0.001
        )
        cpu.atoms.p.grad = gradient.clone()
        live.atoms.p.grad = gradient.cuda()
        reference.step()
        optimizer.step()
        compare(live.atoms.p, cpu.atoms.p, atol=2e-6, rtol=2e-5)
        a, b = (
            optimizer.state[live.atoms.p],
            reference.base_optimizer.state[cpu.atoms.p],
        )
        for key in ("exp_avg", "exp_avg_sq", "step"):
            compare(a[key], b[key], atol=2e-8, rtol=2e-5)
    storage = io.BytesIO()
    torch.save(
        {"model": live.state_dict(), "optimizer": optimizer.state_dict()}, storage
    )
    storage.seek(0)
    checkpoint = torch.load(storage, weights_only=False)
    resumed = make_cst(16, 64, seed=151).cuda()
    before_load = copy.deepcopy(resumed.state_dict())
    resumed.load_state_dict(checkpoint["model"])
    configure_native(resumed)
    other = cst_optimizer(resumed, lr=0.0003, weight_decay=0.0, foreach=False)
    other.load_state_dict(checkpoint["optimizer"])
    for model, opt in ((live, optimizer), (resumed, other)):
        model.atoms.p.grad = torch.full_like(model.atoms.p, 0.001)
        opt.step()
    versions = assert_model_restore(
        live.state_dict(), resumed.state_dict(), before_load
    )
    _assert_tree_equal(optimizer.state_dict(), other.state_dict())
    return {
        "controlled_gradient_updates": 2,
        "parameters_and_moments_checked": True,
        "serialized_resume_next_step_exact": True,
        "cache_generation_load_contract": versions,
    }


def task_step_and_resume_check():
    from .ffn_contract import ModelConfig
    from .ffn_model import LanguageModel

    config = ModelConfig(width=16, layers=2, heads=2, context=8, vocabulary=64)
    model = LanguageModel(config, "cst", seed=251).cuda()
    configure_native(model)
    optimizer = cst_optimizer(model, lr=0.0003, weight_decay=0.0, foreach=False)
    tokens = torch.randint(
        0, 64, (2, 9), generator=torch.Generator().manual_seed(252)
    ).cuda()
    x, y = tokens[:, :-1], tokens[:, 1:]
    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.endswith("atoms.p")
    }

    def step(m, opt):
        opt.zero_grad(set_to_none=True)
        _, loss = m(x, y)
        loss.backward()
        for name, p in m.named_parameters():
            if name.endswith("atoms.p"):
                assert (
                    p.grad is not None
                    and torch.isfinite(p.grad).all()
                    and p.grad.abs().max() > 0
                )
        opt.step()
        return float(loss.detach())

    first_loss = step(model, optimizer)
    assert all(
        not torch.equal(before[n], p)
        for n, p in model.named_parameters()
        if n in before
    )
    storage = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, storage
    )
    storage.seek(0)
    state = torch.load(storage, weights_only=False)
    resumed = LanguageModel(config, "cst", seed=251).cuda()
    before_load = copy.deepcopy(resumed.state_dict())
    configure_native(resumed)
    resumed.load_state_dict(state["model"])
    other = cst_optimizer(resumed, lr=0.0003, weight_decay=0.0, foreach=False)
    other.load_state_dict(state["optimizer"])
    versions = assert_model_restore(
        model.state_dict(), resumed.state_dict(), before_load
    )
    _assert_tree_equal(optimizer.state_dict(), other.state_dict())
    left, right = step(model, optimizer), step(resumed, other)
    errors = [
        compare(p, dict(resumed.named_parameters())[n], atol=3e-6, rtol=3e-5)[
            "max_abs_error"
        ]
        for n, p in model.named_parameters()
    ]
    for (n, p), (_, q) in zip(model.named_parameters(), resumed.named_parameters()):
        a, b = optimizer.state[p], other.state[q]
        for key in ("exp_avg", "exp_avg_sq", "step"):
            compare(a[key], b[key], atol=3e-7, rtol=3e-5)
    return {
        "cst_projections": 4,
        "all_cst_sites_received_finite_nonzero_gradients_and_changed": True,
        "first_loss": first_loss,
        "next_loss": left,
        "resumed_next_loss": right,
        "serialized_state_exact_before_step": True,
        "next_task_step_fp32_tolerance_passed": True,
        "max_parameter_error": max(errors),
        "cache_generation_load_contract": versions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cohort",
        choices=("small", "target", "expanded", "capacity", "radius"),
        default="small",
    )
    parser.add_argument(
        "--q", type=float, choices=(0.0, 0.25, 0.5, 0.75, 1.0), default=0.0
    )
    args = parser.parse_args(argv)
    if args.cohort == "radius" and args.q != 0.0:
        parser.error("radius cohort declares its own q0/.25/.5 family")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "state": "running",
        "cohort": args.cohort,
        "requested_q": args.q,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cases": [],
        "source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in ("ffn_qualification.py", "ffn_model.py", "ffn_contract.py")
        },
    }

    def save():
        (args.output / "qualification.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    try:
        result["quadrature"] = independent_quadrature_check()
        save()
        print("Independent quadrature passed", flush=True)
        cases = (
            [(16, 64, 7), (64, 16, 7)]
            if args.cohort == "small"
            else [
                (128, 512, 128),
                (512, 128, 128),
                (256, 1024, 128),
                (1024, 256, 128),
                (256, 256, 32),
                (64, 1024, 32),
                (1024, 64, 32),
            ]
        )
        if args.cohort == "expanded":
            cases += [
                (128, 128, 128),
                (512, 2048, 128),
                (2048, 512, 128),
                (128, 2048, 128),
                (2048, 128, 128),
            ]
        coefficient_cases = [(ni, no, tokens, 8) for ni, no, tokens in cases]
        if args.cohort == "capacity":
            coefficient_cases = [
                (ni, no, 128, c)
                for c in (8, 16, 32)
                for ni, no in ((256, 1024), (1024, 256))
            ]
        radius_cases = [
            (ni, no, tokens, c, args.q, 0.02) for ni, no, tokens, c in coefficient_cases
        ]
        if args.cohort == "radius":
            radius_cases = [
                (ni, no, 128, 8, q, 0.02 if ni < no else 0.02 / math.sqrt(8))
                for q in (0.0, 0.25, 0.5)
                for ni, no in ((256, 1024), (1024, 256))
            ]
        for ni, no, tokens, coefficient, q, std in radius_cases:
            result["cases"].append(
                qualify_case(ni, no, tokens, coefficient=coefficient, q=q, std=std)
            )
            save()
            print(
                f"PASS {ni}->{no}, c{coefficient}, q{q}, B2 T{tokens}, all Atom gradients",
                flush=True,
            )
        result["optimizer"] = optimizer_and_resume_check()
        print("Controlled optimizer and exact serialized resume passed", flush=True)
        result["task_step"] = task_step_and_resume_check()
        result["state"] = "passed"
        save()
        print("Optimizer and serialized resume passed", flush=True)
    except Exception as error:
        result["state"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
