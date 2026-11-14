"""Independent-radius FFN admission using existing model and VJP/restore gates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torchcst import CSTLinear

from .ffn_contract import ModelConfig, qualification_covers_model
from .ffn_model import (
    LanguageModel,
    configure_native,
    cst_optimizer,
    make_cst,
    parameter_seed,
)
from .ffn_qualification import (
    compare,
    qualify_case,
    reference_vjp,
    task_step_and_resume_check,
)
from .ffn_radius import convert_layer


def quadrature(layer, p):
    """Raw T8 Gauss quadrature with LIVE shared physical radius on both axes."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes = p.new_tensor(nodes)
    weights = p.new_tensor(weights)
    factors = []
    for n, column in [(layer.in_features, 3), (layer.out_features, 2)]:
        rows = []
        for radius, center in zip(p[:, 1], p[:, column]):
            tile = 1
            while float(radius.detach()) > tile:
                tile *= 2
            count = math.ceil(2 * float(radius.detach()) / tile) + 1
            period = count * tile
            center = center.remainder(n)
            first = (
                math.floor((float(center.detach()) + 0.5) / tile - (count - 1) / 2)
                * tile
            )
            sites = torch.arange(first, first + count * tile, dtype=p.dtype)

            def integrate(left, right, radius=radius, period=period):
                u = (right - left)[..., None] / 2 * nodes + (right + left)[
                    ..., None
                ] / 2
                beta = 1 / torch.cos(torch.pi * radius / period)
                v = beta * torch.cos(torch.pi * u / period)
                a, b = torch.ones_like(v), v
                for _ in range(2, 9):
                    a, b = b, 2 * v * b - a
                value = b / torch.cosh(8 * torch.acosh(beta))
                return (right - left) / 2 * (value.square() * weights).sum(-1)

            raw = integrate(sites - 0.5 - center, sites + 0.5 - center)
            norm = integrate(p.new_tensor(-period / 2), p.new_tensor(period / 2))
            rows.append(
                p.new_zeros(n).scatter_add(0, sites.long().remainder(n), raw / norm)
            )
        factors.append(torch.stack(rows).T)
    return factors[0], factors[1] * p[:, 0][None, :]


def quadrature_gate():
    base = make_cst(
        16,
        64,
        std=0.001,
        radius_birth=0.5,
        radius_envelope="rational_zero_opening",
        zero_opening_scale=0.001,
    )
    live = convert_layer(base, "learned").double()
    with torch.no_grad():
        live.atoms.p[:, 1] = torch.linspace(0.63, 2.7, live.atoms.count)
        live.atoms.p[:, 2] = torch.linspace(0.13, 63.31, live.atoms.count)
        live.atoms.p[:, 3] = torch.linspace(0.17, 15.37, live.atoms.count)
    # Bound the independent quadrature gate; every field of these atoms is checked.
    p = live.atoms.p[:4].detach().requires_grad_()
    x = torch.randn(
        3, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(330)
    )
    dy = torch.randn(
        3, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(331)
    )

    def values(function):
        tx = x.detach().requires_grad_()
        pp = p.detach().requires_grad_()
        v, u = function(pp)
        y = (tx @ v) @ u.T
        gx, gp = torch.autograd.grad(y, (tx, pp), dy)
        return y, gx, gp

    actual = values(live._factor_atoms)
    expected = values(lambda pp: quadrature(live, pp))
    checks = {
        n: compare(a, b, atol=2e-9, rtol=2e-7)
        for n, a, b in zip(["y", "dx", "dp"], actual, expected)
    }
    assert expected[2][:, 1].abs().max() > 0
    return {
        "atoms_checked": 4,
        "checks": checks,
        "radius_derivative_live": True,
        "both_axes_share_radius": True,
    }


def physical_case(ni, no, std, seed, config):
    polar = make_cst(
        ni,
        no,
        std=std,
        seed=seed,
        coefficient=config.coefficient,
        q=config.q,
        amplitude_max=config.amplitude_max,
        w_c=config.w_c,
        **config.envelope_settings(),
    )
    layer = convert_layer(polar, "learned")
    base = torch.randn(
        2, 128, ni * 2, generator=torch.Generator().manual_seed(seed + 1)
    )
    x = base[..., ::2]
    dy = torch.randn(2, 128, no, generator=torch.Generator().manual_seed(seed + 2))
    expected = reference_vjp(layer, x, dy)
    live = copy.deepcopy(layer).cuda()
    configure_native(live, linear_backend="levels")
    xx = base.cuda()[..., ::2].requires_grad_()
    y = live(xx)
    grads = torch.autograd.grad(y, (xx, live.atoms.p), dy.cuda(), retain_graph=True)
    again = torch.autograd.grad(y, (xx, live.atoms.p), dy.cuda())
    for a, b in zip(grads, again):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert grads[1][:, 1].abs().max() > 0
    metrics = {
        n: compare(a, b) for n, a, b in zip(["y", "dx", "dp"], [y, *grads], expected)
    }
    return {
        **config.recipe(),
        **config.envelope_settings(),
        "in": ni,
        "out": no,
        "atoms": live.atoms.count,
        "batch": 2,
        "tokens": 128,
        "coefficient": config.coefficient,
        "q": config.q,
        "amplitude_max": config.amplitude_max,
        "w_c": config.w_c,
        "radius_birth": config.radius_birth,
        "all_atoms_checked": True,
        "checks": metrics,
        "retained_backward_exact": True,
        "radius_gradient_nonzero": True,
    }


def matched_initialization(config):
    models = {
        mode: LanguageModel(replace(config, radius_mode=mode), "cst", 17)
        for mode in ["polar", "learned", "fixed"]
    }
    ordinary = {
        n: p.detach()
        for n, p in models["polar"].named_parameters()
        if not n.endswith("atoms.p")
    }
    converted = {
        n: m for n, m in models["learned"].named_modules() if isinstance(m, CSTLinear)
    }
    frozen = {
        n: m for n, m in models["fixed"].named_modules() if isinstance(m, CSTLinear)
    }
    for mode, m in models.items():
        for n, p in m.named_parameters():
            if n in ordinary:
                torch.testing.assert_close(p, ordinary[n], atol=0, rtol=0)
    for n, m in converted.items():
        torch.testing.assert_close(m.atoms.p, frozen[n].atoms.p, atol=0, rtol=0)
    x = torch.randint(
        0,
        config.vocabulary,
        (2, config.context),
        generator=torch.Generator().manual_seed(997),
    ).cuda()
    target = x.roll(-1, 1)
    logits = {}
    policy = {}
    for mode, m in models.items():
        m.cuda()
        configure_native(m)
        before = {
            n: layer.atoms.p.detach().clone()
            for n, layer in m.named_modules()
            if isinstance(layer, CSTLinear)
        }
        y, loss = m(x, target)
        logits[mode] = y.detach().cpu()
        loss.backward()
        policy[mode] = {}
        for n, layer in m.named_modules():
            if not isinstance(layer, CSTLinear):
                continue
            g = layer.atoms.p.grad
            assert torch.isfinite(g).all() and g.abs().max() > 0
            if mode == "learned":
                assert g[:, 1].abs().max() > 0
            if mode == "fixed":
                assert torch.count_nonzero(g[:, 1]) == 0
        optimizer = cst_optimizer(
            m,
            atom_lr_multiplier=16,
            lr=0.0003,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            foreach=False,
        )
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        for n, layer in m.named_modules():
            if not isinstance(layer, CSTLinear):
                continue
            changed = int(torch.count_nonzero(layer.atoms.p[:, 1] != before[n][:, 1]))
            if mode == "learned":
                assert changed > 0
            if mode == "fixed":
                assert changed == 0
                assert (
                    torch.count_nonzero(optimizer.state[layer.atoms.p]["exp_avg"][:, 1])
                    == 0
                )
            policy[mode][n] = {
                "second_field_changed_atoms": changed,
                "radius_mode": mode,
            }
        m.cpu()
        del optimizer, y, loss
    checks = {
        mode: compare(logits[mode], logits["polar"]) for mode in ["learned", "fixed"]
    }
    torch.testing.assert_close(logits["learned"], logits["fixed"], atol=0, rtol=0)
    return {
        "shared_ordinary_parameters_exact": True,
        "physical_learned_fixed_initial_p_exact": True,
        "physical_initial_logits_exact": True,
        "polar_vs_physical_initial_logits": checks,
        "policy": policy,
        "scope": "Polar-to-physical conversion agrees within FP32 gates; physical learned/fixed are bit-identical",
    }


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == "NVIDIA RTX 6000 Ada Generation"
    config = ModelConfig(
        width=512,
        heads=8,
        linear_backend="levels",
        radius_envelope="rational_zero_opening",
        radius_birth=0.5,
        zero_opening_scale=0.001,
        atom_lr_multiplier=16,
    )
    files = [
        "ffn_model.py",
        "ffn_contract.py",
        "ffn_qualification.py",
        "ffn_radius.py",
        "ffn_run.py",
        "ffn_radius_qualification.py",
    ]
    source = {
        n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
        for n in files
    }
    all_results = {
        "state": "running",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "source_sha256": source,
    }

    def save():
        (args.output / "qualification-all.json").write_text(
            json.dumps(all_results, indent=2) + "\n"
        )

    save()
    try:
        all_results["physical_quadrature"] = quadrature_gate()
        save()
        all_results["matched_initialization"] = matched_initialization(config)
        save()
        physical_cases = []
        polar_cases = []
        for ni, no, std, name in [
            (512, 2048, 0.02, "block0.up"),
            (2048, 512, 0.02 / (8**0.5), "block0.down"),
        ]:
            seed = parameter_seed(17, name)
            physical_cases.append(
                physical_case(ni, no, std, seed, replace(config, radius_mode="learned"))
            )
            polar_cases.append(
                qualify_case(
                    ni,
                    no,
                    128,
                    seed=seed,
                    std=std,
                    linear_backend="levels",
                    **config.envelope_settings(),
                )
            )
            all_results["physical_cases"] = physical_cases
            all_results["polar_cases"] = polar_cases
            save()
        for mode in ["polar", "learned", "fixed"]:
            c = replace(config, radius_mode=mode)
            rows = copy.deepcopy(polar_cases if mode == "polar" else physical_cases)
            if mode != "polar":
                for row in rows:
                    row["radius_mode"] = mode
            gate = task_step_and_resume_check(
                full_task=True, ffn_kind="cst", task_config=c, atom_lr_multiplier=16
            )
            assert gate["cst_projections"] == 8 and gate["max_parameter_error"] == 0
            result = {
                "state": "passed",
                "cohort": "recipe",
                "gpu": all_results["gpu"],
                "torch": all_results["torch"],
                "cuda": all_results["cuda"],
                "source_sha256": source,
                "task_recipes": [c.recipe()],
                "task_model_configs": [asdict(c)],
                "model_kinds": [{"ffn_kind": "cst", "attention_kind": "dense"}],
                "cases": rows,
                "task_step": gate,
                "initialization": all_results["matched_initialization"],
                "physical_quadrature": all_results["physical_quadrature"],
            }
            assert qualification_covers_model(result, c, ffn_kind="cst")
            (args.output / (mode + ".json")).write_text(
                json.dumps(result, indent=2) + "\n"
            )
            all_results.setdefault("modes", {})[mode] = gate
            save()
        all_results["state"] = "passed"
        save()
        print("INDEPENDENT RADIUS QUALIFICATION PASSED", flush=True)
    except BaseException as e:
        all_results.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main()
