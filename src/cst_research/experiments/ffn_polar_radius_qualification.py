"""Qualify the Polar-to-physical composition and projected-Adam pair on Ada."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .ffn_contract import ModelConfig, qualification_covers_model
from .ffn_model import LanguageModel, make_cst, parameter_seed
from .ffn_polar_radius import (
    MODES,
    PolarRadiusProjection,
    mapped_parameters,
)
from .ffn_qualification import compare, reference_vjp, task_step_and_resume_check
from .ffn_radius import convert_layer, probe_radius
from .ffn_radius_qualification import quadrature, quadrature_gate
from .polar_radius_chain_rule import check_chain_rule


def composition_oracle():
    source = make_cst(
        16,
        64,
        std=0.001,
        radius_birth=0.5,
        radius_envelope="rational_zero_opening",
        zero_opening_scale=0.001,
    ).double()
    ref = convert_layer(source, "learned").double()
    a = torch.tensor([-0.9, -0.3, -0.03, 0.03, 0.3, 0.9], dtype=torch.float64)
    q = torch.tensor([0.1, 0.25, 0.4, 0.6, 0.8, 0.9], dtype=torch.float64)
    rho = (1 + 3 * q).sqrt()
    p = torch.stack(
        (
            rho * a,
            rho * (1 - a * a).sqrt(),
            torch.arange(6, dtype=torch.float64) * 3.7 + 0.17,
            torch.arange(6, dtype=torch.float64) * 1.7 + 0.23,
        ),
        1,
    )
    x = torch.randn(
        3, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(618)
    )
    dy = torch.randn(
        3, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(619)
    )
    result = {}

    def loss(pp, live, quad=False):
        physical = mapped_parameters(source, pp, live=live)
        v, u = quadrature(ref, physical) if quad else ref._factor_atoms(physical)
        return (((x @ v) @ u.T) * dy).sum()

    for live in [False, True]:
        z = p.detach().requires_grad_()
        actual = torch.autograd.grad(loss(z, live), z)[0]
        z = p.detach().requires_grad_()
        expected = torch.autograd.grad(loss(z, live, True), z)[0]
        result[str(live)] = compare(actual, expected, atol=2e-9, rtol=2e-7)
        if live:
            eps = 1e-6
            fd = torch.empty_like(p)
            for i in range(6):
                for j in range(4):
                    delta = torch.zeros_like(p)
                    delta[i, j] = eps
                    fd[i, j] = (loss(p + delta, True) - loss(p - delta, True)) / (
                        2 * eps
                    )
            result["live_central_difference"] = compare(
                actual, fd, atol=2e-7, rtol=2e-5
            )
    # The projected-domain inward slope is explicitly live at BOTH endpoints.
    for dtype in [torch.float32, torch.float64]:
        for endpoint in [1.0, 2.0]:
            pp = torch.tensor(
                [[0.6 * endpoint, 0.8 * endpoint, 3.17, 1.23]],
                dtype=dtype,
                requires_grad=True,
            )
            expected = mapped_parameters(source, pp, live=True)
            cg = torch.autograd.grad(expected[:, 1].sum(), pp)[0]
            gpu_source = copy.deepcopy(source).cuda()
            pg = pp.detach().cuda().requires_grad_()
            value = mapped_parameters(gpu_source, pg, live=True)
            gg = torch.autograd.grad(value[:, 1].sum(), pg)[0]
            result[f"{dtype}-{endpoint}"] = {
                "value": compare(value, expected),
                "gradient": compare(gg, cg),
            }
            assert (cg[:, :2].double() * pp[:, :2].double() / endpoint).sum() > 0
    return result


def mapped_case(ni, no, std, seed, config):
    base = make_cst(
        ni, no, std=std, seed=seed, coefficient=8, q=0, **config.envelope_settings()
    )
    raw = torch.randn(2, 128, ni * 2, generator=torch.Generator().manual_seed(seed + 1))
    x = raw[..., ::2]
    dy = torch.randn(2, 128, no, generator=torch.Generator().manual_seed(seed + 2))
    rows = {}
    for mode in MODES:
        layer = PolarRadiusProjection(copy.deepcopy(base), mode)
        expected = reference_vjp(layer, x, dy)
        live = copy.deepcopy(layer).cuda()
        xx = raw.cuda()[..., ::2].requires_grad_()
        yy = live(xx)
        grads = torch.autograd.grad(
            yy, (xx, live.atoms.p), dy.cuda(), retain_graph=True
        )
        again = torch.autograd.grad(yy, (xx, live.atoms.p), dy.cuda())
        for a, b in zip(grads, again):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        rows[mode] = {
            **replace(config, radius_mode=mode).recipe(),
            **config.envelope_settings(),
            "in": ni,
            "out": no,
            "atoms": 8192,
            "batch": 2,
            "tokens": 128,
            "all_atoms_checked": True,
            "retained_backward_exact": True,
            "checks": {
                n: compare(a, b)
                for n, a, b in zip(["y", "dx", "dp"], [yy, *grads], expected)
            },
        }
    return rows


def matching_and_memory(config):
    models = {
        mode: LanguageModel(replace(config, radius_mode=mode), "cst", 17)
        for mode in MODES
    }
    for a, b in zip(models[MODES[0]].parameters(), models[MODES[1]].parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    x = torch.randint(
        0, config.vocabulary, (2, 128), generator=torch.Generator().manual_seed(997)
    ).cuda()
    target = x.roll(-1, 1)
    logits = {}
    diagnostics = {}
    memory = {}
    for mode, m in models.items():
        m.cuda()
        logits[mode] = m(x).detach().cpu()
        diagnostics[mode] = probe_radius(m, x, target)
        assert len(diagnostics[mode]["layers"]) == 8
        for row in diagnostics[mode]["layers"].values():
            assert row["physical_radius_partial"]["max_abs"] > 0
            assert (row["radius_task_gradient"]["max_abs"] > 0) == (
                mode == "polar_live"
            )
        # No update or data selection: repeat complete backward/cleanup to
        # expose retained functional-bindings/graphs before learner admission.
        allocated = []
        for step in range(32):
            m.zero_grad(set_to_none=True)
            _, loss = m(x, target)
            loss.backward()
            del loss
            torch.cuda.synchronize()
            allocated.append(torch.cuda.memory_allocated())
        m.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        # GC cycles may make individual samples saw-tooth. A 128MiB envelope
        # bounds all32 forwards/backwards; no forced collection inside the loop.
        assert max(allocated) - min(allocated) < 128 * 2**20
        memory[mode] = {
            "backwards": 32,
            "allocated_bytes": allocated,
            "spread_bytes": max(allocated) - min(allocated),
            "after_gc_bytes": torch.cuda.memory_allocated(),
        }
        m.cpu()
    torch.testing.assert_close(logits[MODES[0]], logits[MODES[1]], atol=0, rtol=0)
    return {
        "initial_all_parameters_exact": True,
        "initial_logits_exact": True,
        "initial_diagnostics": diagnostics,
        "retention_gate": memory,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == "NVIDIA RTX 6000 Ada Generation"
    config = ModelConfig(
        width=512,
        heads=8,
        linear_backend="levels",
        atom_lr_multiplier=16,
        radius_birth=0.5,
        radius_envelope="rational_zero_opening",
        zero_opening_scale=0.001,
        radius_mode="polar_detached",
    )
    names = [
        "ffn_model.py",
        "ffn_contract.py",
        "ffn_run.py",
        "ffn_radius.py",
        "ffn_qualification.py",
        "ffn_polar_radius.py",
        "ffn_polar_radius_qualification.py",
        "polar_radius_chain_rule.py",
    ]
    source = {
        n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
        for n in names
    }
    result = {
        "state": "running",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": source,
    }

    def save():
        (args.output / "qualification-all.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    save()
    try:
        result["physical_quadrature"] = quadrature_gate()
        result["chain_algebra"] = check_chain_rule()
        result["composition"] = composition_oracle()
        save()
        result["initialization"] = matching_and_memory(config)
        save()
        cases = {mode: [] for mode in MODES}
        for ni, no, std, name in [
            (512, 2048, 0.02, "block0.up"),
            (2048, 512, 0.02 / (8**0.5), "block0.down"),
        ]:
            rows = mapped_case(ni, no, std, parameter_seed(17, name), config)
            for mode in MODES:
                cases[mode].append(rows[mode])
            result["cases"] = cases
            save()
        for mode in MODES:
            c = replace(config, radius_mode=mode)
            gate = task_step_and_resume_check(
                full_task=True, ffn_kind="cst", task_config=c, atom_lr_multiplier=16
            )
            assert gate["cst_projections"] == 8 and gate["max_parameter_error"] == 0
            receipt = {
                "state": "passed",
                "cohort": "recipe",
                "gpu": result["gpu"],
                "torch": result["torch"],
                "cuda": result["cuda"],
                "source_sha256": source,
                "cases": cases[mode],
                "task_recipes": [c.recipe()],
                "task_model_configs": [asdict(c)],
                "model_kinds": [{"ffn_kind": "cst", "attention_kind": "dense"}],
                "task_step": gate,
            }
            assert qualification_covers_model(receipt, c, ffn_kind="cst")
            (args.output / (mode + ".json")).write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            result.setdefault("modes", {})[mode] = gate
            save()
        result["state"] = "passed"
        save()
        print("MAPPED POLAR PAIR QUALIFICATION PASSED", flush=True)
    except BaseException as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
