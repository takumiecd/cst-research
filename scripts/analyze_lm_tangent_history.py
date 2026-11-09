"""Read-only separation of local Atom geometry, Adam history and finite response.

No training continuation, learning-rate search, or model improvement is performed.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from audit_lm_updates import (
    accumulate,
    clone_state,
    endpoint,
    finish,
    load_optimizer_snapshot,
)
from lm_tangent_geometry import (
    FactorTangent,
    LinearTangent,
    adam_directions,
    cosine,
    dot,
    factor_partials,
    free_projection,
    masked_projection,
    tangent_projection,
)

CASES = {
    "dense": ("1.up", "1.down"),
    "dst-all": ("1.up", "1.down"),
    "cst-up": ("1.up",),
    "cst-all": ("1.up", "1.down"),
}
FIELDS = ("amplitude", "radius_fraction", "output_center", "input_center")


def read(path):
    return json.loads(path.read_text())


def relative_error(actual, expected):
    return math.sqrt(
        dot(actual - expected, actual - expected) / max(dot(expected, expected), 1e-30)
    )


def group_for(optimizer, parameter):
    return next(
        g for g in optimizer.param_groups if any(p is parameter for p in g["params"])
    )


def pack(parameter, bias, value, bias_value, mask=None):
    del parameter, bias
    return torch.cat(
        (value[mask] if mask is not None else value.flatten(), bias_value.flatten())
    )


def bound_tangent(v, layer, kind):
    if kind != "cst":
        return v
    count = layer.map.atoms.p.numel()
    p = layer.map.atoms.p.detach()
    d = v[:count].reshape_as(p).clone()
    cap = layer.map.kernel.scalar("amplitude_max").to(p)
    d[:, 0] = torch.where(
        ((p[:, 0] >= cap) & (d[:, 0] > 0)) | ((p[:, 0] <= -cap) & (d[:, 0] < 0)),
        0,
        d[:, 0],
    )
    d[:, 1] = torch.where(
        ((p[:, 1] <= 0) & (d[:, 1] < 0)) | ((p[:, 1] >= 1) & (d[:, 1] > 0)), 0, d[:, 1]
    )
    return torch.cat((d.flatten(), v[count:]))


def route_signature(layer, point):
    """Only declaration routing metadata, not an alternative kernel evaluator."""
    from torchcst._backends.torch.kernels.radius_cell_mass import _relative_centers
    from torchcst._backends.torch.parameterizations.radius_envelope import from_state
    from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes

    op = layer.execution_declaration()
    axes = radius_cell_mass_axes(op.charts, op.kernel)
    radius = from_state(layer.kernel, point)[:, 1].double()
    result = []
    for axis, col in zip(axes, (2, 3)):
        tile = torch.full_like(radius, axis.binding.base_tile_sites, dtype=torch.int64)
        while bool((radius > tile * axis.spacing).any()):
            tile = torch.where(
                radius > tile * axis.spacing, tile * axis.binding.merge_factor, tile
            )
        count = (2 * radius / (tile * axis.spacing)).ceil().long() + 1
        center = _relative_centers(point[:, col], axis.origin, axis.period)
        first = (
            ((center + axis.spacing / 2) / (tile * axis.spacing) - (count - 1) / 2)
            .floor()
            .long()
        )
        result.append(
            torch.stack((tile, count, first.remainder(axis.site_count // tile)), 1)
        )
    return torch.cat(result, 1)


def captured_step(model, optimizer, batches, step, sites):
    from cst_research.experiments.ffn_radius_fraction_qualification import assert_nested
    from cst_research.experiments.ffn_selective import projections

    parameters = dict(model.named_parameters())
    old = clone_state(model)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    state = copy.deepcopy(optimizer.state_dict())
    cpu_rng, cuda_rng = (
        torch.get_rng_state().clone(),
        torch.cuda.get_rng_state().clone(),
    )
    model.train()
    _, captured = accumulate(model, optimizer, batches, capture=True)
    norm = finish(model, optimizer, step)
    result = {}
    try:
        for site in sites:
            layer = projections(model)[site]
            is_cst = hasattr(layer.map, "atoms")
            param = layer.map.atoms.p if is_cst else layer.map.weight
            name = next(n for n, p in parameters.items() if p is param)
            biasname = next(n for n, p in parameters.items() if p is layer.bias)
            group, bg = group_for(optimizer, param), group_for(optimizer, layer.bias)
            rates = torch.full_like(param, group["lr"])
            if is_cst:
                configured = optimizer.get_field_lrs()[
                    next(i for i, g in enumerate(optimizer.param_groups) if g is group)
                ]
                for k, field in enumerate(FIELDS):
                    rates[:, k] = configured.get(field, group["lr"])
            delta = param.detach() - old[name]
            if is_cst:
                for k, period in (
                    (2, layer.map.out_features),
                    (3, layer.map.in_features),
                ):
                    delta[:, k] = (
                        (
                            param[:, k].detach().double()
                            - old[name][:, k].double()
                            + period / 2
                        ).remainder(period)
                        - period / 2
                    ).to(delta)
            mask = layer.map.mask if hasattr(layer.map, "mask") else None
            pstate, bstate = optimizer.state[param], optimizer.state[layer.bias]
            vectors = adam_directions(
                param.grad.detach(),
                pstate["exp_avg"],
                pstate["exp_avg_sq"],
                int(pstate["step"]),
                rates,
                group["eps"],
                group["betas"],
            )
            bvectors = adam_directions(
                layer.bias.grad.detach(),
                bstate["exp_avg"],
                bstate["exp_avg_sq"],
                int(bstate["step"]),
                torch.full_like(layer.bias, bg["lr"]),
                bg["eps"],
                bg["betas"],
            )
            row = {
                "vectors": {
                    k: pack(param, layer.bias, v, bvectors[k], mask).clone()
                    for k, v in vectors.items()
                },
                "actual_vector": pack(
                    param, layer.bias, delta, layer.bias.detach() - old[biasname], mask
                ).clone(),
                "x": torch.cat(
                    [
                        v["input"].reshape(-1, v["input"].shape[-1])
                        for v in captured[site]
                    ]
                ),
                "g": torch.cat(
                    [
                        v["gradient"].reshape(-1, v["gradient"].shape[-1])
                        for v in captured[site]
                    ]
                ),
                "old_y": torch.cat(
                    [
                        v["output"].reshape(-1, v["output"].shape[-1])
                        for v in captured[site]
                    ]
                ),
                "old_p": old[name].detach().cpu().clone(),
                "new_p": param.detach().cpu().clone(),
                "bias": old[biasname].clone(),
                "bias_delta": layer.bias.detach().clone() - old[biasname],
                "gradient": pack(
                    param,
                    layer.bias,
                    param.grad.detach(),
                    layer.bias.grad.detach(),
                    mask,
                ).clone(),
                "mask": mask.clone() if mask is not None else None,
            }
            with torch.no_grad():
                row["finite_delta"] = (
                    torch.cat([layer(x) for x in row["x"].split(256)]) - row["old_y"]
                )
            result[site] = row
    finally:
        endpoint(model, old, old, [])
        with torch.no_grad():
            for n, b in model.named_buffers():
                b.copy_(buffers[n])
        load_optimizer_snapshot(optimizer, state)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    assert_nested(dict(model.named_parameters()), old)
    assert_nested(dict(model.named_buffers()), buffers)
    assert_nested(optimizer.state_dict(), state)
    return result, float(norm)


def analyze_map(layer, row, kind, factors, primary, extended=False):
    x, g = row["x"], row["g"]
    target = -g
    if kind == "cst":
        vi, uo, dis, dos = factors
        op = FactorTangent(x, vi, uo, dis, dos)
        reference_y = op.b @ uo.T + row["bias"]
    else:
        op = LinearTangent(
            x,
            layer.map.out_features
            if hasattr(layer.map, "out_features")
            else row["old_y"].shape[1],
            row["mask"],
        )
        reference_y = x @ row["old_p"].to(x).T + row["bias"]
    yerror = relative_error(reference_y, row["old_y"])
    assert yerror <= 0.005, yerror
    # Compare factor-cache VJP to the original task gradient after undoing clipping.
    reference_gradient = op.vjp(g)
    gradient = row["gradient"]
    clip = min(1.0, 1.0 / (row["global_gradient_norm"] + 1e-6))
    if row["mask"] is not None:
        # The cache uses only active coordinates, as the optimizer does.
        pass
    error = relative_error(reference_gradient * clip, gradient)
    assert error <= 0.005, error
    result = {
        "reference_y_relative_error": yerror,
        "task_vjp_relative_error": error,
        "tokens": len(x),
        "outputs": target.shape[1],
        "actual_finite_cosine": cosine(row["finite_delta"], target),
        "directions": {},
        "primary_projection": primary,
    }
    for name, vector in {
        **row["vectors"],
        "actual_quantized_displacement": row["actual_vector"],
    }.items():
        for bounded in (False, True) if kind == "cst" else (False,):
            v = bound_tangent(vector, layer, kind) if bounded else vector
            scale = max(float(v.abs().max()), 1e-30)
            action = op.jvp(v / scale)
            key = name + ("_feasible_tangent" if bounded else "")
            result["directions"][key] = {
                "cosine": cosine(action, target),
                "norm_before_arbitrary_direction_rescaling": float(
                    vector.double().square().sum().sqrt()
                ),
            }
    linear = op.jvp(row["actual_vector"])
    result["actual_linearization"] = {
        "tangent_cosine": cosine(linear, target),
        "finite_cosine": result["actual_finite_cosine"],
        "linear_finite_cosine": cosine(linear, row["finite_delta"]),
        "relative_defect": relative_error(row["finite_delta"], linear),
    }
    if primary:
        unrestricted, singular = free_projection(x, target)
        result["unrestricted_linear_tangent"] = {
            "cosine": cosine(unrestricted, target),
            **singular,
        }
        if kind == "cst":
            projected, audit = tangent_projection(
                op,
                target,
                row["actual_vector"],
                max_iterations=2048 if extended else 256,
            )
            result["atom_tangent_projection"] = {
                "cosine": cosine(projected, target),
                **audit,
            }
            cpu = copy.deepcopy(layer.map).cpu()
            old_routes = route_signature(cpu, row["old_p"])
            result["finite_response"] = []
            count = row["old_p"].numel()
            delta = row["actual_vector"][:count].reshape_as(row["old_p"]).cpu()
            old_reference = reference_y
            for epsilon in (
                (1.0, 0.1, 0.01, 0.001, 0.0001) if extended else (1.0, 0.1, 0.01)
            ):
                point = row["old_p"] + epsilon * delta
                for k, period in (
                    (2, layer.map.out_features),
                    (3, layer.map.in_features),
                ):
                    point[:, k].remainder_(period)
                with torch.no_grad():
                    vi, uo = cpu.operator.factors(point)
                    change = (
                        (x @ vi.to(x)) @ uo.to(x).T
                        + row["bias"]
                        + epsilon * row["bias_delta"]
                        - old_reference
                    )
                routes = route_signature(cpu, point)
                changed = (routes != old_routes).any(1)
                entry = {
                    "epsilon": epsilon,
                    "finite_cosine": cosine(change, target),
                    "linear_finite_cosine": cosine(epsilon * linear, change),
                    "relative_defect": relative_error(change, epsilon * linear),
                    "changed_route_fraction": float(changed.float().mean()),
                    "finite_norm": math.sqrt(dot(change, change)),
                    "tangent_norm": epsilon * math.sqrt(dot(linear, linear)),
                }
                if extended:
                    mask = changed.to(x.device)
                    changed_delta = (x @ vi[:, changed].to(x)) @ uo[:, changed].to(
                        x
                    ).T - (x @ op.vi[:, mask]) @ op.uo[:, mask].T
                    changed_linear = epsilon * op.atom_subset_jvp(
                        row["actual_vector"], mask
                    )
                    residual = change - epsilon * linear
                    changed_residual = changed_delta - changed_linear
                    remaining = residual - changed_residual
                    total_norm = max(math.sqrt(dot(residual, residual)), 1e-30)
                    entry.update(
                        changed_atom_defect_norm_ratio=math.sqrt(
                            dot(changed_residual, changed_residual)
                        )
                        / total_norm,
                        unchanged_atom_defect_norm_ratio=math.sqrt(
                            dot(remaining, remaining)
                        )
                        / total_norm,
                        route_component_change_fractions=[
                            float((routes[:, k] != old_routes[:, k]).float().mean())
                            for k in range(6)
                        ],
                    )
                if epsilon == 1:
                    entry["reference_native_delta_relative_error"] = relative_error(
                        change, row["finite_delta"]
                    )
                    assert entry["reference_native_delta_relative_error"] <= 0.01
                result["finite_response"].append(entry)
        elif row["mask"] is not None:
            projected, audit = masked_projection(x, target, row["mask"])
            result["masked_tangent_projection"] = {
                "cosine": cosine(projected, target),
                **audit,
            }
    return result


def cross_batch_analysis(layer, batches, site, kind, factors):
    names = (
        "historical_m_historical_v",
        "current_g_historical_v",
        "current_g_current_g2",
        "current_gradient",
    )
    matrices = {name: [[None] * len(batches) for _ in batches] for name in names}
    for j, (_, rows) in enumerate(batches):
        target_row = rows[site]
        x = target_row["x"]
        goal = -target_row["g"]
        op = (
            FactorTangent(x, *factors)
            if kind == "cst"
            else LinearTangent(x, goal.shape[1], target_row["mask"])
        )
        for i, (_, sources) in enumerate(batches):
            for name in names:
                v = bound_tangent(sources[site]["vectors"][name], layer, kind)
                matrices[name][i][j] = cosine(
                    op.jvp(v / max(float(v.abs().max()), 1e-30)), goal
                )
    return {
        "source_offsets": [k for k, _ in batches],
        "target_offsets": [k for k, _ in batches],
        "cosines": matrices,
        "scope": "Directions computed on one TRAIN batch and evaluated on other TRAIN batches at identical Parameters; no updates or selection.",
    }


def main():
    from cst_research.experiments.ffn_contract import ModelConfig, learning_rate
    from cst_research.experiments.ffn_model import configure_native, parameter_digest
    from cst_research.experiments.ffn_run import sha, token_batch, verify_data
    from cst_research.experiments.ffn_selective import (
        make_model,
        make_optimizer,
        projections,
    )

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extended-analysis", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = read(args.base_run / "receipt.json")
    assert parent["state"] == "passed" and parent["source_unchanged"]
    assert (
        torch.cuda.get_device_name(0) == parent["gpu"] == "NVIDIA GeForce RTX 4090"
        and torch.__version__ == parent["torch"]
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_root = Path(parent["protocol"]["data"])
    assert (
        sha(data_root / "manifest.json") == parent["protocol"]["data_manifest_sha256"]
    )
    _, data = verify_data(data_root)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    summary = {
        "state": "running",
        "script_sha256": sha(Path(__file__)),
        "geometry_sha256": sha(Path(__file__).with_name("lm_tangent_geometry.py")),
        "data_manifest_sha256": parent["protocol"]["data_manifest_sha256"],
        "checkpoint_step": 4882,
        "records": [],
        "cross_batch": [],
        "training_updates": 0,
        "scope": "Read-only local geometry/history attribution; no continuation, LR search, teacher or adoption",
    }
    target = args.output / "summary.json"

    def save():
        target.write_text(json.dumps(summary, indent=2) + "\n")

    save()
    try:
        for arm, sites in CASES.items():
            s = read(args.base_run / arm / "summary.json")
            checkpoint = args.base_run / arm / "checkpoint-4999168-tokens.pt"
            assert sha(checkpoint) == s["milestone_checkpoint_sha256"][checkpoint.name]
            saved = torch.load(
                checkpoint, map_location="cpu", weights_only=False, mmap=True
            )
            m = make_model(
                ModelConfig(**s["config"]), s["kind"], tuple(s["sites"]), 17
            ).cuda()
            m.load_state_dict(saved["model"])
            if m.kind == "cst":
                configure_native(m)
            o = make_optimizer(m)
            load_optimizer_snapshot(o, saved["optimizer"])
            for group in o.param_groups:
                group["lr"] = learning_rate(4883, 19531) * group.get("lr_scale", 1)
            before = parameter_digest(m)
            cached = {}
            if m.kind == "cst":
                for site in sites:
                    cpu = copy.deepcopy(projections(m)[site].map).cpu()
                    f = factor_partials(cpu, cpu.atoms.p.detach())
                    cached[site] = (
                        f[0].cuda(),
                        f[1].cuda(),
                        [x.cuda() for x in f[2]],
                        [x.cuda() for x in f[3]],
                    )
                    print("FACTOR CACHE PASS", arm, site, flush=True)
                    del f, cpu
            retained = []
            for offset in (0, 32, 64):
                batches = []
                for micro in range(4):
                    first = ((4882 + offset) * 4 + micro) * 2
                    batches.append(
                        token_batch(
                            data["train"], order[first : first + 2] * 128, 128, "cuda"
                        )
                    )
                rows, norm = captured_step(m, o, batches, 4883, sites)
                for site in sites:
                    row = rows[site]
                    row["global_gradient_norm"] = norm
                    result = analyze_map(
                        projections(m)[site],
                        row,
                        m.kind,
                        cached.get(site),
                        offset == 0,
                        args.extended_analysis,
                    )
                    result.update(
                        arm=arm,
                        site=site,
                        batch_offset_steps=offset,
                        checkpoint_sha256=s["milestone_checkpoint_sha256"][
                            checkpoint.name
                        ],
                    )
                    summary["records"].append(result)
                    save()
                    print("ATTRIBUTION PASS", arm, site, offset, flush=True)
                assert parameter_digest(m) == before
                assert all(
                    int(state["step"]) == saved["actual_adam_updates"]
                    for state in o.state.values()
                )
                if args.extended_analysis:
                    retained.append((offset, rows))
                del rows, batches
                torch.cuda.empty_cache()
            if args.extended_analysis:
                for site in sites:
                    cross = cross_batch_analysis(
                        projections(m)[site], retained, site, m.kind, cached.get(site)
                    )
                    cross.update(arm=arm, site=site)
                    summary["cross_batch"].append(cross)
                    save()
            assert sha(checkpoint) == s["milestone_checkpoint_sha256"][checkpoint.name]
            del m, o, saved, cached, retained
            torch.cuda.empty_cache()
        assert len(summary["records"]) == 21
        summary.update(state="passed", checkpoint_files_unchanged=True)
        save()
    except BaseException as e:
        summary.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main()
