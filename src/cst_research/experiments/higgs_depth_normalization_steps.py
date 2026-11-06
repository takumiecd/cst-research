"""Observation-only native-step recorder. No numerical optimizer implementation."""

from __future__ import annotations

from . import higgs_depth_normalization as experiment
from . import higgs_depth_study as stock


def quantiles(value):
    import torch

    v = value.detach().cpu().double().reshape(-1)
    if not bool(torch.isfinite(v).all()):
        raise FloatingPointError("nonfinite diagnostic value")
    return dict(
        zip(
            ("min", "q25", "median", "q75", "q99", "max"),
            torch.quantile(
                v, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.99, 1.0], dtype=torch.float64)
            ).tolist(),
            strict=True,
        )
    )


def fields(name, value):
    if name.endswith("atoms.p"):
        return {
            label: value[:, i]
            for i, label in enumerate(("polar_s", "polar_t", "mu_output", "mu_input"))
        }
    return {"all": value}


def displacement(before, after, *, torus_period=None):
    delta = after.double() - before.double()
    if torus_period is not None:
        delta = (delta + torus_period / 2).remainder(torus_period) - torus_period / 2
    result = stock.tensor_statistics(delta)
    base = float(before.double().square().mean().sqrt())
    result["relative_RMS"] = result["rms"] / base if base else None
    result["before_RMS_exact_zero"] = base == 0
    result["minimum_signed_torus"] = torus_period is not None
    return result


def adam_denominator(state, group):
    """Explicit reporting reconstruction from native m/v/clock and actual group options.

    CUDA fused Adam uses FP32 opmath coefficients for FP32 parameters. This
    reconstruction is independently scalar-qualified, not an exposed native
    denominator buffer or an exact-bitwise kernel-intermediate claim.
    """
    import torch

    if group["amsgrad"] or group["maximize"] or not group["fused"]:
        raise ValueError(
            "diagnostic reconstruction requires qualified fused non-AMSGrad ordinary AdamW"
        )
    m, v, clock = state["exp_avg"], state["exp_avg_sq"], state["step"]
    if m.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("reporting reconstruction requires FP32 states")
    beta1, beta2 = group["betas"]
    b1, b2 = m.new_tensor(beta1), m.new_tensor(beta2)
    step = clock.to(m.device, dtype=m.dtype)
    bc1, bc2 = 1 - b1.pow(step), 1 - b2.pow(step)
    sqrt_vhat = (v / bc2).sqrt()
    denominator = sqrt_vhat + group["eps"]
    return (
        m / bc1,
        denominator,
        sqrt_vhat,
        {
            "step": float(clock),
            "beta1_config": beta1,
            "beta2_config": beta2,
            "beta1_opmath": float(b1),
            "beta2_opmath": float(b2),
            "bias_correction1": float(bc1),
            "bias_correction2": float(bc2),
            "eps": group["eps"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "semantics": "FP32-opmath reporting reconstruction from actual native state and actual group; scalar-qualified, not native buffer capture",
        },
    )


def parameters(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def gradients(model):
    return {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}


def moments(model, optimizer):
    return {
        name: stock.clone_cpu(optimizer.state[p])
        for name, p in model.named_parameters()
        if p in optimizer.state
    }


def observed_step(model, optimizer, x, labels, *, step, retain_raw=True):
    """One CE/backward/native optimizer update, with observations outside its arithmetic."""
    import torch

    optimizer.zero_grad(set_to_none=True)
    detailed = {}
    logits = experiment.trace(model, x, detailed=detailed)[-1]
    for value in detailed.values():
        if value.requires_grad:
            value.retain_grad()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    before = parameters(model)
    raw_gradients = gradients(model)
    forward = {
        key: {
            **stock.tensor_statistics(value),
            "channel_output_RMS_quantiles": quantiles(
                value.detach().double().square().mean(0).sqrt()
            ),
            "channels_with_any_nonzero_fraction": float(
                (value.detach() != 0).any(0).double().mean()
            ),
            "channels_with_any_positive_fraction": float(
                (value.detach() > 0).any(0).double().mean()
            ),
            "channel_scope": "current task B4 only; not permanent neuron activity",
        }
        for key, value in detailed.items()
    }
    backward = {
        key: None if value.grad is None else stock.tensor_statistics(value.grad)
        for key, value in detailed.items()
    }
    relu = {}
    for site in experiment.SITES:
        pre, post = forward[str(site)], forward[str(site + 1)]
        relu[str(site)] = {
            "rho": (post["rms"] / pre["rms"]) ** 2 if pre["rms"] else None,
            "pre_RMS_exact_zero": pre["rms"] == 0,
            "positive_fraction": 1 - post["zero_fraction"],
        }
    captured = {}

    def capture_proposal(original):
        captured["prepared_gradients"] = gradients(model)
        original()
        captured["proposal_parameters"] = parameters(model)
        captured["proposal_moments"] = moments(model, optimizer)

    if hasattr(optimizer, "_adamw_proposal"):
        if "_adamw_proposal" in optimizer.__dict__:
            raise ValueError("unexpected existing optimizer observation override")
        native = optimizer._adamw_proposal
        optimizer._adamw_proposal = lambda: capture_proposal(native)
        try:
            optimizer.step()
        finally:
            del optimizer.__dict__["_adamw_proposal"]
    else:
        capture_proposal(optimizer.step)
    accepted = parameters(model)
    if set(optimizer.state) != set(model.parameters()) or any(
        not bool(torch.isfinite(p).all()) for p in accepted.values()
    ):
        raise FloatingPointError(
            "native accepted parameters/optimizer ownership failed"
        )
    summaries = {}
    for name, p in model.named_parameters():
        group = next(
            g for g in optimizer.param_groups if any(q is p for q in g["params"])
        )
        state = captured["proposal_moments"][name]
        if float(state["step"]) != step:
            raise ValueError("native optimizer clock is not actual task update")
        _, denominator, sqrt_vhat, contract = adam_denominator(state, group)
        row = {"actual_group": contract, "fields": {}}
        for field, old in fields(name, before[name]).items():
            prepared = fields(name, captured["prepared_gradients"][name])[field]
            raw = fields(name, raw_gradients[name])[field]
            proposed = fields(name, captured["proposal_parameters"][name])[field]
            final = fields(name, accepted[name])[field]
            denom = fields(name, denominator)[field]
            rootv = fields(name, sqrt_vhat)[field]
            eps = contract["eps"]
            row["fields"][field] = {
                "raw_task_gradient": stock.tensor_statistics(raw),
                "actual_prepared_gradient": stock.tensor_statistics(prepared),
                "raw_prepared_byte_equal": torch.equal(raw, prepared),
                "actual_m": stock.tensor_statistics(
                    fields(name, state["exp_avg"])[field]
                ),
                "actual_v": stock.tensor_statistics(
                    fields(name, state["exp_avg_sq"])[field]
                ),
                "denominator_quantiles": quantiles(denom),
                "sqrt_vhat_over_eps_quantiles": quantiles(rootv / eps),
                "fraction_sqrt_vhat_lt_eps": float((rootv < eps).double().mean()),
                "proposal_displacement": displacement(old, proposed),
                "accepted_displacement": displacement(
                    old,
                    final,
                    torus_period=model[0].out_features
                    if field.startswith("mu_")
                    else None,
                ),
                "retraction_change": displacement(
                    proposed,
                    final,
                    torus_period=model[0].out_features
                    if field.startswith("mu_")
                    else None,
                ),
            }
        summaries[name] = row
    result = {
        "step": step,
        "task_CE": float(loss.detach()),
        "forward": forward,
        "backward": backward,
        "relu": relu,
        "parameters": summaries,
        "optimizer_type": type(optimizer).__module__
        + "."
        + type(optimizer).__qualname__,
        "raw_retained": retain_raw,
    }
    raw = None
    if retain_raw:
        raw = {
            "step": step,
            "inputs": x.detach().cpu(),
            "labels": labels.detach().cpu(),
            "logits": logits.detach().cpu(),
            "task_CE": loss.detach().cpu(),
            "forward": {k: v.detach().cpu().clone() for k, v in detailed.items()},
            "backward": {
                k: None if v.grad is None else v.grad.detach().cpu().clone()
                for k, v in detailed.items()
            },
            "before_parameters": before,
            "raw_gradients": raw_gradients,
            **captured,
            "accepted": stock.checkpoint(
                model, optimizer, 0, step, {"case": model._diagnostic_case}, None
            ),
        }
    return result, raw
