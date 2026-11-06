"""One-step diagnostic with no optimizer override or module/backward hooks."""

from __future__ import annotations

from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_depth_normalization_steps as accepted_contract
from . import higgs_depth_study as stock
from . import higgs_multilayer_conditioning as study
from . import higgs_multilayer_conditioning_state as state


def stable_displacement(before, after, *, period=None):
    import torch

    delta = after.double() - before.double()
    encoded_changed = float((after != before).double().mean())
    if period is not None:
        # Preserve tiny already-in-range differences; never add halfperiod first.
        outside = delta.abs() > period / 2
        turns = torch.floor((delta + period / 2) / period)
        delta = torch.where(outside, delta - turns * period, delta)
    rms = float(before.double().square().mean().sqrt())
    return {
        **stock.tensor_statistics(delta),
        "quantiles": accepted_contract.quantiles(delta),
        "relative_RMS": float(delta.square().mean().sqrt()) / rms if rms else None,
        "encoded_changed_fraction": encoded_changed,
        "physical_nonzero_fraction": float((delta != 0).double().mean()),
        "minimum_torus": period is not None,
        "period": period,
        "halfperiod_ties": "retain signed original difference",
    }


def forward_backward(model, inputs, labels):
    import torch

    exact.require(
        torch.is_grad_enabled() and inputs.shape == (4, 28) and labels.shape == (4,),
        "grad-enabled actual B4 forward required",
    )
    for module in model.modules():
        exact.require(
            not module._forward_hooks
            and not module._forward_pre_hooks
            and not module._backward_hooks,
            "observation requires no module/backward hooks",
        )
    detailed = {}
    logits = study.trace(model, inputs, detailed=detailed, retain_grad=True)[-1]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    forward = {
        key: {
            **stock.tensor_statistics(value),
            "channel_RMS_quantiles": accepted_contract.quantiles(
                value.detach().double().square().mean(0).sqrt()
            ),
            "channels_any_nonzero_fraction": float(
                (value.detach() != 0).any(0).double().mean()
            ),
        }
        for key, value in detailed.items()
    }
    backward = {
        key: None if value.grad is None else stock.tensor_statistics(value.grad)
        for key, value in detailed.items()
    }
    sites = model._linear_atom_configuration["internal_module_indices"]
    relu = {
        str(i): {
            "rho": (forward[str(i + 1)]["rms"] / forward[str(i)]["rms"]) ** 2
            if forward[str(i)]["rms"]
            else None,
            "positive_fraction": 1 - forward[str(i + 1)]["zero_fraction"],
        }
        for i in sites
    }
    raw = {
        "inputs": inputs.detach().cpu().clone(),
        "labels": labels.detach().cpu().clone(),
        "logits": logits.detach().cpu().clone(),
        "task_CE": loss.detach().cpu(),
        "forward": {k: v.detach().cpu().clone() for k, v in detailed.items()},
        "backward": {
            k: None if v.grad is None else v.grad.detach().cpu().clone()
            for k, v in detailed.items()
        },
    }
    return (
        logits,
        loss,
        {
            "task_CE": float(loss.detach()),
            "forward": forward,
            "backward": backward,
            "relu": relu,
        },
        raw,
    )


def observed_step(
    model, optimizer, inputs, labels, indices, identity, *, step=1, epoch=0, totals=None
):
    exact.require(
        "_adamw_proposal" not in optimizer.__dict__
        and not optimizer._optimizer_step_pre_hooks
        and not optimizer._optimizer_step_post_hooks,
        "native optimizer no instance override/step hooks required",
    )
    optimizer.zero_grad(set_to_none=True)
    before = accepted_contract.parameters(model)
    logits, loss, summary, raw = forward_backward(model, inputs, labels)
    raw_gradients = accepted_contract.gradients(model)
    # No observer is inserted into native prepare/proposal/finish.
    optimizer.step()
    post_gradients = accepted_contract.gradients(model)
    exact.assert_equal(
        raw_gradients,
        post_gradients,
        "source-qualified-FlatTorus-identity-prepare/poststep-gradient",
    )
    state.check_optimizer(model, optimizer, step)
    accepted = state.snapshot(model, optimizer, step, identity, epoch=epoch)
    fields = {}
    for name, point in model.named_parameters():
        group = next(
            g for g in optimizer.param_groups if any(q is point for q in g["params"])
        )
        _, denom, sqrt_vhat, contract = accepted_contract.adam_denominator(
            optimizer.state[point], group
        )
        contract.update(
            {
                "role": group.get("role"),
                "amplitude_weight_decay": group.get("amplitude_weight_decay", 0),
                "scope": "accepted-state/source-opmath reporting reconstruction; canonicalFlatTorus momenttransport identity; NOT capturednativeintermediate",
            }
        )
        rows = {}
        for key, previous in accepted_contract.fields(name, before[name]).items():
            gradient = accepted_contract.fields(name, raw_gradients[name])[key]
            rootv = accepted_contract.fields(name, sqrt_vhat.detach().cpu())[key]
            final = accepted_contract.fields(name, accepted["model"][name])[key]
            active = gradient != 0
            rows[key] = {
                "actual_raw_task_gradient": stock.tensor_statistics(gradient),
                "accepted_m": stock.tensor_statistics(
                    accepted_contract.fields(name, optimizer.state[point]["exp_avg"])[
                        key
                    ]
                ),
                "accepted_v": stock.tensor_statistics(
                    accepted_contract.fields(
                        name, optimizer.state[point]["exp_avg_sq"]
                    )[key]
                ),
                "denominator_quantiles": accepted_contract.quantiles(
                    accepted_contract.fields(name, denom)[key]
                ),
                "sqrt_vhat_over_eps_quantiles": accepted_contract.quantiles(
                    rootv / contract["eps"]
                ),
                "fraction_sqrt_vhat_lt_eps_all": float(
                    (rootv < contract["eps"]).double().mean()
                ),
                "actual_FP32_nonzero_gradient_elements": int(active.sum()),
                "fraction_sqrt_vhat_lt_eps_on_nonzero_gradient": float(
                    (rootv[active] < contract["eps"]).double().mean()
                )
                if bool(active.any())
                else None,
                "accepted_displacement_including_retraction_and_amp_decay": stable_displacement(
                    previous,
                    final,
                    period=model._linear_atom_configuration["width"]
                    if key.startswith("mu_")
                    else None,
                ),
            }
        fields[name] = {"actual_group": contract, "fields": rows}
    summary.update(
        {
            "step": step,
            "case": model._conditioning_case,
            "parameters": fields,
            "no_optimizer_monkeypatch_or_backward_hooks": True,
            "raw_poststep_gradients_exact": True,
            "prepared_gradient_scope": "canonicalFlatTorus sourceidentity and actualraw/poststepbyteequality; no prepared-phase buffer capture",
            "denominator_scope": "accepted-state reporting reconstruction scalar-qualified separately, not actualnativebuffer",
            "accepted_delta_scope": "includes native retraction/amplitudeWD; no proposalbuffer attribution",
        }
    )
    raw.update(
        {
            "indices": indices.detach().cpu().clone(),
            "before_parameters": before,
            "raw_gradients": raw_gradients,
            "poststep_gradients": post_gradients,
            "accepted": accepted,
            "summary": summary,
        }
    )
    if totals is not None:
        totals[0].add_(loss.detach() * 4)
        totals[1].add_((logits.detach().argmax(1) == labels).sum())
    return summary, raw


def endpoint_probe(saved, inputs, labels, indices, device):
    """Independent loaded copy; backward only, never an optimizer update on source."""
    model, optimizer, baseline, load_proof = state.load(saved, device)
    optimizer.zero_grad(set_to_none=True)
    _, _, summary, raw = forward_backward(model, inputs, labels)
    actual = state.snapshot(
        model,
        optimizer,
        saved["updates"],
        saved["configuration"]["identity"],
        epoch=saved["epoch"],
    )
    exact.assert_equal(
        {k: v for k, v in state.core_state(actual).items() if k != "gradients"},
        {k: v for k, v in baseline.items() if k != "gradients"},
        "endpoint-probe-only-declared-copy-gradients-may-change",
    )
    raw.update(
        {
            "indices": indices.detach().cpu().clone(),
            "probe_gradients": accepted_contract.gradients(model),
            "checkpoint_updates": saved["updates"],
            "native_load": load_proof,
            "optimizer_updates_performed": 0,
            "original_checkpoint_model_optimizer_grads_not_modified": True,
        }
    )
    return summary, raw
