"""New six-case states: exact declared owners, native generation-only load boundary."""

from __future__ import annotations

from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_depth_study as stock
from . import higgs_linear_atom_state as qualified_state
from . import higgs_multilayer_conditioning as study
from .higgs_dst_comparison import json_sha

make_optimizer = stock.make_optimizer
core_state = qualified_state.core_state
check_optimizer = qualified_state.check_optimizer


def core_digest(checkpoint):
    """CanonicalSHA of COMPLETE typed core, including all row-ID metadata/moments."""
    return json_sha(exact.fingerprint(core_state(checkpoint)))


def snapshot(model, optimizer, updates, identity, *, epoch=0):
    return stock.checkpoint(
        model,
        optimizer,
        epoch,
        updates,
        {
            "case": model._conditioning_case,
            "model_configuration": model._conditioning_configuration,
            "gains": model.fixed_output_gains.detach().cpu().tolist(),
            "identity": identity,
        },
        None,
    )


def owners(model):
    return qualified_state.owner_versions(
        model, {"method": model._conditioning_configuration["method"]}
    )


def load(saved, device, *, cpu_reference=False):
    from benchmarks.training.higgs_verify import finite_fields, tensor_fields

    exact.require(
        not finite_fields(dict(tensor_fields(saved))),
        "nonfinite saved model/gradient/moments",
    )
    config = saved["configuration"]
    shape = config["model_configuration"]["factory_configuration"]
    model = study.make_model(
        config["case"],
        gains=config["gains"],
        width=shape["width"],
        hidden=shape["hidden_layers"],
    ).to(device)
    exact.assert_equal(
        model._conditioning_configuration,
        config["model_configuration"],
        "native-load-caseconfig",
    )
    optimizer, _ = make_optimizer(model, cpu_reference=cpu_reference)
    before = owners(model)
    for key in before:
        value = saved["model"][key]["layout_version"]
        exact.require(
            type(value) is int and value >= 0, "invalid saved native generation"
        )
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    for name, point in model.named_parameters():
        grad = saved["gradients"][name]
        point.grad = None if grad is None else grad.to(device).clone()
    expected = {
        key: max(value + 1, saved["model"][key]["layout_version"])
        for key, value in before.items()
    }
    exact.require(
        owners(model) == expected,
        "native max(before+1,saved) actual owner transition differs",
    )
    actual = snapshot(
        model, optimizer, saved["updates"], config["identity"], epoch=saved["epoch"]
    )
    exact.assert_equal(
        core_state(actual),
        qualified_state.with_generations(saved, expected),
        "native-load-all-other-core-bytes",
    )
    return (
        model,
        optimizer,
        core_state(actual),
        {
            "rule": "public975 max(before+1,saved) at exactactualCSTowners only; no counterwrite",
            "before": before,
            "serialized": {k: saved["model"][k]["layout_version"] for k in before},
            "loaded": expected,
            "actual_postload_core_sha256": core_digest(actual),
            "expected_only_native_generations_adjusted_core_sha256": core_digest(
                qualified_state.with_generations(saved, expected)
            ),
        },
    )
