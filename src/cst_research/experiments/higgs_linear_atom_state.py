"""Typed full states and exact declared-owner native load generations for H4/H8."""

from __future__ import annotations

import copy

from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_depth_study as inherited
from . import higgs_linear_atom_scaling as study
from .higgs_depth_normalization_learning import core_state


def make_optimizer(model, *, cpu_reference=False):
    return inherited.make_optimizer(model, cpu_reference=cpu_reference)


def snapshot(model, optimizer, epoch, updates, case, identity):
    return inherited.checkpoint(
        model,
        optimizer,
        epoch,
        updates,
        {
            "case": case,
            "case_configuration": model._linear_atom_configuration,
            "identity": identity,
            "optimizer_recipe": "unchanged stock q1/public975 lr.0003",
        },
        None,
    )


def owner_versions(model, case):
    from torchcst.nn import CSTModule

    cfg = model._linear_atom_configuration
    expected = (
        set()
        if case["method"] == "dense"
        else {
            str(i) + ".atom_state._extra_state" for i in cfg["internal_module_indices"]
        }
    )
    owners = {
        name + ".atom_state._extra_state": site.atom_state.layout_version
        for name, site in model.named_modules()
        if isinstance(site, CSTModule)
    }
    exact.require(
        set(owners) == expected, "exact declared actual CST owner inventory differs"
    )
    exact.require(
        all(type(v) is int and v >= 0 for v in owners.values()),
        "invalid native generation",
    )
    return owners


def metadata_copy(value):
    """Copy comparison metadata only; never duplicate or mutate a tensor's bytes."""
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        return {key: metadata_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(metadata_copy(item) for item in value)
    return copy.deepcopy(value)


def with_generations(checkpoint, generations):
    result = metadata_copy(core_state(checkpoint))
    declared = {
        key for key in result["model"] if key.endswith(".atom_state._extra_state")
    }
    exact.require(
        declared == set(generations), "generation comparison owner paths differ"
    )
    for key, version in generations.items():
        fields = result["model"][key]
        exact.require(
            set(fields) == {"format_version", "shape", "layout_version"}
            and fields["format_version"] == 1
            and type(version) is int
            and version >= 0,
            "native atom schema differs",
        )
        fields["layout_version"] = version
    return result


def load_state(case, saved, device, *, cpu_reference=False):
    exact.require(saved["configuration"]["case"] == case, "checkpoint case differs")
    model = study.make_case(case).to(device)
    exact.assert_equal(
        saved["configuration"]["case_configuration"],
        model._linear_atom_configuration,
        "saved-model-configuration",
    )
    optimizer, _ = make_optimizer(model, cpu_reference=cpu_reference)
    before = owner_versions(model, case)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    for name, point in model.named_parameters():
        value = saved["gradients"][name]
        point.grad = None if value is None else value.to(device).clone()
    expected_versions = {}
    for key, current in before.items():
        old = saved["model"][key]["layout_version"]
        exact.require(type(old) is int and old >= 0, "invalid serialized generation")
        expected_versions[key] = max(current + 1, old)
    exact.require(
        owner_versions(model, case) == expected_versions,
        "native max(before+1,saved) generation transition differs",
    )
    actual = snapshot(
        model,
        optimizer,
        saved["epoch"],
        saved["updates"],
        case,
        saved["configuration"]["identity"],
    )
    exact.assert_equal(
        core_state(actual),
        with_generations(saved, expected_versions),
        "native-load-all-other-core-bytes",
    )
    return (
        model,
        optimizer,
        core_state(actual),
        {
            "rule": "unchanged public975 max(before+1,saved), only actualcfg owners; no counterwrite",
            "before": before,
            "serialized": {k: saved["model"][k]["layout_version"] for k in before},
            "loaded": expected_versions,
            "actual_postload_core": exact.fingerprint(core_state(actual)),
        },
    )


def check_optimizer(model, optimizer, updates):
    import torch
    from torchcst.nn import CSTModule

    parameters = list(model.parameters())
    expected = model._linear_atom_configuration["hidden_layers"] + 3
    owned = [point for group in optimizer.param_groups for point in group["params"]]
    exact.require(
        len(parameters) == expected
        and len(owned) == len({id(p) for p in owned}) == expected
        and {id(p) for p in owned} == {id(p) for p in parameters},
        "optimizer ownership mustcover exact H+3 points",
    )
    exact.require(
        set(optimizer.state) == set(parameters), "optimizer moment inventory differs"
    )
    for point in parameters:
        state = optimizer.state[point]
        exact.require(
            {"step", "exp_avg", "exp_avg_sq"} <= set(state)
            and float(state["step"]) == updates,
            "native point clock/moments differ",
        )
        for key in ("exp_avg", "exp_avg_sq"):
            exact.require(
                state[key].shape == point.shape
                and state[key].dtype == point.dtype
                and state[key].device == point.device
                and bool(torch.isfinite(state[key]).all()),
                "moment shape/dtype/device/finite differs",
            )
        exact.require(bool((state["exp_avg_sq"] >= 0).all()), "negative second moment")
    for site in model.modules():
        if isinstance(site, CSTModule):
            stored = site.atom_state.optimizer_state.fields
            for key in ("step", "exp_avg", "exp_avg_sq"):
                exact.require(key in stored, "model-side native moment missing")
                exact.assert_equal(
                    stored[key],
                    optimizer.state[site.atoms.p][key],
                    "model-side-optimizer-coherence/" + key,
                )
