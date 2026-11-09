"""Short task learning with matched raw envelope rows and explicit M16 opt-in."""

from dataclasses import replace

import torch
from torchcst import ChebyshevSpec, CSTLinear, FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.connected_cell_mass_coalesced.envelope import (
    CONNECTED_ENVELOPE,
    ConnectedEnvelopeAlgorithm,
)
from torchcst.kernels.spec import RadiusConnectionSpec

from .ffn_dst_run import config
from .ffn_model import configure_native, parameter_digest
from .ffn_selective import SITE_SETS, projections
from .ffn_selective import make_model as selective_model

ARMS = ("dense", "dst-all", "legacy-m8", "connected-m16")
STEPS = 976
SCHEDULE_STEPS = 20000000 // 1024
EVALUATIONS = (0, 200, 256, 400, 600, 800, 976)
OBSERVATIONS = (1, 256, 976)
SITES = ("1.up", "1.down")


def model_for(arm, device="cuda"):
    if arm not in ARMS:
        raise ValueError("unknown fixed arm")
    kind = "dense" if arm == "dense" else "dst" if arm == "dst-all" else "cst"
    model = selective_model(config(), kind, () if kind == "dense" else SITE_SETS["all"])
    initial = parameter_digest(model)
    if arm == "connected-m16":
        for projection in projections(model).values():
            old = projection.map
            spec = replace(
                old.kernel.spec,
                revision=10,
                profiles=tuple(
                    replace(
                        p,
                        profile=ChebyshevSpec(modes=16),
                        connection=RadiusConnectionSpec(),
                    )
                    for p in old.kernel.spec.profiles
                ),
            )
            projection.map = CSTLinear(
                *old.cst_charts(),
                atoms=old.atoms.p.detach().clone(),
                kernel=spec,
                dtype=torch.float32,
                backend="factored",
            )
            projection.map.initialization_report = {
                "method": "exact raw rows copied from legacy M8 variance initializer",
                "source": old.initialization_report,
                "not_recalibrated_to_m16": True,
            }
        assert parameter_digest(model) == initial
    model = model.to(device)
    if kind == "cst" and device == "cuda":
        if arm == "legacy-m8":
            configure_native(model)
        else:
            registry = make_algorithm_registry()
            registry.register(ConnectedEnvelopeAlgorithm())
            for projection in projections(model).values():
                projection.map.selector = FixedSelector(
                    CONNECTED_ENVELOPE, registry=registry
                )
    model.study_arm = arm
    return model


def set_rate(optimizer, step):
    from .ffn_contract import learning_rate

    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step, SCHEDULE_STEPS) * group.get("lr_scale", 1.0)


def step(model, optimizer, batches, clock):
    from . import ffn_dst

    model.train()
    optimizer.zero_grad(set_to_none=True)
    set_rate(optimizer, clock)
    loss_sum = 0.0
    for x, y in batches:
        _, loss = model(x, y)
        (loss / len(batches)).backward()
        loss_sum += float(loss.detach())
    if model.kind == "dst":
        event = ffn_dst.finish_step(model, optimizer, clock, SCHEDULE_STEPS)
    else:
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        event = {"gradient_norm_before_clip": float(norm), "topology": None}
    return loss_sum / len(batches), event


class Observation:
    """Only records the first microbatch's first 32 task rows; no state mutation."""

    def __init__(self, model):
        self.records = {}
        self.handles = []
        for name in SITES:
            module = projections(model)[name].map
            row = {
                "operator": module.execution_declaration(),
                "old_p": module.atoms.p.detach().cpu().clone(),
            }
            self.records[name] = row

            def hook(_module, inputs, output, row=row):
                if "x" in row:
                    return
                row["x"] = (
                    inputs[0]
                    .detach()
                    .reshape(-1, inputs[0].shape[-1])[:32]
                    .cpu()
                    .clone()
                )
                row["old_y_native"] = (
                    output.detach().reshape(-1, output.shape[-1])[:32].cpu().clone()
                )

                def gradient(g):
                    row["g"] = g.detach().reshape(-1, g.shape[-1])[:32].cpu().clone()

                output.register_hook(gradient)

            self.handles.append(module.register_forward_hook(hook))

    def finish(self, model):
        for h in self.handles:
            h.remove()
        for name, row in self.records.items():
            row["new_p"] = projections(model)[name].map.atoms.p.detach().cpu().clone()
            assert "g" in row
        return self.records


def physical_reference(operator):
    """FP64 physical factors with the original profile and connection declaration."""
    from torchcst import kernels

    spec = operator.kernel.parameterization
    kernel = kernels.product(
        *(replace(p, connection=None) for p in operator.kernel.profiles),
        coordinates=kernels.physical(
            radius_init=(spec.radius_min, spec.radius_max), radius_floor=spec.radius_min
        ),
    )
    if operator.kernel.revision == 10:
        kernel = replace(kernel, revision=9, profiles=operator.kernel.profiles)
    return CSTLinear(
        *operator.charts,
        atoms=1,
        kernel=kernel,
        dtype=torch.float64,
        backend="factored",
    )
