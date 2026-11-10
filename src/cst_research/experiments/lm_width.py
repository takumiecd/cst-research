"""Task models for the fixed-c64 width study; no new operator mathematics."""

import torch

from . import ffn_dst
from .ffn_model import LanguageModel, parameter_seed
from .ffn_selective import SITE_SETS, make_model, make_optimizer, projections
from .lm_width_contract import ARMS, WidthConfig, slots_per_map


def model_for(arm, width, device="cuda"):
    kind, _ = ARMS[arm]
    cfg = WidthConfig(width=width)
    if kind == "cst":
        from torchcst import FixedSelector, make_algorithm_registry
        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
            RADIUS_ENVELOPE_WIDE_AMPLITUDE_WIDTH_LM,
            RadiusEnvelopeWideAmplitudeWidthLMAlgorithm,
        )

        m = make_model(cfg, "cst", SITE_SETS["all"])
        m = m.to(device)
        if device == "cuda":
            registry = make_algorithm_registry()
            registry.register(RadiusEnvelopeWideAmplitudeWidthLMAlgorithm())
            for p in projections(m).values():
                p.map.selector = FixedSelector(
                    RADIUS_ENVELOPE_WIDE_AMPLITUDE_WIDTH_LM, registry=registry
                )
    else:
        m = LanguageModel(cfg, "dense", 17)
        m.ffn_selected_sites = ()
        if kind == "dst":
            for i, block in enumerate(m.blocks):
                for direction in ("up", "down"):
                    p = getattr(block.ffn, direction)
                    p.map = ffn_dst.MaskedLinear(
                        p.map.weight,
                        cfg.atoms_per_projection,
                        torch.Generator().manual_seed(
                            parameter_seed(17, f"block{i}.{direction}.dst-mask")
                        ),
                    )
            m.kind = "dst"
            m.ffn_selected_sites = SITE_SETS["all"]
        m = m.to(device)
    m.study_arm = arm
    return m


def optimizer_for(model):
    kind, rate = ARMS[model.study_arm]
    if kind == "dst":
        return ffn_dst.make_optimizer(model, rate)
    optimizer = make_optimizer(model)
    if kind == "dense":
        for group in optimizer.param_groups:
            group["lr_scale"] = rate / 0.0003
            group["lr"] = rate
    return optimizer


def active_slots(model):
    if model.kind == "dense":
        return {}
    expected = slots_per_map(model.config.width)
    result = (
        {n: len(p.map.atoms.p) for n, p in projections(model).items()}
        if model.kind == "cst"
        else {n: int(p.mask.sum()) for n, p in ffn_dst.layers(model).items()}
    )
    assert len(result) == 8 and set(result.values()) == {expected}
    return result
