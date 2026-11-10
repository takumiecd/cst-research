"""M8 capacity-only follow-up; prior checked trajectories remain external controls."""

from dataclasses import replace

from .ffn_dst_run import config
from .ffn_model import configure_native
from .ffn_selective import SITE_SETS, make_model, make_optimizer, projections

ARMS = {"m8-c8": 8, "m8-c16": 16}


def optimizer_for(model):
    return make_optimizer(model)


def model_for(arm, device="cuda"):
    from torchcst import FixedSelector, make_algorithm_registry
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_c16 import (
        RADIUS_ENVELOPE_WIDE_AMPLITUDE_C16_LM,
        RadiusEnvelopeWideAmplitudeC16LMAlgorithm,
    )

    cfg = replace(config(), coefficient=ARMS[arm])
    model = make_model(cfg, "cst", SITE_SETS["all"]).to(device)
    if device == "cuda":
        if ARMS[arm] == 8:
            configure_native(model)
        else:
            registry = make_algorithm_registry()
            registry.register(RadiusEnvelopeWideAmplitudeC16LMAlgorithm())
            for p in projections(model).values():
                p.map.selector = FixedSelector(
                    RADIUS_ENVELOPE_WIDE_AMPLITUDE_C16_LM, registry=registry
                )
    model.study_arm = arm
    return model
