"""M8 Atom-budget line search with dense-mask, active-count-matched DST."""

from dataclasses import asdict, dataclass

import torch

from . import ffn_dst
from .ffn_contract import ModelConfig, atom_count
from .ffn_dst_run import config
from .ffn_model import LanguageModel, parameter_seed
from .ffn_selective import SITE_SETS, make_model, make_optimizer, projections

COEFFICIENTS = (16, 24, 32, 48, 64, 96, 128)
SPARSE_LRS = (0.003, 0.01, 0.03)
ARMS = (
    {"dense": ("dense", 16, None)}
    | {f"m8-c{c}": ("cst", c, None) for c in COEFFICIENTS}
    | {f"dst-c{c}-lr{lr:g}": ("dst", c, lr) for c in COEFFICIENTS for lr in SPARSE_LRS}
)


@dataclass(frozen=True)
class AtomBudgetConfig(ModelConfig):
    """Admit only this pilot's Atom budgets, keeping ordinary config guards intact."""

    def __post_init__(self):
        ModelConfig(**(asdict(self) | {"coefficient": 16}))
        if type(self.coefficient) is not int or self.coefficient not in COEFFICIENTS:
            raise ValueError("unsupported pilot Atom coefficient")


def model_for(arm, device="cuda"):
    kind, coefficient, _ = ARMS[arm]
    cfg = AtomBudgetConfig(**(asdict(config()) | {"coefficient": coefficient}))
    if kind == "cst":
        from torchcst import FixedSelector, make_algorithm_registry
        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_budget import (
            RADIUS_ENVELOPE_WIDE_AMPLITUDE_BUDGET_LM,
            RadiusEnvelopeWideAmplitudeBudgetLMAlgorithm,
        )

        m = make_model(cfg, "cst", SITE_SETS["all"]).to(device)
        if device == "cuda":
            registry = make_algorithm_registry()
            registry.register(RadiusEnvelopeWideAmplitudeBudgetLMAlgorithm())
            for p in projections(m).values():
                p.map.selector = FixedSelector(
                    RADIUS_ENVELOPE_WIDE_AMPLITUDE_BUDGET_LM, registry=registry
                )
        m.study_arm = arm
        return m
    m = LanguageModel(cfg, "dense", 17)
    if kind == "dense":
        m.ffn_selected_sites = ()
        m.study_arm = arm
        return m.to(device)
    for i, block in enumerate(m.blocks):
        for direction in ("up", "down"):
            projection = getattr(block.ffn, direction)
            old = projection.map
            projection.map = ffn_dst.MaskedLinear(
                old.weight,
                atom_count(old.in_features, old.out_features, coefficient),
                torch.Generator().manual_seed(
                    parameter_seed(17, f"block{i}.{direction}.dst-mask")
                ),
            )
    m.kind = "dst"
    m.ffn_selected_sites = SITE_SETS["all"]
    m.study_arm = arm
    return m.to(device)


def optimizer_for(model):
    if model.kind != "dst":
        return make_optimizer(model)
    return ffn_dst.make_optimizer(model, ARMS[model.study_arm][2])


def active_slots(model):
    if model.kind == "dense":
        return {}
    expected = atom_count(512, 2048, model.config.coefficient)
    if model.kind == "cst":
        return {
            f"{i}.{d}": int(getattr(b.ffn, d).map.atoms.p.shape[0])
            for i, b in enumerate(model.blocks)
            for d in ("up", "down")
        }
    result = {
        name: int(layer.mask.sum()) for name, layer in ffn_dst.layers(model).items()
    }
    assert len(result) == 8 and all(count == expected for count in result.values())
    return result
