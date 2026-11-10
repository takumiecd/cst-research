"""Owner criterion: one CST Atom versus one active DST mask position."""

from dataclasses import replace

import torch

from . import ffn_dst
from .ffn_contract import atom_count
from .ffn_dst_run import config
from .ffn_model import LanguageModel, parameter_seed
from .ffn_selective import SITE_SETS, make_optimizer
from .lm_m8_capacity import model_for as m8_model

SPARSE_LRS = (0.001, 0.003, 0.01)
ARMS = {"m8-c16": ("cst", 16, None)} | {
    f"dst-c{c}-lr{lr:g}": ("dst", c, lr) for c in (8, 16) for lr in SPARSE_LRS
}


def model_for(arm, device="cuda"):
    kind, coefficient, _ = ARMS[arm]
    if kind == "cst":
        return m8_model(arm, device=device)
    cfg = replace(config(), coefficient=coefficient)
    m = LanguageModel(cfg, "dense", 17)
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
    if model.kind == "cst":
        return make_optimizer(model)
    return ffn_dst.make_optimizer(model, ARMS[model.study_arm][2])


def active_slots(model):
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
