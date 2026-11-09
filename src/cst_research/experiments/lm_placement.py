"""Fixed-count connected M16 FFN placement experiment; no runtime changes."""

from dataclasses import replace
from itertools import combinations

import torch
from torchcst import ChebyshevSpec, CSTLinear, FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.connected_cell_mass_coalesced.envelope import (
    CONNECTED_ORDERED_ENVELOPE,
    ConnectedOrderedEnvelopeAlgorithm,
)
from torchcst.kernels.spec import RadiusConnectionSpec

from .ffn_dst_run import config
from .ffn_model import parameter_digest
from .ffn_selective import make_model, projections

PAIRS = {
    f"pair{i}{j}": tuple(f"{b}.{d}" for b in (i, j) for d in ("up", "down"))
    for i, j in combinations(range(4), 2)
}
MATCHED = PAIRS | {
    "up-even": ("0.up", "1.down", "2.up", "3.down"),
    "down-even": ("0.down", "1.up", "2.down", "3.up"),
}
ARMS = (
    {"dense": ("dense", ())}
    | {name: ("cst", sites) for name, sites in MATCHED.items()}
    | {
        "all-cst": ("cst", tuple(f"{i}.{d}" for i in range(4) for d in ("up", "down"))),
        "dst-pair12": ("dst", PAIRS["pair12"]),
    }
)
STEPS = 5000000 // 1024
EVALUATIONS = tuple(sorted({0, 256, 976, 2441, STEPS, *range(200, STEPS, 200)}))
SIGNALS = (0, 976, 2441, STEPS)
OFFSETS = (0, 4096, 8192, 12288)


def model_for(arm, device="cuda"):
    kind, sites = ARMS[arm]
    model = make_model(config(), kind, sites)
    initial = parameter_digest(model)
    if kind == "cst":
        for name in sites:
            projection = projections(model)[name]
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
            projection.map.initialization_report = old.initialization_report
        assert initial == parameter_digest(model)
    model = model.to(device)
    if kind == "cst" and device == "cuda":
        registry = make_algorithm_registry()
        registry.register(ConnectedOrderedEnvelopeAlgorithm())
        for name in sites:
            projections(model)[name].map.selector = FixedSelector(
                CONNECTED_ORDERED_ENVELOPE, registry=registry
            )
    model.study_arm = arm
    return model
