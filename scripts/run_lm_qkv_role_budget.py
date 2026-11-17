"""Compare three allocations of the same Atom budget in width-2048 QKV."""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_lm_fp32_seed_replication as fp32
import run_lm_qkv_2048 as base
import run_lm_qkv_local_sweep as sweep
import torch
from torch import nn
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
    RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM,
    RadiusEnvelopeDenseFourierWeightLMAlgorithm,
)

ROLES = ("query", "key", "value")
ALLOCATIONS = {
    "q128-k64-v64": (128, 64, 64),
    "q64-k128-v64": (64, 128, 64),
    "q64-k64-v128": (64, 64, 128),
}
EXPERIMENT_COUNTS = {"c64": 64 * sweep.WIDTH, "c128": 128 * sweep.WIDTH}
SCHEDULE = ((17, ("q64-k64-v128", "q128-k64-v64", "q64-k128-v64")),)


def cst_projection(reference, config, seed, index, role, coefficient):
    projection_config = SimpleNamespace(
        **(asdict(config) | {"coefficient": coefficient})
    )
    return reference.Projection(
        config.width,
        config.width,
        config=projection_config,
        kind="cst",
        std=config.init_std,
        seed=reference.parameter_seed(seed, f"block{index}.attention.{role[0]}"),
    )


def configure_projection(layer, coefficient, registry):
    assert len(layer.atoms.p) == coefficient * sweep.WIDTH
    assert not torch.count_nonzero(layer.atoms.p[:, 1])
    assert layer.execution_declaration().kernel.parameterization.radius_max == 512
    layer.selector = FixedSelector(
        RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM, registry=registry
    )


class SplitQKV(nn.Module):
    def __init__(self, reference, config, seed, index, coefficients):
        super().__init__()
        for role, coefficient in zip(ROLES, coefficients, strict=True):
            setattr(
                self,
                role,
                cst_projection(reference, config, seed, index, role, coefficient),
            )

    def forward(self, x):
        return torch.cat((self.query(x), self.key(x), self.value(x)), dim=-1)


def build_model(reference, dense, arm, seed, *, allocations=ALLOCATIONS):
    model = copy.deepcopy(dense)
    sites = {}
    if arm == "dense":
        return model, sites

    coefficients = allocations[arm]
    registry = make_algorithm_registry()
    registry.register(RadiusEnvelopeDenseFourierWeightLMAlgorithm())
    with patch.object(base.radius_envelope, "USE_RATIONAL_LOWER", False):
        for index, block in enumerate(model.blocks):
            block.qkv = SplitQKV(reference, model.config, seed, index, coefficients)
            for role, coefficient in zip(ROLES, coefficients, strict=True):
                layer = getattr(block.qkv, role).map
                configure_projection(layer, coefficient, registry)
                sites[f"{index}.{role}"] = layer

    assert len(sites) == 12
    expected_atoms = len(model.blocks) * sweep.WIDTH * sum(coefficients)
    assert sum(len(site.atoms.p) for site in sites.values()) == expected_atoms
    for name, parameter in model.named_parameters():
        if ".qkv." not in name:
            assert torch.equal(parameter, dense.get_parameter(name)), name
    model.kind = "cst"
    return model, sites


def main(args):
    assert fp32.sha(args.paired_anchor) == sweep.PAIRED_ANCHOR_SHA
    anchor = json.loads(args.paired_anchor.read_text())
    assert anchor["state"] == "completed" and anchor["identity"]["steps"] == 4882

    # The inherited qualification probes the first layer, which is a Q layer.
    probe_counts = {
        arm: coefficients[0] * sweep.WIDTH for arm, coefficients in ALLOCATIONS.items()
    }
    identity = {
        "study": "lm-qkv-role-budget-20261117",
        "role_budget_driver_sha256": fp32.sha(Path(__file__)),
        "initialization_adapter_sha256": fp32.sha(Path(sweep.__file__)),
        "paired_anchor_sha256": sweep.PAIRED_ANCHOR_SHA,
        "roles": list(ROLES),
        "role_coefficients": {k: list(v) for k, v in ALLOCATIONS.items()},
        "total_atoms_per_model": 2097152,
        "base_count_scope": "First projection probed in the full output and gradient audit.",
    }
    with (
        patch.multiple(
            base,
            TORCHCST_COMMIT=sweep.TORCHCST_COMMIT,
            MANIFEST_SHA=sweep.MANIFEST_SHA,
            COUNTS=probe_counts,
            SCHEDULE=SCHEDULE,
            model_for=build_model,
            prepare=sweep.make_preparer(anchor),
        ),
        patch.object(sweep, "COUNTS", EXPERIMENT_COUNTS),
    ):
        base.main(args, extra_identity=identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--paired-anchor", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
