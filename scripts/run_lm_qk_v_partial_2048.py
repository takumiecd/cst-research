"""Compare QK-only and V-only CST projections at width 2048."""

import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import run_lm_fp32_seed_replication as fp32
import run_lm_qkv_2048 as base
import run_lm_qkv_down_2048 as pilot
import run_lm_qkv_local_sweep as sweep
import run_lm_qkv_role_budget as roles
import torch
from torch import nn
from torchcst import init, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
    RadiusEnvelopeDenseFourierWeightLMAlgorithm,
)

PILOT_SHA = "6e45ab1380e79682da48be606f8014dc4ffa08749e7b836e74a04abf66956d22"
SEED = 17
ACTIVE = {"qk-only": {"query": 48, "key": 48}, "v-only": {"value": 64}}
PROBE_ATOMS = {
    arm: sweep.WIDTH * next(iter(replaced.values())) for arm, replaced in ACTIVE.items()
}


class PartialQKV(nn.Module):
    def __init__(self, original, replacements, dense_slice):
        super().__init__()
        self.replaced = nn.ModuleDict(replacements)
        weights = original.weight[dense_slice]
        bias = original.bias[dense_slice]
        self.ordinary = nn.Linear(
            original.in_features,
            len(bias),
            device=weights.device,
            dtype=weights.dtype,
        )
        with torch.no_grad():
            self.ordinary.weight.copy_(weights)
            self.ordinary.bias.copy_(bias)
        assert torch.equal(self.ordinary.weight, weights)
        assert torch.equal(self.ordinary.bias, bias)
        self.ordinary_first = dense_slice.start == 0

    def forward(self, x):
        ordinary = self.ordinary(x)
        selected = [projection(x) for projection in self.replaced.values()]
        parts = (ordinary, *selected) if self.ordinary_first else (*selected, ordinary)
        return torch.cat(parts, dim=-1)


def build_model(reference, dense, arm, seed):
    model = copy.deepcopy(dense)
    sites = {}
    if arm == "dense":
        return model, sites

    coefficients = ACTIVE[arm]
    width = model.config.width
    dense_slice = (
        slice(2 * width, 3 * width) if arm == "qk-only" else slice(0, 2 * width)
    )
    registry = make_algorithm_registry()
    registry.register(RadiusEnvelopeDenseFourierWeightLMAlgorithm())
    with patch.object(base.radius_envelope, "USE_RATIONAL_LOWER", False):
        for index, block in enumerate(model.blocks):
            replacements = {
                role: roles.cst_projection(
                    reference, model.config, seed, index, role, coefficient
                )
                for role, coefficient in coefficients.items()
            }
            block.qkv = PartialQKV(block.qkv, replacements, dense_slice)
            for role, projection in replacements.items():
                roles.configure_projection(projection.map, coefficients[role], registry)
                sites[f"{index}.{role}"] = projection.map
    assert len(sites) == len(coefficients) * len(model.blocks)
    for name, parameter in model.named_parameters():
        if ".qkv." not in name:
            assert torch.equal(parameter, dense.get_parameter(name)), name
    model.kind = "cst"
    return model, sites


def main(args):
    assert fp32.sha(args.pilot_summary) == PILOT_SHA
    old = json.loads(args.pilot_summary.read_text())
    assert old["state"] == "completed"
    assert old["identity"]["extension"]["qkv_coefficients"] == [48, 48, 64]
    controls = old["seeds"][str(SEED)]
    for key in ("dense", "qkv"):
        assert controls[key]["checkpoint_audit"] == "passed"
        assert controls[key]["completed_steps"] == 4882
    prepare = base.prepare
    load_native = fp32.load_native

    def verify_native():
        native, digest = load_native()
        assert digest == old["identity"]["native_sha256"]
        return native, digest

    def prepare_matched(reference, dense, arm, seed):
        assert seed == SEED
        assert (
            reference.parameter_digest(dense)
            == controls["dense"]["initial_parameters_sha256"]
        )
        with patch.object(init, "variance_scaling_", sweep.gpu_variance_scaling):
            model, sites, metadata = prepare(reference, dense, arm, seed)
        if arm != "dense":
            for site, layer in sites.items():
                digest = reference.parameter_digest(layer)
                assert (
                    digest
                    == controls["qkv"]["sites"][site]["initial_parameters_sha256"]
                )
                metadata["sites"][site]["initial_parameters_sha256"] = digest
            metadata["matched_dense_slices"] = [
                role for role in roles.ROLES if role not in ACTIVE[arm]
            ]
            print(
                "MATCHED_INITIALIZATION",
                arm,
                len(sites),
                metadata["matched_dense_slices"],
                flush=True,
            )
        return model, sites, metadata

    identity = {
        "study": "lm-qk-v-partial-2048-20261118",
        "driver_sha256": fp32.sha(Path(__file__)),
        "controller_sha256": fp32.sha(Path(pilot.__file__)),
        "allocation_builder_sha256": fp32.sha(Path(roles.__file__)),
        "initialization_adapter_sha256": fp32.sha(Path(sweep.__file__)),
        "pilot_summary_sha256": PILOT_SHA,
        "partial_coefficients": ACTIVE,
    }
    with (
        patch.object(fp32, "load_native", verify_native),
        patch.object(sweep, "COUNTS", PROBE_ATOMS),
        patch.multiple(
            base,
            TORCHCST_COMMIT=pilot.TORCHCST_COMMIT,
            MANIFEST_SHA=pilot.MANIFEST_SHA,
            COUNTS=PROBE_ATOMS,
            SCHEDULE=((SEED, tuple(ACTIVE)),),
            model_for=build_model,
            prepare=prepare_matched,
        ),
    ):
        base.main(args, extra_identity=identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
