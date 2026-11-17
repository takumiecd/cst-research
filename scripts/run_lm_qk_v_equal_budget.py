"""Add the missing QK32 and V96 arms for two equal-Atom comparisons."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import run_lm_fp32_seed_replication as fp32
import run_lm_qk_v_partial_2048 as partial
import run_lm_qkv_2048 as base
import run_lm_qkv_local_sweep as sweep
import run_lm_qkv_role_budget as roles
from torchcst import init

TORCHCST_COMMIT = "b515dd4345343296386715802e32e107bbbaf45c"
MANIFEST_SHA = "8a7bd80c95090353e57384ffe3237083fbe35ffc584a4825df580f857f13f849"
ACTIVE = {"qk-only": {"query": 32, "key": 32}, "v-only": {"value": 96}}
COUNTS = {
    arm: sweep.WIDTH * next(iter(coefficients.values()))
    for arm, coefficients in ACTIVE.items()
}
SCHEDULE = (
    (17, ("qk-only", "v-only")),
    (23, ("v-only", "qk-only")),
    (42, ("qk-only", "v-only")),
)


def main(args):
    assert fp32.sha(args.dense_anchor) == sweep.PAIRED_ANCHOR_SHA
    dense_anchor = json.loads(args.dense_anchor.read_text())
    assert dense_anchor["state"] == "completed"
    prepare = base.prepare
    dense_checks = set()

    def prepare_matched(reference, dense, arm, seed):
        if seed not in dense_checks:
            expected = dense_anchor["seeds"][str(seed)]["dense"]
            assert (
                reference.parameter_digest(dense)
                == expected["initial_parameters_sha256"]
            )
            dense_checks.add(seed)
        with patch.object(init, "variance_scaling_", sweep.gpu_variance_scaling):
            model, sites, metadata = prepare(reference, dense, arm, seed)
        if sites:
            for site, layer in sites.items():
                metadata["sites"][site]["initial_parameters_sha256"] = (
                    reference.parameter_digest(layer)
                )
            metadata["matched_dense_slices"] = [
                role for role in roles.ROLES if role not in ACTIVE[arm]
            ]
        return model, sites, metadata

    identity = {
        "study": "lm-qk-v-equal-budget-2048-20261118",
        "driver_sha256": fp32.sha(Path(__file__)),
        "partial_driver_sha256": fp32.sha(Path(partial.__file__)),
        "allocation_builder_sha256": fp32.sha(Path(roles.__file__)),
        "initialization_adapter_sha256": fp32.sha(Path(sweep.__file__)),
        "dense_anchor_sha256": sweep.PAIRED_ANCHOR_SHA,
        "partial_coefficients": ACTIVE,
        "comparisons": [
            {"total_coefficient": 64, "query_key": [32, 32], "value": 64},
            {"total_coefficient": 96, "query_key": [48, 48], "value": 96},
        ],
    }
    with (
        patch.object(sweep, "COUNTS", COUNTS),
        patch.object(partial, "ACTIVE", ACTIVE),
        patch.multiple(
            base,
            TORCHCST_COMMIT=TORCHCST_COMMIT,
            MANIFEST_SHA=MANIFEST_SHA,
            COUNTS=COUNTS,
            SCHEDULE=SCHEDULE,
            model_for=partial.build_model,
            prepare=prepare_matched,
        ),
    ):
        base.main(args, extra_identity=identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--dense-anchor", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
