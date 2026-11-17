"""Measure V-heavy width-2048 QKV allocations at two Atom budgets."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import run_lm_fp32_seed_replication as fp32
import run_lm_qkv_2048 as base
import run_lm_qkv_local_sweep as sweep
import run_lm_qkv_role_budget as roles

CONTROL = "qkv-c64"
V_HEAVY = "q56-k56-v80"
SMALL = "q48-k48-v64"
ALLOCATIONS = {CONTROL: (64, 64, 64), V_HEAVY: (56, 56, 80), SMALL: (48, 48, 64)}
SCHEDULE = (
    (17, (V_HEAVY, CONTROL, SMALL)),
    (23, (CONTROL, SMALL, V_HEAVY)),
    (42, (SMALL, V_HEAVY, CONTROL)),
)


def main(args):
    assert fp32.sha(args.paired_anchor) == sweep.PAIRED_ANCHOR_SHA
    anchor = json.loads(args.paired_anchor.read_text())
    assert anchor["state"] == "completed" and anchor["identity"]["steps"] == 4882
    assert all(
        anchor["seeds"][str(seed)][CONTROL]["state"] == "completed"
        for seed, _ in SCHEDULE
    )

    uniform_model = base.model_for

    def build_model(reference, dense, arm, seed):
        if arm in ("dense", CONTROL):
            return uniform_model(reference, dense, arm, seed)
        return roles.build_model(reference, dense, arm, seed, allocations=ALLOCATIONS)

    probe_counts = {
        arm: coefficients[0] * sweep.WIDTH for arm, coefficients in ALLOCATIONS.items()
    }
    identity = {
        "study": "lm-qkv-v-allocation-20261117",
        "driver_sha256": fp32.sha(Path(__file__)),
        "allocation_builder_sha256": fp32.sha(Path(roles.__file__)),
        "initialization_adapter_sha256": fp32.sha(Path(sweep.__file__)),
        "paired_anchor_sha256": sweep.PAIRED_ANCHOR_SHA,
        "roles": list(roles.ROLES),
        "role_coefficients": {arm: list(cs) for arm, cs in ALLOCATIONS.items()},
        "base_count_scope": "First Q projection used for the independent output and gradient audit.",
    }
    with patch.multiple(
        base,
        TORCHCST_COMMIT=sweep.TORCHCST_COMMIT,
        MANIFEST_SHA=sweep.MANIFEST_SHA,
        COUNTS=probe_counts,
        SCHEDULE=SCHEDULE,
        model_for=build_model,
        prepare=sweep.make_preparer(anchor),
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
