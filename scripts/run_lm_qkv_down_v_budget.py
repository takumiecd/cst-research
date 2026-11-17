"""Replay the width-2048 combined arm with a larger V-heavy QKV budget."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import run_lm_fp32_seed_replication as fp32
import run_lm_qkv_2048 as base
import run_lm_qkv_down_2048 as pilot
import run_lm_qkv_local_sweep as sweep
import run_lm_qkv_role_budget as roles
from torchcst import init

PILOT_SHA = "6e45ab1380e79682da48be606f8014dc4ffa08749e7b836e74a04abf66956d22"
COEFFICIENTS = (56, 56, 80)
SEED = 17
ARM = "qkv-down"
PROBE_COUNTS = {ARM: COEFFICIENTS[0] * sweep.WIDTH}
INITIALIZATION_COUNTS = PROBE_COUNTS | {
    "value": COEFFICIENTS[2] * sweep.WIDTH,
    "down": pilot.DOWN_ATOMS,
}


def main(args):
    assert fp32.sha(args.paired_anchor) == sweep.PAIRED_ANCHOR_SHA
    assert fp32.sha(args.qkv_anchor) == pilot.QKV_ANCHOR_SHA
    assert fp32.sha(args.pilot_summary) == PILOT_SHA
    dense_anchor = json.loads(args.paired_anchor.read_text())
    qkv_anchor = json.loads(args.qkv_anchor.read_text())
    previous = json.loads(args.pilot_summary.read_text())
    assert previous["state"] == "completed"
    prior_identity = previous["identity"]
    old = previous["seeds"][str(SEED)][ARM]
    assert old["checkpoint_audit"] == "passed" and old["completed_steps"] == 4882
    assert prior_identity["extension"]["qkv_coefficients"] == [48, 48, 64]
    assert prior_identity["extension"]["atoms_per_down_projection"] == pilot.DOWN_ATOMS
    expected_qkv = qkv_anchor["seeds"][str(SEED)]["q56-k56-v80"]
    prepare = pilot.make_preparer(dense_anchor, qkv_anchor)
    load_native = fp32.load_native

    def verify_native():
        native, digest = load_native()
        assert digest == prior_identity["native_sha256"]
        return native, digest

    def prepare_verified(reference, dense, arm, seed):
        model, sites, metadata = prepare(reference, dense, arm, seed)
        if arm == ARM:
            assert seed == SEED
            with patch.object(init, "variance_scaling_", sweep.gpu_variance_scaling):
                qkv_model, qkv_sites = roles.build_model(
                    reference, dense, "qkv", seed, allocations={"qkv": COEFFICIENTS}
                )
            assert (
                reference.parameter_digest(qkv_model)
                == expected_qkv["initial_parameters_sha256"]
            )
            for name in qkv_sites:
                assert metadata["sites"][name]["initial_parameters_sha256"] == (
                    reference.parameter_digest(qkv_sites[name])
                )
            for i in range(4):
                name = f"{i}.down"
                assert metadata["sites"][name] == old["sites"][name], name
            metadata["matched_projection_initialization"] = {
                "down_sites": 4,
                "qkv_sites": len(qkv_sites),
                "qkv_only_initial_parameters_sha256": expected_qkv[
                    "initial_parameters_sha256"
                ],
            }
            print(
                "MATCHED_INITIALIZATION",
                json.dumps(metadata["matched_projection_initialization"]),
                flush=True,
            )
        return model, sites, metadata

    identity = {
        "study": "lm-qkv-down-v-budget-20261118",
        "driver_sha256": fp32.sha(Path(__file__)),
        "pilot_driver_sha256": fp32.sha(Path(pilot.__file__)),
        "allocation_builder_sha256": fp32.sha(Path(roles.__file__)),
        "initialization_adapter_sha256": fp32.sha(Path(sweep.__file__)),
        "paired_anchor_sha256": sweep.PAIRED_ANCHOR_SHA,
        "qkv_anchor_sha256": pilot.QKV_ANCHOR_SHA,
        "pilot_summary_sha256": PILOT_SHA,
        "qkv_coefficients": list(COEFFICIENTS),
        "atoms_per_down_projection": pilot.DOWN_ATOMS,
        "oracle_scope": "First Q output and gradient oracle and the first q update at all sixteen sites.",
    }
    with (
        patch.object(pilot, "QKV_COEFFICIENTS", COEFFICIENTS),
        patch.object(fp32, "load_native", verify_native),
        patch.object(sweep, "COUNTS", INITIALIZATION_COUNTS),
        patch.multiple(
            base,
            TORCHCST_COMMIT=pilot.TORCHCST_COMMIT,
            MANIFEST_SHA=pilot.MANIFEST_SHA,
            COUNTS=PROBE_COUNTS,
            SCHEDULE=((SEED, (ARM,)),),
            model_for=pilot.build_model,
            prepare=prepare_verified,
        ),
    ):
        base.main(args, extra_identity=identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--paired-anchor", type=Path, required=True)
    parser.add_argument("--qkv-anchor", type=Path, required=True)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
