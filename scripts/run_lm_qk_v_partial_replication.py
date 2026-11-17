"""Run paired Dense, full-QKV, QK-only and V-only seeds 23 and 42."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import run_lm_fp32_seed_replication as fp32
import run_lm_qk_v_partial_2048 as partial
import run_lm_qkv_2048 as base
import run_lm_qkv_down_2048 as pilot
import run_lm_qkv_local_sweep as sweep
import run_lm_qkv_role_budget as roles
from torchcst import init

SCHEDULE = (
    (23, ("v-only", "dense", "qkv", "qk-only")),
    (42, ("qkv", "qk-only", "v-only", "dense")),
)
COUNTS = {"qkv": 48 * sweep.WIDTH, **partial.PROBE_ATOMS}


def build_model(reference, dense, arm, seed):
    if arm == "qkv":
        return pilot.build_model(reference, dense, arm, seed)
    return partial.build_model(reference, dense, arm, seed)


def main(args):
    assert fp32.sha(args.pilot_summary) == partial.PILOT_SHA
    assert fp32.sha(args.dense_anchor) == sweep.PAIRED_ANCHOR_SHA
    assert fp32.sha(args.qkv_anchor) == pilot.QKV_ANCHOR_SHA
    previous = json.loads(args.pilot_summary.read_text())
    dense_anchor = json.loads(args.dense_anchor.read_text())
    qkv_anchor = json.loads(args.qkv_anchor.read_text())
    assert {d["state"] for d in (previous, dense_anchor, qkv_anchor)} == {"completed"}
    prepare = base.prepare
    load_native = fp32.load_native
    site_digests = {}

    def verify_native():
        native, digest = load_native()
        assert digest == previous["identity"]["native_sha256"]
        return native, digest

    def prepare_matched(reference, dense, arm, seed):
        key = str(seed)
        expected_dense = dense_anchor["seeds"][key]["dense"][
            "initial_parameters_sha256"
        ]
        assert reference.parameter_digest(dense) == expected_dense
        with patch.object(init, "variance_scaling_", sweep.gpu_variance_scaling):
            model, sites, metadata = prepare(reference, dense, arm, seed)
        if arm == "qkv":
            expected_qkv = qkv_anchor["seeds"][key]["q48-k48-v64"]
            assert (
                metadata["initial_parameters_sha256"]
                == expected_qkv["initial_parameters_sha256"]
            )
        for site, layer in sites.items():
            digest = reference.parameter_digest(layer)
            assert site_digests.setdefault((seed, site), digest) == digest
            metadata["sites"][site]["initial_parameters_sha256"] = digest
        if arm in partial.ACTIVE:
            metadata["matched_dense_slices"] = [
                role for role in roles.ROLES if role not in partial.ACTIVE[arm]
            ]
        return model, sites, metadata

    identity = {
        "study": "lm-qk-v-partial-replication-2048-20261118",
        "driver_sha256": fp32.sha(Path(__file__)),
        "partial_driver_sha256": fp32.sha(Path(partial.__file__)),
        "controller_sha256": fp32.sha(Path(pilot.__file__)),
        "allocation_builder_sha256": fp32.sha(Path(roles.__file__)),
        "initialization_adapter_sha256": fp32.sha(Path(sweep.__file__)),
        "pilot_summary_sha256": partial.PILOT_SHA,
        "dense_anchor_sha256": sweep.PAIRED_ANCHOR_SHA,
        "qkv_anchor_sha256": pilot.QKV_ANCHOR_SHA,
        "partial_coefficients": partial.ACTIVE,
        "qkv_coefficients": list(pilot.QKV_COEFFICIENTS),
    }
    with (
        patch.object(fp32, "load_native", verify_native),
        patch.object(sweep, "COUNTS", COUNTS),
        patch.multiple(
            base,
            TORCHCST_COMMIT=pilot.TORCHCST_COMMIT,
            MANIFEST_SHA=pilot.MANIFEST_SHA,
            COUNTS=COUNTS,
            SCHEDULE=SCHEDULE,
            model_for=build_model,
            prepare=prepare_matched,
        ),
    ):
        base.main(args, extra_identity=identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--dense-anchor", type=Path, required=True)
    parser.add_argument("--qkv-anchor", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
