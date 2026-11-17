"""Compare Dense and two budgets of the selected partial placement at width 4096."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import run_lm_fp32_seed_replication as fp32
import run_lm_qk_v_equal_budget as equal
import run_lm_qk_v_partial_2048 as partial
import run_lm_qkv_2048 as base
import run_lm_qkv_down_2048 as pilot
import run_lm_qkv_local_sweep as sweep
import run_lm_qkv_role_budget as roles
from torchcst import init

WIDTH = 4096
HEADS = 16
ARMS = ("dense", "low", "high")
BUDGETS = {
    "qk-only": {"low": {"query": 32, "key": 32}, "high": {"query": 48, "key": 48}},
    "v-only": {"low": {"value": 64}, "high": {"value": 96}},
}


def adapt_reference(reference):
    reference_config = reference.WidthConfig
    restore = reference.restore_checkpoint

    @dataclass(frozen=True)
    class WideConfig(reference_config):
        width: int = WIDTH
        heads: int = HEADS

        def __post_init__(self):
            assert (self.width, self.heads) == (WIDTH, HEADS)
            reference_config(**(asdict(self) | {"width": 2048, "heads": 8}))

    def batches_at(data, order, clock):
        starts = order[(clock - 1) * 8 : clock * 8] * 128
        return [
            reference.token_batch(data["train"], starts[i : i + 1], 128, "cuda")
            for i in range(len(starts))
        ]

    def restore_without_duplicate_gpu_state(path, model, optimizer, clock, digest):
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        base.release_cuda()
        restore(path, model, optimizer, clock, digest)

    reference.WidthConfig = WideConfig
    reference.batches_at = batches_at
    reference.restore_checkpoint = restore_without_duplicate_gpu_state


def main(args):
    selection = json.loads(args.selection.read_text())
    assert fp32.sha(args.budget_summary) == selection["equal_budget_summary_sha256"]
    summary = json.loads(args.budget_summary.read_text())
    assert summary["state"] == "completed"
    assert (
        summary["identity"]["extension"]["study"]
        == "lm-qk-v-equal-budget-2048-20261118"
    )
    assert set(selection["placements"]) == set(BUDGETS)
    placement = min(
        selection["placements"],
        key=lambda name: (
            selection["placements"][name]["mean_confirmation_ce"],
            selection["placements"][name]["mean_update_ms"],
            name,
        ),
    )
    budgets = BUDGETS[placement]
    counts = {
        arm: WIDTH * next(iter(coefficients.values()))
        for arm, coefficients in budgets.items()
    }
    import_reference = fp32.import_reference
    load_native = fp32.load_native
    prepare = base.prepare

    def import_wide_reference(run):
        reference, anchor = import_reference(run)
        adapt_reference(reference)
        return reference, anchor

    def verify_native():
        native, digest = load_native()
        assert digest == summary["identity"]["native_sha256"]
        return native, digest

    def build_model(reference, dense, arm, seed):
        if arm == "dense":
            return partial.build_model(reference, dense, arm, seed)
        with patch.object(partial, "ACTIVE", {placement: budgets[arm]}):
            return partial.build_model(reference, dense, placement, seed)

    def prepare_gpu(reference, dense, arm, seed):
        with patch.object(init, "variance_scaling_", sweep.gpu_variance_scaling):
            model, sites, metadata = prepare(reference, dense, arm, seed)
        if sites:
            for name, layer in sites.items():
                metadata["sites"][name]["initial_parameters_sha256"] = (
                    reference.parameter_digest(layer)
                )
            metadata["matched_dense_slices"] = [
                role for role in roles.ROLES if role not in budgets[arm]
            ]
        return model, sites, metadata

    identity = {
        "study": "lm-qk-v-scale-4096-20261118",
        "driver_sha256": fp32.sha(Path(__file__)),
        "partial_driver_sha256": fp32.sha(Path(partial.__file__)),
        "allocation_builder_sha256": fp32.sha(Path(roles.__file__)),
        "initialization_adapter_sha256": fp32.sha(Path(sweep.__file__)),
        "timing_controller_sha256": fp32.sha(Path(pilot.__file__)),
        "selection_sha256": fp32.sha(args.selection),
        "equal_budget_summary_sha256": selection["equal_budget_summary_sha256"],
        "placement": placement,
        "role_coefficients": budgets,
        "train_sequences_per_microbatch": 1,
    }
    with (
        patch.object(fp32, "import_reference", import_wide_reference),
        patch.object(fp32, "load_native", verify_native),
        patch.object(sweep, "COUNTS", counts),
        patch.object(sweep, "WIDTH", WIDTH),
        patch.object(pilot, "ARMS", ARMS),
        patch.multiple(
            base,
            WIDTH=WIDTH,
            TORCHCST_COMMIT=equal.TORCHCST_COMMIT,
            MANIFEST_SHA=equal.MANIFEST_SHA,
            COUNTS=counts,
            SCHEDULE=((17, ARMS),),
            model_for=build_model,
            prepare=prepare_gpu,
        ),
    ):
        base.main(
            args, extra_identity=identity, benchmark_preflight=pilot.benchmark_preflight
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--budget-summary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
