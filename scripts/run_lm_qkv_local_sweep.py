"""Sweep nearby uniform QKV coefficients on the frozen width-2048 protocol."""

import argparse
import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

import run_lm_fp32_seed_replication as fp32
import run_lm_qkv_2048 as base
import torch
from benchmarks.cuda.linear import lm_qkv_initialization as calibration
from torchcst import init
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import lm_width
from torchcst._backends.torch.initialization import match_variance

TORCHCST_COMMIT = "0532fe8227009b201c35d7af378c01fdcabeacef"
MANIFEST_SHA = "d80ced2e4bbf14e57693e28becc0b24507c56d2e874711e13a0e72bfaabaae14"
PAIRED_ANCHOR_SHA = "15ce5cc578b817fe94b39e0dfd84ac268520d6a36a615c2e4c23e8ff23fbceda"
WIDTH = 2048
COEFFICIENTS = (64, 48, 56, 72, 80)
COUNTS = {f"qkv-c{coefficient}": WIDTH * coefficient for coefficient in COEFFICIENTS}
SCHEDULE = (
    (17, ("qkv-c64", "qkv-c48", "qkv-c72", "qkv-c56", "qkv-c80")),
    (23, ("qkv-c56", "qkv-c80", "qkv-c48", "qkv-c72")),
    (42, ("qkv-c72", "qkv-c48", "qkv-c80", "qkv-c56")),
)


def gpu_variance_scaling(layer, variance, *, radius_fraction=0.0, generator=None):
    assert radius_fraction == 0 and layer.in_features == layer.out_features == WIDTH
    assert len(layer.atoms.p) in COUNTS.values()
    assert layer.atoms.p.grad is None
    _, evaluate = calibration.gpu_evaluator(lm_width.width_extension(), layer)
    rows, report = match_variance(
        layer.kernel,
        layer.cst_charts(),
        layer.atoms.p,
        variance=variance,
        q=radius_fraction,
        generator=generator,
        energy_evaluator=evaluate,
    )
    with torch.no_grad():
        layer.atoms.p.copy_(rows)
    return init.InitializationReport(
        scheme="variance_scaling",
        radius_fraction=radius_fraction,
        variance=variance,
        **report,
    )


def make_preparer(anchor):
    prepare = base.prepare
    cpu_calibrate = init.variance_scaling_
    verified_dense = set()

    def prepare_gpu(reference, dense, arm, seed):
        control = anchor["seeds"][str(seed)]["qkv-c64"]
        if seed not in verified_dense:
            assert (
                reference.parameter_digest(dense)
                == control["common_dense_initial_sha256"]
            )
            verified_dense.add(seed)
        with patch.object(init, "variance_scaling_", gpu_variance_scaling):
            model, sites, metadata = prepare(reference, dense, arm, seed)
        if arm == "qkv-c64":
            assert (
                metadata["initial_parameters_sha256"]
                == control["initial_parameters_sha256"]
            )
        if seed == 17 and sites and arm != "qkv-c64":
            site = "0.query"
            initialized = sites[site]
            cpu_layer = copy.deepcopy(initialized)
            parameter_seed = calibration.site_seed(seed, site)
            start = time.monotonic()
            with patch.object(base.radius_envelope, "USE_RATIONAL_LOWER", False):
                receipt = cpu_calibrate(
                    cpu_layer,
                    model.config.init_std**2,
                    radius_fraction=0,
                    generator=torch.Generator().manual_seed(parameter_seed + 1),
                )
            relative_energy_difference = abs(
                receipt.expected_energy
                / initialized.initialization_report["expected_energy"]
                - 1
            )
            assert torch.equal(cpu_layer.atoms.p, initialized.atoms.p), arm
            metadata["gpu_cpu_init_gate"] = {
                "site": site,
                "bitwise_equal": True,
                "cpu_seconds": time.monotonic() - start,
                "relative_energy_difference": relative_energy_difference,
            }
            print("CPU_GPU_INIT_MATCH", arm, relative_energy_difference, flush=True)
        return model, sites, metadata

    return prepare_gpu


def main(args):
    assert fp32.sha(args.paired_anchor) == PAIRED_ANCHOR_SHA
    anchor = json.loads(args.paired_anchor.read_text())
    assert anchor["state"] == "completed"
    assert anchor["identity"]["config"]["width"] == WIDTH
    assert anchor["identity"]["steps"] == 4882
    for seed, _ in SCHEDULE:
        assert anchor["seeds"][str(seed)]["qkv-c64"]["state"] == "completed"
    identity = {
        "study": "lm-qkv-local-sweep-20261117",
        "sweep_driver_sha256": fp32.sha(Path(__file__)),
        "paired_anchor_sha256": PAIRED_ANCHOR_SHA,
        "gpu_calibration_sha256": fp32.sha(Path(calibration.__file__)),
        "calibration": "CPU random stream and FP64 radius-bank routing with GPU FP32 energies.",
    }
    with patch.multiple(
        base,
        TORCHCST_COMMIT=TORCHCST_COMMIT,
        MANIFEST_SHA=MANIFEST_SHA,
        COUNTS=COUNTS,
        SCHEDULE=SCHEDULE,
        prepare=make_preparer(anchor),
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
