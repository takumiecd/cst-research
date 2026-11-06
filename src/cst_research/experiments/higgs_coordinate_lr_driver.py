"""Shared-L4 script for the separate bounded position-LR pilot; no provisioning."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.cwd() / "src"))
from cst_research.experiments import higgs_dst_confirmation_driver as common

COMMIT = "97580582ed28c4e494bf5c2656bbe94a85c544c3"
ARCHIVE_SHA = "edbb585de56e304887b118069b7c7ecd9e45ae5fe2e08a5afca828ad8544eb5f"
DATA_SHA = "62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac"
DRIVER = "src/cst_research/experiments/higgs_coordinate_lr_driver.py"


def run(args):
    sys.dont_write_bytecode = True
    source = Path.cwd()
    inputs = source / "research-inputs"
    output = Path(os.environ["CST_JOB_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[0-9a-f]{40}", args.consumer_commit):
        raise ValueError("full consumer commit required")
    manifest_path = inputs / "consumer-source-manifest.json"
    # Reuse the exact immutable-consumer guard; additionally bind this injected
    # script rather than treating the inherited DST driver as executed code.
    manifest = common.verify_consumer(
        source, manifest_path, args.consumer_commit, source / common.DRIVER_PATH
    )
    if manifest["files"].get(DRIVER) != common.sha(Path(__file__)):
        raise ValueError("executed coordinate driver differs")
    if common.sha(inputs / "screen.npz") != DATA_SHA:
        raise ValueError("pilot dataset differs")
    external_manifest = inputs / "original-source-manifest.json"
    digest = common.sha(external_manifest)
    cache = Path("/content/cst-research-position-lr") / args.consumer_commit / digest
    with tempfile.TemporaryDirectory(prefix="cst-position-verified-") as temporary:
        staging = Path(temporary).resolve()
        external = common.extract_verified_external(
            inputs / "original-source.tar.gz",
            external_manifest,
            staging,
            commit=COMMIT,
            archive_sha=ARCHIVE_SHA,
        )
        bootstrap = common.load_bootstrap(staging)
        staged_inputs = staging / "research-inputs"
        staged_inputs.mkdir()
        common.write(
            staged_inputs / "source-manifest.json",
            {"commit": COMMIT, "files": external["files"]},
        )
        for name in ("screen.npz", "data-manifest.json"):
            shutil.copy2(inputs / name, staged_inputs / name)
        with bootstrap.exclusive_cache(cache):
            consumer = common.immutable_consumer(source, cache, manifest)
            original, marker = bootstrap.prepare_source_cache(
                staging, cache / "original", COMMIT
            )
            runtime = bootstrap.probe_runtime()
            proof = common.read(inputs / "original-qualification-proof.json")
            from importlib.util import module_from_spec, spec_from_file_location

            spec = spec_from_file_location(
                "_verified_position_original_campaign",
                original / "benchmarks/training/higgs_confirmation_campaign.py",
            )
            campaign = module_from_spec(spec)
            spec.loader.exec_module(campaign)
            original_identity = campaign.experiment_identity(
                inputs / "screen.npz",
                common.sha(inputs / "original-campaign-source-manifest.json"),
            )
            if (
                proof["status"] != "passed"
                or proof["identity"] != original_identity
                or proof["runtime_fingerprint"] != runtime
                or len(proof["raw_audit"]["raw_oracles"]) != 10
                or len(proof["raw_audit"]["raw_epoch_pairs"]) != 16
                or "136 passed" not in proof["raw_audit"]["gpu_tests"]
            ):
                raise ValueError(
                    "actual original975 qualification/source/runtime certificate differs"
                )
            if common.read(inputs / "original-campaign-source-manifest.json") != {
                "commit": COMMIT,
                "files": external["files"],
            }:
                raise ValueError("original campaign source inventory differs")
            sys.path.insert(0, str(consumer / "src"))
            from cst_research.experiments.higgs_coordinate_lr import json_sha, protocol

            identity = {
                "schema_version": 1,
                "consumer_commit": args.consumer_commit,
                "consumer_manifest_sha256": common.sha(manifest_path),
                "external_source_commit": COMMIT,
                "external_source_manifest_sha256": digest,
                "external_archive_sha256": ARCHIVE_SHA,
                "dataset_sha256": DATA_SHA,
                "protocol_sha256": json_sha(protocol()),
                "runtime_fingerprint": runtime,
                "original_qualification_proof_sha256": common.sha(
                    inputs / "original-qualification-proof.json"
                ),
            }
            reference = inputs / "original-control-reference.json"
            if reference.exists():
                identity["original_control_reference_sha256"] = common.sha(reference)
                shutil.copy2(reference, output / "original-control-reference.json")
            common.write(output / "experiment-identity.json", identity)
            common.write(
                output / "bootstrap-source-verification.json",
                {
                    "identity": identity,
                    "external_cache": marker,
                    "original_qualification": proof,
                    "source_scope": "unchanged975 runtime and benchmark; explicit separate consumer factory/checkpoint/step instrumentation",
                },
            )
            shutil.copy2(external_manifest, output / "original-source-manifest.json")
            dependencies = bootstrap.prepare_dependencies(original, cache / "original")
            environment = os.environ.copy()
            environment.update(
                PYTHONPATH=os.pathsep.join(
                    [
                        str(dependencies),
                        str(consumer / "src"),
                        str(original / "src"),
                        str(original),
                    ]
                ),
                PATH=str(dependencies / "bin") + os.pathsep + environment["PATH"],
                PYTHONDONTWRITEBYTECODE="1",
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                MAX_JOBS="2",
                TORCH_CUDA_ARCH_LIST="8.9",
                TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
            )
            command = [
                sys.executable,
                "-m",
                "cst_research.experiments.higgs_coordinate_lr",
                "--stage",
                args.stage,
                "--data",
                str(original / "research-inputs/screen.npz"),
                "--output",
                str(output),
                "--external-source",
                str(original),
                "--external-manifest",
                str(output / "original-source-manifest.json"),
                "--identity-file",
                str(output / "experiment-identity.json"),
            ]
            if reference.exists():
                command += ["--control-reference", str(reference)]
            if args.stage == "followup":
                if (
                    args.seed not in (29, 43)
                    or args.selected_ratio not in (0.1, 10.0)
                    or not args.pilot_decision_sha256
                ):
                    raise ValueError(
                        "followup requires paired seed, chosen ratio and verified pilot decision digest"
                    )
                pilot = inputs / "pilot-decision.json"
                if common.sha(pilot) != args.pilot_decision_sha256:
                    raise ValueError("pilot decision input digest differs")
                command += [
                    "--seed",
                    str(args.seed),
                    "--selected-ratio",
                    str(args.selected_ratio),
                    "--pilot-decision",
                    str(pilot),
                    "--pilot-decision-sha256",
                    args.pilot_decision_sha256,
                ]
            common.write(
                output / "bootstrap-command.json",
                {"argv": command, "automatic_followup": False},
            )
            result = None
            failure = None
            try:
                result = subprocess.run(
                    command, cwd=original, env=environment, check=False
                )
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, command)
                bootstrap.prepare_source_cache(staging, cache / "original", COMMIT)
                common.immutable_consumer(source, cache, manifest)
            except BaseException as error:
                failure = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                common.write(
                    output / "bootstrap-status.json",
                    {
                        "stage": args.stage,
                        "returncode": None if result is None else result.returncode,
                        "failure": failure,
                        "confirmation_evaluated": False,
                        "test_evaluated": False,
                    },
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--stage", choices=("pilot", "followup"), required=True)
    parser.add_argument("--seed", type=int, choices=(29, 43))
    parser.add_argument("--selected-ratio", type=float, choices=(0.1, 10.0))
    parser.add_argument("--pilot-decision-sha256")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
