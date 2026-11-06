"""Foreground StageA native16 only; immutable consumer and shared verified975 cache."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.cwd() / "src"))

ENTRY = "src/cst_research/experiments/higgs_linear_atom_driver.py"


def destinations(output, cache_root, source):
    output, cache, source = (
        Path(output).resolve(),
        Path(cache_root).resolve(),
        Path(source).resolve(),
    )
    if (
        output.exists()
        or output == cache
        or output.is_relative_to(cache)
        or cache.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from compilation cache required")
    if (
        output.is_relative_to(source)
        or cache.is_relative_to(source)
        or source.is_relative_to(cache)
    ):
        raise ValueError("output/cache outside immutable consumer required")
    return output, cache


def cache_binding(pins, runtime):
    """Independent of consumer commit: StageB must reuse this exact source abspath."""
    return {
        "external_source_manifest_sha256": pins["original-source-manifest.json"],
        "external_source_archive_sha256": pins["original-source.tar.gz"],
        "runtime_pin_sha256": pins["runtime-pin.json"],
        "runtime_pin": runtime,
        "purpose": "fixed-public975-compile-path; not qualification or mutable campaign state",
    }


def run(args):
    from cst_research.experiments import higgs_depth_normalization_driver as inherited
    from cst_research.experiments import higgs_dst_confirmation_driver as common
    from cst_research.experiments import higgs_linear_atom_qualification as study
    from cst_research.experiments.higgs_coordinate_lr import verify_external
    from cst_research.experiments.higgs_depth_study import SOURCE, SOURCE_MANIFEST

    source = Path.cwd().resolve()
    inputs = source / "research-inputs"
    manifest_file = inputs / "consumer-source-manifest.json"
    manifest = common.verify_consumer(
        source, manifest_file, args.consumer_commit, source / common.DRIVER_PATH
    )
    if manifest["files"].get(ENTRY) != common.sha(Path(__file__)):
        raise ValueError("executed StageA driver bytes differ from committed source")
    pins = study.verify_inputs(inputs)
    args.output, cache_root = destinations(args.output, args.cache_root, source)
    cache = cache_root / SOURCE_MANIFEST / pins["runtime-pin.json"]
    args.output.mkdir(parents=True)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="linear-verified-", dir=Path(tempfile.gettempdir()).resolve(strict=True)
    ) as temporary:
        staged = Path(temporary)
        common.extract_verified_external(
            inputs / "original-source.tar.gz",
            inputs / "original-source-manifest.json",
            staged,
            commit=SOURCE,
            archive_sha=inherited.ARCHIVE_SHA,
        )
        bootstrap = common.load_bootstrap(staged)
        with bootstrap.exclusive_cache(cache):
            inherited.bind_cache(
                cache, cache_binding(pins, common.read(inputs / "runtime-pin.json"))
            )
            original = cache / "external975"
            if not original.exists():
                building = cache / "external975-building"
                if building.exists():
                    raise ValueError(
                        "incomplete immutable cache retained; owner review required"
                    )
                shutil.copytree(staged, building, copy_function=shutil.copy2)
                building.rename(original)
            verify_external(
                SimpleNamespace(
                    external_source=original,
                    external_manifest=inputs / "original-source-manifest.json",
                    identity={"external_source_manifest_sha256": SOURCE_MANIFEST},
                )
            )
            consumer_cache = cache / "consumers" / args.consumer_commit
            consumer_cache.mkdir(parents=True, exist_ok=True)
            consumer = common.immutable_consumer(source, consumer_cache, manifest)
            environment = os.environ.copy()
            environment.update(
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=os.pathsep.join(
                    (str(consumer / "src"), str(original / "src"), str(original))
                ),
                MAX_JOBS="2",
                TORCH_CUDA_ARCH_LIST="8.9",
                TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
            )
            command = [
                sys.executable,
                "-B",
                "-m",
                "cst_research.experiments.higgs_linear_atom_qualification",
                "--consumer-commit",
                args.consumer_commit,
                "--consumer-manifest",
                str(manifest_file),
                "--external-source",
                str(original),
                "--external-manifest",
                str(inputs / "original-source-manifest.json"),
                "--inputs",
                str(inputs),
                "--output",
                str(args.output / "science"),
            ]
            common.write(
                args.output / "source-verification.json",
                {
                    "consumer_commit": args.consumer_commit,
                    "consumer_manifest_sha256": common.sha(manifest_file),
                    "external_source_commit": SOURCE,
                    "verified_inputs": pins,
                    "cache_binding": cache_binding(
                        pins, common.read(inputs / "runtime-pin.json")
                    ),
                    "argv": command,
                    "timeout": args.timeout,
                    "qualification_only": True,
                    "task_training_updates": 0,
                    "automatic_retry": False,
                },
            )
            for name in pins:
                if name != "train-development.npz" and name != "original-source.tar.gz":
                    shutil.copy2(inputs / name, args.output / name)
            try:
                with (args.output / "foreground.log").open("w") as log:
                    inherited.foreground(
                        command,
                        cwd=original,
                        environment=environment,
                        log=log,
                        timeout=args.timeout,
                    )
            finally:
                files = common.inventory(args.output)
                common.write(
                    args.output / "recovery-index.json",
                    {
                        "kind": "StageA-qualification-retained-evidence-only-NOT-campaign-state",
                        "consumer_commit": args.consumer_commit,
                        "files": {n: common.sha(p) for n, p in files.items()},
                        "completed_summary_present": (
                            args.output / "science/run-summary.json"
                        ).is_file(),
                        "task_training_updates": 0,
                        "automatic_resume_or_retry": False,
                    },
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "cache-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if not 60 <= args.timeout <= 7200:
        raise ValueError("bounded60..7200 foreground timeout required")
    run(args)


if __name__ == "__main__":
    main()
