"""Direct Hub adapter; verify unchanged source/prior pins, no provisioning/retry."""

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

ENTRY = "src/cst_research/experiments/higgs_depth_normalization_learning_driver.py"


def write_destinations(output, cache, prior_root, source):
    """Resolve destinations without writing into either immutable input tree."""
    output, cache = Path(output).resolve(), Path(cache).resolve()
    immutable = (
        Path(prior_root).resolve(strict=True),
        Path(source).resolve(strict=True),
    )
    if output.exists() or any(
        output.is_relative_to(root) or cache.is_relative_to(root) for root in immutable
    ):
        raise ValueError(
            "fresh output/cache outside immutable prior evidence and consumer required"
        )
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError("separate output and compilation cache required")
    return output, cache


def run(args):
    from cst_research.experiments import (
        higgs_depth_normalization_driver as inherited_driver,
    )
    from cst_research.experiments import higgs_depth_normalization_learning as study
    from cst_research.experiments import (
        higgs_depth_normalization_learning_evidence as prior,
    )
    from cst_research.experiments import higgs_dst_confirmation_driver as common
    from cst_research.experiments.higgs_coordinate_lr import verify_external

    source = Path.cwd().resolve()
    inputs = source / "research-inputs"
    expected = {
        "consumer-source-manifest.json",
        "original-source-manifest.json",
        "original-source.tar.gz",
        "train-development.npz",
        "data-manifest.json",
        "runtime-pin.json",
        *prior.PROOFS,
    }
    if {p.name for p in inputs.iterdir()} != expected or any(
        not p.is_file() or p.is_symlink() for p in inputs.iterdir()
    ):
        raise ValueError(
            "exact ten source/data/runtime/prior-proof inputs required; no heldout"
        )
    manifest_file = inputs / "consumer-source-manifest.json"
    manifest = common.verify_consumer(
        source, manifest_file, args.consumer_commit, source / common.DRIVER_PATH
    )
    if manifest["files"].get(ENTRY) != common.sha(Path(__file__)):
        raise ValueError("executed learning Hub adapter differs")
    pins = {
        "original-source.tar.gz": inherited_driver.ARCHIVE_SHA,
        "original-source-manifest.json": study.stock.SOURCE_MANIFEST,
        "train-development.npz": study.DATA_SHA,
        "data-manifest.json": study.MANIFEST_SHA,
        "runtime-pin.json": prior.RUNTIME_SHA,
        **prior.PROOFS,
    }
    for name, digest in pins.items():
        if common.sha(inputs / name) != digest:
            raise ValueError("input SHA mismatch: " + name)
    args.prior_root = args.prior_root.resolve(strict=True)
    args.output, cache = write_destinations(
        args.output,
        args.cache_root / args.consumer_commit / study.stock.SOURCE_MANIFEST,
        args.prior_root,
        source,
    )
    args.output.mkdir(parents=True)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="learning-verified-",
        dir=Path(tempfile.gettempdir()).resolve(strict=True),
    ) as temporary:
        staged = Path(temporary)
        common.extract_verified_external(
            inputs / "original-source.tar.gz",
            inputs / "original-source-manifest.json",
            staged,
            commit=study.stock.SOURCE,
            archive_sha=inherited_driver.ARCHIVE_SHA,
        )
        bootstrap = common.load_bootstrap(staged)
        with bootstrap.exclusive_cache(cache):
            original = cache / "external975"
            if not original.exists():
                constructing = cache / "external975-building"
                if constructing.exists():
                    raise ValueError(
                        "incomplete source build retained; owner review required"
                    )
                shutil.copytree(staged, constructing, copy_function=shutil.copy2)
                constructing.rename(original)
            verify_external(
                SimpleNamespace(
                    external_source=original,
                    external_manifest=inputs / "original-source-manifest.json",
                    identity={
                        "external_source_manifest_sha256": study.stock.SOURCE_MANIFEST
                    },
                )
            )
            inherited_driver.bind_cache(
                cache,
                {
                    "consumer_commit": args.consumer_commit,
                    "consumer_manifest_sha256": common.sha(manifest_file),
                    "external_source_manifest_sha256": study.stock.SOURCE_MANIFEST,
                    "dataset_sha256": study.DATA_SHA,
                    "runtime_pin": common.read(inputs / "runtime-pin.json"),
                },
            )
            consumer = common.immutable_consumer(source, cache, manifest)
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
                "cst_research.experiments.higgs_depth_normalization_learning",
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
                "--prior-root",
                str(args.prior_root),
                "--output",
                str(args.output / "science"),
            ]
            common.write(
                args.output / "source-verification.json",
                {
                    "new_consumer_commit": args.consumer_commit,
                    "consumer_manifest_sha256": common.sha(manifest_file),
                    "inherited_model_source": prior.PRIOR_COMMIT,
                    "inputs": {
                        name: common.sha(inputs / name) for name in sorted(expected)
                    },
                    "prior_evidence_readonly": str(args.prior_root),
                    "argv": command,
                    "timeout": args.timeout,
                    "no_prior_raw_copy": True,
                    "automatic_retry": False,
                },
            )
            for name in (
                "consumer-source-manifest.json",
                "original-source-manifest.json",
                "data-manifest.json",
                "runtime-pin.json",
                *prior.PROOFS,
            ):
                shutil.copy2(inputs / name, args.output / name)
            try:
                with (args.output / "foreground.log").open("w") as log:
                    inherited_driver.foreground(
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
                        "kind": "actual-learning-attempt-retained-files",
                        "new_consumer_commit": args.consumer_commit,
                        "files": {n: common.sha(p) for n, p in files.items()},
                        "completed_summary_present": (
                            args.output / "science/run-summary.json"
                        ).is_file(),
                        "prior_raw_referenced_not_copied": True,
                        "automatic_resume_or_retry": False,
                    },
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "cache-root", "prior-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if not 60 <= args.timeout <= 7200:
        raise ValueError("bounded60..7200 foreground timeout required")
    run(args)


if __name__ == "__main__":
    main()
