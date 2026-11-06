"""Verified foreground Hub-script adapter; no allocation, queue, install or retry."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.cwd() / "src"))

ARCHIVE_SHA = "edbb585de56e304887b118069b7c7ecd9e45ae5fe2e08a5afca828ad8544eb5f"
ENTRY = "src/cst_research/experiments/higgs_depth_normalization_driver.py"


def foreground(command, *, cwd, environment, log, timeout):
    """Own and reap the exact child process group, including compiler children."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def bind_cache(cache, binding):
    from cst_research.experiments import higgs_dst_confirmation_driver as common

    marker = cache / "runtime-source-binding.json"
    if marker.exists():
        if marker.is_symlink() or common.read(marker) != binding:
            raise ValueError("existing compile cache runtime/source binding differs")
    else:
        if (cache / "torch-extensions").exists():
            raise ValueError("unbound existing compilation cache")
        common.write(marker, binding)


def run(args):
    from cst_research.experiments import higgs_depth_normalization as study
    from cst_research.experiments import higgs_dst_confirmation_driver as common
    from cst_research.experiments.higgs_coordinate_lr import verify_external
    from cst_research.experiments.higgs_depth_data import DATA_SHA, MANIFEST_SHA

    source = Path.cwd().resolve()
    args.output = args.output.resolve()
    inputs = source / "research-inputs"
    expected_inputs = {
        "consumer-source-manifest.json",
        "original-source-manifest.json",
        "original-source.tar.gz",
        "train-development.npz",
        "data-manifest.json",
        "runtime-pin.json",
    }
    if {p.name for p in inputs.iterdir()} != expected_inputs or any(
        not p.is_file() or p.is_symlink() for p in inputs.iterdir()
    ):
        raise ValueError(
            "exact six source/data/runtime inputs required; no future payload or symlink"
        )
    manifest_file = inputs / "consumer-source-manifest.json"
    manifest = common.verify_consumer(
        source, manifest_file, args.consumer_commit, source / common.DRIVER_PATH
    )
    if manifest["files"].get(ENTRY) != common.sha(Path(__file__)):
        raise ValueError("executed diagnostic adapter differs from frozen consumer")
    for name, digest in [
        ("original-source.tar.gz", ARCHIVE_SHA),
        ("original-source-manifest.json", study.stock.SOURCE_MANIFEST),
        ("train-development.npz", DATA_SHA),
        ("data-manifest.json", MANIFEST_SHA),
        ("runtime-pin.json", args.runtime_sha256),
    ]:
        if common.sha(inputs / name) != digest:
            raise ValueError("input digest mismatch: " + name)
    if args.output.exists():
        raise ValueError("fresh output required; no automatic probe replay")
    args.output.mkdir(parents=True)
    cache = (
        args.cache_root.resolve() / args.consumer_commit / study.stock.SOURCE_MANIFEST
    )
    cache.mkdir(parents=True, exist_ok=True)
    # Existing frozen bootstrap provides only the exclusive host-local filesystem
    # lock. Its L4 identity/data-specific bootstrap is intentionally not invoked.
    with tempfile.TemporaryDirectory(prefix="normdiag-verified-") as temporary:
        staged = Path(temporary)
        common.extract_verified_external(
            inputs / "original-source.tar.gz",
            inputs / "original-source-manifest.json",
            staged,
            commit=study.stock.SOURCE,
            archive_sha=ARCHIVE_SHA,
        )
        bootstrap = common.load_bootstrap(staged)
        with bootstrap.exclusive_cache(cache):
            original = cache / "external975"
            if not original.exists():
                staging = cache / "external975-building"
                if staging.exists():
                    raise ValueError(
                        "incomplete previous source build retained; explicit owner recovery required"
                    )
                shutil.copytree(staged, staging, copy_function=shutil.copy2)
                staging.rename(original)
            verify_external(
                SimpleNamespace(
                    external_source=original,
                    external_manifest=inputs / "original-source-manifest.json",
                    identity={
                        "external_source_manifest_sha256": study.stock.SOURCE_MANIFEST
                    },
                )
            )
            bind_cache(
                cache,
                {
                    "consumer_commit": args.consumer_commit,
                    "consumer_manifest_sha256": common.sha(manifest_file),
                    "external_source_manifest_sha256": study.stock.SOURCE_MANIFEST,
                    "dataset_sha256": DATA_SHA,
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
                TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
            )
            runtime = common.read(inputs / "runtime-pin.json")
            capability = runtime["capability"]
            if capability not in ([8, 6], [8, 9], [8, 0]):
                raise ValueError("no approved diagnostic architecture")
            environment["TORCH_CUDA_ARCH_LIST"] = ".".join(map(str, capability))
            command = [
                sys.executable,
                "-B",
                "-m",
                "cst_research.experiments.higgs_depth_normalization_run",
                "--consumer-commit",
                args.consumer_commit,
                "--consumer-manifest",
                str(manifest_file),
                "--external-source",
                str(original),
                "--external-manifest",
                str(inputs / "original-source-manifest.json"),
                "--data",
                str(inputs / "train-development.npz"),
                "--data-manifest",
                str(inputs / "data-manifest.json"),
                "--runtime-file",
                str(inputs / "runtime-pin.json"),
                "--runtime-sha256",
                args.runtime_sha256,
                "--output",
                str(args.output / "science"),
            ]
            common.write(
                args.output / "source-verification.json",
                {
                    "consumer_commit": args.consumer_commit,
                    "consumer_manifest_sha256": common.sha(manifest_file),
                    "external_commit": study.stock.SOURCE,
                    "external_manifest_sha256": study.stock.SOURCE_MANIFEST,
                    "inputs": {
                        n: common.sha(inputs / n) for n in sorted(expected_inputs)
                    },
                    "argv": command,
                    "timeout_seconds": args.timeout,
                    "automatic_retry": False,
                },
            )
            for name in (
                "consumer-source-manifest.json",
                "original-source-manifest.json",
                "data-manifest.json",
                "runtime-pin.json",
            ):
                shutil.copy2(inputs / name, args.output / name)
            try:
                with (args.output / "foreground.log").open("w") as log:
                    foreground(
                        command,
                        cwd=original,
                        environment=environment,
                        log=log,
                        timeout=args.timeout,
                    )
            finally:
                # Recoverability index, never a synthetic success/qualification or
                # permission to continue past the fixed8 updates. No raw duplication.
                files = common.inventory(args.output)
                common.write(
                    args.output / "recovery-index.json",
                    {
                        "kind": "retained-attempt-file-inventory",
                        "consumer_commit": args.consumer_commit,
                        "files": {n: common.sha(p) for n, p in files.items()},
                        "completed_summary_present": (
                            args.output / "science/run-summary.json"
                        ).is_file(),
                        "automatic_resume_or_retry": False,
                        "scope": "Recover exact retained attempt from outer result archive; never reuse partial diagnostic state as fresh qualification.",
                    },
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if not 60 <= args.timeout <= 5400:
        raise ValueError("bounded foreground timeout60..5400 seconds required")
    run(args)


if __name__ == "__main__":
    main()
