"""Shared-pool script for immutable external-source DST development phases.

No provisioning or direct Colab operations. Use pool --script with staged
research-inputs and --consumer-commit FULL_SHA. External verified f377 utilities
own dependencies and recovery serialization; no torchcst runtime is vendored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

EXTERNAL_COMMIT = "f37729e61dc11a57e76103aa2457c2be11c4c306"
EXTERNAL_ARCHIVE_SHA = (
    "90ac777b939cd9107dc0fbdbdc2492f26e2897fd4ada51e97d4905bb166e385f"
)
DATA_SHA = "62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac"
DRIVER_PATH = "src/cst_research/experiments/higgs_dst_comparison_driver.py"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe_path(root, name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or path.as_posix() != name
        or name == "."
    ):
        raise ValueError("unsafe source/artifact path")
    target = Path(root)
    for part in path.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError("symlink in frozen source path")
    return target


def inventory(root):
    result = {}
    for path in Path(root).rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("source tree contains a nonregular entry")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path
    return result


def verify_consumer(source, manifest_path, commit, executed_driver):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("consumer commit must be a full lowercase SHA")
    manifest = read(manifest_path)
    if (
        manifest.get("commit") != commit
        or not isinstance(manifest.get("files"), dict)
        or not manifest["files"]
    ):
        raise ValueError("consumer source manifest identity invalid")
    for name, digest in manifest["files"].items():
        if name.startswith(("research-inputs/", ".git/")) or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise ValueError("invalid tracked consumer source entry")
        if sha(safe_path(source, name)) != digest:
            raise ValueError("consumer source hash mismatch: " + name)
    supplied = {
        name
        for name in inventory(source)
        if not name.startswith("research-inputs/") and name != "__pool_driver__.py"
    }
    if supplied != set(manifest["files"]) or manifest["files"].get(DRIVER_PATH) != sha(
        executed_driver
    ):
        raise ValueError(
            "consumer source inventory or executed pool-driver bytes differ"
        )
    return manifest


def immutable_consumer(source, cache, manifest):
    expected = manifest["files"]
    target = cache / "consumer-source"
    if not target.exists():
        target.mkdir()
        for name in expected:
            destination = safe_path(target, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(safe_path(source, name), destination)
    observed = inventory(target)
    if set(observed) != set(expected) or any(
        sha(observed[name]) != value for name, value in expected.items()
    ):
        raise ValueError("immutable consumer cache inventory/bytes changed")
    return target


def extract_verified_external(archive_path, manifest_path, destination):
    manifest = read(manifest_path)
    if (
        manifest.get("source_commit") != EXTERNAL_COMMIT
        or manifest.get("archive_sha256") != EXTERNAL_ARCHIVE_SHA
        or sha(archive_path) != EXTERNAL_ARCHIVE_SHA
    ):
        raise ValueError("external archive differs from frozen f377 identity")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("external source manifest lacks files")
    for name, value in expected.items():
        safe_path(destination, name)
        if name.startswith(("research-inputs/", ".git/")) or not re.fullmatch(
            r"[0-9a-f]{64}", value
        ):
            raise ValueError("invalid external source manifest entry")
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            target = safe_path(destination, member.name.rstrip("/"))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if (
                not member.isfile()
                or member.name not in expected
                or member.name in seen
            ):
                raise ValueError("external archive inventory is unsafe or differs")
            seen.add(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                shutil.copyfileobj(archive.extractfile(member), output)
            os.utime(target, (member.mtime, member.mtime))
            if sha(target) != expected[member.name]:
                raise ValueError("external archived source hash mismatch")
    if seen != set(expected):
        raise ValueError("external archive omits frozen source files")
    return manifest


def load_bootstrap(source):
    path = source / "benchmarks/training/higgs_confirmation_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_verified_dst_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_recovery(inputs, bootstrap, state, binding):
    capsule, receipt_path = (
        inputs / "recovery-state.tar.gz",
        inputs / "recovery-state-receipt.json",
    )
    if capsule.exists() != receipt_path.exists():
        raise ValueError("recovery requires both capsule and receipt")
    if capsule.exists():
        receipt = read(receipt_path)
        if (
            receipt.get("file") != "recovery-state.tar.gz"
            or receipt.get("sha256") != sha(capsule)
            or receipt.get("bytes") != capsule.stat().st_size
        ):
            raise ValueError("recovery receipt/archive bytes differ")
        return bootstrap.restore_state(capsule, state, binding)
    return None


def run(args, cache_root=Path("/content/cst-research-higgs-dst")):
    sys.dont_write_bytecode = True
    if not re.fullmatch(r"[0-9a-f]{40}", args.consumer_commit):
        raise ValueError("consumer commit must be a full lowercase SHA")
    source, inputs = Path.cwd(), Path.cwd() / "research-inputs"
    output = Path(os.environ["CST_JOB_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    consumer_manifest_path = inputs / "consumer-source-manifest.json"
    consumer_manifest = verify_consumer(
        source, consumer_manifest_path, args.consumer_commit, Path(__file__)
    )
    if sha(inputs / "screen.npz") != DATA_SHA:
        raise ValueError("prepared HIGGS archive differs")
    data_manifest = read(inputs / "data-manifest.json")
    artifact = data_manifest.get("artifacts", {}).get("screen.npz")
    if (artifact.get("sha256") if isinstance(artifact, dict) else artifact) != DATA_SHA:
        raise ValueError("prepared data manifest differs")
    external_manifest_path = inputs / "torchcst-source-manifest.json"
    external_manifest_sha = sha(external_manifest_path)
    cache = Path(cache_root) / args.consumer_commit / external_manifest_sha
    with tempfile.TemporaryDirectory(prefix="cst-dst-verified-source-") as temporary:
        staging = Path(temporary).resolve()
        external_manifest = extract_verified_external(
            inputs / "torchcst-source.tar.gz", external_manifest_path, staging
        )
        bootstrap = load_bootstrap(staging)
        staged_inputs = staging / "research-inputs"
        staged_inputs.mkdir()
        write(
            staged_inputs / "source-manifest.json",
            {"commit": EXTERNAL_COMMIT, "files": external_manifest["files"]},
        )
        for name in ("screen.npz", "data-manifest.json"):
            shutil.copy2(inputs / name, staged_inputs / name)
        with bootstrap.exclusive_cache(cache):
            consumer = immutable_consumer(source, cache, consumer_manifest)
            external, marker = bootstrap.prepare_source_cache(
                staging, cache / "external", EXTERNAL_COMMIT
            )
            runtime = bootstrap.probe_runtime()
            if (cache / "runtime.json").exists() and read(
                cache / "runtime.json"
            ) != runtime:
                raise ValueError("existing cache runtime versions changed")
            write(cache / "runtime.json", runtime)
            sys.path.insert(0, str(consumer / "src"))
            from cst_research.experiments import higgs_dst_comparison as campaign

            binding = {
                "schema_version": 1,
                "consumer_commit": args.consumer_commit,
                "consumer_manifest_sha256": sha(consumer_manifest_path),
                "external_source_commit": EXTERNAL_COMMIT,
                "external_source_manifest_sha256": external_manifest_sha,
                "external_archive_sha256": EXTERNAL_ARCHIVE_SHA,
                "dataset_sha256": DATA_SHA,
                "data_manifest_sha256": sha(inputs / "data-manifest.json"),
                "protocol_sha256": campaign.json_sha(campaign.protocol()),
                "runtime_fingerprint": runtime,
            }
            state = cache / "campaign-state"
            restoration = require_recovery(inputs, bootstrap, state, binding)
            state.mkdir(exist_ok=True)
            binding_path = state / bootstrap.STATE_BINDING
            if binding_path.exists():
                if read(binding_path) != binding:
                    raise ValueError("campaign state binding differs")
            elif bootstrap.regular_files(state):
                raise ValueError("existing campaign state is unbound")
            else:
                write(binding_path, binding)
            write(output / "campaign-identity.json", binding)
            write(
                output / "bootstrap-source-verification.json",
                {
                    **binding,
                    "consumer_source": str(consumer),
                    "external_source": str(external),
                    "external_cache": marker,
                    "restoration": restoration,
                },
            )
            result, failure = None, None
            try:
                deps = bootstrap.prepare_dependencies(external, cache / "external")
                environment = os.environ.copy()
                environment.update(
                    PYTHONPATH=os.pathsep.join(
                        (
                            str(deps),
                            str(consumer / "src"),
                            str(external / "src"),
                            str(external),
                        )
                    ),
                    PATH=str(deps / "bin") + os.pathsep + environment["PATH"],
                    PYTHONDONTWRITEBYTECODE="1",
                    PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                    PYTEST_ADDOPTS="-p no:cacheprovider",
                    MAX_JOBS="2",
                    TORCH_CUDA_ARCH_LIST="8.9",
                    TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
                )
                argv = [
                    sys.executable,
                    "-m",
                    "cst_research.experiments.higgs_dst_comparison",
                    "--stage",
                    args.stage,
                    "--external-source",
                    str(external),
                    "--data",
                    str(external / "research-inputs/screen.npz"),
                    "--output",
                    str(output),
                    "--campaign-state",
                    str(state),
                    "--identity-file",
                    str(output / "campaign-identity.json"),
                ]
                for key in ("width", "kind", "divisor", "seed"):
                    value = getattr(args, key)
                    if value is not None:
                        argv += ["--" + key, str(value)]
                if (inputs / "original-selection-index.json").exists():
                    argv += [
                        "--original-selection-index",
                        str(inputs / "original-selection-index.json"),
                    ]
                write(
                    output / "bootstrap-command.json",
                    {"argv": argv, "stage": args.stage},
                )
                result = subprocess.run(
                    argv, cwd=external, env=environment, check=False
                )
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, argv)
                # Reuse existing verified-cache entry points to detect source changes.
                bootstrap.prepare_source_cache(
                    staging, cache / "external", EXTERNAL_COMMIT
                )
                immutable_consumer(source, cache, consumer_manifest)
            except BaseException as error:
                failure = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                write(
                    output / "bootstrap-status.json",
                    {
                        "stage": args.stage,
                        "returncode": None if result is None else result.returncode,
                        "failure": failure,
                    },
                )
                bootstrap.archive_state(state, output, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument(
        "--stage",
        choices=("qualify", "tune", "freeze-width", "confirm", "assess"),
        required=True,
    )
    parser.add_argument("--width", type=int, choices=(1024, 2048))
    parser.add_argument("--kind", choices=("static", "rigl"))
    parser.add_argument("--divisor", type=int, choices=(1000, 500, 200, 100))
    parser.add_argument("--seed", type=int, choices=(29, 43, 71))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
