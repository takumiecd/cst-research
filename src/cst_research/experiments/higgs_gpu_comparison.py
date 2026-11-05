"""Bounded HIGGS A100 MIG comparison using an unchanged external benchmark.

No torchcst runtime is vendored here. The source manifest, adapter manifest,
exact device, independent numerical references and eager/Graph trajectories
must pass before the two one-epoch timing pilots run. No heldout evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

SOURCE_COMMIT = "97580582ed28c4e494bf5c2656bbe94a85c544c3"
DATA_SHA256 = "62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac"
GPU_NAME = "NVIDIA A100 80GB PCIe MIG 3g.40gb"
GPU_BYTES = 42144366592
WIDTH = 2048
DIVISOR = 1000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def persist(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe_relative(name):
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and path.as_posix() == name
    )


def verify_source(root, manifest_path, expected_commit):
    root = Path(root).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    if (
        not re.fullmatch(r"[0-9a-f]{40}", expected_commit)
        or manifest["source_commit"] != expected_commit
    ):
        raise ValueError("manifest/source commit mismatch")
    expected = manifest["files"]
    if not expected or any(not safe_relative(name) for name in expected):
        raise ValueError("invalid source inventory")
    for name, digest in expected.items():
        path = root / name
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or path.is_symlink()
            or not path.is_file()
            or sha(path) != digest
        ):
            raise ValueError(f"frozen source hash mismatch: {name}")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if actual != set(expected):
        raise ValueError("frozen source inventory mismatch")
    return {
        "commit": expected_commit,
        "manifest_sha256": sha(manifest_path),
        "files_verified": len(expected),
        "archive_sha256": manifest["archive_sha256"],
    }


def require_device(properties, count):
    if (
        count != 1
        or properties["name"] != GPU_NAME
        or properties["capability"] != [8, 0]
        or properties["sms"] != 42
        or properties["memory_bytes"] != GPU_BYTES
    ):
        raise ValueError(
            "requires the explicitly reserved A100 MIG 3g.40gb/42SM device"
        )


def command(kind, mode, data, output, qualification=False):
    if kind not in ("dense", "cst") or mode not in ("eager", "graph"):
        raise ValueError("unknown fixed comparison arm")
    result = [
        sys.executable,
        "-m",
        "benchmarks.training.higgs_confirmation",
        "--data",
        str(data),
        "--output",
        str(output),
        "--kind",
        kind,
        "--width",
        str(WIDTH),
        "--divisor",
        str(DIVISOR),
        "--seed",
        "17",
        "--lr",
        "0.001",
        "--mode",
        mode,
        "--epochs",
        "2" if qualification else "1",
    ]
    if qualification:
        result += [
            "--steps-per-epoch",
            "16",
            "--development-rows",
            "32",
            "--retain-epoch-checkpoints",
        ]
    return result


def run_logged(argv, log_path, *, cwd, env, timeout):
    """Foreground child with an incremental file log and a hard timeout."""
    print("START " + Path(log_path).name, flush=True)
    with Path(log_path).open("w") as log:
        subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
        )
    print("DONE " + Path(log_path).name, flush=True)


def require_trajectory(comparison, left_updates, right_updates, clocks, expected):
    if (
        not comparison["all_byte_equal"]
        or not (left_updates == right_updates == expected)
        or len(clocks) != 5
        or not all(clock == expected for clock in clocks)
    ):
        raise ValueError("whole-step eager/Graph model/gradient/Adam/clock mismatch")


def qualify(args, record, env):
    import torch
    from benchmarks.training.full_step_graph import exact, tensors
    from benchmarks.training.higgs_confirmation import make_model
    from benchmarks.training.owner_window_fp32 import errors, quadrature
    from benchmarks.training.radius_local_reference import local_reference
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    destination = args.output / "qualification"
    destination.mkdir()
    qualification = {
        "status": "running",
        "oracles": [],
        "trajectories": [],
        "tests": None,
    }
    record["qualification"] = qualification
    persist(args.output / "experiment.json", record)
    # Original stock/finite/metadata/topology/capture tests run unchanged.
    tests = [
        "tests/test_higgs_confirmation.py",
        "tests/test_higgs_graph_cohort.py",
        "tests/test_graph_topology_guard.py",
        "tests/test_graph_stock_qualification.py",
    ]
    run_logged(
        [sys.executable, "-m", "pytest", "-q", *tests, "-p", "no:cacheprovider"],
        destination / "gpu-tests.log",
        cwd=args.torchcst_source,
        env=env,
        timeout=1200,
    )
    qualification["tests"] = (
        (destination / "gpu-tests.log").read_text().splitlines()[-1]
    )
    persist(args.output / "experiment.json", record)
    torch.manual_seed(20261105)
    model = make_model(WIDTH, DIVISOR, 17, "cst")[2]
    parameters = model.atoms.p.detach().clone()
    if parameters.shape != (4194, 4):
        raise ValueError("unexpected exact low-density atom count")
    x, dy = torch.randn(4, WIDTH), torch.randn(4, WIDTH)
    reference = copy.deepcopy(model).double()
    layout = hierarchy_from_operator(model.declaration())
    for label, subset, quadrature_reference in (
        ("all-atoms-FP64", parameters, False),
        ("quadrature31", parameters[:31], True),
    ):
        rx, rp = x.double().requires_grad_(), subset.double().requires_grad_()
        expected_y = (
            quadrature(reference, rx, rp)
            if quadrature_reference
            else local_reference(reference, rx, rp)
        )
        expected = (
            expected_y.detach(),
            *torch.autograd.grad(expected_y, (rx, rp), dy.double()),
        )
        xx, pp = x.cuda().requires_grad_(), subset.cuda().requires_grad_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if quadrature_reference:
                entry = executor._extension().research_owner_window_fp32_geometry_sort32_flat_c
                actual_y = entry(
                    xx,
                    pp,
                    layout.input.native_config(),
                    layout.output.native_config(),
                    False,
                )
            else:
                entry = executor._extension().research_higgs_confirmation_owner_window_fp32_geometry_sort32_flat_c
                actual_y = entry(
                    xx,
                    pp,
                    layout.input.native_config(),
                    layout.output.native_config(),
                    True,
                )
            actual = (
                actual_y.detach().cpu(),
                *[
                    value.cpu()
                    for value in torch.autograd.grad(actual_y, (xx, pp), dy.cuda())
                ],
            )
        torch.cuda.synchronize()
        metrics = {
            name: errors(a, b)
            for name, a, b in zip(("Y", "dX", "dP"), actual, expected)
        }
        path = destination / (label + ".pt")
        torch.save(
            {"actual": actual, "expected": expected, "P": subset, "X": x, "dY": dy},
            path,
        )
        qualification["oracles"].append(
            {
                "name": label,
                "atoms": len(subset),
                "file": path.name,
                "sha256": sha(path),
                "metrics": metrics,
            }
        )
        persist(args.output / "experiment.json", record)
        if not all(metric["pass"] for metric in metrics.values()):
            raise ValueError("independent Y/dX/all-parameter oracle failed")
    for kind in ("dense", "cst"):
        paths = {}
        for mode in ("eager", "graph"):
            path = destination / (kind + "-" + mode)
            run_logged(
                command(kind, mode, args.data, path, True),
                destination / (kind + "-" + mode + ".log"),
                cwd=args.torchcst_source,
                env=env,
                timeout=1200,
            )
            paths[mode] = path
        for epoch in (1, 2):
            left, right = [
                torch.load(
                    paths[mode] / f"epoch-{epoch}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for mode in ("eager", "graph")
            ]
            comparison = exact(dict(tensors(left)), dict(tensors(right)))
            clocks = [
                float(item["step"]) for item in left["optimizer"]["state"].values()
            ]
            require_trajectory(
                comparison, left["updates"], right["updates"], clocks, epoch * 16
            )
            qualification["trajectories"].append(
                {
                    "kind": kind,
                    "epoch": epoch,
                    "updates": epoch * 16,
                    "comparison": comparison,
                    "clocks": clocks,
                }
            )
            persist(args.output / "experiment.json", record)
    qualification["status"] = "passed"
    persist(args.output / "experiment.json", record)
    persist(destination / "qualification.json", qualification)


def runtime():
    import torch

    properties = torch.cuda.get_device_properties(0)
    device = {
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "sms": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
    }
    require_device(device, torch.cuda.device_count())
    import triton

    return {
        "device": device,
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        ),
    }


def run(args):
    sys.dont_write_bytecode = True
    args.torchcst_source = args.torchcst_source.resolve()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if args.output.exists():
        raise FileExistsError("experiment output must be fresh")
    args.output.mkdir(parents=True)
    record = {
        "schema_version": 1,
        "study": "higgs-gpu-comparison-20261105",
        "status": "running",
        "qualification": None,
        "pilots": [],
        "confirmation_evaluated": False,
        "test_evaluated": False,
        "scope": "A100 MIG versus earlier L4 environment comparison; not hardware-only causality or a fully tuned quality study",
    }
    begin = time.perf_counter()
    try:
        record["torchcst_source"] = verify_source(
            args.torchcst_source, args.torchcst_manifest, SOURCE_COMMIT
        )
        record["adapter_source"] = verify_source(
            Path(__file__).resolve().parents[3],
            args.adapter_manifest,
            args.adapter_commit,
        )
        if sha(args.data) != DATA_SHA256:
            raise ValueError("prepared HIGGS identity differs")
        record["dataset_sha256"] = DATA_SHA256
        env = os.environ.copy()
        env.update(
            PYTHONDONTWRITEBYTECODE="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            MAX_JOBS="2",
            TORCH_CUDA_ARCH_LIST="8.0",
        )
        env["TORCH_EXTENSIONS_DIR"] = str(args.output.parent / "torch-extensions")
        os.environ["TORCH_EXTENSIONS_DIR"] = env["TORCH_EXTENSIONS_DIR"]
        env["PYTHONPATH"] = os.pathsep.join(
            (str(args.torchcst_source / "src"), str(args.torchcst_source))
        )
        os.environ.update(
            {
                key: env[key]
                for key in (
                    "MAX_JOBS",
                    "TORCH_CUDA_ARCH_LIST",
                    "PYTHONDONTWRITEBYTECODE",
                )
            }
        )
        sys.path[:0] = [str(args.torchcst_source / "src"), str(args.torchcst_source)]
        import torch

        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        record["runtime"] = runtime()
        persist(args.output / "experiment.json", record)
        qualify(args, record, env)
        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor

        binary = Path(executor._extension().__file__)
        retained_binary = args.output / binary.name
        shutil.copy2(binary, retained_binary)
        record["native_extension"] = {
            "file": retained_binary.name,
            "sha256": sha(retained_binary),
            "loaded_sha256": sha(binary),
        }
        persist(args.output / "experiment.json", record)
        # No pilot process may start before all numerical and trajectory gates pass.
        for kind in ("dense", "cst"):
            path = args.output / ("pilot-" + kind)
            run_logged(
                command(kind, "graph", args.data, path),
                args.output / ("pilot-" + kind + ".log"),
                cwd=args.torchcst_source,
                env=env,
                timeout=1200,
            )
            learning = json.loads((path / "learning.json").read_text())
            if (
                learning["total_updates"] != 25000
                or learning["training_rows"] != 100000
                or learning["development_rows"] != 50000
                or learning["confirmation_evaluated"]
                or learning["test_evaluated"]
            ):
                raise ValueError("pilot budget/partition differs")
            record["pilots"].append(
                {
                    "kind": kind,
                    "learning_sha256": sha(path / "learning.json"),
                    "learning": learning,
                }
            )
            persist(args.output / "experiment.json", record)
        # Confirm experiment execution did not mutate either frozen source tree.
        verify_source(args.torchcst_source, args.torchcst_manifest, SOURCE_COMMIT)
        verify_source(
            Path(__file__).resolve().parents[3],
            args.adapter_manifest,
            args.adapter_commit,
        )
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["wall_seconds_including_qualification"] = time.perf_counter() - begin
        persist(args.output / "experiment.json", record)
        manifest = {
            path.relative_to(args.output).as_posix(): sha(path)
            for path in sorted(args.output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        persist(args.output / "manifest.json", manifest)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchcst-source", type=Path, required=True)
    parser.add_argument("--torchcst-manifest", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--adapter-commit", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
