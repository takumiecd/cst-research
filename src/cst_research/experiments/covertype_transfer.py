"""Eager research-local Covertype cells over unchanged public torchcst975."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from .covertype_data import (
    DATA_SHA,
    HELDOUT_SHA,
    MANIFEST_SHA,
    array_record,
    load_development,
    specification,
)
from .higgs_depth_study import checkpoint, clone_cpu, reference_forward
from .higgs_dst_comparison import file_sha, json_sha, read, write

SOURCE = "97580582ed28c4e494bf5c2656bbe94a85c544c3"
SOURCE_MANIFEST = "17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931"
METHODS = ("dense", "cst-1000", "cst-100")
LRS = (0.0001, 0.0003, 0.001)
SEEDS = (29, 43, 71)
UPDATES_PER_EPOCH = 100000 // 4


def numerical_contract():
    return {
        "chart": {"geometry": "flat_torus", "spacing": 1.0},
        "kernel": {
            "composition": "product",
            "profile": "ChebyshevSpec",
            "modes": 8,
            "merge_factor": 2,
            "amplitude_max": 1.0,
            "w_c": 0.3,
            "radius_bounds": {
                "minimum": 0.5,
                "birth": 4.0,
                "maximum": 50.0,
                "upper_floor": 1.0,
            },
            "kappa": 3.0,
            "lower_kappa": 3.0,
            "upper_decay_power": 1.0,
            "alpha_init": 0.0,
            "radial_regularization": 0.1,
            "activity_gain": 1.0,
            "activity_mode": "finite_chord",
            "dormant_expansion_rate": 0.0,
        },
        "linear_plan": {
            "algorithm_id": "cuda_radius_cell_mass",
            "revision": "v1",
            "schedule": "global_owner_window_fp32_geometry_sort32_flat_c",
            "input_direct_levels": 2,
            "output_direct_levels": 2,
        },
        "update_plan": {
            "algorithm_id": "cuda_radius_flat_torus_polar_update_fused",
            "revision": "v2",
        },
        "optimizer": {
            "ordinary_weight_decay": 0.01,
            "atom_raw_weight_decay": 0.0,
            "amplitude_weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "fused": True,
            "capturable": False,
        },
        "initialization": {
            "ordinary_input": "seed+1",
            "ordinary_output": "seed+2",
            "middle": "seed+3",
            "ordinary": "torch.nn.Linear default reset_parameters",
            "atoms": "public CSTLinear FP32 default polar-radius initialization; no gain calibration",
        },
    }


def protocol():
    return {
        "version": "covertype-transfer-v1",
        "source_commit": SOURCE,
        "data": specification(),
        "prepared_artifacts": {
            "train_development_sha256": DATA_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "closed_heldout_sha256": HELDOUT_SHA,
        },
        "architecture": [54, 1024, 1024, 7],
        "numerical_contract": numerical_contract(),
        "middle_bias": False,
        "dense_io_trainable": True,
        "methods": list(METHODS),
        "atoms": [1048, 10485],
        "batch_size": 4,
        "dtype": "FP32",
        "TF32": False,
        "mode": "eager",
        "position_multiplier": 1,
        "development_seed": 17,
        "hpo_lrs": list(LRS),
        "hpo_epochs": 3,
        "single_edge": {"lower": 0.00003, "upper": 0.003},
        "retune": "all methods fresh top2 LR x10epochs",
        "confirmation_seeds": list(SEEDS),
        "final_epochs": 10,
        "selection": "lowest best-development CE; earliest epoch; lower LR on exact tie",
        "heldout_gate": "all9 frozen before explicit separately invoked evaluator; no formal NI",
        "metrics": ["CE", "accuracy", "macroF1", "macroOvRAUC"],
        "auc": "external975 binary_auc per class on float64 softmax probabilities",
        "selected_state_gap": "full train and dev evaluated at identical selected checkpoint outside training clock",
        "formal_noninferiority": False,
        "universal_alpha_optimum": False,
        "geographic_generalization": False,
    }


def configuration(method, width=1024):
    if method not in METHODS or width not in (128, 1024):
        raise ValueError("undeclared model configuration")
    divisor = None if method == "dense" else int(method.split("-")[1])
    return {
        "cohort": "covertype-transfer-v1",
        "id": method,
        "width": width,
        "divisor": divisor,
        "alpha": None if divisor is None else 1 / divisor,
        "atoms": None if divisor is None else width * width // divisor,
        "architecture": [54, width, width, 7],
    }


def make_model(method, seed, width=1024):
    import torch
    from torch import nn
    from torchcst import (
        ChebyshevSpec,
        CSTLinear,
        FixedSelector,
        RadiusBounds,
        chart_presets,
        kernels,
        make_algorithm_registry,
    )

    cfg = configuration(method, width)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        first = nn.Linear(54, width)
        torch.manual_seed(seed + 2)
        last = nn.Linear(width, 7)
        torch.manual_seed(seed + 3)
        if method == "dense":
            middle = nn.Linear(width, width, bias=False)
        else:
            import torchcst

            source = Path(torchcst.__file__).resolve().parents[2]
            catalogue = read(
                source / "benchmarks/cuda/linear/plans-radius-cell-mass.json"
            )["plans"]
            selected = next(
                row["plan"]
                for row in catalogue
                if row["plan"]["recipe"].get("schedule")
                == "global_owner_window_fp32_geometry_sort32_flat_c"
            )
            registry = make_algorithm_registry()
            chart = chart_presets.regular_grid(
                width, spacing=1.0, geometry="flat_torus"
            )
            kernel = kernels.product(
                kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
                coordinates=kernels.polar(
                    amplitude_max=1.0,
                    w_c=0.3,
                    input_bounds=RadiusBounds(
                        minimum=0.5, birth=4.0, maximum=50.0, upper_floor=1.0
                    ),
                ),
            )
            middle = CSTLinear(
                chart,
                chart,
                atoms=cfg["atoms"],
                kernel=kernel,
                dtype=torch.float32,
                selector=FixedSelector(registry.load_plan(selected), registry=registry),
            )
    return nn.Sequential(first, nn.ReLU(), middle, nn.ReLU(), last)


def make_optimizer(model, lr, cpu_reference=False):
    import torch
    from torchcst import CSTAdamW, FixedSelector, make_algorithm_registry
    from torchcst.nn import CSTModule

    atoms = [m.atoms.p for m in model.modules() if isinstance(m, CSTModule)]
    ids = {id(p) for p in atoms}
    ordinary = [p for p in model.parameters() if id(p) not in ids]
    groups = [{"params": ordinary, "weight_decay": 0.01, "role": "ordinary"}]
    if atoms:
        groups.append(
            {
                "params": atoms,
                "weight_decay": 0.0,
                "amplitude_weight_decay": 0.01,
                "role": "atoms",
            }
        )
    options = {
        "lr": lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "fused": not cpu_reference,
        "capturable": False,
        "amsgrad": False,
        "maximize": False,
    }
    if not atoms:
        return torch.optim.AdamW(groups, **options)
    registry = make_algorithm_registry()
    kwargs = (
        {}
        if cpu_reference
        else {
            "update_selector": FixedSelector(
                registry.load_plan(
                    {
                        "schema_version": 1,
                        "algorithm_id": "cuda_radius_flat_torus_polar_update_fused",
                        "algorithm_revision": "v2",
                        "recipe": {},
                    }
                ),
                registry=registry,
            )
        }
    )
    return CSTAdamW(
        groups, model=model, check_finite=cpu_reference, **kwargs, **options
    )


def metrics(logits, labels, binary_auc=None):
    import numpy as np

    if binary_auc is None:
        from benchmarks.training.higgs_density import binary_auc
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if (
        logits.shape != (len(labels), 7)
        or labels.ndim != 1
        or labels.dtype.kind not in "iu"
        or not np.isfinite(logits).all()
        or set(np.unique(labels)) != set(range(7))
    ):
        raise ValueError("seven finite class scores and all seven labels required")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    losses = np.log(exp.sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    predictions = logits.argmax(axis=1)
    confusion = np.bincount(7 * labels + predictions, minlength=49).reshape(7, 7)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(
        2 * np.diag(confusion),
        denominator,
        out=np.zeros(7, dtype=np.float64),
        where=denominator > 0,
    )
    aucs = [
        binary_auc(probabilities[:, k], (labels == k).astype(np.int64))
        for k in range(7)
    ]
    return {
        "loss": float(losses.mean()),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1.mean()),
        "macro_ovr_auc": float(np.mean(aucs)),
        "per_class_auc": aucs,
        "confusion_matrix": confusion.tolist(),
        "examples": len(labels),
    }


def evaluate(model, x, y):
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        logits = torch.cat(
            [model(batch).detach().cpu() for batch in x.split(4)]
        ).numpy()
    labels = y.detach().cpu().numpy()
    result = metrics(logits, labels)
    torch.cuda.synchronize()
    return result, logits, labels, time.perf_counter() - start


def runtime(device):
    import torch

    if torch.cuda.device_count() != 1:
        raise ValueError("exactly one reviewed CUDA device must be visible")
    properties = torch.cuda.get_device_properties(device)
    drivers = set(
        subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(drivers) != 1:
        raise ValueError("unambiguous NVIDIA driver version required")
    nvcc = subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "python": platform.python_version(),
        "triton": importlib.metadata.version("triton"),
        "numpy": importlib.metadata.version("numpy"),
        "driver": next(iter(drivers)),
        "nvcc": nvcc,
        "gpu_profile": {
            "capability": list(torch.cuda.get_device_capability(device)),
            "multiprocessors": properties.multi_processor_count,
            "memory_bytes": properties.total_memory,
        },
    }


def validate_checkpoint(raw, row, identity):
    import torch
    from benchmarks.training.higgs_verify import finite_fields, tensor_fields

    config = raw["configuration"]
    cfg = configuration(row["method"])
    if (
        any(config.get(k) != v for k, v in identity.items())
        or any(config.get(k) != v for k, v in cfg.items())
        or config["seed"] != row["seed"]
        or config["lr"] != row["lr"]
        or config["epochs"] != 10
        or raw["epoch"] != row["best_epoch"]
        or type(row["best_epoch"]) is not int
        or not 1 <= row["best_epoch"] <= 10
        or raw["updates"] != UPDATES_PER_EPOCH * row["best_epoch"]
        or raw["development"] != row["development"]
        or config.get("numerical_contract") != numerical_contract()
    ):
        raise ValueError("selected raw checkpoint configuration differs")
    states = raw["optimizer"]["state"]
    if (
        len(states) != 5
        or any(float(s["step"]) != raw["updates"] for s in states.values())
        or finite_fields(dict(tensor_fields(raw)))
    ):
        raise ValueError("selected raw optimizer clocks/finite gate failed")
    model = raw["model"]
    middle = "2.weight" if row["method"] == "dense" else "2.atom_state.atoms.p"
    middle_shape = (1024, 1024) if row["method"] == "dense" else (cfg["atoms"], 4)
    if (
        tuple(model["0.weight"].shape) != (1024, 54)
        or tuple(model["0.bias"].shape) != (1024,)
        or tuple(model["4.bias"].shape) != (7,)
        or tuple(model["4.weight"].shape) != (7, 1024)
        or tuple(model[middle].shape) != middle_shape
    ):
        raise ValueError("selected raw model architecture differs")
    expected_groups = (
        [("ordinary", ("0.weight", "0.bias", "2.weight", "4.weight", "4.bias"))]
        if row["method"] == "dense"
        else [
            ("ordinary", ("0.weight", "0.bias", "4.weight", "4.bias")),
            ("atoms", (middle,)),
        ]
    )
    groups = raw["optimizer"]["param_groups"]
    gradients = raw["gradients"]
    expected_names = [name for _, names in expected_groups for name in names]
    owned = [identifier for group in groups for identifier in group["params"]]
    if (
        len(groups) != len(expected_groups)
        or len(owned) != len(set(owned))
        or set(owned) != set(states)
        or set(gradients) != set(expected_names)
    ):
        raise ValueError("selected optimizer/gradient ownership differs")
    for group, (role, names) in zip(groups, expected_groups):
        if (
            group.get("role") != role
            or len(group["params"]) != len(names)
            or group["lr"] != row["lr"]
            or tuple(group["betas"]) != (0.9, 0.999)
            or group["eps"] != 1e-8
            or group["amsgrad"] is not False
            or group["weight_decay"] != (0.0 if role == "atoms" else 0.01)
            or (role == "atoms" and group["amplitude_weight_decay"] != 0.01)
        ):
            raise ValueError("selected optimizer group contract differs")
        for identifier, name in zip(group["params"], names):
            point = model[name]
            state = states[identifier]
            gradient = gradients[name]
            if (
                point.dtype != torch.float32
                or not isinstance(gradient, torch.Tensor)
                or gradient.shape != point.shape
                or gradient.dtype != point.dtype
                or set(state) != {"step", "exp_avg", "exp_avg_sq"}
            ):
                raise ValueError("selected gradient/moment schema differs")
            if (
                state["step"].ndim != 0
                or state["step"].dtype != torch.float32
                or any(
                    state[key].shape != point.shape or state[key].dtype != point.dtype
                    for key in ("exp_avg", "exp_avg_sq")
                )
            ):
                raise ValueError("selected moment shape/dtype differs")
    if row["method"] != "dense":
        from torchcst import CSTAdamW

        manifest = raw["optimizer"].get("cst", {})
        contract = f"{CSTAdamW.__module__}.{CSTAdamW.__qualname__}"
        expected_manifest_groups = [
            [(name, tuple(model[name].shape)) for name in names]
            for _, names in expected_groups
        ]
        store = model["2.atom_state.optimizer_state._extra_state"]
        atom_state = states[groups[1]["params"][0]]
        if (
            manifest.get("optimizer_type") != contract
            or manifest.get("groups") != expected_manifest_groups
            or store.get("format_version") != 1
            or store.get("atom_count") != cfg["atoms"]
            or store.get("contract") != contract
            or manifest.get("state_specs") != store.get("field_specs")
            or store.get("field_specs", {}).get("exp_avg") != 0
            or store.get("field_specs", {}).get("exp_avg_sq") != 0
            or store.get("field_specs", {}).get("step", "missing") is not None
            or set(store.get("fields", {})) != set(atom_state)
            or any(
                store["fields"][key].dtype != value.dtype
                or store["fields"][key].shape != value.shape
                or not torch.equal(store["fields"][key], value)
                for key, value in atom_state.items()
            )
        ):
            raise ValueError("selected CST model-side optimizer store differs")


def verify_selected_development(directory, best, result, logits, labels, row_ids):
    import numpy as np

    current = {"logits": logits, "labels": labels, "row_ids": row_ids}
    with np.load(directory / "best-dev-predictions.npz", allow_pickle=False) as source:
        if set(source.files) != set(current):
            raise ValueError("saved selected prediction inventory differs")
        saved = {name: source[name].copy() for name in source.files}
    records = {name: array_record(value) for name, value in current.items()}
    if result != best["development"] or any(
        records[name] != array_record(saved[name]) for name in current
    ):
        raise ValueError("selected-development reload predictions/metadata differ")
    proof = {
        "status": "passed",
        "checkpoint_sha256": best["checkpoint_sha256"],
        "array_bytes": records,
        "metrics_equal": True,
        "comparison": "dtype/shape/C-order array bytes; NPZ archive byte equality not required",
    }
    write(directory / "selected-development-verification.json", proof)
    return proof


def qualify(cfg, seed, lr, output, device):
    import torch
    from benchmarks.training.full_step_graph import ordinary_step
    from benchmarks.training.higgs_verify import (
        byte_equal,
        finite_fields,
        tensor_fields,
    )
    from benchmarks.training.owner_window_fp32 import errors

    output.mkdir()
    generator = torch.Generator().manual_seed(seed + 800)
    model = make_model(cfg["id"], seed).to(device)
    x = torch.randn((4, 54), generator=generator).to(device).requires_grad_()
    reference = copy.deepcopy(model).cpu().double()
    rx = x.detach().cpu().double().requires_grad_()
    dy = torch.randn((4, 7), generator=generator).to(device)
    actual_y = model(x)
    expected_y = reference_forward(reference, rx)
    actual = (actual_y, *torch.autograd.grad(actual_y, (x, *model.parameters()), dy))
    expected = (
        expected_y,
        *torch.autograd.grad(
            expected_y, (rx, *reference.parameters()), dy.cpu().double()
        ),
    )
    comparisons = [errors(a, b) for a, b in zip(actual, expected)]
    torch.save(
        {
            "actual": clone_cpu(actual),
            "expected": clone_cpu(expected),
            "configuration": cfg,
        },
        output / "full-network-vjp.pt",
    )
    if not all(c["pass"] for c in comparisons):
        raise ValueError(
            "Covertype independent public dense_weight Y/dX/allgradient gate failed"
        )
    model = make_model(cfg["id"], seed).to(device)
    optimizer = make_optimizer(model, lr)
    tx = torch.randn((32, 54), generator=generator).to(device)
    ty = torch.arange(32, device=device) % 7
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    midway = None
    for step in range(8):
        ordinary_step(
            model,
            optimizer,
            tx,
            ty,
            torch.arange(4 * step, 4 * step + 4, device=device),
            totals,
        )
        if step == 3:
            midway = checkpoint(model, optimizer, 0, 4, cfg, None)
    final = checkpoint(model, optimizer, 0, 8, cfg, None)
    resumed = make_model(cfg["id"], seed).to(device)
    restored = make_optimizer(resumed, lr)
    resumed.load_state_dict(midway["model"])
    restored.load_state_dict(midway["optimizer"])
    for step in range(4, 8):
        ordinary_step(
            resumed,
            restored,
            tx,
            ty,
            torch.arange(4 * step, 4 * step + 4, device=device),
            totals,
        )
    after = checkpoint(resumed, restored, 0, 8, cfg, None)
    a, b = dict(tensor_fields(final)), dict(tensor_fields(after))
    equal = a.keys() == b.keys() and all(byte_equal(a[k], b[k]) for k in a)
    if (
        not equal
        or finite_fields(a)
        or len(optimizer.state) != 5
        or any(float(s["step"]) != 8 for s in optimizer.state.values())
    ):
        raise ValueError(
            "Covertype eight/four resume/finite/ownership/clock gate failed"
        )
    torch.save(
        {"midway": midway, "final": final, "resumed": after}, output / "eager-resume.pt"
    )
    record = {
        "status": "passed",
        "configuration": cfg,
        "vjp": comparisons,
        "eight_steps_four_resume_tensor_equal": equal,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    write(output / "qualification.json", record)
    return record


def admit_sources(args):
    from . import higgs_dst_confirmation_driver as common
    from .higgs_coordinate_lr import verify_external

    consumer = Path(__file__).resolve().parents[3]
    manifest = read(args.consumer_manifest)
    if (
        not re.fullmatch(r"[0-9a-f]{40}", args.consumer_commit)
        or manifest.get("commit") != args.consumer_commit
    ):
        raise ValueError("full committed consumer identity required")
    files = common.inventory(consumer)
    files = {
        n: p
        for n, p in files.items()
        if not n.startswith("research-inputs/") and n != "__pool_driver__.py"
    }
    if set(files) != set(manifest["files"]) or any(
        file_sha(common.safe_path(consumer, n)) != v
        for n, v in manifest["files"].items()
    ):
        raise ValueError("immutable Covertype consumer source differs")
    if (
        file_sha(Path(__file__))
        != manifest["files"]["src/cst_research/experiments/covertype_transfer.py"]
    ):
        raise ValueError("executed Covertype module differs")
    if (
        args.data.name != "train-development.npz"
        or file_sha(args.data) != DATA_SHA
        or file_sha(args.data_manifest) != args.data_manifest_sha256
        or args.data_manifest_sha256 != MANIFEST_SHA
    ):
        raise ValueError("frozen train/development payload differs")
    verify_external(
        SimpleNamespace(
            external_source=args.external_source,
            external_manifest=args.external_manifest,
            identity={"external_source_manifest_sha256": SOURCE_MANIFEST},
        )
    )
    sys.path[:0] = [str(args.external_source / "src"), str(args.external_source)]
    return {
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "external_commit": SOURCE,
        "external_manifest_sha256": SOURCE_MANIFEST,
        "data_sha256": file_sha(args.data),
        "data_manifest_sha256": args.data_manifest_sha256,
        "protocol_sha256": json_sha(protocol()),
    }


def run(args):
    startup_begin = time.perf_counter()
    identity = admit_sources(args)
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("cell output must be outside immutable source")
    import numpy as np
    import torch
    from benchmarks.training.full_step_graph import ordinary_step
    from benchmarks.training.higgs_confirmation import _configure
    from benchmarks.training.higgs_verify import (
        byte_equal,
        finite_fields,
        tensor_fields,
        tensor_hash,
    )

    cfg = configuration(args.method)
    if (
        args.epochs not in (3, 10)
        or args.seed not in (17, *SEEDS)
        or args.lr not in (*LRS, 0.00003, 0.003)
    ):
        raise ValueError("undeclared cell budget")
    if args.seed != 17 and args.epochs != 10:
        raise ValueError("paired seeds use10epochs")
    arrays, _ = load_development(
        args.data, args.data_manifest, args.data_manifest_sha256
    )
    args.output.mkdir(parents=True, exist_ok=False)
    _configure()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(
            "GPU training requires explicit device admission; CPU checks are separate"
        )
    observed_runtime = runtime(device)
    if (
        getattr(args, "expected_runtime", None) is not None
        and args.expected_runtime != observed_runtime
    ):
        raise ValueError("campaign runtime changed before qualification/training")
    startup_seconds = time.perf_counter() - startup_begin
    start = time.perf_counter()
    qual = qualify(cfg, args.seed, args.lr, args.output / "qualification", device)
    qualification_seconds = time.perf_counter() - start
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    model = make_model(args.method, args.seed).to(device)
    optimizer = make_optimizer(model, args.lr)
    config = {
        **cfg,
        **identity,
        "numerical_contract": numerical_contract(),
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": 4,
        "mode": "eager",
        "TF32": False,
    }
    torch.save(
        checkpoint(model, optimizer, 0, 0, config, None),
        args.output / "initialization.pt",
    )
    tx, ty, vx, vy = (
        torch.from_numpy(arrays[k]).to(device)
        for k in ("x_train", "y_train", "x_development", "y_development")
    )
    preparation_seconds = time.perf_counter() - start - qualification_seconds
    rows = []
    best = None
    updates = 0
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(
            len(tx),
            generator=torch.Generator().manual_seed(args.seed + 100 + epoch - 1),
        )
        indices = order.to(device)
        for total in totals:
            total.zero_()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        for step in range(UPDATES_PER_EPOCH):
            ordinary_step(
                model, optimizer, tx, ty, indices[step * 4 : step * 4 + 4], totals
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        updates += UPDATES_PER_EPOCH
        memory = {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        }
        raw = checkpoint(model, optimizer, epoch, updates, config, None)
        fields = dict(tensor_fields(raw))
        if (
            finite_fields(fields)
            or len(optimizer.state) != 5
            or any(float(s["step"]) != updates for s in optimizer.state.values())
        ):
            raise ValueError("training finite/clock gate failed")
        dev, logits, labels, evaluation_seconds = evaluate(model, vx, vy)
        after = checkpoint(model, optimizer, epoch, updates, config, None)
        after_fields = dict(tensor_fields(after))

        if fields.keys() != after_fields.keys() or any(
            not byte_equal(value, after_fields[name]) for name, value in fields.items()
        ):
            raise ValueError("evaluation changed model/optimizer/gradient state")
        raw["development"] = dev
        torch.save(raw, args.output / f"epoch-{epoch}.pt")
        row = {
            "epoch": epoch,
            "updates": updates,
            "online_train_loss": float(totals[0]) / 100000,
            "online_train_accuracy": float(totals[1]) / 100000,
            "development": dev,
            "training_seconds": seconds,
            "evaluation_seconds": evaluation_seconds,
            "peak_bytes": memory,
            "order_sha256": tensor_hash(order)["sha256"],
        }
        rows.append(row)
        if best is None or dev["loss"] < best["development"]["loss"]:
            torch.save(raw, args.output / "best-dev.pt")
            np.savez_compressed(
                args.output / "best-dev-predictions.npz",
                logits=logits,
                labels=labels,
                row_ids=arrays["row_ids_development"],
            )
            best = {
                "epoch": epoch,
                "development": dev,
                "checkpoint_sha256": file_sha(args.output / "best-dev.pt"),
                "predictions_sha256": file_sha(
                    args.output / "best-dev-predictions.npz"
                ),
            }
            write(args.output / "best-dev.json", best)
        learning = {
            "configuration": config,
            "epochs": rows,
            "best": best,
            "qualification": qual,
            "qualification_seconds": qualification_seconds,
            "preparation_seconds": preparation_seconds,
            "startup_seconds": startup_seconds,
            **observed_runtime,
            "heldout_evaluated": False,
            "timing_scope": "synchronized complete B4 eager step; evaluation/preparation/I/O separately excluded",
        }
        write(args.output / "learning.json", learning)
    shutil = __import__("shutil")
    shutil.copy2(args.output / f"epoch-{args.epochs}.pt", args.output / "final.pt")
    model.load_state_dict(
        torch.load(args.output / "best-dev.pt", map_location=device, weights_only=True)[
            "model"
        ]
    )
    selected = {}
    selected_tensors = dict(tensor_fields(clone_cpu(model.state_dict())))
    for name, x, y in (("train", tx, ty), ("development", vx, vy)):
        result, logits, labels, seconds = evaluate(model, x, y)
        after = dict(tensor_fields(clone_cpu(model.state_dict())))
        if selected_tensors.keys() != after.keys() or any(
            not byte_equal(value, after[name])
            for name, value in selected_tensors.items()
        ):
            raise ValueError("same-selected-state evaluation changed model tensors")
        if name == "development":
            verify_selected_development(
                args.output, best, result, logits, labels, arrays["row_ids_development"]
            )
        np.savez_compressed(
            args.output / f"selected-{name}-predictions.npz",
            logits=logits,
            labels=labels,
            row_ids=arrays["row_ids_" + name],
        )
        selected[name] = {
            "metrics": result,
            "evaluation_seconds": seconds,
            "predictions_sha256": file_sha(
                args.output / f"selected-{name}-predictions.npz"
            ),
        }
    selected["checkpoint_sha256"] = best["checkpoint_sha256"]
    selected["CE_gap"] = (
        selected["development"]["metrics"]["loss"]
        - selected["train"]["metrics"]["loss"]
    )
    write(args.output / "selected-state-gap.json", selected)
    return learning


def add_common(parser):
    for name in (
        "consumer-manifest",
        "external-source",
        "external-manifest",
        "data",
        "data-manifest",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--device", default="cuda")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
