"""New native16 qualification only; no task training or inherited admission ticket."""

from __future__ import annotations

import copy
import gc
import hashlib
from pathlib import Path

from . import higgs_depth_gain as norms
from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_linear_atom_scaling as study
from . import higgs_linear_atom_state as state
from .higgs_dst_comparison import file_sha, write
from .higgs_linear_atom_reference import full_reference

SCALES = (1.0, 1e24)
RAW_BUDGET = 8 * 2**30


def budget(root):
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if size > RAW_BUDGET:
        raise ValueError(
            "explicit8GiB retained-raw budget exceeded; preserve partial evidence"
        )
    return size


def gates(actual, expected, fields, scale):
    from benchmarks.training.owner_window_fp32 import errors

    component = {
        name: errors(a, r) for name, a, r in zip(fields, actual, expected, strict=True)
    }
    relative = {}
    coverage = {}
    rounding = {}
    for name, a, r in zip(fields[1:], actual[1:], expected[1:], strict=True):
        relative[name] = norms.gradient_norm_gate(a, r)
        if name.endswith("atoms.p"):
            relative[name]["packed_columns"] = [
                norms.gradient_norm_gate(a[:, c], r[:, c]) for c in range(4)
            ]
        nonzero = r != 0
        rounding[name] = {
            "reference_nonzero_elements": int(nonzero.sum()),
            "actual_zero_among_reference_nonzero": int(((a == 0) & nonzero).sum()),
            "reference_nonzero_FP32_cast_zero": int(((r.float() == 0) & nonzero).sum()),
        }
        if scale == SCALES[1] and (name == "dX" or name.endswith("atoms.p")):
            coverage[name] = bool((r.abs() > 3e-5 + 3e-4 * r.abs()).any()) and bool(
                (a != 0).any()
            )
            if name.endswith("atoms.p"):
                for c in range(4):
                    coverage[name + f"/column{c}"] = bool(
                        (r[:, c].abs() > 3e-5 + 3e-4 * r[:, c].abs()).any()
                    ) and bool((a[:, c] != 0).any())
    passed = (
        (scale != 1 or all(v["pass"] for v in component.values()))
        and all(
            v["pass"] and all(p["pass"] for p in v.get("packed_columns", []))
            for v in relative.values()
        )
        and all(coverage.values())
    )
    return {
        "scale": scale,
        "normal_elementwise_mandatory": scale == 1,
        "amplified_componentwise_is_diagnostic": scale != 1,
        "elementwise": component,
        "relative_fields_and_packed_columns": relative,
        "critical_material_coverage": coverage,
        "rounded_zero_tail_disclosure": rounding,
        "pass": passed,
        "gradients_applied_to_optimizer": False,
    }


def quadrature_check(model, case, output, device):
    import torch
    from benchmarks.training.cell_mass_mnist import make_radius_cst
    from benchmarks.training.owner_window_fp32 import errors, quadrature
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans
    from torchcst.nn import CSTModule

    if case["method"] == "dense":
        return None
    label = f"N{case['width']}-c{case['c']}"
    path = output / (label + ".pt")
    marker = output / (label + ".json")
    site = next(m for m in model if isinstance(m, CSTModule))
    p = site.atoms.p[:31].detach().cpu().clone()
    digest = hashlib.sha256(p.numpy().tobytes()).hexdigest()
    if marker.exists():
        from .higgs_dst_comparison import read

        saved = read(marker)
        exact.require(
            saved["status"] == "passed"
            and saved["first31_P_sha256"] == digest
            and file_sha(path) == saved["raw_sha256"],
            "shared quadrature raw/point identity changed",
        )
        return saved
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2917)
        small = make_radius_cst(
            case["width"],
            31,
            execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
        )
    with torch.no_grad():
        small.atoms.p.copy_(p)
    reference = copy.deepcopy(small).double()
    generator = torch.Generator().manual_seed(1917 + case["width"] + case["c"])
    x = torch.randn(4, case["width"], generator=generator)
    dy = torch.randn(4, case["width"], generator=generator)
    rx = x.double().requires_grad_()
    rp = p.double().requires_grad_()
    ry = quadrature(reference, rx, rp)
    expected = (ry.detach(), *torch.autograd.grad(ry, (rx, rp), dy.double()))
    small = small.to(device)
    xx = x.to(device).requires_grad_()
    yy = small(xx)
    actual = (
        yy.detach().cpu(),
        *[
            value.detach().cpu()
            for value in torch.autograd.grad(yy, (xx, small.atoms.p), dy.to(device))
        ],
    )
    report = {
        name: errors(a, r)
        for name, a, r in zip(("Y", "dX", "all31_dP"), actual, expected, strict=True)
    }
    torch.save({"P": p, "X": x, "dY": dy, "actual": actual, "expected": expected}, path)
    record = {
        "status": "passed" if all(v["pass"] for v in report.values()) else "failed",
        "label": label,
        "first31_P_sha256": digest,
        "reference": "unchanged975 independent48-point polynomial quadrature",
        "errors": report,
        "raw_sha256": file_sha(path),
    }
    write(marker, record)
    exact.require(
        record["status"] == "passed", "independent quadrature31 bridge failed"
    )
    return record


def qualify_case(case, identity, output, shared_quadrature, device):
    import torch
    from benchmarks.training.full_step_graph import ordinary_step
    from benchmarks.training.higgs_verify import (
        byte_equal,
        finite_fields,
        tensor_fields,
    )

    output.mkdir()
    model = study.make_case(case).to(device)
    optimizer, _ = state.make_optimizer(model)
    initial = state.snapshot(model, optimizer, 0, 0, case, identity)
    torch.save(initial, output / "initial-state.pt")
    exact.require(
        len(list(model.parameters())) == case["hidden_layers"] + 3
        and sum(p.numel() for p in model.parameters())
        == model._linear_atom_configuration["trainable_scalars"],
        "exact parameter ownership/algebra differs",
    )
    quadrature = quadrature_check(model, case, shared_quadrature, device)
    reference = copy.deepcopy(model).cpu().double()
    probe = torch.randn(4, 28, generator=torch.Generator().manual_seed(817))
    names = ["Y", "dX", *[name for name, _ in model.named_parameters()]]
    reports = []
    forward_bytes = []
    reference_forward_bytes = []
    for scale in SCALES:
        x = probe.to(device).requires_grad_()
        y = model(x)
        dy = (
            torch.tensor(
                [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]],
                dtype=torch.float32,
                device=device,
            )
            * scale
        )
        gradients = torch.autograd.grad(y, (x, *model.parameters()), dy)
        actual = (y.detach().cpu(), *[g.detach().cpu() for g in gradients])
        ry, rdx, rp = full_reference(
            reference, probe.double(), dy.detach().cpu().double(), chunk_atoms=256
        )
        expected = (ry, rdx, *rp.values())
        exact.require(list(rp) == names[2:], "oracle all-parameter inventory differs")
        raw_file = f"vjp-scale-{scale:.0e}.pt"
        torch.save(
            {
                "actual": actual,
                "expected": expected,
                "fields": names,
                "X": probe,
                "dY": dy.detach().cpu(),
                "scale": scale,
                "reference": "full actualN/allatoms/FP64 onechunk256/manual fullreverse",
                "no_amplified_optimizer_step": True,
            },
            output / raw_file,
        )
        report = gates(actual, expected, names, scale)
        report["raw_file"] = raw_file
        report["raw_sha256"] = file_sha(output / raw_file)
        reports.append(report)
        write(output / "vjp-report.json", reports)
        exact.require(
            report["pass"],
            "new full-network normal/relative/material oracle failed; no tolerance fallback",
        )
        forward_bytes.append(actual[0])
        reference_forward_bytes.append(expected[0])
        exact.require(
            all(p.grad is None for p in model.parameters()),
            "autograd.grad qualification leaked into optimizer gradients",
        )
        del gradients, actual, expected, dy, y, x, ry, rdx, rp
    exact.require(
        byte_equal(*forward_bytes) and byte_equal(*reference_forward_bytes),
        "amplification changed forward bytes",
    )
    exact.assert_equal(
        state.core_state(state.snapshot(model, optimizer, 0, 0, case, identity)),
        state.core_state(initial),
        "VJP-mutated-initial-native-state",
    )
    initial_fingerprint = exact.fingerprint(state.core_state(initial))
    del model, optimizer, reference, initial, forward_bytes, reference_forward_bytes
    gc.collect()
    torch.cuda.empty_cache()
    # Fresh models: amplified VJPs can never become Adam inputs/moments.
    model = study.make_case(case).to(device)
    optimizer, _ = state.make_optimizer(model)
    exact.assert_equal(
        exact.fingerprint(
            state.core_state(state.snapshot(model, optimizer, 0, 0, case, identity))
        ),
        initial_fingerprint,
        "fresh-normal-qualification-state0",
    )
    probe = torch.randn(32, 28, generator=torch.Generator().manual_seed(1817)).to(
        device
    )
    labels = torch.arange(32, device=device) % 2
    torch.save(
        {
            "probe": probe.detach().cpu(),
            "labels": labels.detach().cpu(),
            "B4_orders": torch.arange(32).reshape(8, 4),
            "generator_seed": 1817,
            "role": "synthetic normal CE qualification only; not task data",
        },
        output / "normal-inputs.pt",
    )
    totals = (probe.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    midway = None
    for step in range(8):
        ordinary_step(
            model,
            optimizer,
            probe,
            labels,
            torch.arange(4 * step, 4 * step + 4, device=device),
            totals,
        )
        state.check_optimizer(model, optimizer, step + 1)
        if step == 3:
            midway = state.snapshot(model, optimizer, 0, 4, case, identity)
    final = state.snapshot(model, optimizer, 0, 8, case, identity)
    torch.save(midway, output / "normal-step4.pt")
    torch.save(final, output / "normal-step8.pt")
    resumed, restored, _, load_report = state.load_state(case, midway, device)
    loaded_versions = state.owner_versions(resumed, case)
    for step in range(4, 8):
        ordinary_step(
            resumed,
            restored,
            probe,
            labels,
            torch.arange(4 * step, 4 * step + 4, device=device),
            totals,
        )
        state.check_optimizer(resumed, restored, step + 1)
    after = state.snapshot(resumed, restored, 0, 8, case, identity)
    torch.save(after, output / "resumed-step8.pt")
    exact.require(
        state.owner_versions(resumed, case) == loaded_versions,
        "generation changed after native load boundary",
    )
    exact.assert_equal(
        state.core_state(after),
        state.with_generations(final, loaded_versions),
        "8-vs4-native-load-plus4-all-other-bytes",
    )
    for snap in (midway, final, after):
        exact.require(
            not finite_fields(dict(tensor_fields(snap))),
            "nonfinite native state/gradients/moments",
        )
        exact.require(
            bool((snap["model"]["fixed_output_gains"] == 1).all()),
            "unit fixed gains mutated",
        )
    record = {
        "status": "passed-new-native-case-qualification",
        "case": case,
        "identity": identity,
        "VJP_scales": list(SCALES),
        "normal_elementwise_and_relative_packed_gates_passed": True,
        "amplified_componentwise_failures_preserved_as_diagnostic": reports[1][
            "elementwise"
        ],
        "amplified_VJP_never_used_for_optimizer": True,
        "normal_true_B4_loss_updates": 8,
        "synthetic_normal_inputs_file": "normal-inputs.pt",
        "synthetic_normal_inputs_sha256": file_sha(output / "normal-inputs.pt"),
        "totals_scope": "scratch accumulator includes8 uninterrupted plus4 resume steps; no learning metric claimed; admission compares model/grad/moments/clocks",
        "additional_resumed_normal_loss_updates": 4,
        "owned_point_tensors": case["hidden_layers"] + 3,
        "native_load": load_report,
        "all_other_resume_bytes_exact": True,
        "quadrature31": quadrature,
        "files": {p.name: file_sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        "task_dataset_or_development_evaluated": False,
        "public_Graph_qualification_claim": False,
    }
    write(output / "qualification.json", record)
    del model, optimizer, resumed, restored, midway, final, after, probe, labels
    gc.collect()
    torch.cuda.empty_cache()
    return record


def qualify_all(identity, output, device):
    import torch

    from .higgs_depth_normalization_run import scalar_adam_gate

    output.mkdir()
    scalar = output / "native-scalar-Adam"
    scalar.mkdir()
    scalar_adam_gate(scalar, device)
    shared = output / "independent-quadrature31"
    shared.mkdir()
    cases = output / "cases"
    cases.mkdir()
    passed = []
    for case in study.pilot_cases():
        print("native qualification " + case["id"], flush=True)
        result = qualify_case(case, identity, cases / case["id"], shared, device)
        passed.append(
            {
                "case": case,
                "qualification_sha256": file_sha(
                    cases / case["id"] / "qualification.json"
                ),
                "initial_state_sha256": result["files"]["initial-state.pt"],
            }
        )
        write(
            output / "progress.json",
            {
                "status": "qualification-only-in-progress",
                "passed_case_ids": [p["case"]["id"] for p in passed],
                "task_training_updates": 0,
            },
        )
        budget(output)
    source = Path(__file__).resolve().parent
    sources = {p.name: file_sha(p) for p in source.glob("higgs_linear_atom*.py")}
    marker = {
        "status": "passed-all16-new-native-qualification-only-NOT-400k-training",
        "identity": identity,
        "cases": passed,
        "qualified_factory_reference_state_qualifier_source": sources,
        "synthetic_normal_updates_per_case": 8,
        "additional_resume_updates_per_case": 4,
        "normal_VJP_scale": 1.0,
        "amplified_VJP_scale": 1e24,
        "amplified_used_for_optimizer": False,
        "native_scalar_Adam_passed": True,
        "raw_bytes": budget(output),
        "task_training_updates": 0,
        "task_dataset_or_development_evaluated": False,
        "initial_state8_used_to_start_task_training": False,
        "training_entrypoint_phase_not_implemented_here": True,
        "runtime_cuda_arch": list(torch.cuda.get_device_capability(0)),
    }
    write(output / "all16-native-qualification.json", marker)
    return marker


RECIPE_AUDIT_SHA = "eefdef7e8b7598232c137a6f8589c74f1a511051d13bcbd8fe0b5a9a01dc8e84"
COTANGENT_DECISION_SHA = (
    "b86e9d69f211768239aaeda30b76863d16519f1bb13ee27ce0a25b30f79eeb31"
)
ENTRY = "src/cst_research/experiments/higgs_linear_atom_driver.py"


def input_pins():
    from .higgs_depth_data import DATA_SHA, MANIFEST_SHA
    from .higgs_depth_normalization_driver import ARCHIVE_SHA
    from .higgs_depth_normalization_learning_evidence import RUNTIME_SHA
    from .higgs_depth_study import SOURCE_MANIFEST

    return {
        "original-source.tar.gz": ARCHIVE_SHA,
        "original-source-manifest.json": SOURCE_MANIFEST,
        "train-development.npz": DATA_SHA,
        "data-manifest.json": MANIFEST_SHA,
        "runtime-pin.json": RUNTIME_SHA,
        "recipe-selection-audit.json": RECIPE_AUDIT_SHA,
        "cotangent-decision.json": COTANGENT_DECISION_SHA,
    }


def verify_inputs(inputs):
    """Hash exactly8 inputs; qualification never decodes task NPZ arrays."""
    from .higgs_depth_data import validate_manifest
    from .higgs_dst_comparison import read

    pins = input_pins()
    exact.require(
        {p.name for p in inputs.iterdir()} == {*pins, "consumer-source-manifest.json"}
        and all(p.is_file() and not p.is_symlink() for p in inputs.iterdir()),
        "exact8 regular inputs required; no heldout or unknown payload",
    )
    for name, digest in pins.items():
        exact.require(file_sha(inputs / name) == digest, "input SHA mismatch: " + name)
    validate_manifest(read(inputs / "data-manifest.json"))
    decision = read(inputs / "cotangent-decision.json")
    exact.require(
        decision["chosen_qualification_VJP_scales"] == list(SCALES)
        and decision["amplified_VJP_applied_to_optimizer"] is False
        and decision["native_GPU_admission_or_training_claim"] is False,
        "pre-GPU cotangent decision differs",
    )
    return {
        **pins,
        "consumer-source-manifest.json": file_sha(
            inputs / "consumer-source-manifest.json"
        ),
    }


def run(args):
    import sys
    from types import SimpleNamespace

    from . import higgs_depth_normalization_run as native_runtime
    from . import higgs_dst_confirmation_driver as common
    from .higgs_coordinate_lr import verify_external
    from .higgs_depth_study import SOURCE, SOURCE_MANIFEST
    from .higgs_dst_comparison import json_sha, read

    consumer = Path(__file__).resolve().parents[3]
    manifest = common.verify_consumer(
        consumer,
        args.consumer_manifest,
        args.consumer_commit,
        consumer / common.DRIVER_PATH,
    )
    exact.require(
        manifest["files"][ENTRY] == file_sha(consumer / ENTRY), "driver bytes differ"
    )
    inputs = verify_inputs(args.inputs)
    verify_external(
        SimpleNamespace(
            external_source=args.external_source,
            external_manifest=args.external_manifest,
            identity={"external_source_manifest_sha256": SOURCE_MANIFEST},
        )
    )
    sys.path[:0] = [str(args.external_source / "src"), str(args.external_source)]
    import torch
    import torchcst
    from benchmarks.training.higgs_confirmation import _configure

    exact.require(
        Path(torchcst.__file__)
        .resolve()
        .is_relative_to(args.external_source.resolve()),
        "public runtime imported outside verified975",
    )
    _configure()
    runtime = native_runtime.runtime_fingerprint()
    exact.require(
        native_runtime.validate_runtime(runtime) == "RTX4090"
        and runtime == read(args.inputs / "runtime-pin.json"),
        "exact reviewed4090 runtime required",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "stage": "A-new-native16-qualification-only",
        "consumer_commit": args.consumer_commit,
        "consumer_manifest_sha256": file_sha(args.consumer_manifest),
        "protocol_sha256": json_sha(study.protocol()),
        "external_source_commit": SOURCE,
        "external_source_manifest_sha256": SOURCE_MANIFEST,
        "runtime_fingerprint": runtime,
        "runtime_pin_sha256": inputs["runtime-pin.json"],
        "data_sha256": inputs["train-development.npz"],
        "data_manifest_sha256": inputs["data-manifest.json"],
        "verified_inputs": inputs,
    }
    write(
        args.output / "run-manifest.json",
        {
            "identity": identity,
            "protocol": study.protocol(),
            "argv": sys.argv,
            "dataset_decoded": False,
        },
    )
    result = qualify_all(
        identity, args.output / "qualification", torch.device("cuda:0")
    )
    write(args.output / "run-summary.json", result)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-commit", required=True)
    for name in (
        "consumer-manifest",
        "external-source",
        "external-manifest",
        "inputs",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
