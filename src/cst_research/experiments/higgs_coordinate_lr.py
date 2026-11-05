"""Bounded dev-only position-LR pilot through unchanged external975 eager code."""

from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import math
import types
from pathlib import Path

from .higgs_dst_comparison import file_sha, json_sha, read, write

SOURCE_COMMIT = "97580582ed28c4e494bf5c2656bbe94a85c544c3"
SOURCE_MANIFEST_SHA = "17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931"
DATA_SHA = "62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac"
RATIOS = (0.1, 1.0, 10.0)
LR = 0.0003
SAMPLE_STEPS = {1, 2, 4, 8, 16, 64, 256, 1024, 4096, 12500, 25000, 50000, 75000}


def protocol():
    return {
        "version": "higgs-position-lr-pilot-v1",
        "external_source_commit": SOURCE_COMMIT,
        "width": 1024,
        "divisor": 1000,
        "atoms": 1048,
        "position_columns": [2, 3],
        "packed_layout": [
            "polar_s",
            "polar_t",
            "physical_output_center",
            "physical_input_center",
        ],
        "base_lr": LR,
        "position_multipliers": list(RATIOS),
        "ordinary_lr": LR,
        "coupled_polar_lr": LR,
        "amplitude_weight_decay": 0.01,
        "raw_atom_weight_decay": 0.0,
        "mode": "eager",
        "dtype": "FP32",
        "TF32": False,
        "batch_size": 4,
        "train_rows": 100000,
        "dev_partition": [0, 50000],
        "epochs": 3,
        "development_seed": 17,
        "qualification": "reused actual975 qualification+runtime; new scalar/ULP/moment/clock/resume CPU gates and fresh short CUDA eager/save-reload gates",
        "advance": {
            "minimum_dev_ce_gain": 0.001,
            "maximum_auc_drop": 0.001,
            "selection": "minimum best-dev CE; lower ratio on exact tie",
            "extra_paired_seeds": [29, 43],
            "followup_epochs": 10,
        },
        "scope": "budget-local exploratory; no formal NI, independently learned width, Graph eligibility or speed-adoption claim",
        "confirmation_evaluated": False,
        "test_evaluated": False,
    }


def decision(arms, qualification_passed):
    if set(arms) != set(RATIOS):
        raise ValueError("exactly three declared pilot arms required")
    for row in arms.values():
        if not all(math.isfinite(row[k]) for k in ("loss", "roc_auc")):
            raise ValueError("nonfinite development metric")
    control = arms[1.0]
    eligible = [
        r
        for r in (0.1, 10.0)
        if qualification_passed
        and control["loss"] - arms[r]["loss"] >= 0.001
        and arms[r]["roc_auc"] >= control["roc_auc"] - 0.001
    ]
    selected = min(eligible, key=lambda r: (arms[r]["loss"], r)) if eligible else None
    return {
        "status": "advance-exploratory"
        if selected is not None
        else "stop-no-short-pilot-evidence",
        "selected_position_multiplier": selected,
        "qualification_passed": qualification_passed,
        "arms": {str(k): v for k, v in arms.items()},
        "eligible_ratios": eligible,
        "followup_seeds": [29, 43] if selected is not None else [],
        "confirmation_evaluated": False,
        "test_evaluated": False,
    }


def clone_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(v) for v in value)
    return copy.deepcopy(value)


def construct_optimizer(
    reference, model, ratio, *, force_adapter=False, cpu_reference=False
):
    """Reuse the exact external factory code with one explicit class substitution."""
    from .higgs_coordinate_optimizer import PositionLRCSTAdamW

    factory = reference
    if ratio != 1.0 or force_adapter:
        globals_copy = dict(
            reference.__globals__,
            CSTAdamW=functools.partial(
                PositionLRCSTAdamW,
                position_multiplier=ratio,
                force_recompute_for_qualification=force_adapter,
            ),
        )
        factory = types.FunctionType(
            reference.__code__,
            globals_copy,
            reference.__name__,
            reference.__defaults__,
            reference.__closure__,
        )
    if cpu_reference:
        # CPU qualification changes only device policy, never production kwargs.
        from torchcst import CSTAdamW

        cls = PositionLRCSTAdamW if ratio != 1.0 or force_adapter else CSTAdamW
        groups = [
            {
                "params": [*model[0].parameters(), *model[4].parameters()],
                "weight_decay": 0.01,
            },
            {
                "params": [model[2].atoms.p],
                "weight_decay": 0.0,
                "amplitude_weight_decay": 0.01,
            },
        ]
        opts = {"model": model, "lr": LR, "fused": False, "check_finite": True}
        if cls is PositionLRCSTAdamW:
            opts["position_multiplier"] = ratio
            opts["force_recompute_for_qualification"] = force_adapter
        optimizer = cls(groups, **opts)
        contract = {"lr": LR, "cpu_reference": True}
    else:
        optimizer, contract = factory(model, "cst", LR)
    metadata = {
        "version": 1,
        "source_commit": SOURCE_COMMIT,
        "position_multiplier": ratio,
        "position_lr": LR * ratio,
        "ordinary_lr": LR,
        "coupled_polar_lr": LR,
        "amplitude_weight_decay": 0.01,
        "position_columns": [2, 3],
    }

    def save_metadata(_optimizer, state):
        state["coordinate_lr_pilot"] = metadata
        return state

    def validate_metadata(_optimizer, state):
        if state.get("coordinate_lr_pilot") != metadata:
            raise ValueError("position LR checkpoint contract differs")

    optimizer.register_state_dict_post_hook(save_metadata)
    optimizer.register_load_state_dict_pre_hook(validate_metadata)
    return optimizer, {**contract, "coordinate_lr_pilot": metadata}


class Observer:
    def __init__(self):
        self.steps = 0
        self.samples = []
        self.optimizer = None
        self.contract = None

    def wrap(self, reference):
        def step(model, optimizer, *args):
            self.steps += 1
            sample = self.steps in SAMPLE_STEPS or self.steps % 25000 == 0
            point = model[2].atoms.p
            before = (
                clone_cpu({"point": point, "state": optimizer.state.get(point, {})})
                if sample
                else None
            )
            result = reference(model, optimizer, *args)
            if sample:
                self.samples.append(
                    {
                        "step": self.steps,
                        "before": before,
                        "after": clone_cpu(
                            {
                                "point": point,
                                "gradient": point.grad,
                                "state": optimizer.state[point],
                                "shadow_clock": getattr(
                                    optimizer, "_last_position_clock", None
                                ),
                            }
                        ),
                    }
                )
            return result

        return step


@contextlib.contextmanager
def inject(runner, ratio, observer):
    """Only benchmark globals are replaced; restore all of them even on failure."""
    saved = {
        k: getattr(runner, k)
        for k in ("make_optimizer", "ordinary_step", "_checkpoint")
    }

    def factory(model, kind, lr):
        if kind != "cst" or lr != LR:
            raise ValueError("pilot fixes CST and base LR")
        optimizer, contract = construct_optimizer(saved["make_optimizer"], model, ratio)
        observer.optimizer, observer.contract = optimizer, contract
        return optimizer, contract

    def checkpoint(
        model, optimizer, gradients, epoch, updates, development, configuration
    ):
        record = saved["_checkpoint"](
            model, optimizer, gradients, epoch, updates, development, configuration
        )
        record["configuration"] = {
            **configuration,
            "position_lr_multiplier": ratio,
            "adapter_protocol": protocol()["version"],
        }
        record["optimizer_contract"] = observer.contract
        return record

    runner.make_optimizer = factory
    runner.ordinary_step = observer.wrap(saved["ordinary_step"])
    runner._checkpoint = checkpoint
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(runner, name, value)


def diagnostics(samples, kernel):
    import torch
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )

    kernel = copy.deepcopy(kernel).cpu()
    rows = []
    for sample in samples:
        p = sample["after"]["point"]
        prior = sample["before"]["point"]
        amplitude, radius_input, radius_output = decoded_values(kernel, p)
        delta = (
            torch.remainder(p[:, 2:4].double() - prior[:, 2:4].double() + 512.0, 1024.0)
            - 512.0
        )

        def stats(values):
            return {
                "min": float(values.min()),
                "mean": float(values.double().mean()),
                "max": float(values.max()),
            }

        rows.append(
            {
                "step": sample["step"],
                "physical_output_center": stats(p[:, 2]),
                "physical_input_center": stats(p[:, 3]),
                "polar_q": stats(p[:, :2].double().square().sum(1)),
                "amplitude": stats(amplitude),
                "physical_input_radius": stats(radius_input),
                "physical_output_radius": stats(radius_output),
                "output_center_step": stats(delta[:, 0]),
                "input_center_step": stats(delta[:, 1]),
                "position_zero_fraction_by_axis": [
                    (delta[:, i] == 0).double().mean().item() for i in (0, 1)
                ],
                "mean_abs_position_step_by_axis": [
                    delta[:, i].abs().mean().item() for i in (0, 1)
                ],
                "master_clock": float(sample["after"]["state"]["step"]),
                "shadow_clock": None
                if sample["after"]["shadow_clock"] is None
                else float(sample["after"]["shadow_clock"]),
            }
        )
    return rows


def qualify(runner, data, output):
    """Small fresh CUDA eager/optimizer/save-reload gate; no development scoring."""
    import numpy as np
    import torch
    from benchmarks.training.higgs_verify import (
        byte_equal,
        finite_fields,
        tensor_fields,
    )

    from .higgs_coordinate_oracle import adam_scalar, retract_scalar

    output.mkdir()
    runner._configure()
    with np.load(data, allow_pickle=False) as archive:
        tx = torch.from_numpy(archive["x_train"][:32].copy()).cuda()
        ty = torch.from_numpy(archive["y_train"][:32].copy()).cuda()

    def equal(a, b):
        aa, bb = dict(tensor_fields(a)), dict(tensor_fields(b))
        return aa.keys() == bb.keys() and all(byte_equal(aa[k], bb[k]) for k in aa)

    record = {
        "status": "running",
        "source_commit": SOURCE_COMMIT,
        "checks": [],
        "confirmation_evaluated": False,
        "test_evaluated": False,
    }
    write(output / "qualification.json", record)
    # Direct LR/ULP, independent scalar Adam+Polar law, contiguous fused state.
    scalar_states = {}
    scalar_optimizer_states = {}
    for ratio in RATIOS:
        model = runner.make_model(1024, 1000, 17, "cst").cuda()
        optimizer, _ = construct_optimizer(runner.make_optimizer, model, ratio)
        point = model[2].atoms.p
        with torch.no_grad():
            point[:, 0] = 0.2
            point[:, 1] = 1.2
            point[:, 2] = 513.00006103515625
            point[:, 3] = 1023.99999
            point[1, 2] = 0.000001
            point[1, 3] = 1023.9999
        before = clone_cpu(point)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        point.grad[1, 3] = -1.0
        gradient_before = clone_cpu(point.grad)
        optimizer.step()
        torch.cuda.synchronize()
        actual = clone_cpu(point)
        expected_rows = []
        for row in (0, 1):
            proposal = [
                adam_scalar(
                    float(before[row, col]),
                    float(gradient_before[row, col]),
                    0.0,
                    0.0,
                    1,
                    LR * (ratio if col >= 2 else 1.0),
                )[0]
                for col in range(4)
            ]
            proposal = [float(torch.tensor(v, dtype=torch.float32)) for v in proposal]
            expected = retract_scalar(
                before[row].tolist(),
                proposal,
                radial_regularization=float(
                    model[2].kernel.scalar("radial_regularization")
                ),
            )
            expected_rows.append(expected)
        expected_rows = torch.tensor(expected_rows, dtype=torch.float32)
        if not torch.equal(actual[:2, 2:], expected_rows[:, 2:]):
            raise ValueError("CUDA direct position-LR/ULP/torus crossing oracle failed")
        if not torch.allclose(
            actual[:2, :2], expected_rows[:, :2], rtol=3e-5, atol=3e-6
        ):
            raise ValueError("CUDA scalar Polar oracle failed")
        expected = expected_rows[0]
        if any(float(s["step"]) != 1 for s in optimizer.state.values()):
            raise ValueError("CUDA Adam master clock failed")
        if ratio != 1.0 and float(optimizer._last_position_clock) != 1:
            raise ValueError("CUDA shadow clock failed")
        for parameter, state in optimizer.state.items():
            # CUDA fused AdamW casts beta to the parameter dtype before
            # forming (1-beta); Python literals .1/.001 are different FP32
            # coefficients. Keep the existing tolerances unchanged.
            beta1, beta2 = optimizer.defaults["betas"]
            gradient = parameter.grad
            expected_mean = gradient * (1 - gradient.new_tensor(beta1))
            expected_square = gradient.square() * (1 - gradient.new_tensor(beta2))
            if not torch.allclose(
                state["exp_avg"], expected_mean, atol=1e-7, rtol=1e-6
            ) or not torch.allclose(
                state["exp_avg_sq"], expected_square, atol=1e-9, rtol=1e-6
            ):
                raise ValueError("CUDA fused contiguous moment/column integrity failed")
        scalar_states[ratio] = clone_cpu(model.state_dict())
        scalar_optimizer_states[ratio] = clone_cpu(optimizer.state_dict()["state"])
        beta1, beta2 = optimizer.defaults["betas"]
        record["cuda_fused_beta_rounding"] = {
            "scope": "qualification expectation only; no optimizer, tolerance, training or selection change",
            "dtype": str(point.dtype),
            "beta1_stored": float(point.new_tensor(beta1)),
            "beta2_stored": float(point.new_tensor(beta2)),
            "first_mean_coefficient": float(1 - point.new_tensor(beta1)),
            "first_square_coefficient": float(1 - point.new_tensor(beta2)),
        }
        torch.save(
            {
                "model": scalar_states[ratio],
                "before": before,
                "after": actual,
                "expected_row0": expected,
                "expected_rows": expected_rows,
                "gradient": gradient_before,
                "optimizer": clone_cpu(optimizer.state_dict()),
            },
            output / f"ratio-{ratio:g}-scalar.pt",
        )
        record["checks"].append(
            {
                "ratio": ratio,
                "direct_position_scalar_and_ulp": True,
                "polar_scalar": True,
                "clock": True,
            }
        )
    for ratio in (0.1, 10.0):
        stock_fields = dict(tensor_fields(scalar_optimizer_states[1.0]))
        arm_fields = dict(tensor_fields(scalar_optimizer_states[ratio]))
        if stock_fields.keys() != arm_fields.keys() or any(
            not byte_equal(stock_fields[k], arm_fields[k]) for k in stock_fields
        ):
            raise ValueError(
                "CUDA same-gradient stock/arm moment or master-clock bytes differ"
            )
        if not torch.equal(
            scalar_states[ratio]["2.atom_state.atoms.p"][:, :2],
            scalar_states[1.0]["2.atom_state.atoms.p"][:, :2],
        ):
            raise ValueError("CUDA position recomputation changed Polar fields")
        for name in ("0.weight", "0.bias", "4.weight", "4.bias"):
            if not torch.equal(scalar_states[ratio][name], scalar_states[1.0][name]):
                raise ValueError(
                    "CUDA position recomputation changed ordinary parameters"
                )
    control = None
    for ratio, force in [(1.0, False), (1.0, True), (0.1, False), (10.0, False)]:
        label = f"ratio-{ratio:g}" + ("-adapter" if force else "")
        model = runner.make_model(1024, 1000, 17, "cst").cuda()
        optimizer, _ = construct_optimizer(
            runner.make_optimizer, model, ratio, force_adapter=force
        )
        totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
        initial = clone_cpu(model.state_dict())
        midway = None
        for step in range(8):
            runner.ordinary_step(
                model,
                optimizer,
                tx,
                ty,
                torch.arange(4 * step, 4 * step + 4, device="cuda"),
                totals,
            )
            if step == 3:
                midway = clone_cpu(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
                )
        final = clone_cpu(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "gradients": {n: p.grad for n, p in model.named_parameters()},
            }
        )
        resumed = runner.make_model(1024, 1000, 17, "cst").cuda()
        restored, _ = construct_optimizer(
            runner.make_optimizer, resumed, ratio, force_adapter=force
        )
        resumed.load_state_dict(midway["model"])
        restored.load_state_dict(midway["optimizer"])
        for step in range(4, 8):
            runner.ordinary_step(
                resumed,
                restored,
                tx,
                ty,
                torch.arange(4 * step, 4 * step + 4, device="cuda"),
                totals,
            )
        end = clone_cpu(
            {
                "model": resumed.state_dict(),
                "optimizer": restored.state_dict(),
                "gradients": {n: p.grad for n, p in resumed.named_parameters()},
            }
        )
        if finite_fields(dict(tensor_fields(final))) or finite_fields(
            dict(tensor_fields(end))
        ):
            raise FloatingPointError("nonfinite CUDA qualification state/gradient")
        if not equal(final, end):
            raise ValueError("CUDA eager save/reload trajectory differs")
        if ratio == 1.0 and not force:
            control = final
        if force and not equal(control, final):
            raise ValueError("CUDA all1 adapter differs from native stock")
        torch.save(
            {"initial": initial, "midway": midway, "final": final, "resumed": end},
            output / (label + "-trajectory.pt"),
        )
        record["checks"].append(
            {
                "ratio": ratio,
                "adapter1": force,
                "eager_steps": 8,
                "resume_steps": 4,
                "tensor_bytes_equal": True,
            }
        )
    record.update(
        status="passed",
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
    )
    write(output / "qualification.json", record)
    return record


def run_arm(runner, args, ratio, seed, output, epochs):
    import torch

    observer = Observer()
    train_args = argparse.Namespace(
        data=args.data,
        output=output,
        kind="cst",
        width=1024,
        divisor=1000,
        seed=seed,
        lr=LR,
        epochs=epochs,
        mode="eager",
        steps_per_epoch=0,
        development_rows=50000,
        retain_epoch_checkpoints=True,
        evaluate_confirmation=False,
        evaluate_only_checkpoint=None,
        evaluation_output=None,
        position_lr_multiplier=ratio,
    )
    with inject(runner, ratio, observer):
        result = runner.run(train_args)
    if (
        result["total_updates"] != epochs * 25000
        or result["graph_replays"] != 0
        or len(result["epochs"]) != epochs
    ):
        raise ValueError("pilot budget/mode differs")
    if result["confirmation_evaluated"] or result["test_evaluated"]:
        raise ValueError("pilot opened heldout data")
    if any(row["updates"] != 25000 * row["epoch"] for row in result["epochs"]):
        raise ValueError("pilot epoch budget differs")
    torch.save(observer.samples, output / "sampled-updates.pt")
    write(
        output / "physical-diagnostics.json",
        diagnostics(observer.samples, observer.optimizer.model[2].kernel),
    )
    # The external runner retained epoch3 rather than final.pt; audit an explicit
    # identical-byte file so the unchanged generic checkpoint reader applies.
    import shutil

    from .higgs_dst_evidence import audit_raw_cell

    shutil.copy2(output / f"epoch-{epochs}.pt", output / "final.pt")
    audit = audit_raw_cell(output, args.external_source, dst=False, data=args.data)
    write(output / "raw-audit.json", audit)
    write(
        output / "adapter.json",
        {
            "protocol": protocol(),
            "position_multiplier": ratio,
            "identity": args.identity,
            "factory": "same external function code with explicit CSTAdamW class substitution for non1",
            "restored_benchmark_globals": True,
            "graph_qualification_modified": False,
            "timing_scope": "eager scientific reference including sparse diagnostic sampling; no speed claim",
        },
    )
    return result


def verify_external(args):
    import sys

    manifest = read(args.external_manifest)
    if (
        manifest.get("source_commit") != SOURCE_COMMIT
        or manifest.get("archive_sha256")
        != "edbb585de56e304887b118069b7c7ecd9e45ae5fe2e08a5afca828ad8544eb5f"
        or file_sha(args.external_manifest)
        != args.identity["external_source_manifest_sha256"]
        or file_sha(args.external_manifest) != SOURCE_MANIFEST_SHA
    ):
        raise ValueError("external975 source manifest differs")
    from .higgs_dst_comparison_driver import inventory, safe_path

    actual = {
        n: p
        for n, p in inventory(args.external_source).items()
        if not n.startswith((".job-deps/", "research-inputs/"))
    }
    if set(actual) != set(manifest["files"]):
        raise ValueError("external975 source inventory differs")
    for name, digest in manifest["files"].items():
        if file_sha(safe_path(args.external_source, name)) != digest:
            raise ValueError("external975 source bytes changed")
    if "torchcst" in sys.modules and not Path(
        sys.modules["torchcst"].__file__
    ).resolve().is_relative_to(args.external_source.resolve()):
        raise ValueError("a different torchcst runtime is already imported")


def run(args):
    args.identity = read(args.identity_file)
    verify_external(args)
    from benchmarks.training import higgs_confirmation as runner

    if args.stage == "pilot" and (
        args.seed is not None
        or args.selected_ratio is not None
        or args.pilot_decision is not None
    ):
        raise ValueError(
            "pilot is fixed seed17 and three ratios; conditional flags are followup-only"
        )
    if file_sha(args.data) != DATA_SHA:
        raise ValueError("pilot dataset differs")
    if args.identity["protocol_sha256"] != json_sha(protocol()):
        raise ValueError("pilot protocol binding differs")
    args.output.mkdir(parents=True, exist_ok=True)
    ratios = RATIOS if args.stage == "pilot" else (1.0, args.selected_ratio)
    seed = 17 if args.stage == "pilot" else args.seed
    if args.stage == "followup":
        if seed not in (29, 43) or args.selected_ratio not in (0.1, 10.0):
            raise ValueError("bounded paired followup only")
        if (
            not args.pilot_decision
            or file_sha(args.pilot_decision) != args.pilot_decision_sha256
        ):
            raise ValueError("root-reviewed pilot decision digest required")
        prior = read(args.pilot_decision)
        if (
            prior["identity"] != args.identity
            or prior["decision"]["status"] != "advance-exploratory"
            or prior["decision"]["selected_position_multiplier"] != args.selected_ratio
        ):
            raise ValueError("followup not admitted by matching pilot decision")
    qualification = qualify(runner, args.data, args.output / "qualification")
    results = {}
    for ratio in ratios:
        results[ratio] = run_arm(
            runner,
            args,
            ratio,
            seed,
            args.output / f"ratio-{ratio:g}",
            3 if args.stage == "pilot" else 10,
        )
    orders = [
        [r["order_sha256"] for r in result["epochs"]] for result in results.values()
    ]
    if any(value != orders[0] for value in orders):
        raise ValueError("pilot data order differs between arms")
    import torch
    from benchmarks.training.higgs_verify import tensor_fields, tensor_hash

    initial_hashes = []
    for ratio in results:
        retained = torch.load(
            args.output / f"ratio-{ratio:g}/initialization.pt",
            map_location="cpu",
            weights_only=True,
        )
        initial_hashes.append(
            {k: tensor_hash(v) for k, v in tensor_fields(retained["model"])}
        )
    if any(value != initial_hashes[0] for value in initial_hashes):
        raise ValueError("pilot initial model tensors differ")
    inits = [result["initial_development"] for result in results.values()]
    if any(value != inits[0] for value in inits):
        raise ValueError("pilot initialization output differs between arms")
    if args.stage == "pilot" and args.control_reference:
        if file_sha(args.control_reference) != args.identity.get(
            "original_control_reference_sha256"
        ):
            raise ValueError("original control diagnostic input changed")
        reference = read(args.control_reference)
        if any(
            reference[k] != v
            for k, v in {
                "source_commit": SOURCE_COMMIT,
                "width": 1024,
                "divisor": 1000,
                "seed": 17,
                "lr": LR,
                "dataset_sha256": DATA_SHA,
            }.items()
        ):
            raise ValueError("original control diagnostic source/config differs")
        comparisons = []
        for old, new in zip(reference["epochs"], results[1.0]["epochs"]):
            comparisons.append(
                {
                    "epoch": new["epoch"],
                    "order_hash_exact": old["order_sha256"] == new["order_sha256"],
                    "state_hashes_exact": old["state"] == new["state"],
                    "gradient_hashes_exact": old["gradients"] == new["gradients"],
                    "development_metric_deltas": {
                        k: new["development"][k] - old["development"][k]
                        for k in ("loss", "accuracy", "roc_auc")
                    },
                }
            )
        write(
            args.output / "original-control-diagnostic.json",
            {
                "reference_learning_sha256": reference["learning_sha256"],
                "reference_mode": "graph",
                "pilot_mode": "eager",
                "comparisons": comparisons,
                "scope": "diagnostic only; no output adjustment or substituted control",
            },
        )
    selected = {
        r: result["best_development_checkpoint"]["development"]
        for r, result in results.items()
    }
    value = {
        "identity": args.identity,
        "protocol": protocol(),
        "stage": args.stage,
        "seed": seed,
        "qualification_sha256": file_sha(
            args.output / "qualification/qualification.json"
        ),
        "arms": {
            str(r): {
                "directory": f"ratio-{r:g}",
                "learning_sha256": file_sha(args.output / f"ratio-{r:g}/learning.json"),
                "checkpoint_sha256": result["best_development_checkpoint"][
                    "checkpoint_sha256"
                ],
            }
            for r, result in results.items()
        },
        "decision": decision(selected, qualification["status"] == "passed")
        if args.stage == "pilot"
        else {"status": "exploratory-paired-followup-completed"},
        "confirmation_evaluated": False,
        "test_evaluated": False,
    }
    write(args.output / "pilot-decision.json", value)
    write(args.output / "pilot.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "followup"), required=True)
    for name in (
        "data",
        "output",
        "external-source",
        "external-manifest",
        "identity-file",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(29, 43))
    parser.add_argument("--selected-ratio", type=float, choices=(0.1, 10.0))
    parser.add_argument("--pilot-decision", type=Path)
    parser.add_argument("--pilot-decision-sha256")
    parser.add_argument("--control-reference", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
