"""Fresh arm qualification using the unchanged reference/gates/native load path."""

from __future__ import annotations

import copy
import gc

from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_linear_atom_qualification as fixed_gates
from . import higgs_multilayer_conditioning_observation as observation
from . import higgs_multilayer_conditioning_reference as reference
from . import higgs_multilayer_conditioning_state as state
from . import higgs_width_amplitude_init as study
from .higgs_dst_comparison import file_sha, write

SCALES = (1.0, 1e24)


def qualify_arm(
    arm,
    gains,
    identity,
    output,
    quad_output,
    device,
    *,
    model_factory=None,
    runtime_case=None,
):
    factory = study.make_model if model_factory is None else model_factory
    case = study.ARMS[arm] if runtime_case is None else runtime_case
    import torch
    from benchmarks.training.full_step_graph import ordinary_step
    from benchmarks.training.higgs_verify import (
        byte_equal,
        finite_fields,
        tensor_fields,
    )

    output.mkdir()
    model = factory(arm, gains=gains).to(device)
    optimizer, _ = state.make_optimizer(model)
    initial = state.snapshot(model, optimizer, 0, identity)
    torch.save(initial, output / "initial-state.pt")
    quad = fixed_gates.quadrature_check(
        model,
        {"method": ("dense" if case == "dense-none" else "cst"), "width": 1024, "c": 4},
        quad_output,
        device,
    )
    ref = copy.deepcopy(model).cpu().double()
    x0 = torch.randn(4, 28, generator=torch.Generator().manual_seed(817))
    names = ["Y", "dX", *[name for name, _ in model.named_parameters()]]
    reports = []
    forwards = []
    expected_forwards = []
    for scale in SCALES:
        x = x0.to(device).requires_grad_()
        y = model(x)
        dy = (
            torch.tensor(
                [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]],
                dtype=torch.float32,
                device=device,
            )
            * scale
        )
        actual = (
            y.detach().cpu(),
            *[
                v.detach().cpu()
                for v in torch.autograd.grad(y, (x, *model.parameters()), dy)
            ],
        )
        yy, dx, dp = reference.full_reference(
            ref, x0.double(), dy.detach().cpu().double(), chunk_atoms=256
        )
        expected = (yy, dx, *dp.values())
        exact.require(list(dp) == names[2:], "full H8 parameter inventory differs")
        filename = f"vjp-scale-{scale:.0e}.pt"
        torch.save(
            {
                "actual": actual,
                "expected": expected,
                "fields": names,
                "X": x0,
                "dY": dy.detach().cpu(),
                "scale": scale,
                "reference": "newmanualFP64Dense/ReLU/LN/gainreverse +unchanged975chunkedallatoms256",
                "optimizer_step": False,
            },
            output / filename,
        )
        record = fixed_gates.gates(actual, expected, names, scale)
        record.update({"raw_file": filename, "raw_sha256": file_sha(output / filename)})
        reports.append(record)
        write(output / "vjp-report.json", reports)
        exact.require(
            record["pass"],
            "new amplitude-init full-network VJP failed; no fallback",
        )
        forwards.append(actual[0])
        expected_forwards.append(expected[0])
        del actual, expected, yy, dx, dp, dy, x, y
    exact.require(
        byte_equal(*forwards) and byte_equal(*expected_forwards),
        "cotangent changedforward",
    )
    exact.assert_equal(
        state.core_state(state.snapshot(model, optimizer, 0, identity)),
        state.core_state(initial),
        "VJP-mutatedinitial0",
    )
    initial_fingerprint = state.core_digest(initial)
    del model, optimizer, initial, ref, forwards, expected_forwards
    gc.collect()
    torch.cuda.empty_cache()
    inputs = torch.randn(32, 28, generator=torch.Generator().manual_seed(1817)).to(
        device
    )
    labels = torch.arange(32, device=device) % 2
    orders = torch.arange(32).reshape(8, 4)
    torch.save(
        {
            "inputs": inputs.cpu(),
            "labels": labels.cpu(),
            "orders": orders,
            "seed": 1817,
            "synthetic_not_taskdata": True,
        },
        output / "normal-inputs.pt",
    )

    def fresh():
        m = factory(arm, gains=gains).to(device)
        o, _ = state.make_optimizer(m)
        exact.assert_equal(
            state.core_digest(state.snapshot(m, o, 0, identity)),
            initial_fingerprint,
            "freshnormal0",
        )
        return m, o

    model, optimizer = fresh()
    totals = (inputs.new_zeros(()), torch.zeros((), dtype=torch.int64, device=device))
    for step, indices in enumerate(orders, 1):
        ordinary_step(model, optimizer, inputs, labels, indices.to(device), totals)
        state.check_optimizer(model, optimizer, step)
        if step == 4:
            middle = state.snapshot(model, optimizer, 4, identity)
    final = state.snapshot(model, optimizer, 8, identity)
    torch.save(middle, output / "normal-step4.pt")
    torch.save(final, output / "normal-step8.pt")
    resumed, loaded, _, load_proof = state.load(middle, device)
    for step in range(4, 8):
        ordinary_step(resumed, loaded, inputs, labels, orders[step].to(device), totals)
        state.check_optimizer(resumed, loaded, step + 1)
    after = state.snapshot(resumed, loaded, 8, identity)
    exact.assert_equal(
        state.core_state(after),
        state.qualified_state.with_generations(final, state.owners(resumed)),
        "native8-vs4load4-allotherbytes",
    )
    torch.save(after, output / "resumed-step8.pt")
    watched, watched_optimizer = fresh()
    first_summary, first_raw = observation.observed_step(
        watched, watched_optimizer, inputs[:4], labels[:4], orders[0], identity, step=1
    )
    torch.save(first_raw, output / "observed-first-step.pt")
    write(output / "observed-first-step.json", first_summary)
    for step in range(1, 8):
        ordinary_step(
            watched, watched_optimizer, inputs, labels, orders[step].to(device), totals
        )
    watched8 = state.snapshot(watched, watched_optimizer, 8, identity)
    exact.assert_equal(
        state.core_state(watched8),
        state.core_state(final),
        "observedfirst+7bare-vs8unchangedordinary-step",
    )
    torch.save(watched8, output / "observed-native8.pt")
    for raw in (middle, final, after, watched8):
        exact.require(
            not finite_fields(dict(tensor_fields(raw))),
            "nonfinite qualificationmodel/grad/moments",
        )
    record = {
        "status": "passed-new-amplitude-arm-native-qualification",
        "arm": arm,
        "case": case,
        "gains": gains,
        "identity": identity,
        "VJP_scales": list(SCALES),
        "normal_elementwise_relative_packed_and_material_passed": True,
        "amplified_componentwise_failures_preserved": reports[1]["elementwise"],
        "amplified_gradient_used_for_optimizer": False,
        "synthetic_normal_updates": 8,
        "additional_resume_updates": 4,
        "additional_observer_parity_updates": 8,
        "total_qualification_updates": 20,
        "task_training_updates": 0,
        "native_load": load_proof,
        "quad31": quad,
        "observer_first+7bare_equals_ordinary8_allcorebytes": True,
        "observer_optimizer_monkeypatch_or_module_backwardhooks": False,
        "owned_parameter_tensors": 11,
        "files": {p.name: file_sha(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write(output / "qualification.json", record)
    del (
        model,
        optimizer,
        resumed,
        loaded,
        watched,
        watched_optimizer,
        middle,
        final,
        after,
        watched8,
        first_raw,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return record
