"""Compare a shared Atom decoder with a frozen rational-envelope QKV run."""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__:
    from .shared_atom_decoder import SharedAtomDecoder
else:
    from shared_atom_decoder import SharedAtomDecoder


class DecodedLinear(nn.Module):
    def __init__(self, layer, decoder, amplitude_scale):
        super().__init__()
        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.radius_envelope import (
            physical_operator,
        )
        from torchcst._backends.torch.parameterizations.radius_envelope import (
            prepare_constants,
        )

        self.layer = layer
        # The enclosing model registers this same decoder once.
        object.__setattr__(self, "decoder", decoder)
        self.amplitude_scale = amplitude_scale
        self.operator = physical_operator(layer.execution_declaration())
        spec = layer.execution_declaration().kernel.parameterization
        self.constants = prepare_constants(
            layer.atoms.p,
            radius_min=spec.radius_min,
            radius_max=spec.radius_max,
            tau=spec.tau,
            amplitude_max=spec.amplitude_max,
        )

    def decoded(self):
        return self.decoder.decode(
            self.layer.atoms.p,
            amplitude_scale=self.amplitude_scale,
            amplitude_max=32.0,
            periods=(self.operator.out_features, self.operator.in_features),
        )

    def forward(self, x):
        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
            LM_PHYSICAL_WIDTH_LEVELS,
            _physical_dispatcher,
        )
        from torchcst._backends.torch.parameterizations.radius_envelope import (
            prepared_physical_parameters,
        )
        from torchcst.operators.execution import LinearBinding, LinearInputs

        physical = prepared_physical_parameters(self.decoded(), self.constants)
        flat = x.reshape(-1, self.operator.in_features).contiguous()
        y = _physical_dispatcher().run(
            LinearBinding(self.operator, physical),
            LinearInputs(flat),
            plan=LM_PHYSICAL_WIDTH_LEVELS,
        )
        return y.reshape(*x.shape[:-1], self.operator.out_features)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_reference(args):
    record = read(args.reference / "rational_learned" / "result.json")
    arm = record["arms"]["rational_learned"]
    assert record["state"] == arm["state"] == "completed"
    assert arm["checkpoint_audit"] == "passed"
    assert sha(args.reference / "train.py") == record["script_sha256"]
    assert sha(args.protocol) == record["source_protocol_sha256"]
    protocol = read(args.protocol)
    overrides = {}
    for manifest in args.source_changes:
        for name, change in read(manifest)["changes"].items():
            overrides[name] = change.get("after", change.get("experiment"))
    patch = record["source_change"]
    assert (
        overrides.get(patch["path"], protocol["source_files"][patch["path"]])
        == patch["before"]
    )
    overrides[patch["path"]] = patch["after"]
    for name, digest in protocol["source_files"].items():
        assert sha(args.reference / "source" / name) == overrides.get(name, digest), (
            name
        )
    spec = importlib.util.spec_from_file_location(
        "qkv_reference", args.reference / "train.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver, record, protocol


def attach(model, projections, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model.decoder = SharedAtomDecoder().to(next(model.parameters()).device)
    scale = float(
        torch.cat([p.map.atoms.p[:, 0].detach() for p in projections.values()])
        .square()
        .mean()
        .sqrt()
    )
    assert scale > 0
    for p in projections.values():
        layer = p.map
        p.map = DecodedLinear(layer, model.decoder, scale)
        # Coordinates are periodic and can wrap by one chart period.
        raw = layer.atoms.p.detach()
        decoded = p.map.decoded().detach()
        torch.testing.assert_close(decoded[:, :2], raw[:, :2], atol=0, rtol=0)
        periods = raw.new_tensor([layer.out_features, layer.in_features])
        torch.testing.assert_close(
            decoded[:, 2:], raw[:, 2:].remainder(periods), atol=0, rtol=0
        )
    return scale


def make_optimizer(model, projections):
    latent = [p.map.layer.atoms.p for p in projections.values()]
    shared = list(model.decoder.parameters())
    special = {id(p) for p in (*latent, *shared)}
    ordinary = [p for p in model.parameters() if id(p) not in special]
    opt = torch.optim.AdamW(
        [
            {"params": ordinary, "lr_scale": 1.0},
            {"params": latent, "lr_scale": 32.0},
            {"params": shared, "lr_scale": 1.0},
        ],
        lr=0.0003,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    return opt, ordinary + latent, shared


@torch.no_grad()
def decoded_geometry(projections):
    from torchcst._backends.torch.parameterizations.radius_envelope import (
        prepared_physical_parameters,
    )

    rows = torch.cat(
        [
            prepared_physical_parameters(p.map.decoded(), p.map.constants)[:, :2]
            for p in projections.values()
        ]
    )
    amplitude, radius = rows.unbind(1)
    bands = (
        (0.0, 0.01),
        (0.01, 0.03),
        (0.03, 0.1),
        (0.1, 0.3),
        (0.3, 0.6),
        (0.6, None),
    )
    result = []
    for lo, hi in bands:
        mask = (amplitude.abs() >= lo) & (
            amplitude.abs() < (float("inf") if hi is None else hi)
        )
        selected = radius[mask]
        result.append(
            {
                "band": [lo, hi],
                "count": len(selected),
                "median_radius": float(selected.median()) if len(selected) else None,
            }
        )
    return result


def batches_for(train, data, order, clock):
    starts = order[(clock - 1) * 8 : clock * 8] * 128
    return [
        train.token_batch(data["train"], starts[j : j + 2], 128, "cuda")
        for j in range(0, 8, 2)
    ]


def train_run(args, train, record, protocol):
    from .ffn_contract import learning_rate
    from torchcst._backends.torch.parameterizations import radius_envelope

    assert torch.cuda.get_device_name() == record["gpu"]
    assert torch.version.cuda == record["cuda"] and torch.__version__ == record["torch"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_path = Path(protocol["data"])
    assert sha(data_path / "manifest.json") == record["data_manifest_sha256"]
    _, data = train.verify_data(data_path)
    order = np.random.default_rng(905).permutation(20000000 // 128)
    assert hashlib.sha256(order.tobytes()).hexdigest() == record["data_order_sha256"]
    os.environ["CST_RESEARCH_RADIUS_CAP"] = "512"
    model = train.model_for()
    assert train.parameter_digest(model) == record["initial_parameters_sha256"]
    radius_envelope.USE_RATIONAL_LOWER = True
    model.cuda()
    projections = train.projections(model)
    x, y = batches_for(train, data, order, 1)[0]
    raw = [p.map.atoms.p for p in projections.values()]
    _, direct_loss = model(x, y)
    direct_grad = torch.autograd.grad(direct_loss, raw)
    scale = attach(model, projections, record["protocol"]["seed"])
    _, decoded_loss = model(x, y)
    torch.testing.assert_close(decoded_loss, direct_loss, atol=1e-6, rtol=0)
    opt, core, shared = make_optimizer(model, projections)
    grads = torch.autograd.grad(decoded_loss, raw + shared)
    for actual, expected in zip(grads[: len(raw)], direct_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-3)
    assert any(bool(g.any()) for g in grads[len(raw) :])
    del direct_loss, decoded_loss, direct_grad, grads, x, y

    initial = train.development(model, data["development"])
    assert abs(initial["ce"] - record["initial_development"]["ce"]) <= 1e-6
    shared_before = [p.detach().clone() for p in shared]
    arm = record["arms"]["rational_learned"]
    report = {
        "state": "running",
        "completed_steps": 0,
        "target_steps": args.steps,
        "reference_result_sha256": sha(
            args.reference / "rational_learned" / "result.json"
        ),
        "runner_sha256": sha(Path(__file__)),
        "decoder_sha256": sha(Path(__file__).with_name("shared_atom_decoder.py")),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "data_manifest_sha256": record["data_manifest_sha256"],
        "data_order_sha256": record["data_order_sha256"],
        "architecture": [4, 16, 4],
        "shared_scalars": sum(p.numel() for p in shared),
        "initial_parameters_sha256": record["initial_parameters_sha256"],
        "initial_amplitude_rms": scale,
        "initial_development": initial,
        "initial_loss_and_latent_grad_match": True,
        "decoder_gradient_clip": 0.1,
        "evaluations": {},
        "checkpoints": {},
    }
    path = args.output / "result.json"
    times = []
    peak_allocated = 0
    peak_reserved = 0
    started = time.monotonic()
    train.write(path, report)
    try:
        with (args.output / "curve.jsonl").open("x") as curve:
            for clock in range(1, args.steps + 1):
                model.train()
                opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:
                    group["lr"] = (
                        learning_rate(clock, train.SCHEDULE_STEPS) * group["lr_scale"]
                    )
                batches = batches_for(train, data, order, clock)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                loss_total = 0.0
                for batch in batches:
                    _, loss = model(*batch)
                    (loss / len(batches)).backward()
                    loss_total += float(loss.detach()) / len(batches)
                core_norm = torch.nn.utils.clip_grad_norm_(
                    core, 1.0, error_if_nonfinite=True
                )
                shared_norm = torch.nn.utils.clip_grad_norm_(
                    shared, 0.1, error_if_nonfinite=True
                )
                opt.step()
                with torch.no_grad():
                    for param in raw:
                        param[:, 0].clamp_(-32.0, 32.0)
                        param[:, 1].clamp_(0.0, 1.0)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                if clock > 10 or args.steps <= 10:
                    times.append(elapsed)
                    peak_allocated = max(
                        peak_allocated, torch.cuda.max_memory_allocated()
                    )
                    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                if clock == 1:
                    assert any(
                        not torch.equal(a, b)
                        for a, b in zip(shared, shared_before, strict=True)
                    )
                row = {
                    "step": clock,
                    "train_ce": loss_total,
                    "step_seconds": elapsed,
                    "core_grad_norm": float(core_norm),
                    "shared_grad_norm": float(shared_norm),
                }
                if clock in train.EVAL_STEPS or clock == args.steps:
                    row["development"] = train.development(model, data["development"])
                    row["decoded_radius_by_amplitude"] = decoded_geometry(projections)
                    ref = arm["evaluations"].get(str(clock))
                    if ref is not None:
                        row["direct_ce"] = ref["development"]["ce"]
                        row["delta_ce"] = row["development"]["ce"] - row["direct_ce"]
                    report["evaluations"][clock] = row
                    print(json.dumps(row), flush=True)
                if clock in train.SAVE_STEPS or clock == args.steps:
                    ckpt = args.output / f"decoder-{clock}.pt"
                    temp = ckpt.with_suffix(".tmp")
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "step": clock,
                            "amplitude_scale": scale,
                            "data_manifest_sha256": record["data_manifest_sha256"],
                            "data_order_sha256": record["data_order_sha256"],
                        },
                        temp,
                    )
                    temp.replace(ckpt)
                    report["checkpoints"][clock] = sha(ckpt)
                curve.write(json.dumps(row, allow_nan=False) + "\n")
                report["completed_steps"] = clock
                if clock % 50 == 0 or "development" in row:
                    curve.flush()
                    train.write(path, report)

        assert sha(ckpt) == report["checkpoints"][args.steps]
        saved = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        assert all(int(state["step"]) == args.steps for state in opt.state.values())
        restored = train.development(model, data["development"])
        assert abs(restored["ce"] - row["development"]["ce"]) <= 1e-6
        report.update(
            state="completed",
            final_development=restored,
            checkpoint_audit="passed",
            curve_sha256=sha(args.output / "curve.jsonl"),
            median_step_seconds=statistics.median(times),
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except BaseException as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        train.write(path, report)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-changes", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4882)
    args = parser.parse_args()
    train, record, protocol = load_reference(args)
    for module, name in (
        ("torchcst", "torchcst"),
        ("cst_research.experiments.ffn_model", "cst-research"),
    ):
        path = Path(importlib.util.find_spec(module).origin).resolve()
        assert path.is_relative_to((args.reference / "source" / name).resolve()), path
    if not 1 <= args.steps <= train.LONG_STEPS:
        parser.error(f"steps must be in [1, {train.LONG_STEPS}]")
    args.output.mkdir(parents=False)
    train_run(args, train, record, protocol)


if __name__ == "__main__":
    main()
