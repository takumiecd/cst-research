"""Qualify the exact public-envelope FFN model and its checkpoint updates."""

import argparse
import io
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torchcst import CSTLinear
from torchcst._backends.torch.parameterizations.radius_envelope import (
    from_state,
    prepare_constants,
    prepared_physical_parameters,
)

from .ffn_contract import ModelConfig
from .ffn_model import LanguageModel, configure_native, cst_optimizer, parameter_digest
from .ffn_radius_fraction import snapshot
from .ffn_run import sha, token_batch, verify_data

TAUS = (0.003, 0.01, 0.03, 0.1, 0.3)


def config_for(
    tau,
    *,
    vocabulary=50257,
    radius_fraction=0.0,
    amplitude_max=1.0,
    linear_backend="levels",
    initialization_mode="moments",
    radius_fraction_lr_multiplier=1.0,
):
    return ModelConfig(
        width=512,
        heads=8,
        vocabulary=vocabulary,
        linear_backend=linear_backend,
        radius_mode="envelope",
        tau=tau,
        atom_lr_multiplier=16.0,
        radius_fraction=radius_fraction,
        amplitude_max=amplitude_max,
        initialization_mode=initialization_mode,
        radius_fraction_lr_multiplier=radius_fraction_lr_multiplier,
    )


def options(model):
    return cst_optimizer(
        model,
        atom_lr_multiplier=16.0,
        lr=0.0003,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        foreach=False,
    )


def step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def assert_nested(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested(left, right)
    else:
        assert a == b


def assert_model_resume(model, restored):
    """Numerical state is exact; load invalidates old placement certificates."""
    left, right = model.state_dict(), restored.state_dict()
    assert left.keys() == right.keys()
    transitions = {}
    for key in left:
        if key.endswith("atom_state._extra_state"):
            a, b = left[key], right[key]
            assert a.keys() == b.keys() == {"format_version", "shape", "layout_version"}
            assert a["format_version"] == b["format_version"] == 1
            assert a["shape"] == b["shape"]
            # Restored models are newly constructed: set_extra_state deliberately
            # uses max(current_generation+1, saved_generation).
            assert b["layout_version"] == max(1, a["layout_version"])
            transitions[key] = [a["layout_version"], b["layout_version"]]
        else:
            assert_nested(left[key], right[key])
    return transitions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        help="Explicit tau/fraction/amplitude/backend configurations",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert sha(args.data / "manifest.json") == args.data_manifest_sha256
    manifest, data = verify_data(args.data)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == "NVIDIA RTX 6000 Ada Generation"
    configs = (
        [config_for(tau, vocabulary=manifest["vocabulary"]) for tau in TAUS]
        if args.cases is None
        else [
            config_for(vocabulary=manifest["vocabulary"], **case)
            for case in json.loads(args.cases.read_text())
        ]
    )
    names = [
        "ffn_model.py",
        "ffn_contract.py",
        "ffn_run.py",
        "ffn_radius.py",
        "ffn_envelope.py",
        "ffn_radius_fraction.py",
        "ffn_probe_initialization.py",
        "ffn_fraction_updates.py",
        Path(__file__).name,
    ]
    result = {
        "state": "running",
        "cohort": "recipe",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "data_manifest_sha256": args.data_manifest_sha256,
        "source_sha256": {n: sha(Path(__file__).parent / n) for n in names},
        "task_model_configs": [asdict(c) for c in configs],
        "task_recipes": [c.recipe() for c in configs],
        "cases": [],
        "models": [],
    }
    target = args.output / "qualification.json"

    def save():
        target.write_text(json.dumps(result, indent=2) + "\n")

    save()
    x, y = token_batch(data["train"], [0, 128], 128, "cuda")
    try:
        for config in configs:
            model = LanguageModel(config, "cst", 17).cuda()
            configure_native(model)
            calibration = None
            if config.initialization_mode == "probe_rms":
                from .ffn_probe_initialization import calibrate_model

                calibration = calibrate_model(model, x[:1], second_probe=x[1:])
            initial_parameter_sha256 = parameter_digest(model)
            layers = [m for m in model.modules() if isinstance(m, CSTLinear)]
            assert len(layers) == 8 and all(
                m.atoms.p.shape == (8192, 4) for m in layers
            )
            # Compare physical FP32 rows and every map partial with the independent
            # reference construction, including actual moment-matched atoms.
            for layer in layers:
                p = layer.atoms.p
                spec = layer.kernel.spec.parameterization
                constants = prepare_constants(
                    p,
                    **{
                        n: getattr(spec, n)
                        for n in ("radius_min", "radius_max", "tau", "amplitude_max")
                    },
                )
                old = from_state(layer.kernel, p)
                new = prepared_physical_parameters(p, constants)
                torch.testing.assert_close(old, new, atol=0, rtol=0)
                ga = torch.autograd.grad(old.square().sum(), p)[0]
                gb = torch.autograd.grad(new.square().sum(), p)[0]
                torch.testing.assert_close(ga, gb, atol=0, rtol=0)
            # Two outstanding forwards must retain their own autograd saves.
            _, loss1 = model(x, y)
            _, loss2 = model(x, y)
            loss1.backward()
            grads = {
                n: p.grad.clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            model.zero_grad(set_to_none=True)
            loss2.backward()
            for n, p in model.named_parameters():
                if n in grads:
                    torch.testing.assert_close(p.grad, grads[n], atol=0, rtol=0)
            model.zero_grad(set_to_none=True)
            opt = options(model)
            from .ffn_fraction_updates import FractionUpdates

            updates_observer = FractionUpdates(model)
            initial_fraction = [m.atoms.p[:, 1].detach().clone() for m in layers]
            loss = step(model, opt, x, y)
            updates_observer.update()
            observed_updates = updates_observer.report(opt)
            assert any(
                not torch.equal(m.atoms.p[:, 1], f)
                for m, f in zip(layers, initial_fraction)
            )
            record = {
                "tau": config.tau,
                "config": asdict(config),
                "loss": loss,
                "initialization": [m.initialization_report for m in layers],
                "initial_fraction_learned": True,
                "initial_parameter_sha256": initial_parameter_sha256,
                "outstanding_forwards_equal": True,
                "snapshot": snapshot(model, gradients=True),
                "initial_probe_calibration": calibration,
                "observed_updates": observed_updates,
            }
            # Exact serialized optimizer/moment/step-counter continuation at tau .03.
            if config.tau == 0.03 or (
                config.amplitude_max > 1 and config.radius_fraction in (0.5, 1.0)
            ):
                payload = io.BytesIO()
                torch.save(
                    {"model": model.state_dict(), "optimizer": opt.state_dict()},
                    payload,
                )
                payload.seek(0)
                saved = torch.load(payload, weights_only=False)
                restored = LanguageModel(config, "cst", 17).cuda()
                configure_native(restored)
                restored.load_state_dict(saved["model"])
                other = options(restored)
                other.load_state_dict(saved["optimizer"])
                step(model, opt, x, y)
                step(restored, other, x, y)
                record["load_layout_generations"] = assert_model_resume(model, restored)
                assert len(record["load_layout_generations"]) == 8
                assert_nested(opt.state_dict(), other.state_dict())
                record["serialized_next_step_bitexact"] = True
                del restored, other, saved, payload
            for ni, no in [(512, 2048), (2048, 512)]:
                result["cases"].append(
                    dict(**config.recipe(), **{"in": ni, "out": no, "atoms": 8192})
                )
            result["models"].append(record)
            save()
            print(
                "PASS tau/fraction/cap",
                config.tau,
                config.radius_fraction,
                config.amplitude_max,
                flush=True,
            )
            del model, opt, layers, grads, loss1, loss2
            torch.cuda.empty_cache()
        result["state"] = "passed"
        save()
    except BaseException as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
