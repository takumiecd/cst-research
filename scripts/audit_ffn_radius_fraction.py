"""Read-only trained checkpoint audit against encoded-row FP64 references.

Run with the frozen experiment source on PYTHONPATH, outside that source tree.
"""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torchcst import CSTLinear, kernels
from torchcst._backends.torch.parameterizations.radius_envelope import from_state

from cst_research.experiments.ffn_contract import ModelConfig
from cst_research.experiments.ffn_model import LanguageModel, configure_native
from cst_research.experiments.ffn_run import evaluate, sha, verify_data


def layer_oracle(layer, seed):
    cpu = copy.deepcopy(layer).cpu()
    p = cpu.atoms.p
    physical = from_state(cpu.kernel, p)
    op = cpu.execution_declaration()
    reference = CSTLinear(
        *op.charts,
        atoms=1,
        dtype=torch.float64,
        backend="factored",
        kernel=kernels.product(
            *op.kernel.profiles,
            coordinates=kernels.physical(radius_init=(0.5, 50.0), radius_floor=0.5),
        ),
    )
    vi, uo = reference.operator.factors(physical.double())
    data = torch.Generator().manual_seed(seed)
    x = torch.randn(7, layer.in_features * 2, generator=data)
    dy = torch.randn(7, layer.out_features, generator=data)
    cx = x.double()[:, ::2].requires_grad_()
    cy = (cx @ vi) @ uo.T
    cg = torch.autograd.grad(cy, (cx, p), dy.double())
    gx = x.cuda()[:, ::2].requires_grad_()
    gy = layer(gx)
    gg = torch.autograd.grad(gy, (gx, layer.atoms.p), dy.cuda())
    errors = []
    for actual, expected in zip((gy, *gg), (cy, *cg)):
        actual = actual.detach().cpu().double()
        expected = expected.detach().double()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=4e-3)
        errors.append(float((actual - expected).abs().max()))
    return {
        "rows": 7,
        "noncontiguous_input": True,
        "max_abs_y_dx_dp": errors,
        "atol": 1e-4,
        "rtol": 4e-3,
        "reference": "FP64 factored physical kernels, retaining original FP32 physical-row encoding and complete map VJP",
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.run.resolve()
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["status"] == "passed" and receipt["source_unchanged"]
    protocol = receipt["protocol"]
    assert (
        torch.cuda.get_device_name(0) == receipt["gpu"]
        and torch.__version__ == receipt["torch"]
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert (
        sha(Path(protocol["data"]) / "manifest.json")
        == protocol["data_manifest_sha256"]
    )
    _, data = verify_data(Path(protocol["data"]))
    result = {
        "state": "running",
        "audit_source_sha256": sha(Path(__file__)),
        "models": [],
        "confirmation_evaluated": False,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for row in receipt["results"]:
            if row["stage"] != "long":
                continue
            directory = root / row["label"]
            summary = json.loads((directory / "summary.json").read_text())
            checkpoint = directory / "checkpoint.pt"
            assert sha(checkpoint) == summary["checkpoint_sha256"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = ModelConfig(**saved["config"])
            assert saved["updates"] == summary["completed_updates"] == 976
            assert saved["completed_tokens"] == summary["completed_tokens"] == 999424
            assert saved["seed"] == row["seed"] and saved["kind"] == row["kind"]
            assert asdict(config) == summary["config"]
            assert saved["data_manifest_sha256"] == protocol["data_manifest_sha256"]
            model = LanguageModel(config, row["kind"], row["seed"]).cuda()
            model.load_state_dict(saved["model"])
            if row["kind"] == "cst":
                configure_native(model)
            states = saved["optimizer"]["state"]
            for state in states.values():
                assert int(state["step"]) == 976
                for name in ("exp_avg", "exp_avg_sq"):
                    assert (
                        state[name].dtype == torch.float32
                        and torch.isfinite(state[name]).all()
                    )
            observed = evaluate(model, data["development"], 2)["ce"]
            expected = summary["latest_development"]["ce"]
            assert abs(observed - expected) <= 1e-6, (observed, expected)
            record = {
                **row,
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "reproduced_development_ce": observed,
                "ce_absolute_difference": abs(observed - expected),
                "optimizer_states_checked": len(states),
                "linear_oracles": [],
            }
            if row["kind"] == "cst":
                layers = [m for m in model.modules() if isinstance(m, CSTLinear)]
                assert len(layers) == 8
                for layer in layers:
                    p = layer.atoms.p
                    assert torch.isfinite(p).all() and (p[:, 0].abs() <= 1).all()
                    assert ((p[:, 1] >= 0) & (p[:, 1] <= 1)).all()
                for i, layer in enumerate(
                    (model.blocks[0].ffn.up.map, model.blocks[0].ffn.down.map)
                ):
                    record["linear_oracles"].append(
                        layer_oracle(layer, row["seed"] + i + 640)
                    )
            result["models"].append(record)
            save()
            print("AUDIT PASS", row["label"], flush=True)
            del model, saved, states
            torch.cuda.empty_cache()
        assert len(result["models"]) == 9
        result["state"] = "passed"
        save()
    except BaseException as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
