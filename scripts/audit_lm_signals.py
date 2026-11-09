"""Read-only post-hoc activation/cotangent audit of task-trained LM checkpoints.

Run with the frozen training source on PYTHONPATH, after GPU training has ended.
No optimizer updates, Dense-teacher objective, or confirmation data are used.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def digest_parameters(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def statistics(x, *, spectrum=False):
    x = x.detach().float()
    assert torch.isfinite(x).all()
    rms = float(x.double().square().mean().sqrt())
    sample = x.reshape(-1)
    sample = sample[:: max(1, math.ceil(sample.numel() / 8192))]
    result = {
        "mean": float(x.double().mean()),
        "std": float(x.double().std(unbiased=False)),
        "rms": rms,
        "max_abs": float(x.abs().max()),
        "exact_zero_fraction": float((x == 0).float().mean()),
        "near_zero_relative_fraction": float(
            (x.abs() <= max(rms * 1e-3, 1e-30)).float().mean()
        ),
        "positive_fraction": float((x > 0).float().mean()),
        "sample_abs_quantiles": torch.quantile(
            sample.abs(), torch.tensor([0.1, 0.5, 0.9, 0.99], device=x.device)
        )
        .cpu()
        .tolist(),
        "quantile_sample_values": sample.numel(),
    }
    if spectrum and x.ndim >= 2:
        matrix = x.reshape(-1, x.shape[-1])
        matrix = matrix[:: max(1, math.ceil(matrix.shape[0] / 128))]
        matrix = matrix - matrix.mean(0, keepdim=True)
        gram = matrix @ matrix.T
        eigen = torch.linalg.eigvalsh(gram.double()).clamp_min(0)
        total = eigen.sum()
        result["sample_feature_spectrum"] = {
            "rows": matrix.shape[0],
            "features": matrix.shape[1],
            "participation_rank": float(total.square() / eigen.square().sum())
            if float(total) > 0
            else 0.0,
            "stable_rank": float(total / eigen.max()) if float(total) > 0 else 0.0,
            "scope": "Centered covariance on at most128 token rows, not full operator rank or representational capacity",
        }
    return result


@torch.no_grad()
def attention_statistics(packed, heads):
    b, t, three = packed.shape
    d = three // 3
    assert d % heads == 0
    q, k, v = [
        a.detach().float().reshape(b, t, heads, d // heads).transpose(1, 2)
        for a in packed.chunk(3, -1)
    ]
    score = (q @ k.transpose(-1, -2)) / math.sqrt(d // heads)
    allowed = torch.ones(t, t, device=score.device, dtype=torch.bool).tril()
    prob = score.masked_fill(~allowed, -torch.inf).softmax(-1)
    entropy = -(prob * prob.clamp_min(1e-30).log()).sum(-1)
    normalized = (
        entropy[..., 1:] / torch.arange(2, t + 1, device=score.device).float().log()
    )
    return {
        "q_rms": float(q.double().square().mean().sqrt()),
        "k_rms": float(k.double().square().mean().sqrt()),
        "v_rms": float(v.double().square().mean().sqrt()),
        "legal_logit_std": float(score[..., allowed].double().std(unbiased=False)),
        "entropy_nats_mean": float(entropy.double().mean()),
        "entropy_over_log_available_keys": float(normalized.double().mean())
        if t > 1
        else 0.0,
        "maximum_probability_mean": float(prob.max(-1).values.double().mean()),
        "scope": "Explicit causal softmax from actual Q/K; read-only diagnostic, native SDPA forward unchanged",
    }


def probe(model, x, y):
    model.train()
    model.zero_grad(set_to_none=True)
    values = {}
    attention = {}
    handles = []
    selected = {
        n: m
        for n, m in model.named_modules()
        if n
        and (
            n.split(".")[-1]
            in (
                "norm1",
                "norm2",
                "qkv",
                "attention_out",
                "up",
                "down",
                "ffn",
                "final_norm",
            )
            or (n.startswith("blocks.") and n.count(".") == 1)
        )
    }
    try:
        for name, module in selected.items():

            def capture(_module, inputs, output, name=name):
                if not isinstance(output, torch.Tensor):
                    return
                row = {"output": statistics(output, spectrum=True)}
                values[name] = row
                if inputs and inputs[0].is_floating_point():
                    row["input"] = statistics(inputs[0])
                    if inputs[0].requires_grad:

                        def input_gradient(g):
                            row["input_gradient"] = statistics(g)

                        handles.append(inputs[0].register_hook(input_gradient))
                if output.requires_grad:

                    def output_gradient(g):
                        row["output_gradient"] = statistics(g)

                    handles.append(output.register_hook(output_gradient))
                if name.endswith(".qkv"):
                    attention[name] = attention_statistics(output, model.config.heads)

            handles.append(module.register_forward_hook(capture))
        _, loss = model(x, y)
        loss.backward()
        gradients = {
            n: {
                "rms": float(p.grad.detach().double().square().mean().sqrt()),
                "max_abs": float(p.grad.detach().abs().max()),
                **(
                    {
                        "field_rms": p.grad.detach()
                        .double()
                        .square()
                        .mean(0)
                        .sqrt()
                        .cpu()
                        .tolist()
                    }
                    if n.endswith("atoms.p")
                    else {}
                ),
            }
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        assert all(math.isfinite(v["rms"]) for v in gradients.values())
        return {
            "loss": float(loss.detach()),
            "tokens": y.numel(),
            "layers": values,
            "attention": attention,
            "parameter_gradients": gradients,
        }
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)


def main():
    from cst_research.experiments.ffn_contract import ModelConfig
    from cst_research.experiments.ffn_model import configure_native
    from cst_research.experiments.ffn_run import sha, token_batch, verify_data
    from cst_research.experiments.ffn_selective import make_model

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base-run", type=Path, required=True)
    p.add_argument("--placement-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    roots = [args.base_run, args.placement_run]
    receipts = [json.loads((r / "receipt.json").read_text()) for r in roots]
    assert all(r["state"] == "passed" and r["source_unchanged"] for r in receipts)
    assert all(receipts[0][key] == receipts[1][key] for key in ("gpu", "torch", "cuda"))
    assert (
        receipts[0]["gpu"] == torch.cuda.get_device_name(0)
        and receipts[0]["torch"] == torch.__version__
    )
    assert (
        receipts[0]["protocol"]["data_manifest_sha256"]
        == receipts[1]["protocol"]["data_manifest_sha256"]
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_path = Path(receipts[0]["protocol"]["data"])
    assert (
        sha(data_path / "manifest.json")
        == receipts[0]["protocol"]["data_manifest_sha256"]
    )
    _, data = verify_data(data_path)
    result = {
        "state": "running",
        "evaluator_sha256": sha(Path(__file__)),
        "scope": "Post-hoc TRAIN-only observation; original task labels, no training or confirmation evaluation; no causal or operator-rank claim",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "sources": [r["protocol"]["source_archive_sha256"] for r in receipts],
        "models": [],
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    out = (args.output / "signals.jsonl").open("x")
    try:
        for root, receipt in zip(roots, receipts):
            for arm in receipt["results"]:
                directory = root / arm["arm"]
                s = json.loads((directory / "summary.json").read_text())
                old = {
                    v["step"]: v
                    for v in map(
                        json.loads,
                        (directory / "mechanisms.jsonl").read_text().splitlines(),
                    )
                }
                files = {
                    int(n.split("-")[1]) // 1024: (directory / n, h)
                    for n, h in s["milestone_checkpoint_sha256"].items()
                }
                for step in (0, 976, 4882):
                    saved = None
                    checkpoint_sha = None
                    if step:
                        path, checkpoint_sha = files[step]
                        assert sha(path) == checkpoint_sha
                        saved = torch.load(
                            path, map_location="cpu", weights_only=False, mmap=True
                        )
                    model = make_model(
                        ModelConfig(**s["config"]), s["kind"], tuple(s["sites"]), 17
                    ).cuda()
                    if saved:
                        model.load_state_dict(saved["model"])
                    if s["kind"] == "cst":
                        configure_native(model)
                    before = digest_parameters(model)
                    cpu_rng = torch.get_rng_state().clone()
                    cuda_rng = torch.cuda.get_rng_state().clone()
                    scopes = [("original128", [0])] + (
                        [("additional512", [0, 4096, 8192, 12288])]
                        if step == 4882
                        else []
                    )
                    for label, starts in scopes:
                        x, y = token_batch(data["train"], starts, 128, "cuda")
                        observed = probe(model, x, y)
                        if label == "original128":
                            expected = old[step]["probe"]
                            assert abs(observed["loss"] - expected["loss"]) <= 1e-6
                            for name, v in expected["parameters"].items():
                                torch.testing.assert_close(
                                    torch.tensor(
                                        observed["parameter_gradients"][name]["rms"],
                                        dtype=torch.float64,
                                    ),
                                    torch.tensor(
                                        v["gradient_rms"], dtype=torch.float64
                                    ),
                                    atol=1e-10,
                                    rtol=1e-5,
                                )
                        row = {
                            "arm": arm["arm"],
                            "kind": s["kind"],
                            "sites": s["sites"],
                            "step": step,
                            "probe_scope": label,
                            "checkpoint_sha256": checkpoint_sha,
                            **observed,
                        }
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                    assert before == digest_parameters(model)
                    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(
                        cuda_rng, torch.cuda.get_rng_state()
                    )
                    result["models"].append(
                        {
                            "arm": arm["arm"],
                            "step": step,
                            "checkpoint_sha256": checkpoint_sha,
                            "parameter_and_rng_unchanged": True,
                            "original_probe_ce_and_ffn_gradient_rms_reproduced": True,
                        }
                    )
                    save()
                    print("SIGNAL AUDIT PASS", arm["arm"], step, flush=True)
                    del model, saved
                    torch.cuda.empty_cache()
        assert len(result["models"]) == 51
        result["state"] = "passed"
        save()
    except BaseException as e:
        result.update(state="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise
    finally:
        out.close()


if __name__ == "__main__":
    main()
