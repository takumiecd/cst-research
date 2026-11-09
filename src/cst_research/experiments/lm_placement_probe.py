"""Read-only task probes of projections, residual branches and cotangents."""

import torch

from .ffn_model import parameter_digest
from .ffn_selective import projections, rms


def cosine(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else None


def features(value):
    x = value.detach().reshape(-1, value.shape[-1]).double()
    x = x - x.mean(0, keepdim=True)
    eigen = torch.linalg.eigvalsh(x @ x.T).clamp_min(0)
    total = eigen.sum()
    if total <= 0:
        return {"centered_energy": 0.0, "stable_rank": None, "entropy_rank": None}
    p = eigen / total
    return {
        "centered_energy": float(total),
        "stable_rank": float(total / eigen[-1]),
        "entropy_rank": float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum())),
        "leading_energy_share": float(p[-1]),
    }


def probe(model, x, y, *, retain=False):
    """Unscaled/unclipped single-context CE gradients, outside optimizer/timing."""
    was_training = model.training
    before = parameter_digest(model)
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    model.train()
    model.zero_grad(set_to_none=True)
    records, captured, handles = {}, {}, []
    modules = {}
    for i, block in enumerate(model.blocks):
        modules[f"block{i}"] = block
        modules[f"{i}.norm2"] = block.norm2
        modules[f"{i}.ffn"] = block.ffn
        modules[f"{i}.attention_out"] = block.attention_out
    modules.update(projections(model))
    try:
        for name, module in modules.items():

            def capture(_module, inputs, output, name=name):
                xx = inputs[0]
                row = {"input_rms": rms(xx), "output_rms": rms(output)}
                if name.endswith((".ffn", ".up", ".down")) or name.startswith("block"):
                    row["output_features"] = features(output)
                records[name] = row
                if retain:
                    captured[name] = {
                        "x": xx.detach().float().cpu().clone(),
                        "y": output.detach().float().cpu().clone(),
                    }

                def dx(g):
                    row["input_gradient_rms"] = rms(g)
                    if retain:
                        captured[name]["dx"] = g.detach().float().cpu().clone()

                def dy(g):
                    row["output_gradient_rms"] = rms(g)
                    if retain:
                        captured[name]["dy"] = g.detach().float().cpu().clone()

                xx.register_hook(dx)
                output.register_hook(dy)

            handles.append(module.register_forward_hook(capture))
        logits, loss = model(x, y)
        loss.backward()
        fields = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            assert torch.isfinite(p.grad).all()
            row = {"parameter_rms": rms(p), "gradient_rms": rms(p.grad)}
            if name.endswith("atoms.p"):
                row["field_gradient_rms"] = (
                    p.grad.detach().double().square().mean(0).sqrt().tolist()
                )
            fields[name] = row
        # norm2 input is the residual stream immediately before FFN addition.
        # Capture this small same-basis branch only for the retained context.
        residual = {}
        if retain:
            for i in range(len(model.blocks)):
                base = captured[f"{i}.norm2"]["x"]
                delta = captured[f"{i}.ffn"]["y"]
                g = captured[f"{i}.ffn"]["dy"]
                residual[str(i)] = {
                    "ffn_over_residual_rms": rms(delta) / max(rms(base), 1e-30),
                    "ffn_residual_cosine": cosine(delta, base),
                    "ffn_negative_cotangent_cosine": cosine(delta, -g),
                    "residual_gradient_rms": rms(captured[f"{i}.norm2"]["dx"]),
                }
        result = {
            "ce": float(loss.detach()),
            "layers": records,
            "parameters": fields,
            "residual": residual,
            "correct": int((logits.argmax(-1) == y).sum()),
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    assert parameter_digest(model) == before
    now = dict(model.named_buffers())
    assert now.keys() == buffers.keys()
    assert all(torch.equal(v, now[k]) for k, v in buffers.items())
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    return result, captured
