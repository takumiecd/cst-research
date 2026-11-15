"""Compare four and five trainable Atom latents with a frozen shared decoder."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .physical_atom_decoder import PhysicalAtomDecoder

WIDTHS = (784, 128, 10)
BATCH_SIZE = 256
TRAIN_COUNT = 55_000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


class DecoderLayer(nn.Module):
    """Evaluate the physical Atom sum after decoding rectangular chart coordinates."""

    def __init__(self, inputs, outputs, extra_latent):
        from torchcst import CSTLinear, chart_presets, kernels

        super().__init__()
        layouts = {784: (16, 7), 128: (1, 2), 10: (5, 2)}

        def cell_mass(size):
            base, merge = layouts[size]
            return kernels.cell_mass(base_tile_sites=base, merge_factor=merge)

        self.bounds = (outputs, inputs)
        self.max_radius = min(self.bounds) / 2
        self.cst = CSTLinear(
            *(
                chart_presets.regular_grid(n, geometry="flat_torus")
                for n in self.bounds
            ),
            atoms=(inputs * outputs) // 16,
            kernel=kernels.product(
                *(cell_mass(n) for n in self.bounds),
                coordinates=kernels.physical(
                    radius_init=(1.0, 2.0), amplitude_std=0.35, radius_floor=0.5
                ),
            ),
            backend="factored",
        )
        self.bias = nn.Parameter(torch.zeros(outputs))
        with torch.no_grad():
            self.cst.atoms.p[:, 1].log_()
        self.fifth = (
            nn.Parameter(self.cst.atoms.p.new_zeros(self.cst.atom_count))
            if extra_latent
            else None
        )

    def physical(self, decoder):
        latent = self.cst.atoms.p
        periods = latent.new_tensor(self.bounds)
        scale = decoder.period / periods
        common = decoder.preclip(
            torch.cat((latent[:, :2], latent[:, 2:] * scale), dim=1)
        )
        amp, log_r, out, inp = common.unbind(1)
        if self.fifth is not None:
            extra = self.fifth.tanh()
            amplitude_scale = decoder.amplitude_scale
            amp = amp + extra * amplitude_scale * (amp / amplitude_scale).tanh()
            log_r = log_r - extra
        return torch.stack(
            (
                amp.clamp(-32, 32),
                log_r.clamp(math.log(0.5), math.log(self.max_radius))
                .exp()
                .clamp(0.5, self.max_radius),
                (out / scale[0]).remainder(periods[0]),
                (inp / scale[1]).remainder(periods[1]),
            ),
            dim=1,
        )

    def forward(self, x, decoder):
        from torchcst._backends.torch.operators.linear import factors

        inputs, outputs = factors(self.cst.operator, self.physical(decoder))
        return F.linear(x, outputs @ inputs.T, self.bias)


class Classifier(nn.Module):
    def __init__(self, decoder, *, extra_latent, seed):
        super().__init__()
        self.decoder = deepcopy(decoder).requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.layers = nn.ModuleList(
                DecoderLayer(i, o, extra_latent) for i, o in itertools.pairwise(WIDTHS)
            )

    def forward(self, x):
        hidden, output = self.layers
        return output(F.gelu(hidden(x, self.decoder)), self.decoder)


def stats(tensor):
    value = tensor.detach().abs()
    assert torch.isfinite(value).all()
    return {
        "count": value.numel(),
        "nonzero": int(value.count_nonzero()),
        "rms": float(value.square().mean().sqrt()),
        "median_abs": float(value.median()),
        "max_abs": float(value.max()),
    }


def first_gradients(model, loss):
    gradients = torch.autograd.grad(
        loss, [p for p in model.parameters() if p.requires_grad]
    )
    return dict(
        zip(
            (name for name, p in model.named_parameters() if p.requires_grad),
            gradients,
            strict=True,
        )
    )


@torch.no_grad()
def evaluate(model, x, y):
    logits = model(x)
    return {
        "ce": float(F.cross_entropy(logits, y)),
        "accuracy": float((logits.argmax(1) == y).float().mean()),
        "examples": len(y),
    }


@torch.no_grad()
def factored_oracle(model, inputs):
    from torchcst._backends.torch.operators.dispatch import factored

    expected = inputs
    for index, layer in enumerate(model.layers):
        expected = (
            factored(layer.cst, expected, layer.physical(model.decoder)) + layer.bias
        )
        if index == 0:
            expected = F.gelu(expected)
    torch.testing.assert_close(model(inputs), expected, rtol=1e-5, atol=2e-6)


def train(name, model, x, y, indices, val_x, val_y):
    with_fifth = isinstance(model, Classifier) and model.layers[0].fifth is not None
    if isinstance(model, Classifier):
        atoms = [layer.cst.atoms.p for layer in model.layers]
        if with_fifth:
            atoms.extend(layer.fifth for layer in model.layers)
        groups = [
            {"params": atoms, "lr": 0.02},
            {"params": [layer.bias for layer in model.layers]},
        ]
        weights = {n: p.clone() for n, p in model.decoder.named_parameters()}
    else:
        groups = model.parameters()
        weights = {}
    optimizer = torch.optim.Adam(groups, lr=0.003)
    assert not optimizer.state
    accumulated, first = 0.0, []
    started = time.perf_counter()
    for step, idx in enumerate(indices.split(BATCH_SIZE), 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        loss.backward()
        optimizer.step()
        accumulated += loss.detach().item() * len(idx)
        if step == 1 and with_fifth:
            first = [
                {
                    "gradient": stats(layer.fifth.grad),
                    "after_update": stats(layer.fifth),
                }
                for layer in model.layers
            ]
            assert all(d["after_update"]["nonzero"] > 0 for d in first)
        if step % 50 == 0:
            print(name, "step", step, "train CE", float(loss.detach()), flush=True)
    epoch_seconds = time.perf_counter() - started
    assert all(int(s["step"]) == step for s in optimizer.state.values())
    if weights:
        for n, param in model.decoder.named_parameters():
            assert torch.equal(param, weights[n]) and param.grad is None
    fifth = [stats(layer.fifth) for layer in model.layers] if with_fifth else []
    result = {
        "steps": step,
        "train_ce": accumulated / len(indices),
        "epoch_seconds": epoch_seconds,
        "validation": evaluate(model, val_x, val_y),
        "fifth_first_step": first,
        "fifth_final": fifth,
    }
    print(name, json.dumps(result, allow_nan=False), flush=True)
    return result


def main():
    import torchcst
    from benchmarks.training.cell_mass_mnist import load_mnist

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--warm-function", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    data, data_receipt = load_mnist(args.data)
    warm = torch.load(args.warm_function, map_location="cpu", weights_only=True)
    decoder = PhysicalAtomDecoder(warm["amplitude_scale"], period=warm["period"])
    decoder.load_state_dict(warm["decoder"])
    # The official test set remains untouched during candidate selection.
    split = torch.randperm(60_000, generator=torch.Generator().manual_seed(19))
    train_indices, val_indices = split.split((TRAIN_COUNT, 60_000 - TRAIN_COUNT))
    order = train_indices[
        torch.randperm(TRAIN_COUNT, generator=torch.Generator().manual_seed(args.seed))
    ]
    x = data["train_x"].float().div_(255)
    y = data["train_y"]
    train_x = x[train_indices]
    mean, std = train_x.mean().item(), train_x.std(unbiased=False).item()
    x.sub_(mean).div_(std)
    del train_x
    val_x, val_y = x[val_indices], y[val_indices]
    four = Classifier(decoder, extra_latent=False, seed=args.seed)
    five = Classifier(decoder, extra_latent=True, seed=args.seed)
    initial_geometries = []
    first_x, first_y = x[order[:BATCH_SIZE]], y[order[:BATCH_SIZE]]
    with torch.no_grad():
        for a, b in zip(four.layers, five.layers, strict=True):
            p0, p1 = a.physical(four.decoder), b.physical(five.decoder)
            assert torch.equal(p0, p1)
            initial_geometries.append({"atoms": len(p0), "sha256": digest(p0)})
        torch.testing.assert_close(four(first_x), five(first_x), rtol=0, atol=0)
    factored_oracle(four, first_x[:16])
    grad4 = first_gradients(four, F.cross_entropy(four(first_x), first_y))
    grad5 = first_gradients(five, F.cross_entropy(five(first_x), first_y))
    for name, value in grad4.items():
        torch.testing.assert_close(value, grad5[name], rtol=0, atol=0)
    for key in ("layers.0.fifth", "layers.1.fifth"):
        assert stats(grad5[key])["nonzero"] > 0, key
    initial_val = evaluate(four, val_x, val_y)
    print("INITIAL_GEOMETRY", initial_geometries, "CE", initial_val, flush=True)
    # This fully dense arm is a task sanity check with its ordinary initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        dense = nn.Sequential(nn.Linear(784, 128), nn.GELU(), nn.Linear(128, 10))
    source = Path(torchcst.__file__).resolve().parents[2]
    receipt = {
        "protocol": {
            "widths": WIDTHS,
            "epochs": 1,
            "batch_size": BATCH_SIZE,
            "seed": args.seed,
            "split_seed": 19,
            "train_examples": len(order),
            "val_examples": len(val_indices),
            "train_order_sha256": digest(order),
            "validation_indices_sha256": digest(val_indices),
            "data_sha256": data_receipt["sha256"],
            "normalization": {"mean": mean, "std": std},
            "atom_lr": 0.02,
            "bias_and_dense_lr": 0.003,
            "warm_decoder_sha256": sha(args.warm_function),
            "runner_sha256": sha(Path(__file__)),
            "torchcst_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "torch": torch.__version__,
            "device": "cpu",
            "platform": platform.platform(),
            "threads": args.threads,
            "timing": "batches, forward, backward and Adam; excludes validation",
        },
        "initial_geometry": initial_geometries,
        "initial_validation_cst": initial_val,
        "factored_oracle": "passed",
        "initial_logits_and_common_gradients": "exact",
        "preparation_seconds": time.perf_counter() - started,
        "arms": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, model in (("four", four), ("five", five), ("dense", dense)):
        receipt["arms"][name] = train(name, model, x, y, order, val_x, val_y)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
