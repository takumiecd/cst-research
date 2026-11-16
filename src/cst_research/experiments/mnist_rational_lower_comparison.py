"""Port the historical rational lower envelope to the paired MNIST classifier."""

import argparse
import importlib.util
import json
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .match_physical_atom_init import matched_latents
from .mnist_fifth_latent_search import (
    dataset,
    epoch_orders,
    save_checkpoint,
    write_report,
)
from .mnist_fixed_decoder_latents import (
    Classifier,
    DecoderLayer,
    cuda_oracle,
    digest,
    evaluate,
    sha,
    stats,
    train,
)
from .physical_atom_decoder import PhysicalAtomDecoder

SEEDS = (17, 29)
ARMS = ("rational_q", "input_center")


def historical_formula(source, receipt):
    arm = receipt["arms"]["rational_learned"]
    if (
        receipt["state"] != "completed"
        or arm["checkpoint_audit"] != "passed"
        or sha(source) != receipt["source_change"]["after"]
    ):
        raise ValueError("historic envelope source and completed receipt must match")
    spec = importlib.util.spec_from_file_location("historical_radius_envelope", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.USE_RATIONAL_LOWER = True
    return module.physical_parameters


class RationalLayer(DecoderLayer):
    def __init__(self, *args, formula):
        super().__init__(*args)
        self.formula = formula

    def physical(self, decoder):
        return self.formula(
            self.cst.atoms.p,
            radius_min=0.5,
            radius_max=self.max_radius,
            tau=0.03,
            amplitude_max=32.0,
        )


class DenseHeadClassifier(Classifier):
    def __init__(self, rational, head):
        nn.Module.__init__(self)
        self.decoder = rational.decoder
        self.layers = nn.ModuleList([rational.layers[0]])
        self.head = deepcopy(head)

    def forward(self, x):
        return self.head(F.gelu(self.layers[0](x, self.decoder)))


def dense_head_pair(rational, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dense = nn.Sequential(nn.Linear(784, 128), nn.GELU(), nn.Linear(128, 10))
    hybrid = DenseHeadClassifier(rational, dense[-1])
    for actual, expected in zip(
        hybrid.head.parameters(), dense[-1].parameters(), strict=True
    ):
        assert torch.equal(actual, expected)
    return {"cst_dense": hybrid, "dense": dense}


@torch.no_grad()
def matched_models(decoder, formula, seed):
    input_center = Classifier(
        decoder, extra_latent=True, seed=seed, fifth_coupling=(0.0, 0.0, 0.0, 1.0)
    )
    rational = Classifier(
        decoder,
        extra_latent=False,
        seed=seed,
        layer_factory=partial(RationalLayer, formula=formula),
    )
    double = deepcopy(decoder).double()
    records = []
    for learned, analytic in zip(input_center.layers, rational.layers, strict=True):
        old_physical = learned.physical(input_center.decoder)
        raw = learned.cst.atoms.p
        scale = decoder.period / raw.new_tensor(learned.bounds)
        source_latent = torch.cat((raw[:, :2], raw[:, 2:] * scale), dim=1)
        parameters = analytic.cst.atoms.p
        parameters.copy_(old_physical)
        parameters[:, 1].zero_()
        expected = analytic.physical(rational.decoder)
        expected_1024 = expected.clone()
        expected_1024[:, 2:] *= scale
        target = double.preclip(source_latent.double())
        target[:, 0] = expected[:, 0]
        target[:, 1] = expected[:, 1].double().log()
        center_shift = expected_1024[:, 2:].double() - target[:, 2:]
        target[:, 2:] += (center_shift + decoder.period / 2).remainder(
            decoder.period
        ) - decoder.period / 2
        restored, diff, okay = matched_latents(
            double, target, source_latent, decoder, expected_1024
        )
        if not okay:
            raise ValueError(
                f"initial decoder inversion did not match: {diff.tolist()}"
            )
        restored[:, 2:] /= scale
        raw.copy_(restored)
        records.append(
            {
                "atoms": len(expected),
                "before_alignment_radius_median": float(old_physical[:, 1].median()),
                "matched_radius_median": float(expected[:, 1].median()),
                "radius_max": analytic.max_radius,
                "max_abs_latent_change": (
                    source_latent - torch.cat((raw[:, :2], raw[:, 2:] * scale), dim=1)
                )
                .abs()
                .amax(0)
                .tolist(),
            }
        )
    return {"rational_q": rational, "input_center": input_center}, records


@torch.no_grad()
def check_geometry(models):
    rational, learned = (models[name] for name in ARMS)
    errors = []
    for a, b in zip(rational.layers, learned.layers, strict=True):
        desired = a.physical(rational.decoder)
        actual = b.physical(learned.decoder)
        periods = desired.new_tensor(a.bounds)
        difference = actual - desired
        difference[:, 1] = (actual[:, 1] / desired[:, 1]).log()
        difference[:, 2:] = (difference[:, 2:] + periods / 2).remainder(
            periods
        ) - periods / 2
        bound = difference.new_tensor([6e-7, 6e-6, 1.5e-4, 1.5e-4])
        error = difference.abs().amax(0)
        if not bool((error <= bound).all()):
            raise ValueError(f"initial physical geometry mismatch: {error.tolist()}")
        errors.append(error.tolist())
    return errors


@torch.no_grad()
def project_rational_atoms(model):
    for layer in model.layers:
        p = layer.cst.atoms.p
        p[:, 0].clamp_(-32, 32)
        p[:, 1].clamp_(0, 1)


def run(args):
    torch.set_num_threads(6)
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090"
    ):
        raise RuntimeError("requires NVIDIA GeForce RTX 4090")
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    old = json.loads(args.history.read_text())
    formula = historical_formula(args.envelope_source, old)
    previous = json.loads(args.mnist_reference.read_text())
    manifest = previous["protocol"]
    full_cst = None
    if args.full_cst_reference:
        full_cst = json.loads(args.full_cst_reference.read_text())
        assert full_cst["state"] == "completed"
        assert full_cst["protocol"]["mnist_reference_sha256"] == sha(
            args.mnist_reference
        )
    arm_names = ("cst_dense", "dense") if full_cst else ARMS
    x, y, fit, val, test_x, test_y, data_sha = dataset(args.data)
    assert data_sha == manifest["data_sha256"]
    assert digest(val) == manifest["validation_indices_sha256"]
    assert sha(args.warm_function) == manifest["warm_decoder_sha256"]
    assert (
        torch.__version__ == manifest["torch"]
        and torch.version.cuda == manifest["cuda"]
    )
    order = {seed: epoch_orders(fit, seed) for seed in SEEDS}
    assert all(
        [digest(row) for row in rows] == manifest["epoch_order_sha256"][str(seed)]
        for seed, rows in order.items()
    )
    warm = torch.load(args.warm_function, weights_only=True, map_location="cpu")
    decoder = PhysicalAtomDecoder(warm["amplitude_scale"], period=warm["period"])
    decoder.load_state_dict(warm["decoder"])
    examples = order[SEEDS[0]][0][:16]
    val_x, val_y = x[val].cuda(), y[val].cuda()
    test_x, test_y = test_x.cuda(), test_y.cuda()
    gpu_x, gpu_y = x.cuda(), y.cuda()
    protocol = {
        "seeds": list(SEEDS),
        "configuration": list(arm_names),
        "full_cst_reference_sha256": sha(args.full_cst_reference) if full_cst else None,
        "epochs": len(order[SEEDS[0]]),
        "mnist_reference_sha256": sha(args.mnist_reference),
        "historical_receipt_sha256": sha(args.history),
        "historical_envelope_source_sha256": sha(args.envelope_source),
        "source_sha256": {
            name: sha(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "mnist_fixed_decoder_latents.py",
                "mnist_fifth_latent_search.py",
                "match_physical_atom_init.py",
                "physical_atom_decoder.py",
            )
        },
        "torchcst_commit": manifest["torchcst_commit"],
        "execution": manifest["execution"],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "historical_tinystories_ce": old["arms"]["rational_learned"][
            "final_development"
        ]["ce"],
        "radius_upper": [64.0] if full_cst else [64.0, 5.0],
        "radius_lower": old["protocol"]["lower"],
        "q_start": 0.0,
        "q_project": [0.0, 1.0],
        "tau_constant_unused_by_rational_lower": 0.03,
        "optimizer": "MNIST Adam with atom lr .02, bias lr .003, post-update a/q projection for rational_q",
        "warm_decoder_sha256": sha(args.warm_function),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "result.json"
    report = (
        json.loads(destination.read_text())
        if destination.is_file()
        else {
            "state": "running",
            "protocol": protocol,
            "seeds": {},
        }
    )
    assert report["protocol"] == protocol
    if report["state"] == "completed":
        return
    write_report(destination, report)

    def fit_arm(seed, name, model, record, cuda_orders):
        directory = args.output / str(seed) / name
        directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(directory.glob("epoch-*.pt"))
        state = (
            torch.load(existing[-1], map_location="cpu", weights_only=True)
            if existing
            else None
        )
        first_q = list(state.get("first_q", [])) if state else []
        rational = name in ("rational_q", "cst_dense")

        @torch.no_grad()
        def project_q():
            project_rational_atoms(model)
            if not first_q:
                first_q.extend(
                    {
                        "gradient": stats(layer.cst.atoms.p.grad[:, 1]),
                        "after_update": stats(layer.cst.atoms.p[:, 1]),
                    }
                    for layer in model.layers
                )
                assert all(v["after_update"]["nonzero"] > 0 for v in first_q)

        def after_epoch(checkpoint):
            checkpoint["first_q"] = first_q
            save_checkpoint(directory, checkpoint)
            record["arms"][name] = {
                "complete": False,
                "epochs": checkpoint["epochs"],
            }
            write_report(destination, report)

        result = train(
            f"{seed}/{name}",
            model,
            gpu_x,
            gpu_y,
            cuda_orders,
            val_x,
            val_y,
            resume=state,
            on_epoch=after_epoch,
            after_update=project_q if rational else None,
        )
        record["arms"][name] = {
            **result,
            "test": evaluate(model, test_x, test_y),
            "complete": True,
            "first_q": first_q,
            "final_q": [stats(layer.cst.atoms.p[:, 1]) for layer in model.layers]
            if rational
            else [],
        }
        write_report(destination, report)

    for seed, orders in order.items():
        record = report["seeds"].setdefault(str(seed), {"arms": {}})
        if all(record["arms"].get(name, {}).get("complete") for name in arm_names):
            continue
        print("MATCHING_INITIAL_ATOMS", seed, flush=True)
        cpu, initial = matched_models(decoder, formula, seed)
        if full_cst:
            rational = cpu["rational_q"]
            reference = full_cst["seeds"][str(seed)]
            assert (
                initial[0]["matched_radius_median"]
                == reference["initial_geometry"][0]["matched_radius_median"]
            )
            assert initial[0]["atoms"] == reference["initial_geometry"][0]["atoms"]
            cpu = dense_head_pair(rational, seed)
            initial = initial[:1]
            prior = reference["arms"]["rational_q"]
            record["full_cst_reference"] = {
                key: prior[key]
                for key in ("epochs", "validation", "test", "training_seconds")
            }
        gpu = deepcopy(cpu)
        for model in gpu.values():
            model.cuda()
        if seed == SEEDS[0]:
            key = arm_names[0]
            report["cuda_oracle"] = cuda_oracle(
                cpu[key], gpu[key], x[examples], y[examples]
            )
        record["initial_geometry"] = initial
        record["trainable_parameters"] = {
            name: sum(p.numel() for p in model.parameters() if p.requires_grad)
            for name, model in gpu.items()
        }
        record["initial_validation"] = {
            name: evaluate(model, val_x, val_y) for name, model in gpu.items()
        }
        if full_cst:
            record["initial_first_layer_sha256"] = digest(
                gpu["cst_dense"].layers[0].physical(gpu["cst_dense"].decoder)
            )
            print(
                "INITIAL_HYBRID",
                seed,
                json.dumps(record["trainable_parameters"]),
                flush=True,
            )
        else:
            record["max_physical_error"] = check_geometry(gpu)
            assert (
                abs(
                    record["initial_validation"]["rational_q"]["ce"]
                    - record["initial_validation"]["input_center"]["ce"]
                )
                < 3e-6
            )
            print(
                "INITIAL_GEOMETRY_CHECKED",
                seed,
                json.dumps(record["max_physical_error"]),
                flush=True,
            )
        write_report(destination, report)
        cuda_orders = [batch_order.cuda() for batch_order in orders]
        for name, model in gpu.items():
            if record["arms"].get(name, {}).get("complete"):
                continue
            fit_arm(seed, name, model, record, cuda_orders)
    report["state"] = "completed"
    report["wall_seconds"] = time.perf_counter() - started
    write_report(destination, report)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for key in (
        "data",
        "warm-function",
        "history",
        "envelope-source",
        "mnist-reference",
        "output",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--full-cst-reference", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
