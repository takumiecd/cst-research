"""Compare fixed MNIST pixel traversals for a rational CST first layer."""

import argparse
import itertools
import json
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch

from .mnist_fifth_latent_search import (
    dataset,
    epoch_orders,
    save_checkpoint,
    write_report,
)
from .mnist_fixed_decoder_latents import cuda_oracle, digest, evaluate, sha, train
from .mnist_rational_lower_comparison import (
    dense_head_pair,
    historical_formula,
    matched_models,
    project_rational_atoms,
)
from .physical_atom_decoder import PhysicalAtomDecoder

SIDE = 28
SEEDS = (17, 29)
ORDERS = ("row", "serpentine", "morton", "hilbert")


def hilbert_xy(index, size):
    x = y = 0
    span = 1
    while span < size:
        rx = (index // 2) & 1
        ry = (index ^ rx) & 1
        if ry == 0:
            if rx == 1:
                x, y = span - 1 - x, span - 1 - y
            x, y = y, x
        x += span * rx
        y += span * ry
        span *= 2
        index //= 4
    return x, y


def pixel_orders():
    def morton_index(pixel):
        y, x = divmod(pixel, SIDE)
        return sum(
            ((x >> bit & 1) << (2 * bit)) + ((y >> bit & 1) << (2 * bit + 1))
            for bit in range(5)
        )

    row = torch.arange(SIDE**2).reshape(SIDE, SIDE)
    snake = row.clone()
    snake[1::2] = snake[1::2].flip(1)
    hilbert = [hilbert_xy(i, 32) for i in range(32**2)]
    assert all(
        abs(a - c) + abs(b - d) == 1 for (a, b), (c, d) in itertools.pairwise(hilbert)
    )
    return {
        "row": row.flatten(),
        "serpentine": snake.flatten(),
        "morton": torch.tensor(sorted(range(SIDE**2), key=morton_index)),
        "hilbert": torch.tensor(
            [y * SIDE + x for x, y in hilbert if x < SIDE and y < SIDE]
        ),
    }


def pixel_locality(permutation):
    rank = permutation.argsort().reshape(SIDE, SIDE)
    distance = torch.cat(
        ((rank[:, 1:] - rank[:, :-1]).flatten(), (rank[1:] - rank[:-1]).flatten())
    ).abs()
    distance = torch.minimum(distance, SIDE**2 - distance).float()
    return {
        "mean_torus_distance": float(distance.mean()),
        "median_torus_distance": float(distance.median()),
        "fraction_within_8_cells": float((distance <= 8).float().mean()),
    }


def run(args):
    torch.set_num_threads(6)
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090"
    ):
        raise RuntimeError("requires NVIDIA GeForce RTX 4090")
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    baseline = json.loads(args.reference.read_text())
    protocol_source = json.loads(args.orders_reference.read_text())["protocol"]
    assert baseline["state"] == "completed"
    assert baseline["protocol"]["mnist_reference_sha256"] == sha(args.orders_reference)
    assert baseline["protocol"]["historical_receipt_sha256"] == sha(args.history)
    assert baseline["protocol"]["historical_envelope_source_sha256"] == sha(
        args.envelope_source
    )
    assert baseline["protocol"]["warm_decoder_sha256"] == sha(args.warm_function)
    assert baseline["protocol"]["gpu"] == torch.cuda.get_device_name()
    assert baseline["protocol"]["torch"] == torch.__version__
    assert baseline["protocol"]["cuda"] == torch.version.cuda
    formula = historical_formula(
        args.envelope_source, json.loads(args.history.read_text())
    )
    x, y, fit, val, test_x, test_y, data_sha = dataset(args.data)
    assert data_sha == protocol_source["data_sha256"]
    assert digest(val) == protocol_source["validation_indices_sha256"]
    cpu_orders = {seed: epoch_orders(fit, seed) for seed in SEEDS}
    assert all(
        [digest(row) for row in orders]
        == protocol_source["epoch_order_sha256"][str(seed)]
        for seed, orders in cpu_orders.items()
    )
    warm = torch.load(args.warm_function, map_location="cpu", weights_only=True)
    decoder = PhysicalAtomDecoder(warm["amplitude_scale"], period=warm["period"])
    decoder.load_state_dict(warm["decoder"])
    permutations = pixel_orders()
    for p in permutations.values():
        assert torch.equal(p.sort().values, torch.arange(SIDE**2))
    protocol = {
        "screen_seed": SEEDS[0],
        "confirmation_seed": SEEDS[1],
        "screen_epochs": 2,
        "final_epochs": 5,
        "selection": "lowest epoch-two validation CE among the three non-row layouts",
        "reference_sha256": sha(args.reference),
        "orders_reference_sha256": sha(args.orders_reference),
        "source_sha256": {
            name: sha(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "mnist_rational_lower_comparison.py",
                "mnist_fixed_decoder_latents.py",
                "mnist_fifth_latent_search.py",
                "match_physical_atom_init.py",
                "physical_atom_decoder.py",
            )
        },
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torchcst_commit": baseline["protocol"]["torchcst_commit"],
        "execution": baseline["protocol"]["execution"],
        "layout_indices": {name: p.tolist() for name, p in permutations.items()},
        "locality": {name: pixel_locality(p) for name, p in permutations.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "result.json"
    report = (
        json.loads(target.read_text())
        if target.is_file()
        else {
            "state": "running",
            "protocol": protocol,
            "seeds": {},
        }
    )
    assert report["protocol"] == protocol
    if report["state"] == "completed":
        return
    write_report(target, report)
    cpu_x, cpu_y = x, y
    x, y = x.cuda(), y.cuda()
    test_x, test_y, val = test_x.cuda(), test_y.cuda(), val.cuda()
    gpu_orders = {s: [p.cuda() for p in batches] for s, batches in cpu_orders.items()}

    def prepare(seed):
        matched, _ = matched_models(decoder, formula, seed)
        model = dense_head_pair(matched["rational_q"], seed)["cst_dense"]
        if seed == SEEDS[0]:
            probe = cpu_orders[seed][0][:16]
            report["cuda_oracle"] = cuda_oracle(
                model,
                deepcopy(model).cuda(),
                cpu_x[probe].index_select(1, permutations["hilbert"]),
                cpu_y[probe],
            )
        model.cuda()
        physical = model.layers[0].physical(model.decoder)
        assert (
            digest(physical)
            == baseline["seeds"][str(seed)]["initial_first_layer_sha256"]
        )
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert (
            parameters
            == baseline["seeds"][str(seed)]["trainable_parameters"]["cst_dense"]
        )
        record = report["seeds"].setdefault(str(seed), {"arms": {}})
        record["initial_first_layer_sha256"] = digest(physical)
        record["trainable_parameters"] = parameters
        record["baseline"] = {
            name: baseline["seeds"][str(seed)]["arms"][name]
            for name in ("cst_dense", "dense")
        }
        write_report(target, report)
        return model, record["arms"]

    def fit_layout(seed, name, epochs, initial, arms):
        previous = arms.get(name, {})
        if previous.get("complete") and len(previous["epochs"]) >= epochs:
            return
        began = time.perf_counter()
        pix = permutations[name].cuda()
        inputs = x if name == "row" else x.index_select(1, pix)
        model = deepcopy(initial)
        directory = args.output / str(seed) / name
        directory.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(directory.glob("epoch-*.pt"))
        state = (
            torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
            if checkpoints
            else None
        )
        print(
            "START",
            seed,
            name,
            "from",
            len(state["epochs"]) if state else 0,
            "to",
            epochs,
            flush=True,
        )

        def after_epoch(checkpoint):
            save_checkpoint(directory, checkpoint)
            arms[name] = {
                "complete": False,
                "epochs": checkpoint["epochs"],
                "phase_wall_seconds": previous.get("phase_wall_seconds", []),
            }
            write_report(target, report)

        result = train(
            f"{seed}/{name}",
            model,
            inputs,
            y,
            gpu_orders[seed][:epochs],
            inputs[val],
            y[val],
            resume=state,
            on_epoch=after_epoch,
            after_update=partial(project_rational_atoms, model),
        )
        arms[name] = {
            **result,
            "complete": True,
            "phase_wall_seconds": previous.get("phase_wall_seconds", [])
            + [time.perf_counter() - began],
        }
        if epochs == 5:
            arms[name]["test"] = evaluate(model, test_x.index_select(1, pix), test_y)
        write_report(target, report)

    initial, arms = prepare(SEEDS[0])
    for name in ORDERS:
        fit_layout(SEEDS[0], name, 2, initial, arms)
    first = baseline["seeds"][str(SEEDS[0])]["arms"]["cst_dense"]
    assert all(
        abs(a["validation"]["ce"] - b["validation"]["ce"]) < 1e-7
        for a, b in zip(arms["row"]["epochs"][:2], first["epochs"][:2], strict=True)
    )
    ranked = sorted(ORDERS[1:], key=lambda n: arms[n]["epochs"][1]["validation"]["ce"])
    winner = ranked[0]
    report["selected_layout"] = winner
    report["epoch_two_rank_non_row"] = ranked
    write_report(target, report)
    print("SELECTED", winner, flush=True)
    fit_layout(SEEDS[0], winner, 5, initial, arms)
    initial, arms = prepare(SEEDS[1])
    fit_layout(SEEDS[1], winner, 5, initial, arms)
    report["state"] = "completed"
    report["wall_seconds"] = time.perf_counter() - started
    write_report(target, report)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for key in (
        "data",
        "warm-function",
        "history",
        "envelope-source",
        "reference",
        "orders-reference",
        "output",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
