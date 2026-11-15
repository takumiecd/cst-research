"""Screen fixed fifth-latent couplings in a shared MNIST CUDA process."""

import argparse
import json
import time
from pathlib import Path

import torch

from .mnist_fixed_decoder_latents import (
    BATCH_SIZE,
    DEFAULT_COUPLING,
    TRAIN_COUNT,
    Classifier,
    cuda_oracle,
    digest,
    evaluate,
    sha,
    train,
)
from .physical_atom_decoder import PhysicalAtomDecoder

CANDIDATES = {
    "four": None,
    "current": DEFAULT_COUPLING,
    "amplitude": (1.0, 0.0, 0.0, 0.0),
    "radius": (0.0, -1.0, 0.0, 0.0),
    "amp_radius_same_sign": (1.0, 1.0, 0.0, 0.0),
    "amp_radius_weaker": (0.5, -0.5, 0.0, 0.0),
    "amp_radius_stronger": (2.0, -2.0, 0.0, 0.0),
    "amp_heavy": (2.0, -0.5, 0.0, 0.0),
    "radius_heavy": (0.5, -2.0, 0.0, 0.0),
    "output_center": (0.0, 0.0, 1.0, 0.0),
    "input_center": (0.0, 0.0, 0.0, 1.0),
    "both_centers": (0.0, 0.0, 1.0, 1.0),
    "amp_radius_both_centers": (1.0, -1.0, 1.0, 1.0),
}
SCREEN_SEED = 17
CONFIRM_SEED = 29
SCREEN_EPOCHS = 2
FINAL_EPOCHS = 5
PROMOTED_COUNT = 3
CONTROLS = ("four", "current")


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def epoch_orders(train_indices, seed):
    generator = torch.Generator().manual_seed(seed)
    return [
        train_indices[torch.randperm(len(train_indices), generator=generator)]
        for _ in range(FINAL_EPOCHS)
    ]


def save_checkpoint(directory, state):
    name = f"epoch-{len(state['epochs'])}.pt"
    destination = directory / name
    temporary = destination.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)


def dataset(root):
    from benchmarks.training.cell_mass_mnist import load_mnist

    data, receipt = load_mnist(root)
    x, y = data["train_x"].float().div_(255), data["train_y"]
    permutation = torch.randperm(60_000, generator=torch.Generator().manual_seed(19))
    fit, validation = permutation.split((TRAIN_COUNT, 60_000 - TRAIN_COUNT))
    sample = x[fit]
    mean, std = sample.mean().item(), sample.std(unbiased=False).item()
    x.sub_(mean).div_(std)
    test_x = data["t10k_x"].float().div_(255).sub_(mean).div_(std)
    return x, y, fit, validation, test_x, data["t10k_y"], receipt["sha256"]


def rank(arms, order, epoch):
    return sorted(
        order,
        key=lambda name: (arms[name]["epochs"][epoch - 1]["validation"]["ce"], name),
    )


def run(args):
    torch.set_num_threads(6)
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090"
    ):
        raise RuntimeError("requires NVIDIA GeForce RTX 4090")
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    x, y, fit, val, test_x, test_y, data_sha = dataset(args.data)
    reference = json.loads(args.reference.read_text())
    manifest = reference["protocol"]
    assert data_sha == manifest["data_sha256"]
    assert digest(val) == manifest["validation_indices_sha256"]
    assert sha(args.warm_function) == manifest["warm_decoder_sha256"]
    assert manifest["device"] == torch.cuda.get_device_name()
    assert (
        manifest["torch"] == torch.__version__
        and manifest["cuda"] == torch.version.cuda
    )
    cpu_orders = {seed: epoch_orders(fit, seed) for seed in (SCREEN_SEED, CONFIRM_SEED)}
    assert [digest(row) for row in cpu_orders[SCREEN_SEED]] == manifest[
        "epoch_order_sha256"
    ]
    warm = torch.load(args.warm_function, weights_only=True, map_location="cpu")
    decoder = PhysicalAtomDecoder(warm["amplitude_scale"], period=warm["period"])
    decoder.load_state_dict(warm["decoder"])

    def candidate(name, seed):
        coupling = CANDIDATES[name]
        return Classifier(
            decoder,
            extra_latent=coupling is not None,
            seed=seed,
            fifth_coupling=coupling or DEFAULT_COUPLING,
        )

    checks = cpu_orders[SCREEN_SEED][0][:16]
    cpu = candidate("amp_radius_both_centers", SCREEN_SEED)
    oracle = cuda_oracle(
        cpu,
        candidate("amp_radius_both_centers", SCREEN_SEED).cuda(),
        x[checks],
        y[checks],
    )
    print("CUDA_ORACLE", oracle, flush=True)
    val_x, val_y = x[val].cuda(), y[val].cuda()
    x, y = x.cuda(), y.cuda()
    test_x, test_y = test_x.cuda(), test_y.cuda()
    orders = {seed: [row.cuda() for row in rows] for seed, rows in cpu_orders.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / "result.json"
    protocol = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "execution": manifest["execution"],
        "torchcst_commit": manifest["torchcst_commit"],
        "reference_sha256": sha(args.reference),
        "warm_decoder_sha256": sha(args.warm_function),
        "data_sha256": data_sha,
        "epoch_order_sha256": {
            str(seed): [digest(row) for row in rows]
            for seed, rows in cpu_orders.items()
        },
        "validation_indices_sha256": digest(val),
        "couplings": {
            name: list(c) if c is not None else None for name, c in CANDIDATES.items()
        },
        "source_sha256": {
            file.name: sha(file)
            for file in (
                Path(__file__),
                Path(__file__).with_name("mnist_fixed_decoder_latents.py"),
                Path(__file__).with_name("physical_atom_decoder.py"),
            )
        },
        "screen_seed": SCREEN_SEED,
        "confirmation_seed": CONFIRM_SEED,
        "screen_epochs": SCREEN_EPOCHS,
        "final_epochs": FINAL_EPOCHS,
        "promoted_count": PROMOTED_COUNT,
        "batch_size": BATCH_SIZE,
    }
    report = (
        json.loads(result_file.read_text())
        if result_file.is_file()
        else {
            "state": "running",
            "protocol": protocol,
            "oracle": oracle,
            "preparation_seconds": time.perf_counter() - started,
            "arms": {str(seed): {} for seed in orders},
        }
    )
    if report["protocol"] != protocol:
        raise ValueError("existing output belongs to a different protocol")
    if report["state"] == "completed":
        return
    write_report(result_file, report)

    def geometry(model):
        return [digest(layer.physical(model.decoder)) for layer in model.layers]

    initial = {seed: geometry(candidate("four", seed).cuda()) for seed in orders}
    first_images = {seed: x[indices[0][:16]] for seed, indices in orders.items()}
    with torch.no_grad():
        first_logits = {
            seed: candidate("four", seed).cuda()(batch)
            for seed, batch in first_images.items()
        }

    def fit_arm(seed, name, epochs):
        arms = report["arms"][str(seed)]
        previous = arms.get(name, {})
        if len(previous.get("epochs", [])) >= epochs and previous.get("complete"):
            return
        arm_started = time.perf_counter()
        model = candidate(name, seed).cuda()
        if geometry(model) != initial[seed]:
            raise ValueError(f"initial physical geometry differs for {seed}/{name}")
        with torch.no_grad():
            torch.testing.assert_close(
                model(first_images[seed]), first_logits[seed], rtol=2e-6, atol=2e-7
            )
        directory = args.output / str(seed) / name
        directory.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(directory.glob("epoch-*.pt"))
        resume = (
            torch.load(checkpoints[-1], weights_only=True, map_location="cpu")
            if checkpoints
            else None
        )
        if resume is not None and len(resume["epochs"]) > epochs:
            raise ValueError(f"unexpected future checkpoint for {seed}/{name}")
        print(
            "START",
            seed,
            name,
            "from epoch",
            len(resume["epochs"]) if resume else 0,
            "to",
            epochs,
            flush=True,
        )

        def after_epoch(state):
            save_checkpoint(directory, state)
            arms[name] = {
                "complete": False,
                "epochs": state["epochs"],
                "fifth_first_step": state["fifth_first_step"],
                "initial_physical_sha256": initial[seed],
            }
            write_report(result_file, report)

        result = train(
            f"{seed}/{name}",
            model,
            x,
            y,
            orders[seed][:epochs],
            val_x,
            val_y,
            resume=resume,
            on_epoch=after_epoch,
        )
        arms[name] = {
            **result,
            "complete": True,
            "initial_physical_sha256": initial[seed],
            "phase_wall_seconds": previous.get("phase_wall_seconds", [])
            + [time.perf_counter() - arm_started],
        }
        if seed == CONFIRM_SEED and epochs == FINAL_EPOCHS:
            arms[name]["held_out_test"] = evaluate(model, test_x, test_y)
        write_report(result_file, report)

    for name in CANDIDATES:
        fit_arm(SCREEN_SEED, name, SCREEN_EPOCHS)
    arms = report["arms"][str(SCREEN_SEED)]
    novel = [name for name in CANDIDATES if name not in CONTROLS]
    screened = rank(arms, novel, SCREEN_EPOCHS)
    finalists = tuple(screened[:PROMOTED_COUNT])
    report["screen_rank"] = rank(arms, CANDIDATES, SCREEN_EPOCHS)
    report["promoted"] = list(finalists)
    write_report(result_file, report)
    print("PROMOTED", finalists, flush=True)

    for name in (*CONTROLS, *finalists):
        fit_arm(SCREEN_SEED, name, FINAL_EPOCHS)
    # Validate the best novel coupling on a fresh model/data-order seed.
    winner = rank(arms, finalists, FINAL_EPOCHS)[0]
    report["confirmation_candidate"] = winner
    write_report(result_file, report)
    print("CONFIRMING", winner, flush=True)
    for name in (*CONTROLS, winner):
        fit_arm(CONFIRM_SEED, name, FINAL_EPOCHS)
    report["state"] = "completed"
    write_report(result_file, report)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--warm-function", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
