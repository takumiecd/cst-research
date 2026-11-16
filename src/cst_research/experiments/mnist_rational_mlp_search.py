"""Search fixed MLP corrections to the rational Atom chart with paired rollouts."""

import argparse
import json
import math
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from statistics import mean

import torch
from torch import nn

from .mnist_fifth_latent_search import (
    dataset,
    epoch_orders,
    save_checkpoint,
    write_report,
)
from .mnist_fixed_decoder_latents import (
    DEFAULT_COUPLING,
    cuda_oracle,
    digest,
    evaluate,
    sha,
    train,
)
from .mnist_rational_lower_comparison import (
    RationalLayer,
    dense_head_pair,
    historical_formula,
    matched_models,
    project_rational_atoms,
)
from .physical_atom_decoder import PhysicalAtomDecoder

SEEDS = (17, 29)
DIRECTIONS = 6
HIDDEN = 16
HEAD_RMS = 0.15
DIRECTION_SEED = 6517
FEATURE_SEED = 4129
SCREEN_EPOCHS = 2
FINAL_EPOCHS = 5
FINALISTS = 2


class ResidualMLP(nn.Module):
    """Add shared corrections to (amplitude, log radius, both centers)."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(FEATURE_SEED)
            self.features = nn.Sequential(
                nn.Linear(6, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU()
            )
            self.head = nn.Linear(HIDDEN, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.requires_grad_(False)

    def forward(self, raw, base, bounds, max_radius):
        amplitude, fraction, out, inp = raw.unbind(1)
        out_angle, in_angle = out * (math.tau / bounds[0]), inp * (math.tau / bounds[1])
        features = torch.stack(
            (
                torch.asinh(amplitude / 0.3),
                fraction,
                out_angle.sin(),
                out_angle.cos(),
                in_angle.sin(),
                in_angle.cos(),
            ),
            dim=1,
        )
        da, dr, do, di = (
            fraction[:, None] * self.head(self.features(features))
        ).unbind(1)
        return torch.stack(
            (
                (base[:, 0] + 0.3 * da).clamp(-32, 32),
                (base[:, 1] * dr.exp()).clamp(0.5, max_radius),
                base[:, 2] + 2 * do,
                base[:, 3] + 2 * di,
            ),
            dim=1,
        )


class ResidualLayer(RationalLayer):
    def physical(self, decoder):
        return decoder(
            self.cst.atoms.p, super().physical(decoder), self.bounds, self.max_radius
        )


def add_decoder(analytic, formula):
    model = deepcopy(analytic)
    layer = ResidualLayer(784, 128, False, DEFAULT_COUPLING, formula=formula)
    layer.load_state_dict(analytic.layers[0].state_dict())
    model.layers[0] = layer
    model.decoder = ResidualMLP()
    return model


def paired_directions(size, *, seed=DIRECTION_SEED, anchor=None):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((size, DIRECTIONS), generator=generator, dtype=torch.float64)
    if anchor is not None:
        noise[:, 0].copy_(anchor)
    unit, _ = torch.linalg.qr(noise)
    if anchor is not None and float(unit[:, 0] @ anchor) < 0:
        unit[:, 0].neg_()
    return unit.T.to(torch.float32).contiguous()


def screening_gaps(names, scores):
    return {
        name: {str(seed): ce[name] - ce["origin"] for seed, ce in scores.items()}
        for name in names
        if name != "origin"
    }


def rank_by_worst_seed(gaps):
    return sorted(
        gaps,
        key=lambda name: (max(gaps[name].values()), mean(gaps[name].values()), name),
    )


@torch.no_grad()
def set_head(model, vector):
    head = model.decoder.head
    weight, bias = vector.split((head.weight.numel(), head.bias.numel()))
    head.weight.copy_(weight.reshape_as(head.weight))
    head.bias.copy_(bias)


def reward_update(directions, scores, scale):
    signed_gains = torch.tensor(
        [
            scores[f"dir{i + 1}_minus"] - scores[f"dir{i + 1}_plus"]
            for i in range(DIRECTIONS)
        ],
        dtype=directions.dtype,
    )
    update = signed_gains @ directions
    norm = update.norm()
    return update if float(norm) == 0 else update * (0.5 * scale / norm)


def run(args):
    torch.set_num_threads(6)
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090"
    ):
        raise RuntimeError("requires NVIDIA GeForce RTX 4090")
    torch.backends.cuda.matmul.allow_tf32 = False
    began = time.perf_counter()
    baseline = json.loads(args.reference.read_text())
    previous = json.loads(args.orders_reference.read_text())
    old = json.loads(args.history.read_text())
    base_protocol = baseline["protocol"]
    order_protocol = previous["protocol"]
    assert baseline["state"] == previous["state"] == "completed"
    assert base_protocol["mnist_reference_sha256"] == sha(args.orders_reference)
    for name, path in (
        ("historical_receipt_sha256", args.history),
        ("historical_envelope_source_sha256", args.envelope_source),
        ("warm_decoder_sha256", args.warm_function),
    ):
        assert base_protocol[name] == sha(path)
    assert (base_protocol["torch"], base_protocol["cuda"]) == (
        torch.__version__,
        torch.version.cuda,
    )
    formula = historical_formula(args.envelope_source, old)
    x, y, fit, val, test_x, test_y, data_sha = dataset(args.data)
    assert data_sha == order_protocol["data_sha256"]
    assert digest(val) == order_protocol["validation_indices_sha256"]
    cpu_orders = {seed: epoch_orders(fit, seed) for seed in SEEDS}
    assert all(
        [digest(order) for order in orders]
        == order_protocol["epoch_order_sha256"][str(seed)]
        for seed, orders in cpu_orders.items()
    )
    warm = torch.load(args.warm_function, weights_only=True, map_location="cpu")
    decoder = PhysicalAtomDecoder(warm["amplitude_scale"], period=warm["period"])
    decoder.load_state_dict(warm["decoder"])
    width = 4 * (HIDDEN + 1)
    prior_round = None
    head_rms, direction_seed, finalists = HEAD_RMS, DIRECTION_SEED, FINALISTS
    screen_seeds = SEEDS[:1]
    anchor = None
    if args.previous_round:
        prior_round = json.loads(args.previous_round.read_text())
        prior = prior_round["protocol"]
        assert prior_round["state"] == "completed"
        assert prior["reference_sha256"] == sha(args.reference)
        assert prior["orders_reference_sha256"] == sha(args.orders_reference)
        assert prior["feature_seed"] == FEATURE_SEED
        head_rms = prior["head_rms"] / 2
        direction_seed, finalists = 8329, 3
        screen_seeds = SEEDS
        anchor = torch.tensor(prior["candidates"]["dir6_plus"], dtype=torch.float64)
    directions = paired_directions(width, seed=direction_seed, anchor=anchor)
    scale = head_rms * math.sqrt(width)
    candidates = {"origin": torch.zeros(width)}
    for i, direction in enumerate(directions, 1):
        candidates[f"dir{i}_plus"] = scale * direction
        candidates[f"dir{i}_minus"] = -scale * direction

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "result.json"
    protocol = {
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
        "torchcst_commit": base_protocol["torchcst_commit"],
        "execution": base_protocol["execution"],
        "data_sha256": data_sha,
        "validation_indices_sha256": digest(val),
        "epoch_order_sha256": {
            str(seed): [digest(order) for order in orders]
            for seed, orders in cpu_orders.items()
        },
        "screen_seeds": list(screen_seeds),
        "final_seeds": list(SEEDS),
        "prior_round_sha256": sha(args.previous_round) if prior_round else None,
        "first_direction": "previous dir6_plus" if prior_round else "random",
        "finalist_count": finalists,
        "screen_epochs": SCREEN_EPOCHS,
        "final_epochs": FINAL_EPOCHS,
        "candidates": {name: vector.tolist() for name, vector in candidates.items()},
        "direction_seed": direction_seed,
        "feature_seed": FEATURE_SEED,
        "architecture": [6, HIDDEN, HIDDEN, 4],
        "head_rms": head_rms,
        "correction": "multiply all four shared corrections by the Atom q",
        "promotion": "rank by worst paired CE gap at epoch two, then mean gap; include reward_update",
    }
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
    cpu_x, cpu_y = x, y
    val_x, val_y = x[val].cuda(), y[val].cuda()
    x, y = x.cuda(), y.cuda()
    test_x, test_y = test_x.cuda(), test_y.cuda()
    orders = {s: [o.cuda() for o in batches] for s, batches in cpu_orders.items()}
    write_report(target, report)

    def prepare(seed):
        base, _ = matched_models(decoder, formula, seed)
        analytic = dense_head_pair(base["rational_q"], seed)["cst_dense"]
        common = add_decoder(analytic, formula)
        if seed == SEEDS[0]:
            probe = cpu_orders[seed][0][:16]
            oracle = deepcopy(common)
            set_head(oracle, candidates["dir1_plus"])
            with torch.no_grad():
                oracle.layers[0].cst.atoms.p[:, 1].fill_(0.35)
            report["cuda_oracle"] = cuda_oracle(
                oracle, deepcopy(oracle).cuda(), cpu_x[probe], cpu_y[probe]
            )
        common.cuda()
        desired = baseline["seeds"][str(seed)]["initial_first_layer_sha256"]
        assert digest(common.layers[0].physical(common.decoder)) == desired
        assert not common.layers[0].cst.atoms.p[:, 1].count_nonzero()
        initial_batch = x[orders[seed][0][:16]]
        with torch.no_grad():
            initial_logits = common(initial_batch)
        record = report["seeds"].setdefault(str(seed), {"arms": {}})
        record["initial_first_layer_sha256"] = desired
        record["trainable_parameters"] = sum(
            p.numel() for p in common.parameters() if p.requires_grad
        )
        record["frozen_mlp_parameters"] = sum(
            p.numel() for p in common.decoder.parameters()
        )
        record["baseline"] = {
            name: baseline["seeds"][str(seed)]["arms"][name]
            for name in ("cst_dense", "dense")
        }
        write_report(target, report)
        return common, desired, initial_batch, initial_logits, record["arms"]

    def fit_arm(seed, name, budget, prepared):
        common, expected, images, logits, arms = prepared
        previous = arms.get(name, {})
        if previous.get("complete") and len(previous["epochs"]) >= budget:
            return
        started = time.perf_counter()
        model = deepcopy(common)
        set_head(model, candidates[name])
        with torch.no_grad():
            assert digest(model.layers[0].physical(model.decoder)) == expected
            torch.testing.assert_close(model(images), logits, rtol=0, atol=0)
        directory = args.output / str(seed) / name
        directory.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(directory.glob("epoch-*.pt"))
        state = (
            torch.load(checkpoints[-1], weights_only=True, map_location="cpu")
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
            budget,
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
            x,
            y,
            orders[seed][:budget],
            val_x,
            val_y,
            resume=state,
            on_epoch=after_epoch,
            after_update=partial(project_rational_atoms, model),
        )
        arms[name] = {
            **result,
            "complete": True,
            "head_sha256": digest(candidates[name]),
            "phase_wall_seconds": previous.get("phase_wall_seconds", [])
            + [time.perf_counter() - started],
        }
        if budget == FINAL_EPOCHS:
            arms[name]["test"] = evaluate(model, test_x, test_y)
        write_report(target, report)

    prepared = {seed: prepare(seed) for seed in screen_seeds}
    for name in candidates:
        for seed in screen_seeds:
            fit_arm(seed, name, SCREEN_EPOCHS, prepared[seed])
            if name == "origin":
                reference = baseline["seeds"][str(seed)]["arms"]["cst_dense"]
                origin = prepared[seed][-1][name]
                assert all(
                    abs(a["validation"]["ce"] - b["validation"]["ce"]) <= 1e-7
                    for a, b in zip(
                        origin["epochs"],
                        reference["epochs"][:SCREEN_EPOCHS],
                        strict=True,
                    )
                )
    scores = {
        seed: {
            name: prepared[seed][-1][name]["epochs"][SCREEN_EPOCHS - 1]["validation"][
                "ce"
            ]
            for name in candidates
        }
        for seed in screen_seeds
    }
    average = {name: mean(ce[name] for ce in scores.values()) for name in candidates}
    update = reward_update(directions, average, scale)
    candidates["reward_update"] = update
    report["reward_update"] = update.tolist()
    for seed in screen_seeds:
        fit_arm(seed, "reward_update", SCREEN_EPOCHS, prepared[seed])
        scores[seed]["reward_update"] = prepared[seed][-1]["reward_update"]["epochs"][
            SCREEN_EPOCHS - 1
        ]["validation"]["ce"]
    report["screen_ce"] = {str(seed): ce for seed, ce in scores.items()}
    gaps = screening_gaps(candidates, scores)
    ranked = rank_by_worst_seed(gaps)
    promoted = ranked[:finalists]
    report["screen_ce_gap_to_origin"] = gaps
    report["screen_rank"] = ranked
    report["promoted"] = promoted
    write_report(target, report)
    print("PROMOTED", promoted, flush=True)
    for seed in SEEDS:
        if seed not in prepared:
            prepared[seed] = prepare(seed)
        for name in promoted:
            fit_arm(seed, name, FINAL_EPOCHS, prepared[seed])

    report["final_validation_ce_gap_to_rational"] = {
        name: {
            str(seed): prepared[seed][-1][name]["validation"]["ce"]
            - baseline["seeds"][str(seed)]["arms"]["cst_dense"]["validation"]["ce"]
            for seed in SEEDS
        }
        for name in promoted
    }
    report["state"] = "completed"
    report["wall_seconds"] = time.perf_counter() - began
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
    parser.add_argument(
        "--previous-round",
        type=Path,
        help="Screen all candidates on both seeds at half the prior perturbation scale",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
