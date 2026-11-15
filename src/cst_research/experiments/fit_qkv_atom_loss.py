"""Fit immediate loss predictors with whole-batch cross-validation."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

FIELDS = ("amplitude", "radius", "output_center", "input_center")


def physical_rows(raw):
    a, q, out, inp = raw.double().unbind(-1)
    lower = 0.5 + 3.5 / (1 + 3 * (a / 0.3).square())
    radius = ((1 - q) * lower.log() + q * torch.full_like(q, 512.0).log()).exp()
    return torch.stack((a, radius, out, inp), dim=-1).to(raw.dtype)


def differences(before, after):
    direct = after[..., :2] - before[..., :2]
    centers = (after[..., 2:] - before[..., 2:] + 512).remainder(1024) - 512
    return torch.cat((direct, centers), dim=-1)


def features(group):
    """Pool Atom-level values, actual moves, and gradients without using CE."""
    before = group["physical"]
    after = physical_rows(group["proposed_raw"])
    delta = differences(before, after)
    delta[0].zero_()
    gradient = group["physical_gradients"]
    dot = delta * gradient
    radius = before[:, 1]
    units = torch.stack(
        (
            before[:, 0].abs().clamp_min(0.03),
            radius,
            radius.clamp_min(1),
            radius.clamp_min(1),
        ),
        dim=1,
    )
    moves = (delta / units).asinh()
    scaled_grad = (gradient * units * 1e4).asinh()
    values = torch.cat(
        (
            before[:, :1],
            radius.log().unsqueeze(1),
            (before[:, 2:] * (torch.pi / 512)).sin(),
            (before[:, 2:] * (torch.pi / 512)).cos(),
        ),
        dim=1,
    )
    lower = physical_rows(
        torch.stack(
            (before[:, 0], torch.zeros_like(radius), before[:, 2], before[:, 3]), dim=1
        )
    )[:, 1]
    state = torch.cat((values, (radius / lower).log().unsqueeze(1), scaled_grad), dim=1)
    interactions = torch.stack(
        [
            (moves[:, :, i] * moves[:, :, j]).mean(1)
            for i in range(4)
            for j in range(i + 1, 4)
        ],
        dim=1,
    )
    position_alignment = torch.einsum("cni,nj->cij", moves, values).flatten(1) / len(
        before
    )
    pooled = torch.cat(
        (
            moves.mean(1),
            moves.abs().mean(1),
            moves.square().mean(1),
            dot.sum(1),
            dot.abs().sum(1),
            (moves * scaled_grad).mean(1),
            interactions,
            position_alignment,
            state.mean(0).expand(len(after), -1),
            state.std(0, unbiased=False).expand(len(after), -1),
        ),
        dim=1,
    )
    return pooled, dot.sum((1, 2))


def read_part(root, part):
    path = root / part["filename"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != part["sha256"]:
        raise ValueError(f"shard hash mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def prepare(root, manifest):
    out = []
    for part in manifest["parts"]:
        payload = read_part(root, part)
        xx, gg, yy = [], [], []
        for group in payload["groups"]:
            x, dot = features(group)
            xx.append(x)
            gg.append(dot)
            yy.append(torch.tensor([row[1] for row in group["observations"]]))
        out.append(
            {
                "step": part["step"],
                "pair": part["pair"],
                "file": part["filename"],
                "x": torch.stack(xx),
                "dot": torch.stack(gg),
                "y": torch.stack(yy),
                "shortlist": part["shortlist"],
            }
        )
    return out


def fit(rows):
    x = torch.cat([row["x"] for row in rows])
    y = torch.cat([row["y"] for row in rows])
    # Equalize typical label scales across contexts without using held-out labels.
    group_weight = y.square().mean(dim=1).sqrt().clamp_min(0.001).reciprocal()
    feature_mean = x.mean(dim=(0, 1))
    feature_scale = x.std(dim=(0, 1), unbiased=False).clamp_min(1e-8)
    xx = ((x - feature_mean) / feature_scale).clamp(-10, 10)
    xx0 = xx[:, :1]
    target = y * 1000
    weight = (group_weight / group_weight.mean()).unsqueeze(1)
    xdiff = (xx - xx0).flatten(0, 1).double()
    w = weight.expand_as(y).flatten().double()
    matrix = xdiff.T @ (xdiff * w[:, None])
    beta = torch.linalg.solve(
        matrix + torch.eye(matrix.shape[0], dtype=matrix.dtype) * (0.002 * len(w)),
        xdiff.T @ (target.flatten().double() * w),
    ).float()
    nets = []
    for seed in (17, 29, 41):
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(x.shape[-1], 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
        for _ in range(100):
            for ids in torch.randperm(len(x)).split(8):
                prediction = model(xx[ids]).squeeze(-1) - model(xx0[ids]).squeeze(-1)
                loss = (weight[ids] * (prediction - target[ids]).square()).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        nets.append(model.eval())
    return feature_mean, feature_scale, beta, nets


@torch.no_grad()
def predict(context, fitted):
    mean, scale, beta, models = fitted
    xx = ((context["x"] - mean) / scale).clamp(-10, 10)
    cv = torch.stack([model(xx).squeeze(-1) for model in models]).mean(0)
    return {
        "gradient": context["dot"],
        "ridge": (xx - xx[:, :1]) @ beta / 1000,
        "mlp": (cv - cv[:, :1]) / 1000,
    }


def assess(context, scores):
    y = context["y"]
    shape = y.shape
    oracle = float(y.min())
    out = {}
    for name, score in scores.items():
        group, action = divmod(int(score.argmin()), shape[1])
        eligible = [
            (float(score[r["group"], r["index"]]), r) for r in context["shortlist"]
        ]
        best = min(eligible, key=lambda entry: entry[0])[1]
        rank = score.flatten().argsort(stable=True)
        ranked_truth = y.flatten()[rank]
        out[name] = {
            "group": group,
            "action": action,
            "selected_dev_delta": float(y[group, action]),
            "oracle_dev_delta": oracle,
            "dev_regret": float(y[group, action]) - oracle,
            "dev_best_of_top10_delta": float(ranked_truth[:10].min()),
            "saved_shortlist_confirmation_delta": best["delta_confirmation"],
            "saved_shortlist_family": best["family"],
        }
    return out


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest_bytes = (args.dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["state"] != "completed":
        raise ValueError("exploration must be completed")
    data = prepare(args.dataset, manifest)
    assert sum(row["y"].numel() for row in data) == 27648
    out = []
    for pair in sorted({row["pair"] for row in data}):
        fitted = fit([row for row in data if row["pair"] != pair])
        for context in data:
            if context["pair"] == pair:
                out.append(
                    {
                        "step": context["step"],
                        "pair": pair,
                        "file": context["file"],
                        "selectors": assess(context, predict(context, fitted)),
                    }
                )
        print("finished batch fold", pair, flush=True)
    report = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cv": "leave one batch pair out across all three checkpoints",
        "methods": "physical gradient Taylor, ridge, pooled features MLP 32x16 ensemble",
        "fields": FIELDS,
        "contexts": out,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
