"""Compare physical Atom charts at fixed, shared latent locations."""

import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .plot_physical_atom_decoder import inspect, physical_delta
else:
    from plot_physical_atom_decoder import inspect, physical_delta


@torch.no_grad()
def plot(reference, trained, samples, units, output, terminal=False):
    import matplotlib.pyplot as plt

    t = torch.linspace(-1, 1, 65, dtype=samples.dtype)
    a0, b0 = reference(samples), trained(samples)
    fig, axes = plt.subplots(4, 4, figsize=(12.5, 10), sharex=True)
    fig.subplots_adjust(
        left=0.106, right=0.98, top=0.87, bottom=0.125, wspace=0.39, hspace=0.37
    )
    fig.suptitle(
        (
            "Terminal-CE reward versus ordinary Adam"
            if terminal
            else "How 32-update reward changes the shared QKV Atom decoder"
        ),
        x=0.106,
        y=0.96,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.106,
        0.915,
        (
            "Terminal-CE reward (purple) and ordinary Adam (green)"
            if terminal
            else "ES response minus ordinary Adam response at the same latent inputs"
        ),
        fontsize=11,
        color="#465262",
    )
    captions = ("Amplitude", "Log radius", "Output center", "Input center")
    changes = []
    for j in range(4):
        points = samples[None].expand(len(t), -1, -1).clone()
        points[:, :, j] += t[:, None] * units[j]
        a = physical_delta(a0, reference(points), reference.period) / units
        b = physical_delta(b0, trained(points), trained.period) / units
        difference = b - a
        low, mid, high = torch.quantile(
            difference, samples.new_tensor([0.1, 0.5, 0.9]), dim=1
        )
        changes.append(mid[-1].tolist())
        if terminal:
            quantiles = samples.new_tensor([0.1, 0.5, 0.9])
            a_quantiles = torch.quantile(a, quantiles, dim=1)
            b_quantiles = torch.quantile(b, quantiles, dim=1)
        for i in range(4):
            ax = axes[i, j]
            if terminal:
                for values, color in (
                    (a_quantiles, "#277568"),
                    (b_quantiles, "#554187"),
                ):
                    lo, median, hi = values[:, :, i]
                    ax.fill_between(t, lo, hi, color=color, alpha=0.12)
                    ax.plot(t, median, color=color, lw=1.6)
            else:
                ax.fill_between(t, low[:, i], high[:, i], color="#66558a", alpha=0.18)
                ax.plot(t, mid[:, i], color="#554187", lw=1.6)
            ax.axhline(0, color="#9eabb6", ls="--", lw=0.8, zorder=0)
            ax.set_xlim(-1, 1)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(captions[j] + " latent", fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel("Δ " + captions[i], fontsize=9)
    fig.text(
        0.106,
        0.045,
        "Each row shows a physical output. Values are normalized. "
        "Bands span the 10th–90th percentile across the same sampled Atoms.",
        fontsize=9,
        color="#465262",
    )
    fig.savefig(output, dpi=165, facecolor="white")
    plt.close(fig)
    return torch.tensor(changes).T.tolist()


def run(directory, dest, terminal=False):
    report = json.loads((directory / "result.json").read_text())
    if report["state"] != "completed":
        raise ValueError("expected a finished reward experiment")
    ordinary_name, reward_name = (
        ("ordinary_adam", "reward_updated_fixed")
        if terminal
        else ("joint_adam_32", "trained_es")
    )
    original, a, summary_a = inspect(directory / f"function-{ordinary_name}.pt")
    trained, b, summary_b = inspect(directory / f"function-{reward_name}.pt")
    for name, summary in ((ordinary_name, summary_a), (reward_name, summary_b)):
        assert report["function_snapshots"][name] == summary["snapshot_sha256"]
    assert torch.equal(a, b)
    assert summary_a["units"] == summary_b["units"]
    value = plot(
        original,
        trained,
        a,
        a.new_tensor(summary_a["units"]),
        dest.with_suffix(".png"),
        terminal,
    )
    dest.with_suffix(".json").write_text(
        json.dumps(
            {
                "atom_count": len(a),
                "input_units": summary_a["units"],
                "ordinary_adam": summary_a,
                "reward_trained": summary_b,
                "paired_response_difference_at_plus_one_input_unit": value,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.directory, args.output, args.terminal)


if __name__ == "__main__":
    main()
