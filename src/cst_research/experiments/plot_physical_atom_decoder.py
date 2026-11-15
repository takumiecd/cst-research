"""Plot the learned four-input chart at actual QKV Atom locations."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

if __package__:
    from .physical_atom_decoder import PhysicalAtomDecoder
else:
    from physical_atom_decoder import PhysicalAtomDecoder


def physical_delta(base, sample, period):
    amp = sample[..., 0] - base[..., 0]
    radius = (sample[..., 1] / base[..., 1]).log()
    centers = (sample[..., 2:] - base[..., 2:] + period / 2).remainder(
        period
    ) - period / 2
    return torch.cat((amp.unsqueeze(-1), radius.unsqueeze(-1), centers), dim=-1)


def inspect(snapshot):
    data = torch.load(snapshot, map_location="cpu", weights_only=True)
    decoder = PhysicalAtomDecoder(data["amplitude_scale"], period=data["period"])
    decoder.load_state_dict(data["decoder"], strict=True)
    decoder.eval()
    samples = torch.cat(tuple(data["latents"].values())).requires_grad_(True)
    physical = decoder(samples)
    units = samples.new_tensor(
        [data["amplitude_scale"], 1.0, data["period"] / 32, data["period"] / 32]
    )
    rows = []
    for i in range(4):
        output = physical[:, i]
        if i == 1:
            output = output.log()
        jacobian = torch.autograd.grad(output.sum(), samples, retain_graph=i != 3)[0]
        rows.append(jacobian * units / units[i])
    jacobian = torch.stack(rows, dim=1)
    with torch.no_grad():
        base = decoder.base(samples)
        learned = physical_delta(base, physical, decoder.period) / units
        summary = {
            "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "step": data["step"],
            "development_ce": data["development"],
            "atoms": len(samples),
            "fields": decoder.fields,
            "units": units.tolist(),
            "absolute_correction_in_units_median": learned.abs()
            .median(dim=0)
            .values.tolist(),
            "absolute_jacobian_median": jacobian.abs().median(dim=0).values.tolist(),
            "signed_jacobian_median": jacobian.median(dim=0).values.tolist(),
            "absolute_jacobian_p90": torch.quantile(
                jacobian.abs(), 0.9, dim=0
            ).tolist(),
        }
    return decoder, samples.detach(), summary


@torch.no_grad()
def plot(decoder, latents, summary, output):
    import matplotlib.pyplot as plt

    units = latents.new_tensor(summary["units"])
    t = torch.linspace(-1, 1, 65, dtype=latents.dtype)
    ref = decoder(latents)[None]
    base = decoder.base(latents)[None]
    figure, axes = plt.subplots(4, 4, figsize=(12.5, 10), sharex=True)
    figure.subplots_adjust(
        left=0.11, right=0.98, top=0.86, bottom=0.12, wspace=0.31, hspace=0.27
    )
    figure.suptitle(
        f"Shared QKV Atom decoder after {summary['step']:,} updates",
        x=0.11,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.11,
        0.922,
        f"TinyStories development CE {summary['development_ce']:.4f}"
        f"  ·  {summary['atoms']:,} sampled Atom latents",
        fontsize=11,
        color="#4e5966",
    )
    captions = ("Amplitude", "Log radius", "Output center", "Input center")
    for j in range(4):
        points = latents[None].repeat(len(t), 1, 1)
        points[:, :, j] += t[:, None] * units[j]
        prediction = physical_delta(ref, decoder(points), decoder.period) / units
        reference = physical_delta(base, decoder.base(points), decoder.period) / units
        low, median, high = torch.quantile(
            prediction, latents.new_tensor([0.1, 0.5, 0.9]), dim=1
        )
        unlearned = reference.median(dim=1).values
        for i in range(4):
            ax = axes[i, j]
            ax.fill_between(t, low[:, i], high[:, i], color="#2a8175", alpha=0.20)
            ax.plot(t, median[:, i], color="#1b695f", lw=1.7)
            ax.plot(t, unlearned[:, i], color="#746a7c", ls="--", lw=1.0)
            ax.axhline(0, color="#b6bcc3", lw=0.65, zorder=0)
            ax.set_xlim(-1, 1)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(captions[j] + " latent", fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel("Δ " + captions[i], fontsize=10)
    figure.text(
        0.11,
        0.047,
        "One input varies at a time. Curves show the median and 10–90% range in normalized units. "
        "Dashed lines show the initial independent chart.",
        fontsize=9,
        color="#4e5966",
    )
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    decoder, latents, summary = inspect(args.snapshot)
    plot(decoder, latents, summary, args.output.with_suffix(".png"))
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
