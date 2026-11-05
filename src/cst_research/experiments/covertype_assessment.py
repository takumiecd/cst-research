"""Explicit heldout evaluation only after all nine selected checkpoints seal."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import covertype_transfer as cell
from .covertype_campaign import require, validate_learning, verify_files
from .covertype_data import HELDOUT_SHA, MANIFEST_SHA, array_record
from .higgs_dst_comparison import file_sha, read, write
from .higgs_dst_comparison_driver import safe_path


def audit_freeze(state, freeze_file, expected_sha, identity):
    require(file_sha(freeze_file) == expected_sha, "explicit sealed freeze SHA differs")
    freeze = read(freeze_file)
    require(
        freeze["identity"] == identity and freeze["heldout_evaluated"] is False,
        "freeze identity/heldout scope differs",
    )
    selection_file = state / "selected-configurations.json"
    require(
        file_sha(selection_file) == freeze["selection_sha256"],
        "selected configurations changed",
    )
    selected = read(selection_file)
    require(
        selected["identity"] == identity
        and selected["heldout_evaluated"] is False
        and len(selected["selected"]) == 3
        and {r["method"] for r in selected["selected"]} == set(cell.METHODS),
        "selected family incomplete",
    )
    rows = freeze["models"]
    require(
        len(rows) == 9
        and {(r["method"], r["seed"]) for r in rows}
        == {(m, s) for m in cell.METHODS for s in cell.SEEDS},
        "nine models must freeze globally before heldout access",
    )
    for row in rows:
        relative = Path(row["directory"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "unsafe sealed directory",
        )
        directory = safe_path(state, row["directory"])
        completion = directory / "cell-complete.json"
        require(
            file_sha(completion) == row["completion_sha256"], "completion proof changed"
        )
        record = read(completion)
        require(record["files"] == row["files"], "row inventory changed")
        verify_files(directory, record)
        require(row["epochs"] == 10, "all selected models need common10epoch horizon")
        config = next(r for r in selected["selected"] if r["method"] == row["method"])
        require(
            row["lr"] == config["lr"],
            "model LR differs from frozen development selection",
        )
        learning = read(directory / "learning.json")
        chosen = validate_learning(
            learning, row["method"], row["seed"], row["lr"], 10, identity
        )
        require(
            chosen["epoch"] == row["best_epoch"]
            and chosen["development"] == row["development"]
            and learning["best"] == read(directory / "best-dev.json")
            and learning["best"]["checkpoint_sha256"]
            == file_sha(directory / "best-dev.pt"),
            "selected checkpoint proof differs",
        )
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cell.add_common(parser)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--frozen-index-sha256", required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "fresh assessment output required")
    require(
        not args.output.resolve().is_relative_to(
            Path(cell.__file__).resolve().parents[3]
        ),
        "assessment output must be outside immutable source",
    )
    identity = cell.admit_sources(args)
    # NO heldout path is opened or hashed before this complete global raw gate.
    freeze = audit_freeze(
        args.state, args.frozen_index, args.frozen_index_sha256, identity
    )
    import numpy as np
    import torch
    from benchmarks.training.higgs_confirmation import _configure

    from .covertype_data import specification

    # All nine raw checkpoint configurations/moments/clocks pass before any
    # heldout bytes are opened, hashed, decompressed or sent to an evaluator.
    for row in freeze["models"]:
        raw = torch.load(
            args.state / row["directory"] / "best-dev.pt",
            map_location="cpu",
            weights_only=True,
        )
        cell.validate_checkpoint(raw, row, identity)
    device = torch.device(args.device)
    require(device.type == "cuda", "explicit admitted CUDA evaluator required")
    require(cell.runtime(device) == freeze["runtime"], "evaluation runtime differs")
    manifest = read(args.data_manifest)
    require(
        file_sha(args.data_manifest) == args.data_manifest_sha256 == MANIFEST_SHA
        and manifest["specification"] == specification(),
        "data manifest differs",
    )
    artifact = manifest["artifacts"]["future-heldout.npz"]
    require(
        file_sha(args.heldout) == artifact["sha256"] == HELDOUT_SHA,
        "heldout artifact differs",
    )
    with np.load(args.heldout, allow_pickle=False) as source:
        require(
            set(source.files) == {"x_heldout", "y_heldout", "row_ids_heldout"},
            "heldout payload inventory differs",
        )
        values = {k: source[k].copy() for k in source.files}
    require(
        all(array_record(v) == artifact["arrays"][k] for k, v in values.items()),
        "heldout array bytes differ",
    )
    require(
        values["x_heldout"].shape == (50000, 54)
        and values["y_heldout"].shape == (50000,),
        "heldout cohort differs",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    _configure()
    x = torch.from_numpy(values["x_heldout"]).to(device)
    y = torch.from_numpy(values["y_heldout"]).to(device)
    results = []
    for row in freeze["models"]:
        directory = args.state / row["directory"]
        model = cell.make_model(row["method"], row["seed"]).to(device)
        checkpoint = torch.load(
            directory / "best-dev.pt", map_location=device, weights_only=True
        )
        require(
            all(checkpoint["configuration"].get(k) == v for k, v in identity.items()),
            "checkpoint identity differs",
        )
        require(
            checkpoint["configuration"]["id"] == row["method"]
            and checkpoint["configuration"]["seed"] == row["seed"]
            and checkpoint["configuration"]["lr"] == row["lr"]
            and checkpoint["epoch"] == row["best_epoch"],
            "checkpoint/configuration row differs",
        )
        model.load_state_dict(checkpoint["model"])
        metrics, logits, labels, seconds = cell.evaluate(model, x, y)
        filename = f"{row['method']}-seed{row['seed']}.npz"
        np.savez_compressed(
            args.output / filename,
            logits=logits,
            labels=labels,
            row_ids=values["row_ids_heldout"],
        )
        results.append(
            {
                "method": row["method"],
                "seed": row["seed"],
                "metrics": metrics,
                "evaluation_seconds": seconds,
                "predictions_file": filename,
                "predictions_sha256": file_sha(args.output / filename),
            }
        )
        del model
    write(
        args.output / "assessment.json",
        {
            "identity": identity,
            "freeze_sha256": args.frozen_index_sha256,
            "heldout_sha256": artifact["sha256"],
            "models": results,
            "exploratory_only": True,
            "formal_noninferiority": False,
            "geographic_generalization": False,
        },
    )


if __name__ == "__main__":
    main()
