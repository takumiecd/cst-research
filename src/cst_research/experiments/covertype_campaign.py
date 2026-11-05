"""Bounded dev-only HPO and nine-model sealing; never reads heldout payloads."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from . import covertype_transfer as cell
from .higgs_dst_comparison import file_sha, read, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank(candidate):
    return candidate["best_development_loss"], candidate["best_epoch"], candidate["lr"]


def select(candidates):
    require(bool(candidates), "no development candidates")
    return min(candidates, key=rank)


def edge_extension(candidates):
    require({r["lr"] for r in candidates} == set(cell.LRS), "base LR grid incomplete")
    winner = select(candidates)["lr"]
    return (
        0.00003
        if winner == min(cell.LRS)
        else 0.003
        if winner == max(cell.LRS)
        else None
    )


def validate_screen(record, method, identity):
    rows = record["candidates"]
    base = [r for r in rows if r["lr"] in cell.LRS]
    extra = edge_extension(base)
    require(
        record["identity"] == identity
        and record["edge_lr"] == extra
        and record["selected"] == select(rows),
        "screen decision differs",
    )
    require(
        len(rows) == 3 + int(extra is not None)
        and {r["lr"] for r in rows}
        == set(cell.LRS) | ({extra} if extra is not None else set()),
        "missing or excess LR extension",
    )
    require(
        all(
            r["method"] == method and r["seed"] == 17 and r["epochs"] == 3 for r in rows
        ),
        "screen cell scope differs",
    )


def verify_files(directory, record):
    from .higgs_dst_comparison_driver import inventory, safe_path

    observed = inventory(directory)
    require(
        set(observed) == set(record["files"]) | {"cell-complete.json"},
        "raw cell inventory changed",
    )
    for name, digest in record["files"].items():
        require(
            file_sha(safe_path(directory, name)) == digest, "raw cell bytes changed"
        )


def validate_learning(learning, method, seed, lr, epochs, identity):
    config = learning["configuration"]
    require(
        all(config.get(k) == v for k, v in identity.items()),
        "cell source/data/protocol identity differs",
    )
    require(
        config["id"] == method
        and config["seed"] == seed
        and config["lr"] == lr
        and config["epochs"] == epochs
        and config["batch_size"] == 4
        and config["TF32"] is False
        and config["mode"] == "eager",
        "cell budget/configuration differs",
    )
    rows = learning["epochs"]
    require(
        len(rows) == epochs and learning["heldout_evaluated"] is False,
        "cell horizon/heldout scope differs",
    )
    require(
        [r["updates"] for r in rows]
        == [cell.UPDATES_PER_EPOCH * e for e in range(1, epochs + 1)],
        "cell update budget differs",
    )
    chosen = min(rows, key=lambda r: (r["development"]["loss"], r["epoch"]))
    require(
        learning["best"]["epoch"] == chosen["epoch"]
        and learning["best"]["development"] == chosen["development"],
        "best checkpoint is not earliest minimum CE",
    )
    require(
        learning["qualification"]["status"] == "passed",
        "actual cell qualification missing",
    )
    return chosen


class Campaign:
    def __init__(self, args):
        self.args = args
        self.state = args.state
        require(
            not self.state.resolve().is_relative_to(
                Path(cell.__file__).resolve().parents[3]
            ),
            "campaign state must be outside immutable source",
        )
        self.identity = cell.admit_sources(args)
        self.state.mkdir(parents=True, exist_ok=True)
        write(self.state / "identity.json", self.identity, immutable=True)
        require(
            not (self.state / "heldout-freeze.json").exists() or args.stage == "seal",
            "sealed family is immutable",
        )

    def saved(self, name, value):
        path = self.state / name
        write(path, value, immutable=True)
        return value

    def cell(self, method, seed, lr, epochs):
        label = f"{method}-seed{seed}-lr{lr:g}-{epochs}ep"
        directory = self.state / "cells" / label
        if directory.exists():
            require(
                (directory / "cell-complete.json").exists(),
                "partial/uncertain cell exists; inspect before explicit recovery",
            )
            retained = read(directory / "cell-complete.json")
            verify_files(directory, retained)
            learning = read(directory / "learning.json")
        else:
            learning = cell.run(
                SimpleNamespace(
                    **(
                        vars(self.args)
                        | {
                            "method": method,
                            "seed": seed,
                            "lr": lr,
                            "epochs": epochs,
                            "output": directory,
                            "expected_runtime": read(self.state / "runtime.json")
                            if (self.state / "runtime.json").exists()
                            else None,
                        }
                    )
                )
            )
            from .higgs_dst_comparison_driver import inventory

            retained = {
                "files": {n: file_sha(p) for n, p in inventory(directory).items()}
            }
            write(directory / "cell-complete.json", retained)
        chosen = validate_learning(learning, method, seed, lr, epochs, self.identity)
        runtime = {
            k: learning[k]
            for k in (
                "torch",
                "cuda",
                "gpu",
                "gpu_profile",
                "python",
                "triton",
                "driver",
                "nvcc",
                "numpy",
            )
        }
        self.saved("runtime.json", runtime)
        best = read(directory / "best-dev.json")
        require(
            best["checkpoint_sha256"] == file_sha(directory / "best-dev.pt")
            and best == learning["best"],
            "best metadata/checkpoint changed",
        )
        return {
            "method": method,
            "seed": seed,
            "lr": lr,
            "epochs": epochs,
            "best_epoch": chosen["epoch"],
            "best_development_loss": chosen["development"]["loss"],
            "development": chosen["development"],
            "directory": directory.relative_to(self.state).as_posix(),
            "files": retained["files"],
            "completion_sha256": file_sha(directory / "cell-complete.json"),
        }

    def screen(self, method):
        candidates = [self.cell(method, 17, lr, 3) for lr in cell.LRS]
        extra = edge_extension(candidates)
        if extra is not None:
            candidates.append(self.cell(method, 17, extra, 3))
        return self.saved(
            f"screen/{method}.json",
            {
                "identity": self.identity,
                "candidates": candidates,
                "selected": select(candidates),
                "edge_lr": extra,
            },
        )

    def retune(self, method):
        screened = read(self.state / "screen" / f"{method}.json")
        validate_screen(screened, method, self.identity)
        require(
            all(
                row == self.cell(method, 17, row["lr"], 3)
                for row in screened["candidates"]
            ),
            "screen raw evidence differs",
        )
        top = sorted(screened["candidates"], key=rank)[:2]
        candidates = [self.cell(method, 17, row["lr"], 10) for row in top]
        return self.saved(
            f"retune/{method}.json",
            {
                "identity": self.identity,
                "screen_sha256": file_sha(self.state / "screen" / f"{method}.json"),
                "fresh_from_initialization": True,
                "candidates": candidates,
                "selected": select(candidates),
            },
        )

    def freeze(self):
        selected = []
        for method in cell.METHODS:
            record = read(self.state / "retune" / f"{method}.json")
            require(
                record["identity"] == self.identity
                and record["selected"] == select(record["candidates"])
                and len(record["candidates"]) == 2,
                "retune incomplete",
            )
            screened = read(self.state / "screen" / f"{method}.json")
            validate_screen(screened, method, self.identity)
            require(
                record["screen_sha256"]
                == file_sha(self.state / "screen" / f"{method}.json")
                and [r["lr"] for r in record["candidates"]]
                == [r["lr"] for r in sorted(screened["candidates"], key=rank)[:2]]
                and record["fresh_from_initialization"] is True,
                "retune is not fresh true top2",
            )
            require(
                all(
                    row == self.cell(method, 17, row["lr"], 10)
                    for row in record["candidates"]
                ),
                "retune raw evidence differs",
            )
            chosen = record["selected"]
            require(chosen["epochs"] == 10, "final horizon differs")
            # Reverify complete raw bytes at selection freeze.
            self.cell(method, 17, chosen["lr"], 10)
            selected.append(chosen)
        return self.saved(
            "selected-configurations.json",
            {
                "identity": self.identity,
                "selected": selected,
                "selection_uses_development_only": True,
                "heldout_evaluated": False,
            },
        )

    def confirm(self, method, seed):
        require(seed in cell.SEEDS, "paired confirmation seed required")
        freeze = read(self.state / "selected-configurations.json")
        require(freeze["identity"] == self.identity, "selection identity differs")
        chosen = next(r for r in freeze["selected"] if r["method"] == method)
        result = self.cell(method, seed, chosen["lr"], 10)
        return self.saved(f"confirm/{method}/{seed}.json", result)

    def seal(self):
        freeze = read(self.state / "selected-configurations.json")
        require(freeze["identity"] == self.identity, "selection identity differs")
        rows = []
        for selected in freeze["selected"]:
            for seed in cell.SEEDS:
                row = read(self.state / "confirm" / selected["method"] / f"{seed}.json")
                require(
                    row == self.cell(selected["method"], seed, selected["lr"], 10),
                    "paired model differs from frozen configuration",
                )
                rows.append(row)
        require(
            len(rows) == 9
            and {(r["method"], r["seed"]) for r in rows}
            == {(m, s) for m in cell.METHODS for s in cell.SEEDS},
            "all nine distinct models required",
        )
        import torch

        for row in rows:
            raw = torch.load(
                self.state / row["directory"] / "best-dev.pt",
                map_location="cpu",
                weights_only=True,
            )
            cell.validate_checkpoint(raw, row, self.identity)
        record = {
            "schema_version": 1,
            "identity": self.identity,
            "runtime": read(self.state / "runtime.json"),
            "selection_sha256": file_sha(self.state / "selected-configurations.json"),
            "models": rows,
            "heldout_evaluated": False,
            "formal_noninferiority": False,
        }
        self.saved("heldout-freeze.json", record)
        self.saved(
            "heldout-freeze-receipt.json",
            {
                "file": "heldout-freeze.json",
                "sha256": file_sha(self.state / "heldout-freeze.json"),
            },
        )
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cell.add_common(parser)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("screen", "retune", "freeze", "confirm", "seal"),
        required=True,
    )
    parser.add_argument("--method", choices=cell.METHODS)
    parser.add_argument("--seed", type=int, choices=cell.SEEDS)
    args = parser.parse_args()
    require(
        (args.method is not None) == (args.stage in ("screen", "retune", "confirm")),
        "method is required only for individual method phases",
    )
    require(
        (args.seed is not None) == (args.stage == "confirm"),
        "paired seed is required only for confirmation",
    )
    campaign = Campaign(args)
    if args.stage in ("screen", "retune", "confirm"):
        require(args.method is not None, "method required")
    if args.stage == "screen":
        campaign.screen(args.method)
    elif args.stage == "retune":
        campaign.retune(args.method)
    elif args.stage == "freeze":
        campaign.freeze()
    elif args.stage == "confirm":
        campaign.confirm(args.method, args.seed)
    else:
        campaign.seal()


if __name__ == "__main__":
    main()
