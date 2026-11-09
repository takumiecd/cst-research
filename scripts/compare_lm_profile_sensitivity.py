"""Paired CPU degree/connection VJPs on retained original TRAIN contexts."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels
from torchcst._backends.torch.parameterizations.radius_envelope import (
    physical_parameters,
)
from torchcst.kernels.spec import RadiusConnectionSpec

from cst_research.experiments.ffn_run import sha


def reference(ni, no, modes, connected):
    kernel = kernels.product(
        kernels.cell_mass(ChebyshevSpec(modes=modes)),
        coordinates=kernels.physical(radius_init=(0.5, 50.0), radius_floor=0.5),
    )
    if connected:
        kernel = replace(
            kernel,
            revision=9,
            profiles=tuple(
                replace(p, connection=RadiusConnectionSpec()) for p in kernel.profiles
            ),
        )
    return CSTLinear(
        chart_presets.regular_grid(no, spacing=1.0, geometry="flat_torus"),
        chart_presets.regular_grid(ni, spacing=1.0, geometry="flat_torus"),
        atoms=1,
        kernel=kernel,
        dtype=torch.float64,
        backend="factored",
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--previous-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    old = a.previous_run
    geometry_path = old / "geometry/geometry.json"
    previous = json.loads(geometry_path.read_text())
    hashes = json.loads((old / "analysis/analysis.json").read_text())["tensor_hashes"]
    assert previous["state"] == "passed"
    result = {
        "state": "running",
        "geometry_sha256": sha(geometry_path),
        "scope": "same retained raw Atom rows, original TRAIN X and task cotangent; CPU FP64 factors after FP32 envelope decode; degree/connection paired local VJP, not retrained M8 causal attribution",
        "sampling": "all-cst and pair12, all selected sites, first up to2 witnesses per preexisting geometry category",
        "records": [],
        "input_hashes": {},
    }
    cache = {}
    for record in previous["records"]:
        arm, site = record["arm"], record["site"]
        if arm not in ("all-cst", "pair12"):
            continue
        if arm not in cache:
            path = old / arm / "signals-4882.pt"
            assert sha(path) == hashes[f"{arm}/signals-4882.pt"]
            result["input_hashes"][arm] = sha(path)
            cache[arm] = torch.load(path, map_location="cpu", weights_only=True)
        context = cache[arm][site]
        x = context["x"].reshape(-1, context["x"].shape[-1]).double()
        dy = context["dy"].reshape(-1, context["dy"].shape[-1]).double()
        for sample in record["samples"]:
            raw = torch.tensor(
                sample["raw_rows"][:2], dtype=torch.float32, requires_grad=True
            )
            cells = []
            for modes in (8, 16):
                for connected in (False, True):
                    ref = reference(x.shape[-1], dy.shape[-1], modes, connected)
                    physical = physical_parameters(
                        raw,
                        radius_min=0.5,
                        radius_max=50.0,
                        tau=0.003,
                        amplitude_max=32.0,
                    ).double()
                    vi, vo = ref.operator.factors(physical)
                    y = (x @ vi) @ vo.T
                    grad = torch.autograd.grad(y, raw, dy)[0].double()
                    assert torch.isfinite(grad).all()
                    cell = {
                        "modes": modes,
                        "connected": connected,
                        "raw_gradients": grad.tolist(),
                        "task_dot": float((y.detach() * dy).sum()),
                    }
                    if modes == 16 and connected:
                        expected = torch.tensor(
                            sample["FP64_factor_task_raw_gradients"][:2],
                            dtype=torch.float64,
                        )
                        torch.testing.assert_close(
                            grad, expected, atol=1e-12, rtol=1e-5
                        )
                        cell["reproduces_previous_vjp"] = True
                    cells.append(cell)
            result["records"].append(
                {
                    "arm": arm,
                    "site": site,
                    "kind": sample["kind"],
                    "atom_ids": sample["atom_ids"][:2],
                    "raw_rows": sample["raw_rows"][:2],
                    "cells": cells,
                }
            )
        (a.output / "profile-sensitivity.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print("PROFILE-SENSITIVITY", arm, site, flush=True)
    assert len({(r["arm"], r["site"]) for r in result["records"]}) == 12
    result["state"] = "passed"
    (a.output / "profile-sensitivity.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
