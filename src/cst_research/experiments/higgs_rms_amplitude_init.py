"""Match initial pre-ReLU Dense RMS by changing raw Atom amplitudes only."""

from __future__ import annotations

import copy
import math

from . import higgs_multilayer_conditioning as contract
from . import higgs_multilayer_conditioning_reference as reference

COHORT = "higgs-rms-amplitude-init-v1"
ARM = "rms-init"
SEARCH_TOLERANCE = 0.005
NATIVE_TOLERANCE = 0.01
MAX_COEFFICIENT = 3.2


def trial_atoms(template, width, coefficient):
    import torch

    if not math.isfinite(coefficient) or not 0 <= coefficient <= MAX_COEFFICIENT:
        raise ValueError("bounded finite initializer coefficient required")
    p = template.detach().clone()
    unbounded = p[:, 0] * (math.sqrt(width) * coefficient / 0.1)
    ratio = unbounded.clamp(-1 + 1e-6, 1 - 1e-6)
    p[:, :2] = torch.stack((ratio, (1 - ratio.square()).sqrt()), dim=-1)
    return p, int((unbounded.abs() >= 1 - 1e-6).sum())


def search(evaluate, target):
    """Sign bracket plus bounded bisection; accept measured relative error only."""
    if not math.isfinite(target) or target <= 0:
        raise ValueError("positive finite Dense RMS target required")
    history = []
    best = None
    low, high = 0.0, None

    def attempt(k):
        nonlocal best
        rms, value = evaluate(k)
        if not math.isfinite(rms) or rms < 0:
            raise ValueError("invalid candidate RMS")
        error = abs(rms / target - 1)
        history.append({"coefficient": k, "RMS": rms, "relative_error": error})
        if best is None or error < best[0]:
            best = (error, k, value)
        return rms, error

    for k in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, MAX_COEFFICIENT):
        rms, error = attempt(k)
        if error <= SEARCH_TOLERANCE:
            return best, history
        if rms >= target:
            high = k
            break
        low = k
    if high is not None:
        for _ in range(32):
            middle = (low + high) / 2
            rms, error = attempt(middle)
            if error <= SEARCH_TOLERANCE:
                return best, history
            if rms < target:
                low = middle
            else:
                high = middle
    raise ValueError("RMS initializer could not meet target: " + str(history))


def make_model(atoms, *, width=1024, hidden=8):
    import torch

    model = contract.make_model("cst-none", width=width, hidden=hidden)
    sites = model._linear_atom_configuration["internal_module_indices"]
    if set(atoms) != set(sites):
        raise ValueError("exact internal initializer inventory required")
    with torch.no_grad():
        for site in sites:
            p = atoms[site]
            if (
                p.dtype != torch.float32
                or p.shape != model[site].atoms.p.shape
                or not bool(torch.isfinite(p).all())
            ):
                raise ValueError("finite FP32 initialized Atoms required")
            model[site].atoms.p.copy_(p)
    return model


def calibrate(features, *, width=1024, hidden=8, progress=None):
    import torch

    if (
        features.device.type != "cpu"
        or features.dtype != torch.float32
        or features.shape != (256, 28)
    ):
        raise ValueError("fixed first256 TRAIN features only")
    dense = contract.make_model("dense-none", width=width, hidden=hidden)
    original = contract.make_model("cst-none", width=width, hidden=hidden)
    dense_ref = copy.deepcopy(dense).double()
    candidate = copy.deepcopy(original).double()
    sites = candidate._linear_atom_configuration["internal_module_indices"]
    with torch.no_grad():
        target_traces = reference.forward(dense_ref, features.double(), collect=True)[1]
        value = features.double()
        atoms, records, reference_outputs = {}, [], {}
        for index, layer in enumerate(candidate):
            if index not in sites:
                value = reference.affine_or_relu(layer, value)
                continue
            template = original[index].atoms.p.detach().clone()
            target = float(target_traces[str(index)].square().mean().sqrt())

            def evaluate(k, *, template=template, layer=layer, layer_input=value):
                p, clipped = trial_atoms(template, width, k)
                layer.atoms.p.copy_(p.double())
                y = reference.affine_or_relu(layer, layer_input, chunk_atoms=256)
                return float(y.square().mean().sqrt()), (p, y, clipped)

            best, history = search(evaluate, target)
            error, k, (p, y, clipped) = best
            layer.atoms.p.copy_(p.double())
            atoms[index] = p
            value = y
            reference_outputs[str(index)] = y.detach().clone()
            records.append(
                {
                    "site": index,
                    "coefficient": k,
                    "normal_std_before_clip": k * math.sqrt(width / len(p)),
                    "Dense_pre_ReLU_RMS": target,
                    "CST_pre_ReLU_RMS": float(y.square().mean().sqrt()),
                    "relative_error": error,
                    "Dense_post_ReLU_RMS": float(
                        target_traces[str(index + 1)].square().mean().sqrt()
                    ),
                    "CST_post_ReLU_RMS": float(y.relu().square().mean().sqrt()),
                    "clipped_atoms": clipped,
                    "atoms": len(p),
                    "search_history": history,
                }
            )
            if progress is not None:
                progress(atoms, records)
    result = make_model(atoms, width=width, hidden=hidden)
    with torch.no_grad():
        for site in sites:
            if not torch.equal(
                original[site].atoms.p[:, 2:], result[site].atoms.p[:, 2:]
            ):
                raise ValueError("calibration changed centers")
    return (
        atoms,
        {
            "records": records,
            "gain": [1.0] * len(sites),
            "normalization": "none",
            "calibration_before_learning_only": True,
            "labels_or_DEV_used": False,
            "search_relative_tolerance": SEARCH_TOLERANCE,
            "native_relative_tolerance": NATIVE_TOLERANCE,
            "no_runtime_output_rescaling": True,
        },
        {
            "features": features.clone(),
            "Dense_FP64_pre_ReLU": {str(i): target_traces[str(i)] for i in sites},
            "CST_FP64_pre_ReLU": reference_outputs,
        },
    )


def native_rms_witness(atoms, features, calibration, device):
    import torch

    candidate = make_model(atoms).to(device)
    dense = contract.make_model("dense-none").to(device)
    result = {}
    with torch.no_grad():
        for label, model in [("Dense", dense), ("CST", candidate)]:
            traces = {str(i): [] for i in (2, 4, 6, 8, 10, 12, 14)}
            for start in range(0, 256, 4):
                values = contract.trace(model, features[start : start + 4].to(device))
                for i, items in traces.items():
                    items.append(values[int(i)].detach().cpu())
            result[label + "_native_pre_ReLU"] = {
                i: torch.cat(v) for i, v in traces.items()
            }
    for row in calibration["records"]:
        key = str(row["site"])
        for label in ("Dense", "CST"):
            rms = float(
                result[label + "_native_pre_ReLU"][key].double().square().mean().sqrt()
            )
            row[label + "_native_pre_ReLU_RMS"] = rms
            if abs(rms / row["Dense_pre_ReLU_RMS"] - 1) > NATIVE_TOLERANCE:
                raise ValueError("native initialized RMS did not match Dense")
    return result
