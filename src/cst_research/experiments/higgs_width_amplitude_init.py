"""Paired raw-amplitude initialization experiment; fixed public975 operators."""

from __future__ import annotations

import math

from . import higgs_multilayer_conditioning as model_contract

COHORT = "higgs-width-amplitude-init-v1"
ARMS = {
    "dense-control": "dense-none",
    "old-init": "cst-none",
    "width-init": "cst-none",
    "old-init-gain": "cst-fixed-gain",
}
WIDTH, HIDDEN, ATOMS, SEED = 1024, 8, 4096, 17
MILESTONES = model_contract.MILESTONES


def physical_initialization_record(model):
    import torch
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )

    result = []
    with torch.no_grad():
        for index in model._linear_atom_configuration["internal_module_indices"]:
            layer = model[index]
            if not hasattr(layer, "atoms"):
                continue
            amplitude, ri, ro = decoded_values(layer.kernel, layer.atoms.p)
            result.append(
                {
                    "site": index,
                    "amplitude_RMS": float(amplitude.square().mean().sqrt()),
                    "amplitude_max_abs": float(amplitude.abs().max()),
                    "input_radius_min": float(ri.min()),
                    "input_radius_max": float(ri.max()),
                    "output_radius_min": float(ro.min()),
                    "output_radius_max": float(ro.max()),
                }
            )
    return result


def make_model(arm, *, gains=None, width=WIDTH, hidden=HIDDEN):
    """Keep the architecture contract unchanged; the arm is in experiment identity.

    A checkpoint's factory reconstructs its shape and loads its serialized Atom
    values. This uses the unchanged native load path for new initial values too.
    """
    import torch

    if arm not in ARMS:
        raise ValueError("undeclared initialization arm")
    model = model_contract.make_model(
        ARMS[arm], gains=gains, width=width, hidden=hidden
    )
    if arm == "width-init":
        if not bool((model.fixed_output_gains == 1).all()):
            raise ValueError("new raw initialization requires unit gains")
        with torch.no_grad():
            for index in model._linear_atom_configuration["internal_module_indices"]:
                layer = model[index]
                p = layer.atoms.p
                # The exact public initializer uses alpha_init=0, amplitude_max=1
                # and (normal draw, sqrt(1-draw^2)) polar encoding. Pair the same
                # draw and centers, changing std by sqrt(N). At N1024 the factor
                # is exactly32, matching std0.05 vs0.0015625 in FP32.
                if (
                    float(layer.kernel.scalar("alpha_init")) != 0
                    or float(layer.kernel.scalar("amplitude_max")) != 1
                ):
                    raise ValueError(
                        "qualified zero-alpha unit-amplitude initializer required"
                    )
                ratio = (p[:, 0] * math.sqrt(width)).clamp(-1 + 1e-6, 1 - 1e-6)
                p[:, :2].copy_(
                    torch.stack((ratio, (1 - ratio.square()).sqrt()), dim=-1)
                )
    model._amplitude_init_arm = arm
    return model


def initialization_policy(arm):
    if arm not in ARMS:
        raise ValueError("undeclared initialization arm")
    return {
        "arm": arm,
        "model_contract_case": ARMS[arm],
        "physical_amplitude_std": None
        if arm == "dense-control"
        else 0.1 / math.sqrt(ATOMS / WIDTH)
        if arm == "width-init"
        else 0.1 / math.sqrt(ATOMS),
        "formula": "0.1/sqrt(A/N)"
        if arm == "width-init"
        else "Dense legacy"
        if arm == "dense-control"
        else "0.1/sqrt(A)",
        "normalization": "none",
        "gain": "same fixed256 TRAIN pre-ReLU Dense RMS calibration as prior study"
        if arm == "old-init-gain"
        else "all ones",
        "ordinary_IO_initializer": "unchanged legacy-linear-default",
        "coordinate_draws_and_standardized_amplitude_draws_paired": True,
        "radius_changes_with_new_amplitude_intentionally_allowed": arm == "width-init",
        "symmetric_clip": [-1 + 1e-6, 1 - 1e-6],
    }


def protocol():
    return {
        "version": COHORT,
        "arms": list(ARMS),
        "initialization": [initialization_policy(a) for a in ARMS],
        "width": WIDTH,
        "hidden_layers": HIDDEN,
        "internal_CST_maps": HIDDEN - 1,
        "atoms_per_CST_map": ATOMS,
        "seed": SEED,
        "epochs": 1,
        "task_updates_per_arm": 25000,
        "total_task_updates": 100000,
        "batch_size": 4,
        "learning_rate": 0.0003,
        "position_multiplier": 1,
        "optimizer": "unchanged stockq1 fused CSTAdamW/AdamW, eps1e-8, ordinary/amplitude WD0.01, raw-Atom WD0",
        "dtype": "FP32",
        "TF32": False,
        "mode": "eager",
        "milestones": list(MILESTONES),
        "selection": "fixedfinal25000; no bestDEV, HPO or earlystop",
        "probes": "same first TRAIN B4 on loaded copies at0/1000/5000/10000/15000/20000/25000; no optimizer update in probes; source fullcore remains unchanged",
        "native_qualification": "fresh full-network FP64 manual VJP dy1/dy1e24 with unchanged gates, independent31Atom quadrature and scalarAdam, 8bare vs4load4 vsobservedfirst+7 exact fullcores before task training",
        "task_data": "TRAIN100k rows[1000000,1100000), DEV50k[1100000,1150000); no primary confirmation or officialtest",
        "precision_scope": "retained arrays independently recountable; denominator summaries are accepted-state reconstructions, not native intermediate captures",
        "timing_memory_scope": "diagnostic synchronized bare updates and allocated/reserved peaks; probe, evaluation, calibration and IO cost separate",
        "raw_cap_bytes": 6 * 2**30,
        "JSON_cap_bytes": 64 * 2**20,
        "automatic_followup": False,
        "formal_NI_Atomorder_or_initialization_optimality_claim": False,
    }
