"""Data-free He/Xavier moment targets for pinned public975 Polar CST maps.

Experimental research utility, not a torchcst public default. The scalar search
accounts for amplitude-dependent radii and the declared symmetric clipping.
"""

from __future__ import annotations

import copy
import math

from . import higgs_multilayer_conditioning as contract

COHORT = "higgs-fan-initialization-v1"
TOLERANCE = 0.005
MAX_SCALE = 32.0


def target_energy(scheme, fan_in, fan_out, *, mode="fan_in", gain=None):
    if any(type(n) is not int or n < 1 for n in (fan_in, fan_out)):
        raise ValueError("positive integer logical fan sizes required")
    if scheme not in ("he", "xavier") or mode not in ("fan_in", "fan_out"):
        raise ValueError("explicit he/xavier and fan_in/fan_out required")
    gain = (math.sqrt(2) if scheme == "he" else 1.0) if gain is None else gain
    if isinstance(gain, bool) or not math.isfinite(gain) or gain <= 0:
        raise ValueError("positive finite gain required")
    variance = (
        gain**2 / (fan_in if mode == "fan_in" else fan_out)
        if scheme == "he"
        else gain**2 * 2 / (fan_in + fan_out)
    )
    return variance * fan_in * fan_out


def encode(template, scale, q, amplitude_max=1.0):
    import torch

    if (
        isinstance(q, bool)
        or not math.isfinite(q)
        or not 0 <= q <= 1
        or not math.isfinite(scale)
        or not 0 <= scale <= MAX_SCALE
        or not math.isfinite(amplitude_max)
        or amplitude_max <= 0
    ):
        raise ValueError("bounded finite q/scale and positive amplitude_max required")
    # Recover the saved original nominal standardized Gaussian draw; original
    # initialization is alpha_init=0, amplitude_max=1 and std=.1/sqrt(A).
    z = template[:, 0] * (math.sqrt(len(template)) / 0.1)
    ratio_unclipped = z * (scale / amplitude_max)
    ratio = ratio_unclipped.clamp(-1 + 1e-6, 1 - 1e-6)
    rho = math.sqrt(1 + 3 * q)
    p = template.detach().clone()
    p[:, :2] = torch.stack((ratio, (1 - ratio.square()).sqrt()), -1) * rho
    return p, int((ratio_unclipped.abs() >= 1 - 1e-6).sum())


def atom_energies(layer, p, *, chunk_atoms=128):
    """Actual FP32 Atoms evaluated with FP64 factors; no dense W allocation.

    For independent symmetric signs, conditional expected ||W||_F^2 equals
    sum_a ||a_a Phi_a||_F^2, even with overlapping atoms and amplitude-dependent
    radii: geometry depends on amplitude squared, not its sign.
    """
    import torch

    values = []
    with torch.no_grad():
        for first in range(0, len(p), chunk_atoms):
            x, y = layer._factor_atoms(p[first : first + chunk_atoms].double())
            values.append(x.square().sum(0) * y.square().sum(0))
    return torch.cat(values)


def solve(evaluate, target):
    if not math.isfinite(target) or target <= 0:
        raise ValueError("positive finite energy target required")
    history = []
    lower, upper = 0.0, None

    def attempt(scale):
        energy, payload = evaluate(scale)
        if not math.isfinite(energy) or energy < 0:
            raise ValueError("finite nonnegative energy required")
        error = abs(energy / target - 1)
        history.append(dict(scale=scale, energy=energy, relative_error=error))
        return energy, error, payload

    for scale in (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, MAX_SCALE):
        energy, error, payload = attempt(scale)
        if error <= TOLERANCE:
            return scale, payload, history
        if energy >= target:
            upper = scale
            break
        lower = scale
    if upper is not None:
        for _ in range(32):
            middle = (lower + upper) / 2
            energy, error, payload = attempt(middle)
            if error <= TOLERANCE:
                return middle, payload, history
            if energy < target:
                lower = middle
            else:
                upper = middle
    raise ValueError("fan target not reached within declared search: " + str(history))


def initialize_layer(layer, *, scheme, q, mode="fan_in", gain=None):
    import torch
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )

    if (
        layer.atoms.p.dtype != torch.float32
        or layer.atoms.p.device.type != "cpu"
        or float(layer.kernel.scalar("amplitude_max")) != 1
        or float(layer.kernel.scalar("alpha_init")) != 0
    ):
        raise ValueError(
            "original CPU FP32 public975 unit-amplitude zero-q template required"
        )
    original = layer.atoms.p.detach().clone()
    ref = copy.deepcopy(layer).double()
    # Chart site counts, not number of raw Atom coordinates.
    nin = layer.input_chart.features
    nout = layer.output_chart.features
    target = target_energy(scheme, nin, nout, mode=mode, gain=gain)

    def evaluate(scale):
        p, clipped = encode(original, scale, q)
        energies = atom_energies(ref, p)
        return float(energies.sum()), (p, clipped, energies)

    scale, (p, clipped, energies), history = solve(evaluate, target)
    amplitude, ri, ro = decoded_values(ref.kernel, p.double())
    return (
        p,
        dict(
            scheme=scheme,
            mode=mode,
            gain=gain,
            q_init=q,
            fan_in=nin,
            fan_out=nout,
            target_expected_Frobenius_squared=target,
            observed_conditional_sign_energy=float(energies.sum()),
            relative_error=abs(float(energies.sum()) / target - 1),
            nominal_normal_std_before_clip=scale,
            physical_amplitude_RMS=float(amplitude.square().mean().sqrt()),
            clipped_atoms=clipped,
            atoms=len(p),
            input_radius_min=float(ri.min()),
            input_radius_max=float(ri.max()),
            output_radius_min=float(ro.min()),
            output_radius_max=float(ro.max()),
            independent_sign_expectation_NOT_realized_W_energy=True,
            data_features_or_Dense_reference_used=False,
            search_history=history,
        ),
        energies,
    )


def make_model(method, *, scheme, atoms=None):
    import torch

    if method not in ("dense", "cst"):
        raise ValueError("dense/cst required")
    model = contract.make_model("dense-none" if method == "dense" else "cst-none")
    sites = model._linear_atom_configuration["internal_module_indices"]
    if method == "cst" and (atoms is None or set(atoms) != set(sites)):
        raise ValueError("all seven initialized Atom tables required")
    with torch.no_grad():
        for number, site in enumerate(sites):
            if method == "dense":
                generator = torch.Generator().manual_seed(17 + 100 + number)
                if scheme == "he":
                    torch.nn.init.kaiming_normal_(
                        model[site].weight,
                        mode="fan_in",
                        nonlinearity="relu",
                        generator=generator,
                    )
                elif scheme == "xavier":
                    torch.nn.init.xavier_normal_(
                        model[site].weight, gain=1.0, generator=generator
                    )
                else:
                    raise ValueError("he/xavier required")
            else:
                if (
                    atoms[site].shape != model[site].atoms.p.shape
                    or atoms[site].dtype != torch.float32
                    or not bool(torch.isfinite(atoms[site]).all())
                ):
                    raise ValueError(
                        "finite shape-matched initialized FP32 Atoms required"
                    )
                model[site].atoms.p.copy_(atoms[site])
    return model


def calibrate(*, scheme, q, progress=None):
    model = contract.make_model("cst-none")
    atoms, records, energies = {}, [], {}
    for site in model._linear_atom_configuration["internal_module_indices"]:
        p, record, energy = initialize_layer(model[site], scheme=scheme, q=q)
        atoms[site], energies[site] = p, energy
        records.append(dict(site=site, **record))
        if progress is not None:
            progress(atoms, records)
    return atoms, records, energies
