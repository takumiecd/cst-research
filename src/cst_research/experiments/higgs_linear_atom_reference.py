"""Full NONE-network FP64 oracle, using unchanged975 one-chunk CST VJPs."""

from __future__ import annotations


def full_reference(model, inputs, cotangent=None, *, chunk_atoms=256, trace=None):
    """All current sites/atoms/parameters; no [A,N,N] or full-network autograd graph.

    Only the chosen NONE/unit-gain family is supported. One original975 reference
    chunk graph lives at a time; ordinary/ReLU reverse rules are explicit FP64.
    Returned tensors are detached. This is qualification tooling, not a backend.
    """
    import torch
    from benchmarks.cuda.linear.radius_cell_mass import chunked_reference
    from torch import nn
    from torchcst.nn import CSTModule

    cfg = model._linear_atom_configuration
    if cfg["normalization"] != "none" or not bool(
        (model.fixed_output_gains == 1).all()
    ):
        raise ValueError(
            "this exact manual oracle supports NONE and unit fixed gains only"
        )
    if (
        inputs.device.type != "cpu"
        or inputs.dtype != torch.float64
        or inputs.shape != (4, 28)
        or any(
            p.device.type != "cpu" or p.dtype != torch.float64
            for p in model.parameters()
        )
    ):
        raise ValueError("full B4 CPU FP64 model/input required")
    if type(chunk_atoms) is not int or chunk_atoms < 1:
        raise ValueError("positive integer chunk bound required")
    if cotangent is not None and (
        cotangent.shape != (4, 2)
        or cotangent.dtype != torch.float64
        or cotangent.device.type != "cpu"
    ):
        raise ValueError("global CPU FP64 B4x2 cotangent required")
    if not bool(torch.isfinite(inputs).all()) or any(
        not bool(torch.isfinite(p).all()) for p in model.parameters()
    ):
        raise ValueError("finite CPU input and model parameters required")
    if cotangent is not None and not bool(torch.isfinite(cotangent).all()):
        raise ValueError("finite global cotangent required")
    layers = list(model)
    if (
        len(layers) != 2 * cfg["hidden_layers"] + 1
        or cfg["internal_maps"] != cfg["hidden_layers"] - 1
    ):
        raise ValueError("reference topology metadata differs")
    activations = [inputs.detach()]
    with torch.no_grad():
        for index, layer in enumerate(layers):
            previous = activations[-1]
            if isinstance(layer, CSTModule):
                current = chunked_reference(layer, previous, chunk_atoms=chunk_atoms)
            elif isinstance(layer, nn.Linear):
                current = previous @ layer.weight.T
                if layer.bias is not None:
                    current = current + layer.bias
            elif isinstance(layer, nn.ReLU):
                current = torch.where(
                    previous > 0, previous, torch.zeros_like(previous)
                )
            else:
                raise TypeError("unexpected network module in reference")
            activations.append(current.detach())
            if trace is not None:
                trace[str(index)] = {
                    "RMS": float(current.square().mean().sqrt()),
                    "max_abs": float(current.abs().max()),
                    "zero_fraction": float((current == 0).double().mean()),
                    "positive_fraction": float((current > 0).double().mean()),
                }
    if cotangent is None:
        return activations[-1]
    gradients = {}
    dy = cotangent.detach()
    for index in reversed(range(len(layers))):
        layer = layers[index]
        previous = activations[index]
        if isinstance(layer, nn.ReLU):
            dy = dy * (previous > 0)
        elif isinstance(layer, nn.Linear):
            gradients[str(index) + ".weight"] = dy.T @ previous
            if layer.bias is not None:
                gradients[str(index) + ".bias"] = dy.sum(dim=0)
            dy = dy @ layer.weight.detach()
        elif isinstance(layer, CSTModule):
            with torch.enable_grad():
                _, dy, dp = chunked_reference(
                    layer, previous, dy, chunk_atoms=chunk_atoms
                )
            names = [name for name, _ in layer.named_parameters()]
            if len(names) != 1:
                raise ValueError("CST site must expose one packed parameter")
            gradients[str(index) + "." + names[0]] = dp
        else:
            raise TypeError("unexpected reverse module")
    ordered = {name: gradients[name].detach() for name, _ in model.named_parameters()}
    if set(gradients) != set(ordered):
        raise ValueError("reference parameter inventory differs")
    return activations[-1], dy.detach(), ordered


def reference_array_estimate(width, hidden_layers, c, *, chunk_atoms=256):
    """Known array terms only, NOT a host-RSS bound or measured complexity claim."""
    from .higgs_linear_atom_scaling import exact_atoms

    atoms = exact_atoms(width, c)
    layers = hidden_layers - 1
    return {
        "width": width,
        "hidden_layers": hidden_layers,
        "atoms_per_site": atoms,
        "chunk_atoms": chunk_atoms,
        "forbidden_atom_matrix_FP64_bytes_per_site": 8 * atoms * width * width,
        "dense_export_FP64_bytes_per_site": 8 * width * width,
        "full_factor_pair_FP64_bytes_per_site": 16 * atoms * width,
        "one_chunk_factor_pair_FP64_bytes": 16 * min(chunk_atoms, atoms) * width,
        "retained_forward_activations_FP64_bytes": 8
        * 4
        * (28 + 2 + 2 * hidden_layers * width),
        "packed_parameters_and_gradients_FP64_bytes": 2 * 8 * 4 * atoms * layers,
        "scope": "accounted array terms only; onechunk autograd/profile intermediates, Python/Torch allocator, DenseIO and gradients require actual isolated-process RSS/time measurement; no [A,N,N] construction",
    }
