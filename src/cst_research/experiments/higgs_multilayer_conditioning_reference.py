"""New H8 full-network manual FP64 reverse including no-affine LN/fixed gains."""

from __future__ import annotations

from .higgs_multilayer_conditioning import EPS


def normalize(value, kind):
    if kind == "none":
        return value, None
    if kind != "ln":
        raise ValueError("reference supports explicit NONE or LN only")
    centered = value - value.mean(-1, keepdim=True)
    inverse = (centered.square().mean(-1, keepdim=True) + EPS).rsqrt()
    normalized = centered * inverse
    return normalized, (normalized.detach(), inverse.detach())


def norm_reverse(dy, cache):
    if cache is None:
        return dy
    normalized, inverse = cache
    return inverse * (
        dy
        - dy.mean(-1, keepdim=True)
        - normalized * (dy * normalized).mean(-1, keepdim=True)
    )


def affine_or_relu(layer, value, *, chunk_atoms=256):
    import torch
    from benchmarks.cuda.linear.radius_cell_mass import chunked_reference
    from torch import nn
    from torchcst.nn import CSTModule

    if isinstance(layer, CSTModule):
        return chunked_reference(layer, value, chunk_atoms=chunk_atoms)
    if isinstance(layer, nn.Linear):
        result = value @ layer.weight.T
        return result if layer.bias is None else result + layer.bias
    if isinstance(layer, nn.ReLU):
        return torch.where(value > 0, value, torch.zeros_like(value))
    raise TypeError("undeclared full-network module")


def forward(model, inputs, *, collect=False, chunk_atoms=256):
    import torch

    cfg = model._conditioning_configuration
    sites = model._linear_atom_configuration["internal_module_indices"]
    value = inputs
    traces = {}
    with torch.no_grad():
        for index, layer in enumerate(model):
            value = affine_or_relu(layer, value, chunk_atoms=chunk_atoms)
            if index in sites:
                value, _ = normalize(value, cfg["normalization"])
                value = value * model.fixed_output_gains[sites.index(index)]
            traces[str(index)] = value.detach()
    return (value, traces) if collect else value


def full_reference(model, inputs, cotangent, *, chunk_atoms=256):
    import torch
    from benchmarks.cuda.linear.radius_cell_mass import chunked_reference
    from torch import nn
    from torchcst.nn import CSTModule

    cfg = model._conditioning_configuration
    layout = model._linear_atom_configuration
    if (
        inputs.shape != (4, 28)
        or inputs.dtype != torch.float64
        or inputs.device.type != "cpu"
        or cotangent.shape != (4, 2)
        or cotangent.dtype != torch.float64
        or cotangent.device.type != "cpu"
        or any(
            p.dtype != torch.float64 or p.device.type != "cpu"
            for p in model.parameters()
        )
    ):
        raise ValueError("full B4 CPU FP64 model/X/globaldY required")
    if not bool(
        torch.isfinite(inputs).all() and torch.isfinite(cotangent).all()
    ) or any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("nonfinite reference input/cotangent/parameter")
    if (
        type(chunk_atoms) is not int
        or chunk_atoms < 1
        or len(model) != 2 * layout["hidden_layers"] + 1
        or cfg["epsilon"] != EPS
    ):
        raise ValueError("reference topology/chunk/epsilon differs")
    sites = layout["internal_module_indices"]
    activations, caches = [inputs.detach()], {}
    with torch.no_grad():
        for index, layer in enumerate(model):
            value = affine_or_relu(layer, activations[-1], chunk_atoms=chunk_atoms)
            if index in sites:
                value, cache = normalize(value, cfg["normalization"])
                caches[index] = cache
                value = value * model.fixed_output_gains[sites.index(index)]
            activations.append(value.detach())
    dy, gradients = cotangent.detach(), {}
    for index in reversed(range(len(model))):
        layer, previous = model[index], activations[index]
        if index in sites:
            dy = dy * model.fixed_output_gains[sites.index(index)]
            dy = norm_reverse(dy, caches[index])
        if isinstance(layer, nn.ReLU):
            dy = dy * (previous > 0)
        elif isinstance(layer, nn.Linear):
            gradients[f"{index}.weight"] = dy.T @ previous
            if layer.bias is not None:
                gradients[f"{index}.bias"] = dy.sum(0)
            dy = dy @ layer.weight.detach()
        elif isinstance(layer, CSTModule):
            with torch.enable_grad():
                _, dy, dp = chunked_reference(
                    layer, previous, dy, chunk_atoms=chunk_atoms
                )
            names = [n for n, _ in layer.named_parameters()]
            if len(names) != 1:
                raise ValueError("CST must own exactone packedP")
            gradients[f"{index}." + names[0]] = dp
        else:
            raise TypeError("undeclared reverse module")
    ordered = {n: gradients[n].detach() for n, _ in model.named_parameters()}
    if ordered.keys() != dict(model.named_parameters()).keys() or set(ordered) != set(
        gradients
    ):
        raise ValueError("full gradient owner inventory differs")
    return activations[-1], dy.detach(), ordered
