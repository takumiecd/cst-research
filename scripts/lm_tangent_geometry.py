"""Analysis-only Jacobian actions from public factors; never updates a model."""

import math

import torch


def dot(a, b):
    return float((a.double() * b.double()).sum())


def cosine(a, b):
    aa, bb = dot(a, a), dot(b, b)
    return dot(a, b) / math.sqrt(aa * bb) if aa > 0 and bb > 0 else None


def factor_partials(layer, p, chunk=128):
    """Atoms are independent columns: four public-factor JVPs recover each column.

    The CPU reference's discrete routing stays fixed in its mathematical VJP.
    This is the same declared one-sided derivative at envelope endpoints.
    """
    assert p.device.type == "cpu"
    vi, uo, dis, dos = [], [], [[] for _ in range(4)], [[] for _ in range(4)]
    for first in range(0, len(p), chunk):
        point = p[first : first + chunk].detach().clone()
        values = layer.operator.factors(point)
        vi.append(values[0].detach())
        uo.append(values[1].detach())
        for k in range(4):
            direction = torch.zeros_like(point)
            direction[:, k] = 1
            _, tangent = torch.autograd.functional.jvp(
                layer.operator.factors, point, direction
            )
            dis[k].append(tangent[0].detach())
            dos[k].append(tangent[1].detach())
    return (
        torch.cat(vi, 1),
        torch.cat(uo, 1),
        [torch.cat(x, 1) for x in dis],
        [torch.cat(x, 1) for x in dos],
    )


class FactorTangent:
    def __init__(self, x, vi, uo, dis, dos):
        self.x, self.vi, self.uo = x, vi, uo
        self.b = x @ vi
        self.cs = [x @ d for d in dis]
        self.dos = dos
        self.atoms = vi.shape[1]
        self.outputs = uo.shape[0]
        self.size = self.atoms * 4 + self.outputs

    def jvp(self, v):
        p, bias = v[: self.atoms * 4].reshape(self.atoms, 4), v[self.atoms * 4 :]
        left = sum(c * p[:, k] for k, c in enumerate(self.cs))
        right = sum(d * p[:, k] for k, d in enumerate(self.dos))
        return left @ self.uo.T + self.b @ right.T + bias

    def vjp(self, g):
        left, right = g @ self.uo, g.T @ self.b
        fields = torch.stack(
            [(left * c).sum(0) + (right * d).sum(0) for c, d in zip(self.cs, self.dos)],
            1,
        )
        return torch.cat((fields.flatten(), g.sum(0)))

    def column_norms(self):
        b = self.b.double()
        u = self.uo.double()
        norms = []
        for c, d in zip(self.cs, self.dos):
            c, d = c.double(), d.double()
            squared = (
                c.square().sum(0) * u.square().sum(0)
                + b.square().sum(0) * d.square().sum(0)
                + 2 * (c * b).sum(0) * (u * d).sum(0)
            )
            norms.append(squared.clamp_min(0).sqrt())
        return torch.cat(
            (
                torch.stack(norms, 1).flatten(),
                torch.full(
                    (self.outputs,),
                    math.sqrt(len(self.x)),
                    device=self.x.device,
                    dtype=torch.float64,
                ),
            )
        ).to(self.x)


class LinearTangent:
    def __init__(self, x, outputs, mask=None):
        self.x, self.outputs, self.mask = x, outputs, mask
        self.weight_shape = (outputs, x.shape[1])
        self.weights = int(mask.sum()) if mask is not None else outputs * x.shape[1]
        self.size = self.weights + outputs

    def jvp(self, v):
        if self.mask is None:
            weight = v[: self.weights].reshape(self.weight_shape)
        else:
            weight = self.x.new_zeros(self.weight_shape)
            weight[self.mask] = v[: self.weights]
        return self.x @ weight.T + v[self.weights :]

    def vjp(self, g):
        weight = g.T @ self.x
        return torch.cat(
            (weight[self.mask] if self.mask is not None else weight.flatten(), g.sum(0))
        )


def free_projection(x, target, rtol=1e-7):
    """All dW/db directions on this exact X, not a Dense teacher or fitting task."""
    augmented = torch.cat(
        (x.double(), torch.ones(len(x), 1, device=x.device, dtype=torch.float64)), 1
    )
    u, s, _ = torch.linalg.svd(augmented, full_matrices=False)
    keep = s > s.max() * rtol
    q = u[:, keep]
    projection = q @ (q.T @ target.double())
    return projection.to(target), {
        "numerical_rank": int(keep.sum()),
        "rows": len(x),
        "columns": augmented.shape[1],
        "relative_singular_cutoff": rtol,
        "maximum_singular": float(s.max()),
        "minimum_retained_singular": float(s[keep].min()),
    }


def masked_projection(x, target, mask, rtol=1e-7, chunk=16):
    result = torch.empty_like(target)
    ranks = []
    for first in range(0, mask.shape[0], chunk):
        rows = mask[first : first + chunk]
        count = rows.sum(1)
        assert torch.equal(count, count[:1].expand_as(count))
        indices = rows.nonzero()[:, 1].reshape(len(rows), int(count[0]))
        matrices = x[:, indices].permute(1, 0, 2).double()
        matrices = torch.cat(
            (
                matrices,
                torch.ones(len(rows), len(x), 1, device=x.device, dtype=torch.float64),
            ),
            2,
        )
        u, s, _ = torch.linalg.svd(matrices, full_matrices=False)
        keep = s > s[:, :1] * rtol
        goals = target[:, first : first + len(rows)].T.double()
        coefficients = torch.einsum("bmk,bm->bk", u, goals) * keep
        result[:, first : first + len(rows)] = torch.einsum(
            "bmk,bk->bm", u, coefficients
        ).T.to(target)
        ranks.extend(keep.sum(1).tolist())
    return result, {
        "minimum_row_rank": min(ranks),
        "maximum_row_rank": max(ranks),
        "relative_singular_cutoff": rtol,
    }


def tangent_projection(operator, target, initial, max_iterations=256, tolerance=1e-4):
    """Column-normalized CGLS. Incomplete solves are reachable lower bounds only."""
    norms = operator.column_norms()
    keep = norms > norms.max() * 1e-10
    scale = torch.where(keep, norms.reciprocal(), torch.zeros_like(norms))
    forward = lambda x: operator.jvp(scale * x)
    transpose = lambda x: scale * operator.vjp(x)
    # A one-dimensional candidate initializes a valid lower bound.
    actual = operator.jvp(initial)
    coefficient = dot(actual, target) / max(dot(actual, actual), 1e-30)
    x = (
        torch.where(
            keep,
            initial / torch.where(keep, scale, torch.ones_like(scale)),
            torch.zeros_like(initial),
        )
        * coefficient
    )
    residual = target - forward(x)
    s = transpose(residual)
    direction = s.clone()
    ss = dot(s, s)
    reference = max(math.sqrt(dot(transpose(target), transpose(target))), 1e-30)
    trace = []
    for k in range(max_iterations + 1):
        normal_residual = math.sqrt(ss) / reference
        if k in (0, 8, 16, 32, 64, 128, 256) or normal_residual <= tolerance:
            prediction = forward(x)
            trace.append(
                {
                    "iteration": k,
                    "normal_residual": normal_residual,
                    "captured_target_norm_ratio": math.sqrt(
                        max(0, dot(target, target) - dot(residual, residual))
                        / max(dot(target, target), 1e-30)
                    ),
                    "direction_cosine": cosine(prediction, target),
                }
            )
        if normal_residual <= tolerance or k == max_iterations:
            break
        q = forward(direction)
        qq = dot(q, q)
        if qq <= 1e-40:
            break
        alpha = ss / qq
        x += alpha * direction
        residual -= alpha * q
        new = transpose(residual)
        newss = dot(new, new)
        direction = new + (newss / max(ss, 1e-40)) * direction
        s, ss = new, newss
    prediction = forward(x)
    # Recompute residual explicitly so recurrence drift cannot claim convergence.
    exact = target - prediction
    residual_norm = math.sqrt(dot(transpose(exact), transpose(exact))) / reference
    return prediction, {
        "iterations": k,
        "normal_residual_recomputed": residual_norm,
        "converged": residual_norm <= tolerance,
        "tolerance": tolerance,
        "column_relative_cutoff": 1e-10,
        "kept_columns": int(keep.sum()),
        "total_columns": operator.size,
        "trace": trace,
        "scope": "Unconstrained, numerically resolved local tangent; ignores one-sided bounds. Nonconverged value is not a structural ceiling.",
    }


def adam_directions(gradient, first, second, step, rate, eps=1e-8, betas=(0.9, 0.95)):
    """Moments are the post-current-gradient values, with original clipping/rates."""
    m = first / (1 - betas[0] ** step)
    historical_denominator = (second / (1 - betas[1] ** step)).sqrt() + eps
    current_denominator = gradient.abs() + eps
    return {
        "historical_m_historical_v": -rate * m / historical_denominator,
        "current_g_historical_v": -rate * gradient / historical_denominator,
        "historical_m_current_g2": -rate * m / current_denominator,
        "current_g_current_g2": -rate * gradient / current_denominator,
        "current_gradient": -rate * gradient,
    }
