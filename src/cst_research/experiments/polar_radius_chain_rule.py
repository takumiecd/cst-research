"""Check the proposed full Polar-radius derivative on CPU; no learner/update law."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torchcst._backends.torch.parameterizations.polar_amp_width import _width_bounds
from torchcst._backends.torch.parameterizations.radius_cell_mass import decoded_values

from .ffn_model import make_cst
from .ffn_radius import convert_layer


def check_chain_rule():
    source = make_cst(
        16,
        64,
        std=0.001,
        radius_birth=0.5,
        radius_envelope="rational_zero_opening",
        zero_opening_scale=0.001,
    ).double()
    physical = convert_layer(source, "learned").double()
    amplitudes = torch.tensor([-0.9, -0.3, -0.03, 0.03, 0.3, 0.9], dtype=torch.float64)
    qs = torch.tensor([0.1, 0.25, 0.4, 0.6, 0.8, 0.9], dtype=torch.float64)
    rho = (1 + 3 * qs).sqrt()
    p = torch.stack(
        [
            rho * amplitudes,
            rho * (1 - amplitudes.square()).sqrt(),
            torch.tensor([-0.23, 3.17, 9.31, 17.43, 31.11, 64.17]),
            torch.tensor([-0.17, 1.23, 3.41, 7.31, 15.17, 16.23]),
        ],
        1,
    )
    x = torch.randn(
        3, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(4129)
    )
    dy = torch.randn(
        3, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(4130)
    )

    def decode(z, live=True):
        a, r, ro = decoded_values(source.kernel, z)
        torch.testing.assert_close(r, ro, atol=0, rtol=0)
        return torch.cat(
            [a[:, None], (r if live else r.detach())[:, None], z[:, 2:]], 1
        )

    def loss_physical(z):
        v, u = physical._factor_atoms(z)
        return (((x @ v) @ u.T) * dy).sum()

    def loss(z, live=True):
        return loss_physical(decode(z, live))

    z = p.detach().requires_grad_()
    value = loss(z)
    full = torch.autograd.grad(value, z)[0]
    z = p.detach().requires_grad_()
    stopped = torch.autograd.grad(loss(z, False), z)[0]
    y = decode(p).detach().requires_grad_()
    g = torch.autograd.grad(loss_physical(y), y)[0]
    a = y[:, 0].detach().requires_grad_()
    q = qs
    _, lo, hi = _width_bounds(
        source.kernel,
        a,
        q,
        side="input",
        scalar_names=source.kernel.spec.parameterization.width_scalar_names,
    )
    dlo = torch.autograd.grad(lo.sum(), a, retain_graph=True)[0]
    dhi = torch.autograd.grad(hi.sum(), a)[0]
    da = torch.stack(
        [p[:, 1].square() / rho.pow(3), -p[:, 0] * p[:, 1] / rho.pow(3)], 1
    )
    dq = 2 * p[:, :2] / 3
    r = y[:, 1].detach()
    ra = r * ((1 - q) * dlo / lo + q * dhi / hi)
    rq = r * (hi / lo).log()
    manual = torch.cat(
        [
            (g[:, 0] + g[:, 1] * ra)[:, None] * da + (g[:, 1] * rq)[:, None] * dq,
            g[:, 2:],
        ],
        1,
    )
    torch.testing.assert_close(full, manual, atol=3e-10, rtol=3e-8)
    finite = []
    for eps in [1e-6, 5e-7]:
        fd = torch.empty_like(p)
        for i in range(p.shape[0]):
            for j in range(4):
                perturb = torch.zeros_like(p)
                perturb[i, j] = eps
                fd[i, j] = (loss(p + perturb) - loss(p - perturb)) / (2 * eps)
        torch.testing.assert_close(full, fd, atol=2e-7, rtol=2e-5)
        finite.append(
            {"epsilon": eps, "max_absolute_error": float((full - fd).abs().max())}
        )
    normal = p[:, :2] / rho[:, None]
    radial = (full[:, :2] * normal).sum(1)
    expected_radial = g[:, 1] * rq * 2 * rho / 3
    assert radial.abs().max() > 1e-6
    torch.testing.assert_close(radial, expected_radial, atol=3e-10, rtol=3e-8)
    assert (stopped[:, :2] * normal).sum(1).abs().max() < 1e-12
    # Endpoints of q's clamp are nonsmooth: test the radius map's inward derivatives,
    # not an invalid central-difference claim at q=0/1.
    aa = torch.tensor([-0.3, 0.03, 0.3], dtype=torch.float64)

    def radii(q):
        return _width_bounds(
            source.kernel,
            aa,
            torch.full_like(aa, q),
            side="input",
            scalar_names=source.kernel.spec.parameterization.width_scalar_names,
        )[0]

    _, l, u = _width_bounds(
        source.kernel,
        aa,
        None,
        side="input",
        scalar_names=source.kernel.spec.parameterization.width_scalar_names,
    )
    eps = 1e-7
    endpoints = {}
    for q in [0.0, 1.0]:
        expected = (l if q == 0 else u) * (u / l).log()
        fd = (
            (radii(eps) - radii(0)) / eps
            if q == 0
            else (radii(1) - radii(1 - eps)) / eps
        )
        torch.testing.assert_close(fd, expected, atol=2e-6, rtol=2e-6)
        live_q = torch.full_like(aa, q, requires_grad=True)
        current_r = _width_bounds(
            source.kernel,
            aa,
            live_q,
            side="input",
            scalar_names=source.kernel.spec.parameterization.width_scalar_names,
        )[0]
        current_slope = torch.autograd.grad(current_r.sum(), live_q)[0]
        if q == 1:
            # The existing detached-radius value helper selects U at q>=1.
            # It is not a qualified live-radius endpoint derivative.
            assert torch.count_nonzero(current_slope) == 0
            assert torch.count_nonzero(expected) == len(expected)
        endpoints[str(q)] = {
            "side": "right/inward" if q == 0 else "left/inward",
            "max_absolute_error": float((fd - expected).abs().max()),
            "inward_drdq": expected.tolist(),
            "current_value_helper_autograd_drdq": current_slope.tolist(),
        }
    result = {
        "state": "passed",
        "scope": "CPU FP64 mathematical derivative qualification only; no live-radius Polar GPU learner/optimizer implemented or run.",
        "atoms_checked": 6,
        "fields_per_atom": 4,
        "q_values": qs.tolist(),
        "manual_chain_max_absolute_error": float((full - manual).abs().max().detach()),
        "finite_differences": finite,
        "full_radial_gradient": radial.tolist(),
        "detached_radial_gradient": (stopped[:, :2] * normal).sum(1).tolist(),
        "endpoints": endpoints,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    result = check_chain_rule()
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
