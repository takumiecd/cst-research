"""Private HIGGS-only position-LR experiment over an unchanged CSTAdamW step.

Import this adapter only after the external975 source has been verified. It is
not a public optimizer or a replacement backend. Packed atoms are [s,t,mu_o,mu_i].
"""

from __future__ import annotations

import math

import torch
from torch.optim._functional import adamw
from torchcst import CSTAdamW
from torchcst.geometry.spec import FlatTorusGeometrySpec
from torchcst.kernels.parameterizations import PolarAmpRadiusSpec

EXTERNAL_COMMIT = "97580582ed28c4e494bf5c2656bbe94a85c544c3"


class PositionLRCSTAdamW(CSTAdamW):
    """Recompute the position Adam proposal at its actual LR before FP rounding.

    Native AdamW owns the complete ordinary/polar proposal and master clock.
    Only positions use a second functional call from their original contiguous
    state. Strided packed views are never passed to a fused AdamW implementation.
    The normal CST preparation, radius law, amplitude decay and retraction run
    unchanged around this method.
    """

    def __init__(
        self,
        *args,
        position_multiplier=1.0,
        force_recompute_for_qualification=False,
        **kwargs,
    ):
        if (
            type(position_multiplier) not in (int, float)
            or not math.isfinite(position_multiplier)
            or position_multiplier not in (0.1, 1.0, 10.0)
        ):
            raise ValueError("this bounded pilot permits position multipliers .1,1,10")
        if type(force_recompute_for_qualification) is not bool:
            raise TypeError("qualification flag must be bool")
        self.position_multiplier = float(position_multiplier)
        self._force_recompute_for_qualification = force_recompute_for_qualification
        self._last_position_clock = None
        super().__init__(*args, **kwargs)
        if len(self._sites) != 1:
            raise ValueError("pilot requires exactly one managed CST middle layer")
        site = self._sites[0]
        declaration = site.kernel.declaration()
        point = site.atoms.p
        if (
            type(declaration.parameterization) is not PolarAmpRadiusSpec
            or declaration.update.id != "polar_activity_radius"
            or point.ndim != 2
            or point.shape[1] != 4
            or point.dtype not in (torch.float32, torch.float64)
            or any(
                type(chart.geometry.spec) is not FlatTorusGeometrySpec
                or chart.geometry.spec.intrinsic_dim != 1
                for chart in site.cst_charts()
            )
        ):
            raise ValueError(
                "pilot requires packed radius Polar atoms and two scalar flat-torus centers"
            )
        atom_groups = [
            g for g in self.param_groups if any(p is point for p in g["params"])
        ]
        if len(atom_groups) != 1 or len(atom_groups[0]["params"]) != 1:
            raise ValueError("pilot requires one dedicated packed-atom group")

    def _manifest(self):
        return {
            **super()._manifest(),
            "position_lr_pilot": {
                "external_source_commit": EXTERNAL_COMMIT,
                "position_multiplier": self.position_multiplier,
                "columns": [2, 3],
                "force_recompute_for_qualification": self._force_recompute_for_qualification,
                "rule": "direct AdamW position LR from old contiguous state; native polar proposal and master clock",
            },
        }

    @torch.no_grad()
    def _adamw_proposal(self):
        if (
            self.position_multiplier == 1.0
            and not self._force_recompute_for_qualification
        ):
            # No extra _init_group, subtraction, multiplication or copy at all1.
            return super()._adamw_proposal()
        # load_state_dict replaces group dictionaries; resolve the current one.
        point = self._bound_parameters[self._sites[0]]
        group = next(
            g
            for g in self.param_groups
            if len(g["params"]) == 1 and g["params"][0] is point
        )
        params, grads, means, squares, maxima, clocks = [], [], [], [], [], []
        has_complex = self._init_group(
            group, params, grads, means, squares, maxima, clocks
        )
        if not params:
            return super()._adamw_proposal()
        if len(params) != 1 or has_complex or group["weight_decay"] != 0:
            raise ValueError("unexpected packed-atom AdamW layout/decay")
        # clone(contiguous_format) also creates independent storage for A=1.
        contiguous = lambda value: (
            value[:, 2:4].detach().clone(memory_format=torch.contiguous_format)
        )
        position = contiguous(params[0])
        gradient = contiguous(grads[0])
        mean = contiguous(means[0])
        square = contiguous(squares[0])
        maximum = contiguous(maxima[0]) if group["amsgrad"] else None
        clock = clocks[0].detach().clone()
        super()._adamw_proposal()
        beta1, beta2 = group["betas"]
        adamw(
            [position],
            [gradient],
            [mean],
            [square],
            [maximum] if maximum is not None else [],
            [clock],
            foreach=group["foreach"],
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            amsgrad=group["amsgrad"],
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"] * self.position_multiplier,
            weight_decay=0.0,
            eps=group["eps"],
            maximize=group["maximize"],
            has_complex=False,
        )
        params[0][:, 2:4].copy_(position)
        means[0][:, 2:4].copy_(mean)
        squares[0][:, 2:4].copy_(square)
        if maximum is not None:
            maxima[0][:, 2:4].copy_(maximum)
        # Diagnostic only. The true packed step was advanced once by native AdamW.
        self._last_position_clock = clock
