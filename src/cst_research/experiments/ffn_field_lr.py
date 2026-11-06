"""Research-only Polar proposal-rate experiment; not a torchcst public API.

Rates scale AdamW's *normalized proposal*, not its input gradients. The joint
(s,t) block controls amplitude direction; its radius/activity still follows the
existing Polar law with the same explicit outer step size in both study arms.
Changing amplitude motion can therefore change activity intrinsically.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from torchcst import CSTLinear, CSTOptimizer, FixedSelector, make_algorithm_registry
from torchcst.geometry.spec import FlatTorusGeometrySpec
from torchcst.kernels.parameterizations import PolarAmpRadiusSpec


@dataclass(frozen=True)
class FieldRateRecipe:
    """Absolute peak rates; all follow the same scalar schedule factor."""

    ordinary: float = 0.0003
    amplitude: float = 0.0048
    input_center: float = 0.0048
    output_center: float = 0.0048
    outer_clock: float = 0.0048

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative scalar")
        if self.ordinary == 0 or self.outer_clock == 0:
            raise ValueError("ordinary and outer_clock peak rates must be positive")

    def manifest(self):
        return {
            "version": 1,
            "peak_field_lrs": asdict(self),
            "schedule": "common factor applied through ordinary/outer group lr",
            "proposal": "scale joint(s,t), output_center, input_center after AdamW",
            "activity": "existing Polar policy at outer_clock; motion coupling retained",
            "scope": "PolarAmpRadiusSpec, 1D flat-torus axes, raw/logical decay0",
        }


def recipe_for(arm):
    """Declared short-study rates; not an optimality claim or public default."""
    rates = {
        "all16": 0.0048,
        "amplitude1-centers16": 0.0003,
        "amplitude8-centers16": 0.0024,
    }
    if arm not in rates:
        raise ValueError("unknown field LR arm")
    return FieldRateRecipe(amplitude=rates[arm])


class FieldRateOptimizer(CSTOptimizer):
    """Restricted experimental wrapper, retaining one Parameter/state per Atom table.

    The exact AdamW base preserves standard moment interpretation. Its pre-hook
    sees projected gradients and its post-hook runs before CST retraction. No
    gradients or moment accumulators are scaled. Checkpoints bind the immutable
    field-rate recipe and reject mismatches before loading any state.
    """

    def __init__(self, model, recipe=FieldRateRecipe(), *, native=True):
        if not isinstance(recipe, FieldRateRecipe):
            raise TypeError("recipe must be FieldRateRecipe")
        self.recipe = recipe
        sites = [m for m in model.modules() if isinstance(m, CSTLinear)]
        if not sites:
            raise ValueError("at least one CSTLinear required")
        for site in sites:
            if (
                type(site.kernel.spec.parameterization) is not PolarAmpRadiusSpec
                or site.atoms.p.ndim != 2
                or site.atoms.p.shape[1] != 4
                or any(
                    type(c.geometry.spec) is not FlatTorusGeometrySpec
                    or c.geometry.spec.intrinsic_dim != 1
                    or c.trainable
                    for c in site.cst_charts()
                )
            ):
                raise ValueError(
                    "research adapter requires fixed 1D flat-torus Polar radius sites"
                )
        atoms = [site.atoms.p for site in sites]
        atom_ids = {id(p) for p in atoms}
        ordinary = [p for p in model.parameters() if id(p) not in atom_ids]
        self._field_atom_ids = atom_ids
        self._scaled_blocks = tuple(
            (first, last, rate / recipe.outer_clock)
            for first, last, rate in (
                (0, 2, recipe.amplitude),
                (2, 3, recipe.output_center),
                (3, 4, recipe.input_center),
            )
            if rate != recipe.outer_clock
        )
        groups = []
        if ordinary:
            groups.append({"params": ordinary, "lr": recipe.ordinary, "lr_scale": 1.0})
        groups.append(
            {
                "params": atoms,
                "lr": recipe.outer_clock,
                "lr_scale": recipe.outer_clock / recipe.ordinary,
            }
        )
        base = torch.optim.AdamW(
            groups,
            lr=recipe.ordinary,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            foreach=False,
        )
        selector = None
        if native:
            if any(p.device.type != "cuda" or p.dtype != torch.float32 for p in atoms):
                raise ValueError(
                    "native study requires CUDA FP32; use native=False explicitly for CPU reference"
                )
            from torchcst._backends.cuda.algorithms.polar_update.plans import (
                RADIUS_FLAT_TORUS_FUSED,
            )

            selector = FixedSelector(
                RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry()
            )
        elif any(p.device.type != "cpu" for p in atoms):
            raise ValueError("reference study requires CPU")
        super().__init__(
            base, model=model, update_selector=selector, check_finite=False
        )
        self._proposal_snapshots = []
        self._proposal_hooks = (
            base.register_step_pre_hook(self._before_proposal),
            base.register_step_post_hook(self._after_proposal),
        )

    @torch.no_grad()
    def _before_proposal(self, base, args, kwargs):
        if self._proposal_snapshots:
            raise RuntimeError(
                "unfinished prior proposal; do not retry failed optimizer steps"
            )
        for group in base.param_groups:
            if (
                group["weight_decay"] != 0
                or group.get("amplitude_weight_decay", 0) != 0
            ):
                raise ValueError(
                    "field LR study requires zero raw and logical weight decay"
                )
            for point in group["params"]:
                if id(point) not in self._field_atom_ids or point.grad is None:
                    continue
                for first, last, ratio in self._scaled_blocks:
                    self._proposal_snapshots.append(
                        (point, first, last, ratio, point[:, first:last].clone())
                    )

    @torch.no_grad()
    def _after_proposal(self, base, args, kwargs):
        for point, first, last, ratio, old in self._proposal_snapshots:
            proposal = point[:, first:last]
            proposal.copy_(old + (proposal - old) * ratio)
        self._proposal_snapshots.clear()

    def state_dict(self):
        result = super().state_dict()
        result["ffn_field_lr"] = self.recipe.manifest()
        return result

    def load_state_dict(self, state_dict):
        if state_dict.get("ffn_field_lr") != self.recipe.manifest():
            raise ValueError("field LR checkpoint recipe differs")
        return super().load_state_dict(
            {k: v for k, v in state_dict.items() if k != "ffn_field_lr"}
        )
