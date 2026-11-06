"""Draft A=cN deep-model factory and parameter algebra; no training execution path."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction

from . import higgs_depth_normalization as existing_norm
from . import higgs_depth_study as inherited

COHORT = "higgs-linear-atom-scaling-draft-v1"
ORDINARY_INIT = "legacy-linear-default"
PUBLIC_CST_WIDTHS = (128, 1024, 2048, 4096, 8192, 16384)


def rational_c(value):
    """No binary-float conversion or silent rounding of the exact atom budget."""
    if type(value) is int:
        result = Fraction(value)
    elif isinstance(value, Fraction):
        result = value
    elif isinstance(value, str):
        try:
            result = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("c requires an explicit positive rational") from error
    else:
        raise ValueError(
            "use integer, Fraction or exact rational string c; floats/bools refused"
        )
    if result <= 0:
        raise ValueError("positive c required")
    return result


def exact_atoms(width, c):
    if type(width) is not int or width < 4:
        raise ValueError("integer width >=4 required")
    budget = rational_c(c) * width
    if budget.denominator != 1:
        raise ValueError("A=cN must be integral; no floor/ceil or budget rounding")
    return budget.numerator


def fraction_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def configuration(
    *, method, width, hidden_layers, c, normalization, ordinary_initialization
):
    if (
        method not in ("dense", "cst")
        or type(hidden_layers) is not int
        or hidden_layers not in (4, 8)
    ):
        raise ValueError("draft supports Dense/CST and Hhidden4/8 only")
    if type(width) is not int or width < 4:
        raise ValueError("integer width >=4 required")
    if normalization not in ("none", "rms", "ln"):
        raise ValueError("explicit draft normalization required")
    if ordinary_initialization != ORDINARY_INIT:
        raise ValueError(
            "only explicit legacy initializer prototype exists; recipe not sealed"
        )
    if method == "dense":
        if c is not None:
            raise ValueError("Dense has no atom budget")
        atom_c, atoms = None, None
    else:
        if width not in PUBLIC_CST_WIDTHS:
            raise ValueError("CST factory retains public975 helper width guard")
        atom_c, atoms = rational_c(c), exact_atoms(width, c)
    internal = hidden_layers - 1
    return {
        "cohort": COHORT,
        "method": method,
        "width": width,
        "hidden_layers": hidden_layers,
        "internal_maps": internal,
        "architecture": [28] + [width] * hidden_layers + [2],
        "internal_module_indices": list(range(2, 2 * hidden_layers, 2)),
        "output_module_index": 2 * hidden_layers,
        "internal_bias": False,
        "dense_io_trainable": True,
        "c_exact": None if atom_c is None else fraction_record(atom_c),
        "atoms_per_cst_layer": atoms,
        "effective_alpha_exact": None
        if atom_c is None
        else fraction_record(atom_c / width),
        "normalization": normalization,
        "norm_eps": existing_norm.EPS,
        "norm_affine": False,
        "ordinary_initialization": ordinary_initialization,
        "initialization_recipe_sealed": False,
        "normalization_recipe_sealed": False,
        "trainable_scalars": internal
        * (width * width if method == "dense" else 4 * atoms)
        + 31 * width
        + 2,
        "trainable_tensor_count": internal + 4,
        "fixed_gain_buffer_scalars": internal,
        "normalization_trainable_scalars": 0,
    }


def make_model(
    *,
    method,
    width,
    hidden_layers,
    c,
    normalization,
    ordinary_initialization,
    seed=17,
    fixed_gains=None,
):
    """Explicit prototype; CPU reference construction is not CUDA admission."""
    import torch
    from benchmarks.training.cell_mass_mnist import make_radius_cst
    from torch import nn
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

    if torch.get_default_dtype() != torch.float32:
        raise ValueError(
            "prototype initialization requires ordinary FP32 default dtype"
        )
    cfg = configuration(
        method=method,
        width=width,
        hidden_layers=hidden_layers,
        c=c,
        normalization=normalization,
        ordinary_initialization=ordinary_initialization,
    )
    if type(seed) is not int:
        raise ValueError("explicit integer seed required")
    # Construct IO first under their own streams, exactly like the H4 parent.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        first = nn.Linear(28, width)
        torch.manual_seed(seed + 2)
        last = nn.Linear(width, 2)
        layers = [first, nn.ReLU()]
        for index in range(cfg["internal_maps"]):
            torch.manual_seed(seed + (1000 if method == "cst" else 100) + index)
            layer = (
                make_radius_cst(
                    width,
                    cfg["atoms_per_cst_layer"],
                    execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
                )
                if method == "cst"
                else nn.Linear(width, width, bias=False)
            )
            layers.extend((layer, nn.ReLU()))
        layers.append(last)

    class ScalingSequential(nn.Sequential):
        def forward(self, value):
            return trace(self, value)[-1]

    model = ScalingSequential(*layers)
    gains = [1.0] * cfg["internal_maps"] if fixed_gains is None else fixed_gains
    gain_tensor = torch.as_tensor(gains, dtype=torch.float32)
    if (
        gain_tensor.shape != (cfg["internal_maps"],)
        or not bool(torch.isfinite(gain_tensor).all())
        or not bool(((gain_tensor >= 1e-6) & (gain_tensor <= 1e6)).all())
    ):
        raise ValueError("explicit positive fixed gains in inherited bounds required")
    model.register_buffer("fixed_output_gains", gain_tensor.clone())
    model._linear_atom_configuration = cfg
    model._cst_research_cohort = COHORT
    return model


def trace(model, value):
    cfg = model._linear_atom_configuration
    sites = cfg["internal_module_indices"]
    values = []
    for index, layer in enumerate(model):
        value = layer(value)
        if index in sites:
            value = existing_norm.norm(value, cfg["normalization"])
            value = value * model.fixed_output_gains[sites.index(index)]
        values.append(value)
    return values


def reference_forward(model, value):
    """CPU algebra oracle: public dense_weight + matmul, independent norm formulas."""
    import torch
    from torch import nn
    from torchcst.nn import CSTModule

    cfg = model._linear_atom_configuration
    sites = cfg["internal_module_indices"]
    for index, layer in enumerate(model):
        if isinstance(layer, CSTModule):
            value = value @ layer.dense_weight().T
        elif isinstance(layer, nn.Linear):
            value = value @ layer.weight.T
            if layer.bias is not None:
                value = value + layer.bias
        else:
            value = torch.where(value > 0, value, torch.zeros_like(value))
        if index in sites:
            if cfg["normalization"] == "rms":
                value = value / torch.sqrt(
                    torch.mean(value * value, dim=-1, keepdim=True) + existing_norm.EPS
                )
            elif cfg["normalization"] == "ln":
                centered = value - torch.mean(value, dim=-1, keepdim=True)
                value = centered / torch.sqrt(
                    torch.mean(centered * centered, dim=-1, keepdim=True)
                    + existing_norm.EPS
                )
            value = value * model.fixed_output_gains[sites.index(index)]
    return value


def protocol():
    return {
        "version": COHORT,
        "status": "PREPARED_STAGE_A_QUALIFICATION_ONLY_NOT_EXECUTED",
        "external_source_commit": inherited.SOURCE,
        "external_source_manifest_sha256": inherited.SOURCE_MANIFEST,
        "architecture": "28→N×H→2; Hhidden=4/8, L=H−1=3/7 biasfree internal maps, HReLUs, allDenseIO trainable",
        "widths_planned": [1024, 2048, 4096],
        "later_width_not_pilot": 8192,
        "atom_rule": "exact A=cN per CST map, no rounding",
        "parameter_formulas": {
            "dense": "L*N^2+31*N+2",
            "cst": "(4*c*L+31)*N+2",
            "no_affine_norm_trainable_overhead": 0,
            "fixed_gain_buffer_scalars": "L, not learned parameters",
        },
        "parameter_scaling_scope": "linear N only for fixed c AND fixed L; linear L at fixed N/c; not measured O(N) time/peak memory",
        "kernel": {
            "factory": "unchanged public975 make_radius_cst",
            "cell_mass": True,
            "Chebyshev_modes": 8,
            "merge_factor": 2,
            "w_c": 0.3,
            "spacing": 1.0,
            "plan": "GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C",
            "optimizer": "unchanged stockCSTAdamW q1; no position-LR adapter",
        },
        "construction_seed_streams": {
            "input": "seed+1",
            "output": "seed+2",
            "Dense_internal_j": "seed+100+j",
            "CST_internal_j": "seed+1000+j",
        },
        "legacy_H4_exact_compatibility_c": {
            "A104_at_N1024": "13/128",
            "A1048_at_N1024": "131/128",
        },
        "recipe_sealed": False,
        "normalization_choice": "none",
        "fixed_gain_calibration_choice": "none; unit fixed gains",
        "ordinary_initialization_choice": ORDINARY_INIT,
        "prototype_initializer": {
            "implemented": ORDINARY_INIT,
            "public_behavior": "unchanged torch.nn.Linear reset_parameters; biasfree internal Uniform±1/sqrt(N), not He-ReLU gain",
            "H8_adequacy": "UNMEASURED; do not assume a usable deepDense baseline or silently change initialization",
        },
        "planned_candidate_pilot_NOT_authorized_for_execution": {
            "widths": [1024, 2048],
            "hidden_layers": [4, 8],
            "c": [1, 4, 16],
            "seed": 17,
            "proposed_epochs": 1,
            "proposed_B4_updates_per_case": 25000,
            "CST_cases": 12,
            "Dense_stock_controls": 4,
            "additional_matched_norm_Dense_controls": 0,
            "total_cases": 16,
            "total_main_updates": 400000,
            "lr": 0.0003,
            "normalization": "none",
            "fixed_gains": 1,
            "calibration": False,
            "width4096_or8192_included": False,
            "short_budget_role": "screen fitting/conditioning only; no Dense-quality or threshold claim",
        },
        "before_any_training": [
            "completed current9 evidence reviewed; NONE user correction recorded, source/native qualification still pending",
            "confirm usable Dense-stock and matched-norm Dense when nontrivial norm is chosen",
            "new width/depth independent Y/dX/all-parameter VJP +complete-step/moment/clock/resume GPU admission; currentH4 qualification does not coverH8/newwidth",
            "freeze source/data/recipe/budget/runtime and resource review; no hidden initialization adjustment",
        ],
        "final_quality_claim_requires": [
            "proper independent Dense/CST LR tuning at common budgets",
            "paired3seeds and frozen family/margins before fresh final evaluation",
            "fixedc width validation4096, later8192 only separately approved",
        ],
        "recipe_selection_evidence_sha256": "eefdef7e8b7598232c137a6f8589c74f1a511051d13bcbd8fe0b5a9a01dc8e84",
        "pre_GPU_user_steering": "tentative LN20-case idea superseded before execution; main NONE16cases, LN only separate later conditional diagnostic with matchedDense, never automaticfallback",
        "qualification_cotangent_scales": [1.0, 1e24],
        "pre_GPU_cotangent_decision_sha256": "b86e9d69f211768239aaeda30b76863d16519f1bb13ee27ce0a25b30f79eeb31",
        "amplified_cotangent_used_for_optimizer": False,
        "raw_evidence_budget_bytes": 8589934592,
        "storage_plan": "retain all compressed SHA-bound raw locally + immutable remote source; per-case owned temporary extraction/admission, no permanent whole decompressed duplicate; freshfree minus conservativecompressed/maxscratch/primaryreserve mustleave>=8GiB; rootseals actualvalues",
        "stage_A": {
            "entrypoint": "cst_research.experiments.higgs_linear_atom_driver",
            "scope": "new16 actualwidth/depth/count native qualification only; no400k task training",
            "requires_native_GPU": True,
            "normal_elementwise": "abs(error)<=3e-5+3e-4*abs(reference), mandatory",
            "relative_each_gradient_and_each_packed_column": "relativeL2<=3e-4 AND maxerror/refmax<=3e-4; FP64 normalized norms; exactzero reference requires exactzeroactual",
            "amplified_componentwise": "record every failure; diagnostic, never relabel as passed",
            "material_critical_fields": "dy1e24 dX/everypackedcolumn: any abs(reference)>3e-5+3e-4*abs(reference) AND anyactualnonzero; allnormalFP32rounded-zero tails reported",
            "independent_bridge": "48-point polynomial quadrature actual31atoms perN/c, fullY/dX/all31packedP; first31actualP fingerprint shared acrossdepth only if exact",
            "optimizer": "fresh synthetic32inputs/labels/8B4orders retained,8normalCE and4resume; NO amplifiedgradient optimizer step; fullmodel/grad/m/v/clock/nativeownerstore bytes",
            "load": "only exactdeclared actual3/7CST owners max(before+1,saved); allothercorebytesliteral; actualPOSTLOAD baseline",
            "outputs": "rawinitial0/twoactual+expectedVJPs/syntheticinputs/normal4/normal8/resumed8/quadrature/scalar/proof+source/runtime/dataidentities",
            "Stage_B": "separate reviewedsource must preserve allStageAfiles and bind actualpassedraw16/source/runtime/data; taskstarts fresh0, neverstate8",
        },
        "training_entrypoint_implemented": False,
        "protocol_inspection_GPU_operations": False,
        "confirmation_or_test_payload_staged_or_scored": False,
        "formal_claim": False,
    }


def pilot_cases():
    """Fixed sixteen-case NONE screen; no density/LR/normalization CLI expansion."""
    result = []
    for width in (1024, 2048):
        for hidden_layers in (4, 8):
            result.append(
                {
                    "id": f"N{width}-H{hidden_layers}-dense-stock",
                    "width": width,
                    "hidden_layers": hidden_layers,
                    "method": "dense",
                    "c": None,
                }
            )
            for c in (1, 4, 16):
                result.append(
                    {
                        "id": f"N{width}-H{hidden_layers}-cst-c{c}",
                        "width": width,
                        "hidden_layers": hidden_layers,
                        "method": "cst",
                        "c": c,
                    }
                )
    return result


def make_case(case):
    if case not in pilot_cases():
        raise ValueError("case is outside fixed16 NONE pilot")
    return make_model(
        method=case["method"],
        width=case["width"],
        hidden_layers=case["hidden_layers"],
        c=case["c"],
        normalization="none",
        ordinary_initialization=ORDINARY_INIT,
        seed=17,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(protocol(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
