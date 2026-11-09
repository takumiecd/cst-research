"""Read-only finite update vs declared JVP at recorded task-LM updates."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torchcst import CSTLinear, FixedSelector, make_algorithm_registry
from torchcst._backends.torch.parameterizations.radius_envelope import (
    physical_parameters,
)

from cst_research.experiments.lm_connected import physical_reference

FIELDS = ("amplitude", "radius_fraction", "output_center", "input_center", "all_fields")


def statistics(finite, linear, g):
    d, j, target = finite.double(), linear.double(), -g.double()
    dn, jn, tn = (float(v.norm()) for v in (d, j, target))

    def cosine(a, b, an, bn):
        return float((a * b).sum()) / (an * bn) if an > 0 and bn > 0 else None

    return {
        "finite_norm": dn,
        "prediction_norm": jn,
        "finite_over_prediction_norm": dn / jn if jn else None,
        "linear_finite_cosine": cosine(d, j, dn, jn),
        "relative_defect_to_prediction": float((d - j).norm()) / jn if jn else None,
        "finite_task_cosine": cosine(d, target, dn, tn),
        "predicted_task_cosine": cosine(j, target, jn, tn),
        "task_projection_finite": float((target * d).sum()),
        "task_projection_predicted": float((target * j).sum()),
    }


def displacement(old, new, ni, no):
    v = new.double() - old.double()
    for k, period in ((2, no), (3, ni)):
        v[:, k] = (v[:, k] + period / 2).remainder(period) - period / 2
    return v.to(old)


def candidate(old, new, v, field, epsilon, ni, no):
    point = old.clone()
    cols = range(4) if field == 4 else (field,)
    for k in cols:
        point[:, k] = new[:, k] if epsilon == 1 else old[:, k] + epsilon * v[:, k]
    point[:, 2].remainder_(no)
    point[:, 3].remainder_(ni)
    return point


def native_layer(operator, p):
    m = CSTLinear(
        *operator.charts,
        atoms=p.clone(),
        kernel=operator.kernel,
        dtype=torch.float32,
        backend="factored",
    ).cuda()
    registry = make_algorithm_registry()
    if operator.kernel.revision == 10:
        from torchcst._backends.cuda.algorithms.linear.connected_cell_mass_coalesced.envelope import (
            CONNECTED_ORDERED_ENVELOPE,
            ConnectedOrderedEnvelopeAlgorithm,
        )

        registry.register(ConnectedOrderedEnvelopeAlgorithm())
        plan = CONNECTED_ORDERED_ENVELOPE
    else:
        from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.radius_envelope import (
            RADIUS_ENVELOPE_WIDE_AMPLITUDE_LM,
            RadiusEnvelopeWideAmplitudeLMAlgorithm,
        )

        registry.register(RadiusEnvelopeWideAmplitudeLMAlgorithm())
        plan = RADIUS_ENVELOPE_WIDE_AMPLITUDE_LM
    m.selector = FixedSelector(plan, registry=registry)
    return m


def analyze(row):
    op = row["operator"]
    ni, no = op.in_features, op.out_features
    old, new = row["old_p"], row["new_p"]
    v = displacement(old, new, ni, no)
    reference = physical_reference(op)
    spec = op.kernel.parameterization
    settings = {
        k: getattr(spec, k)
        for k in ("radius_min", "radius_max", "tau", "amplitude_max")
    }
    x = row["x"].double()
    g = row["g"].double()
    module = native_layer(op, old)
    native_x = row["x"].cuda()
    with torch.no_grad():
        native_old = module(native_x).cpu().double()
    torch.testing.assert_close(
        native_old, row["old_y_native"].double(), atol=1e-5, rtol=1e-5
    )

    def action(p):
        physical = physical_parameters(p, **settings).double()
        vi, vo = reference.operator.factors(physical)
        return (x @ vi) @ vo.T

    # One chunk is a complete additive Atom contribution, not a teacher target.
    result = {
        "fields": {},
        "rows": len(x),
        "atoms": len(old),
        "input_sites": ni,
        "output_sites": no,
        "scope": "declared JVP at raw FP32 rows; independent FP64 factors after encoded physical map; actual task cotangent, bias fixed",
        "field_delta_rms": v.double().square().mean(0).sqrt().tolist(),
    }
    baseline = torch.zeros(len(x), no, dtype=torch.float64)
    for first in range(0, len(old), 128):
        baseline += action(old[first : first + 128]).detach()
    torch.testing.assert_close(native_old, baseline, atol=1e-4, rtol=4e-3)
    for field, name in enumerate(FIELDS):
        direction = v.clone()
        if field < 4:
            direction[:, [k for k in range(4) if k != field]] = 0
        linear = torch.zeros_like(baseline)
        for first in range(0, len(old), 128):
            p = old[first : first + 128].requires_grad_()
            _, j = torch.autograd.functional.jvp(
                action, p, direction[first : first + 128]
            )
            linear += j.detach()
        records = []
        for epsilon in (1.0, 0.1, 0.01):
            point = candidate(old, new, v, field, epsilon, ni, no)
            value = torch.zeros_like(baseline)
            for first in range(0, len(old), 128):
                value += action(point[first : first + 128]).detach()
            change = value - baseline
            r = {"epsilon": epsilon, **statistics(change, epsilon * linear, g)}
            if epsilon == 1:
                with torch.no_grad():
                    module.atoms.p.copy_(point.cuda())
                    finite = module(native_x).cpu().double() - native_old
                error = float((finite - change).norm())
                scale = float(change.norm())
                r["native_finite_vs_reference"] = {
                    "max_abs": float((finite - change).abs().max()),
                    "relative_l2": error / max(scale, 1e-30),
                }
                # Small updates can be dominated by FP32 output subtraction; retain both metrics.
                assert (
                    r["native_finite_vs_reference"]["max_abs"] <= 2e-4
                    or r["native_finite_vs_reference"]["relative_l2"] <= 0.01
                ), r
                r["native_actual"] = statistics(finite, linear, g)
            records.append(r)
        result["fields"][name] = records
        print("FIELD", name, records[0]["linear_finite_cosine"], flush=True)
    del module
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = {
        "state": "running",
        "records": [],
        "epsilons": [1, 0.1, 0.01],
        "fields": list(FIELDS),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for arm in ("legacy-m8", "connected-m16"):
        for path in sorted(
            (a.run / arm).glob("probe-*.pt"), key=lambda p: int(p.stem.split("-")[-1])
        ):
            raw = torch.load(path, map_location="cpu", weights_only=False)
            for site, row in raw["records"].items():
                entry = {
                    "arm": arm,
                    "step": raw["step"],
                    "site": site,
                    "probe_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    **analyze(row),
                }
                summary["records"].append(entry)
                (a.output / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
    summary["state"] = "passed"
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
