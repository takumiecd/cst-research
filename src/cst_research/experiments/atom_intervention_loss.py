"""Collect paired task-loss responses to interventions on physical Atom parameters."""

from __future__ import annotations

import math


def collect_pair(
    *,
    loss,
    gradient_batch,
    evaluation_batch,
    atom_parameters,
    transform_rows,
    amplitude_edges,
    group_size,
    groups_per_band,
    actions,
    radius_limits,
    center_periods,
    amplitude_limit,
    seed,
):
    """Return sampled Atom features and group-level labels for one batch pair.

    loss(batch) must return a scalar Torch tensor with the model in deterministic
    evaluation mode. transform_rows(callback) is a context manager calling
    callback(site, physical_rows) at each selected site's forward. Atom
    parameters and physical rows both have one row per Atom in the same order.
    Physical columns are amplitude, radius, output center and input center.
    """
    import torch

    if gradient_batch is evaluation_batch:
        raise ValueError("use distinct feature and evaluation batches")
    lo, hi = radius_limits
    if not (0 < lo < hi < math.inf and group_size > 0 and groups_per_band > 0):
        raise ValueError(
            "positive group counts and finite ordered radius limits required"
        )
    edges = tuple(amplitude_edges)
    if (
        len(edges) < 2
        or edges[0] != 0
        or edges[-1] != math.inf
        or any(not a < b for a, b in zip(edges, edges[1:]))
    ):
        raise ValueError("amplitude edges must increase from zero to infinity")
    actions = tuple(tuple(action) for action in actions)
    if not actions or any(
        len(action) != 4 or not all(map(math.isfinite, action)) for action in actions
    ):
        raise ValueError("actions must be finite vectors of length four")
    periods = tuple(center_periods)
    if len(periods) != 2 or not all(0 < period < math.inf for period in periods):
        raise ValueError("two positive finite periods required")

    if not 0 < amplitude_limit <= math.inf:
        raise ValueError("positive amplitude limit required")

    names = tuple(atom_parameters)
    if not names:
        raise ValueError("select at least one Atom site")
    params = tuple(atom_parameters.values())
    captured = {}

    def capture(site, rows):
        if site in captured:
            raise ValueError(
                f"selected site called twice in the gradient batch: {site}"
            )
        captured[site] = rows
        return rows

    def scalar(batch):
        value = loss(batch)
        if value.ndim or not bool(torch.isfinite(value)):
            raise ValueError("loss must return a finite scalar")
        return value

    with torch.enable_grad(), transform_rows(capture):
        feature_loss = scalar(gradient_batch)
    if captured.keys() != atom_parameters.keys():
        raise ValueError(
            "gradient forward did not visit the selected Atom sites exactly"
        )
    physical = tuple(captured[name] for name in names)
    if any(
        x.ndim != 2 or x.shape[1] != 4 or len(x) != len(p)
        for p, x in zip(params, physical)
    ):
        raise ValueError(
            "each physical Atom table must have four fields and match its storage"
        )
    gradients = torch.autograd.grad(feature_loss, (*params, *physical))
    gradient_loss = float(feature_loss.detach())
    raw_grad, phys_grad = gradients[: len(names)], gradients[len(names) :]
    values = torch.cat([x.detach().cpu() for x in physical])
    grads = torch.cat([x.detach().cpu() for x in phys_grad])
    raw = torch.cat([x.detach().cpu() for x in params])
    raw_grads = torch.cat([x.detach().cpu() for x in raw_grad])
    captured.clear()
    del physical, gradients, feature_loss
    if not bool(
        torch.isfinite(values).all()
        and torch.isfinite(grads).all()
        and torch.isfinite(raw).all()
        and torch.isfinite(raw_grads).all()
    ):
        raise ValueError("non-finite Atom values or gradients")
    amplitude, radius = values[:, :2].unbind(1)
    if bool(((radius < lo) | (radius > hi)).any()):
        raise ValueError("baseline physical radii exceed the supplied limits")
    features = torch.stack(
        (
            amplitude.abs(),
            radius.log(),
            amplitude.sign() * grads[:, 0],
            radius * grads[:, 1],
            values[:, 2].remainder(periods[0]) / periods[0],
            values[:, 3].remainder(periods[1]) / periods[1],
            radius * grads[:, 2],
            radius * grads[:, 3],
        ),
        dim=1,
    )

    def change(rows, action):
        da, dr, dout, din = action
        result = rows.clone()
        if da:
            result[:, 0] = (rows[:, 0] * math.exp(da)).clamp(
                -amplitude_limit, amplitude_limit
            )
        if dr:
            result[:, 1] = (rows[:, 1] * math.exp(dr)).clamp(lo, hi)
        for col, delta, period in zip((2, 3), (dout, din), periods):
            if delta:
                result[:, col] = (rows[:, col] + delta * rows[:, 1]).remainder(period)
        return result

    offsets = [0]
    for p in params:
        offsets.append(offsets[-1] + len(p))
    generator = torch.Generator().manual_seed(seed)

    groups = []
    observations = []
    with torch.no_grad():
        baseline = float(scalar(evaluation_batch))
        for low, high in zip(edges, edges[1:]):
            eligible = (
                ((amplitude.abs() >= low) & (amplitude.abs() < high))
                .nonzero()
                .flatten()
            )
            if not eligible.numel():
                continue
            order = eligible[torch.randperm(len(eligible), generator=generator)]
            for indices in order[: group_size * groups_per_band].split(group_size):
                group = len(groups)
                by_site = {}
                for i, name in enumerate(names):
                    mask = (indices >= offsets[i]) & (indices < offsets[i + 1])
                    if mask.any():
                        by_site[name] = (indices[mask] - offsets[i]).to(
                            params[i].device
                        )
                group_rows = values[indices]
                groups.append(
                    {
                        "group": group,
                        "amplitude_band": [low, None if high == math.inf else high],
                        "atom_indices": indices,
                        "raw_values": raw[indices],
                        "raw_gradients": raw_grads[indices],
                        "physical_values": values[indices],
                        "physical_gradients": grads[indices],
                        "features": features[indices],
                        "actual_deltas": [],
                    }
                )
                for action in actions:
                    changed = change(group_rows, action).double()
                    original = group_rows.double()
                    actual = changed - original
                    actual[:, 1] = (changed[:, 1] / original[:, 1]).log()
                    for col, period in zip((2, 3), periods):
                        actual[:, col] = (actual[:, col] + period / 2).remainder(
                            period
                        ) - period / 2
                    groups[-1]["actual_deltas"].append(actual)

                    def intervene(site, rows):
                        if site not in by_site:
                            return rows
                        selected = by_site[site]
                        modified = rows.clone()
                        modified[selected] = change(rows[selected], action)
                        return modified

                    with transform_rows(intervene):
                        measured = float(scalar(evaluation_batch))
                    if not any(action) and not math.isclose(
                        measured, baseline, abs_tol=1e-7, rel_tol=0
                    ):
                        raise RuntimeError("zero intervention changed the loss")
                    observations.append(
                        {
                            "group": group,
                            "action": action,
                            "changed_atoms": int((actual != 0).any(dim=1).sum()),
                            "baseline_loss": baseline,
                            "intervention_loss": measured,
                            "delta_loss": measured - baseline,
                        }
                    )
        if not math.isclose(
            float(scalar(evaluation_batch)), baseline, abs_tol=1e-7, rel_tol=0
        ):
            raise RuntimeError("baseline loss changed after intervention restoration")
    return {
        "sites": names,
        "site_offsets": offsets,
        "feature_columns": (
            "abs_amplitude",
            "log_radius",
            "grad_abs_amplitude_at_fixed_radius",
            "grad_log_radius",
            "output_center_fraction",
            "input_center_fraction",
            "grad_output_center_in_radii",
            "grad_input_center_in_radii",
        ),
        "physical_columns": ("amplitude", "radius", "output_center", "input_center"),
        "actions": actions,
        "action_columns": (
            "log_amplitude_scale",
            "log_radius_scale",
            "output_shift_in_radii",
            "input_shift_in_radii",
        ),
        "actual_delta_columns": (
            "amplitude",
            "log_radius",
            "output_center",
            "input_center",
        ),
        "gradient_batch_loss": gradient_loss,
        "groups": groups,
        "observations": observations,
    }
