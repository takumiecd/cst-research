"""Read-only per-Atom fraction/radius update diagnostics, outside update timers."""

import torch
from torchcst import CSTLinear
from torchcst._backends.torch.parameterizations.radius_envelope import from_state


def stats(value):
    x = value.detach().cpu().double().flatten()
    return {
        "mean_signed": float(x.mean()),
        "mean_absolute": float(x.abs().mean()),
        "rms": float(x.square().mean().sqrt()),
        "p90_absolute": float(torch.quantile(x.abs(), 0.9)),
        "max_absolute": float(x.abs().max()),
    }


class FractionUpdates:
    @torch.no_grad()
    def __init__(self, model):
        self.layers = {
            n: m for n, m in model.named_modules() if isinstance(m, CSTLinear)
        }
        if getattr(model.config, "radius_mode", None) != "envelope" or not self.layers:
            raise ValueError("fraction diagnostics require explicit envelope sites")
        self.initial_q = {
            n: m.atoms.p[:, 1].detach().cpu().double().clone()
            for n, m in self.layers.items()
        }
        self.previous = {n: q.clone() for n, q in self.initial_q.items()}
        self.last_delta = {n: torch.zeros_like(q) for n, q in self.initial_q.items()}
        self.path = {n: torch.zeros_like(q) for n, q in self.initial_q.items()}
        self.initial_r = {
            n: from_state(m.kernel, m.atoms.p)[:, 1].detach().cpu().double()
            for n, m in self.layers.items()
        }
        self.steps = 0

    @torch.no_grad()
    def update(self):
        for name, layer in self.layers.items():
            q = layer.atoms.p[:, 1].detach().cpu().double()
            delta = q - self.previous[name]
            self.path[name].add_(delta.abs())
            self.last_delta[name] = delta
            self.previous[name] = q.clone()
        self.steps += 1

    @torch.no_grad()
    def report(self, optimizer):
        rates = optimizer.get_field_lrs()
        groups = {
            id(p): (i, g)
            for i, g in enumerate(optimizer.param_groups)
            for p in g["params"]
        }
        result = {}
        for name, layer in self.layers.items():
            p = layer.atoms.p
            q = p[:, 1].detach().cpu().double()
            current_r = from_state(layer.kernel, p)[:, 1].detach().cpu().double()
            counter = p.detach().clone()
            counter[:, 1].copy_(self.initial_q[name].to(counter))
            amplitude_r = (
                from_state(layer.kernel, counter)[:, 1].detach().cpu().double()
            )
            f_component = current_r - amplitude_r
            a_component = amplitude_r - self.initial_r[name]
            torch.testing.assert_close(
                current_r - self.initial_r[name],
                f_component + a_component,
                atol=0,
                rtol=0,
            )
            dq = q - self.initial_q[name]
            row = {
                "last_fraction_delta": stats(self.last_delta[name]),
                "net_fraction_delta": stats(dq),
                "fraction_path_length": stats(self.path[name]),
                "fraction_increased": float((dq > 0).double().mean()),
                "fraction_decreased": float((dq < 0).double().mean()),
                "fraction_unchanged": float((dq == 0).double().mean()),
                "at_zero": float((q == 0).double().mean()),
                "at_one": float((q == 1).double().mean()),
                "net_radius_delta": stats(current_r - self.initial_r[name]),
                "radius_fraction_component": stats(f_component),
                "radius_amplitude_component": stats(a_component),
            }
            index, group = groups[id(p)]
            state = optimizer.state.get(p, {})
            row["effective_fraction_lr"] = rates[index]["radius_fraction"]
            if "exp_avg" in state:
                step = int(state["step"])
                beta1, beta2 = group["betas"]
                m = state["exp_avg"][:, 1].detach().cpu().double() / (1 - beta1**step)
                v = state["exp_avg_sq"][:, 1].detach().cpu().double() / (
                    1 - beta2**step
                )
                proposal = -row["effective_fraction_lr"] * m / (v.sqrt() + group["eps"])
                row["analytic_adam_fraction_delta"] = stats(proposal)
                row["outward_momentum_at_boundary"] = float(
                    (((q == 0) & (proposal < 0)) | ((q == 1) & (proposal > 0)))
                    .double()
                    .mean()
                )
            else:
                row["analytic_adam_fraction_delta"] = None
            result[name] = row
        return {
            "steps_observed": self.steps,
            "layers": result,
            "scope": "Actual per-Atom post-projection updates; path sums every update. Radius net decomposes exactly at current amplitude and initial fraction. Adam delta is a FP64 analytic estimate from stored FP32 moments before bounds/rounding, not an exact raw-proposal trace. Diagnostics do not modify model, gradients, optimizer or RNG.",
        }
