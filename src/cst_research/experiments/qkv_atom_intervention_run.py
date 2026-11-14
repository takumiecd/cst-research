"""Collect TinyStories QKV loss responses from the frozen rational-L run."""

import argparse
import importlib.util
import json
import os
from contextlib import contextmanager
from itertools import combinations, product
from pathlib import Path

import torch

if __package__:
    from .atom_intervention_loss import collect_pair
else:
    from atom_intervention_loss import collect_pair


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def actions():
    """Signed singles, all signed two-field combinations, and joint moves."""
    sizes = (0.10, 0.10, 0.10, 0.10)
    yield (0.0,) * 4
    for count in (1, 2, 4):
        for fields in combinations(range(4), count):
            for signs in product((-1, 1), repeat=count):
                values = [0.0] * 4
                for field, sign in zip(fields, signs):
                    values[field] = sizes[field] * sign
                yield tuple(values)


@contextmanager
def transform_physical(projections, callback):
    """Intercept prepared rows at the site that owns them."""
    from torchcst._backends.torch.parameterizations import radius_envelope

    original = radius_envelope.prepared_physical_parameters
    sites = []
    handles = []

    def enter(_module, _args, name):
        sites.append(name)

    def leave(_module, _args, _output):
        sites.pop()

    def wrapped(rows, constants):
        return callback(sites[-1], original(rows, constants))

    try:
        for name, projection in projections.items():
            handles.append(
                projection.map.register_forward_pre_hook(
                    lambda module, inputs, name=name: enter(module, inputs, name)
                )
            )
            handles.append(
                projection.map.register_forward_hook(leave, always_call=True)
            )
        radius_envelope.prepared_physical_parameters = wrapped
        yield
    finally:
        radius_envelope.prepared_physical_parameters = original
        for handle in handles:
            handle.remove()


def load_reference(reference_probe):
    # The probe supplies the archived checkpoint, data and source verification.
    probe = load_script(reference_probe.resolve())
    train = load_script(probe.BASE / "train.py")
    record = probe.read(probe.BASE / probe.ARM / "result.json")
    assert record["state"] == "completed"
    assert record["arms"][probe.ARM]["checkpoint_audit"] == "passed"
    protocol = probe.verify_source(record)
    assert train.sha(probe.BASE / "train.py") == record["script_sha256"]
    data_path = Path(protocol["data"])
    assert train.sha(data_path / "manifest.json") == record["data_manifest_sha256"]
    _, data = train.verify_data(data_path)
    assert torch.cuda.get_device_name() == record["gpu"]
    assert torch.__version__ == record["torch"] and torch.version.cuda == record["cuda"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CST_RESEARCH_RADIUS_CAP"] = "512"
    from torchcst._backends.torch.parameterizations import radius_envelope

    model = train.model_for()
    assert train.parameter_digest(model) == record["initial_parameters_sha256"]
    radius_envelope.USE_RATIONAL_LOWER = True
    return probe, train, record, data, model.cuda().eval()


def run(args):
    probe, train, record, data, model = load_reference(args.reference_probe)
    projections = train.projections(model)
    params = {site: p.map.atoms.p for site, p in projections.items()}
    names = tuple(params)
    args.output.mkdir()

    def transform_rows(callback):
        return transform_physical(projections, callback)

    def loss(batch):
        return model(*batch)[1]

    manifest = {
        "state": "running",
        "parts": [],
        "gpu": record["gpu"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "data_manifest_sha256": record["data_manifest_sha256"],
        "runner_sha256": train.sha(Path(__file__)),
        "collector_sha256": train.sha(
            Path(__file__).with_name("atom_intervention_loss.py")
        ),
        "source_protocol_sha256": record["source_protocol_sha256"],
        "checkpoints": record["arms"][probe.ARM]["checkpoints"],
        "loss": "mean next-token CE",
    }
    path = args.output / "manifest.json"
    train.write(path, manifest)
    try:
        for step in (976, 4882):
            model.load_state_dict(probe.load_checkpoint(record, step)["model"])
            expected = record["arms"][probe.ARM]["evaluations"][str(step)][
                "development"
            ]["ce"]
            assert (
                abs(train.development(model, data["development"])["ce"] - expected)
                <= 1e-6
            )
            for pair in range(args.pairs):
                # Gradient and label windows use separate frozen story splits.
                starts = [pair * 258, pair * 258 + 129]
                a = train.token_batch(data["train"], starts, 128, "cuda")
                b = train.token_batch(data["development"], starts, 128, "cuda")
                payload = collect_pair(
                    loss=loss,
                    gradient_batch=a,
                    evaluation_batch=b,
                    atom_parameters=params,
                    transform_rows=transform_rows,
                    amplitude_edges=(0.0, 0.01, 0.03, 0.1, 0.3, 0.6, float("inf")),
                    group_size=args.group_size,
                    groups_per_band=1,
                    actions=actions(),
                    radius_limits=(0.5, 512.0),
                    center_periods=(1024, 1024),
                    amplitude_limit=32.0,
                    seed=17 + pair,
                )
                assert payload["sites"] == names
                payload["checkpoint_step"] = step
                payload["batch_pair"] = pair
                payload["batches"] = {
                    "gradient": {"split": "train", "starts": starts},
                    "evaluation": {"split": "development", "starts": starts},
                }
                part = args.output / f"step-{step}-pair-{pair}.pt"
                temporary = part.with_suffix(".tmp")
                torch.save(payload, temporary)
                temporary.replace(part)
                manifest["parts"].append(
                    {
                        "filename": part.name,
                        "sha256": train.sha(part),
                        "step": step,
                        "batch_pair": pair,
                        "groups": len(payload["groups"]),
                        "rows": len(payload["observations"]),
                    }
                )
                train.write(path, manifest)
                print(json.dumps(manifest["parts"][-1]), flush=True)
        manifest["state"] = "completed"
    except BaseException as exc:
        manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        train.write(path, manifest)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference-probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=1024)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 31 or args.group_size < 1:
        parser.error("require 1 to 31 batch pairs and a positive group size")
    run(args)


if __name__ == "__main__":
    main()
