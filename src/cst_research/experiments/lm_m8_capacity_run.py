"""One fresh c16 arm; source-qualified c8/Dense/DST controls are not rerun."""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import torch

from .ffn_model import parameter_digest
from .ffn_radius_fraction_qualification import assert_model_resume, assert_nested
from .ffn_run import sha, token_batch, verify_data
from .ffn_selective import active_ffn_scalars, projections
from .lm_connected import step
from .lm_geometry_trial_run import audit, full_oracle, train, write
from .lm_m8_capacity import model_for, optimizer_for
from .lm_placement_probe import probe


def prior(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(sha(root / n) == h for n, h in manifest["files"].items())
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    assert json.loads((root / "audit/audit.json").read_text())["state"] == "passed"
    assert (
        receipt["protocol"]["source_commits"]["cst-research"]
        == "76e80b6816269f63b02ff85ad262746f14754726"
    )
    return json.loads((root / "qualification/qualification.json").read_text())


def ordinary(model):
    return {
        n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
        for n, p in model.named_parameters()
        if not (".ffn." in n and ".map." in n)
    }


def qualify(output, data, previous):
    old = prior(previous)
    base = model_for("m8-c8", device="cpu")
    assert (
        parameter_digest(base) == old["arms"]["legacy-m8"]["initial_parameters_sha256"]
    )
    shared = ordinary(base)
    del base
    model = model_for("m8-c16")
    assert ordinary(model) == shared
    digest = parameter_digest(model)
    assert digest == old["arms"]["m16-c16-e8"]["initial_parameters_sha256"]
    assert active_ffn_scalars(model) == old["arms"]["dst-c16"]["active_ffn_scalars"]
    result = {
        "state": "running",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "prior_manifest_sha256": sha(previous / "manifest.json"),
        "m8_c8_initial_parameters_match": True,
        "ordinary_parameters_match": True,
        "m8_c16_raw_initial_parameters_match_prior_m16_c16": True,
        "arms": {
            "m8-c16": {
                "initial_parameters_sha256": digest,
                "active_ffn_scalars": active_ffn_scalars(model),
            }
        },
    }
    row = result["arms"]["m8-c16"]
    row["oracles"] = {
        s: full_oracle(projections(model)[s].map) for s in ("1.up", "1.down")
    }
    write(output / "qualification.json", result)
    assert all(v["passed"] for v in row["oracles"].values())
    opt = optimizer_for(model)
    assert [g["eps"] for g in opt.param_groups] == [1e-8, 1e-8]
    batches = [token_batch(data["train"], [0, 128], 128, "cuda")]
    step(model, opt, batches, 1)
    stream = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    other = model_for("m8-c16")
    other.load_state_dict(saved["model"])
    oo = optimizer_for(other)
    oo.load_state_dict(copy.deepcopy(saved["optimizer"]))
    probe(model, *token_batch(data["train"], [0], 128, "cuda"), retain=True)
    step(model, opt, batches, 2)
    step(other, oo, batches, 2)
    assert_model_resume(model, other)
    assert_nested(opt.state_dict(), oo.state_dict())
    row["serialized_resume_and_probe_next_update_exact"] = True
    result["state"] = "passed"
    write(output / "qualification.json", result)
    print("QUALIFIED m8-c16", flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("qualify", "train", "audit"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--data-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous-run", type=Path, required=True)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--run", type=Path)
    a = p.parse_args()
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 3090"
    assert torch.cuda.get_device_capability() == (8, 6)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha(a.data / "manifest.json") == a.data_manifest_sha256
    _, data = verify_data(a.data)
    a.output.mkdir(parents=True, exist_ok=False)
    if a.mode == "qualify":
        qualify(a.output, data, a.previous_run)
    elif a.mode == "audit":
        audit(
            a.run,
            a.output,
            data,
            arm_names=("m8-c16",),
            model_builder=model_for,
            optimizer_builder=optimizer_for,
        )
    else:
        q = json.loads(a.qualification.read_text())
        assert q["state"] == "passed" and q["torch"] == torch.__version__
        prior(a.previous_run)
        train(
            "m8-c16",
            a.output,
            data,
            a.data_manifest_sha256,
            q,
            model_builder=model_for,
            optimizer_builder=optimizer_for,
            declaration=("cst", 16, 1e-8, "legacy-m8"),
        )


if __name__ == "__main__":
    main()
