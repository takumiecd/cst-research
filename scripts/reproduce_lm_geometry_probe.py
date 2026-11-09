"""Synchronized, exact qualification-prefix reproduction; no performance claim."""

import argparse
import copy
import io
import json
import os
from pathlib import Path

import torch

from cst_research.experiments.ffn_radius_fraction_qualification import (
    assert_model_resume,
    assert_nested,
)
from cst_research.experiments.ffn_run import sha, token_batch, verify_data
from cst_research.experiments.ffn_selective import projections
from cst_research.experiments.lm_connected import step
from cst_research.experiments.lm_geometry_trial import model_for, optimizer_for
from cst_research.experiments.lm_geometry_trial_run import full_oracle
from cst_research.experiments.lm_placement_probe import probe


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--data-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 3090"
    assert torch.cuda.get_device_capability() == (8, 6)
    assert sha(a.data / "manifest.json") == a.data_manifest_sha256
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {
        "state": "running",
        "stages": [],
        "synchronized_debug_only": True,
        "script_sha256": sha(Path(__file__)),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }

    def record(stage):
        torch.cuda.synchronize()
        report["stages"].append(stage)
        (a.output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PASSED", stage, flush=True)

    try:
        _, data = verify_data(a.data)
        m = model_for("m16-c8-e8")
        o = optimizer_for(m)
        record("construction")
        for s in ("1.up", "1.down"):
            result = full_oracle(projections(m)[s].map)
            report.setdefault("oracles", {})[s] = result
            record("oracle-finished-" + s)
            assert result["passed"]
        batch = [token_batch(data["train"], [0, 128], 128, "cuda")]
        step(m, o, batch, 1)
        record("first-256-row-update")
        stream = io.BytesIO()
        torch.save({"model": m.state_dict(), "optimizer": o.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        other = model_for("m16-c8-e8")
        other.load_state_dict(saved["model"])
        oo = optimizer_for(other)
        oo.load_state_dict(copy.deepcopy(saved["optimizer"]))
        torch.save(saved, a.output / "before-probe.pt")
        report["before_probe_sha256"] = sha(a.output / "before-probe.pt")
        record("serialized-other-restored")
        probe(m, *token_batch(data["train"], [0], 128, "cuda"), retain=True)
        record("original-128-row-probe")
        step(m, o, batch, 2)
        record("original-second-update")
        step(other, oo, batch, 2)
        record("restored-second-update")
        assert_model_resume(m, other)
        assert_nested(o.state_dict(), oo.state_dict())
        record("next-update-exact")
        report["state"] = "passed"
    except BaseException as e:
        report.update(state="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (a.output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
