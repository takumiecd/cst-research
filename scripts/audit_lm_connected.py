"""Read-only final checkpoint identity and optimizer-clock audit."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from cst_research.experiments.ffn_model import parameter_digest
from cst_research.experiments.lm_connected import ARMS, model_for


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(4)
    report = {"state": "running", "script_sha256": sha(Path(__file__)), "arms": {}}
    for arm in ARMS:
        summary = json.loads((a.run / arm / "summary.json").read_text())
        assert summary["state"] == "completed"
        for name, h in summary["checkpoints"].items():
            assert sha(a.run / arm / name) == h
        saved = torch.load(
            a.run / arm / "checkpoint-976.pt", map_location="cpu", weights_only=False
        )
        assert saved["step"] == 976 and saved["config"] == summary["config"]
        m = model_for(arm, device="cpu")
        m.load_state_dict(saved["model"])
        digest = parameter_digest(m)
        assert digest == summary["final_parameters_sha256"]
        states = saved["optimizer"]["state"]
        clocks = sorted({int(v["step"]) for v in states.values() if "step" in v})
        assert clocks == ([973] if arm == "dst-all" else [976]), clocks
        for v in states.values():
            for k in ("exp_avg", "exp_avg_sq"):
                if k in v:
                    assert v[k].dtype == torch.float32 and torch.isfinite(v[k]).all()
        report["arms"][arm] = {
            "checkpoint_sha256": summary["checkpoints"],
            "final_parameter_digest_matches": True,
            "parameter_sha256": digest,
            "optimizer_clocks": clocks,
            "moment_dtype": "float32",
            "moments_finite": True,
        }
        print("AUDIT PASS", arm, flush=True)
        del saved, m
    report["state"] = "passed"
    a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
