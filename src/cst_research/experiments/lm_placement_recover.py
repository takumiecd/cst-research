"""Recover completed trajectories without rewriting the failed original evidence."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from .ffn_model import parameter_digest
from .ffn_run import sha
from .lm_placement import STEPS, model_for


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--revalidation", type=Path, required=True)
    a = p.parse_args()
    accepted = json.loads(a.revalidation.read_text())
    assert accepted["state"] == "passed" and accepted["independent_witness"]["passed"]
    old_manifest = json.loads((a.old / "manifest.json").read_text())
    result = {
        "state": "running",
        "old_job": a.old.name,
        "scope": "reuse completed trajectories; original component-gate failure retained, no extra training updates",
        "source_file_hashes": {},
        "arms": {},
    }
    for arm in ("dense", "dst-pair12", "pair12"):
        source = a.old / arm
        target = a.new / arm
        target.mkdir(exist_ok=False)
        summary = json.loads((source / "summary.json").read_text())
        assert summary["completed_steps"] == STEPS
        for path in source.iterdir():
            if path.suffix in (".json", ".jsonl"):
                h = sha(path)
                assert old_manifest["files"][str(path.relative_to(a.old))] == h
                shutil.copyfile(path, target / path.name)
                result["source_file_hashes"][str(path.relative_to(a.old))] = h
            elif path.suffix == ".pt":
                (target / path.name).symlink_to(path.resolve())
        for name, h in summary["checkpoints"].items():
            assert sha(target / name) == h
        if arm == "pair12":
            shutil.copyfile(target / "summary.json", target / "original-summary.json")
            saved = torch.load(
                source / f"checkpoint-{STEPS}.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert saved["step"] == STEPS and saved["arm"] == arm
            m = model_for(arm, "cpu")
            m.load_state_dict(saved["model"])
            assert summary["actual_adam_updates"] == STEPS
            for state in saved["optimizer"]["state"].values():
                assert int(state["step"]) == STEPS
                assert all(
                    state[k].dtype == torch.float32 and torch.isfinite(state[k]).all()
                    for k in ("exp_avg", "exp_avg_sq")
                )
            summary.update(
                state="completed",
                completed_tokens=STEPS * 1024,
                final_parameters_sha256=parameter_digest(m),
                final_oracles=accepted["arms"][arm],
                completion_resolution="post-failure per-field FP32 numerical admission; original-summary.json and parent failure preserved; no extra updates",
                original_componentwise_oracle_passed=False,
            )
            (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            del m, saved
        result["arms"][arm] = {
            "original_state": json.loads((source / "summary.json").read_text())[
                "state"
            ],
            "derived_state": summary["state"],
            "completed_steps": STEPS,
        }
    (a.new / "qualification").mkdir(exist_ok=False)
    shutil.copyfile(
        a.old / "qualification/qualification.json",
        a.new / "qualification/qualification.json",
    )
    result.update(
        state="passed",
        qualification_sha256=sha(a.new / "qualification/qualification.json"),
        numerical_revalidation_sha256=sha(a.revalidation),
    )
    (a.new / "trajectory-reuse.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
