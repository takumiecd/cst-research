"""Evaluate learned FFN-branch dependence on the task; no training updates."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from cst_research.experiments.ffn_model import parameter_digest
from cst_research.experiments.ffn_run import sha, verify_data
from cst_research.experiments.ffn_selective_run import development
from cst_research.experiments.lm_placement import ARMS, STEPS, model_for


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((a.run / "receipt.json").read_text())
    assert receipt["state"] == "passed" and receipt["source_unchanged"]
    assert json.loads((a.run / "audit/audit.json").read_text())["state"] == "passed"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090"
    protocol = receipt["protocol"]
    data_root = Path(protocol["data"])
    assert sha(data_root / "manifest.json") == protocol["data_manifest_sha256"]
    _, data = verify_data(data_root)
    result = {
        "state": "running",
        "source_commits": protocol["source_commits"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_manifest_sha256": protocol["data_manifest_sha256"],
        "scope": "post-hoc task DEV dependence under zeroing one trained FFN residual branch; no training updates, not additive attribution or an optimal-model claim",
        "arms": {},
    }
    for arm, (kind, sites) in ARMS.items():
        m = model_for(arm)
        summary = json.loads((a.run / arm / "summary.json").read_text())
        path = a.run / arm / f"checkpoint-{STEPS}.pt"
        assert sha(path) == summary["checkpoints"][path.name]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(saved["model"])
        del saved
        expected = summary["final_parameters_sha256"]
        assert parameter_digest(m) == expected
        buffers = {k: v.detach().clone() for k, v in m.named_buffers()}
        cpu_rng, gpu_rng = (
            torch.get_rng_state().clone(),
            torch.cuda.get_rng_state().clone(),
        )
        curves = [
            json.loads(v)
            for v in (a.run / arm / "curves.jsonl").read_text().splitlines()
        ]
        expected_dev = next(v["development"] for v in curves if v["step"] == STEPS)
        baseline = development(m, data["development"])
        assert abs(baseline["ce"] - expected_dev["ce"]) <= 1e-6
        rows = []
        for index, block in enumerate(m.blocks):
            handle = block.ffn.register_forward_hook(
                lambda module, args, output: torch.zeros_like(output)
            )
            try:
                dev = development(m, data["development"])
            finally:
                handle.remove()
            row = {
                "block": index,
                "branch_kind": "".join(
                    ("C" if kind == "cst" else "S")
                    if f"{index}.{role}" in sites
                    else "D"
                    for role in ("up", "down")
                ),
                "development": dev,
                "delta_ce": dev["ce"] - baseline["ce"],
                "delta_correct_next_tokens": dev["correct_next_tokens"]
                - baseline["correct_next_tokens"],
            }
            rows.append(row)
        restored = development(m, data["development"])
        assert restored == baseline
        assert parameter_digest(m) == expected
        now = dict(m.named_buffers())
        assert buffers.keys() == now.keys() and all(
            torch.equal(v, now[k]) for k, v in buffers.items()
        )
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(
            gpu_rng, torch.cuda.get_rng_state()
        )
        result["arms"][arm] = {
            "baseline": baseline,
            "zeroed_ffn_branches": rows,
            "restored_exact": True,
            "parameters_buffers_rng_unchanged": True,
        }
        (a.output / "ablation.json").write_text(json.dumps(result, indent=2) + "\n")
        print("ABLATION", arm, [round(r["delta_ce"], 6) for r in rows], flush=True)
        del m
        torch.cuda.empty_cache()
    result["state"] = "passed"
    (a.output / "ablation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
