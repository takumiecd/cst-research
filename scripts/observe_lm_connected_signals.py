"""Read-only initial/final block signal observer for completed short LM arms."""

import argparse
import json
from pathlib import Path

import torch

from cst_research.experiments.ffn_model import parameter_digest
from cst_research.experiments.ffn_run import sha, token_batch, verify_data
from cst_research.experiments.ffn_selective import mechanism_probe
from cst_research.experiments.lm_connected import ARMS, model_for


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((a.run / "receipt.json").read_text())
    assert parent["state"] == "passed" and parent["source_unchanged"]
    data_root = Path(parent["protocol"]["data"])
    assert (
        sha(data_root / "manifest.json") == parent["protocol"]["data_manifest_sha256"]
    )
    _, data = verify_data(data_root)
    q = json.loads((a.run / "qualification/qualification.json").read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "state": "running",
        "scope": "read-only copies, original TRAIN labels, no optimizer or training updates",
        "script_sha256": sha(Path(__file__)),
        "parent_source_commits": parent["protocol"]["source_commits"],
        "data_manifest_sha256": parent["protocol"]["data_manifest_sha256"],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "records": [],
    }
    for arm in ARMS:
        for clock in (0, 976):
            m = model_for(arm)
            if clock:
                checkpoint = a.run / arm / "checkpoint-976.pt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                m.load_state_dict(saved["model"])
                del saved
                expected = json.loads((a.run / arm / "summary.json").read_text())[
                    "final_parameters_sha256"
                ]
            else:
                expected = q["arms"][arm]["initial_parameters_sha256"]
            assert parameter_digest(m) == expected
            before = {k: v.detach().clone() for k, v in m.named_buffers()}
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            for offset in (0, 4096, 8192, 12288):
                x, y = token_batch(data["train"], [offset], 128, "cuda")
                probe = mechanism_probe(m, x, y)
                result["records"].append(
                    {
                        "arm": arm,
                        "step": clock,
                        "train_offset": offset,
                        "parameter_sha256": expected,
                        "probe": probe,
                    }
                )
            assert parameter_digest(m) == expected
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(
                cuda_rng, torch.cuda.get_rng_state()
            )
            now = dict(m.named_buffers())
            assert before.keys() == now.keys()
            assert all(torch.equal(v, now[k]) for k, v in before.items())
            (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print("SIGNALS", arm, clock, "PASS", flush=True)
            del m
            torch.cuda.empty_cache()
    result["state"] = "passed"
    result["parameters_buffers_rng_unchanged"] = True
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
