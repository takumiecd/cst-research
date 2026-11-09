"""Post-training DEV token accuracy, without changing frozen learning/selection.

Use the frozen experiment source on PYTHONPATH. This supplementary evaluator is
separately hash-identified; it does not replace the original CE primary metric.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from cst_research.experiments import ffn_dst
from cst_research.experiments.ffn_contract import ModelConfig
from cst_research.experiments.ffn_model import LanguageModel, configure_native
from cst_research.experiments.ffn_run import sha, token_batch, verify_data


@torch.no_grad()
def metrics(model, tokens):
    model.eval()
    length = model.config.context
    windows = (len(tokens) - 1) // length
    total, count, correct = 0.0, 0, 0
    for j in range(0, windows, 2):
        starts = [k * length for k in range(j, min(j + 2, windows))]
        x, y = token_batch(tokens, starts, length, "cuda")
        logits, loss = model(x, y)
        total += float(loss) * y.numel()
        correct += int((logits.argmax(dim=-1) == y).sum())
        count += y.numel()
    ce = total / count
    return {
        "ce": ce,
        "perplexity": math.exp(ce),
        "tokens": count,
        "correct_next_tokens": correct,
        "next_token_top1_accuracy": correct / count,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    root = args.run.resolve()
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["status"] == "passed" and receipt["audits_passed"]
    assert receipt["gpu"] == torch.cuda.get_device_name(0)
    assert receipt["torch"] == torch.__version__
    assert all(
        sha(root / "source" / n) == h
        for n, h in json.loads((root / "source-files.json").read_text()).items()
    )
    protocol = receipt["protocol"]
    data_path = Path(protocol["data"])
    assert sha(data_path / "manifest.json") == protocol["data_manifest_sha256"]
    _, data = verify_data(data_path)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "state": "running",
        "scope": "Post-hoc supplementary teacher-forced next-token top1 on the same8192 DEV targets; no confirmation or selection change, no free-generation quality claim",
        "evaluator_sha256": sha(Path(__file__)),
        "source_archive_sha256": protocol["source_archive_sha256"],
        "data_manifest_sha256": protocol["data_manifest_sha256"],
        "gpu": receipt["gpu"],
        "torch": receipt["torch"],
        "models": [],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for row in receipt["results"]:
            directory = root / row["label"]
            summary = json.loads((directory / "summary.json").read_text())
            curves = [
                json.loads(x)
                for x in (directory / "curves.jsonl").read_text().splitlines()
            ]
            for name, digest in summary["milestone_checkpoint_sha256"].items():
                checkpoint = directory / name
                assert sha(checkpoint) == digest
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                config = ModelConfig(**saved["config"])
                model = (
                    ffn_dst.make_model(config, saved["seed"])
                    if row["kind"] == "dst"
                    else LanguageModel(config, row["kind"], saved["seed"])
                ).cuda()
                model.load_state_dict(saved["model"])
                if row["kind"] == "cst":
                    configure_native(model)
                observed = metrics(model, data["development"])
                expected = next(
                    c["development"]["ce"]
                    for c in curves
                    if c["step"] == saved["updates"] and "development" in c
                )
                assert (
                    observed["tokens"] == 8192
                    and abs(observed["ce"] - expected) <= 1e-6
                )
                result["models"].append(
                    {
                        "kind": row["kind"],
                        "step": saved["updates"],
                        "completed_tokens": saved["completed_tokens"],
                        "checkpoint_sha256": digest,
                        "recorded_ce": expected,
                        "ce_absolute_difference": abs(observed["ce"] - expected),
                        **observed,
                    }
                )
                print(json.dumps(result["models"][-1]), flush=True)
                save()
                del model, saved
                torch.cuda.empty_cache()
        assert len(result["models"]) == 15
        result["state"] = "passed"
        save()
    except BaseException as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
