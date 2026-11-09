"""Load separately verified jobs without rewriting their execution receipts."""

import hashlib
import json


def load_inputs(root, recovery=None):
    rows, directories = [], {}
    for index, job in enumerate([root] + ([recovery] if recovery else [])):
        integrity = json.loads((job / "retrieval-integrity.json").read_text())
        assert integrity["state"] == "passed"
        data = job / "retrieved"
        receipt = json.loads((data / "receipt.json").read_text())
        audit = json.loads((data / "trained-audit.json").read_text())
        assert audit["state"] == "passed"
        if receipt["status"] != "passed":
            excluded = audit["excluded_cli_rejection"]
            assert recovery and receipt["status"] == "failed"
            assert excluded["source_unchanged_reverified"]
            assert (
                excluded["original_receipt_sha256"]
                == hashlib.sha256((data / "receipt.json").read_bytes()).hexdigest()
            )
        for row in receipt["results"]:
            label = row["label"]
            public_label = f"recovery-{label}" if index else label
            rows.append(
                {
                    **row,
                    "label": public_label,
                    "origin_label": label,
                    "origin_job": job.name,
                }
            )
            directories[public_label] = data / label
    return rows, directories
