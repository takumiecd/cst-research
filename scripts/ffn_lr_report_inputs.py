"""Load separately verified jobs without rewriting their execution receipts."""

import hashlib
import json


def load_inputs(root, recovery=None, dst_run=None):
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
    if any(r.get("long_horizon_comparison") for r in rows):
        assert recovery is None and dst_run is None
        assert len(rows) == 3 and {r["kind"] for r in rows} == {"dense", "dst", "cst"}
        assert all(r.get("long_horizon_comparison") for r in rows)
        assert {r["updates"] for r in rows} == {19531}
        assert {r["seed"] for r in rows} == {17}
        base = root / "retrieved"
        da = json.loads((base / "dst-audit/audit.json").read_text())
        ca = json.loads((base / "trained-audit.json").read_text())
        assert da["state"] == ca["state"] == "passed"
        assert len(da["models"]) == 5 and len(ca["models"]) == 10
        summaries = {
            r["kind"]: json.loads(
                (directories[r["label"]] / "summary.json").read_text()
            )
            for r in rows
        }
        assert len({s["data_order_sha256"] for s in summaries.values()}) == 1
        assert all(s["completed_tokens"] == 19999744 for s in summaries.values())
        for key in (
            "width",
            "layers",
            "heads",
            "context",
            "vocabulary",
            "expansion",
            "attention_kind",
            "feature_order",
        ):
            assert len({s["config"][key] for s in summaries.values()}) == 1
        assert (
            summaries["dst"]["active_ffn_scalar_slots_including_bias"]
            == summaries["cst"]["actual_ffn_parameters"]
            == 272384
        )
        assert summaries["dst"]["actual_adam_updates"] == 19473
        assert len(summaries["dst"]["topology_events"]) == 58
    if dst_run is not None:
        base = root / "retrieved"
        data = dst_run / "retrieved"
        assert (
            json.loads((dst_run / "retrieval-integrity.json").read_text())["state"]
            == "passed"
        )
        reference = json.loads((data / "receipt.json").read_text())
        primary = json.loads((base / "receipt.json").read_text())
        assert reference["status"] == "passed" and reference["source_unchanged"]
        assert all(primary[k] == reference[k] for k in ("gpu", "torch", "cuda"))
        assert (
            primary["protocol"]["data_manifest_sha256"]
            == reference["protocol"]["data_manifest_sha256"]
        )
        a = json.loads((base / "qualification/qualification.json").read_text())
        b = json.loads((data / "qualification/qualification.json").read_text())
        assert all(
            a["source_sha256"][n] == b["source_sha256"][n]
            for n in ("ffn_model.py", "ffn_contract.py", "ffn_run.py")
        )
        selected = min(
            (r for r in reference["results"] if r["kind"] == "dst"),
            key=lambda r: (r["development_ce"], r["sparse_peak_lr"]),
        )
        assert selected["label"] == reference["selection"]["label"]
        original = selected["label"]
        selected = {
            **selected,
            "label": "dst-" + original,
            "origin_label": original,
            "origin_job": dst_run.name,
            "development_selected": True,
        }
        exemplar = next(r for r in rows if r["kind"] == "cst")
        assert (
            selected["seed"] == exemplar["seed"]
            and selected["updates"] == exemplar["updates"]
        )
        left = json.loads((directories[exemplar["label"]] / "summary.json").read_text())
        right = json.loads((data / original / "summary.json").read_text())
        assert all(
            left["config"][n] == right["config"][n]
            for n in (
                "width",
                "layers",
                "heads",
                "context",
                "vocabulary",
                "expansion",
                "attention_kind",
                "feature_order",
            )
        )
        assert (
            right["active_ffn_scalar_slots_including_bias"]
            == left["actual_ffn_parameters"]
        )
        rows.append(selected)
        directories[selected["label"]] = data / original
    return rows, directories
