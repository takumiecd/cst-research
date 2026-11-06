"""StageB read-only admission of actual StageA evidence; no GPU or tensor imports."""

from __future__ import annotations

import re
from pathlib import Path

from . import higgs_depth_normalization_learning_evidence as exact
from . import higgs_linear_atom_scaling as model_contract
from .higgs_depth_data import DATA_SHA, MANIFEST_SHA
from .higgs_depth_study import SOURCE, SOURCE_MANIFEST
from .higgs_dst_comparison import file_sha, read

PARENT_COMMIT = "446f25f1ec339f1e3acb4cf84d96ac5825306ab7"
PARENT_MANIFEST = "717bcbcb88dde83fe619cad547bd9ffbc9092e35be5787fdcad7f1eda2859db4"
PARENT_PROTOCOL = "9a34b7fc5a30d58b47c36db25ed15b148dae7c347fbd4898beb0fda8e7bc7b28"
REPORT_STATUS = "passed-independent-StageA16-qualification-only-NOT-task-training"
PREFIX = "result/science/qualification"


def sealed_sha(path, expected):
    exact.require(
        isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected),
        "owner-sealed fullSHA required",
    )
    exact.require(
        path.is_file() and not path.is_symlink() and file_sha(path) == expected,
        "owner-sealed proof bytes differ",
    )


def strict_files(root, expected):
    """Consumer source is independently manifest-checked; all actual result files exact."""
    observed = {}
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        exact.require(
            not path.is_symlink() and (path.is_dir() or path.is_file()),
            "nonregular StageA evidence",
        )
        if name == "consumer" or name.startswith("consumer/"):
            continue
        if path.is_file():
            observed[name] = path
    exact.require(
        set(observed) == set(expected),
        "StageA result inventory differs; no extra sidecars in rawroot",
    )
    exact.verify_file_inventory(root, expected)
    return expected


def verify_stage_a(
    root, admission, admission_sha, release, release_sha, current_consumer, runtime
):
    """Real full raw hash pass before any torch.load; owner-pinned scientific audit separate."""
    root = Path(root).resolve(strict=True)
    sealed_sha(admission, admission_sha)
    sealed_sha(release, release_sha)
    report = read(admission)
    required = {
        "status": REPORT_STATUS,
        "consumer_commit": PARENT_COMMIT,
        "consumer_manifest_sha256": PARENT_MANIFEST,
        "protocol_sha256": PARENT_PROTOCOL,
        "external_source_commit": SOURCE,
        "external_source_manifest_sha256": SOURCE_MANIFEST,
        "runtime_pin_sha256": exact.RUNTIME_SHA,
        "runtime_fingerprint": runtime,
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "task_training_updates": 0,
        "all16_passed": True,
        "root_file_release_sha256": release_sha,
    }
    for name, value in required.items():
        exact.require(
            report.get(name) == value,
            "actual independent StageA report differs: " + name,
        )
    outer = exact.safe_member(root, "outer-receipt.json")
    sealed_sha(outer, report["parent_outer_receipt_sha256"])
    receipt = read(outer)
    exact.require(
        receipt["driver_exit_code"] == 0
        and type(receipt["driver_exit_code"]) is int
        and receipt["completed_summary_present"] is True
        and receipt["consumer_commit"] == PARENT_COMMIT
        and receipt["runtime_pin_sha256"] == exact.RUNTIME_SHA
        and receipt["automatic_retry"] is False,
        "StageA outer normal completed identity required",
    )
    expected = {
        **receipt["files"],
        "outer-receipt.json": {
            "sha256": report["parent_outer_receipt_sha256"],
            "bytes": outer.stat().st_size,
        },
    }
    exact.require(
        report["verified_files"] == expected,
        "independent report must bind entire actual receipt+outer inventory",
    )
    strict_files(root, expected)
    parent_manifest = exact.safe_member(root, "result/consumer-source-manifest.json")
    sealed_sha(parent_manifest, PARENT_MANIFEST)
    source = read(parent_manifest)
    exact.require(
        source["commit"] == PARENT_COMMIT and len(source["files"]) == 114,
        "original StageA114source inventory differs",
    )
    for name, digest in source["files"].items():
        exact.require(
            file_sha(exact.safe_member(current_consumer, name)) == digest,
            "StageB must preserve ALL114StageA files: " + name,
        )
    summary_file = exact.safe_member(root, "result/science/run-summary.json")
    summary = read(summary_file)
    exact.require(
        summary["status"]
        == "passed-all16-new-native-qualification-only-NOT-400k-training"
        and summary["task_training_updates"] == 0
        and summary["task_dataset_or_development_evaluated"] is False
        and summary["initial_state8_used_to_start_task_training"] is False
        and summary["amplified_used_for_optimizer"] is False
        and summary["normal_VJP_scale"] == 1
        and summary["amplified_VJP_scale"] == 1e24,
        "StageA actual16summary incomplete",
    )
    identity = summary["identity"]
    for name, value in {
        "consumer_commit": PARENT_COMMIT,
        "consumer_manifest_sha256": PARENT_MANIFEST,
        "protocol_sha256": PARENT_PROTOCOL,
        "runtime_fingerprint": runtime,
        "data_sha256": DATA_SHA,
        "data_manifest_sha256": MANIFEST_SHA,
        "external_source_commit": SOURCE,
        "external_source_manifest_sha256": SOURCE_MANIFEST,
    }.items():
        exact.require(
            identity[name] == value, "actual StageA producer identity differs: " + name
        )
    for name, digest in summary[
        "qualified_factory_reference_state_qualifier_source"
    ].items():
        exact.require(
            file_sha(
                exact.safe_member(
                    current_consumer, "src/cst_research/experiments/" + name
                )
            )
            == digest,
            "qualified StageA source changed",
        )
    cases = model_contract.pilot_cases()
    exact.require(
        [row["case"] for row in summary["cases"]] == cases
        and [row["case"] for row in report["cases"]] == cases,
        "ordered exact16 qualified cases required",
    )
    imported = {}
    for case, raw_row, audited in zip(
        cases, summary["cases"], report["cases"], strict=True
    ):
        base = f"{PREFIX}/cases/{case['id']}"
        qname, initial = base + "/qualification.json", base + "/initial-state.pt"
        exact.require(
            audited["qualification_file"] == qname
            and audited["initial_state_file"] == initial
            and audited["qualification_sha256"]
            == raw_row["qualification_sha256"]
            == expected[qname]["sha256"]
            and audited["initial_state_sha256"]
            == raw_row["initial_state_sha256"]
            == expected[initial]["sha256"],
            "actual audited case/rawinitial binding differs",
        )
        record = read(exact.safe_member(root, qname))
        exact.require(
            record["case"] == case
            and record["identity"] == identity
            and record["status"] == "passed-new-native-case-qualification"
            and record["normal_true_B4_loss_updates"] == 8
            and record["additional_resumed_normal_loss_updates"] == 4
            and record["all_other_resume_bytes_exact"] is True
            and record["amplified_VJP_never_used_for_optimizer"] is True
            and record["owned_point_tensors"] == case["hidden_layers"] + 3
            and record["task_dataset_or_development_evaluated"] is False,
            "native case qualification incomplete",
        )
        for name, digest in record["files"].items():
            full = base + "/" + name
            exact.safe_member(root, full)
            exact.require(
                expected[full]["sha256"] == digest, "native raw cell hash differs"
            )
        imported[case["id"]] = {
            "case": case,
            "initial_state_file": initial,
            "initial_state_sha256": raw_row["initial_state_sha256"],
            "qualification_sha256": raw_row["qualification_sha256"],
        }
    return {
        "status": "verified-readonly-actual-StageA16-for-new-StageB-fresh0",
        "producer_identity": identity,
        "independent_admission_sha256": admission_sha,
        "root_file_release_sha256": release_sha,
        "parent_outer_receipt_sha256": report["parent_outer_receipt_sha256"],
        "files_verified_in_place": expected,
        "cases": imported,
        "task_training_updates_inherited": 0,
        "qualification_states_used_as_training_state": False,
    }


def load_initial(root, proof, case):
    import torch

    row = proof["cases"][case["id"]]
    path = exact.safe_member(root, row["initial_state_file"])
    sealed_sha(path, row["initial_state_sha256"])
    raw = torch.load(path, map_location="cpu", weights_only=True)
    exact.require(
        raw["epoch"] == raw["updates"] == 0
        and raw["configuration"]["case"] == case
        and raw["configuration"]["identity"] == proof["producer_identity"]
        and not raw["optimizer"]["state"]
        and all(v is None for v in raw["gradients"].values()),
        "only qualified freshinitial0 with empty moments/Nonegrads allowed",
    )
    return raw
