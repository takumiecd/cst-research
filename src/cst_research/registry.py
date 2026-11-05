"""Small, read-only study registry and explicit frozen-protocol checks."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path, PurePosixPath

STUDY_ID = "higgs-alpha-confirmation-20261105"
STATUSES = {"PLANNED", "RUNNING", "COMPLETED", "STOPPED", "ARCHIVED"}


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and str(path) == value
        and value != "."
        and ":" not in path.parts[0]
    )


def validate_study(study):
    """Return errors without claiming runner correctness or evidence availability."""
    if not isinstance(study, dict):
        return ["study must be a JSON object"]
    errors = []
    if study.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(study.get("id"), str) or not re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*", study["id"]
    ):
        errors.append("id must be a lowercase hyphenated identifier")
    if not isinstance(study.get("status"), str) or study["status"] not in STATUSES:
        errors.append("invalid study status")
    for key in ("title", "status_note", "runner_entrypoint", "source_pull_request"):
        if not isinstance(study.get(key), str) or not study[key]:
            errors.append(f"{key} must be a nonempty string")
    if not re.fullmatch(
        r"https://github\.com/[\w.-]+/[\w.-]+", str(study.get("source_repository", ""))
    ):
        errors.append("source_repository must be a public GitHub repository URL")
    if not re.fullmatch(r"[0-9a-f]{40}", str(study.get("source_commit", ""))):
        errors.append("source_commit must be a full lowercase Git SHA")
    if not safe_relative(study.get("protocol_source_path")):
        errors.append("protocol_source_path must be a safe repository-relative path")
    evidence = study.get("evidence", {})
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
    else:
        if evidence.get("root_env") != "CST_RESEARCH_EVIDENCE_ROOT":
            errors.append("evidence.root_env must be CST_RESEARCH_EVIDENCE_ROOT")
        if not safe_relative(evidence.get("relative_path")):
            errors.append(
                "evidence.relative_path must be a safe relative locator, never an absolute user path"
            )
    if not isinstance(study.get("protocol"), dict):
        errors.append("protocol must be an object")
    elif study.get("id") == STUDY_ID:
        errors.extend(_validate_frozen_higgs(study))
    return errors


def _validate_frozen_higgs(study):
    """Guard the exact registered design; new designs need new study IDs."""
    protocol = study["protocol"]
    expected = {
        (
            "data",
            "prepared_archive_sha256",
        ): "62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac",
        ("training", "dtype"): "FP32",
        ("training", "tf32"): False,
        ("architecture", "widths"): [1024, 2048],
        ("architecture", "alphas"): [0.001, 0.002, 0.005, 0.01],
        ("architecture", "divisors"): [1000, 500, 200, 100],
        ("architecture", "all_layers_trainable"): True,
        ("architecture", "middle_bias"): False,
        ("data", "training_rows"): [0, 100000],
        ("data", "standardization_fit_rows"): [0, 10400000],
        ("data", "development_validation_indices"): [0, 50000],
        ("data", "confirmation_validation_indices"): [50000, 100000],
        ("data", "official_test_rows"): [10500000, 11000000],
        ("data", "official_test_evaluated"): False,
        ("training", "development_seed"): 17,
        ("training", "confirmation_seeds"): [29, 43, 71],
        ("training", "base_learning_rates"): [0.0001, 0.0003, 0.001, 0.003],
        ("training", "batch_size"): 4,
        ("training", "base_epochs"): 10,
        ("training", "confirmation_model_count"): 30,
        ("statistics", "accuracy_margin"): 0.002,
        ("statistics", "roc_auc_margin"): 0.002,
        ("statistics", "family_frozen"): True,
    }
    errors = []
    if (
        study.get("source_commit") != "97580582ed28c4e494bf5c2656bbe94a85c544c3"
        or study.get("runner_entrypoint")
        != "benchmarks.training.higgs_confirmation_campaign"
    ):
        errors.append("frozen HIGGS source/runner differs")
    for (group, key), wanted in expected.items():
        actual_group = protocol.get(group, {})
        if not isinstance(actual_group, dict) or actual_group.get(key) != wanted:
            errors.append(f"frozen HIGGS protocol differs: {group}.{key}")
    training = protocol.get("training", {})
    statistics = protocol.get("statistics", {})
    if isinstance(training, dict):
        edge = training.get("edge_extension", {})
        horizon = training.get("common_horizon_per_width", {})
        if not isinstance(edge, dict) or any(
            edge.get(k) != v
            for k, v in {
                "maximum_additions_per_group": 1,
                "lower": 0.00003,
                "upper": 0.01,
                "epochs": 10,
            }.items()
        ):
            errors.append("frozen HIGGS edge extension differs")
        if (
            not isinstance(horizon, dict)
            or horizon.get("base") != 10
            or horizon.get("extended") != 20
        ):
            errors.append("frozen HIGGS common horizon differs")
        elif any(
            horizon.get(k) != v
            for k, v in {
                "trigger_best_epoch_at_least": 9,
                "trigger_loss_epochs": [8, 10],
                "trigger_loss_improvement_gt": 0.001,
                "retune_top_k": 2,
                "tail_best_epoch_at_least": 19,
                "tail_loss_epochs": [18, 20],
                "tail_loss_improvement_gt": 0.001,
            }.items()
        ):
            errors.append("frozen HIGGS horizon trigger/retuning rule differs")
    if isinstance(statistics, dict):
        primary = statistics.get("primary", {})
        if not isinstance(primary, dict) or any(
            primary.get(k) != v
            for k, v in {
                "method": "intersection-union of two co-primary one-sided paired-seed noninferiority t tests",
                "degrees_of_freedom": 2,
                "configuration_count": 8,
                "configuration_p": "max(p_accuracy,p_roc_auc)",
                "familywise_alpha": 0.05,
            }.items()
        ):
            errors.append("frozen HIGGS primary IUT8 differs")
    data = protocol.get("data", {})
    if not isinstance(data, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(data.get("prepared_archive_sha256", ""))
    ):
        errors.append("prepared dataset archive requires a SHA256")
    return errors


def load_studies(directory=None):
    root = (
        Path(directory)
        if directory is not None
        else files("cst_research").joinpath("studies")
    )
    records = []
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if not path.name.endswith(".json"):
            continue
        try:
            study = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise ValueError(f"cannot read {path.name}: {error}") from error
        errors = validate_study(study)
        if isinstance(study, dict) and path.name != f"{study.get('id')}.json":
            errors.append("filename must match study id")
        if errors:
            raise ValueError(f"{path.name}: " + "; ".join(errors))
        records.append(study)
    if not records:
        raise ValueError("registry contains no studies")
    return records
