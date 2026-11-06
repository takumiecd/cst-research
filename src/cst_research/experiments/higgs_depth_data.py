"""Admission of the frozen new-row depth data, without opening heldout payloads.

The raw preparation script and its full-source/array hashes are retained outside
Git. This reader only accepts the four-array train/development artifact. Historical
feature-statistics exposure is separate from documented model/label use.
"""

from __future__ import annotations

import hashlib

from .higgs_dst_comparison import file_sha, read

RAW_SHA = "ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698"
DATA_SHA = "58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3"
MANIFEST_SHA = "530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526"
RANGES = {
    "train": [1000000, 1100000],
    "development": [1100000, 1150000],
    "future_confirmation": [1150000, 1200000],
}


def data_spec():
    return {
        "version": "higgs-depth-new-rows-v1",
        "raw_compressed_sha256": RAW_SHA,
        "official_train_rows": [0, 10500000],
        "official_test_reserved": [10500000, 11000000],
        "ranges": {k: list(v) for k, v in RANGES.items()},
        "normalization_fit": list(RANGES["train"]),
        "statistics_dtype": "float64",
        "std_floor": 1e-6,
        "application": "(raw_float32-mean_float32)/std_float32",
        "documented_previous_exposure": {
            "model_training": [0, 100000],
            "retained_raw": [0, 1000000],
            "feature_statistics_fit": [0, 10400000],
            "development": [10400000, 10450000],
            "confirmation_reserved": [10450000, 10500000],
            "manifest_sha256": "e668148a325d1a0a6ad70c5b2bad2ff4c754b4651c8f121ced4e8b24e06914fd",
        },
        "exposure_scope": "known prior HIGGS artifacts; no universal claim about all historical CST label use",
        "formal_confirmation_ready": False,
        "official_test_evaluated": False,
    }


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def validate_spec(spec):
    if spec != data_spec():
        raise ValueError("depth split/preprocessing specification differs")
    values = list(spec["ranges"].values())
    if any(overlaps(a, b) for i, a in enumerate(values) for b in values[i + 1 :]):
        raise ValueError("depth raw splits overlap")
    old = spec["documented_previous_exposure"]
    if any(
        overlaps(v, old["retained_raw"])
        or overlaps(v, old["development"])
        or overlaps(v, old["confirmation_reserved"])
        for v in values
    ):
        raise ValueError(
            "new splits overlap documented previous retained/validation rows"
        )
    return {
        "model_and_label_ranges_disjoint_from_documented_previous": True,
        "historical_feature_statistics_exposure": {
            k: overlaps(v, old["feature_statistics_fit"])
            for k, v in spec["ranges"].items()
        },
        "current_normalization_fit_only_new_train": True,
        "formal_confirmation_ready": False,
        "scope": spec["exposure_scope"],
    }


def validate_manifest(manifest):
    expected_ranges = {
        "training": RANGES["train"],
        "development": RANGES["development"],
        "future_confirmation": RANGES["future_confirmation"],
    }
    if (
        manifest.get("source_compressed_sha256") != RAW_SHA
        or manifest.get("source_bytes") != 2816407858
        or manifest.get("raw_rows_zero_based_half_open") != expected_ranges
        or manifest.get("training_artifact_contains_future_confirmation") is not False
        or manifest.get("future_confirmation_evaluated") is not False
        or manifest.get("official_test_accessed") is not False
        or manifest.get("official_test_reserved_range") != [10500000, 11000000]
    ):
        raise ValueError("depth raw provenance/partition/heldout scope differs")
    normalization = manifest["normalization"]
    expected = {
        "fit_rows": RANGES["train"],
        "statistics_dtype": "float64",
        "raw_dtype": "float32",
        "std_floor": 1e-6,
        "ddof": 0,
        "application": "(raw_float32 - mean_float32) / std_float32",
    }
    if any(normalization.get(k) != v for k, v in expected.items()):
        raise ValueError("depth normalization is not new-train-only")
    exposure = manifest["exposure"]
    if (
        exposure.get("prior_retained_training") != [0, 1000000]
        or exposure.get("prior_current_validation") != [10400000, 10500000]
        or exposure.get("prior_aggregate_feature_statistics") != [0, 10400000]
        or exposure.get("old_feature_statistics_included_new_rows") is not True
        or exposure.get("new_normalization_uses_only_new_training_rows") is not True
    ):
        raise ValueError("depth historical feature/model exposure disclosure differs")
    return validate_spec(data_spec())


def load_pilot(data, manifest_path):
    import numpy as np

    if file_sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("depth manifest digest differs")
    manifest = read(manifest_path)
    validate_manifest(manifest)
    artifact = manifest["artifacts"]["train-development.npz"]
    if file_sha(data) != DATA_SHA or artifact["sha256"] != DATA_SHA:
        raise ValueError("depth train/development artifact digest differs")
    expected = {"x_train", "y_train", "x_development", "y_development"}
    with np.load(data, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError(
                "pilot payload has missing or unapproved arrays, including heldout data"
            )
        values = {name: archive[name].copy() for name in expected}
    for name, rows in (("train", 100000), ("development", 50000)):
        x, y = values["x_" + name], values["y_" + name]
        if (
            x.shape != (rows, 28)
            or x.dtype != np.float32
            or y.shape != (rows,)
            or y.dtype != np.int64
        ):
            raise ValueError("depth array shape/dtype differs")
        if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
            raise ValueError("invalid depth data values")
    for name, value in values.items():
        metadata = artifact["arrays"][name]
        if (
            metadata["shape"] != list(value.shape)
            or metadata["dtype"] != str(value.dtype)
            or metadata["sha256"]
            != hashlib.sha256(value.tobytes(order="C")).hexdigest()
        ):
            raise ValueError("depth normalized array hash differs")
    normalized_train = values["x_train"].astype(np.float64)
    if (
        np.abs(normalized_train.mean(0)).max() > 1e-5
        or np.abs(normalized_train.std(0) - 1).max() > 1e-5
    ):
        raise ValueError("depth train-only standardization consistency check failed")
    return values, manifest
