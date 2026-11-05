"""Frozen Covertype split/preprocessing; heldout artifact is separate from training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import zipfile
from pathlib import Path

from .higgs_dst_comparison import file_sha, json_sha, read, write

ZIP_SHA = "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d"
GZIP_SHA = "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"
DATA_SHA = "010f150bab3a700fbcde1106daf21e83cf28fe9024732d29c25756144a07e416"
MANIFEST_SHA = "4a6026b41b0abe9594a0320e6455dee17e3fc8ecddcb3aff97657a75b7ae4bcd"
HELDOUT_SHA = "7220f0d7282c83732c203325f523da180e4701d8a001647ea0eff4cdc0dbee4f"
COUNTS = (100000, 50000, 50000)


def specification():
    return {
        "version": "covertype-random-transfer-v1",
        "rows": 581012,
        "features": 54,
        "classes": 7,
        "source_zip_sha256": ZIP_SHA,
        "source_gzip_sha256": GZIP_SHA,
        "permutation": {"generator": "numpy.PCG64", "seed": 20261106, "rows": 581012},
        "split_sizes": dict(zip(("train", "development", "heldout"), COUNTS)),
        "normalization": {
            "columns": [0, 10],
            "fit": "train only",
            "statistics_dtype": "float64",
            "ddof": 0,
            "std_floor": 1e-6,
            "application": "float64 then float32",
            "binary_columns_unchanged": [10, 54],
        },
        "label_mapping": "raw1..7 to0..6",
        "geographic_generalization": False,
        "formal_noninferiority": False,
        "official_benchmark_comparison": False,
    }


def array_record(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def partition(raw, counts=COUNTS):
    """Pure preprocessing, with smaller explicit counts only for CPU unit tests."""
    import numpy as np

    if raw.ndim != 2 or raw.shape[1] != 55 or sum(counts) > len(raw):
        raise ValueError("Covertype row/feature shape differs")
    if (
        not np.isfinite(raw).all()
        or not np.isin(raw[:, 10:54], (0, 1)).all()
        or not np.isin(raw[:, 54], range(1, 8)).all()
    ):
        raise ValueError("invalid Covertype values")
    order = np.random.Generator(np.random.PCG64(20261106)).permutation(len(raw))
    train = raw[order[: counts[0]], :10].astype(np.float64)
    mean = train.mean(axis=0)
    std = np.maximum(train.std(axis=0, ddof=0), 1e-6)
    result = {}
    start = 0
    for name, count in zip(("train", "development", "heldout"), counts):
        ids = order[start : start + count].astype(np.int64)
        start += count
        x = raw[ids, :54].astype(np.float32).copy()
        x[:, :10] = ((x[:, :10].astype(np.float64) - mean) / std).astype(np.float32)
        result[name] = {
            "x": x,
            "y": (raw[ids, 54] - 1).astype(np.int64),
            "row_ids": ids,
        }
    return result, {
        "mean_float64": mean.tolist(),
        "std_float64": std.tolist(),
        "full_permutation_sha256": array_record(order)["sha256"],
    }


def prepare(archive, output):
    import numpy as np

    if file_sha(archive) != ZIP_SHA:
        raise ValueError("UCI zip digest differs")
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        members = [n for n in source.namelist() if Path(n).name == "covtype.data.gz"]
        if len(members) != 1:
            raise ValueError("ambiguous UCI data member")
        payload = source.read(members[0])
    if hashlib.sha256(payload).hexdigest() != GZIP_SHA:
        raise ValueError("UCI gzip member differs")
    raw = np.loadtxt(
        io.BytesIO(gzip.decompress(payload)), delimiter=",", dtype=np.float32
    )
    if raw.shape != (581012, 55):
        raise ValueError("full UCI rows differ")
    splits, normalization = partition(raw)
    development = {
        f"{key}_{split}": value
        for split in ("train", "development")
        for key, value in splits[split].items()
    }
    heldout = {f"{key}_heldout": value for key, value in splits["heldout"].items()}
    np.savez_compressed(output / "train-development.npz", **development)
    np.savez_compressed(output / "future-heldout.npz", **heldout)
    manifest = {
        "schema_version": 1,
        "specification": specification(),
        "specification_sha256": json_sha(specification()),
        "normalization": normalization,
        "artifacts": {},
        "heldout_evaluated": False,
        "training_artifact_contains_heldout": False,
        "citation": "Blackard, J. (1998). Covertype. UCI. https://doi.org/10.24432/C50K5N",
        "source_url": "https://archive.ics.uci.edu/static/public/31/covertype.zip",
        "license": "CC BY 4.0",
    }
    for name, values in (
        ("train-development.npz", development),
        ("future-heldout.npz", heldout),
    ):
        manifest["artifacts"][name] = {
            "sha256": file_sha(output / name),
            "arrays": {k: array_record(v) for k, v in values.items()},
        }
    write(output / "data-manifest.json", manifest)
    return manifest


def load_development(data, manifest_path, expected_manifest_sha):
    import numpy as np

    if (
        file_sha(manifest_path) != expected_manifest_sha
        or expected_manifest_sha != MANIFEST_SHA
    ):
        raise ValueError("data manifest digest differs")
    manifest = read(manifest_path)
    if (
        manifest["specification"] != specification()
        or manifest["specification_sha256"] != json_sha(specification())
        or manifest["heldout_evaluated"] is not False
        or manifest["training_artifact_contains_heldout"] is not False
    ):
        raise ValueError("data protocol/heldout scope differs")
    artifact = manifest["artifacts"]["train-development.npz"]
    if file_sha(data) != artifact["sha256"] or artifact["sha256"] != DATA_SHA:
        raise ValueError("train/development artifact changed")
    expected = {
        f"{key}_{split}"
        for split in ("train", "development")
        for key in ("x", "y", "row_ids")
    }
    with np.load(data, allow_pickle=False) as source:
        if set(source.files) != expected:
            raise ValueError("unexpected training payload, including heldout")
        arrays = {k: source[k].copy() for k in expected}
    for split, count in zip(("train", "development"), COUNTS):
        x, y, ids = (arrays[k + "_" + split] for k in ("x", "y", "row_ids"))
        if (
            x.shape != (count, 54)
            or x.dtype != np.float32
            or y.shape != (count,)
            or y.dtype != np.int64
            or ids.shape != (count,)
            or ids.dtype != np.int64
            or len(np.unique(ids)) != count
            or not np.isfinite(x).all()
            or not np.isin(x[:, 10:], (0, 1)).all()
            or set(np.unique(y)) != set(range(7))
        ):
            raise ValueError("invalid training/development arrays")
        for key in ("x", "y", "row_ids"):
            name = key + "_" + split
            if array_record(arrays[name]) != artifact["arrays"][name]:
                raise ValueError("array digest differs")
    if np.intersect1d(arrays["row_ids_train"], arrays["row_ids_development"]).size:
        raise ValueError("train/development overlap")
    return arrays, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output)


if __name__ == "__main__":
    main()
