"""Frozen TinyStories prefixes and GPT-2 BPE token files; never loads a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import sys
import urllib.request
from array import array
from pathlib import Path

DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATASET = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
END_OF_STORY = "<|endoftext|>"


def split_stories(lines):
    """Split whole stories before token packing, independent of line boundaries."""
    pieces = []
    for line in lines:
        parts = line.split(END_OF_STORY)
        for index, part in enumerate(parts):
            pieces.append(part)
            if index < len(parts) - 1:
                story = "".join(pieces).strip()
                if story:
                    yield story
                pieces = []
    story = "".join(pieces).strip()
    if story:
        yield story


def remote_stories(filename):
    request = urllib.request.Request(
        DATASET + DATASET_REVISION + "/" + filename,
        headers={"User-Agent": "cst-research-ffn/1"},
    )
    with (
        urllib.request.urlopen(request, timeout=90) as response,
        io.TextIOWrapper(response, encoding="utf-8", errors="strict") as text,
    ):
        yield from split_stories(text)


def story_digest(story):
    return hashlib.sha256(story.encode("utf-8")).hexdigest()


def write_prefix(stories, destination, count, encoding, *, forbidden=None):
    """Consume whole documents, save only the requested token prefix.

    The normalized-story digest covers every consumed accepted document,
    including the discarded tail of the final story. It is not a full upstream
    file digest. Returned story IDs prevent reuse across train/dev/confirmation.
    """
    forbidden = set() if forbidden is None else forbidden
    ids, duplicate_count = set(), 0
    tokens = array("I")
    normalized = hashlib.sha256()
    while len(tokens) < count:
        story = next(stories)
        digest = story_digest(story)
        if digest in forbidden or digest in ids:
            duplicate_count += 1
            continue
        ids.add(digest)
        normalized.update(story.encode("utf-8"))
        normalized.update(b"\x00")
        tokens.extend(encoding.encode_ordinary(story))
        tokens.append(encoding.eot_token)
    del tokens[count:]
    if tokens.itemsize != 4:
        raise RuntimeError("uint32 token storage required")
    if sys.byteorder != "little":
        tokens.byteswap()
    with destination.open("xb") as stream:
        tokens.tofile(stream)
    payload_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "filename": destination.name,
        "tokens": count,
        "dtype": "<u4",
        "sha256": payload_digest,
        "accepted_stories": len(ids),
        "duplicates_excluded": duplicate_count,
        "normalized_consumed_stories_sha256": normalized.hexdigest(),
    }, ids


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=20000001)
    args = parser.parse_args(argv)
    if not 1000001 <= args.train_tokens <= 20000001:
        parser.error("bounded train prefix must contain1M+1 through20M+1 tokens")
    import tiktoken

    encoding = tiktoken.get_encoding("gpt2")
    assert encoding.n_vocab == 50257 and encoding.eot_token == 50256
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "state": "preparing",
        "dataset": "roneneldan/TinyStories",
        "revision": DATASET_REVISION,
        "tokenizer": "tiktoken:gpt2",
        "tiktoken_version": importlib.metadata.version("tiktoken"),
        "vocabulary": encoding.n_vocab,
        "files": {},
        "packing": "whole-story split first, GPT2 ordinary encoding plus EOS; uint32 little-endian",
        "source_scope": "consumed normalized story prefixes, not full upstream file retrieval",
    }

    def save():
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    try:
        train, train_ids = write_prefix(
            iter(remote_stories("TinyStories-train.txt")),
            args.output / "train.bin",
            args.train_tokens,
            encoding,
        )
        manifest["files"]["train"] = train
        save()
        print("Train prefix prepared", train, flush=True)
        validation = iter(remote_stories("TinyStories-valid.txt"))
        dev, dev_ids = write_prefix(
            validation,
            args.output / "development.bin",
            8193,
            encoding,
            forbidden=train_ids,
        )
        manifest["files"]["development"] = dev
        save()
        confirm, confirm_ids = write_prefix(
            validation,
            args.output / "confirmation.bin",
            32769,
            encoding,
            forbidden=train_ids | dev_ids,
        )
        manifest["files"]["confirmation"] = confirm
        assert not (
            train_ids & dev_ids or train_ids & confirm_ids or dev_ids & confirm_ids
        )
        for name, ids in [
            ("train", train_ids),
            ("development", dev_ids),
            ("confirmation", confirm_ids),
        ]:
            p = args.output / (name + "-story-sha256.txt")
            p.write_text("\n".join(sorted(ids)) + "\n")
            manifest["files"][name]["story_hash_file_sha256"] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
        # Fingerprint the actual encoding, rather than relying only on its name.
        state = hashlib.sha256()
        for token, rank in sorted(
            encoding._mergeable_ranks.items(), key=lambda item: item[1]
        ):
            state.update(len(token).to_bytes(4, "little"))
            state.update(token)
            state.update(rank.to_bytes(4, "little"))
        state.update(json.dumps(encoding._special_tokens, sort_keys=True).encode())
        manifest["encoding_sha256"] = state.hexdigest()
        manifest["story_hash_sets_disjoint"] = True
        manifest["state"] = "prepared"
        save()
        print("All token files prepared, story hashes disjoint", flush=True)
    except Exception as error:
        manifest["state"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
