"""Read-only registry commands; intentionally no training launcher."""

from __future__ import annotations

import argparse
import json
import sys

from cst_research.config import evidence_location
from cst_research.registry import load_studies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        help="validate/read a draft registry directory instead of packaged studies",
    )
    parser.add_argument(
        "--config", help="local TOML evidence locator; env root takes precedence"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list registered studies")
    listing.add_argument("--json", action="store_true")
    show = commands.add_parser("show", help="show a study manifest")
    show.add_argument("study_id")
    show.add_argument(
        "--evidence",
        action="store_true",
        help="also resolve private local evidence locator",
    )
    validate = commands.add_parser(
        "validate",
        help="validate metadata and frozen design; no evidence/GPU validation",
    )
    validate.add_argument("study_id", nargs="?")
    validate.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        studies = load_studies(args.registry)
        requested = getattr(args, "study_id", None)
        selected = [
            study for study in studies if not requested or study["id"] == requested
        ]
        if not selected:
            raise ValueError(f"unknown study: {requested}")
        if args.command == "list":
            rows = [
                {key: study[key] for key in ("id", "status", "title")}
                for study in studies
            ]
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                for row in rows:
                    print(f"{row['id']}\t{row['status']}\t{row['title']}")
        elif args.command == "show":
            study = selected[0]
            result = (
                {
                    "study": study,
                    "local_evidence": evidence_location(study, args.config),
                }
                if args.evidence
                else study
            )
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        else:
            result = {
                "valid": True,
                "studies": [s["id"] for s in selected],
                "scope": "Registry metadata and declared protocol only; no external evidence, runner, GPU or result validation.",
            }
            print(
                json.dumps(result, indent=2)
                if args.json
                else f"Valid registry: {len(selected)} study/studies. {result['scope']}"
            )
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"cst-research: {error}", file=sys.stderr)
        return 1
