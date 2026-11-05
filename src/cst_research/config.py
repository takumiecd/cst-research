"""Optional local evidence locator; never downloads or modifies evidence."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def evidence_location(study, config_path=None):
    config_path = config_path or os.environ.get("CST_RESEARCH_CONFIG")
    root_value = os.environ.get("CST_RESEARCH_EVIDENCE_ROOT")
    origin = "CST_RESEARCH_EVIDENCE_ROOT" if root_value else None
    if not root_value and config_path:
        path = Path(config_path).expanduser().resolve()
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        evidence = config.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError("local config [evidence] must be a table")
        root_value = evidence.get("root")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError("local config requires a nonempty [evidence] root")
        root = Path(root_value).expanduser()
        root_value = str(root if root.is_absolute() else path.parent / root)
        origin = "local config"
    if not root_value:
        return {"configured": False, "path": None, "exists": None, "origin": None}
    location = (
        Path(root_value).expanduser().resolve() / study["evidence"]["relative_path"]
    )
    return {
        "configured": True,
        "path": str(location),
        "exists": location.is_dir(),
        "origin": origin,
    }
