from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = PROJECT_ROOT / "config" / "defaults.json"
SUBGRAPH_PATH = PROJECT_ROOT / "config" / "subgraph.json"


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_defaults(path: str | Path = DEFAULTS_PATH) -> dict:
    return load_json(path)


def load_subgraph(path: str | Path = SUBGRAPH_PATH) -> dict:
    return load_json(path)
