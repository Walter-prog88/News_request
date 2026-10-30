from __future__ import annotations

import json
from pathlib import Path

from .models import QuerySpec


def load_query_specs(path: str | Path) -> list[QuerySpec]:
    config_path = Path(path)
    try:
        root = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON config: {config_path}") from exc

    if not isinstance(root, dict) or not isinstance(root.get("queries"), list):
        raise ValueError("config root must contain a queries list")

    specs = [QuerySpec.from_mapping(item) for item in root["queries"]]
    if not specs:
        raise ValueError("config must contain at least one query")
    ids = [spec.dataset_id for spec in specs]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate dataset ids: {duplicates}")
    return specs

