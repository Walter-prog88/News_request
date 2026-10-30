"""YAML configuration loading with strict validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, TypeAdapter

from .paths import CONFIG_DIR
from .types import AssetSpec, ExperimentSpec, MacroConfig

T = TypeVar("T", bound=BaseModel)


def read_yaml(path: str | Path) -> Any:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = CONFIG_DIR / resolved
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_assets(path: str | Path = "assets.yml") -> dict[str, AssetSpec]:
    payload = read_yaml(path)
    assets = TypeAdapter(list[AssetSpec]).validate_python(payload["assets"])
    return {asset.asset_id: asset for asset in assets}


def load_experiments(path: str | Path) -> list[ExperimentSpec]:
    payload = read_yaml(path)
    return TypeAdapter(list[ExperimentSpec]).validate_python(payload["experiments"])


def load_macro_config(path: str | Path = "macro_us.yml") -> MacroConfig:
    """Load the point-in-time US macro universe."""

    return MacroConfig.model_validate(read_yaml(path))


def load_validation_protocol(path: str | Path = "validation.yml") -> dict[str, Any]:
    """Load the locked walk-forward, inference and robustness protocol."""

    payload = read_yaml(path)
    required = {
        "walk_forward",
        "execution",
        "bootstrap",
        "validated_thresholds",
        "neural_increment_thresholds",
        "parameter_robustness",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"validation protocol is missing sections: {missing}")
    return payload
