"""Stable project paths used by CLI, tests and the dashboard."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    override = os.getenv("INDEX_TIMING_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ROOT / "reports"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated"


def ensure_runtime_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, RUNS_DIR, MODELS_DIR, GENERATED_REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)

