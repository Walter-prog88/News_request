"""Public configuration and result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Market(StrEnum):
    CHINA = "CN"
    HONG_KONG = "HK"
    UNITED_STATES = "US"


class ValidationStatus(StrEnum):
    VALIDATED = "Validated"
    PROMISING = "Promising"
    RISK_OVERLAY = "Risk overlay"
    FAILED = "Failed"
    INSUFFICIENT_HISTORY = "Insufficient history"


class AssetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    name: str
    studies: list[Literal["caitong", "zhongyou", "us_extension"]]
    signal_symbol: str
    execution_symbol: str | None = None
    provider: str
    market: Market
    currency: str
    calendar: str
    start: date | None = None
    one_way_cost_bps: float = Field(ge=0)
    cash_series: str | None = None
    proprietary: bool = False
    proxy_for: str | None = None
    notes: str = ""


class DatasetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    provider: str
    symbol: str
    retrieved_at: datetime
    as_of: date
    path: Path
    sha256: str
    row_count: int = Field(ge=0)
    min_date: date | None = None
    max_date: date | None = None
    vintage: str | None = None
    is_proxy: bool = False
    warnings: list[str] = Field(default_factory=list)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    study: Literal["caitong", "zhongyou"]
    asset_ids: list[str]
    model: str
    feature_set: str
    train_end: date
    test_start: date
    test_end: date | None = None
    seeds: list[int] = Field(default_factory=lambda: list(range(10)))
    assumptions: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class MacroSeriesSpec(BaseModel):
    """Configuration contract for a point-in-time macro feature series."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    series_id: str = Field(alias="id", min_length=1)
    group: Literal[
        "rates_liquidity",
        "risk_credit",
        "dollar_commodity",
        "high_frequency_economy",
        "low_frequency_macro",
    ]
    frequency: Literal["daily", "weekly", "monthly"]
    use_alfred_vintage: bool = False

    @field_validator("series_id")
    @classmethod
    def normalize_series_id(cls, value: str) -> str:
        return value.strip().upper()


class MacroConfig(BaseModel):
    """Macro universe and its conservative availability convention."""

    model_config = ConfigDict(extra="forbid")

    availability_rule: Literal["next_trading_day_when_release_time_unknown"]
    series: list[MacroSeriesSpec]

    @model_validator(mode="after")
    def unique_series_ids(self) -> MacroConfig:
        ids = [spec.series_id for spec in self.series]
        if len(ids) != len(set(ids)):
            raise ValueError("macro series ids must be unique")
        return self


@dataclass(slots=True)
class BacktestResult:
    asset_id: str
    experiment_id: str
    daily: pd.DataFrame
    metrics: dict[str, float | int | str | None]
    status: ValidationStatus = ValidationStatus.FAILED
    diagnostics: dict[str, Any] | None = None
