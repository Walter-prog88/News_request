"""Caitong broad-index timing features with point-in-time joins.

The source report does not disclose a complete factor dictionary.  This module
therefore keeps the disclosed factor *families* separate and records every
point-in-time source explicitly.  A macro value is joined on its availability
timestamp, never on the economic observation period it describes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TECHNICAL_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)
CAITONG_FEATURE_GROUPS: tuple[str, ...] = (
    "technical",
    "valuation_fund",
    "rates_commodities_overseas",
    "low_frequency_macro",
)
CAITONG_FEATURE_SETS: tuple[str, ...] = (
    "technical",
    "technical_plus_valuation_fund",
    "technical_plus_market_global",
    "full",
)
FeatureGroup = Literal[
    "technical",
    "valuation_fund",
    "rates_commodities_overseas",
    "low_frequency_macro",
]


@dataclass(frozen=True, slots=True)
class PointInTimeSource:
    """A feature source indexed by, or containing, its first available date.

    ``release_timestamp_col=None`` means the frame index is the availability
    timestamp.  Observation dates must not be substituted for release dates.
    ``prefix`` is mandatory in spirit (and automatically derived when omitted)
    so similarly named fields from different sources cannot silently collide.
    """

    name: str
    group: FeatureGroup
    data: pd.DataFrame
    release_timestamp_col: str | None = None
    value_columns: tuple[str, ...] | None = None
    prefix: str | None = None


@dataclass(frozen=True, slots=True)
class CaitongFeatureBundle:
    """Feature matrix plus auditable family membership and assumptions."""

    frame: pd.DataFrame
    groups: Mapping[str, tuple[str, ...]]
    assumptions: tuple[str, ...]
    source_metadata: tuple[Mapping[str, object], ...]

    def columns_for(self, feature_set: str = "full") -> tuple[str, ...]:
        """Return ordered columns for a nested ablation feature set."""

        aliases = {
            "technical": ("technical",),
            "technical_plus_valuation_fund": ("technical", "valuation_fund"),
            "technical_plus_market_global": (
                "technical",
                "valuation_fund",
                "rates_commodities_overseas",
            ),
            "full": CAITONG_FEATURE_GROUPS,
        }
        if feature_set in self.groups:
            requested = (feature_set,)
        elif feature_set in aliases:
            requested = aliases[feature_set]
        else:
            raise ValueError(
                f"Unknown feature_set={feature_set!r}; expected one of "
                f"{sorted(set(aliases) | set(self.groups))}"
            )
        return tuple(column for group in requested for column in self.groups.get(group, ()))


def _datetime_index(frame: pd.DataFrame | pd.Series, *, name: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"), name="date")
    if index.hasnans:
        raise ValueError(f"{name} contains missing timestamps")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted by date")
    if index.has_duplicates:
        raise ValueError(f"{name} contains duplicate timestamps")
    return index


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window, min_periods=window).mean()
    loss = -delta.clip(upper=0.0).rolling(window, min_periods=window).mean()
    relative_strength = _safe_divide(gain, loss)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    # The limiting cases are well-defined even though the ratio is infinite.
    rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
    rsi = rsi.mask((loss == 0.0) & (gain == 0.0), 50.0)
    return rsi


def build_technical_features(
    market: pd.DataFrame,
    *,
    windows: Sequence[int] = TECHNICAL_WINDOWS,
) -> pd.DataFrame:
    """Build causal price/volume features at the report's four horizons.

    Required field: ``close``.  Optional fields are ``open``, ``high``, ``low``,
    ``volume``, ``amount`` and ``turnover``.  Features only use observations at
    or before the row timestamp.  No back-filling is performed.
    """

    if "close" not in market:
        raise ValueError("market must contain a 'close' column")
    if tuple(windows) != tuple(sorted(set(windows))) or any(window < 2 for window in windows):
        raise ValueError("windows must be unique, increasing integers >= 2")

    result = pd.DataFrame(index=_datetime_index(market, name="market"))
    close = _numeric(market["close"])
    close.index = result.index
    one_day_return = close.pct_change(fill_method=None)

    optional: dict[str, pd.Series] = {}
    for name in ("open", "high", "low", "volume", "amount", "turnover"):
        if name in market:
            optional[name] = _numeric(market[name]).set_axis(result.index)

    result["return_1d"] = one_day_return
    if "open" in optional:
        result["intraday_return"] = _safe_divide(close, optional["open"]) - 1.0
    if {"high", "low"}.issubset(optional):
        result["intraday_range"] = _safe_divide(
            optional["high"] - optional["low"], close.shift(1)
        )

    for window in windows:
        rolling_close = close.rolling(window, min_periods=window)
        close_mean = rolling_close.mean()
        close_std = rolling_close.std(ddof=0)
        result[f"return_{window}d"] = close.pct_change(window, fill_method=None)
        result[f"realized_vol_{window}d"] = (
            one_day_return.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)
        )
        result[f"close_to_ma_{window}d"] = _safe_divide(close, close_mean) - 1.0
        result[f"close_zscore_{window}d"] = _safe_divide(close - close_mean, close_std)
        result[f"rsi_{window}d"] = _rsi(close, window) / 100.0

        if {"high", "low"}.issubset(optional):
            rolling_high = optional["high"].rolling(window, min_periods=window).max()
            rolling_low = optional["low"].rolling(window, min_periods=window).min()
            result[f"range_position_{window}d"] = _safe_divide(
                close - rolling_low, rolling_high - rolling_low
            )

        for activity in ("volume", "amount", "turnover"):
            if activity not in optional:
                continue
            values = optional[activity]
            rolling_mean = values.rolling(window, min_periods=window).mean()
            result[f"{activity}_to_ma_{window}d"] = _safe_divide(values, rolling_mean) - 1.0
            result[f"{activity}_change_{window}d"] = values.pct_change(
                window, fill_method=None
            ).replace([np.inf, -np.inf], np.nan)

        if "volume" in optional:
            signed_volume = np.sign(one_day_return.fillna(0.0)) * optional["volume"]
            obv = signed_volume.cumsum()
            result[f"obv_change_{window}d"] = obv.diff(window)

    return result.replace([np.inf, -np.inf], np.nan)


def point_in_time_join(
    base: pd.DataFrame,
    source: PointInTimeSource,
    *,
    keep_available_at: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Backward-as-of join using a source's first-publication timestamps."""

    base_index = _datetime_index(base, name="base")
    source_frame = source.data.copy()
    if source.release_timestamp_col is None:
        available_at = pd.to_datetime(source_frame.index, errors="raise")
    else:
        if source.release_timestamp_col not in source_frame:
            raise ValueError(
                f"{source.name}: missing release column {source.release_timestamp_col!r}"
            )
        available_at = pd.to_datetime(
            source_frame.pop(source.release_timestamp_col), errors="raise"
        )
    if pd.isna(available_at).any():
        raise ValueError(f"{source.name}: release timestamps contain missing values")

    if source.value_columns is None:
        value_columns = tuple(
            column
            for column in source_frame.columns
            if column not in {"observation_date", "period", "vintage"}
        )
    else:
        value_columns = source.value_columns
    missing = sorted(set(value_columns) - set(source_frame.columns))
    if missing:
        raise ValueError(f"{source.name}: missing value columns {missing}")
    if not value_columns:
        raise ValueError(f"{source.name}: no value columns selected")

    prefix = source.prefix or source.name
    rename = {column: f"{prefix}__{column}" for column in value_columns}
    collisions = sorted(set(rename.values()) & set(base.columns))
    if collisions:
        raise ValueError(f"{source.name}: feature columns already exist: {collisions}")
    right = source_frame.loc[:, list(value_columns)].rename(columns=rename)
    right.insert(0, "__available_at", pd.DatetimeIndex(available_at))
    right = right.sort_values("__available_at").drop_duplicates("__available_at", keep="last")

    left = base.copy()
    left.index = base_index
    left.insert(0, "__date", base_index)
    joined = pd.merge_asof(
        left.sort_values("__date"),
        right,
        left_on="__date",
        right_on="__available_at",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("__date")
    joined.index.name = "date"
    if keep_available_at:
        joined = joined.rename(columns={"__available_at": f"{prefix}__available_at"})
    else:
        joined = joined.drop(columns="__available_at")
    return joined, tuple(rename.values())


def assemble_caitong_features(
    market: pd.DataFrame,
    *,
    pit_sources: Iterable[PointInTimeSource] = (),
    windows: Sequence[int] = TECHNICAL_WINDOWS,
) -> CaitongFeatureBundle:
    """Assemble the four nested Caitong factor families.

    External source values are accepted only through :class:`PointInTimeSource`,
    which makes their release-date semantics explicit.
    """

    features = build_technical_features(market, windows=windows)
    groups: dict[str, list[str]] = {group: [] for group in CAITONG_FEATURE_GROUPS}
    groups["technical"].extend(features.columns)
    metadata: list[Mapping[str, object]] = []
    seen_names: set[str] = set()

    for source in pit_sources:
        if source.name in seen_names:
            raise ValueError(f"Duplicate point-in-time source name: {source.name!r}")
        seen_names.add(source.name)
        features, added = point_in_time_join(features, source)
        groups[source.group].extend(added)
        metadata.append(
            {
                "name": source.name,
                "group": source.group,
                "release_timestamp_col": source.release_timestamp_col or "index",
                "columns": list(added),
                "join": "backward_asof",
            }
        )

    assumptions = (
        "研报未披露完整因子清单；本实现按技术量价、估值资金、利率商品海外、低频宏观分组。",
        "所有外部因子按首次可得时间向后as-of合并，不以经济观察期回填。",
        "技术量价窗口固定为5/10/20/60个交易日，且仅使用当日及此前数据。",
    )
    return CaitongFeatureBundle(
        frame=features,
        groups={group: tuple(columns) for group, columns in groups.items()},
        assumptions=assumptions,
        source_metadata=tuple(metadata),
    )


def make_caitong_targets(
    close: pd.Series,
    *,
    horizons: Sequence[int] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Create forward returns, binary directions and their realization dates."""

    if any(horizon < 1 for horizon in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must contain unique positive integers")
    index = _datetime_index(close, name="close")
    prices = _numeric(close).set_axis(index)
    output = pd.DataFrame(index=index)
    dates = pd.Series(index, index=index)
    for horizon in horizons:
        forward = prices.shift(-horizon) / prices - 1.0
        output[f"forward_return_{horizon}d"] = forward
        output[f"target_up_{horizon}d"] = (forward > 0.0).astype(float).where(forward.notna())
        output[f"target_available_at_{horizon}d"] = dates.shift(-horizon)
    return output
