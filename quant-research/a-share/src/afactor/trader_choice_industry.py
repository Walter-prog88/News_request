"""Auditable industry, sector, and static-theme features for trader choice.

This module is intentionally independent from the disclosed-buy labels.  Every
public function accepts only a stock/date panel and source data; no label frame
or selected-stock list is accepted.  The time clock is explicit:

* an analyst-report industry label published on calendar date ``d`` becomes
  usable on the first trading session strictly after ``d`` (``T+1``);
* Shenwan index returns and amount features use observations through ``T-1``;
* ``daily_basic`` fields are joined only from the exact previous trading
  session; and
* the local Shenwan membership table and CSI technology memberships are
  current/static snapshots.  Their output columns contain ``static`` and
  ``oracle`` so they cannot be mistaken for point-in-time classifications.

The static classifications are useful for the user's stated oracle experiment,
but must not be used as historical point-in-time evidence in a forward test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


_PANEL_KEYS = ("decision_date", "symbol")
_REPORT_OUTPUT_COLUMNS = (
    "report_fine_industry_pit",
    "report_industry_publish_time",
    "report_industry_effective_date",
    "report_industry_age_sessions",
    "report_industry_age_calendar_days",
)
_DAILY_BASIC_VALUE_COLUMNS = (
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "circ_mv",
)


@dataclass(frozen=True)
class IndustryFeaturePaths:
    """Default local sources for the oracle industry-feature experiment."""

    astock_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    sw_index_root: Path = Path(
        "/path/to/user/Desktop/A-Share-Algorithm-Lab/"
        "regime_detector/data/sw_industry"
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "astock_root", Path(self.astock_root))
        object.__setattr__(self, "sw_index_root", Path(self.sw_index_root))

    @property
    def reports_file(self) -> Path:
        return (
            self.astock_root
            / "data_8:21"
            / "source_raw"
            / "eastmoney_analyst_reports_raw.csv"
        )

    @property
    def static_sw_file(self) -> Path:
        return self.astock_root / "stock_sw_industry.csv"

    @property
    def daily_basic_file(self) -> Path:
        return self.astock_root / "data_8:21" / "daily_basic.csv"

    @property
    def tech_constituents_file(self) -> Path:
        return (
            self.astock_root
            / "data_8:21"
            / "index_cons"
            / "csindex_cons_snapshot_20260821.csv"
        )


def _normal_date(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("date must be valid")
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _date_series(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    compact = text.str.fullmatch(r"\d{8}", na=False)
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if compact.any():
        out.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce"
        )
    if (~compact).any():
        parsed = pd.to_datetime(text.loc[~compact], errors="coerce", utc=True)
        out.loc[~compact] = parsed.dt.tz_convert(None).dt.normalize()
    return out


def _datetime_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def _standard_symbol(value: object) -> str | pd.NA:
    """Normalize common Eastmoney/Tushare/CSI security-code forms."""

    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if not text:
        return pd.NA
    if "." in text:
        left, right = text.split(".", 1)
        if left in {"SH", "SZ", "BJ"} and right.isdigit():
            return f"{right.zfill(6)}.{left}"
        if right in {"SH", "SZ", "BJ"} and left.isdigit():
            return f"{left.zfill(6)}.{right}"
    digits = "".join(character for character in text if character.isdigit())
    if not 1 <= len(digits) <= 6:
        return pd.NA
    digits = digits.zfill(6)
    if digits.startswith(("6", "900")):
        suffix = "SH"
    elif digits.startswith(("0", "3", "200")):
        suffix = "SZ"
    elif digits.startswith(("4", "8", "9")):
        suffix = "BJ"
    else:
        return pd.NA
    return f"{digits}.{suffix}"


def _calendar(values: Sequence[object] | pd.Index) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(pd.Index(values), errors="coerce", utc=True)
    parsed = parsed[~parsed.isna()].tz_convert(None).normalize()
    result = pd.DatetimeIndex(parsed).drop_duplicates().sort_values()
    if len(result) < 2:
        raise ValueError("trading_dates must contain at least two valid sessions")
    return result


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_PANEL_KEYS).difference(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    result = panel.copy()
    result["decision_date"] = _date_series(result["decision_date"])
    result["symbol"] = result["symbol"].map(_standard_symbol).astype("string")
    if result[list(_PANEL_KEYS)].isna().any(axis=None):
        raise ValueError("panel contains invalid decision_date or symbol values")
    if result.duplicated(list(_PANEL_KEYS), keep=False).any():
        raise ValueError("panel must be unique by (decision_date, symbol)")
    result["_industry_original_order"] = np.arange(len(result), dtype=np.int64)
    return result


def _finish_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.sort_values("_industry_original_order", kind="mergesort")
        .drop(columns="_industry_original_order")
        .reset_index(drop=True)
    )


def _read_csv_or_frame(
    source: pd.DataFrame | str | Path,
    *,
    usecols: Sequence[str] | None = None,
    dtype: object | None = None,
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)


def _next_sessions(
    raw_dates: pd.Series, trading_dates: pd.DatetimeIndex
) -> pd.Series:
    """Map a calendar date to the first trading session strictly after it."""

    normalized = _date_series(raw_dates)
    positions = trading_dates.searchsorted(normalized, side="right")
    values = np.full(len(normalized), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = normalized.notna().to_numpy() & (positions < len(trading_dates))
    values[valid] = trading_dates.to_numpy()[positions[valid]]
    return pd.Series(values, index=raw_dates.index)


def prepare_report_industry_history(
    reports: pd.DataFrame | str | Path,
    trading_dates: Sequence[object] | pd.Index,
) -> pd.DataFrame:
    """Normalize report-industry observations and assign strict T+1 dates."""

    frame = _read_csv_or_frame(reports)
    required = {"stockCode", "publishDate", "indvInduName"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"reports is missing required columns: {missing}")
    calendar = _calendar(trading_dates)
    out = pd.DataFrame(index=frame.index)
    out["symbol"] = frame["stockCode"].map(_standard_symbol)
    out["report_industry_publish_time"] = _datetime_series(frame["publishDate"])
    out["report_industry_publish_date"] = out[
        "report_industry_publish_time"
    ].dt.normalize()
    out["report_industry_effective_date"] = _next_sessions(
        frame["publishDate"], calendar
    )
    out["report_fine_industry_pit"] = (
        frame["indvInduName"].astype("string").str.strip().replace("", pd.NA)
    )
    out["_report_source_order"] = np.arange(len(out), dtype=np.int64)
    out = out.dropna(
        subset=[
            "symbol",
            "report_industry_publish_time",
            "report_industry_effective_date",
            "report_fine_industry_pit",
        ]
    )
    return out.sort_values(
        [
            "symbol",
            "report_industry_effective_date",
            "report_industry_publish_time",
            "_report_source_order",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def add_report_industry_pit(
    panel: pd.DataFrame,
    reports: pd.DataFrame | str | Path,
    trading_dates: Sequence[object] | pd.Index,
) -> pd.DataFrame:
    """Attach each stock's latest analyst-report fine industry as of session T.

    A report published on T is deliberately unavailable on T.  Its first
    eligible decision is the next session.  Age zero therefore means that the
    label first became available on the decision session.
    """

    result = _prepare_panel(panel)
    calendar = _calendar(trading_dates)
    absent = pd.DatetimeIndex(result["decision_date"].unique()).difference(calendar)
    if len(absent):
        raise ValueError(
            "trading_dates is missing decision sessions: "
            + ", ".join(str(date.date()) for date in absent)
        )
    history = prepare_report_industry_history(reports, calendar)
    for column in _REPORT_OUTPUT_COLUMNS:
        result[column] = pd.NaT if "date" in column or "time" in column else np.nan
    result["report_fine_industry_pit"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )

    calendar_position = pd.Series(np.arange(len(calendar)), index=calendar)
    for decision, indices in result.groupby("decision_date", sort=False).groups.items():
        decision = pd.Timestamp(decision)
        known = history.loc[
            history["report_industry_effective_date"].le(decision)
        ].drop_duplicates("symbol", keep="last")
        if known.empty:
            continue
        lookup = known.set_index("symbol")
        symbols = result.loc[indices, "symbol"].astype(str)
        for column in (
            "report_fine_industry_pit",
            "report_industry_publish_time",
            "report_industry_effective_date",
        ):
            result.loc[indices, column] = symbols.map(lookup[column]).to_numpy()
        effective = pd.to_datetime(
            result.loc[indices, "report_industry_effective_date"], errors="coerce"
        )
        effective_position = effective.map(calendar_position)
        result.loc[indices, "report_industry_age_sessions"] = (
            int(calendar_position.loc[decision]) - effective_position
        ).to_numpy()
        publish_date = pd.to_datetime(
            result.loc[indices, "report_industry_publish_time"], errors="coerce"
        ).dt.normalize()
        result.loc[indices, "report_industry_age_calendar_days"] = (
            decision - publish_date
        ).dt.days.to_numpy()

    source_cutoff = (
        history["report_industry_publish_time"].max()
        if not history.empty
        else pd.NaT
    )
    result["report_industry_available"] = result[
        "report_fine_industry_pit"
    ].notna()
    result["report_industry_source_cutoff"] = source_cutoff
    result["report_industry_asof_rule"] = (
        "publishDate; first usable on next trading session (strict T+1)"
    )
    return _finish_panel(result)


def _deduplicate_symbol_map(
    frame: pd.DataFrame, columns: Sequence[str], *, source_name: str
) -> pd.DataFrame:
    subset = frame[["symbol", *columns]].dropna(subset=["symbol"]).drop_duplicates()
    conflicts = subset.groupby("symbol", sort=False).size().gt(1)
    if conflicts.any():
        examples = conflicts[conflicts].index[:5].tolist()
        raise ValueError(f"{source_name} has conflicting symbol rows: {examples}")
    return subset.drop_duplicates("symbol", keep="last")


def add_static_sw_labels(
    panel: pd.DataFrame,
    stock_sw_industry: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Attach the current Shenwan L1/fine snapshot as explicit oracle fields."""

    result = _prepare_panel(panel)
    frame = _read_csv_or_frame(stock_sw_industry)
    required = {"ts_code", "sw_industry", "fine_industry"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"stock_sw_industry is missing required columns: {missing}")
    mapping = pd.DataFrame(index=frame.index)
    mapping["symbol"] = frame["ts_code"].map(_standard_symbol)
    mapping["sw_l1_static_oracle"] = (
        frame["sw_industry"].astype("string").str.strip().replace("", pd.NA)
    )
    mapping["sw_fine_static_oracle"] = (
        frame["fine_industry"].astype("string").str.strip().replace("", pd.NA)
    )
    mapping = _deduplicate_symbol_map(
        mapping,
        ["sw_l1_static_oracle", "sw_fine_static_oracle"],
        source_name="stock_sw_industry",
    ).set_index("symbol")
    symbols = result["symbol"].astype(str)
    result["sw_l1_static_oracle"] = symbols.map(mapping["sw_l1_static_oracle"])
    result["sw_fine_static_oracle"] = symbols.map(mapping["sw_fine_static_oracle"])
    result["sw_static_oracle_available"] = result["sw_l1_static_oracle"].notna()
    result["sw_static_asof_rule"] = (
        "current/static snapshot without effective dates; oracle experiment only"
    )
    return _finish_panel(result)


def load_sw_index_history(
    directory: str | Path,
    *,
    index_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load the 31 local Shenwan L1 index parquet files into a long frame."""

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(root)
    selected = None if index_codes is None else {str(code) for code in index_codes}
    rows: list[pd.DataFrame] = []
    for path in sorted(root.glob("*.parquet")):
        if not re.fullmatch(r"\d{6}", path.stem):
            continue
        if selected is not None and path.stem not in selected:
            continue
        frame = pd.read_parquet(path, columns=["close", "amount", "name"])
        if frame.empty:
            continue
        piece = frame.reset_index()
        date_column = piece.columns[0]
        piece = piece.rename(columns={date_column: "trade_date", "name": "sw_l1_name"})
        piece["sw_index_code"] = path.stem
        rows.append(piece[["trade_date", "sw_index_code", "sw_l1_name", "close", "amount"]])
    if not rows:
        return pd.DataFrame(
            columns=["trade_date", "sw_index_code", "sw_l1_name", "close", "amount"]
        )
    out = pd.concat(rows, ignore_index=True)
    out["trade_date"] = _date_series(out["trade_date"])
    out["sw_l1_name"] = out["sw_l1_name"].astype("string").str.strip()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    return out.dropna(subset=["trade_date", "sw_index_code", "sw_l1_name"])


def _prepare_sw_index_feature_table(
    sw_index_history: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    required = {"trade_date", "sw_index_code", "sw_l1_name", "close", "amount"}
    missing = sorted(required.difference(sw_index_history.columns))
    if missing:
        raise ValueError(f"sw_index_history is missing required columns: {missing}")
    history = sw_index_history.copy()
    history["trade_date"] = _date_series(history["trade_date"])
    history["sw_index_code"] = history["sw_index_code"].astype("string").str.strip()
    history["sw_l1_name"] = history["sw_l1_name"].astype("string").str.strip()
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history["amount"] = pd.to_numeric(history["amount"], errors="coerce")
    history = history.dropna(subset=["trade_date", "sw_index_code", "sw_l1_name"])
    duplicates = history.duplicated(["trade_date", "sw_index_code"], keep=False)
    if duplicates.any():
        raise ValueError("sw_index_history must be unique by (trade_date, sw_index_code)")

    pieces: list[pd.DataFrame] = []
    for code, group in history.groupby("sw_index_code", sort=True):
        names = group["sw_l1_name"].dropna().unique()
        if len(names) != 1:
            raise ValueError(f"sw_index_code {code} has inconsistent industry names")
        aligned = group.set_index("trade_date")[["close", "amount"]].reindex(trading_dates)
        close = aligned["close"]
        amount = aligned["amount"]
        feature = pd.DataFrame(index=trading_dates)
        feature["sw_index_code"] = str(code)
        feature["sw_l1_name"] = str(names[0])
        # These rows are indexed by the last observable feature date.  The join
        # to a decision uses the exact previous session, never the T row.
        feature["sw_l1_ret5_t1"] = close / close.shift(5) - 1.0
        feature["sw_l1_ret20_t1"] = close / close.shift(20) - 1.0
        prior20 = amount.shift(1).rolling(20, min_periods=16).mean()
        feature["sw_l1_amount_ratio20_t1"] = amount / prior20
        recent5 = amount.rolling(5, min_periods=4).mean()
        earlier20 = amount.shift(5).rolling(20, min_periods=16).mean()
        feature["sw_l1_amount_accel5_20_t1"] = recent5 / earlier20
        feature["sw_index_feature_date"] = feature.index
        pieces.append(feature.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _previous_session_map(
    decisions: pd.Series, trading_dates: pd.DatetimeIndex
) -> pd.Series:
    positions = trading_dates.get_indexer(pd.DatetimeIndex(decisions))
    if (positions < 0).any():
        missing = sorted(pd.DatetimeIndex(decisions)[positions < 0].unique())
        raise ValueError(
            "trading_dates is missing decision sessions: "
            + ", ".join(str(date.date()) for date in missing)
        )
    values = np.full(len(decisions), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = positions > 0
    values[valid] = trading_dates.to_numpy()[positions[valid] - 1]
    return pd.Series(values, index=decisions.index)


def add_sw_index_t1_features(
    panel: pd.DataFrame,
    sw_index_history: pd.DataFrame | str | Path,
    trading_dates: Sequence[object] | pd.Index,
    *,
    industry_column: str = "sw_l1_static_oracle",
) -> pd.DataFrame:
    """Join Shenwan L1 5/20-day return and amount acceleration through T-1.

    ``industry_column`` defaults to the explicitly static/oracle classification.
    A different, genuinely PIT L1 column can be supplied by future callers.
    """

    result = _prepare_panel(panel)
    if industry_column not in result.columns:
        raise ValueError(f"panel is missing industry column: {industry_column}")
    calendar = _calendar(trading_dates)
    if isinstance(sw_index_history, pd.DataFrame):
        history = sw_index_history.copy()
    else:
        history = load_sw_index_history(sw_index_history)
    feature = _prepare_sw_index_feature_table(history, calendar)
    result["sw_index_required_date"] = _previous_session_map(
        result["decision_date"], calendar
    )
    key_columns = ["sw_index_required_date", industry_column]
    feature = feature.rename(
        columns={"sw_index_feature_date": "sw_index_required_date", "sw_l1_name": industry_column}
    )
    keep = [
        "sw_index_required_date",
        industry_column,
        "sw_index_code",
        "sw_l1_ret5_t1",
        "sw_l1_ret20_t1",
        "sw_l1_amount_ratio20_t1",
        "sw_l1_amount_accel5_20_t1",
    ]
    feature = feature[keep].drop_duplicates(key_columns, keep="last")
    result = result.merge(feature, how="left", on=key_columns, sort=False, validate="m:1")
    result["sw_index_feature_available"] = result[
        ["sw_l1_ret5_t1", "sw_l1_ret20_t1"]
    ].notna().all(axis=1)
    result["sw_index_source_cutoff"] = (
        _date_series(history["trade_date"]).max() if not history.empty else pd.NaT
    )
    result["sw_index_asof_rule"] = "exact T-1 index close/amount; no T close/amount"
    return _finish_panel(result)


def _read_daily_basic_dates(
    source: pd.DataFrame | str | Path,
    required_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    columns = ["ts_code", "trade_date", *_DAILY_BASIC_VALUE_COLUMNS]
    if isinstance(source, pd.DataFrame):
        missing = sorted(set(columns).difference(source.columns))
        if missing:
            raise ValueError(f"daily_basic is missing required columns: {missing}")
        frame = source[columns].copy()
        parsed = _date_series(frame["trade_date"])
        return frame.loc[parsed.isin(required_dates)].copy()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"ts_code": "string", "trade_date": "string"},
        chunksize=500_000,
        low_memory=False,
    ):
        parsed = _date_series(chunk["trade_date"])
        take = parsed.isin(required_dates)
        if take.any():
            chunks.append(chunk.loc[take].copy())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)


def add_daily_basic_t1_features(
    panel: pd.DataFrame,
    daily_basic: pd.DataFrame | str | Path,
    trading_dates: Sequence[object] | pd.Index,
) -> pd.DataFrame:
    """Join exact-T-1 turnover, free-turnover, volume ratio and float MV."""

    result = _prepare_panel(panel)
    calendar = _calendar(trading_dates)
    result["daily_basic_required_date"] = _previous_session_map(
        result["decision_date"], calendar
    )
    required_dates = set(result["daily_basic_required_date"].dropna().map(_normal_date))
    frame = _read_daily_basic_dates(daily_basic, required_dates)
    frame["daily_basic_required_date"] = _date_series(frame["trade_date"])
    frame["symbol"] = frame["ts_code"].map(_standard_symbol).astype("string")
    for column in _DAILY_BASIC_VALUE_COLUMNS:
        frame[f"db_{column}_t1"] = pd.to_numeric(frame[column], errors="coerce")
    keep = [
        "daily_basic_required_date",
        "symbol",
        *(f"db_{column}_t1" for column in _DAILY_BASIC_VALUE_COLUMNS),
    ]
    frame = frame[keep].dropna(subset=["daily_basic_required_date", "symbol"])
    duplicates = frame.duplicated(
        ["daily_basic_required_date", "symbol"], keep=False
    )
    if duplicates.any():
        raise ValueError("daily_basic has duplicate rows for a stock/session")
    result = result.merge(
        frame,
        how="left",
        on=["daily_basic_required_date", "symbol"],
        sort=False,
        validate="1:1",
    )
    value_columns = [f"db_{column}_t1" for column in _DAILY_BASIC_VALUE_COLUMNS]
    result["daily_basic_feature_available"] = result[value_columns].notna().any(axis=1)
    result["daily_basic_asof_rule"] = (
        "exact previous trading session (T-1); circ_mv unit follows Tushare: 10k CNY"
    )
    return _finish_panel(result)


def _sanitize_index_code(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip())


def add_static_tech_index_tags(
    panel: pd.DataFrame,
    constituents: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Attach CSI technology-index memberships as static multi-label tags."""

    result = _prepare_panel(panel)
    frame = _read_csv_or_frame(constituents)
    code_column = "index_code" if "index_code" in frame.columns else "指数代码"
    name_column = "index_name" if "index_name" in frame.columns else "指数名称"
    symbol_column = "成分券代码"
    date_column = "日期"
    required = {code_column, name_column, symbol_column, date_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"tech constituents is missing required columns: {missing}")
    source = pd.DataFrame(index=frame.index)
    source["symbol"] = frame[symbol_column].map(_standard_symbol)
    source["index_code"] = frame[code_column].astype("string").str.strip()
    source["index_name"] = frame[name_column].astype("string").str.strip()
    source["snapshot_date"] = _date_series(frame[date_column])
    source = source.dropna(subset=["symbol", "index_code", "index_name", "snapshot_date"])
    if source.empty:
        raise ValueError("tech constituents has no valid membership rows")
    snapshot_date = source["snapshot_date"].max()
    source = source.loc[source["snapshot_date"].eq(snapshot_date)].drop_duplicates(
        ["symbol", "index_code"], keep="last"
    )
    codes = sorted(source["index_code"].astype(str).unique())
    grouped_codes = source.groupby("symbol", sort=False)["index_code"].agg(
        lambda values: "|".join(sorted(set(map(str, values))))
    )
    grouped_names = source.groupby("symbol", sort=False)["index_name"].agg(
        lambda values: "|".join(sorted(set(map(str, values))))
    )
    symbols = result["symbol"].astype(str)
    result["tech_index_codes_static_oracle"] = symbols.map(grouped_codes).fillna("")
    result["tech_index_names_static_oracle"] = symbols.map(grouped_names).fillna("")
    result["tech_index_count_static_oracle"] = (
        result["tech_index_codes_static_oracle"].str.count(r"\|")
        + result["tech_index_codes_static_oracle"].ne("").astype(int)
    )
    result["tech_index_member_static_oracle"] = result[
        "tech_index_count_static_oracle"
    ].gt(0)
    for code in codes:
        column = f"tech_member__{_sanitize_index_code(code)}_static_oracle"
        members = set(
            source.loc[source["index_code"].astype(str).eq(code), "symbol"].astype(str)
        )
        result[column] = symbols.isin(members)
    result["tech_index_snapshot_date"] = snapshot_date
    result["tech_index_static_asof_rule"] = (
        "single current/static constituent snapshot; oracle experiment only"
    )
    return _finish_panel(result)


def add_industry_context_features(
    panel: pd.DataFrame,
    trading_dates: Sequence[object] | pd.Index,
    *,
    reports: pd.DataFrame | str | Path | None = None,
    stock_sw_industry: pd.DataFrame | str | Path | None = None,
    sw_index_history: pd.DataFrame | str | Path | None = None,
    daily_basic: pd.DataFrame | str | Path | None = None,
    tech_constituents: pd.DataFrame | str | Path | None = None,
    paths: IndustryFeaturePaths | None = None,
) -> pd.DataFrame:
    """Attach all five industry/context feature groups in a deterministic order."""

    paths = paths or IndustryFeaturePaths()
    result = add_report_industry_pit(
        panel, reports if reports is not None else paths.reports_file, trading_dates
    )
    result = add_static_sw_labels(
        result,
        stock_sw_industry if stock_sw_industry is not None else paths.static_sw_file,
    )
    result = add_sw_index_t1_features(
        result,
        sw_index_history if sw_index_history is not None else paths.sw_index_root,
        trading_dates,
    )
    result = add_daily_basic_t1_features(
        result,
        daily_basic if daily_basic is not None else paths.daily_basic_file,
        trading_dates,
    )
    result = add_static_tech_index_tags(
        result,
        tech_constituents
        if tech_constituents is not None
        else paths.tech_constituents_file,
    )
    return result


__all__ = [
    "IndustryFeaturePaths",
    "add_daily_basic_t1_features",
    "add_industry_context_features",
    "add_report_industry_pit",
    "add_static_sw_labels",
    "add_static_tech_index_tags",
    "add_sw_index_t1_features",
    "load_sw_index_history",
    "prepare_report_industry_history",
]
