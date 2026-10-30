"""Point-in-time nine-chain membership from analyst reports only.

The classifier in this module is deliberately narrower than the static
semantic upper-bound experiment.  It consumes only a report's security code,
publication timestamp, title, and reported fine-industry text.  A report
published on calendar date ``d`` first becomes usable on the first trading
session strictly after ``d`` (strict ``T+1``), and its evidence expires after
the configured number of trading sessions.

No transaction event, disclosed holding, selected-stock label, realised
return, company name, static Shenwan snapshot, or oracle membership is
accepted by the public API.  The only shared object is the frozen nine-chain
text dictionary from :mod:`afactor.trader_choice_semantic`; matching and the
point-in-time clock are implemented independently here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .trader_choice_semantic import (
    DEFAULT_STATIC_ORACLE_THEMES,
    ThemeDefinition,
)

TRADER_CHAIN_PIT_VERSION = "analyst-report-pit-nine-chain-v1"
TRADER_CHAIN_PIT_ASOF_RULE = (
    "analyst report publishDate; first usable on next trading session "
    "(strict T+1); evidence expires by configured trading-session TTL"
)

# One frozen dictionary is used by the static diagnostic and the deployable
# PIT experiment.  Importing these immutable definitions does not import or
# consult any static membership, disclosed trade, label, or oracle result.
PIT_CHAIN_DEFINITIONS: tuple[ThemeDefinition, ...] = DEFAULT_STATIC_ORACLE_THEMES

_FORBIDDEN_REPORT_COLUMNS = {
    "event_id",
    "event_type",
    "label",
    "label_confidence",
    "direction_confidence",
    "date_confidence",
    "is_selected",
    "selected",
    "is_disclosed_buy",
    "buy_flag",
    "sell_flag",
    "trade_direction",
    "realized_pnl",
    "realised_pnl",
}

_MEMBERSHIP_COLUMNS = (
    "decision_date",
    "symbol",
    "chain_id",
    "chain_name",
    "membership",
    "evidence_age",
    "evidence_age_sessions",
    "source_count",
    "latest_evidence_publish_time",
    "latest_evidence_effective_date",
    "latest_report_fine_industry",
    "matched_keywords",
    "match_rules",
    "evidence_samples",
    "ttl_sessions",
    "mapping_version",
    "asof_rule",
)


@dataclass(frozen=True)
class TraderChainPITConfig:
    """Column names and evidence lifetime for PIT chain membership.

    ``ttl_sessions=N`` means that evidence is active for exactly ``N``
    trading sessions beginning with its strict-T+1 effective session.  Thus,
    its valid ages are ``0`` through ``N - 1``.  ``None`` disables expiry.
    """

    ttl_sessions: int | None = 120
    decision_date_col: str = "decision_date"
    panel_symbol_col: str = "symbol"
    report_symbol_col: str = "stockCode"
    report_publish_col: str = "publishDate"
    report_title_col: str = "title"
    report_fine_industry_col: str = "indvInduName"
    evidence_sample_limit: int = 3

    def __post_init__(self) -> None:
        if self.ttl_sessions is not None and (
            isinstance(self.ttl_sessions, bool) or self.ttl_sessions <= 0
        ):
            raise ValueError("ttl_sessions must be positive or None")
        if self.evidence_sample_limit <= 0:
            raise ValueError("evidence_sample_limit must be positive")
        names = (
            self.decision_date_col,
            self.panel_symbol_col,
            self.report_symbol_col,
            self.report_publish_col,
            self.report_title_col,
            self.report_fine_industry_col,
        )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("configured column names must be non-empty strings")


def _normalised_columns(columns: Iterable[object]) -> set[str]:
    return {str(column).strip().casefold() for column in columns}


def _read_report_columns(
    reports: pd.DataFrame | str | Path,
    config: TraderChainPITConfig,
) -> pd.DataFrame:
    """Read only the four report fields that the classifier is allowed to use."""

    if isinstance(reports, pd.DataFrame):
        available = list(reports.columns)
        source = reports
    else:
        path = Path(reports)
        if not path.exists():
            raise FileNotFoundError(path)
        available = list(pd.read_csv(path, nrows=0).columns)
        source = path

    forbidden = sorted(
        _normalised_columns(available).intersection(_FORBIDDEN_REPORT_COLUMNS)
    )
    if forbidden:
        raise ValueError(
            "analyst reports contain transaction/label columns that this PIT "
            f"classifier is forbidden to read: {forbidden}"
        )

    required = {config.report_symbol_col, config.report_publish_col}
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(f"analyst reports are missing required columns: {missing}")
    text_columns = [
        column
        for column in (config.report_title_col, config.report_fine_industry_col)
        if column in available
    ]
    if not text_columns:
        raise ValueError(
            "analyst reports need a title or fine-industry column: "
            f"{config.report_title_col!r}, {config.report_fine_industry_col!r}"
        )

    usecols = [config.report_symbol_col, config.report_publish_col, *text_columns]
    if isinstance(source, pd.DataFrame):
        return source.loc[:, usecols].copy()
    return pd.read_csv(source, usecols=usecols, low_memory=False)


def _standard_symbol(value: object) -> str | pd.NA:
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
    # CSV readers commonly turn 000001 into 1 or 1.0.
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    digits = "".join(character for character in text if character.isdigit())
    if not digits or len(digits) > 6:
        return pd.NA
    digits = digits.zfill(6)
    if digits.startswith(("6", "900")):
        exchange = "SH"
    elif digits.startswith(("0", "3", "200")):
        exchange = "SZ"
    elif digits.startswith(("4", "8", "9")):
        exchange = "BJ"
    else:
        return pd.NA
    return f"{digits}.{exchange}"


def _datetime_series(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if compact.any():
        result.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce"
        )
    if (~compact).any():
        parsed = pd.to_datetime(text.loc[~compact], errors="coerce", utc=True)
        result.loc[~compact] = parsed.dt.tz_convert(None).to_numpy()
    return result


def _date_series(values: pd.Series) -> pd.Series:
    return _datetime_series(values).dt.normalize()


def _calendar(values: Sequence[object] | pd.Index) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(pd.Index(values), errors="coerce", utc=True)
    parsed = parsed[~parsed.isna()].tz_convert(None).normalize()
    result = pd.DatetimeIndex(parsed).drop_duplicates().sort_values()
    if len(result) < 2:
        raise ValueError("trading_dates must contain at least two valid sessions")
    return result


def _next_sessions(
    raw_dates: pd.Series,
    trading_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Map every publication date to the first session strictly after it."""

    dates = _date_series(raw_dates)
    positions = trading_dates.searchsorted(dates, side="right")
    values = np.full(
        len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    valid = dates.notna().to_numpy() & (positions < len(trading_dates))
    values[valid] = trading_dates.to_numpy()[positions[valid]]
    return pd.Series(values, index=raw_dates.index)


def _keyword_pattern(keywords: Sequence[str]) -> str:
    expressions: list[str] = []
    for keyword in sorted(set(keywords), key=len, reverse=True):
        escaped = re.escape(keyword)
        if keyword.isascii() and keyword.replace("_", "").isalnum():
            expressions.append(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
        else:
            expressions.append(escaped)
    return "(?:" + "|".join(expressions) + ")"


def _definition_match_masks(
    text: pd.Series,
    definition: ThemeDefinition,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    direct = pd.Series(False, index=text.index, dtype=bool)
    if definition.keywords:
        direct = text.str.contains(
            _keyword_pattern(definition.keywords), case=False, regex=True, na=False
        )
    contextual = pd.Series(False, index=text.index, dtype=bool)
    for rule in definition.cooccurrence_rules:
        rule_match = pd.Series(True, index=text.index, dtype=bool)
        for group in rule:
            rule_match &= text.str.contains(
                _keyword_pattern(group), case=False, regex=True, na=False
            )
        contextual |= rule_match
    if definition.context_exclude_keywords:
        contextual &= ~text.str.contains(
            _keyword_pattern(definition.context_exclude_keywords),
            case=False,
            regex=True,
            na=False,
        )
    matched = direct | contextual
    if definition.exclude_keywords:
        excluded = text.str.contains(
            _keyword_pattern(definition.exclude_keywords),
            case=False,
            regex=True,
            na=False,
        )
        matched &= ~excluded
        direct &= ~excluded
        contextual &= ~excluded
    return matched, direct, contextual


def _matching_keywords(text: str, definition: ThemeDefinition) -> str:
    terms = (
        *definition.keywords,
        *(
            value
            for rule in definition.cooccurrence_rules
            for group in rule
            for value in group
        ),
    )
    matches = [
        keyword
        for keyword in dict.fromkeys(terms)
        if re.search(_keyword_pattern((keyword,)), text, flags=re.IGNORECASE)
    ]
    return "|".join(matches)


def _join_unique(values: Iterable[object], separator: str = "|") -> str:
    cleaned = sorted(
        {
            str(value).strip()
            for value in values
            if value is not None and not pd.isna(value) and str(value).strip()
        }
    )
    return separator.join(cleaned)


def prepare_report_chain_evidence(
    reports: pd.DataFrame | str | Path,
    trading_dates: Sequence[object] | pd.Index,
    *,
    config: TraderChainPITConfig | None = None,
) -> pd.DataFrame:
    """Normalize reports, apply strict T+1, and emit matched report evidence."""

    cfg = config or TraderChainPITConfig()
    calendar = _calendar(trading_dates)
    frame = _read_report_columns(reports, cfg)
    base = pd.DataFrame(index=frame.index)
    base["symbol"] = frame[cfg.report_symbol_col].map(_standard_symbol)
    base["report_publish_time"] = _datetime_series(frame[cfg.report_publish_col])
    base["report_publish_date"] = base["report_publish_time"].dt.normalize()
    base["evidence_effective_date"] = _next_sessions(
        frame[cfg.report_publish_col], calendar
    )
    base["report_title"] = (
        frame[cfg.report_title_col].astype("string").fillna("")
        if cfg.report_title_col in frame.columns
        else ""
    )
    base["report_fine_industry"] = (
        frame[cfg.report_fine_industry_col].astype("string").fillna("")
        if cfg.report_fine_industry_col in frame.columns
        else ""
    )
    base["evidence_text"] = (
        base["report_fine_industry"]
        .str.cat(base["report_title"], sep=" | ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip(" |")
    )
    base = base.dropna(
        subset=["symbol", "report_publish_time", "evidence_effective_date"]
    )
    base = base.loc[base["evidence_text"].ne("")].copy()
    # Exact duplicate rows are one public source, regardless of CSV repetition.
    base = base.drop_duplicates(
        [
            "symbol",
            "report_publish_time",
            "report_title",
            "report_fine_industry",
        ],
        keep="first",
    ).reset_index(drop=True)
    base["_source_id"] = np.arange(len(base), dtype=np.int64)

    output_columns = [
        "symbol",
        "chain_id",
        "chain_name",
        "report_publish_time",
        "report_publish_date",
        "evidence_effective_date",
        "report_fine_industry",
        "match_rule",
        "matched_keywords",
        "evidence_text",
        "source_type",
        "mapping_version",
        "asof_rule",
        "_source_id",
    ]
    matches: list[pd.DataFrame] = []
    text = base["evidence_text"].astype("string")
    for definition in PIT_CHAIN_DEFINITIONS:
        matched, direct, contextual = _definition_match_masks(text, definition)
        if not matched.any():
            continue
        match_rule = pd.Series("context", index=base.index, dtype="string")
        match_rule.loc[direct] = "direct"
        match_rule.loc[direct & contextual] = "direct+context"
        item = base.loc[
            matched,
            [
                "symbol",
                "report_publish_time",
                "report_publish_date",
                "evidence_effective_date",
                "report_fine_industry",
                "evidence_text",
                "_source_id",
            ],
        ].copy()
        item["chain_id"] = definition.theme_id
        item["chain_name"] = definition.theme_name
        item["match_rule"] = match_rule.loc[matched].to_numpy()
        item["matched_keywords"] = item["evidence_text"].map(
            lambda value, rule=definition: _matching_keywords(str(value), rule)
        )
        item["source_type"] = "analyst_report_title_fine_industry"
        item["mapping_version"] = TRADER_CHAIN_PIT_VERSION
        item["asof_rule"] = TRADER_CHAIN_PIT_ASOF_RULE
        matches.append(item[output_columns])

    if not matches:
        return pd.DataFrame(columns=output_columns)
    return (
        pd.concat(matches, ignore_index=True)
        .sort_values(
            ["evidence_effective_date", "symbol", "chain_id", "_source_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _prepare_panel_keys(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: TraderChainPITConfig,
) -> pd.DataFrame:
    required = {config.decision_date_col, config.panel_symbol_col}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    keys = pd.DataFrame(index=panel.index)
    keys["decision_date"] = _date_series(panel[config.decision_date_col])
    keys["symbol"] = panel[config.panel_symbol_col].map(_standard_symbol).astype(
        "string"
    )
    if keys.isna().any(axis=None):
        raise ValueError("panel contains invalid decision dates or symbols")
    if keys.duplicated(["decision_date", "symbol"], keep=False).any():
        raise ValueError("panel must be unique by decision date and symbol")
    absent = pd.DatetimeIndex(keys["decision_date"].unique()).difference(calendar)
    if len(absent):
        raise ValueError(
            "trading_dates is missing decision sessions: "
            + ", ".join(str(date.date()) for date in absent)
        )
    return keys


def build_trader_chain_pit_membership(
    panel: pd.DataFrame,
    reports: pd.DataFrame | str | Path,
    trading_dates: Sequence[object] | pd.Index,
    *,
    config: TraderChainPITConfig | None = None,
) -> pd.DataFrame:
    """Build active multi-label chain membership for each stock/decision date.

    Only active memberships are returned.  Therefore every returned row has
    ``membership=True``.  A missing stock/chain row means there was no
    matching report evidence known at that decision time, or all evidence had
    expired.  ``evidence_age`` is the trading-session age of the freshest
    active source; ``source_count`` counts distinct active report records.
    """

    cfg = config or TraderChainPITConfig()
    calendar = _calendar(trading_dates)
    keys = _prepare_panel_keys(panel, calendar, cfg)
    evidence = prepare_report_chain_evidence(reports, calendar, config=cfg)
    if evidence.empty:
        return pd.DataFrame(columns=_MEMBERSHIP_COLUMNS)

    calendar_position = pd.Series(np.arange(len(calendar)), index=calendar)
    effective_position = evidence["evidence_effective_date"].map(calendar_position)
    evidence = evidence.assign(_effective_position=effective_position.to_numpy())
    rows: list[pd.DataFrame] = []
    for decision, decision_keys in keys.groupby("decision_date", sort=True):
        decision = pd.Timestamp(decision)
        decision_position = int(calendar_position.loc[decision])
        symbols = set(decision_keys["symbol"].astype(str))
        active = evidence.loc[
            evidence["_effective_position"].le(decision_position)
            & evidence["symbol"].isin(symbols)
        ].copy()
        if active.empty:
            continue
        active["_evidence_age"] = (
            decision_position - active["_effective_position"]
        ).astype(np.int64)
        if cfg.ttl_sessions is not None:
            active = active.loc[active["_evidence_age"].lt(cfg.ttl_sessions)]
        if active.empty:
            continue

        group_keys = ["symbol", "chain_id", "chain_name"]
        grouped = active.groupby(group_keys, sort=True)
        aggregated = grouped.agg(
            evidence_age=("_evidence_age", "min"),
            source_count=("_source_id", "nunique"),
            matched_keywords=("matched_keywords", _join_unique),
            match_rules=("match_rule", _join_unique),
            evidence_samples=(
                "evidence_text",
                lambda values: " || ".join(
                    list(dict.fromkeys(str(value) for value in values))[
                        : cfg.evidence_sample_limit
                    ]
                )[:1000],
            ),
        ).reset_index()
        latest = (
            active.sort_values(
                [
                    *group_keys,
                    "evidence_effective_date",
                    "report_publish_time",
                    "_source_id",
                ],
                kind="mergesort",
            )
            .drop_duplicates(group_keys, keep="last")
            .loc[
                :,
                [
                    *group_keys,
                    "report_publish_time",
                    "evidence_effective_date",
                    "report_fine_industry",
                ],
            ]
            .rename(
                columns={
                    "report_publish_time": "latest_evidence_publish_time",
                    "evidence_effective_date": "latest_evidence_effective_date",
                    "report_fine_industry": "latest_report_fine_industry",
                }
            )
        )
        aggregated = aggregated.merge(
            latest, on=group_keys, how="left", validate="one_to_one"
        )
        aggregated.insert(0, "decision_date", decision)
        aggregated["membership"] = True
        aggregated["evidence_age_sessions"] = aggregated["evidence_age"]
        aggregated["ttl_sessions"] = (
            cfg.ttl_sessions if cfg.ttl_sessions is not None else pd.NA
        )
        aggregated["mapping_version"] = TRADER_CHAIN_PIT_VERSION
        aggregated["asof_rule"] = TRADER_CHAIN_PIT_ASOF_RULE
        rows.append(aggregated)

    if not rows:
        return pd.DataFrame(columns=_MEMBERSHIP_COLUMNS)
    result = pd.concat(rows, ignore_index=True)
    result["membership"] = result["membership"].astype(bool)
    result["evidence_age"] = result["evidence_age"].astype("Int64")
    result["evidence_age_sessions"] = result["evidence_age_sessions"].astype(
        "Int64"
    )
    result["source_count"] = result["source_count"].astype("Int64")
    result["ttl_sessions"] = result["ttl_sessions"].astype("Int64")
    return result.loc[:, _MEMBERSHIP_COLUMNS].sort_values(
        ["decision_date", "symbol", "chain_id"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "PIT_CHAIN_DEFINITIONS",
    "TRADER_CHAIN_PIT_ASOF_RULE",
    "TRADER_CHAIN_PIT_VERSION",
    "TraderChainPITConfig",
    "build_trader_chain_pit_membership",
    "prepare_report_chain_evidence",
]
