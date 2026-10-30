"""Exchange-calendar auditing for provider market snapshots."""

from __future__ import annotations

import pandas as pd


def align_to_trading_calendar(
    frame: pd.DataFrame,
    calendar_name: str,
    *,
    drop_non_sessions: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Audit dates, drop known holiday rows, and never invent missing prices."""

    if frame.empty:
        return frame.copy(), []
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("calendar alignment requires a DatetimeIndex")
    try:
        import exchange_calendars as xcals
    except ImportError as exc:  # pragma: no cover - locked dependency
        raise RuntimeError("exchange-calendars is required for session auditing") from exc
    calendar = xcals.get_calendar(calendar_name)
    index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    first_supported = pd.Timestamp(calendar.first_session).tz_localize(None).normalize()
    last_supported = pd.Timestamp(calendar.last_session).tz_localize(None).normalize()
    audit_start = max(index.min(), first_supported)
    audit_end = min(index.max(), last_supported)
    warnings: list[str] = []
    unsupported = index[(index < first_supported) | (index > last_supported)]
    if len(unsupported):
        warnings.append(
            f"{calendar_name}日历仅覆盖{first_supported.date()}至"
            f"{last_supported.date()}；保留{len(unsupported)}条覆盖区间外历史且不作休市审计"
        )
    if audit_start > audit_end:
        return frame.copy(), warnings
    sessions = calendar.sessions_in_range(audit_start, audit_end)
    sessions = pd.DatetimeIndex(sessions).tz_localize(None).normalize()
    supported_index = index[(index >= first_supported) & (index <= last_supported)]
    unexpected = supported_index.difference(sessions)
    missing = sessions.difference(supported_index)
    result = frame.copy()
    if len(unexpected):
        rendered = ",".join(date.strftime("%Y-%m-%d") for date in unexpected[:10])
        suffix = "…" if len(unexpected) > 10 else ""
        warnings.append(
            f"{calendar_name}发现{len(unexpected)}条非交易日记录：{rendered}{suffix}"
        )
        if drop_non_sessions:
            result = result.loc[~index.isin(unexpected)].copy()
    if len(missing):
        warnings.append(
            f"{calendar_name}区间内缺少{len(missing)}个交易日；未进行价格填充"
        )
    return result, warnings


__all__ = ["align_to_trading_calendar"]
