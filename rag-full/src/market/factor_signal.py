"""Factor scoring utilities for the investment research workflow.

Supports two modes:
  - neutralize=False  原始分位（向前兼容）
  - neutralize=True   行业 + 市值中性化后的纯 alpha 分位（默认）

中性化步骤：
  1. Winsorize：去极值，截断在 均值 ± 3σ
  2. OLS 回归：factor ~ 行业哑变量 + log(市值)
  3. 取残差，计算截面分位
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.market.data_loader import (
    DEFAULT_TUSHARE_DATA_DIR,
    load_factor_summary,
    load_panel,
    normalize_ts_code,
)


DEFAULT_FACTOR_DIRECTIONS = {
    "fac_eps_revision_60d": 1,
    "fac_turn_20d": 1,
    "fac_vol_20d": 1,
    "fac_eps_yield": 1,
    "fac_log_mv": 1,
    "fac_mf_20d": 1,
    "fac_ep": 1,
    "fac_bp": 1,
}

# fac_log_mv IS market cap — neutralizing it against itself is circular.
# Skip neutralization for this factor.
_SKIP_NEUTRALIZE = {"fac_log_mv"}


def _percentile_rank(series: pd.Series, value: float) -> float | None:
    """Return percentile rank in [0, 1]."""
    valid = series.dropna()
    if valid.empty or pd.isna(value):
        return None
    return float((valid <= value).mean())


def _winsorize(series: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Clip values beyond mean ± n_std * std."""
    valid = series.dropna()
    if len(valid) < 10:
        return series
    mean, std = valid.mean(), valid.std()
    if std == 0:
        return series
    return series.clip(mean - n_std * std, mean + n_std * std)


def _fit_neutralization(
    cross_section: pd.DataFrame,
    factor_col: str,
) -> tuple[pd.Series, tuple | None]:
    """OLS: factor ~ industry_dummies + fac_log_mv  →  (residuals, model).

    model = (dummy_column_names, coefficients) for projecting out-of-sample stocks.
    Returns (raw_series, None) if not enough data.
    """
    data = cross_section[[factor_col, "industry", "fac_log_mv"]].dropna()
    if len(data) < 30:
        return cross_section[factor_col], None

    dummies = pd.get_dummies(data["industry"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(data)), dummies.values, data["fac_log_mv"].values])
    y = data[factor_col].values

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return cross_section[factor_col], None

    residuals = pd.Series(y - X @ coeffs, index=data.index, name=factor_col)
    model = (list(dummies.columns), coeffs)
    return residuals, model


def _apply_model(raw_value: float, industry: str, logmv: float,
                 model: tuple | None) -> float:
    """Compute neutralized residual for a stock using a fitted OLS model."""
    if model is None or pd.isna(raw_value) or pd.isna(logmv):
        return raw_value
    dummy_cols, coeffs = model
    ind_vec = np.zeros(len(dummy_cols))
    if industry in dummy_cols:
        ind_vec[dummy_cols.index(industry)] = 1.0
    x = np.concatenate([[1.0], ind_vec, [logmv]])
    return float(raw_value - x @ coeffs)


def get_industry_peers(
    stock_code: str,
    *,
    data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR,
    top_n: int = 5,
) -> str:
    """Return a Markdown table of top-N peers in the same industry.

    Computes neutralized composite scores for all HS300 stocks in the same
    industry and ranks them. Used as extra context for the LLM.
    """
    ts_code = normalize_ts_code(stock_code)
    panel = load_panel(data_dir)
    date = panel["trade_date"].max()

    cross_section = panel[
        (panel["trade_date"] == date) & (panel["is_hs300"] == True)
    ].copy()

    # Find target stock's industry
    stock_rows = cross_section[cross_section["ts_code"] == ts_code]
    if stock_rows.empty:
        # Stock might not be in HS300 — look it up in the full panel
        full_row = panel[(panel["trade_date"] == date) & (panel["ts_code"] == ts_code)]
        if full_row.empty:
            return ""
        industry = str(full_row.iloc[0].get("industry", ""))
    else:
        industry = str(stock_rows.iloc[0].get("industry", ""))

    if not industry:
        return ""

    peers = cross_section[cross_section["industry"] == industry].copy()
    if len(peers) < 2:
        return ""

    factors = list(DEFAULT_FACTOR_DIRECTIONS.keys())
    rows = []
    for _, row in peers.iterrows():
        scores = []
        for fac in factors:
            if fac not in peers.columns:
                continue
            val = row.get(fac)
            if pd.isna(val):
                continue
            winsorized = _winsorize(peers[fac])
            residuals, model = _fit_neutralization(
                peers.assign(**{fac: winsorized}), fac
            )
            if row.name in residuals.index:
                neu_val = float(residuals[row.name])
            else:
                neu_val = float(val)
            pct = _percentile_rank(residuals, neu_val)
            if pct is not None:
                direction = DEFAULT_FACTOR_DIRECTIONS.get(fac, 1)
                scores.append(pct if direction >= 0 else 1 - pct)

        composite = float(np.mean(scores)) if scores else None
        rows.append({
            "ts_code": row["ts_code"],
            "name": panel[panel["ts_code"] == row["ts_code"]]["ts_code"].iloc[0],
            "composite": composite,
            "eps_rev": _percentile_rank(peers["fac_eps_revision_60d"], row.get("fac_eps_revision_60d")),
            "ep": _percentile_rank(peers["fac_ep"], row.get("fac_ep")),
            "bp": _percentile_rank(peers["fac_bp"], row.get("fac_bp")),
        })

    # Get company names
    name_map = (
        panel[panel["trade_date"] == date][["ts_code"]]
        .drop_duplicates()
        .set_index("ts_code")
    )
    # Use stock_basic if available, else use ts_code
    from src.market.data_loader import get_stock_info
    for r in rows:
        try:
            info = get_stock_info(r["ts_code"])
            r["name"] = info.get("name") or r["ts_code"]
        except Exception:
            r["name"] = r["ts_code"]

    df = pd.DataFrame(rows).sort_values("composite", ascending=False).head(top_n + 1)

    def _pct(v):
        return f"{v * 100:.1f}%" if v is not None and not pd.isna(v) else "N/A"

    lines = [
        f"\n## {industry}行业对比（沪深300成分，{date.strftime('%Y-%m-%d')}）\n",
        "| 股票 | 综合得分 | EPS修正 | EP分位 | BP分位 | 标记 |",
        "|------|---------|--------|--------|--------|------|",
    ]
    for r in df.to_dict("records"):
        marker = " ← 当前" if r["ts_code"] == ts_code else ""
        lines.append(
            f"| {r['name']}({r['ts_code']}) "
            f"| {_pct(r['composite'])} "
            f"| {_pct(r['eps_rev'])} "
            f"| {_pct(r['ep'])} "
            f"| {_pct(r['bp'])} "
            f"|{marker} |"
        )
    return "\n".join(lines) + "\n"


def calculate_factor_score(
    stock_code: str,
    *,
    data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR,
    as_of: str | None = None,
    factors: list[str] | None = None,
    neutralize: bool = True,
) -> dict:
    """Calculate cross-sectional factor percentiles and composite score.

    Args:
        neutralize: If True (default), remove industry and market-cap effects
            before computing percentile ranks. This gives a cleaner alpha signal
            by stripping out sector-level and size-level noise.
    """
    ts_code = normalize_ts_code(stock_code)
    panel = load_panel(data_dir)

    if as_of:
        target_date = pd.to_datetime(as_of)
        available_dates = panel.loc[panel["trade_date"] <= target_date, "trade_date"]
        if available_dates.empty:
            raise ValueError(f"No panel date <= {as_of}")
        date = available_dates.max()
    else:
        date = panel["trade_date"].max()

    cross_section = panel[(panel["trade_date"] == date) & (panel["is_hs300"] == True)].copy()
    stock_row = panel[(panel["trade_date"] == date) & (panel["ts_code"] == ts_code)]
    if stock_row.empty:
        raise ValueError(f"No factor row found for {ts_code} on {date:%Y-%m-%d}")

    factor_summary = load_factor_summary(data_dir)
    factors = factors or list(DEFAULT_FACTOR_DIRECTIONS)

    stock = stock_row.iloc[0]
    stock_industry = str(stock.get("industry", ""))
    stock_logmv = float(stock.get("fac_log_mv", np.nan))

    details = []
    scores = []

    for factor in factors:
        if factor not in panel.columns:
            continue

        raw_value = stock.get(factor)
        raw_value = None if pd.isna(raw_value) else float(raw_value)

        # --- Raw percentile (always computed for comparison) ---
        raw_percentile = _percentile_rank(cross_section[factor], raw_value)

        # --- Neutralized percentile ---
        if neutralize and factor not in _SKIP_NEUTRALIZE and raw_value is not None:
            winsorized = _winsorize(cross_section[factor])
            residuals, model = _fit_neutralization(
                cross_section.assign(**{factor: winsorized}), factor
            )
            # Get the stock's neutralized value
            if stock_row.index[0] in residuals.index:
                # Stock is in the cross-section: use its own residual
                neu_value = float(residuals[stock_row.index[0]])
            else:
                # Stock not in HS300: project using the fitted model
                raw_w = float(_winsorize(
                    pd.Series([raw_value], name=factor)
                ).iloc[0])
                neu_value = _apply_model(raw_w, stock_industry, stock_logmv, model)

            percentile = _percentile_rank(residuals, neu_value)
        else:
            percentile = raw_percentile
            neu_value = raw_value

        if percentile is None:
            oriented = None
        else:
            direction = DEFAULT_FACTOR_DIRECTIONS.get(factor, 1)
            oriented = percentile if direction >= 0 else 1 - percentile
            scores.append(oriented)

        # IC from factor summary
        rank_ic, rank_icir = None, None
        if not factor_summary.empty and "factor" in factor_summary.columns:
            match = factor_summary[factor_summary["factor"] == factor]
            if not match.empty:
                rank_ic = float(match.iloc[0].get("rank_ic_mean", np.nan))
                rank_icir = float(match.iloc[0].get("rank_icir", np.nan))

        details.append(
            {
                "factor": factor,
                "value": raw_value,
                "percentile": percentile,          # neutralized (or raw if skipped)
                "percentile_raw": raw_percentile,  # always raw, for comparison
                "oriented_score": oriented,
                "rank_ic": rank_ic,
                "rank_icir": rank_icir,
            }
        )

    composite_score = float(np.mean(scores)) if scores else None
    signal = (
        "unknown" if composite_score is None
        else "strong" if composite_score >= 0.65
        else "neutral_positive" if composite_score >= 0.50
        else "neutral_negative" if composite_score >= 0.35
        else "weak"
    )

    return {
        "as_of": date.strftime("%Y-%m-%d"),
        "ts_code": ts_code,
        "composite_score": composite_score,
        "signal": signal,
        "neutralized": neutralize,
        "factor_details": details,
    }

