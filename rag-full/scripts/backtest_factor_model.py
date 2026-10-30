"""Backtest composite factor top/bottom quantiles on the local panel."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.market.data_loader import DEFAULT_TUSHARE_DATA_DIR, load_panel
from src.market.factor_signal import (
    DEFAULT_FACTOR_DIRECTIONS,
    _SKIP_NEUTRALIZE,
    _fit_neutralization,
    _winsorize,
)


LOGGER = logging.getLogger(__name__)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def _summarize_returns(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "mean": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "win_rate": np.nan,
            "max_drawdown": np.nan,
            "cumulative": np.nan,
        }

    mean = float(returns.mean())
    annual_return = float((1.0 + mean) ** periods_per_year - 1.0)
    annual_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(annual_return / annual_vol) if annual_vol > 0 else np.nan
    return {
        "mean": mean,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "win_rate": float((returns > 0).mean()),
        "max_drawdown": _max_drawdown(returns),
        "cumulative": float((1.0 + returns).prod() - 1.0),
    }


def _factor_percentiles(
    cross_section: pd.DataFrame,
    factors: list[str],
    *,
    neutralize: bool,
) -> pd.DataFrame:
    scores = pd.DataFrame(index=cross_section.index)
    for factor in factors:
        if factor not in cross_section.columns:
            continue

        if neutralize and factor not in _SKIP_NEUTRALIZE:
            winsorized = _winsorize(cross_section[factor])
            residuals, _ = _fit_neutralization(
                cross_section.assign(**{factor: winsorized}), factor
            )
            values = pd.Series(np.nan, index=cross_section.index, dtype=float)
            values.loc[residuals.index] = residuals
        else:
            values = cross_section[factor].astype(float)

        percentile = values.rank(pct=True)
        direction = DEFAULT_FACTOR_DIRECTIONS.get(factor, 1)
        scores[factor] = percentile if direction >= 0 else 1.0 - percentile

    return scores


def run_backtest(
    *,
    data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR,
    start: str | None = None,
    end: str | None = None,
    quantile: float = 0.2,
    holding_days: int = 5,
    neutralize: bool = True,
    min_names: int = 80,
    factors: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run daily HS300 top/bottom quantile backtest."""
    factors = factors or list(DEFAULT_FACTOR_DIRECTIONS)
    panel = load_panel(data_dir)
    data = panel[panel["is_hs300"] == True].copy()
    if start:
        data = data[data["trade_date"] >= pd.to_datetime(start)]
    if end:
        data = data[data["trade_date"] <= pd.to_datetime(end)]

    rows: list[dict[str, Any]] = []
    for date, cross_section in data.groupby("trade_date", sort=True):
        cols = ["ts_code", "industry", "fac_log_mv", "y_future_5d", *factors]
        available_cols = []
        for col in cols:
            if col in cross_section.columns and col not in available_cols:
                available_cols.append(col)
        cs = cross_section[available_cols].copy()
        cs = cs.dropna(subset=["y_future_5d"])
        if len(cs) < min_names:
            continue

        scores = _factor_percentiles(cs, factors, neutralize=neutralize)
        composite = scores.mean(axis=1, skipna=True)
        valid = cs.assign(composite_score=composite).dropna(subset=["composite_score"])
        if len(valid) < min_names:
            continue

        top_cutoff = valid["composite_score"].quantile(1.0 - quantile)
        bottom_cutoff = valid["composite_score"].quantile(quantile)
        top = valid[valid["composite_score"] >= top_cutoff]
        bottom = valid[valid["composite_score"] <= bottom_cutoff]
        if top.empty or bottom.empty:
            continue

        top_ret = float(top["y_future_5d"].mean())
        bottom_ret = float(bottom["y_future_5d"].mean())
        rows.append(
            {
                "trade_date": date,
                "n_universe": int(len(valid)),
                "n_top": int(len(top)),
                "n_bottom": int(len(bottom)),
                "top_return": top_ret,
                "bottom_return": bottom_ret,
                "long_short_return": top_ret - bottom_ret,
                "top_cutoff": float(top_cutoff),
                "bottom_cutoff": float(bottom_cutoff),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No valid backtest rows generated")

    periods_per_year = 252.0 / holding_days
    summary = {
        "start": results["trade_date"].min().strftime("%Y-%m-%d"),
        "end": results["trade_date"].max().strftime("%Y-%m-%d"),
        "periods": int(len(results)),
        "quantile": quantile,
        "holding_days": holding_days,
        "neutralize": neutralize,
        "factors": factors,
        "top": _summarize_returns(results["top_return"], periods_per_year),
        "bottom": _summarize_returns(results["bottom_return"], periods_per_year),
        "long_short": _summarize_returns(results["long_short_return"], periods_per_year),
    }
    return results, summary


def _fmt_pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value * 100:.2f}%"


def _fmt_num(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def render_markdown(summary: dict[str, Any], csv_path: Path | None = None) -> str:
    """Render a compact Markdown backtest report."""
    lines = [
        "# 因子模型分层回测",
        "",
        f"- 区间：{summary['start']} 至 {summary['end']}",
        f"- 样本期数：{summary['periods']}",
        f"- 股票池：沪深300成分股日截面",
        f"- 分组：top/bottom {summary['quantile']:.0%}",
        f"- 前瞻收益：y_future_{summary['holding_days']}d",
        f"- 中性化：{'行业 + log(市值)' if summary['neutralize'] else '未中性化'}",
        f"- 因子：{', '.join(summary['factors'])}",
        "",
        "| 组合 | 单期均值 | 年化收益 | 年化波动 | Sharpe | 胜率 | 最大回撤 | 累计收益 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = [
        ("Top", summary["top"]),
        ("Bottom", summary["bottom"]),
        ("Long-Short", summary["long_short"]),
    ]
    for label, stats in labels:
        lines.append(
            "| {label} | {mean} | {ann} | {vol} | {sharpe} | {win} | {mdd} | {cum} |".format(
                label=label,
                mean=_fmt_pct(stats["mean"]),
                ann=_fmt_pct(stats["annual_return"]),
                vol=_fmt_pct(stats["annual_vol"]),
                sharpe=_fmt_num(stats["sharpe"]),
                win=_fmt_pct(stats["win_rate"]),
                mdd=_fmt_pct(stats["max_drawdown"]),
                cum=_fmt_pct(stats["cumulative"]),
            )
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 该回测使用每日滚动 5 日前瞻收益，观察值存在重叠，年化指标用于横向比较，不等同于可直接交易收益。",
            "- 当前组合未计交易成本、停牌/涨跌停约束、换手约束和行业暴露限制。",
            "- 中性化逻辑与投研报告中的行业 + 市值 OLS 残差口径保持一致。",
        ]
    )
    if csv_path:
        lines.append(f"- 明细 CSV：`{csv_path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_TUSHARE_DATA_DIR))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument("--min-names", type=int, default=80)
    parser.add_argument("--no-neutralize", action="store_true")
    parser.add_argument("--output", default="reports/factor_backtest.md")
    parser.add_argument("--csv-output", default="reports/factor_backtest_daily.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results, summary = run_backtest(
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        quantile=args.quantile,
        holding_days=args.holding_days,
        neutralize=not args.no_neutralize,
        min_names=args.min_names,
    )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = render_markdown(summary, csv_path=csv_path)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {output_path}")
    print(f"Saved daily results: {csv_path}")


if __name__ == "__main__":
    main()

