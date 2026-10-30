"""Why TWAP execution costs so much: is it real alpha decay or bad prices?

Two things could explain the 88% -> 39% gap. Either the strategy's edge is
genuinely concentrated in the first 15 minutes (so buying later gives it
away), or the TWAP panel carries junk on thin names. This measures both:

  1. TWAP-vs-open dispersion restricted to cells the strategy can actually
     trade (liquidity gate + finite return), not the whole panel.
  2. The selected basket's own open -> TWAP move, which is exactly the
     alpha handed over by waiting 15 minutes.
"""

import json

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, load_env

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
TWAP_PANEL = f"{CACHE}/twap_0930_0945.parquet"
DEV_END, K = 1213, 10


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    P = np.load(PANEL)
    tw = pd.read_parquet(TWAP_PANEL).reindex(
        index=pd.DatetimeIndex(dates)).values

    meta = json.load(open(f"{ZOO}/meta.json"))
    zi = {s: i for i, s in enumerate(meta["symbols"])}
    zdates = pd.DatetimeIndex(meta["dates"])
    keep = zdates.get_indexer(pd.DatetimeIndex(dates))
    assert (keep >= 0).all()
    # FIXED: the first run sliced the zoo grid with [:len(dates)], which is
    # date-shifted garbage; every number it printed is superseded by this run
    op = np.load(f"{ZOO}/continuous_open.npy")[
        :, [zi[s] for s in syms]][keep]
    op = np.where(op > 0, op, np.nan)

    with np.errstate(invalid="ignore"):
        gap = tw / op - 1.0                      # open -> TWAP move
    tradable = G & np.isfinite(gap) & np.isfinite(
        np.vstack([O2O, np.full((1, O2O.shape[1]), np.nan)]))
    g = gap[tradable]
    print(f"可交易样本数 {len(g):,}")
    for q in (0.1, 1, 25, 50, 75, 99, 99.9):
        print(f"  p{q:<5} {np.nanpercentile(g, q)*1e4:9.1f} bp")
    print(f"  mean {np.nanmean(g)*1e4:.1f} bp   "
          f"std {np.nanstd(g)*1e4:.1f} bp")

    # the basket's own open -> TWAP move, day by day
    basket, bench = [], []
    for t in range(DEV_END, len(dates) - 2):
        s = np.where(G[t] & np.isfinite(O2O[t + 1]), P[t], np.nan)
        idx = np.where(np.isfinite(s))[0]
        if len(idx) < K:
            continue
        top = idx[np.argsort(s[idx])][-K:]
        basket.append(np.nanmean(gap[t + 1, top]))
        bench.append(np.nanmean(gap[t + 1, idx]))
    b = np.array(basket, dtype=float)
    u = np.array(bench, dtype=float)
    print(f"\n选中篮子 开盘->TWAP 平均移动 {np.nanmean(b)*1e4:+.1f} bp/日"
          f"  (中位 {np.nanmedian(b)*1e4:+.1f})")
    print(f"全池平均         {np.nanmean(u)*1e4:+.1f} bp/日")
    print(f"超额(篮子-全池)  {np.nanmean(b - u)*1e4:+.1f} bp/日"
          f"  → 年化约 {np.nanmean(b - u)*252*100:+.1f}%")

    json.dump({
        "tradable_cells": int(len(g)),
        "gap_bp": {f"p{q}": round(float(np.nanpercentile(g, q) * 1e4), 1)
                   for q in (0.1, 1, 25, 50, 75, 99, 99.9)},
        "gap_mean_bp": round(float(np.nanmean(g) * 1e4), 2),
        "gap_std_bp": round(float(np.nanstd(g) * 1e4), 2),
        "basket_open_to_twap_bp_per_day": round(float(np.nanmean(b) * 1e4), 2),
        "pool_open_to_twap_bp_per_day": round(float(np.nanmean(u) * 1e4), 2),
        "excess_bp_per_day": round(float(np.nanmean(b - u) * 1e4), 2),
        "excess_annualised_pct": round(float(np.nanmean(b - u) * 252 * 100), 1),
    }, open(f"{OUT}/twap_diag.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
