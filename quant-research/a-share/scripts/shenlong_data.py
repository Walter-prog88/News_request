"""神龙摆尾近端验证 · 数据层。

把本项目已有的三段行情拼成一张 2020-01-02 ~ 2026-08-27 的**不复权** OHLC 面板,
另存复权因子 / ST 标记 / 官方前收盘 / 涨跌停幅度,供形态识别与逐日触价模拟使用。

三段来源:
  A  cache/daily/*.parquet        2020-01-02 ~ 2025-12-31  (分钟数据聚合;前复权,锚在段末)
  B  cache/sealed/*.parquet       2026-01-05 ~ 2026-08-17  (同上,锚在段末)
  C  cache/recent_daily/parts/*   2026-08-18 ~ 2026-08-27  (baostock 原始不复权)
A/B 段用 data_8:21/daily_basic.csv 的不复权收盘逐日逐股反解出比例,再把 O/H/L 一起还原,
因此不依赖对"前复权锚点"的任何猜测。C 段本来就是不复权。

已知局限(写进报告,不在代码里掩盖):
  * 面板 5565 只股票全部存活到 2025 年末 —— 2020-2026 间退市的 ~225 只不在样本内(幸存者偏差)。
  * 成交额用分钟聚合的 amount,不是官方成交量;034 的"放量"条件按成交额执行。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "shenlong" / "panel"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_FIELDS = {"open": "open_auction_px", "high": "high_px", "low": "low_px",
                "close": "close_px", "amount": "amount"}


def _seg(dirname: str) -> dict[str, pd.DataFrame]:
    d = ROOT / "cache" / dirname
    return {k: pd.read_parquet(d / f"{v}.parquet").astype("float64") for k, v in PANEL_FIELDS.items()}


def _recent() -> dict[str, pd.DataFrame]:
    parts = sorted((ROOT / "cache" / "recent_daily" / "parts").glob("*.csv"))
    frames = []
    for p in parts:
        if p.stat().st_size == 0:
            continue
        f = pd.read_csv(p)
        if f.empty:
            continue
        frames.append(f)
    r = pd.concat(frames, ignore_index=True)
    # baostock 代码 sh.600000 -> ts_code 600000.SH
    r["ts_code"] = r.code.str.slice(3) + "." + r.code.str.slice(0, 2).str.upper()
    r["date"] = pd.to_datetime(r.date)
    r = r[r.tradestatus == 1]
    out = {}
    for k, col in (("open", "open"), ("high", "high"), ("low", "low"),
                   ("close", "close"), ("amount", "amount")):
        out[k] = r.pivot(index="date", columns="ts_code", values=col).astype("float64")
    out["_preclose"] = r.pivot(index="date", columns="ts_code", values="preclose").astype("float64")
    out["_isST"] = r.pivot(index="date", columns="ts_code", values="isST").astype("float64")
    return out


def main() -> None:
    print("读 daily_basic(不复权收盘)…", flush=True)
    db = pd.read_csv(ROOT / "data_8:21" / "daily_basic.csv",
                     usecols=["ts_code", "trade_date", "close", "turnover_rate", "circ_mv"],
                     dtype={"ts_code": str})
    db["date"] = pd.to_datetime(db.trade_date, format="%Y%m%d")
    raw_close = db.pivot(index="date", columns="ts_code", values="close").astype("float64")
    turn = db.pivot(index="date", columns="ts_code", values="turnover_rate").astype("float64")
    circ = db.pivot(index="date", columns="ts_code", values="circ_mv").astype("float64")
    del db
    print(f"  raw_close {raw_close.shape} {raw_close.index.min().date()}~{raw_close.index.max().date()}", flush=True)

    print("读 adj_factor…", flush=True)
    af = pd.read_csv(ROOT / "data_8:21" / "adj_factor.csv", dtype={"ts_code": str})
    af["date"] = pd.to_datetime(af.trade_date, format="%Y%m%d")
    cumf = af.pivot(index="date", columns="ts_code", values="adj_factor").astype("float64")
    del af
    print(f"  cumf {cumf.shape}", flush=True)

    segs = []
    for name, dirname in (("A", "daily"), ("B", "sealed")):
        print(f"段 {name}: cache/{dirname}", flush=True)
        s = _seg(dirname)
        idx, cols = s["close"].index, s["close"].columns
        rc = raw_close.reindex(index=idx, columns=cols)
        scale = rc / s["close"]
        # 同一天全市场比例应当高度一致(整段同一锚点);逐股逐日相除本身就是精确还原
        ok = scale.notna()
        print(f"  可还原比例覆盖 {ok.to_numpy().sum()/max(s['close'].notna().to_numpy().sum(),1):.4f}", flush=True)
        out = {k: (s[k] * scale if k != "amount" else s[k]) for k in s}
        out["close"] = rc.where(rc.notna(), out["close"])   # 收盘直接用官方不复权
        segs.append(out)

    print("段 C: cache/recent_daily", flush=True)
    c = _recent()
    segs.append(c)

    cols = sorted(set().union(*[set(s["close"].columns) for s in segs]))
    idx = pd.DatetimeIndex(sorted(set().union(*[set(s["close"].index) for s in segs])))
    panel = {}
    for k in ("open", "high", "low", "close", "amount"):
        acc = pd.DataFrame(np.nan, index=idx, columns=cols)
        for s in segs:
            v = s[k].reindex(index=idx, columns=cols)
            acc = acc.where(v.isna(), v)
        panel[k] = acc.astype("float32")
    print(f"拼接后 {panel['close'].shape} {idx.min().date()}~{idx.max().date()}", flush=True)

    # 官方前收盘:用累计复权因子把除权除息折算进去;C 段直接用 baostock 的 preclose
    cf = cumf.reindex(index=idx, columns=cols).ffill()
    cf = cf.bfill()
    rc = panel["close"].astype("float64")
    pre = rc.shift(1) * (cf.shift(1) / cf)
    pre_c = c["_preclose"].reindex(index=idx, columns=cols)
    pre = pre.where(pre_c.isna(), pre_c)

    # ST 标记
    st = pd.read_csv(ROOT / "data_8:21" / "st_info_latest.csv", dtype=str)
    st.columns = [x.strip().lstrip("﻿") for x in st.columns]
    st["date"] = pd.to_datetime(st.trade_date)
    st["v"] = 1.0
    stp = st.pivot_table(index="date", columns="stock_code", values="v", aggfunc="max")
    stp = stp.reindex(index=idx, columns=cols)
    st_flag = stp.notna()
    st_c = c["_isST"].reindex(index=idx, columns=cols)
    st_flag = st_flag | st_c.eq(1.0).fillna(False)
    print(f"ST 覆盖:平均每日 {st_flag.sum(axis=1).mean():.0f} 只", flush=True)

    # 涨跌停幅度(不含 ST 收窄之外的特殊安排;北交所 30%)
    def rate(code: str) -> float:
        if code.endswith(".BJ"):
            return 0.30
        if code.startswith("688"):
            return 0.20
        if code.startswith(("300", "301")):
            return 0.20      # 2020-08-24 起;之前 10%,下面按日期修正
        return 0.10
    base = pd.Series({cd: rate(cd) for cd in cols})
    lim = pd.DataFrame(np.repeat(base.to_numpy()[None, :], len(idx), axis=0), index=idx, columns=cols)
    gem = [cd for cd in cols if cd.startswith(("300", "301"))]
    lim.loc[lim.index < pd.Timestamp("2020-08-24"), gem] = 0.10
    lim = lim.where(~st_flag, np.minimum(lim, 0.05))          # 主板 ST 5%
    lim.loc[:, [cd for cd in cols if cd.startswith(("300", "301", "688"))]] = \
        lim.loc[:, [cd for cd in cols if cd.startswith(("300", "301", "688"))]].where(
            ~st_flag.loc[:, [cd for cd in cols if cd.startswith(("300", "301", "688"))]], 0.20)

    for k, v in panel.items():
        v.to_parquet(OUT / f"{k}.parquet")
    pre.astype("float32").to_parquet(OUT / "preclose.parquet")
    cf.astype("float64").to_parquet(OUT / "cumf.parquet")
    st_flag.to_parquet(OUT / "is_st.parquet")
    lim.astype("float32").to_parquet(OUT / "limit_rate.parquet")
    turn.reindex(index=idx, columns=cols).astype("float32").to_parquet(OUT / "turnover.parquet")
    circ.reindex(index=idx, columns=cols).astype("float32").to_parquet(OUT / "circ_mv.parquet")
    print("写出", OUT, flush=True)

    # 自检
    r = rc / pre - 1
    up = (r - lim).abs() <= 0.004
    print(f"自检:涨停日占比 {up.to_numpy().sum()/panel['close'].notna().to_numpy().sum():.4f}"
          f"  日均涨停 {up.sum(axis=1).mean():.1f} 只", flush=True)
    bad = (panel["high"] < panel[["open", "close"]].max(axis=1) - 1e-6) | \
          (panel["low"] > panel[["open", "close"]].min(axis=1) + 1e-6)
    print(f"自检:OHLC 越界 {int(bad.to_numpy().sum())} 格", flush=True)


if __name__ == "__main__":
    main()
