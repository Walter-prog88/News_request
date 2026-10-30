"""神龙摆尾 · 第四轮:可执行口径的龙虎榜跟随、涨幅配对对照、背景阈值稳健性。"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shenlong_core import Data, shenlong_signals, simulate
from run_shenlong_controls import stats, nw_t

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "shenlong"
RNG = np.random.default_rng(999)


def main():
    d = Data()
    C, O, elig = d.C, d.O, d.eligible()
    di = {t: k for k, t in enumerate(d.dates)}
    ci = {c: k for k, c in enumerate(d.cols)}

    # ---------- 1. 龙虎榜:能实际执行的口径(次日开盘买,第 N 日收盘卖,双边 0.15%)----------
    print("=== 龙虎榜跟随:可执行口径(次日开盘买入,含成本)===")
    tl = pd.read_csv(ROOT / "data_8:21" / "top_list.csv",
                     usecols=["ts_code", "trade_date", "net_amount", "net_amount_ratio", "circ_mv"],
                     dtype={"ts_code": str})
    tl["date"] = pd.to_datetime(tl.trade_date, format="%Y%m%d")
    tl = tl.groupby(["ts_code", "date"], as_index=False).agg(
        net=("net_amount", "sum"), ratio=("net_amount_ratio", "max"), circ=("circ_mv", "max"))
    tl = tl[tl.ts_code.isin(ci) & tl.date.isin(di)].copy()
    cost = 0.0015
    Onp, Cnp = O.to_numpy(), C.to_numpy()
    ii = tl.date.map(di).to_numpy(); jj = tl.ts_code.map(ci).to_numpy()
    valid = elig.to_numpy()
    # 全池同口径基准:次日开盘买、N 日后收盘卖
    for h in (3, 5, 10, 20):
        buy_i = ii + 1
        sell_i = ii + h
        ok = (sell_i < len(d.dates))
        r = np.full(len(tl), np.nan)
        b = Onp[np.clip(buy_i, 0, len(d.dates) - 1), jj]
        s = Cnp[np.clip(sell_i, 0, len(d.dates) - 1), jj]
        r[ok] = (s[ok] * (1 - cost)) / (b[ok] * (1 + cost)) - 1
        tl[f"r{h}"] = r
        # 同日全池基准(同样次日开盘买、N 日收盘卖)
        univ = (pd.DataFrame(Cnp).shift(-h) / pd.DataFrame(Onp).shift(-1) - 1)
        univ = univ.where(pd.DataFrame(valid))
        mkt = univ.mean(axis=1).to_numpy()
        tl[f"ex{h}"] = tl[f"r{h}"] - mkt[ii] + (cost * 2)  * 0  # 基准不含成本,单列对照
        tl[f"mkt{h}"] = mkt[ii]
    tl["净买占成交"] = tl.ratio
    q = pd.qcut(tl.净买占成交, 10, labels=False, duplicates="drop")
    g = tl.assign(q=q).groupby("q").agg(**{f"净r{h}": (f"r{h}", "mean") for h in (3, 5, 10, 20)},
                                        **{f"超额{h}": (f"ex{h}", "mean") for h in (3, 5, 10, 20)}, n=("r5", "size"))
    print((g * 100).round(2).to_string())
    top = tl[q == q.max()]
    for h in (3, 5, 10, 20):
        x = top[f"ex{h}"].to_numpy()
        print(f"  最高一档 h={h:>2}  净收益 {np.nanmean(top[f'r{h}'])*100:+.2f}%  "
              f"减同日全池 {np.nanmean(x)*100:+.2f}%  NW_t {nw_t(x, h):.2f}  n={int(np.isfinite(x).sum())}")

    # ---------- 2. 涨幅配对对照:同日 + 当日涨幅同分位 ----------
    print("\n=== 对照:同日 + 信号日涨幅同分位(问'除了买在大阳线之后,箱体还值什么')===")
    cl = pd.read_parquet(OUT / "round2_main_closed.parquet")
    dayret = (C / C.shift(1) - 1)
    rows = []
    cols = np.array(d.cols)
    for dt, gg in cl.groupby("signal_date"):
        mask = elig.loc[dt].to_numpy() & np.isfinite(dayret.loc[dt].to_numpy())
        pool = cols[mask]; kv = dayret.loc[dt].to_numpy()[mask]
        order = np.argsort(kv); pool, kv = pool[order], kv[order]
        if len(pool) < 60:
            continue
        for r in gg.itertuples(index=False):
            own = dayret.at[dt, r.code]
            frac = 1 - r.stop / r.entry
            if not np.isfinite(own) or not (0.01 < frac < 0.6):
                continue
            p = int(np.searchsorted(kv, own))
            lo = max(0, min(p - 20, len(pool) - 40))
            for cd in RNG.choice(pool[lo:lo + 40], size=8, replace=False):
                rows.append((cd, dt, frac))
    mc = pd.DataFrame(rows, columns=["code", "signal_date", "stop_frac"])
    mt = simulate(d, mc, limit_guard=True)
    mt = mt[mt.exit_kind.isin(["stop", "target"])]
    print(pd.DataFrame([stats(cl, "神龙Ⅰ 045主版"), stats(mt, "对照·同日+当日涨幅配对")]).round(4).to_string(index=False))
    pair = mt.groupby("signal_date").ret.mean().rename("ctrl")
    pp = cl.set_index("signal_date").ret.to_frame().join(pair).dropna()
    dif = (pp.ret - pp.ctrl).to_numpy()
    print(f"配对差 {dif.mean():+.4f}  NW_t(60) {nw_t(dif, 60):.2f}  n={len(dif)}")

    # ---------- 3. 背景阈值稳健性(事前网格,全部报出)----------
    print("\n=== 背景阈值稳健性:横盘振幅上限 × 前期跌幅下限 ===")
    res = []
    base_bg = d.background
    for amp in (0.15, 0.25, 0.35):
        for drop in (0.10, 0.20, 0.30):
            def bg(amp=amp, drop=drop):
                c = d.C
                lo = c.shift(1).rolling(22, min_periods=22).min()
                hi = c.shift(1).rolling(22, min_periods=22).max()
                old = c.shift(23).rolling(104, min_periods=104).max()
                return (((hi / lo - 1) <= amp) & ((1 - lo / old) >= drop)).fillna(False)
            d.background = bg
            s = shenlong_signals(d)
            t = simulate(d, s, limit_guard=True)
            t = t[t.exit_kind.isin(["stop", "target"])]
            st = stats(t, f"振幅≤{amp} 跌幅≥{drop}")
            res.append(st)
            print(f"  {st['版本']}: n={st['已平']} 胜率 {st['胜率']:.3f} 均值 {st['每笔净均值']:+.4f}")
    d.background = base_bg
    pd.DataFrame(res).to_csv(OUT / "round4_bg_grid.csv", index=False)


if __name__ == "__main__":
    main()
