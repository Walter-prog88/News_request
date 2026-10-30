"""神龙摆尾 · 第三轮:组合对照、龙虎榜资金流(真·"有人在买"的直接证据)、当前候选、边界估计。"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shenlong_core import Data, shenlong_signals
from run_shenlong_controls import nw_t

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "shenlong"


def portfolio(dates, tr, *, slots=20, weight=0.05):
    di = {t: k for k, t in enumerate(dates)}
    tr = tr[tr.entry.notna()].sort_values(["signal_date", "code"]).copy()
    tr["ei"] = tr.signal_date.map(di) + 1
    tr["xi"] = tr.exit_date.map(di)
    tr = tr[tr.xi.notna()].copy()
    tr["xi"] = tr.xi.astype(int)
    busy = np.zeros(len(dates) + 5, dtype=int)
    daily = np.zeros(len(dates))
    n_taken = 0
    for r in tr.itertuples(index=False):
        if busy[r.ei] >= slots:
            continue
        busy[r.ei:r.xi + 1] += 1
        n_taken += 1
        n = max(r.xi - r.ei, 1)
        daily[r.ei:r.xi] += weight * ((1 + r.ret) ** (1 / n) - 1)
    curve = np.cumprod(1 + daily)
    dd = 1 - curve / np.maximum.accumulate(curve)
    yrs = len(dates) / 244
    return {"取用": n_taken, "累计": curve[-1] - 1, "年化": curve[-1] ** (1 / yrs) - 1,
            "最大回撤": dd.max(), "平均占用": busy[:len(dates)].mean()}, curve


def main():
    d = Data()
    cl = pd.read_parquet(OUT / "round2_main_closed.parquet")
    rc = pd.read_parquet(OUT / "round2_ctrl_rand.parquet")
    rc = rc[rc.exit_kind.isin(["stop", "target"])]

    print("=== 组合口径对照(同样 20 档位、每档 5%、先到先得)===")
    a, ca = portfolio(d.dates, cl)
    b, cb = portfolio(d.dates, rc.sample(frac=1.0, random_state=1))
    print("神龙 :", json.dumps({k: round(float(v), 4) for k, v in a.items()}, ensure_ascii=False))
    print("随机 :", json.dumps({k: round(float(v), 4) for k, v in b.items()}, ensure_ascii=False))
    m = d.dates >= pd.Timestamp("2021-01-01")
    base = d.C.pct_change(fill_method=None).where(d.eligible()).mean(axis=1)
    bc = (1 + base[m].fillna(0)).cumprod()
    print(f"全池等权(2021起) 累计 {bc.iloc[-1]-1:+.4f} 年化 {bc.iloc[-1]**(244/m.sum())-1:+.4f}")

    # ---------- 龙虎榜:直接观察"有人在大额净买入"之后会怎样 ----------
    print("\n=== 龙虎榜净买入 → 前向中性化超额 ===")
    tl = pd.read_csv(ROOT / "data_8:21" / "top_list.csv",
                     usecols=["ts_code", "trade_date", "net_amount", "net_amount_ratio",
                              "turnover_rate", "circ_mv", "reason"], dtype={"ts_code": str})
    tl["date"] = pd.to_datetime(tl.trade_date, format="%Y%m%d")
    tl = tl.groupby(["ts_code", "date"], as_index=False).agg(
        net_amount=("net_amount", "sum"), ratio=("net_amount_ratio", "max"),
        circ_mv=("circ_mv", "max"))
    tl = tl[tl.ts_code.isin(d.cols)]
    C, elig = d.C, d.eligible()
    ex = {}
    for h in (1, 5, 10, 20, 60):
        f = (C.shift(-h) / C - 1).where(elig)
        ex[h] = f.sub(f.mean(axis=1), axis=0)
    di = {t: k for k, t in enumerate(d.dates)}
    tl = tl[tl.date.isin(di)]
    ii = tl.date.map(di).to_numpy()
    ci = tl.ts_code.map({c: k for k, c in enumerate(d.cols)}).to_numpy()
    for h in (1, 5, 10, 20, 60):
        tl[f"ex{h}"] = ex[h].to_numpy()[ii, ci]
    tl["净买占成交"] = tl.ratio
    tl["净买占流通市值"] = tl.net_amount / tl.circ_mv.replace(0, np.nan)
    for key in ("净买占成交", "净买占流通市值"):
        q = pd.qcut(tl[key], 10, labels=False, duplicates="drop")
        g = tl.assign(q=q).groupby("q").agg(n=("ex20", "size"), **{f"ex{h}": (f"ex{h}", "mean") for h in (1, 5, 10, 20, 60)})
        print(f"\n-- 按 {key} 十分位(上榜当日收盘起算,中性化)")
        print((g * 100).round(2).to_string())
        top = tl[q == q.max()]
        for h in (5, 20, 60):
            x = top[f"ex{h}"].to_numpy()
            print(f"   最高一档 ex{h} 均值 {np.nanmean(x)*100:+.2f}%  NW_t {nw_t(x, h):.2f}  n={int(np.isfinite(x).sum())}")

    # 首板 + 当日龙虎榜大额净买
    print("\n=== 合格首板 × 当日是否龙虎榜大额净买 ===")
    lu, bg, bb, el = d.limit_up(), d.background(), d.big_body(), d.eligible()
    prior = lu.shift(1, fill_value=False).rolling(22, min_periods=22).sum() == 0
    m2 = (lu & bg & bb & el & prior).to_numpy()
    idx = np.argwhere(m2)
    fb = pd.DataFrame({"code": [d.cols[j] for _, j in idx], "date": [d.dates[i] for i, _ in idx]})
    fb = fb.merge(tl[["ts_code", "date", "净买占成交", "ex5", "ex20", "ex60"]],
                  left_on=["code", "date"], right_on=["ts_code", "date"], how="left")
    fb["上榜"] = fb.净买占成交.notna()
    fb["大额净买"] = fb.净买占成交 > 5
    ii = fb.date.map(di).to_numpy(); ci = fb.code.map({c: k for k, c in enumerate(d.cols)}).to_numpy()
    for h in (5, 20, 60):
        fb[f"ex{h}"] = ex[h].to_numpy()[ii, ci]
    g = fb.groupby(["上榜", "大额净买"]).agg(n=("ex20", "size"), ex5=("ex5", "mean"),
                                          ex20=("ex20", "mean"), ex60=("ex60", "mean"))
    print((g.assign(n=g.n) ).round(4).to_string())

    # ---------- 当前候选(截至面板最后一日)----------
    print("\n=== 截至面板最后一日的形态状态 ===")
    sig = pd.read_parquet(OUT / "main_signals.parquet")
    recent = sig[sig.signal_date >= d.dates[-25]]
    print(f"最近 25 个交易日确认突破的信号 {len(recent)} 个")
    print(recent.sort_values("signal_date").tail(20).to_string(index=False))

    # ---------- 退市幸存者偏差上界 ----------
    sb = pd.read_csv(ROOT / "data_8:21" / "stock_basic.csv", dtype=str)
    dl = sb[sb.delist_date.notna()]
    dl = dl[(dl.delist_date >= "20210101") & (dl.delist_date <= "20260827")]
    rate = len(sig) / (len(d.cols) * (len(d.dates) / 244))
    miss = len(dl) * 1.5 * rate
    print(f"\n退市股 {len(dl)} 只不在面板内;按每股每年 {rate:.3f} 个信号、平均在样 1.5 年估计,"
          f"缺失约 {miss:.0f} 笔;若这些交易平均 -60%,主版每笔净均值由 "
          f"{cl.ret.mean():+.4f} 降至 {(cl.ret.sum() + miss*-0.6)/(len(cl)+miss):+.4f}")


if __name__ == "__main__":
    main()
