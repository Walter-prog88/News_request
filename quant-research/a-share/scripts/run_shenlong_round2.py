"""神龙摆尾 · 第二轮:严格配对对照 + 分段 + 组合 + 龙虎榜资金流。

第一轮的同日随机对照用"信号收盘"估止损距离,这里改成用**每笔真实成交后的止损距离**
逐笔配对,并另加波动率配对与市值配对,排除"神龙只是选到了低波动股"这一解释。
"""
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
RNG = np.random.default_rng(4242)


def matched_control(d: Data, tr: pd.DataFrame, k: int, mode: str) -> pd.DataFrame:
    """mode: 'rand' 同日随机 | 'vol' 同日 + 60日波动同分位 | 'size' 同日 + 流通市值同分位"""
    elig = (d.eligible() & d.C.notna() & d.O.notna())
    cols = np.array(d.cols)
    ret = d.C.pct_change()
    vol = ret.rolling(60, min_periods=40).std()
    circ = d.circ
    key = {"vol": vol, "size": np.log(circ.where(circ > 0))}.get(mode)
    rows = []
    for dt, g in tr.groupby("signal_date"):
        mask = elig.loc[dt].to_numpy()
        pool = cols[mask]
        if len(pool) < 5 * k:
            continue
        if key is not None:
            kv = key.loc[dt].to_numpy()[mask]
            ok = np.isfinite(kv)
            pool, kv = pool[ok], kv[ok]
            order = np.argsort(kv)
            pool, kv = pool[order], kv[order]
        for r in g.itertuples(index=False):
            frac = 1 - r.stop / r.entry
            if not (0.01 < frac < 0.6):
                continue
            if key is None:
                pick = RNG.choice(pool, size=k, replace=False)
            else:
                own = key.loc[dt, r.code]
                if not np.isfinite(own):
                    continue
                p = int(np.searchsorted(kv, own))
                lo = max(0, min(p - 5 * k // 2, len(pool) - 5 * k))
                band = pool[lo:lo + 5 * k]
                pick = RNG.choice(band, size=min(k, len(band)), replace=False)
            for cd in pick:
                rows.append((cd, dt, frac))
    return pd.DataFrame(rows, columns=["code", "signal_date", "stop_frac"])


def portfolio(d: Data, tr: pd.DataFrame, *, slots=20, weight=0.05) -> dict:
    """按信号先后排队占用固定档位;满仓则放弃后续信号。等权、不加杠杆、闲钱不计息。"""
    tr = tr[tr.entry.notna()].sort_values("signal_date").copy()
    di = {t: k for k, t in enumerate(d.dates)}
    tr["ei"] = tr.signal_date.map(di) + 1
    tr["xi"] = tr.exit_date.map(di)
    tr = tr[tr.xi.notna()]
    tr["xi"] = tr.xi.astype(int)
    busy = np.zeros(len(d.dates) + 5, dtype=int)
    taken = []
    for r in tr.itertuples(index=False):
        if busy[r.ei] >= slots:
            continue
        busy[r.ei:r.xi + 1] += 1
        taken.append(r)
    t = pd.DataFrame(taken)
    eq = 1.0
    daily = np.zeros(len(d.dates))
    for r in t.itertuples(index=False):
        n = max(r.xi - r.ei, 1)
        daily[r.ei:r.xi] += weight * ((1 + r.ret) ** (1 / n) - 1)
    curve = np.cumprod(1 + daily)
    yrs = len(d.dates) / 244
    dd = 1 - curve / np.maximum.accumulate(curve)
    return {"取用笔数": len(t), "总笔数": len(tr), "累计": curve[-1] - 1,
            "年化": curve[-1] ** (1 / yrs) - 1, "最大回撤": dd.max(),
            "平均占用档位": busy[:len(d.dates)].mean()}


def main() -> None:
    d = Data()
    sig = shenlong_signals(d)
    tr = simulate(d, sig, limit_guard=True)
    cl = tr[tr.exit_kind.isin(["stop", "target"])].copy()
    print(f"主版:入场 {tr.entry.notna().sum()} 已平 {len(cl)}", flush=True)
    cl.to_parquet(OUT / "round2_main_closed.parquet")

    rows = [stats(cl, "神龙Ⅰ 045主版")]
    ctrl_store = {}
    for mode, label in (("rand", "同日随机"), ("vol", "同日+60日波动配对"), ("size", "同日+流通市值配对")):
        c = matched_control(d, cl, k=8, mode=mode)
        ct = simulate(d, c, limit_guard=True)
        cc = ct[ct.exit_kind.isin(["stop", "target"])]
        ctrl_store[mode] = ct
        ct.to_parquet(OUT / f"round2_ctrl_{mode}.parquet")
        rows.append(stats(cc, f"对照·{label}(逐笔配对止损)"))
        print(f"  对照 {label}: {len(cc)} 笔", flush=True)
    tab = pd.DataFrame(rows)
    print("\n=== 严格配对对照 ===")
    print(tab.round(4).to_string(index=False))
    tab.to_csv(OUT / "round2_controls.csv", index=False)

    # 分年:神龙 vs 同日随机
    rc = ctrl_store["rand"]
    rc = rc[rc.exit_kind.isin(["stop", "target"])]
    a = cl.groupby(cl.signal_date.dt.year).agg(n=("ret", "size"), 胜率=("ret", lambda x: (x > 0).mean()),
                                               均值=("ret", "mean"))
    b = rc.groupby(rc.signal_date.dt.year).agg(n对照=("ret", "size"), 胜率对照=("ret", lambda x: (x > 0).mean()),
                                               均值对照=("ret", "mean"))
    yr = a.join(b)
    yr["差(神龙-随机)"] = yr.均值 - yr.均值对照
    print("\n=== 分年:神龙 vs 同日随机 ===")
    print(yr.round(4).to_string())
    yr.to_csv(OUT / "round2_by_year.csv")

    # 配对差检验:每个信号 vs 其 8 个同日随机对照的均值
    m = rc.groupby(["signal_date"]).ret.mean().rename("ctrl")
    p = cl.set_index("signal_date").ret.to_frame().join(m).dropna()
    dif = (p.ret - p.ctrl).to_numpy()
    print(f"\n配对差 均值 {dif.mean():+.4f}  NW_t(60) {nw_t(dif, 60):.2f}  n={len(dif)}")

    # 组合
    print("\n=== 组合(20档位,每档5%) ===")
    pf = portfolio(d, cl)
    print(json.dumps({k: round(float(v), 4) for k, v in pf.items()}, ensure_ascii=False))
    base = d.C.pct_change().where(d.eligible()).mean(axis=1)
    bc = (1 + base.fillna(0)).cumprod()
    print(f"全池等权基准 累计 {bc.iloc[-1]-1:+.4f} 年化 {bc.iloc[-1]**(244/len(bc))-1:+.4f}")

    # 切片(事前登记的 6 个,全部报出)
    print("\n=== 切片(全部报出,未择优) ===")
    cl2 = cl.copy()
    cl2["board"] = np.where(cl2.code.str.startswith(("300", "301")), "创业板",
                    np.where(cl2.code.str.startswith("688"), "科创板", "主板"))
    cl2["整理天数"] = pd.cut(cl2.consolidation, [4, 7, 12, 20, 1000], labels=["5-7", "8-12", "13-20", ">20"])
    cl2["放量"] = cl2.vol_up
    cl2["市值分位"] = pd.qcut(cl2.code.map(d.circ.ffill().iloc[-1].to_dict()), 3, labels=["小", "中", "大"])
    for key in ("board", "整理天数", "放量", "市值分位"):
        g = cl2.groupby(key, observed=True).agg(n=("ret", "size"), 胜率=("ret", lambda x: (x > 0).mean()),
                                                均值=("ret", "mean"))
        print(f"\n-- {key}"); print(g.round(4).to_string())

    cl.to_parquet(OUT / "round2_main_closed.parquet")
    ctrl_store["rand"].to_parquet(OUT / "round2_ctrl_rand.parquet")


if __name__ == "__main__":
    main()
