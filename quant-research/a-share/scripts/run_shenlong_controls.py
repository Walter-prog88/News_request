"""神龙摆尾 · 对照组。回答:形态本身是否比"同一天随便买 + 同样的止盈止损几何"更好。

对照:
  R  同日随机对照   同一交易日、同一可交易池随机取股,止损距离与配对的真实信号相同,止盈同为 ×1.5
  B  首板即买       合格首板的次日开盘就买,止损 = 首板开盘,不等整理与突破
  N  去掉低位背景   只留首板→回箱→突破,不要求"前期下跌 + 近月横盘"
  K  纯突破         背景合格 + 收阳 + 收盘创 20 日新高,无涨停/箱体条件(止损距离配对)
另报每个信号日的 20/60/120 日**逐日截面中性化**超额,重叠窗口用 Newey-West 修 t。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shenlong_core import Data, shenlong_signals, simulate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "shenlong"
RNG = np.random.default_rng(20260829)


def stats(cl: pd.DataFrame, label: str) -> dict:
    if not len(cl):
        return {"版本": label, "已平": 0}
    w, l = cl[cl.ret > 0], cl[cl.ret <= 0]
    return {"版本": label, "已平": len(cl), "胜率": len(w) / len(cl),
            "平均盈": w.ret.mean(), "平均亏": l.ret.mean(),
            "盈亏比": w.ret.mean() / abs(l.ret.mean()) if len(l) else np.nan,
            "每笔净均值": cl.ret.mean(), "中位持有": cl.hold.median(),
            "止损距离中位": (1 - cl.stop / cl.entry).median()}


def nw_t(x: np.ndarray, lag: int) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan
    e = x - x.mean(); n = len(x)
    s = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(max(s, 1e-18) / n)


def fwd_excess(d: Data, sig: pd.DataFrame, hs=(20, 60, 120)) -> pd.DataFrame:
    """信号日收盘起算的前向收益,减同日全池(可交易)等权均值。"""
    C = d.C
    elig = d.eligible()
    rows = {}
    di = {t: k for k, t in enumerate(d.dates)}
    for h in hs:
        f = C.shift(-h) / C - 1
        f = f.where(elig)
        mkt = f.mean(axis=1)
        ex = f.sub(mkt, axis=0)
        rows[h] = ex
    recs = []
    for r in sig.itertuples(index=False):
        rec = {"code": r.code, "signal_date": r.signal_date}
        for h in hs:
            try:
                rec[f"ex{h}"] = rows[h].at[r.signal_date, r.code]
            except KeyError:
                rec[f"ex{h}"] = np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def random_control(d: Data, sig: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    elig = d.eligible() & d.C.notna() & d.O.notna()
    cols = np.array(d.cols)
    by_date = {t: cols[elig.loc[t].to_numpy()] for t in sig.signal_date.unique()}
    rows = []
    for r in sig.itertuples(index=False):
        pool = by_date[r.signal_date]
        if len(pool) < k:
            continue
        frac = 1 - r.board_open / r.signal_close if r.signal_close > 0 else np.nan
        # 用真实成交的止损距离更贴切:先按信号收盘估,模拟里再按实际买价重算比例
        for cd in RNG.choice(pool, size=k, replace=False):
            rows.append((cd, r.signal_date, frac))
    return pd.DataFrame(rows, columns=["code", "signal_date", "stop_frac"])


def first_board_entries(d: Data) -> pd.DataFrame:
    lu, bg, bb, el = d.limit_up(), d.background(), d.big_body(), d.eligible()
    prior = lu.shift(1, fill_value=False).rolling(22, min_periods=22).sum() == 0
    m = lu & bg & bb & el & prior
    idx = np.argwhere(m.to_numpy())
    O = d.O.to_numpy()
    rows = [(d.cols[j], d.dates[i], O[i, j]) for i, j in idx]
    return pd.DataFrame(rows, columns=["code", "signal_date", "board_open"])


def breakout_entries(d: Data) -> pd.DataFrame:
    bg, el = d.background(), d.eligible()
    hi20 = d.C.shift(1).rolling(20, min_periods=20).max()
    m = bg & el & (d.C > d.O) & (d.C > hi20)
    idx = np.argwhere(m.to_numpy())
    rows = [(d.cols[j], d.dates[i]) for i, j in idx]
    b = pd.DataFrame(rows, columns=["code", "signal_date"])
    return b


def main() -> None:
    d = Data()
    print("生成主版信号…", flush=True)
    sig = shenlong_signals(d)
    main_t = simulate(d, sig, limit_guard=True)
    main_cl = main_t[main_t.exit_kind.isin(["stop", "target"])]
    res = [stats(main_cl, "神龙Ⅰ 045主版(收盘回箱)")]

    print("变体…", flush=True)
    for name, kw in (("045实体回箱", dict(return_box="body")),
                     ("034 要求放量", dict(require_volume=True)),
                     ("legacy 不要求回箱", dict(return_box="none")),
                     ("去掉低位背景", dict(need_background=False))):
        s2 = shenlong_signals(d, **kw)
        t2 = simulate(d, s2, limit_guard=True)
        res.append(stats(t2[t2.exit_kind.isin(["stop", "target"])], f"变体·{name}"))
        print(f"  {name}: {len(s2)} 信号", flush=True)

    print("对照 R 同日随机…", flush=True)
    rc = random_control(d, sig, k=10)
    rc = rc[rc.stop_frac.between(0.01, 0.6)]
    rt = simulate(d, rc, limit_guard=True)
    res.append(stats(rt[rt.exit_kind.isin(["stop", "target"])], "对照R·同日随机(配对止损)"))

    print("对照 B 首板即买…", flush=True)
    fb = first_board_entries(d)
    bt = simulate(d, fb, limit_guard=True)
    res.append(stats(bt[bt.exit_kind.isin(["stop", "target"])], "对照B·合格首板次日就买"))

    print("对照 K 纯突破(配对止损)…", flush=True)
    bo = breakout_entries(d)
    bo = bo.sample(n=min(30000, len(bo)), random_state=7).sort_values("signal_date")
    bo["stop_frac"] = float((1 - main_cl.stop / main_cl.entry).median())
    kt = simulate(d, bo, limit_guard=True)
    res.append(stats(kt[kt.exit_kind.isin(["stop", "target"])], "对照K·背景内20日新高突破"))

    tab = pd.DataFrame(res)
    tab.to_csv(OUT / "controls_summary.csv", index=False)
    print("\n" + tab.round(4).to_string(index=False))

    print("\n前向中性化超额:")
    lines = []
    for label, s in (("神龙主版信号", sig), ("合格首板", fb.assign(board_open=fb.board_open))):
        fe = fwd_excess(d, s)
        for h in (20, 60, 120):
            x = fe[f"ex{h}"].to_numpy()
            lines.append({"组": label, "窗口": h, "n": int(np.isfinite(x).sum()),
                          "中性化超额": np.nanmean(x), "NW_t": nw_t(x, h),
                          "中位": np.nanmedian(x), "胜率>0": np.nanmean(x > 0)})
    fx = pd.DataFrame(lines)
    fx.to_csv(OUT / "fwd_excess.csv", index=False)
    print(fx.round(4).to_string(index=False))

    main_t.to_parquet(OUT / "main_trades.parquet")
    sig.to_parquet(OUT / "main_signals.parquet")
    rt.to_parquet(OUT / "control_random_trades.parquet")
    bt.to_parquet(OUT / "control_firstboard_trades.parquet")


if __name__ == "__main__":
    main()
