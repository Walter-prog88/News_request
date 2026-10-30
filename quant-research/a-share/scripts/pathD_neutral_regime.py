"""路径 D 补充:
C2. age 分桶的中性化超额,但用**日度组合口径**(先在每日截面取等权均值,再对日期取平均),
    与主脚本第 3 节可比;中性化维度 = 成交额十分位 / 申万一级 / 行业×成交额三分位。
F.  市场状态交互:大盘(V1000 等权指数)本身处于 UP / DOWN / 中间态时,
    个股 trend-age 分桶的超额是否变号 —— 直接回答"趋势市里趋势才管用吗"。
G.  可交易化:'age 首次达到 N 当天买入、持有 h 日'的事件超额(N=20/40/60),分训练/检验段。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST

OUT = "/path/to/user/Desktop/A stock market/reports/pathD_neutral_regime.txt"
_buf = []


def w(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _buf.append(s)


def run_age(cond, ok):
    T, N = cond.shape
    age = np.zeros((T, N), np.int32)
    cur = np.zeros(N, np.int32)
    for t in range(T):
        cur = np.where(ok[t], np.where(cond[t], cur + 1, 0), cur)
        age[t] = cur
    return age


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    m = x.mean(); e = x - m
    s = (e * e).sum() / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] * e[:-L]).sum() / n
    return m / np.sqrt(s / n) if s > 0 else np.nan


def cell_demean(R, cells, U, K):
    T, N = R.shape
    out = np.full_like(R, np.nan)
    ok = np.isfinite(R) & U & (cells >= 0)
    for t in range(T):
        m = ok[t]
        if m.sum() < 50:
            continue
        cl = cells[t, m].astype(np.int64); rv = R[t, m]
        cnt = np.bincount(cl, minlength=K)
        mu = np.where(cnt >= 5, np.bincount(cl, weights=rv, minlength=K) / np.maximum(cnt, 1), np.nan)
        out[t, m] = rv - mu[cl]
    return out


AGE_BINS = [(1, 4), (5, 9), (10, 19), (20, 39), (40, 59), (60, 119), (120, 10 ** 6)]


def daily_port(D, sel, minobs=5):
    """D:(T,N) 已中性化收益;sel:(T,N) 桶。返回逐日等权均值序列。"""
    ok = np.isfinite(D) & sel
    n = ok.sum(1)
    s = np.where(ok, D, 0.0).sum(1)
    return np.where(n >= minobs, s / np.maximum(n, 1), np.nan), n


def main():
    p = Panel()
    close = p.raw("close_px"); dates, codes = close.index, close.columns
    T, N = close.shape
    U = ST.universe(p, "V1000").values
    C = close.values.astype(np.float32); okpx = np.isfinite(C)
    M20 = close.rolling(20, min_periods=18).mean().values.astype(np.float32)
    M60 = close.rolling(60, min_periods=50).mean().values.astype(np.float32)
    ok = np.isfinite(C) & np.isfinite(M20) & np.isfinite(M60)
    up = ok & (C > M60) & (M20 > M60)
    dn = ok & (C < M60) & (M20 < M60)
    age_up, age_dn = run_age(up, okpx), run_age(dn, okpx)
    del M20, M60, up, dn

    tw = p.raw("twap_open10")
    R20 = (tw.shift(-21) / tw.shift(-1) - 1.0).values.astype(np.float32)
    del tw

    adv = np.log(p["adv20"].values + 1.0)
    df = pd.read_csv("/path/to/user/Desktop/A stock market/stock_sw_industry.csv", encoding="utf-8-sig")
    m = df.set_index("ts_code")["sw_industry"]
    cats, inv = np.unique(pd.Series(codes).map(m).fillna("NA").values, return_inverse=True)
    inv = inv.astype(np.int16); KI = len(cats)

    IND = np.where(U, np.repeat(inv[None, :], T, 0), -1).astype(np.int16)
    DEC = np.full((T, N), -1, np.int16)
    for t in range(T):
        mm = U[t] & np.isfinite(adv[t])
        if mm.sum() < 100:
            continue
        v = adv[t, mm]
        DEC[t, mm] = np.searchsorted(np.quantile(v, np.arange(1, 10) / 10), v, side="right")
    TER = np.where(DEC >= 0, np.minimum(DEC // 4, 2), -1).astype(np.int16)
    CELL = np.where((IND >= 0) & (TER >= 0), IND * 3 + TER, -1).astype(np.int16)
    ALL = np.where(U, 0, -1).astype(np.int16)
    del adv

    NEU = {"全池": cell_demean(R20, ALL, U, 1),
           "成交额10分位": cell_demean(R20, DEC, U, 10),
           "申万一级": cell_demean(R20, IND, U, KI),
           "行业x额3分位": cell_demean(R20, CELL, U, KI * 3)}
    del ALL, DEC, TER, CELL, IND

    y = dates.year.values
    segs = {"TRAIN 2020-23": y <= 2023, "TEST 2024-25": y >= 2024,
            "2024-09-24起": dates.values >= np.datetime64("2024-09-24")}

    w("=" * 118)
    w("C2. 日度组合口径的中性化超额(h=20 日,单位 %;括号内 NW t)")
    w("=" * 118)
    for lbl, age in (("UP", age_up), ("DOWN", age_dn)):
        w(f"\n  --- {lbl} ---")
        w(f"{'段':<15}{'age桶':<10}{'日均只数':>8}" + "".join(f"{k:>18}" for k in NEU))
        for lab, sm in segs.items():
            for lo, hi in AGE_BINS:
                sel = (age >= lo) & (age <= hi) & U & sm[:, None]
                row = None
                for k, D in NEU.items():
                    x, n = daily_port(D, sel)
                    g = np.isfinite(x)
                    if g.sum() < 30:
                        row = None; break
                    if row is None:
                        row = f"{lab:<15}{f'{lo}-{hi if hi<10**5 else 999}':<10}{n[g].mean():>8.0f}"
                    row += f"{np.nanmean(x[g])*100:>11.2f}({nw_t(x[g],20):>5.2f})"
                if row:
                    w(row)
            w("")

    # ---------------- F. 市场状态交互 ----------------
    w("=" * 118)
    w("F. 大盘状态交互:V1000 等权指数的 UP/DOWN/中间,与个股 trend-age 桶的全池超额(h=20,%)")
    w("=" * 118)
    ret = np.full((T, N), np.nan, np.float32)
    ret[1:] = C[1:] / C[:-1] - 1.0
    ret = np.where(U & np.isfinite(ret) & (np.abs(ret) < 0.5), ret, np.nan)
    with np.errstate(invalid="ignore"):
        mr = np.nanmean(ret, axis=1)
    mr = np.nan_to_num(mr, nan=0.0)
    MI = np.cumprod(1 + mr)
    s = pd.Series(MI)
    mi20, mi60 = s.rolling(20).mean().values, s.rolling(60).mean().values
    mstate = np.where(np.isfinite(mi60) & (MI > mi60) & (mi20 > mi60), 1,
                      np.where(np.isfinite(mi60) & (MI < mi60) & (mi20 < mi60), -1, 0))
    m_age = run_age((mstate == 1)[:, None], np.ones((T, 1), bool))[:, 0]
    w(f"  大盘 UP 天数={int((mstate==1).sum())}  DOWN={int((mstate==-1).sum())}  中间={int((mstate==0).sum())}")
    w(f"  2024-09-24 起大盘 UP 占比={float((mstate[dates.values>=np.datetime64('2024-09-24')]==1).mean())*100:.0f}%")
    D0 = NEU["全池"]
    w(f"\n{'大盘态':<10}{'段':<15}{'age桶':<10}{'日均只数':>8}{'超额%':>9}{'NWt':>7}{'天数':>7}")
    for msname, mm in (("大盘UP", mstate == 1), ("大盘中间", mstate == 0), ("大盘DOWN", mstate == -1)):
        for lab, sm in segs.items():
            if (sm & mm).sum() < 60:
                continue
            for lo, hi in AGE_BINS:
                sel = (age_up >= lo) & (age_up <= hi) & U & (sm & mm)[:, None]
                x, n = daily_port(D0, sel)
                g = np.isfinite(x)
                if g.sum() < 30:
                    continue
                w(f"{msname:<10}{lab:<15}{f'{lo}-{hi if hi<10**5 else 999}':<10}{n[g].mean():>8.0f}"
                  f"{np.nanmean(x[g])*100:>9.2f}{nw_t(x[g],20):>7.2f}{int(g.sum()):>7}")
            w("")

    # ---------------- G. 事件式:age 首次达到 N ----------------
    w("=" * 118)
    w("G. 事件:age 恰好首次达到 N 的当天收盘买入(T+1 开盘),持有 h 日的全池超额(%)")
    w("=" * 118)
    tw = p.raw("twap_open10")
    U0 = np.where(U, 0, -1).astype(np.int16)
    HS = (5, 10, 20, 40, 60)
    DH = {}
    for h in HS:
        RR = (tw.shift(-(1 + h)) / tw.shift(-1) - 1.0).values.astype(np.float32)
        DH[h] = cell_demean(RR, U0, U, 1)
        del RR
    del tw
    w(f"{'段':<15}{'事件':<14}{'事件数':>8}{'日均只数':>8}" + "".join(f"{'h='+str(h):>13}" for h in HS))
    for lab, sm in segs.items():
        for Nn in (20, 40, 60):
            for nm, ev in ((f"UP age={Nn}", age_up == Nn), (f"DOWN age={Nn}", age_dn == Nn)):
                sel = ev & U & sm[:, None]
                if sel.sum() < 500:
                    continue
                row = f"{lab:<15}{nm:<14}{int(sel.sum()):>8}"
                first = True
                for h in HS:
                    x, n = daily_port(DH[h], sel, minobs=3)
                    g = np.isfinite(x)
                    if first:
                        row += f"{n[g].mean():>8.1f}"; first = False
                    row += f"{np.nanmean(x[g])*100:>7.2f}({nw_t(x[g],h):>4.1f})"
                w(row)
        w("")

    with open(OUT, "w") as f:
        f.write("\n".join(_buf))
    w(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
