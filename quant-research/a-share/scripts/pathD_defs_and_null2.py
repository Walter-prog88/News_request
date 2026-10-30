"""路径 D 稳健性:
H. DOWN 侧的随机零模型(与 UP 对称检验)。
I. 更严格的"趋势"定义:
   DEF1 = close>MA60 且 MA20>MA60                       (主定义)
   DEF2 = 多头排列 close>MA20>MA60>MA120                 (教科书上升通道)
   DEF3 = DEF2 且 amt20 > amt120 (量能同步放大)          (量价齐升)
   DEF4 = DEF2 且 距 250 日最高价 <= 5%(创新高型趋势)
   每个定义:每日只数 / hazard 曲线 / 与随机零模型比 / age 桶超额(训练 vs 检验)。
J. 大票 vs 小票:UP age>=20 的超额按成交额十分位分组。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST

OUT = "/path/to/user/Desktop/A stock market/reports/pathD_defs_null2.txt"
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


def fut_run(B, h):
    T, N = B.shape
    nxt = np.zeros((T, N), np.int16); acc = np.zeros(N, np.int16)
    for t in range(T - 1, -1, -1):
        nxt[t] = acc
        acc = np.where(B[t], np.minimum(acc + 1, h), 0).astype(np.int16)
    return nxt


def spell_len(B):
    T, N = B.shape; out = []
    for j in range(N):
        col = B[:, j]
        if not col.any():
            continue
        d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        k = (s > 60) & (e < T)
        if k.any():
            out.append((e - s)[k])
    return np.concatenate(out) if out else np.zeros(0, int)


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    m = x.mean(); e = x - m; s = (e * e).sum() / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] * e[:-L]).sum() / n
    return m / np.sqrt(s / n) if s > 0 else np.nan


KS = [1, 5, 10, 20, 30, 40, 60, 80, 120]


def hz(age, F, U, h, m2=None, minn=250):
    r = []
    for k in KS:
        sel = (age >= max(k - 2, 1)) & (age <= k + 2) & U
        if m2 is not None:
            sel = sel & m2
        r.append(np.nan if sel.sum() < minn else (F[sel] >= h).mean() * 100)
    return "".join("  --  " if not np.isfinite(x) else f"{x:6.1f}" for x in r)


AGE_BINS = [(1, 4), (5, 9), (10, 19), (20, 39), (40, 59), (60, 119), (120, 10 ** 6)]


def daily_ex(R, sel, pool):
    ok = np.isfinite(R)
    s, b = sel & pool & ok, pool & ok
    ns, nb = s.sum(1), b.sum(1)
    Rz = np.where(ok, R, 0.0)
    ms = np.where(ns >= 5, (Rz * s).sum(1) / np.maximum(ns, 1), np.nan)
    mb = np.where(nb >= 50, (Rz * b).sum(1) / np.maximum(nb, 1), np.nan)
    return ms - mb, ns


def main():
    p = Panel()
    close = p.raw("close_px"); dates = close.index
    T, N = close.shape
    U = ST.universe(p, "V1000").values
    C = close.values.astype(np.float32); okpx = np.isfinite(C)
    MA = {k: close.rolling(k, min_periods=int(k * .85)).mean().values.astype(np.float32)
          for k in (20, 60, 120)}
    hi250 = close.rolling(250, min_periods=150).max().values.astype(np.float32)
    amt = p.raw("amount")
    a20 = amt.rolling(20, min_periods=15).mean().values.astype(np.float32)
    a120 = amt.rolling(120, min_periods=90).mean().values.astype(np.float32)
    del amt
    tw = p.raw("twap_open10")
    R20 = (tw.shift(-21) / tw.shift(-1) - 1.0).values.astype(np.float32)
    R40 = (tw.shift(-41) / tw.shift(-1) - 1.0).values.astype(np.float32)
    del tw
    ok = np.isfinite(C) & np.isfinite(MA[20]) & np.isfinite(MA[60])
    ok3 = ok & np.isfinite(MA[120])
    y = dates.year.values
    segs = {"TRAIN 2020-23": y <= 2023, "TEST 2024-25": y >= 2024,
            "2024-09-24起": dates.values >= np.datetime64("2024-09-24")}

    # ---------- H. DOWN 零模型 ----------
    w("=" * 112)
    w("H. DOWN 侧随机零模型:P(状态再连续存活 20 天 | age≈k) %")
    w("=" * 112)
    dn = ok & (C < MA[60]) & (MA[20] < MA[60])
    a_dn = run_age(dn, okpx); f_dn = fut_run(a_dn > 0, 20)
    tail = np.zeros(T, bool); tail[:T - 20] = True
    L = spell_len(a_dn > 0)
    w(f"{'序列':<26}" + "".join(f"k={k:<4}" for k in KS) + "   中位  P90  P(>=60d)%")
    w(f"{'实际 DOWN':<26}{hz(a_dn, f_dn, U, 20, tail[:,None])}"
      f"{np.percentile(L,50):>7.0f}{np.percentile(L,90):>5.0f}{(L>=60).mean()*100:>9.1f}")
    lr = np.log(C[1:] / C[:-1]); lr = np.where(np.isfinite(lr), lr, np.nan)
    with np.errstate(invalid="ignore"):
        mkt = np.nan_to_num(np.nanmean(np.where(U[1:], lr, np.nan), 1), nan=0.0)
    for nm, base, add in (("NULL-1 全收益重排", lr, None), ("NULL-2 只重排残差", lr - mkt[:, None], mkt)):
        rng = np.random.default_rng(7)
        X = base.copy()
        for j in range(N):
            v = np.flatnonzero(np.isfinite(X[:, j]))
            if len(v) > 2:
                X[v, j] = X[rng.permutation(v), j]
        if add is not None:
            X = X + add[:, None]
        Cs = np.vstack([np.ones(N, np.float32),
                        np.exp(np.nancumsum(np.where(np.isfinite(X), X, 0.0), 0))]).astype(np.float32)
        Cs[~okpx] = np.nan
        d = pd.DataFrame(Cs)
        m20 = d.rolling(20, min_periods=18).mean().values.astype(np.float32)
        m60 = d.rolling(60, min_periods=50).mean().values.astype(np.float32)
        o = np.isfinite(Cs) & np.isfinite(m20) & np.isfinite(m60)
        a2 = run_age(o & (Cs < m60) & (m20 < m60), okpx); f2 = fut_run(a2 > 0, 20)
        L2 = spell_len(a2 > 0)
        w(f"{nm+' DOWN':<26}{hz(a2, f2, U, 20, tail[:,None])}"
          f"{np.percentile(L2,50):>7.0f}{np.percentile(L2,90):>5.0f}{(L2>=60).mean()*100:>9.1f}")
        del Cs, d, m20, m60, o, a2, f2
    del lr, a_dn, f_dn, dn

    # ---------- I. 多种趋势定义 ----------
    w("\n" + "=" * 112)
    w("I. 更严格的趋势定义(每个定义:每日只数、hazard、age 桶 h=20/40 全池超额)")
    w("=" * 112)
    hi_near = ok3 & (C >= hi250 * 0.95)
    DEFS = {
        "DEF1 close>MA60,MA20>MA60": ok & (C > MA[60]) & (MA[20] > MA[60]),
        "DEF2 多头排列 C>MA20>MA60>MA120": ok3 & (C > MA[20]) & (MA[20] > MA[60]) & (MA[60] > MA[120]),
        "DEF3 DEF2 且 amt20>amt120": (ok3 & (C > MA[20]) & (MA[20] > MA[60]) & (MA[60] > MA[120])
                                      & np.isfinite(a20) & np.isfinite(a120) & (a20 > a120)),
        "DEF4 DEF2 且 距250日高<=5%": (ok3 & (C > MA[20]) & (MA[20] > MA[60]) & (MA[60] > MA[120])
                                    & np.isfinite(hi250) & hi_near),
    }
    for nm, cond in DEFS.items():
        age = run_age(cond, okpx); F = fut_run(age > 0, 20)
        L = spell_len(age > 0)
        w(f"\n  ### {nm}")
        w(f"    池内日均只数 age>=1:{((age>=1)&U).sum(1).mean():.0f}  age>=20:{((age>=20)&U).sum(1).mean():.0f}"
          f"  age>=60:{((age>=60)&U).sum(1).mean():.0f}")
        w(f"    spell: n={len(L)} 中位={np.percentile(L,50):.0f} P90={np.percentile(L,90):.0f} "
          f"P(>=60d)={(L>=60).mean()*100:.1f}%")
        w(f"    {'P(再撑20天|age≈k)':<22}" + "".join(f"k={k:<4}" for k in KS))
        for lab, sm in segs.items():
            w(f"    {lab:<22}{hz(age, F, U, 20, (sm & tail)[:,None])}")
        w(f"    {'段':<15}{'age桶':<10}{'日均只数':>8}{'h20超额%':>10}{'NWt':>7}{'h40超额%':>10}{'NWt':>7}")
        for lab, sm in segs.items():
            for lo, hi in AGE_BINS:
                sel = (age >= lo) & (age <= hi) & sm[:, None]
                e1, n1 = daily_ex(R20, sel, U & sm[:, None])
                e2, _ = daily_ex(R40, sel, U & sm[:, None])
                g1, g2 = np.isfinite(e1), np.isfinite(e2)
                if g1.sum() < 30:
                    continue
                w(f"    {lab:<15}{f'{lo}-{hi if hi<10**5 else 999}':<10}{n1[g1].mean():>8.0f}"
                  f"{np.nanmean(e1[g1])*100:>10.2f}{nw_t(e1[g1],20):>7.2f}"
                  f"{np.nanmean(e2[g2])*100:>10.2f}{nw_t(e2[g2],40):>7.2f}")
            w("")
        del age, F

    # ---------- J. 大票 vs 小票 ----------
    w("=" * 112)
    w("J. UP(DEF1) age>=20 的 h=20 超额,按当日成交额十分位分组(基准=同十分位池均值)")
    w("=" * 112)
    age = run_age(DEFS["DEF1 close>MA60,MA20>MA60"], okpx)
    la = np.log(p["adv20"].values + 1.0)
    DEC = np.full((T, N), -1, np.int16)
    for t in range(T):
        m = U[t] & np.isfinite(la[t])
        if m.sum() < 100:
            continue
        v = la[t, m]
        DEC[t, m] = np.searchsorted(np.quantile(v, np.arange(1, 10) / 10), v, side="right")
    w(f"{'段':<15}{'成交额十分位':<12}{'日均只数':>8}{'桶收益%':>9}{'同分位基准%':>12}{'超额%':>9}{'NWt':>7}")
    for lab, sm in segs.items():
        for d0 in range(10):
            pool_d = U & (DEC == d0) & sm[:, None]
            sel = (age >= 20) & pool_d
            e, n = daily_ex(R20, sel, pool_d)
            ok_ = np.isfinite(R20)
            nb = pool_d.sum(1)
            mb = np.where(nb >= 30, (np.where(ok_, R20, 0) * (pool_d & ok_)).sum(1) /
                          np.maximum((pool_d & ok_).sum(1), 1), np.nan)
            g = np.isfinite(e)
            if g.sum() < 30:
                continue
            ms = e + mb
            w(f"{lab:<15}{'D'+str(d0+1):<12}{n[g].mean():>8.0f}{np.nanmean(ms[g])*100:>9.2f}"
              f"{np.nanmean(mb[g])*100:>12.2f}{np.nanmean(e[g])*100:>9.2f}{nw_t(e[g],20):>7.2f}")
        w("")

    with open(OUT, "w") as f:
        f.write("\n".join(_buf))
    w(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
