"""路径 D 收尾:右尾与生存的关键汇总数字。
K. 截面分位口径:把每日池内 fwd20 排成 0-1 百分位,按 age 桶报
   平均百分位、P(落入当日 Top10%)、P(落入当日 Top2%)、P(落入当日 Bottom10%)。
   —— 直接回答"趋势状态是否提高抓到大牛股的概率"(均值为负但右尾可能更肥)。
L. 生存汇总:P(T>=2k | T>=k) 与 age=k 时的期望剩余寿命(实际 vs 随机零模型)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST

OUT = "/path/to/user/Desktop/A stock market/reports/pathD_tail.txt"
_buf = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _buf.append(s)


def run_age(cond, ok):
    T, N = cond.shape
    age = np.zeros((T, N), np.int32); cur = np.zeros(N, np.int32)
    for t in range(T):
        cur = np.where(ok[t], np.where(cond[t], cur + 1, 0), cur); age[t] = cur
    return age


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


AGE_BINS = [(1, 4), (5, 9), (10, 19), (20, 39), (40, 59), (60, 119), (120, 10 ** 6)]


def main():
    p = Panel()
    close = p.raw("close_px"); dates = close.index
    T, N = close.shape
    U = ST.universe(p, "V1000").values
    C = close.values.astype(np.float32); okpx = np.isfinite(C)
    M20 = close.rolling(20, min_periods=18).mean().values.astype(np.float32)
    M60 = close.rolling(60, min_periods=50).mean().values.astype(np.float32)
    ok = np.isfinite(C) & np.isfinite(M20) & np.isfinite(M60)
    age_up = run_age(ok & (C > M60) & (M20 > M60), okpx)
    age_dn = run_age(ok & (C < M60) & (M20 < M60), okpx)
    tw = p.raw("twap_open10")
    R = (tw.shift(-21) / tw.shift(-1) - 1.0).values.astype(np.float32)
    del tw, M20, M60

    # 截面百分位
    P = np.full((T, N), np.nan, np.float32)
    for t in range(T):
        m = U[t] & np.isfinite(R[t])
        n = int(m.sum())
        if n < 100:
            continue
        v = R[t, m]
        r = np.argsort(np.argsort(v)).astype(np.float32)
        P[t, m] = r / (n - 1)

    y = dates.year.values
    segs = {"TRAIN 2020-23": y <= 2023, "TEST 2024-25": y >= 2024,
            "2024-09-24起": dates.values >= np.datetime64("2024-09-24")}
    w("=" * 104)
    w("K. fwd20 的当日截面百分位(池内 0=最差 1=最好;随机应为 0.500 / Top10%=10% / Top2%=2%)")
    w("=" * 104)
    for lbl, age in (("UP", age_up), ("DOWN", age_dn)):
        w(f"\n  --- {lbl} ---")
        w(f"{'段':<15}{'age桶':<10}{'股票日数':>10}{'平均百分位':>10}{'Top10%':>9}{'Top2%':>9}{'Bot10%':>9}")
        for lab, sm in segs.items():
            for lo, hi in AGE_BINS:
                sel = (age >= lo) & (age <= hi) & U & sm[:, None] & np.isfinite(P)
                n = int(sel.sum())
                if n < 3000:
                    continue
                v = P[sel]
                w(f"{lab:<15}{f'{lo}-{hi if hi<10**5 else 999}':<10}{n:>10}{v.mean():>10.3f}"
                  f"{(v>=0.90).mean()*100:>9.1f}{(v>=0.98).mean()*100:>9.1f}{(v<=0.10).mean()*100:>9.1f}")
            w("")

    w("=" * 104)
    w("L. 生存汇总:spell 总长度 T 的条件概率(全样本 2020-06 起的完整 spell)")
    w("=" * 104)
    lr = np.log(C[1:] / C[:-1]); lr = np.where(np.isfinite(lr), lr, np.nan)
    rows = {}
    rows["实际 UP"] = spell_len(age_up > 0)
    rows["实际 DOWN"] = spell_len(age_dn > 0)
    rng = np.random.default_rng(11)
    X = lr.copy()
    for j in range(N):
        v = np.flatnonzero(np.isfinite(X[:, j]))
        if len(v) > 2:
            X[v, j] = X[rng.permutation(v), j]
    Cs = np.vstack([np.ones(N, np.float32),
                    np.exp(np.nancumsum(np.where(np.isfinite(X), X, 0.0), 0))]).astype(np.float32)
    Cs[~okpx] = np.nan
    d = pd.DataFrame(Cs)
    m20 = d.rolling(20, min_periods=18).mean().values.astype(np.float32)
    m60 = d.rolling(60, min_periods=50).mean().values.astype(np.float32)
    o = np.isfinite(Cs) & np.isfinite(m20) & np.isfinite(m60)
    rows["NULL 随机游走 UP"] = spell_len(run_age(o & (Cs > m60) & (m20 > m60), okpx) > 0)
    w(f"{'序列':<20}{'n':>8}{'均值':>7}{'中位':>6}{'P90':>6}" +
      "".join(f"{'P(T>=' + str(k) + ')':>11}" for k in (20, 40, 60, 120)) +
      "".join(f"{'P(2k|k)k=' + str(k):>13}" for k in (20, 40, 60)) + f"{'E[剩余|T>=60]':>14}")
    for nm, L in rows.items():
        L = np.asarray(L, float)
        cond = [f"{(L>=2*k).sum()/max((L>=k).sum(),1)*100:>13.1f}" for k in (20, 40, 60)]
        rem = (L[L >= 60] - 60).mean() if (L >= 60).any() else np.nan
        w(f"{nm:<20}{len(L):>8}{L.mean():>7.1f}{np.percentile(L,50):>6.0f}{np.percentile(L,90):>6.0f}"
          + "".join(f"{(L>=k).mean()*100:>11.2f}" for k in (20, 40, 60, 120))
          + "".join(cond) + f"{rem:>14.1f}")
    with open(OUT, "w") as f:
        f.write("\n".join(_buf))
    w(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
