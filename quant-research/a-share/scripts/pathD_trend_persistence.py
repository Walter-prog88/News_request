"""路径 D:趋势的持续性本身(生存分析 + 收益版 + 分年 + 下跌对照)。

状态定义(只用 T 日及之前收盘数据):
  UP[t]   = close[t] > MA60[t] 且 MA20[t] > MA60[t]
  DOWN[t] = close[t] < MA60[t] 且 MA20[t] < MA60[t]
  MA 用 min_periods 略松(MA20>=18, MA60>=50)以耐受停牌。
  停牌日(close 缺失)不判定、状态原样携带、age 不增加。
  age[t] = 该状态已连续为真的天数(含 t)。age>=N 即"连续 N 日符合"。

生存:T = 一段 spell 的总长度。观测到 age==k 等价于 T>=k。
  条件概率 P(再持续 h 天 | 已持续 k 天) = P(T>=k+h | T>=k)
  = 在 age∈[k-2,k+2] 的所有股票日中,状态在 t+1..t+h 全程不破的比例。

收益:fwd_h = twap_open10.shift(-(1+h)) / twap_open10.shift(-1) - 1
  超额 = 桶内当日等权均值 - 当日 V1000 池等权均值,再对日期取平均;NW t 值(lag=h)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST

OUT = "/path/to/user/Desktop/A stock market/reports/pathD_trend_persistence.txt"
_buf = []


def w(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _buf.append(s)


# ---------------- 工具 ----------------
def run_age(cond: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """cond: 条件成立(需 ok 为真); ok: 当日有价。停牌日携带 age 不增。"""
    T, N = cond.shape
    age = np.zeros((T, N), np.int32)
    cur = np.zeros(N, np.int32)
    for t in range(T):
        cur = np.where(ok[t], np.where(cond[t], cur + 1, 0), cur)
        age[t] = cur
    return age


def fut_run(B: np.ndarray, h: int) -> np.ndarray:
    """从 t+1 起连续为真的天数(上限 h)。"""
    T, N = B.shape
    nxt = np.zeros((T, N), np.int16)
    acc = np.zeros(N, np.int16)
    for t in range(T - 1, -1, -1):
        nxt[t] = acc
        acc = np.where(B[t], np.minimum(acc + 1, h), 0).astype(np.int16)
    return nxt


def nw_t(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    m = x.mean()
    e = x - m
    s = (e * e).sum() / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] * e[:-L]).sum() / n
    return m / np.sqrt(s / n) if s > 0 else np.nan


def seg_masks(dates: pd.DatetimeIndex) -> dict:
    y = dates.year.values
    d = {"TRAIN 2020-2023": (y >= 2020) & (y <= 2023),
         "TEST  2024-2025": (y >= 2024) & (y <= 2025)}
    for yy in range(2020, 2026):
        d[f"  {yy}"] = (y == yy)
    d["  2024-09-24起"] = dates.values >= np.datetime64("2024-09-24")
    return d


# ---------------- 主流程 ----------------
def main():
    p = Panel()
    close = p.raw("close_px")
    dates = close.index
    U = ST.universe(p, "V1000").values
    C = close.values.astype(np.float32)
    M20 = close.rolling(20, min_periods=18).mean().values.astype(np.float32)
    M60 = close.rolling(60, min_periods=50).mean().values.astype(np.float32)
    ok = np.isfinite(C) & np.isfinite(M20) & np.isfinite(M60)
    okpx = np.isfinite(C)

    tw = p.raw("twap_open10")
    FR = {h: (tw.shift(-(1 + h)) / tw.shift(-1) - 1.0).values.astype(np.float32) for h in (20, 40)}
    del tw, close, p

    segs = seg_masks(dates)
    conds = {"上升趋势 UP  (close>MA60 且 MA20>MA60)": ok & (C > M60) & (M20 > M60),
             "下跌趋势 DOWN(close<MA60 且 MA20<MA60)": ok & (C < M60) & (M20 < M60)}
    del C, M20, M60

    for lbl, cond in conds.items():
        w("\n" + "=" * 118)
        w(f"### {lbl}")
        w("=" * 118)
        age = run_age(cond, okpx)
        B = age > 0
        analyze(B, age, U, dates, segs, FR)
        del age, B

    with open(OUT, "w") as f:
        f.write("\n".join(_buf))
    w(f"\n[written] {OUT}")


def analyze(B, age, U, dates, segs, FR):
    T = B.shape[0]
    poolcnt = U.sum(1)

    # ---------- 1. 每日符合股票数 ----------
    w("\n--- 1. 每日处于该状态(age>=N)的 V1000 池内股票数 ---")
    w(f"{'段':<20}{'池内总数':>9}{'N=20':>8}{'N=40':>8}{'N=60':>8}{'N=120':>8}"
      f"{'%N20':>7}{'%N60':>7}{'%N120':>7}")
    for lab, sm in segs.items():
        if sm.sum() < 20:
            continue
        pc = poolcnt[sm].mean()
        cnts = {n: ((age >= n) & U).sum(1)[sm].mean() for n in (20, 40, 60, 120)}
        w(f"{lab:<20}{pc:>9.0f}" + "".join(f"{cnts[n]:>8.0f}" for n in (20, 40, 60, 120))
          + f"{cnts[20]/pc*100:>7.1f}{cnts[60]/pc*100:>7.1f}{cnts[120]/pc*100:>7.1f}")

    # ---------- 2a. spell 长度 ----------
    w("\n--- 2a. spell 总长度分布(完整 spell:起点在 2020-06-01 之后、终点在样本内) ---")
    spell_stats(B, U, dates)

    # ---------- 2b. 条件生存概率 ----------
    F20, F40 = fut_run(B, 20), fut_run(B, 40)
    w("\n--- 2b. P(状态再连续存活 h 天 | 已持续 k 天),观测 age∈[k-2,k+2] 且在 V1000 池内 ---")
    hazard(B, age, F20, F40, U, segs, T)
    del F20, F40

    # ---------- 3. 收益版 ----------
    w("\n--- 3. 按 age 分桶:未来 h 日收益、相对 V1000 池等权的超额、胜率 ---")
    ret_by_age(age, U, segs, FR)


def spell_stats(B, U, dates):
    T, N = B.shape
    start_ok = dates.values > np.datetime64("2020-06-01")
    lens_all, lens_pool = [], []
    for j in range(N):
        col = B[:, j]
        if not col.any():
            continue
        d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
        s = np.flatnonzero(d == 1)
        e = np.flatnonzero(d == -1)          # exclusive end
        keep = (s > 0) & (e < T) & start_ok[np.minimum(s, T - 1)]
        if not keep.any():
            continue
        L = (e - s)[keep]
        lens_all.append(L)
        sp = s[keep]
        inp = U[:, j]
        pk = np.array([inp[a:min(a + 20, T)].any() for a in sp], bool)
        lens_pool.append(L[pk])
    for nm, LL in (("全部有效股", lens_all), ("起点20日内在V1000池", lens_pool)):
        L = np.concatenate(LL) if LL else np.zeros(0)
        if len(L) == 0:
            continue
        w(f"  {nm:<22} n_spell={len(L):>7}  mean={L.mean():>6.1f}  P25={np.percentile(L,25):>4.0f}"
          f"  中位={np.percentile(L,50):>4.0f}  P75={np.percentile(L,75):>4.0f}"
          f"  P90={np.percentile(L,90):>4.0f}  P99={np.percentile(L,99):>4.0f}  max={L.max():>4.0f}")
        w("      " + "  ".join(f"P(>={t}d)={(L>=t).mean()*100:.1f}%" for t in (20, 40, 60, 120, 250)))


def hazard(B, age, F20, F40, U, segs, T):
    ks = [1, 5, 10, 20, 30, 40, 60, 80, 120, 160, 200, 250]
    has = {h: np.concatenate([np.ones(T - h, bool), np.zeros(h, bool)]) for h in (20, 40)}
    for h, F in ((20, F20), (40, F40)):
        w(f"\n  P(再连续存活 {h} 天 | age≈k)  %")
        w(f"{'段':<20}" + "".join(f"k={k:<6}" for k in ks) + "  |baseline*")
        for lab, sm in segs.items():
            if sm.sum() < 80:
                continue
            dm = (sm & has[h])[:, None]
            row = [f"{lab:<20}"]
            for k in ks:
                sel = (age >= max(k - 2, 1)) & (age <= k + 2) & U & dm
                row.append(f"{'--':<8}" if sel.sum() < 300 else f"{(F[sel]>=h).mean()*100:<8.1f}")
            base = U & dm                       # 全池任意股票日:未来 h 天全程处于该状态
            row.append(f"  {(F[base]>=h).mean()*100:.1f}")
            w("".join(row))
        w(f"{'  [观测数/千]':<20}" + "".join(
            f"{((age>=max(k-2,1))&(age<=k+2)&U&has[h][:,None]).sum()/1000:<8.0f}" for k in ks))
    w("  *baseline = 全 V1000 池股票日中,未来 h 天全程处于该状态的比例(不看当前 age)")


AGE_BINS = [(1, 4), (5, 9), (10, 19), (20, 39), (40, 59), (60, 119), (120, 249), (250, 10 ** 6)]


def ret_by_age(age, U, segs, FR):
    for h in (20, 40):
        R = FR[h]
        ok = np.isfinite(R)
        Rz = np.where(ok, R, 0.0)
        # 当日池内中位数,用于胜率
        w(f"\n  ==== h={h} 日(twap_open10 口径) ====")
        w(f"{'段':<20}{'age桶':<10}{'日均只数':>8}{'桶收益%':>9}{'池收益%':>9}{'超额%':>9}"
          f"{'NWt':>7}{'胜率%':>8}{'桶日数':>7}")
        for lab, sm in segs.items():
            if sm.sum() < 80:
                continue
            dm = sm[:, None]
            b = U & ok & dm
            nb = b.sum(1)
            mb = np.where(nb >= 50, (Rz * b).sum(1) / np.maximum(nb, 1), np.nan)
            for lo, hi in AGE_BINS:
                s = (age >= lo) & (age <= hi) & U & ok & dm
                ns = s.sum(1)
                ms = np.where(ns >= 5, (Rz * s).sum(1) / np.maximum(ns, 1), np.nan)
                ex = ms - mb
                g = np.isfinite(ex)
                if g.sum() < 30:
                    continue
                win = float((R[s] > 0).mean() * 100)
                nm = f"{lo}-{'inf' if hi > 10**5 else hi}"
                w(f"{lab:<20}{nm:<10}{ns[g].mean():>8.0f}{np.nanmean(ms[g])*100:>9.2f}"
                  f"{np.nanmean(mb[g])*100:>9.2f}{np.nanmean(ex[g])*100:>9.2f}"
                  f"{nw_t(ex[g], h):>7.2f}{win:>8.1f}{int(g.sum()):>7}")
            w("")


if __name__ == "__main__":
    main()
