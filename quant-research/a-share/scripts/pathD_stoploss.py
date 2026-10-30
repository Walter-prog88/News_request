"""路径 D 关键追问:第 K 节显示"老上升趋势"是杠铃分布(右尾更肥、左尾更肥、均值为负)。
若右尾是真的,加上止损把左尾切掉之后,超额能否转正?

规则:T 日收盘出信号 -> T+1 开盘 twap 买入。
      若此后某日 k(k>=1)收盘价 <= 入场日(T)收盘 × (1-X),次日(T+k+1)开盘 twap 卖出;
      否则持有到 T+1+h 开盘 twap 卖出。
      同一规则同时施加在 age 桶与全池基准上,超额 = 桶日均 - 池日均,再对日期平均。
      不含成本;另报止损触发率,便于估算成本影响(单边约 1bp,卖出另加印花税 5bp)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST

OUT = "/path/to/user/Desktop/A stock market/reports/pathD_stoploss.txt"
_buf = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _buf.append(s)


def run_age(cond, ok):
    T, N = cond.shape
    age = np.zeros((T, N), np.int32); cur = np.zeros(N, np.int32)
    for t in range(T):
        cur = np.where(ok[t], np.where(cond[t], cur + 1, 0), cur); age[t] = cur
    return age


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    m = x.mean(); e = x - m; s = (e * e).sum() / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] * e[:-L]).sum() / n
    return m / np.sqrt(s / n) if s > 0 else np.nan


def shift_up(A, s):
    """B[t] = A[t+s]"""
    T = A.shape[0]
    B = np.full_like(A, np.nan)
    if s < T:
        B[:T - s] = A[s:]
    return B


def stop_ret(C, TW, X, h):
    """返回 (实现收益, 是否止损)。"""
    T, N = C.shape
    entry = shift_up(TW, 1)
    hold = shift_up(TW, 1 + h) / entry - 1.0
    out = np.full((T, N), np.nan, np.float32)
    stopped = np.zeros((T, N), bool)
    alive = np.isfinite(entry) & np.isfinite(C)
    thr = C * (1.0 - X)
    for k in range(1, h + 1):
        ck = shift_up(C, k)
        px = shift_up(TW, k + 1)
        hit = alive & np.isfinite(ck) & (ck <= thr) & np.isfinite(px)
        if not hit.any():
            continue
        out[hit] = px[hit] / entry[hit] - 1.0
        stopped |= hit
        alive &= ~hit
    fin = alive & np.isfinite(hold)
    out[fin] = hold[fin]
    return out, stopped


AGE_BINS = [(1, 4), (10, 19), (20, 39), (40, 59), (60, 119), (120, 10 ** 6)]


def main():
    p = Panel()
    close = p.raw("close_px"); dates = close.index
    T, N = close.shape
    U = ST.universe(p, "V1000").values
    C = close.values.astype(np.float32); okpx = np.isfinite(C)
    M20 = close.rolling(20, min_periods=18).mean().values.astype(np.float32)
    M60 = close.rolling(60, min_periods=50).mean().values.astype(np.float32)
    ok = np.isfinite(C) & np.isfinite(M20) & np.isfinite(M60)
    age = run_age(ok & (C > M60) & (M20 > M60), okpx)
    del M20, M60
    TW = p.raw("twap_open10").values.astype(np.float32)

    y = dates.year.values
    segs = {"TRAIN 2020-23": y <= 2023, "TEST 2024-25": y >= 2024,
            "2024-09-24起": dates.values >= np.datetime64("2024-09-24")}

    for h in (40, 60):
        for X in (0.0, 0.08, 0.12):
            if X == 0.0:
                R = shift_up(TW, 1 + h) / shift_up(TW, 1) - 1.0
                S = np.zeros((T, N), bool)
                tag = f"h={h} 无止损"
            else:
                R, S = stop_ret(C, TW, X, h)
                tag = f"h={h} 止损 -{int(X*100)}%"
            w("\n" + "=" * 104)
            w(f"### {tag}(UP 状态 age 桶;超额 = 桶 - 同规则全池)")
            w("=" * 104)
            okR = np.isfinite(R)
            Rz = np.where(okR, R, 0.0)
            w(f"{'段':<15}{'age桶':<10}{'日均只数':>8}{'桶收益%':>9}{'池收益%':>9}{'超额%':>9}{'NWt':>7}"
              f"{'止损率%':>8}{'P(>20%)':>9}{'池P(>20%)':>10}")
            for lab, sm in segs.items():
                b = U & okR & sm[:, None]
                nb = b.sum(1)
                mb = np.where(nb >= 50, (Rz * b).sum(1) / np.maximum(nb, 1), np.nan)
                pb = (R[b] > 0.20).mean() * 100
                for lo, hi in AGE_BINS:
                    s = (age >= lo) & (age <= hi) & b
                    ns = s.sum(1)
                    if (ns >= 5).sum() < 30:
                        continue
                    ms = np.where(ns >= 5, (Rz * s).sum(1) / np.maximum(ns, 1), np.nan)
                    ex = ms - mb; g = np.isfinite(ex)
                    w(f"{lab:<15}{f'{lo}-{hi if hi<10**5 else 999}':<10}{ns[g].mean():>8.0f}"
                      f"{np.nanmean(ms[g])*100:>9.2f}{np.nanmean(mb[g])*100:>9.2f}"
                      f"{np.nanmean(ex[g])*100:>9.2f}{nw_t(ex[g],h):>7.2f}"
                      f"{S[s].mean()*100:>8.1f}{(R[s]>0.20).mean()*100:>9.1f}{pb:>10.1f}")
                w("")
            del R, S, okR, Rz

    with open(OUT, "w") as f:
        f.write("\n".join(_buf))
    w(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
