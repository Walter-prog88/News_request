"""路径 C 收尾之二:把"排雷侧"(负超额)的格子做成日频序列 + 非重叠 t。
结论:唯一在两段都显著、且六年全为负的结构 = 贴 60 日高点 x 高换手率(= 用户所说的"量价齐升")。
"""
import sys, gc, numpy as np, pandas as pd
sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
from afactor.strategy import universe

p = Panel(); dates, codes = p.dates, p.codes; T, N = len(dates), len(codes)
U = universe(p, 'V1000').values.copy()


def lr(pth):
    a = np.load(pth); o = np.full(a.shape, np.nan, np.float32); ok = a != 255
    o[ok] = a[ok].astype(np.float32) / 200.
    return o


RT = lr('/tmp/rk_turn20.npy')          # 由 scripts/pathC_prep_newdata.py 生成
close = p.raw('close_px').values.astype(np.float32); high = p.raw('high_px').values.astype(np.float32)
amount = p.raw('amount').values.astype(np.float32); tw = p.raw('twap_open10').values.astype(np.float32)
p._cache.clear(); gc.collect()


def roll(a, w, fn, mp):
    o = np.empty(a.shape, np.float32)
    for s in range(0, N, 800):
        e = min(s + 800, N)
        o[:, s:e] = getattr(pd.DataFrame(a[:, s:e]).rolling(w, min_periods=mp), fn)().values.astype(np.float32)
    return o


def pr(a):
    o = np.full(a.shape, np.nan, np.float32)
    for s in range(0, T, 300):
        e = min(s + 300, T)
        o[s:e] = pd.DataFrame(np.where(U[s:e], a[s:e], np.nan).astype(np.float32)).rank(axis=1, pct=True).values.astype(np.float32)
    return o


with np.errstate(all='ignore'):
    h = 20; f = np.full((T, N), np.nan, np.float32); den = tw[1:T - h]
    f[:T - 1 - h] = tw[1 + h:] / np.where(den > 0, den, np.nan) - 1.
    f[~np.isfinite(f)] = np.nan; f[~U] = np.nan
    ex = (f - np.nanmean(f, axis=1, keepdims=True)).astype(np.float32)
    hh60 = roll(high, 60, 'max', 40); a5 = roll(amount, 5, 'mean', 4); a60 = roll(amount, 60, 'mean', 40)
    dd60 = pr(close / hh60 - 1.); v1 = pr(a5 / np.where(a60 > 0, a60, np.nan))
    mom = np.full((T, N), np.nan, np.float32); mom[120:] = close[120:] / close[:-120] - 1.; mom = pr(mom)
del tw, high, amount, a5, a60, hh60, close; gc.collect()


def dm(mask):
    m = mask & U & np.isfinite(ex); s = np.where(m, ex, 0.).sum(1); n = m.sum(1)
    return pd.Series(np.where(n >= 5, s / np.maximum(n, 1), np.nan), index=dates), pd.Series(n, index=dates)


def tt(s, step=20):
    v = s.iloc[::step].dropna()
    return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), len(v)


Q = lambda r: np.where(np.isfinite(r), np.clip(r * 5, 0, 4), -1).astype(np.int8)
qd, qv, qm, qt = Q(dd60), Q(v1), Q(mom), Q(RT)
cases = {
    "排雷 A: dd60 Q4(贴60日高) x turn20 Q4(高换手)": (qd == 4) & (qt == 4),
    "排雷 B: mom120 Q4 x turn20 Q4": (qm == 4) & (qt == 4),
    "排雷 C: mom120 Q4 x v1 Q4(近5日放量)": (qm == 4) & (qv == 4),
    "参照 D: mom120 Q4 x v1 Q0(近5日缩量)": (qm == 4) & (qv == 0),
}
for lab, m in cases.items():
    s, n = dm(m); yr = {y: round(float(s[s.index.year == y].mean() * 100), 2) for y in range(2020, 2026)}
    a, ta, na = tt(s[s.index.year <= 2023]); b, tb, nb = tt(s[s.index.year >= 2024])
    print(f"{lab}\n   逐年 {yr}\n   train {a*100:+.2f}% t={ta:+.2f} (n={na}) | test {b*100:+.2f}% t={tb:+.2f} (n={nb}) "
          f"| 每日只数中位 {float(n[n>0].median()):.0f}", flush=True)
