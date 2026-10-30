"""路径 C 收尾:把候选结构做成可交易口径 + 给诚实的统计量。

A) 六条轴的 Q0-Q4 日频多空价差序列:逐年 + 非重叠 20 日 t 值(截面相关已被日聚合吸收)。
B) 唯一两段都为正的格子 box20[Q4] x v2[Q0]:做成每 20 日调仓的等权组合,报年化超额与逐年。
C) 同一格子的"事前可识别"版本:固定阈值(不用截面分位)是否也成立。
D) 与用户结构的直接对照组合。
"""
from __future__ import annotations
import gc, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
from afactor.strategy import universe

OUT = "/path/to/user/Desktop/A stock market/research/pathC_volprice_grid"
NQ, CH = 5, 800
T0 = time.time()
pd.set_option("display.width", 240)


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def rss():
    import psutil, os as _o
    return psutil.Process(_o.getpid()).memory_info().rss / 1e9


p = Panel()
dates, codes = p.dates, p.codes
T, N = len(dates), len(codes)
YR = dates.year.values
U = universe(p, "V1000").values.copy()


def _load_rk(path):
    a = np.load(path)
    out = np.full(a.shape, np.nan, np.float32)
    ok = a != 255
    out[ok] = a[ok].astype(np.float32) / 200.0
    return out


RK_TURN = _load_rk("/tmp/rk_turn20.npy")

close = p.raw("close_px").values.astype(np.float32)
high = p.raw("high_px").values.astype(np.float32)
low = p.raw("low_px").values.astype(np.float32)
amount = p.raw("amount").values.astype(np.float32)
tw = p.raw("twap_open10").values.astype(np.float32)
p._cache.clear(); gc.collect()


def roll(a, w, fn, minp=None):
    minp = minp or max(w // 2, 5)
    out = np.empty(a.shape, np.float32)
    for s in range(0, a.shape[1], CH):
        e = min(s + CH, a.shape[1])
        blk = pd.DataFrame(a[:, s:e])
        out[:, s:e] = getattr(blk.rolling(w, min_periods=minp), fn)().values.astype(np.float32)
        del blk
    gc.collect()
    return out


def pctrank(a):
    out = np.full(a.shape, np.nan, np.float32)
    for s in range(0, T, 300):
        e = min(s + 300, T)
        x = np.where(U[s:e], a[s:e], np.nan).astype(np.float32)
        out[s:e] = pd.DataFrame(x).rank(axis=1, pct=True).values.astype(np.float32)
        del x
    return out


with np.errstate(all="ignore"):
    def fwd(h):
        f = np.full((T, N), np.nan, np.float32)
        den = tw[1:T - h]
        f[: T - 1 - h] = tw[1 + h:] / np.where(den > 0, den, np.nan) - 1.0
        f[~np.isfinite(f)] = np.nan
        f[~U] = np.nan
        return (f - np.nanmean(f, axis=1, keepdims=True)).astype(np.float32)
    ex20 = fwd(20)
del tw; gc.collect()

with np.errstate(all="ignore"):
    ret = np.full((T, N), np.nan, np.float32); ret[1:] = close[1:] / close[:-1] - 1.0
    ret[~np.isfinite(ret)] = np.nan
    a5 = roll(amount, 5, "mean", 4); a20 = roll(amount, 20, "mean", 15)
    a60 = roll(amount, 60, "mean", 40); a120 = roll(amount, 120, "mean", 80)
    hh20 = roll(high, 20, "max", 15); ll20 = roll(low, 20, "min", 15)
    cc252 = roll(close, 252, "max", 150)
    RAW = {}
    RAW["box20"] = (hh20 - ll20) / close
    RAW["v1"] = a5 / np.where(a60 > 0, a60, np.nan)
    RAW["v2"] = a20 / np.where(a120 > 0, a120, np.nan)
    RAW["pos252"] = close / cc252
    mom = np.full((T, N), np.nan, np.float32); mom[120:] = close[120:] / close[:-120] - 1.0
    RAW["mom120"] = mom
    mom20 = np.full((T, N), np.nan, np.float32); mom20[20:] = close[20:] / close[:-20] - 1.0
    RAW["mom20"] = mom20
    R = {k: pctrank(v) for k, v in RAW.items()}
    R["turn20"] = RK_TURN
del a5, a20, a60, a120, hh20, ll20, cc252, ret, high, low, amount, close
gc.collect()
log(f"feats rss={rss():.2f}GB")

Q = {k: np.where(np.isfinite(v), np.clip((v * NQ), 0, NQ - 1), -1).astype(np.int8) for k, v in R.items()}


def daily_mean(mask):
    """每日截面均值(mask 内),返回长度 T 的 Series(不足 5 只记 NaN)。"""
    m = mask & U & np.isfinite(ex20)
    s = np.where(m, ex20, 0.0).sum(1)
    n = m.sum(1)
    out = np.where(n >= 5, s / np.maximum(n, 1), np.nan)
    return pd.Series(out, index=dates), pd.Series(n, index=dates)


def tstat_nonoverlap(s, step=20):
    v = s.iloc[::step].dropna()
    if len(v) < 8:
        return np.nan, np.nan, 0
    return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), len(v)


log("\n" + "=" * 78 + "\nA) 单轴 Q0 - Q4 日频价差:逐年均值(%) 与非重叠 t")
rows = []
for k in ["v1", "v2", "turn20", "pos252", "mom120", "box20"]:
    lo, _ = daily_mean(Q[k] == 0)
    hi, _ = daily_mean(Q[k] == 4)
    sp = lo - hi
    r = {"axis": f"{k} Q0-Q4"}
    for y in range(2020, 2026):
        r[y] = float(sp[sp.index.year == y].mean() * 100)
    for seg, yy in (("train", sp.index.year <= 2023), ("test", sp.index.year >= 2024)):
        m, t, n = tstat_nonoverlap(sp[yy])
        r[seg + "_mean"] = float(m * 100); r[seg + "_t"] = float(t); r[seg + "_n"] = n
    rows.append(r)
axis_tbl = pd.DataFrame(rows).set_index("axis").round(3)
axis_tbl.to_csv(f"{OUT}/axis_spread_by_year.csv")
log("\n" + axis_tbl.to_string())

log("\n" + "=" * 78 + "\nB) 候选格子 box20[Q4] x v2[Q0]:每日成员的等权超额")
CELL = (Q["box20"] == 4) & (Q["v2"] == 0)
s, n = daily_mean(CELL)
r = {}
for y in range(2020, 2026):
    r[y] = float(s[s.index.year == y].mean() * 100)
log("  逐年 ex20(%): " + str({k: round(v, 2) for k, v in r.items()}))
log("  每日成员数中位: " + str(float(n[n > 0].median())))
for seg, yy in (("train", s.index.year <= 2023), ("test", s.index.year >= 2024)):
    m, t, nn = tstat_nonoverlap(s[yy])
    log(f"  {seg}: 非重叠 n={nn}  mean={m*100:+.3f}%/20日  t={t:+.2f}")
m, t, nn = tstat_nonoverlap(s)
log(f"  全样本: 非重叠 n={nn}  mean={m*100:+.3f}%/20日  t={t:+.2f}  "
    f"≈ 年化超额 {((1+m)**12.5-1)*100:+.1f}%")
# 月度胜率
mo = s.iloc[::20].dropna()
log(f"  非重叠期胜率 {(mo>0).mean():.3f} ({int((mo>0).sum())}/{len(mo)})")

log("\n" + "=" * 78 + "\nC) 事前固定阈值版(不依赖截面分位,信息时点更干净)")
with np.errstate(all="ignore"):
    fix = (RAW["box20"] > 0.30) & (RAW["v2"] < 0.65)
s2, n2 = daily_mean(fix)
r2 = {y: float(s2[s2.index.year == y].mean() * 100) for y in range(2020, 2026)}
log("  逐年 ex20(%): " + str({k: round(v, 2) for k, v in r2.items()}) +
    f"  每日只数中位 {float(n2[n2>0].median()):.0f}")
for seg, yy in (("train", s2.index.year <= 2023), ("test", s2.index.year >= 2024)):
    m, t, nn = tstat_nonoverlap(s2[yy])
    log(f"  {seg}: mean={m*100:+.3f}%  t={t:+.2f}  n={nn}")

log("\n  再拆:同样宽箱缩量,按 20 日涨跌拆开")
for lab, extra in (("20日下跌", RAW["mom20"] < 0), ("20日上涨", RAW["mom20"] >= 0)):
    s3, n3 = daily_mean(CELL & extra)
    for seg, yy in (("train", s3.index.year <= 2023), ("test", s3.index.year >= 2024)):
        m, t, nn = tstat_nonoverlap(s3[yy])
        log(f"    {lab} {seg}: mean={m*100:+.3f}%  t={t:+.2f}  每日只数中位 {float(n3[n3>0].median()):.0f}")

log("\n" + "=" * 78 + "\nD) 用户结构 vs 反结构 的日频序列对照")
up = R["mom120"] > 0.7
near = R["pos252"] > 0.8
comps = {
    "用户结构 量价齐升(mom120高+近高点+放量)": up & near & (Q["v2"] == 4),
    "反结构   近高点+缩量": up & near & (Q["v2"] == 0),
    "候选     宽箱+缩量(box20 Q4 x v2 Q0)": CELL,
    "候选∩下跌 宽箱+缩量+20日下跌": CELL & (RAW["mom20"] < 0),
}
for lab, m in comps.items():
    sx, nx = daily_mean(m)
    rr = {y: round(float(sx[sx.index.year == y].mean() * 100), 2) for y in range(2020, 2026)}
    a, ta, na = tstat_nonoverlap(sx[sx.index.year <= 2023])
    b, tb, nb = tstat_nonoverlap(sx[sx.index.year >= 2024])
    log(f"  {lab}\n     逐年 {rr}\n     train {a*100:+.2f}% t={ta:+.2f} | test {b*100:+.2f}% t={tb:+.2f} "
        f"| 每日只数中位 {float(nx[nx>0].median()):.0f}")
log(f"DONE rss={rss():.2f}GB")
