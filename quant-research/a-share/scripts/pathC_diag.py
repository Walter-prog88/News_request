"""路径 C 诊断:
1) 量能轴 vs 价格轴哪个稳定?两段各自的单轴单调性 + 边际检验(控制另一轴)。
2) 唯一两段都强的格子 P5_box20[4] x V2_amt20_120[0] 的画像与时间序列。
3) 用户结构的"阶段化"检验:中长期上涨 -> 回调 -> 缩量横盘 -> 放量突破,四个阶段分别的超额。
4) 控制真实市值/换手率后,量能轴还剩多少。
"""
from __future__ import annotations
import gc, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
from afactor.strategy import universe
from afactor import newdata

OUT = "/path/to/user/Desktop/A stock market/research/pathC_volprice_grid"
NQ = 5
CH = 800
T0 = time.time()
pd.set_option("display.width", 220)


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def rss():
    import psutil, os as _o
    return psutil.Process(_o.getpid()).memory_info().rss / 1e9


p = Panel()
dates, codes = p.dates, p.codes
T, N = len(dates), len(codes)
YR, YM = dates.year.values, (dates.year * 100 + dates.month).values
U = universe(p, "V1000").values.copy()

# --- 换手率/真实市值的截面秩由 scripts/pathC_prep_newdata.py 单独进程预算好 ---
def _load_rk(path):
    a = np.load(path)
    out = np.full(a.shape, np.nan, np.float32)
    ok = a != 255
    out[ok] = a[ok].astype(np.float32) / 200.0
    return out


RK_TURN = _load_rk("/tmp/rk_turn20.npy")
RK_MV = _load_rk("/tmp/rk_logmv.npy")
log(f"newdata ranks loaded rss={rss():.2f}GB")

close = p.raw("close_px").values.astype(np.float32)
high = p.raw("high_px").values.astype(np.float32)
low = p.raw("low_px").values.astype(np.float32)
amount = p.raw("amount").values.astype(np.float32)
tw = p.raw("twap_open10").values.astype(np.float32)
p._cache.clear(); gc.collect()
log(f"raw loaded rss={rss():.2f}GB")


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
    ex20, ex40 = fwd(20), fwd(40)
del tw; gc.collect()

with np.errstate(all="ignore"):
    ret = np.full((T, N), np.nan, np.float32); ret[1:] = close[1:] / close[:-1] - 1.0
    ret[~np.isfinite(ret)] = np.nan
    a20 = roll(amount, 20, "mean", 15); a60 = roll(amount, 60, "mean", 40)
    a120 = roll(amount, 120, "mean", 80); a5 = roll(amount, 5, "mean", 4)
    hh20 = roll(high, 20, "max", 15); ll20 = roll(low, 20, "min", 15)
    hh60 = roll(high, 60, "max", 40); cc252 = roll(close, 252, "max", 150)
    s20 = roll(ret, 20, "std", 15); s60 = roll(ret, 60, "std", 40)

    F = {}
    F["box20"] = (hh20 - ll20) / close
    F["v2"] = a20 / np.where(a120 > 0, a120, np.nan)
    F["v1"] = a5 / np.where(a60 > 0, a60, np.nan)
    F["pos252"] = close / cc252
    F["dd60"] = close / hh60 - 1.0
    F["volconv"] = s20 / np.where(s60 > 0, s60, np.nan)
    mom120 = np.full((T, N), np.nan, np.float32); mom120[120:] = close[120:] / close[:-120] - 1.0
    F["mom120"] = mom120
    mom20 = np.full((T, N), np.nan, np.float32); mom20[20:] = close[20:] / close[:-20] - 1.0
    F["mom20"] = mom20
    prev_h20 = np.roll(hh20, 1, axis=0); prev_h20[0] = np.nan
    brk = np.zeros((T, N), bool); brk[21:] = close[21:] > prev_h20[21:]
    spike = amount > 1.5 * a20

    R = {k: pctrank(v) for k, v in F.items()}
    R["turn20"] = RK_TURN
    R["logmv"] = RK_MV
    F["turn20"] = RK_TURN      # 只用秩,原值不再需要
    F["logmv"] = RK_MV
del a5, a60, a120, hh60, cc252, s20, s60, ll20, prev_h20, ret, high, low
gc.collect()
log(f"feats done rss={rss():.2f}GB")

valid = U & np.isfinite(ex20) & np.isfinite(ex40)
for k in R:
    valid &= np.isfinite(R[k])
didx = np.repeat(np.arange(T, dtype=np.int32)[:, None], N, axis=1)
cidx = np.repeat(np.arange(N, dtype=np.int32)[None, :], T, axis=0)

D = {("rk_" + k): R[k][valid] for k in R}
D.update({("f_" + k): F[k][valid] for k in ("box20", "v2", "mom120", "mom20", "dd60", "turn20", "logmv")})
D["ex20"] = ex20[valid].astype(np.float64)
D["ex40"] = ex40[valid].astype(np.float64)
D["t"] = didx[valid]; D["c"] = cidx[valid]
D["brk"] = brk[valid]; D["spike"] = spike[valid]
D["yr"] = YR[D["t"]]
D["tr"] = D["yr"] <= 2023
del F, R, ex20, ex40, didx, cidx, valid, brk, spike, close, amount, a20, hh20, U, mom120, mom20
gc.collect()
log(f"flat {len(D['ex20'])/1e6:.2f}M rows rss={rss():.2f}GB")

TR, TE = D["tr"], ~D["tr"]


def qcut(r, nq=NQ):
    return np.clip((r * nq).astype(np.int8), 0, nq - 1)


def report(mask_dict, title, y="ex20"):
    rows = []
    for lab, m in mask_dict.items():
        for seg, sm in (("train", TR), ("test", TE)):
            mm = m & sm
            rows.append(dict(cond=lab, seg=seg, n=int(mm.sum()),
                             per_day=float(mm.sum() / len(np.unique(D["t"][sm]))) if mm.sum() else 0.0,
                             ex20=float(D["ex20"][mm].mean()) if mm.sum() > 30 else np.nan,
                             ex40=float(D["ex40"][mm].mean()) if mm.sum() > 30 else np.nan,
                             hit=float((D["ex20"][mm] > 0).mean()) if mm.sum() > 30 else np.nan))
    df = pd.DataFrame(rows).pivot(index="cond", columns="seg", values=["n", "per_day", "ex20", "ex40", "hit"])
    log(f"\n=== {title} ===\n" + df.round(4).to_string())
    return df


# ---------------- 1) 单轴单调性 ----------------
log("\n" + "=" * 70 + "\n1) 单轴五分位(两段)")
rows = []
for k in ["pos252", "dd60", "volconv", "box20", "mom120", "v1", "v2", "turn20"]:
    q = qcut(D["rk_" + k])
    for seg, sm in (("train", TR), ("test", TE)):
        for b in range(NQ):
            m = sm & (q == b)
            rows.append(dict(feat=k, seg=seg, q=b, ex20=D["ex20"][m].mean()))
mono = pd.DataFrame(rows).pivot_table(index=["feat", "seg"], columns="q", values="ex20").mul(100).round(3)
mono["Q4-Q0"] = (mono[4] - mono[0]).round(3)
log("\n" + mono.to_string())

# ---------------- 2) 量能轴在控制价格轴后的边际 ----------------
log("\n" + "=" * 70 + "\n2) 边际:在每个价格分位内,量能 Q0-Q4 的差 (ex20, %)")
for pk in ["pos252", "mom120", "box20"]:
    for vk in ["v1", "v2", "turn20"]:
        pq, vq = qcut(D["rk_" + pk]), qcut(D["rk_" + vk])
        out = {}
        for seg, sm in (("train", TR), ("test", TE)):
            sp = []
            for i in range(NQ):
                a = D["ex20"][sm & (pq == i) & (vq == 0)].mean()
                b = D["ex20"][sm & (pq == i) & (vq == 4)].mean()
                sp.append((a - b) * 100)
            out[seg] = np.round(sp, 2)
        log(f"  {pk} x {vk}: train {out['train']}   test {out['test']}")

log("\n   反过来:在每个量能分位内,价格 Q4-Q0 的差 (ex20, %)")
for pk in ["pos252", "mom120", "box20", "volconv"]:
    for vk in ["v2"]:
        pq, vq = qcut(D["rk_" + pk]), qcut(D["rk_" + vk])
        out = {}
        for seg, sm in (("train", TR), ("test", TE)):
            sp = []
            for j in range(NQ):
                a = D["ex20"][sm & (pq == 4) & (vq == j)].mean()
                b = D["ex20"][sm & (pq == 0) & (vq == j)].mean()
                sp.append((a - b) * 100)
            out[seg] = np.round(sp, 2)
        log(f"  {pk} x {vk}: train {out['train']}   test {out['test']}")

# ---------------- 3) 最好格子画像 ----------------
log("\n" + "=" * 70 + "\n3) P5_box20[Q4] x V2_amt20_120[Q0] 画像")
bq, vq = qcut(D["rk_box20"]), qcut(D["rk_v2"])
CELL = (bq == 4) & (vq == 0)
for seg, sm in (("train", TR), ("test", TE)):
    m = CELL & sm
    base = sm
    log(f"  {seg}: n={m.sum()}  每日 {m.sum()/len(np.unique(D['t'][sm])):.1f} 只  "
        f"ex20={D['ex20'][m].mean()*100:+.2f}%  ex40={D['ex40'][m].mean()*100:+.2f}%  hit={(D['ex20'][m]>0).mean():.3f}")
    for k in ["mom120", "mom20", "dd60", "box20", "v2", "turn20", "logmv"]:
        log(f"      {k}: cell median={np.median(D['f_'+k][m]):+.3f}  池中位数={np.median(D['f_'+k][base]):+.3f}  "
            f"cell 截面秩中位={np.median(D['rk_'+k][m]):.2f}")
# 逐年 + 非重叠 t 统计
yr_ex = pd.Series(D["ex20"][CELL]).groupby(D["yr"][CELL]).agg(["mean", "size"])
log("\n  逐年:\n" + yr_ex.assign(mean=lambda x: (x["mean"] * 100).round(2)).to_string())
nov = np.isin(D["t"][CELL], np.arange(0, T, 20))
y = D["ex20"][CELL][nov]
log(f"  非重叠(每20交易日取一截面)n={len(y)}  mean={y.mean()*100:+.3f}%  "
    f"t={y.mean()/ (y.std()/np.sqrt(len(y))):+.2f}")
# 按日均值序列的 t(每日 cell 均值,更保守)
s = pd.Series(D["ex20"][CELL]).groupby(D["t"][CELL]).mean()
s = s[s.index % 20 == 0]
log(f"  日截面均值序列(每20日取1)n={len(s)}  mean={s.mean()*100:+.3f}%  t={s.mean()/(s.std()/np.sqrt(len(s))):+.2f}")

# 宽箱是"暴涨"还是"暴跌"造成?
for seg, sm in (("train", TR), ("test", TE)):
    m = CELL & sm
    up = D["f_mom20"][m] > 0
    log(f"  {seg}: cell 内 20日涨的占比 {up.mean():.3f};"
        f" 涨的 ex20={D['ex20'][m][up].mean()*100:+.2f}% / 跌的 ex20={D['ex20'][m][~up].mean()*100:+.2f}%")

# 市值控制
log("\n  控制真实市值(市值五分位内的 cell 超额):")
mq = qcut(D["rk_logmv"])
for seg, sm in (("train", TR), ("test", TE)):
    v = [float(D["ex20"][CELL & sm & (mq == k)].mean() * 100) if (CELL & sm & (mq == k)).sum() > 100 else np.nan
         for k in range(NQ)]
    b = [float(D["ex20"][sm & (mq == k)].mean() * 100) for k in range(NQ)]
    log(f"    {seg} cell={np.round(v,2)}  同市值桶基准={np.round(b,2)}  差={np.round(np.array(v)-np.array(b),2)}")

# ---------------- 4) 用户结构的阶段化检验 ----------------
log("\n" + "=" * 70 + "\n4) 用户结构分阶段(先有 120 日中长期上涨,再看当前处于哪一阶段)")
up = D["rk_mom120"] > 0.7          # 中长期趋势
pull = (D["f_dd60"] < -0.08) & (D["f_dd60"] > -0.25)     # 距60日高回调 8-25%
quiet = (D["rk_volconv"] < 0.4) & (D["rk_box20"] < 0.4)  # 横盘震荡(波动收敛+窄箱)
shrink = D["rk_v2"] < 0.3          # 缩量
expand = D["rk_v2"] > 0.7          # 放量
near = D["rk_pos252"] > 0.8        # 贴近年内高点
conds = {
    "S0 只要求 120日涨幅前30%": up,
    "S1 上涨+回调(8-25%)": up & pull,
    "S2 上涨+回调+横盘": up & pull & quiet,
    "S3 上涨+回调+横盘+缩量  <= 用户结构的'蓄势'": up & pull & quiet & shrink,
    "S4 上涨+回调+横盘+放量  ": up & pull & quiet & expand,
    "S5 上涨+回调+横盘+当日放量突破 <= 用户结构的'重新走强'": up & pull & quiet & D["brk"] & D["spike"],
    "S6 上涨+贴近高点+缩量": up & near & shrink,
    "S7 上涨+贴近高点+放量  <= '量价齐升'": up & near & expand,
    "S8 上涨+贴近高点+当日放量突破": up & near & D["brk"] & D["spike"],
    "Z  全池": np.ones(len(D["ex20"]), bool),
}
report(conds, "阶段化结果")

# ---------------- 5) 缩量 vs 放量 的价差按年 ----------------
log("\n" + "=" * 70 + "\n5) '贴近年内高点'内部:缩量(v2 Q0) 减 放量(v2 Q4) 的 ex20 差,逐年")
rows = []
for yv in range(2020, 2026):
    m = (D["yr"] == yv) & near
    a = D["ex20"][m & (vq == 0)].mean(); b = D["ex20"][m & (vq == 4)].mean()
    rows.append(dict(year=yv, low_vol=a * 100, high_vol=b * 100, spread=(a - b) * 100,
                     n_low=int((m & (vq == 0)).sum()), n_high=int((m & (vq == 4)).sum())))
log("\n" + pd.DataFrame(rows).round(2).to_string(index=False))

# ---------------- 6) 300308.SZ 描述 ----------------
tgt = "300308.SZ"
if tgt in list(codes):
    ci = list(codes).index(tgt)
    m = D["c"] == ci
    if m.sum():
        sub = pd.DataFrame({"yr": D["yr"][m], "ex20": D["ex20"][m],
                            "rk_pos252": D["rk_pos252"][m], "rk_v2": D["rk_v2"][m],
                            "rk_box20": D["rk_box20"][m], "rk_mom120": D["rk_mom120"][m]})
        log("\n" + "=" * 70 + f"\n6) {tgt} 逐年:自身未来20日超额 与 其所处状态的截面秩中位数")
        log("\n" + sub.groupby("yr").agg(n=("ex20", "size"), ex20=("ex20", "mean"),
                                         pos252=("rk_pos252", "median"), v2=("rk_v2", "median"),
                                         box20=("rk_box20", "median"), mom120=("rk_mom120", "median")
                                         ).round(3).to_string())
log(f"DONE rss={rss():.2f}GB")
