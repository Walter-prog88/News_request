"""路径 C:量价配合的二维格子搜索(形态/量价序列,不用现有因子库)。

价格轴 6 个 x 量能轴 5 个 = 30 个 5x5 格子表,分 2020-23(搜索段)/ 2024-25(检验段)。
外加用户点名的复合条件("量价齐升"、"缩量横盘后放量突破")。
稳健性:置换检验(按日在池内打乱前瞻收益)给最好格子的零分布。

内存约束:峰值 < 3GB —— 所有 rolling 按列分块;特征算完立即分桶成 int8 后释放。
"""
from __future__ import annotations
import gc, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
from afactor.strategy import universe
from afactor import newdata

OUT = "/path/to/user/Desktop/A stock market/research/pathC_volprice_grid"
os.makedirs(OUT, exist_ok=True)
NQ = 5
CH = 800          # 列分块大小
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def rss():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        return -1.0


# ---------------------------------------------------------------- 数据
p = Panel()
dates, codes = p.dates, p.codes
T, N = len(dates), len(codes)
YR = dates.year.values
log(f"panel {T}x{N}  rss={rss():.2f}GB")

U = universe(p, "V1000").values.copy()               # bool (T,N)
close = p.raw("close_px").values.astype(np.float32)
high = p.raw("high_px").values.astype(np.float32)
low = p.raw("low_px").values.astype(np.float32)
amount = p.raw("amount").values.astype(np.float32)
tw = p.raw("twap_open10").values.astype(np.float32)
p._cache.clear()
gc.collect()
log(f"raw loaded rss={rss():.2f}GB")


# ---------------------------------------------------------------- 工具
def roll(a: np.ndarray, w: int, fn: str, minp: int | None = None) -> np.ndarray:
    """按列分块滚动,float32 输出,内存受控。"""
    minp = minp or max(w // 2, 5)
    out = np.empty(a.shape, np.float32)
    for s in range(0, a.shape[1], CH):
        e = min(s + CH, a.shape[1])
        blk = pd.DataFrame(a[:, s:e])
        out[:, s:e] = getattr(blk.rolling(w, min_periods=minp), fn)().values.astype(np.float32)
        del blk
    gc.collect()
    return out


def pctrank(a: np.ndarray) -> np.ndarray:
    """截面百分位秩(仅 U 内),uint8 0..200,255=NaN。分块按行做。"""
    out = np.full(a.shape, 255, np.uint8)
    for s in range(0, T, 200):
        e = min(s + 200, T)
        x = np.where(U[s:e], a[s:e], np.nan).astype(np.float32)
        r = pd.DataFrame(x).rank(axis=1, pct=True).values
        ok = np.isfinite(r)
        v = np.clip((r[ok] * 200.0), 0, 200).astype(np.uint8)
        blk = np.full(x.shape, 255, np.uint8)
        blk[ok] = v
        out[s:e] = blk
        del x, r, ok, v, blk
    gc.collect()
    return out


def to_bucket(rk: np.ndarray) -> np.ndarray:
    """uint8 百分位(0..200,255=NaN) -> 0..NQ-1,-1=NaN。"""
    b = np.full(rk.shape, -1, np.int8)
    ok = rk != 255
    b[ok] = np.clip((rk[ok].astype(np.int16) * NQ) // 201, 0, NQ - 1).astype(np.int8)
    return b


RANKS: dict[str, np.ndarray] = {}     # uint8 percentile panels


def add_feat(name: str, arr: np.ndarray):
    with np.errstate(all="ignore"):
        RANKS[name] = pctrank(arr)
    log(f"  feat {name}  rss={rss():.2f}GB")


# ---------------------------------------------------------------- 前瞻收益(超额)
def fwd_excess(h: int) -> np.ndarray:
    f = np.full((T, N), np.nan, np.float32)
    hi = 1 + h
    den = tw[1:T - h]
    f[: T - hi] = tw[hi:] / np.where(den > 0, den, np.nan) - 1.0
    f[~np.isfinite(f)] = np.nan
    f[~U] = np.nan
    mu = np.nanmean(f, axis=1, keepdims=True)
    return (f - mu).astype(np.float32)


with np.errstate(all="ignore"):
    ex20 = fwd_excess(20)
    ex40 = fwd_excess(40)
del tw
gc.collect()
log(f"fwd done rss={rss():.2f}GB")

# ---------------------------------------------------------------- 价格轴
log("price-axis ...")
ret = np.full((T, N), np.nan, np.float32)
with np.errstate(all="ignore"):
    ret[1:] = close[1:] / close[:-1] - 1.0
ret[~np.isfinite(ret)] = np.nan

with np.errstate(all="ignore"):
    m252 = roll(close, 252, "max", 150)
    add_feat("P1_pos252", close / m252)                      # 高=贴近 252 日高点
    del m252
    m60 = roll(high, 60, "max", 40)
    add_feat("P2_dd60", close / m60 - 1.0)                   # 高=回撤小
    del m60
    s20 = roll(ret, 20, "std", 15)
    s60 = roll(ret, 60, "std", 40)
    add_feat("P3_volconv", s20 / np.where(s60 > 0, s60, np.nan))   # 低=波动收敛
    del s20, s60
    sabs = roll(np.abs(ret), 60, "sum", 40)
    net = np.full((T, N), np.nan, np.float32)
    net[60:] = close[60:] / close[:-60] - 1.0
    add_feat("P4_patheff60", net / np.where(sabs > 0, sabs, np.nan))  # 高=平滑上升通道
    del sabs, net
    h20m = roll(high, 20, "max", 15)
    l20m = roll(low, 20, "min", 15)
    add_feat("P5_box20", (h20m - l20m) / close)              # 低=窄箱
    del l20m
    mom = np.full((T, N), np.nan, np.float32)
    mom[120:] = close[120:] / close[:-120] - 1.0
    add_feat("P6_mom120", mom)
    del mom, ret, low
gc.collect()
log(f"price feats done rss={rss():.2f}GB")

# ---------------------------------------------------------------- 量能轴
log("volume-axis ...")
with np.errstate(all="ignore"):
    a20 = roll(amount, 20, "mean", 15)
    a60 = roll(amount, 60, "mean", 40)
    a5 = roll(amount, 5, "mean", 4)
    add_feat("V1_amt5_60", a5 / np.where(a60 > 0, a60, np.nan))
    del a5
    a120 = roll(amount, 120, "mean", 80)
    add_feat("V2_amt20_120", a20 / np.where(a120 > 0, a120, np.nan))
    del a120

    # V3 log(成交额) 20 日 OLS 斜率 / 20 日标准差(分块 float64)
    w = 20
    Sxx = w * (w * w - 1) / 12.0
    jc = (np.arange(T, dtype=np.float64) - T / 2.0)[:, None]
    slope = np.empty((T, N), np.float32)
    for s in range(0, N, CH):
        e = min(s + CH, N)
        la = np.log(np.where(amount[:, s:e] > 0, amount[:, s:e], np.nan)).astype(np.float64)
        la = pd.DataFrame(la).ffill(limit=10).values
        B = pd.DataFrame(la).rolling(w, min_periods=w).sum().values
        A = pd.DataFrame(la * jc).rolling(w, min_periods=w).sum().values
        st = jc - (w - 1)
        slope[:, s:e] = ((A - st * B - ((w - 1) / 2.0) * B) / Sxx).astype(np.float32)
        del la, A, B, st
    sd_la = roll(np.log(np.where(amount > 0, amount, np.nan)).astype(np.float32), w, "std", w)
    add_feat("V3_amtslope20", slope * w / np.where(sd_la > 1e-6, sd_la, np.nan))
    del slope, sd_la, jc
    gc.collect()

    turn = newdata.turnover_rate(dates, codes).reindex(index=dates, columns=codes).values.astype(np.float32)
    add_feat("V4_turn20", roll(turn, 20, "mean", 15))
    del turn

    big = (amount > 1.5 * a60).astype(np.float32)
    big[~np.isfinite(amount)] = np.nan
    add_feat("V5_bigvol20", roll(big, 20, "sum", 15))
    del big

    # 复合条件原料(bool,便宜)
    prev_h20 = np.roll(h20m, 1, axis=0)
    prev_h20[0] = np.nan
    breakout = np.zeros((T, N), bool)
    breakout[21:] = close[21:] > prev_h20[21:]
    vol_spike = np.zeros((T, N), bool)
    vol_spike[:] = amount > 1.5 * a20
    del prev_h20, h20m, a20, a60, amount, close, high
gc.collect()
log(f"volume feats done rss={rss():.2f}GB")

PKEYS = ["P1_pos252", "P2_dd60", "P3_volconv", "P4_patheff60", "P5_box20", "P6_mom120"]
VKEYS = ["V1_amt5_60", "V2_amt20_120", "V3_amtslope20", "V4_turn20", "V5_bigvol20"]
buckets = {k: to_bucket(RANKS[k]) for k in PKEYS + VKEYS}
log(f"buckets done rss={rss():.2f}GB")

# ---------------------------------------------------------------- 扁平化
valid = U.copy()
for k in buckets:
    valid &= buckets[k] >= 0
valid &= np.isfinite(ex20) & np.isfinite(ex40)
log(f"valid cells {valid.sum()/1e6:.2f}M")

didx = np.repeat(np.arange(T, dtype=np.int32)[:, None], N, axis=1)
FLAT = {}
for name, ym in (("train", YR <= 2023), ("test", YR >= 2024)):
    m = valid & ym[:, None]
    d = {k: buckets[k][m].astype(np.int32) for k in buckets}
    d["ex20"] = ex20[m].astype(np.float64)
    d["ex40"] = ex40[m].astype(np.float64)
    d["date"] = didx[m]
    for rk in ("P1_pos252", "P6_mom120", "V1_amt5_60", "P3_volconv", "P5_box20"):
        d["rk_" + rk] = RANKS[rk][m].astype(np.float32) / 200.0
    d["brk"] = breakout[m]
    d["spike"] = vol_spike[m]
    FLAT[name] = d
    log(f"  flat {name}: {len(d['ex20'])/1e6:.2f}M rows  rss={rss():.2f}GB")

del buckets, RANKS, ex20, ex40, didx, valid, U, breakout, vol_spike
gc.collect()
log(f"flattened rss={rss():.2f}GB")


# ---------------------------------------------------------------- 格子统计
def grid(d, pk, vk, ycol):
    idx = d[pk] * NQ + d[vk]
    s = np.bincount(idx, weights=d[ycol], minlength=NQ * NQ)
    n = np.bincount(idx, minlength=NQ * NQ)
    return (s / np.maximum(n, 1)).reshape(NQ, NQ), n.reshape(NQ, NQ)


results, rows = {}, []
for pk in PKEYS:
    for vk in VKEYS:
        rec = {"price": pk, "vol": vk}
        for seg in ("train", "test"):
            for h in ("ex20", "ex40"):
                g, n = grid(FLAT[seg], pk, vk, h)
                rec[f"{seg}_{h}"] = g.tolist()
                rec[f"{seg}_{h}_n"] = n.tolist()
        results[f"{pk}|{vk}"] = rec
        gtr, gte = np.array(rec["train_ex20"]), np.array(rec["test_ex20"])
        ntr, nte = np.array(rec["train_ex20_n"]), np.array(rec["test_ex20_n"])
        gm = np.where(ntr >= 3000, gtr, -9)
        i, j = np.unravel_index(np.argmax(gm), gm.shape)
        rows.append(dict(pair=f"{pk}|{vk}", cell=f"({i},{j})",
                         train20=float(gtr[i, j]), test20=float(gte[i, j]),
                         train40=float(np.array(rec["train_ex40"])[i, j]),
                         test40=float(np.array(rec["test_ex40"])[i, j]),
                         n_tr=int(ntr[i, j]), n_te=int(nte[i, j])))
        log(f"  {pk} x {vk}: best({i},{j}) tr20={gtr[i,j]*100:+.2f}% te20={gte[i,j]*100:+.2f}%")

summ = pd.DataFrame(rows).sort_values("train20", ascending=False)
summ.to_csv(f"{OUT}/pair_best_cells.csv", index=False)
log("\n=== 每个 pair 在训练段最好的格子,及其检验段表现 ===\n" + summ.to_string(index=False))


def all_cells(seg, ycol):
    out = []
    for pk in PKEYS:
        for vk in VKEYS:
            g, n = grid(FLAT[seg], pk, vk, ycol)
            for i in range(NQ):
                for j in range(NQ):
                    out.append((pk, vk, i, j, g[i, j], n[i, j]))
    return pd.DataFrame(out, columns=["price", "vol", "i", "j", "ex", "n"])


ac = all_cells("train", "ex20").merge(all_cells("test", "ex20"),
                                      on=["price", "vol", "i", "j"], suffixes=("_tr", "_te"))
ac40 = all_cells("train", "ex40").merge(all_cells("test", "ex40"),
                                        on=["price", "vol", "i", "j"], suffixes=("_tr", "_te"))
ac = ac.merge(ac40[["price", "vol", "i", "j", "ex_tr", "ex_te"]].rename(
    columns={"ex_tr": "ex40_tr", "ex_te": "ex40_te"}), on=["price", "vol", "i", "j"])
ac = ac[ac.n_tr >= 3000].sort_values("ex_tr", ascending=False)
ac.to_csv(f"{OUT}/all_cells.csv", index=False)
log("\n=== Top-20 cells by train ex20 ===\n" + ac.head(20).to_string(index=False))
log("\n=== Bottom-10 cells by train ex20 ===\n" + ac.tail(10).to_string(index=False))
log(f"\n格子层面: 两段同为正占比={((ac.ex_tr>0)&(ac.ex_te>0)).mean():.3f}  "
    f"pearson(tr,te)={np.corrcoef(ac.ex_tr, ac.ex_te)[0,1]:.3f}  "
    f"spearman={ac.ex_tr.corr(ac.ex_te, method='spearman'):.3f}  n_cells={len(ac)}")
# 训练段 top-10% 格子在检验段的平均
k = max(1, len(ac) // 10)
log(f"训练段前 10% 格子({k} 个)在检验段平均 ex20 = {ac.head(k).ex_te.mean()*100:+.3f}%  "
    f"(全部格子检验段均值 {ac.ex_te.mean()*100:+.3f}%)")

# ---------------------------------------------------------------- 置换检验
NPERM = 20
rng = np.random.default_rng(20260827)
d = FLAT["train"]
order = np.argsort(d["date"], kind="stable")
dsorted = d["date"][order]
cut = np.flatnonzero(np.diff(dsorted)) + 1
segs = np.concatenate(([0], cut, [len(dsorted)]))
focus = [(r.price, r.vol, int(r.i), int(r.j)) for r in ac.head(5).itertuples()]
perm_focus = {str(f): [] for f in focus}
perm_max = []
for it in range(NPERM):
    ysort = d["ex20"][order].copy()
    for a, b in zip(segs[:-1], segs[1:]):
        if b > a + 1:
            ysort[a:b] = rng.permutation(ysort[a:b])
    y = np.empty_like(ysort)
    y[order] = ysort
    best = -9.0
    for pk in PKEYS:
        for vk in VKEYS:
            idx = d[pk] * NQ + d[vk]
            s = np.bincount(idx, weights=y, minlength=NQ * NQ)
            n = np.bincount(idx, minlength=NQ * NQ)
            g = s / np.maximum(n, 1)
            g[n < 3000] = -9
            best = max(best, float(g.max()))
            for (fp, fv, fi, fj) in focus:
                if fp == pk and fv == vk:
                    perm_focus[str((fp, fv, fi, fj))].append(float(g[fi * NQ + fj]))
    perm_max.append(best)
    log(f"  perm {it+1}/{NPERM} max={best*100:+.3f}%")

real_max = float(ac.ex_tr.max())
pm = np.array(perm_max)
log(f"\n=== 置换零分布(750 格 max,{NPERM} 次)===\nmean={pm.mean()*100:+.3f}%  sd={pm.std()*100:.3f}%  "
    f"min={pm.min()*100:+.3f}%  max={pm.max()*100:+.3f}%\n真实 max={real_max*100:+.3f}%  "
    f"超过零分布的比例={(pm < real_max).mean():.3f}")
for f in focus:
    v = np.array(perm_focus[str(f)])
    realv = float(ac[(ac.price == f[0]) & (ac.vol == f[1]) & (ac.i == f[2]) & (ac.j == f[3])].ex_tr.iloc[0])
    log(f"  focus {f}: real={realv*100:+.3f}%  null mean={v.mean()*100:+.3f}% sd={v.std()*100:.3f}%  "
        f"z={(realv-v.mean())/max(v.std(),1e-9):+.2f}")

# ---------------------------------------------------------------- 用户点名条件
def cond_stats(d, m, label, seg):
    r = {"seg": seg, "label": label, "n": int(m.sum()), "freq": float(m.mean())}
    if m.sum() < 50:
        r.update(ex20=np.nan, ex40=np.nan, hit20=np.nan)
    else:
        r["ex20"] = float(d["ex20"][m].mean())
        r["ex40"] = float(d["ex40"][m].mean())
        r["hit20"] = float((d["ex20"][m] > 0).mean())
    return r


cond_rows = []
for seg in ("train", "test"):
    d = FLAT[seg]
    pos, v1, vc, bx, mm = (d["rk_P1_pos252"], d["rk_V1_amt5_60"], d["rk_P3_volconv"],
                           d["rk_P5_box20"], d["rk_P6_mom120"])
    conds = {
        "A  量价齐升 pos252>.8 & amt5/60>.8": (pos > .8) & (v1 > .8),
        "A' 量价齐升(严) >.9 & >.9": (pos > .9) & (v1 > .9),
        "A''量价齐升 ∩ mom120 前30%": (pos > .8) & (v1 > .8) & (mm > .7),
        "B  缩量窄箱+放量突破": (vc < .3) & (bx < .3) & d["brk"] & d["spike"],
        "B' 同上 ∩ mom120 前30%": (vc < .3) & (bx < .3) & d["brk"] & d["spike"] & (mm > .7),
        "B''缩量窄箱+放量突破(不要求突破)": (vc < .3) & (bx < .3) & d["spike"],
        "C  对照 近高点但缩量 pos>.8 & amt5/60<.2": (pos > .8) & (v1 < .2),
        "D  对照 全池": np.ones(len(d["ex20"]), bool),
    }
    for lab, m in conds.items():
        cond_rows.append(cond_stats(d, m, lab, seg))
cond = pd.DataFrame(cond_rows)[["seg", "label", "n", "freq", "ex20", "ex40", "hit20"]]
cond.to_csv(f"{OUT}/named_conditions.csv", index=False)
log("\n=== 用户点名条件 ===\n" + cond.to_string(index=False))

# ---------------------------------------------------------------- 焦点格子按年
yr_rows = []
for (fp, fv, fi, fj) in focus:
    for seg in ("train", "test"):
        d = FLAT[seg]
        m = (d[fp] == fi) & (d[fv] == fj)
        yy = YR[d["date"][m]]
        s = pd.Series(d["ex20"][m]).groupby(yy).mean()
        for y_ in s.index:
            yr_rows.append(dict(cell=f"{fp}[{fi}]x{fv}[{fj}]", year=int(y_), ex20=float(s[y_])))
yr = pd.DataFrame(yr_rows)
yr.to_csv(f"{OUT}/focus_cells_by_year.csv", index=False)
log("\n=== 训练段前 5 名格子按年 ex20 ===\n" +
    yr.pivot(index="cell", columns="year", values="ex20").mul(100).round(2).to_string())

json.dump({"grids": results, "perm_max": perm_max, "real_max": real_max,
           "perm_focus": perm_focus, "focus": [list(f) for f in focus]},
          open(f"{OUT}/grids.json", "w"))
log(f"DONE rss={rss():.2f}GB")
