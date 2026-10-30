"""路径 C 补充:真正数据驱动的"形状搜索"——对 60 日归一化(价格路径, 成交额路径)做聚类,
看有没有哪一类形状在两段都有正超额;并把用户的样本股 300308.SZ 落到这些类上做描述性核对。

不猜形状:形状由 k-means 从数据里长出来。K=24。
2020-2023 上训练 k-means 与统计,2024-2025 只做 out-of-sample 打分。
内存:按抽样日期逐日构造窗口,峰值 < 2GB。
"""
from __future__ import annotations
import gc, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
from afactor.strategy import universe

OUT = "/path/to/user/Desktop/A stock market/research/pathC_volprice_grid"
os.makedirs(OUT, exist_ok=True)
W = 60          # 窗口长度
NP = 12         # 降采样点数
K = 24
STEP = 3        # 每 3 个交易日取一个截面
T0 = time.time()


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
close = p.raw("close_px").values.astype(np.float32)
amount = p.raw("amount").values.astype(np.float32)
tw = p.raw("twap_open10").values.astype(np.float32)
p._cache.clear(); gc.collect()
log(f"loaded {T}x{N} rss={rss():.2f}GB")

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

lc = np.log(np.where(close > 0, close, np.nan)).astype(np.float32)
la = np.log(np.where(amount > 0, amount, np.nan)).astype(np.float32)
la = pd.DataFrame(la).ffill(limit=5).values.astype(np.float32)
del amount; gc.collect()

pick = np.arange(W - 1, -1, -(W // NP))[::-1][-NP:]     # 窗口内降采样位置
log(f"pick idx {pick}")

sample_t = np.arange(W + 5, T - 42, STEP)
X, Y20, Y40, DT, CD = [], [], [], [], []
for t in sample_t:
    m = U[t] & np.isfinite(ex20[t]) & np.isfinite(ex40[t])
    if m.sum() < 50:
        continue
    wp = lc[t - W + 1:t + 1, m]                       # (W, k)
    wv = la[t - W + 1:t + 1, m]
    good = np.isfinite(wp).all(0) & np.isfinite(wv).all(0)
    if good.sum() < 50:
        continue
    idx = np.flatnonzero(m)[good]
    wp, wv = wp[:, good], wv[:, good]
    wp = wp - wp[0]                                    # 起点归零
    sp = wp.std(0); sp[sp < 1e-6] = np.nan
    wp = wp / sp                                       # 只留形状
    wv = wv - wv.mean(0)
    sv = wv.std(0); sv[sv < 1e-6] = np.nan
    wv = wv / sv
    f = np.concatenate([wp[pick], wv[pick]], 0).T.astype(np.float32)   # (k, 2*NP)
    ok = np.isfinite(f).all(1)
    X.append(f[ok]); Y20.append(ex20[t, idx][ok]); Y40.append(ex40[t, idx][ok])
    DT.append(np.full(ok.sum(), t, np.int32)); CD.append(idx[ok].astype(np.int32))

X = np.concatenate(X); Y20 = np.concatenate(Y20); Y40 = np.concatenate(Y40)
DT = np.concatenate(DT); CD = np.concatenate(CD)
del lc, la, ex20, ex40, close, U; gc.collect()
log(f"X {X.shape} rss={rss():.2f}GB")

tr = YR[DT] <= 2023
te = ~tr
log(f"train rows {tr.sum()/1e6:.2f}M  test rows {te.sum()/1e6:.2f}M")

from sklearn.cluster import MiniBatchKMeans
rng = np.random.default_rng(7)
sub = rng.choice(np.flatnonzero(tr), size=min(400_000, int(tr.sum())), replace=False)
km = MiniBatchKMeans(n_clusters=K, random_state=7, batch_size=8192, n_init=5, max_iter=300)
km.fit(X[sub])
lab = km.predict(X).astype(np.int16)
log(f"kmeans done rss={rss():.2f}GB")


def clus_tbl(mask, y, name):
    s = np.bincount(lab[mask], weights=y[mask].astype(np.float64), minlength=K)
    n = np.bincount(lab[mask], minlength=K)
    return pd.Series(s / np.maximum(n, 1), name=name), pd.Series(n, name="n_" + name)


e20tr, n_tr = clus_tbl(tr, Y20, "tr20")
e20te, n_te = clus_tbl(te, Y20, "te20")
e40tr, _ = clus_tbl(tr, Y40, "tr40")
e40te, _ = clus_tbl(te, Y40, "te40")
tab = pd.concat([e20tr, e20te, e40tr, e40te, n_tr, n_te], axis=1)
# 描述每个簇的形状:总涨幅、路径效率、末段量比
C = km.cluster_centers_
pp, vv = C[:, :NP], C[:, NP:]
tab["shape_net"] = pp[:, -1]                                   # 归一化后的净涨幅(单位=窗口内 std)
tab["shape_eff"] = pp[:, -1] / np.abs(np.diff(pp, axis=1)).sum(1)
tab["shape_maxdd"] = (pp - np.maximum.accumulate(pp, axis=1)).min(1)
tab["vol_end"] = vv[:, -3:].mean(1)                            # 末段量能(z)
tab["vol_trend"] = vv[:, -4:].mean(1) - vv[:, :4].mean(1)
tab = tab.sort_values("tr20", ascending=False)
tab.to_csv(f"{OUT}/shape_clusters.csv")
log("\n=== 形状簇(按训练段 ex20 排序)===\n" + tab.mul(1).round(4).to_string())
log(f"\n簇层面 spearman(tr20, te20) = {tab.tr20.corr(tab.te20, method='spearman'):.3f}  "
    f"两段同为正的簇 {(((tab.tr20>0)&(tab.te20>0)).sum())}/{K}")
log(f"训练段最好 3 簇在检验段: {tab.head(3).te20.mul(100).round(3).tolist()}  "
    f"训练段最差 3 簇在检验段: {tab.tail(3).te20.mul(100).round(3).tolist()}")

# 置换:按日打乱 Y20,看簇 max 的零分布
order = np.argsort(DT[tr], kind="stable")
ytr = Y20[tr].astype(np.float64)
dtr = DT[tr]
ds = dtr[order]
cut = np.concatenate(([0], np.flatnonzero(np.diff(ds)) + 1, [len(ds)]))
ltr = lab[tr]
pmax = []
for it in range(20):
    ys = ytr[order].copy()
    for a, b in zip(cut[:-1], cut[1:]):
        if b > a + 1:
            ys[a:b] = rng.permutation(ys[a:b])
    y = np.empty_like(ys); y[order] = ys
    s = np.bincount(ltr, weights=y, minlength=K)
    n = np.bincount(ltr, minlength=K)
    g = s / np.maximum(n, 1); g[n < 3000] = -9
    pmax.append(float(g.max()))
pmax = np.array(pmax)
log(f"\n簇置换零分布 max: mean={pmax.mean()*100:+.3f}% sd={pmax.std()*100:.3f}% max={pmax.max()*100:+.3f}%  "
    f"真实 max={tab.tr20.max()*100:+.3f}%  分位={(pmax < tab.tr20.max()).mean():.3f}")

# 打印前 3 / 后 3 簇的形状轮廓
for r in list(tab.index[:3]) + list(tab.index[-3:]):
    log(f"cluster {r}: price {np.round(C[r,:NP],2).tolist()}")
    log(f"            vol   {np.round(C[r,NP:],2).tolist()}")

# 300308.SZ 落到哪些簇
target = "300308.SZ"
if target in list(codes):
    ci = list(codes).index(target)
    m = CD == ci
    log(f"\n300308.SZ 样本数 {m.sum()},其未来 20 日超额均值 {Y20[m].mean()*100:+.2f}%")
    sub_tab = pd.DataFrame({"cluster": lab[m], "ex20": Y20[m], "year": YR[DT[m]]})
    log("按年:\n" + sub_tab.groupby("year").ex20.agg(["mean", "count"]).round(4).to_string())
    log("最常落入的簇:\n" + sub_tab.cluster.value_counts().head(6).to_string())
    log("这些簇的全池训练段 ex20:\n" +
        tab.loc[sub_tab.cluster.value_counts().head(6).index, ["tr20", "te20"]].round(4).to_string())

json.dump({"centers": C.tolist(), "table": tab.reset_index().to_dict("records"),
           "perm_max": pmax.tolist()}, open(f"{OUT}/shape_clusters.json", "w"))
log(f"DONE rss={rss():.2f}GB")
