"""路径 B 第 2 步:训练 + 样本外检验 + 两个必做对照。

模型:HistGradientBoostingRegressor,标签 = 未来 20 日收益的当日截面百分位。
特征组:
  path   : 120 维原始序列(标准化 log 价格路径 60 + 标准化 log 成交额路径 60)
  summ   : 4 维汇总统计量(60日收益/波动/成交额比/距60日高)—— 对照 (i)
  shuf   : 把 60 日窗口内的 (日收益, log成交额) 配对**随机打乱时间顺序**后重新累积成路径,
           训练与检验都用打乱版 —— 对照 (ii)。共用同一个置换 => 保留日收益与成交额的
           同期关系与全部一维分布,只破坏"时间形状"。
  comb   : path + summ (124 维)
条件版:只在当日 mom120 > 截面中位数的强势股子集内训练与检验。
"""
from __future__ import annotations
import sys, time, json, gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
sys.path.insert(0, "/path/to/user/Desktop/A stock market/scripts")
from pathb_build import Builder, OUT, H, WIN, TRAIN_END

RES = OUT / "results.json"
SEED = 20260827


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------- 统计工具 ----------------
def nw_t(x: np.ndarray, bw: int) -> tuple[float, float]:
    """序列均值的 Newey-West t(Bartlett 核,带宽 bw)。返回 (mean, t)。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return float(np.nanmean(x)) if n else np.nan, np.nan
    e = x - x.mean()
    s = float(e @ e) / n
    for l in range(1, min(bw, n - 1) + 1):
        w = 1.0 - l / (bw + 1.0)
        s += 2.0 * w * float(e[l:] @ e[:-l]) / n
    if s <= 0:
        return float(x.mean()), np.nan
    return float(x.mean()), float(x.mean() / np.sqrt(s / n))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n < 50:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def ic_summary(ic: list[float], tag: str) -> dict:
    ic = np.array([v for v in ic if np.isfinite(v)])
    m, t = nw_t(ic, max(5, 2 * H))
    return {"tag": tag, "n_days": int(len(ic)), "ic_mean": round(m, 5),
            "ic_t_nw": round(t, 2) if np.isfinite(t) else None,
            "ic_ir_daily": round(m / ic.std(), 3) if len(ic) and ic.std() > 0 else None,
            "ic_pos_frac": round(float((ic > 0).mean()), 3) if len(ic) else None}


# ---------------- 特征变换 ----------------
def to_path(X: np.ndarray) -> np.ndarray:
    """in-place: 前 60 列的日收益 -> 累积 log 价格路径。"""
    np.cumsum(X[:, :WIN], axis=1, out=X[:, :WIN])
    return X


def shuffle_time(X: np.ndarray, rng: np.random.Generator, chunk: int = 200_000) -> np.ndarray:
    """in-place 打乱窗口内时间顺序(价量共用同一置换),再对价格块 cumsum。"""
    n = len(X)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        perm = np.argsort(rng.random((e - s, WIN)), axis=1)
        X[s:e, :WIN] = np.take_along_axis(X[s:e, :WIN], perm, axis=1)
        X[s:e, WIN:] = np.take_along_axis(X[s:e, WIN:], perm, axis=1)
        np.cumsum(X[s:e, :WIN], axis=1, out=X[s:e, :WIN])
    return X


def make_model(n_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=n_iter, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=500, l2_regularization=1.0, max_bins=128,
        early_stopping=False, random_state=SEED)


# ---------------- 逐日评估(流式) ----------------
def eval_stream(b: Builder, t_list, specs, cond=False):
    """specs: {name: (model, feat_fn)};feat_fn(Xraw, aux, rng) -> 特征矩阵。
    返回 {name: {'ic': [...], 'dec': (nd,10) 平均前瞻收益, 'uni': [...]}}"""
    rng = np.random.default_rng(SEED + 7)
    out = {k: {"ic": [], "dec": [], "uni": [], "t": []} for k in specs}
    for t in t_list:
        r = b.build(t)
        if r is None:
            continue
        for k in out:
            out[k]["t"].append(t)
        Xraw, y, aux, idx = r
        fwd = b.FWD[t, idx]
        if cond:
            m = aux[:, 4] > np.median(aux[:, 4])
            if m.sum() < 200:
                continue
            Xraw, y, aux, fwd = Xraw[m], y[m], aux[m], fwd[m]
            y = pd.Series(y).rank().to_numpy() / (len(y) + 1.0)   # 子集内重排
        for name, (mdl, fn) in specs.items():
            F = fn(Xraw.copy(), aux, rng)
            pr = mdl.predict(F)
            out[name]["ic"].append(spearman(pr, y))
            q = pd.qcut(pd.Series(pr).rank(method="first"), 10, labels=False).to_numpy()
            out[name]["dec"].append([float(fwd[q == g].mean()) for g in range(10)])
            out[name]["uni"].append(float(fwd.mean()))
    for k in out:
        out[k]["dec"] = np.array(out[k]["dec"], float)
    return out


FEAT = {
    "path": lambda X, a, rng: to_path(X),
    "summ": lambda X, a, rng: a[:, :4].copy(),
    "shuf": lambda X, a, rng: shuffle_time(X, rng),
    "comb": lambda X, a, rng: np.hstack([to_path(X), a[:, :4]]),
    "pric": lambda X, a, rng: to_path(X)[:, :WIN],
    "volu": lambda X, a, rng: X[:, WIN:].copy(),
}


def train_feat(key, X, AUX, rng):
    """训练集特征构造(X 为原始未 cumsum 的 120 列,会被就地修改)。"""
    if key == "summ":
        return AUX[:, :4].copy()
    if key == "path":
        return to_path(X)
    if key == "shuf":
        return shuffle_time(X, rng)
    if key == "comb":
        return np.hstack([to_path(X), AUX[:, :4]])
    if key == "pric":
        return np.ascontiguousarray(to_path(X)[:, :WIN])
    if key == "volu":
        return np.ascontiguousarray(X[:, WIN:])
    raise KeyError(key)


def report(res, name, tag, extra=None, dates=None):
    d = res[name]
    dec = d["dec"]; uni = np.array(d["uni"])
    ls = dec[:, 9] - dec[:, 0]
    top = dec[:, 9] - uni
    bot = dec[:, 0] - uni
    m_ls, t_ls = nw_t(ls, max(5, 2 * H))
    m_top, t_top = nw_t(top, max(5, 2 * H))
    m_bot, t_bot = nw_t(bot, max(5, 2 * H))
    r = ic_summary(d["ic"], tag)
    r.update({"h20_LS": round(m_ls, 4), "h20_LS_t": round(t_ls, 2),
              "h20_D10_excess": round(m_top, 4), "h20_D10_t": round(t_top, 2),
              "h20_D1_excess": round(m_bot, 4), "h20_D1_t": round(t_bot, 2),
              "dec_mean": [round(float(v), 4) for v in dec.mean(0)]})
    if dates is not None and len(d["t"]):
        ic = np.array(d["ic"], float)
        yy = dates[np.array(d["t"])].year.to_numpy()
        r["ic_by_year"] = {int(u): round(float(np.nanmean(ic[yy == u])), 4) for u in np.unique(yy)}
    if extra:
        r.update(extra)
    return r


def main():
    results = {}
    meta = json.loads((OUT / "meta.json").read_text())
    dates = pd.to_datetime(meta["dates"])
    DT = np.load(OUT / "train_DT.npy")
    Y = np.load(OUT / "train_Y.npy")
    AUX = np.load(OUT / "train_AUX.npy")
    yr = dates[DT].year.to_numpy()
    log("train rows", len(Y), "years", np.unique(yr))

    b = Builder()
    all_t = np.arange(len(dates))
    test_t = [int(t) for t in all_t
              if dates[t] >= pd.Timestamp("2024-01-01") and t + H + 1 < len(dates)]
    valid23_t = [int(t) for t in all_t
                 if dates[t].year == 2023 and t >= 120 and t + H + 1 < len(dates)]
    log(f"test dates={len(test_t)}  inner-valid(2023) dates={len(valid23_t)}")

    # ---------- A. 内部留出:2020-2022 训练 -> 2023 验证,顺便定 n_iter ----------
    # purge:训练标签窗口不得伸进 2023 验证段
    cut = min(valid23_t) - (H + 1)
    fit_m = DT <= cut
    log("inner fit rows", int(fit_m.sum()), "cut date", dates[cut].date())

    X = np.load(OUT / "train_X.npy")
    Xf = to_path(X[fit_m])
    del X; gc.collect()
    yf = Y[fit_m]
    mdl = HistGradientBoostingRegressor(
        max_iter=50, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
        l2_regularization=1.0, max_bins=128, early_stopping=False,
        random_state=SEED, warm_start=True)
    curve = {}
    best = (None, -9)
    for n_it in [50, 100, 200, 300, 400]:
        mdl.set_params(max_iter=n_it)
        t0 = time.time()
        mdl.fit(Xf, yf)
        rr = eval_stream(b, valid23_t, {"path": (mdl, FEAT["path"])})
        s = ic_summary(rr["path"]["ic"], f"inner2023_path_it{n_it}")
        curve[n_it] = s
        log(f"  n_iter={n_it} fit {time.time()-t0:.0f}s  2023 IC={s['ic_mean']} t={s['ic_t_nw']}")
        if s["ic_mean"] > best[1]:
            best = (n_it, s["ic_mean"])
    N_ITER = best[0]
    results["inner_n_iter_curve"] = {str(k): v for k, v in curve.items()}
    results["n_iter_chosen"] = N_ITER
    log("chosen n_iter", N_ITER)
    del Xf, mdl; gc.collect()

    # 内部留出段(2023)的三个模型:path / summ / shuf
    inner = {}
    AUXf = AUX[fit_m][:, :4].copy()
    for key in ["path", "summ", "shuf", "comb", "pric", "volu"]:
        rng = np.random.default_rng(SEED)
        F = train_feat(key, np.load(OUT / "train_X.npy")[fit_m], AUXf, rng)
        m = make_model(N_ITER); m.fit(F, yf)
        del F; gc.collect()
        rr = eval_stream(b, valid23_t, {key: (m, FEAT[key])})
        inner[key] = report(rr, key, f"inner2023_{key}")
        log("  inner2023", key, inner[key])
        del m; gc.collect()
    results["inner_2023"] = inner
    del AUXf; gc.collect()

    # ---------- B. 全训练段 2020-2023 -> 检验段 2024-2025 ----------
    models = {}
    for key in ["path", "summ", "shuf", "comb", "pric", "volu"]:
        rng = np.random.default_rng(SEED)
        t0 = time.time()
        F = train_feat(key, np.load(OUT / "train_X.npy"), AUX, rng)
        m = make_model(N_ITER); m.fit(F, Y)
        # 训练段内样本内 IC(仅供参考)
        pr = m.predict(F)
        s = pd.DataFrame({"d": DT, "p": pr, "y": Y})
        ins = s.groupby("d").apply(lambda g: spearman(g["p"].to_numpy(), g["y"].to_numpy()),
                                   include_groups=False).to_numpy()
        models[key] = (m, float(np.nanmean(ins)))
        log(f"  fit {key} {time.time()-t0:.0f}s  in-sample IC={np.nanmean(ins):.4f}")
        del F, pr, s; gc.collect()

    specs = {k: (models[k][0], FEAT[k]) for k in models}
    log("streaming test 2024-2025 ...")
    rr = eval_stream(b, test_t, specs)
    results["test_2024_25"] = {k: report(rr, k, f"test_{k}",
                                         {"in_sample_train_ic": round(models[k][1], 4)},
                                         dates=dates)
                              for k in specs}
    for k, v in results["test_2024_25"].items():
        log("  TEST", k, v)

    # 条件版之前先把 test 上"全样本模型限定在强势股子集"的表现也算出来
    log("streaming test on strong subset with unconditional models ...")
    rr2 = eval_stream(b, test_t, specs, cond=True)
    results["test_strong_subset_uncond_model"] = {k: report(rr2, k, f"test_strong_uncond_{k}")
                                                  for k in specs}
    for k, v in results["test_strong_subset_uncond_model"].items():
        log("  TEST-STRONG(uncond model)", k, v)
    del models, specs, rr, rr2; gc.collect()

    # ---------- C. 条件版:只在强势股(mom120 > 当日中位数)里训练与检验 ----------
    log("conditional models ...")
    med = pd.Series(AUX[:, 4]).groupby(DT).transform("median").to_numpy()
    strong = AUX[:, 4] > med
    gs = pd.Series(Y[strong]).groupby(DT[strong])
    Ys = (gs.rank().to_numpy() / (gs.transform("size").to_numpy() + 1.0)).astype(np.float32)
    log("strong rows", int(strong.sum()))
    AUXs = AUX[strong][:, :4].copy()
    cspecs = {}
    for key in ["path", "summ", "shuf", "comb"]:
        rng = np.random.default_rng(SEED)
        F = train_feat(key, np.load(OUT / "train_X.npy")[strong], AUXs, rng)
        m = make_model(N_ITER); m.fit(F, Ys)
        cspecs[key] = (m, FEAT[key])
        log("  fit cond", key)
        del F; gc.collect()
    rr3 = eval_stream(b, test_t, cspecs, cond=True)
    results["test_conditional_strong"] = {k: report(rr3, k, f"test_cond_{k}", dates=dates)
                                          for k in cspecs}
    for k, v in results["test_conditional_strong"].items():
        log("  TEST-COND", k, v)

    RES.write_text(json.dumps(results, indent=1, default=float))
    log("written", RES)


if __name__ == "__main__":
    main()
