"""实验一:个股 × 市场分钟互动特征(第二次分钟扫描)。

自建全市场分钟等权指数(当日有效股票的分钟收益截面均值),提取:
半贝塔四象限(Bollerslev-Patton-Quaedvlieg 2022)、日内 beta/相关、领先-滞后、
共同极端波动、特质波动占比、相对市场的时段超额收益。
全日版 + pre_ 截断版(09:30-14:49,供尾盘口径),输出格式与 build_daily 一致。

    python -m afactor.build_market
"""
from __future__ import annotations
import sys, time, warnings, glob, os, re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from afactor.config import RAW, CACHE, N_BARS, PRE_CUTOFF, load_raw_minute, raw_months

warnings.filterwarnings("ignore")
EPS = 1e-12
TMP_MKT = CACHE / "_tmp_mkt"
TMP_MKT.mkdir(parents=True, exist_ok=True)


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < EPS, np.nan, a / np.where(np.abs(b) < EPS, np.nan, b))


def _corr_vec(x, m):
    """各列 x 与向量 m 的相关。x (T,N), m (T,)"""
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = x - np.nanmean(x, axis=0)
        mm = m - np.nanmean(m)
        num = np.nansum(xm * mm[:, None], axis=0)
        den = np.sqrt(np.nansum(xm ** 2, axis=0) * np.nansum(mm ** 2))
        return _safe_div(num, den)


def _mkt_features(P, prefix=""):
    """P: (T,N) 分钟收盘价。返回 {name: (N,)}。市场 = 当日有效列的分钟收益等权均值。"""
    out = {}
    r = _safe_div(P[1:], P[:-1]) - 1.0
    r = np.clip(r, -0.5, 0.5)
    valid = ~np.isnan(P).all(axis=0)
    with np.errstate(invalid="ignore"):
        mkt = np.nanmean(np.where(valid[None, :], r, np.nan), axis=1)   # (T-1,)
    if not np.isfinite(mkt).any():
        return {}
    mkt2 = np.nansum(mkt ** 2)

    p_i = np.clip(r, 0, None); n_i = np.clip(r, None, 0)
    Pm = np.clip(mkt, 0, None); Nm = np.clip(mkt, None, 0)
    out[prefix + "mkt_beta"] = _safe_div(np.nansum(r * mkt[:, None], axis=0), mkt2)
    out[prefix + "mkt_corr"] = _corr_vec(r, mkt)
    out[prefix + "semibeta_n"] = _safe_div(np.nansum(n_i * Nm[:, None], axis=0), mkt2)
    out[prefix + "semibeta_p"] = _safe_div(np.nansum(p_i * Pm[:, None], axis=0), mkt2)
    out[prefix + "semibeta_mn"] = _safe_div(-np.nansum(p_i * Nm[:, None], axis=0), mkt2)
    out[prefix + "semibeta_mp"] = _safe_div(-np.nansum(n_i * Pm[:, None], axis=0), mkt2)
    out[prefix + "mkt_lead"] = _corr_vec(r[1:], mkt[:-1])       # 个股跟随市场
    out[prefix + "mkt_lag"] = _corr_vec(r[:-1], mkt[1:])        # 个股领先市场

    beta = out[prefix + "mkt_beta"]
    idio = r - beta[None, :] * mkt[:, None]
    var_tot = np.nanvar(r, axis=0); var_idio = np.nanvar(idio, axis=0)
    out[prefix + "idio_share"] = _safe_div(var_idio, var_tot)
    out[prefix + "idio_vol"] = np.sqrt(np.maximum(var_idio, 0.0))

    # 共同极端:个股 |r| 前10% 分钟中,市场也处于 |mkt| 前10% 的比例
    absr = np.abs(r); absm = np.abs(mkt)
    thr_i = np.nanquantile(absr, 0.9, axis=0)
    thr_m = np.nanquantile(absm, 0.9)
    big_i = absr >= thr_i[None, :]
    big_m = (absm >= thr_m)[:, None]
    cnt_i = big_i.sum(axis=0)
    out[prefix + "cojump"] = _safe_div((big_i & big_m).sum(axis=0).astype(float), cnt_i.astype(float))

    # 相对市场的时段超额(收益和近似)
    cum_i = np.nancumsum(np.where(np.isfinite(r), r, 0.0), axis=0)
    cum_m = np.nancumsum(np.where(np.isfinite(mkt), mkt, 0.0))
    T1 = r.shape[0]
    seg = {"open30": (0, min(30, T1)), "am": (0, min(120, T1)),
           "pm": (min(120, T1), T1), "last30": (max(T1 - 30, 0), T1)}
    for k, (a, b) in seg.items():
        if b <= a:
            continue
        ex = (cum_i[b - 1] - (cum_i[a - 1] if a > 0 else 0.0)) - \
             (cum_m[b - 1] - (cum_m[a - 1] if a > 0 else 0.0))
        out[prefix + f"ex_ret_{k}"] = ex
    out[prefix + "ex_ret_day"] = cum_i[-1] - cum_m[-1]
    for k in out:
        out[k] = np.where(valid, out[k], np.nan)
    return out


def process_month(month: str) -> str:
    t0 = time.time()
    C = load_raw_minute("close_adj", month)
    codes = C.columns
    days = C.index.normalize().values
    Ca = C.to_numpy(np.float64)
    uniq, starts, counts = np.unique(days, return_index=True, return_counts=True)
    order = np.argsort(starts)
    uniq, starts, counts = uniq[order], starts[order], counts[order]

    feats, good = {}, []
    for d, s, n in zip(uniq, starts, counts):
        if n != N_BARS:
            continue
        P_ = Ca[s:s + n]
        f = _mkt_features(P_)
        f.update(_mkt_features(P_[:PRE_CUTOFF], prefix="pre_"))
        if not f:
            continue
        for k, v in f.items():
            feats.setdefault(k, []).append(v)
        good.append(pd.Timestamp(d))

    arrs = {k: np.asarray(v, np.float32) for k, v in feats.items()}
    np.savez_compressed(TMP_MKT / f"{month}.npz",
                        dates=np.array([str(d.date()) for d in good]),
                        codes=np.array(list(codes), dtype=object), **arrs)
    return f"{month}: {len(good)}日 x {len(codes)}股 x {len(arrs)}特征, {time.time()-t0:.0f}s"


def main():
    months = raw_months("close_adj")
    todo = [m for m in months if not (TMP_MKT / f"{m}.npz").exists()]
    print(f"待处理 {len(todo)}/{len(months)} 个月", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(process_month, m): m for m in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(todo)}] {fu.result()}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {futs[fu]} FAILED: {type(e).__name__}: {e}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
