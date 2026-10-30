"""分钟数据 -> 日频原料面板(一次扫描,后续所有因子只读日频缓存)。

用法:
    python -m afactor.build_daily            # 全量 80 个月
    python -m afactor.build_daily 202001 202003

输出:cache/_tmp/{YYYYMM}.npz(每月一个),再由 merge_daily.py 合并为
     cache/daily/{feature}.parquet(2020-2025)与 cache/sealed/{feature}.parquet(2026)
"""
from __future__ import annotations
import sys, time, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from afactor.config import (RAW, TMP, N_BARS, IDX_1000, IDX_1130, IDX_1301, IDX_1430,
                            IDX_1450, IDX_1457, IDX_CLOSE_AUCTION, OPEN_TWAP, TAIL_TWAP,
                            PRE_CUTOFF, load_raw_minute, raw_months)

warnings.filterwarnings("ignore")
EPS = 1e-12

# 日内 8 段(等长 30 bar,覆盖 idx1..240)
SEGMENTS = [(1, 30), (31, 60), (61, 90), (91, 120),
            (121, 150), (151, 180), (181, 210), (211, 240)]


# ---------------- 向量化工具(沿 axis=0,输入 (T, N)) ----------------
def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < EPS, np.nan, a / np.where(np.abs(b) < EPS, np.nan, b))


def _corr(x, y):
    """逐列相关系数,NaN 传播。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = x - x.mean(axis=0)
        ym = y - y.mean(axis=0)
        num = (xm * ym).sum(axis=0)
        den = np.sqrt((xm ** 2).sum(axis=0) * (ym ** 2).sum(axis=0))
        return _safe_div(num, den)


def _autocorr(x, lag):
    return _corr(x[lag:], x[:-lag])


def _argpos(x, mode="max"):
    """极值出现的相对时点 [0,1];全 NaN 列返回 NaN。"""
    allnan = np.isnan(x).all(axis=0)
    filled = np.where(np.isnan(x), -np.inf if mode == "max" else np.inf, x)
    idx = (np.argmax if mode == "max" else np.argmin)(filled, axis=0).astype(np.float64)
    idx /= max(x.shape[0] - 1, 1)
    idx[allnan] = np.nan
    return idx


def _gini(v):
    n = v.shape[0]
    s = np.sort(np.where(np.isnan(v), np.nan, v), axis=0)
    w = np.arange(1, n + 1)[:, None]
    tot = s.sum(axis=0)
    return _safe_div(2.0 * (w * s).sum(axis=0), n * tot) - (n + 1.0) / n


def _skew(x):
    m = x.mean(axis=0); sd = x.std(axis=0)
    return _safe_div(((x - m) ** 3).mean(axis=0), sd ** 3)


def _kurt(x):
    m = x.mean(axis=0); sd = x.std(axis=0)
    return _safe_div(((x - m) ** 4).mean(axis=0), sd ** 4) - 3.0


def _smart_money(r, v, vol, frac=0.2):
    """开源证券"聪明钱"因子:S=|r|/sqrt(vol),按 S 降序取累计成交额前 frac 的分钟,
    其 VWAP 与全日 VWAP 之比。>1 表示聪明钱在高位活跃。"""
    s = _safe_div(np.abs(r), np.sqrt(np.maximum(vol, EPS)))
    order = np.argsort(-np.nan_to_num(s, nan=-np.inf), axis=0)
    v_sorted = np.take_along_axis(v, order, axis=0)
    vol_sorted = np.take_along_axis(vol, order, axis=0)
    cum = np.nancumsum(v_sorted, axis=0)
    tot = np.nansum(v_sorted, axis=0)
    prev = cum - v_sorted
    mask = prev < frac * tot                      # 累计到 frac 之前的分钟
    vwap_smart = _safe_div(np.nansum(np.where(mask, v_sorted, 0.0), axis=0),
                           np.nansum(np.where(mask, vol_sorted, 0.0), axis=0))
    vwap_all = _safe_div(np.nansum(v, axis=0), np.nansum(vol, axis=0))
    out = _safe_div(vwap_smart, vwap_all)
    out[np.nansum(v, axis=0) <= EPS] = np.nan
    return out


def _twap(mid, v, lo, hi):
    """窗口 [lo,hi) 内有成交分钟的等时加权均价(TWAP)。"""
    m = mid[lo:hi]; vv = v[lo:hi]
    trad = vv > EPS
    num = np.where(trad, m, np.nan)
    cnt = trad.sum(axis=0)
    px = np.nanmean(num, axis=0)
    px = np.where(cnt == 0, np.nanmean(m, axis=0), px)   # 全无成交则退化为价格均值(实盘=不可成交,由参与率约束拦截)
    return px


# ---------------- 单日特征 ----------------
def _day_features(P, O, V, full=True):
    """P/O/V 形如 (T, N)。full=True 计算全日全量特征;False 计算 14:50 截断的精简集。"""
    out = {}
    T = P.shape[0]
    mid = (O + P) / 2.0
    vol = _safe_div(V, mid)                                # 近似股数
    r = _safe_div(P[1:], P[:-1]) - 1.0                     # (T-1, N) 分钟收益
    r = np.clip(r, -0.5, 0.5)                              # 防除权残留毛刺
    v1 = V[1:]                                             # 与 r 对齐的成交额
    absr = np.abs(r)
    tot_amt = np.nansum(V, axis=0)
    tot_vol = np.nansum(vol, axis=0)
    vwap = _safe_div(tot_amt, tot_vol)
    px_first, px_last = P[0], P[-1]

    p = "" if full else "pre_"

    # --- 收益结构 ---
    out[p + "ret_intraday"] = _safe_div(px_last, px_first) - 1.0
    out[p + "rv"] = np.nansum(r ** 2, axis=0)
    out[p + "rvol"] = np.nanstd(r, axis=0)
    up = r > 0; dn = r < 0
    out[p + "rv_up"] = np.nansum(np.where(up, r ** 2, 0.0), axis=0)
    out[p + "rv_down"] = np.nansum(np.where(dn, r ** 2, 0.0), axis=0)
    out[p + "rsj"] = _safe_div(out[p + "rv_up"] - out[p + "rv_down"], out[p + "rv"])  # 已实现符号跳跃
    out[p + "rskew"] = np.sqrt(r.shape[0]) * _safe_div(np.nansum(r ** 3, axis=0), out[p + "rv"] ** 1.5)
    out[p + "rkurt"] = _kurt(r)
    out[p + "max_abs_r"] = np.nanmax(absr, axis=0)
    out[p + "path_len"] = np.nansum(absr, axis=0)
    out[p + "trend_eff"] = _safe_div(np.abs(_safe_div(px_last, px_first) - 1.0), out[p + "path_len"])
    bpv = (np.pi / 2.0) * np.nansum(absr[1:] * absr[:-1], axis=0)
    out[p + "bpv"] = bpv
    out[p + "jump"] = np.maximum(out[p + "rv"] - bpv, 0.0)
    out[p + "rng"] = _safe_div(np.nanmax(P, axis=0) - np.nanmin(P, axis=0), np.nanmean(P, axis=0))

    # --- 量价关系 ---
    out[p + "corr_r_v"] = _corr(r, v1)
    out[p + "corr_absr_v"] = _corr(absr, v1)
    out[p + "up_amt_share"] = _safe_div(np.nansum(np.where(up, v1, 0.0), axis=0), np.nansum(v1, axis=0))
    out[p + "up_min_ratio"] = np.where(np.isnan(r).all(axis=0), np.nan, up.sum(axis=0) / max(r.shape[0], 1))
    out[p + "vwap_dev"] = _safe_div(px_last, vwap) - 1.0
    out[p + "smart_money"] = _smart_money(r, v1, _safe_div(v1, mid[1:]))
    out[p + "amihud"] = _safe_div(np.nansum(absr, axis=0), tot_amt) * 1e9
    out[p + "amt_hhi"] = _safe_div(np.nansum(V ** 2, axis=0), tot_amt ** 2)
    out[p + "amt_gini"] = _gini(V)
    out[p + "amt_cv"] = _safe_div(np.nanstd(V, axis=0), np.nanmean(V, axis=0))
    out[p + "amt_skew"] = _skew(V)
    srt = np.sort(np.nan_to_num(V, nan=0.0), axis=0)
    out[p + "amt_top5_share"] = _safe_div(srt[-5:].sum(axis=0), tot_amt)
    out[p + "zero_min_ratio"] = np.where(np.isnan(V).all(axis=0), np.nan, (V <= EPS).sum(axis=0) / T)

    # --- 时序结构 ---
    out[p + "r_ac1"] = _autocorr(r, 1)
    out[p + "v_ac1"] = _autocorr(v1, 1)
    tgrid = (np.arange(T) / max(T - 1, 1))[:, None]
    out[p + "amt_time_wgt"] = _safe_div(np.nansum(V * tgrid, axis=0), tot_amt)
    out[p + "t_max_px"] = _argpos(P, "max")
    out[p + "t_min_px"] = _argpos(P, "min")
    out[p + "t_max_amt"] = _argpos(V, "max")

    # --- 大波动分钟的成交结构 ---
    k = max(int(round(0.1 * r.shape[0])), 1)
    ordr = np.argsort(-np.nan_to_num(absr, nan=-np.inf), axis=0)[:k]
    out[p + "bigmove_amt_share"] = _safe_div(np.take_along_axis(v1, ordr, axis=0).sum(axis=0),
                                             np.nansum(v1, axis=0))

    if not full:
        return out

    # ============ 仅全日口径的字段 ============
    out["open_auction_px"] = P[0]
    out["close_px"] = P[IDX_CLOSE_AUCTION]
    out["amount"] = tot_amt
    out["vwap"] = vwap
    out["twap_open10"] = _twap(mid, V, *OPEN_TWAP)
    out["amt_open10"] = np.nansum(V[OPEN_TWAP[0]:OPEN_TWAP[1]], axis=0)
    out["twap_tail8"] = _twap(mid, V, *TAIL_TWAP)
    out["amt_tail8"] = np.nansum(V[TAIL_TWAP[0]:TAIL_TWAP[1]], axis=0)
    out["amt_close_auction"] = V[IDX_CLOSE_AUCTION]
    out["amt_open_auction"] = V[0]
    out["n_trade_min"] = np.where(np.isnan(V).all(axis=0), np.nan, (V > EPS).sum(axis=0))

    # 关键时点收益
    out["ret_open30"] = _safe_div(P[IDX_1000], P[0]) - 1.0
    out["ret_morning"] = _safe_div(P[IDX_1130], P[0]) - 1.0
    out["ret_afternoon"] = _safe_div(P[IDX_CLOSE_AUCTION], P[IDX_1301]) - 1.0
    out["ret_last30"] = _safe_div(P[IDX_CLOSE_AUCTION], P[IDX_1430]) - 1.0
    out["ret_last10"] = _safe_div(P[IDX_CLOSE_AUCTION], P[IDX_1450]) - 1.0
    out["ret_closeauc"] = _safe_div(P[IDX_CLOSE_AUCTION], P[IDX_1457]) - 1.0
    out["ret_mid"] = _safe_div(P[IDX_1430], P[IDX_1000]) - 1.0

    # 时段成交额占比
    out["amt_first30_share"] = _safe_div(np.nansum(V[1:IDX_1000 + 1], axis=0), tot_amt)
    out["amt_last30_share"] = _safe_div(np.nansum(V[IDX_1430 + 1:], axis=0), tot_amt)
    out["amt_last10_share"] = _safe_div(np.nansum(V[IDX_1450 + 1:], axis=0), tot_amt)
    out["amt_morning_share"] = _safe_div(np.nansum(V[1:IDX_1130 + 1], axis=0), tot_amt)
    out["amt_openauc_share"] = _safe_div(V[0], tot_amt)
    out["amt_closeauc_share"] = _safe_div(V[IDX_CLOSE_AUCTION], tot_amt)

    # 8 段收益与成交占比
    for i, (lo, hi) in enumerate(SEGMENTS):
        out[f"seg{i}_ret"] = _safe_div(P[hi], P[lo - 1]) - 1.0
        out[f"seg{i}_amt"] = _safe_div(np.nansum(V[lo:hi + 1], axis=0), tot_amt)

    # 分钟形态:跳空 vs 实体
    gap = _safe_div(O[1:], P[:-1]) - 1.0
    body = _safe_div(P, O) - 1.0
    out["gap_share"] = _safe_div(np.nansum(np.abs(gap), axis=0),
                                 np.nansum(np.abs(gap), axis=0) + np.nansum(np.abs(body[1:]), axis=0))
    out["body_ratio"] = _safe_div(np.nansum(np.abs(body), axis=0), np.nansum(absr, axis=0))
    out["corr_body_r"] = _corr(body[1:], r)
    return out


# ---------------- 月度处理 ----------------
def process_month(month: str) -> str:
    t0 = time.time()
    C = load_raw_minute("close_adj", month)
    O = load_raw_minute("open_adj", month)
    V = load_raw_minute("amt", month)
    codes = C.columns
    assert list(O.columns) == list(codes) and list(V.columns) == list(codes), f"{month} 列不一致"
    assert O.index.equals(C.index) and V.index.equals(C.index), f"{month} 索引不一致"

    days = C.index.normalize().values
    Ca, Oa, Va = C.to_numpy(np.float64), O.to_numpy(np.float64), V.to_numpy(np.float64)
    uniq, starts, counts = np.unique(days, return_index=True, return_counts=True)
    order = np.argsort(starts)
    uniq, starts, counts = uniq[order], starts[order], counts[order]

    feats, bad_days, good = {}, [], []
    for d, s, n in zip(uniq, starts, counts):
        if n != N_BARS:
            bad_days.append((str(pd.Timestamp(d).date()), int(n)))
            continue
        sl = slice(s, s + n)
        P_, O_, V_ = Ca[sl], Oa[sl], Va[sl]
        valid = ~np.isnan(P_).all(axis=0)          # 该股当日是否有数据(未上市/停牌为全 NaN)
        f = _day_features(P_, O_, V_, full=True)
        f.update(_day_features(P_[:PRE_CUTOFF], O_[:PRE_CUTOFF], V_[:PRE_CUTOFF], full=False))
        for k, val in f.items():
            feats.setdefault(k, []).append(np.where(valid, val, np.nan))
        good.append(pd.Timestamp(d))
    arrs = {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}
    np.savez_compressed(TMP / f"{month}.npz",
                        dates=np.array([str(d.date()) for d in good]),
                        codes=np.array(list(codes), dtype=object),
                        **arrs)
    return f"{month}: {len(good)}日 x {len(codes)}股 x {len(arrs)}特征, {time.time()-t0:.0f}s" + \
           (f"  异常日={bad_days}" if bad_days else "")


def main():
    months = raw_months("close_adj")
    if len(sys.argv) > 2:
        lo, hi = sys.argv[1], sys.argv[2]
        months = [m for m in months if lo <= m <= hi]
    todo = [m for m in months if not (TMP / f"{m}.npz").exists()]
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
