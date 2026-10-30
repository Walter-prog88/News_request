"""T5 组合层与统计函数(前缀 inc_t5;预注册 INC_T5_PREREG.md §5、§9、§10)。纯函数,可单测。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TD = 242


# ================================================================ 权重层合成
def combine_weights(Ws: list[np.ndarray], a) -> np.ndarray:
    """W_comb[t] = sum_k a_k[t] * W_k[t];某篮子该日无持仓(W_k[t] 行和 = 0)时该篮子权重置 0、其余归一。
    a:(K,) 常数或 (T,K) 逐日。全部篮子都空的日子 W_comb[t] = 0。返回 float32 (T,N)。"""
    K = len(Ws)
    T = Ws[0].shape[0]
    A = np.broadcast_to(np.asarray(a, np.float64), (T, K)).copy() if np.ndim(a) == 1 else np.asarray(a, np.float64).copy()
    has = np.stack([(W.sum(axis=1) > 1e-12) for W in Ws], axis=1)
    A = np.where(has, A, 0.0)
    s = A.sum(axis=1, keepdims=True)
    A = np.divide(A, s, out=np.zeros_like(A), where=s > 0)
    out = np.zeros(Ws[0].shape, np.float32)
    for k, W in enumerate(Ws):
        out += (A[:, k:k + 1] * W).astype(np.float32)
    return out


def core_weights(K: int, core: int = 0, core_w: float = 0.5) -> np.ndarray:
    """核心-卫星:core 篮子占 core_w,其余 K-1 个平分 1-core_w。"""
    a = np.full(K, (1.0 - core_w) / (K - 1))
    a[core] = core_w
    return a


def erc_weights(excess: pd.DataFrame, window: int = 250, lag: int = 2, min_days: int | None = None) -> np.ndarray:
    """逆 TE 等风险权重(预注册 §5):a_k ∝ 1/TE_k,TE_k = 篮子 k 日超额过去 window 个信号日的标准差(s <= t - lag),
    截到 [0.5/K, 2/K] 后归一;不足 window 日时等权。excess:(T,K) DataFrame,按行位置对齐面板行。返回 (T,K)。"""
    x = excess.to_numpy(np.float64)
    T, K = x.shape
    min_days = window if min_days is None else min_days
    sd = excess.rolling(window, min_periods=min_days).std().to_numpy()
    sd = np.vstack([np.full((lag, K), np.nan), sd[:-lag]]) if lag > 0 else sd      # 第 t 行只用 s <= t-lag
    out = np.full((T, K), 1.0 / K)
    ok = np.isfinite(sd).all(axis=1) & (sd > 0).all(axis=1)
    inv = np.where(ok[:, None], 1.0 / np.where(sd > 0, sd, np.nan), np.nan)
    raw = inv / np.nansum(inv, axis=1, keepdims=True)
    clipped = np.clip(raw, 0.5 / K, 2.0 / K)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    out[ok] = clipped[ok]
    return out


# ================================================================ 生成检验
def hac_regress(y: np.ndarray, x: np.ndarray, q: int = 5) -> dict:
    """y = a + b x + u;a、b 的 Newey-West(Bartlett,maxlags = q)t 值与 R²。"""
    import statsmodels.api as sm
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = len(y)
    if n < 30:
        return dict(n=n, a=np.nan, t_a=np.nan, b=np.nan, t_b=np.nan, r2=np.nan)
    X = sm.add_constant(x)
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": q})
    return dict(n=n, a=float(r.params[0]), t_a=float(r.tvalues[0]), b=float(r.params[1]), t_b=float(r.tvalues[1]),
                r2=float(r.rsquared))


def nonoverlap_sum(x: pd.Series, k: int = 20) -> pd.Series:
    """k 日不重叠累计(算术和),标签取块末日;末尾不满一块的丢弃。"""
    n = (len(x) // k) * k
    v = x.iloc[:n].to_numpy().reshape(-1, k).sum(axis=1)
    return pd.Series(v, index=x.index[k - 1:n:k])


def block_bootstrap_dir(e_new: np.ndarray, e_old: np.ndarray, block: int = 20, n_boot: int = 2000, seed: int = 0
                        ) -> dict:
    """ΔIR = IR(new) - IR(old) 的移动块自助(块长 block 日,成对重采样,IR = 均值 x TD / (std x sqrt(TD)))。"""
    rng = np.random.default_rng(seed)
    m = np.isfinite(e_new) & np.isfinite(e_old)
    a, b = e_new[m], e_old[m]
    n = len(a)

    def ir(x):
        s = x.std(ddof=1)
        return x.mean() * TD / (s * np.sqrt(TD)) if s > 0 else np.nan
    base = ir(a) - ir(b)
    nb = int(np.ceil(n / block))
    starts_max = n - block
    d = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, starts_max + 1, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        d[i] = ir(a[idx]) - ir(b[idx])
    return dict(dir=float(base), lo90=float(np.nanpercentile(d, 5)), hi90=float(np.nanpercentile(d, 95)),
                p_pos=float(np.nanmean(d > 0)))


# ================================================================ 相对水下期
def rel_underwater(rs: pd.Series, rb: pd.Series) -> dict:
    """相对净值 = cumprod(1+rs)/cumprod(1+rb) 的最长水下期(交易日)、水下占比、最大相对回撤、期末是否水下;
    以及"滚动 250 日算术超额为负"的最长连续天数。"""
    a, b = rs.align(rb, join="inner")
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    rel = ((1.0 + a).cumprod() / (1.0 + b).cumprod()).to_numpy()
    peak = np.maximum.accumulate(rel)
    below = rel < peak - 1e-12
    longest = cur = 0
    for v in below:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    rdd = float((rel / peak - 1.0).min()) if len(rel) else np.nan
    r250 = (a - b).rolling(250, min_periods=250).sum()
    neg = (r250 < 0).to_numpy()
    nl = c2 = 0
    for v in neg:
        c2 = c2 + 1 if v else 0
        nl = max(nl, c2)
    return dict(rel_uw_longest_days=int(longest), rel_uw_frac=float(below.mean()) if len(below) else np.nan,
                rel_mdd=rdd, rel_uw_at_end=bool(below[-1]) if len(below) else False, roll250_neg_longest=int(nl))


def te_ir(e: pd.Series) -> dict:
    e = e.dropna()
    te = float(e.std() * np.sqrt(TD))
    ex = float(e.mean() * TD)
    return dict(excess_arith=ex, te=te, ir=ex / te if te > 0 else np.nan)


def jaccard_holdings(Wa: np.ndarray, Wb: np.ndarray, rows: np.ndarray) -> float:
    """月末持仓重合:并集/交集的 Jaccard 逐行平均。"""
    out = []
    for i in rows:
        a, b = Wa[i] > 0, Wb[i] > 0
        u = (a | b).sum()
        if u:
            out.append((a & b).sum() / u)
    return float(np.mean(out)) if out else np.nan


def verdict(p: dict, gates_ok: bool, c_downgrade: bool | None = None) -> dict:
    """预注册 §10.2 + §10.3 的判定规则(含偏离登记 verdict_p1_subwindow)。
    p:dict P1..P7 -> bool;另需 dIR_full、a_full、P5。"""
    all_pass = (all(p[k] for k in ("P1", "P2", "P3", "P4", "P5", "P6", "P7")) and gates_ok
                and p["dIR_full"] > 0 and p["a_full"] > 0)
    neg = (not (p["dIR_full"] > 0)) or (not (p["a_full"] > 0)) or (not p["P5"])
    if all_pass:
        v = "有增量"
    elif neg:
        v = "无增量"
    else:
        v = "证据不足"
    note = ""
    if v == "有增量" and c_downgrade:
        v = "证据不足"
        note = "C 轨指示性证据不支持,降级(§10.3)"
    return dict(verdict=v, note=note)
