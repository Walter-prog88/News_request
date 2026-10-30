"""Barra 归因分辨率阶梯 —— B×GBT 打分面板从 Top20 放宽到 top10%(n=20/100/200/400 + 缓冲带 10%/15%)。

## 目的

`research/BARRA_TOP20.md` §3.1:Top20 周频的 NW se(alpha) ≈ 4.6pp/年,|alpha| ≥ 9pp 才检得出;
而排雷缓冲 20/35(持仓 ~2900 只)的 se ≈ 0.7pp。"好股票那一头有没有信息"从未被正确地测过,
因为尺子太粗。本脚本把持仓面板逐级放宽,报每档的 **alpha 及其 SE(分辨率阶梯)**,
看在多宽的面板上能把 alpha 从流动性暴露里分辨出来。

## 方法

1. 阶梯:scheme="top", n ∈ {20, 100, 200, 400}(≈ V1000 池的 0.5% / 2.5% / 5% / 10%),
   全部 rebalance=5、avg_phases=True(5 相位平均)、OOS 2023–25 逐年掩码后拼接——
   与 run_ml_models.py / run_barra_attribution_top20.py 同一调用。n=20 必须复现 BARRA_TOP20
   (逐年 ann/bench/excess 对账本 ±1pp)。
2. 缓冲带版 scheme="top_buffer":ST.run_strategy 的 top 方案没接缓冲带,故把
   `src/afactor/strategy.py::_run_one_phase` **复制**到本脚本(不改原文件),加一个分支,
   权重函数 `weights_top_buffer(S, U, enter=0.10, exit=0.15)` 镜像 `backtest.weights_exclude_buffer`:
   分位进入最好 10% 才买、跌出最好 15% 才卖、等权。
   **先用既有方案(top n=20 / n=400)对照 ST._run_one_phase 逐相位逐日 max|Δ|<1e-12**,再跑缓冲版。
3. 每档同一套 Barra 归因(照 run_barra_attribution_top20.py::regress):
   excess_t = alpha + Σ β_k f_kt + ε_t,X = barra_factor_returns.csv 的 open10 口径 10 风格 + 行业,
   alpha 年化 ×242,t = nw_t(resid+alpha, q)(q=0/5/10/20),**SE = alpha/t**(核心产出),
   交叉核对 nw_se_ols 的截距 SE;R²、Liquidity β 与贡献、十风格贡献、逐年 alpha、同窗排雷对照。
4. 最宽两档(n=400、缓冲版)额外:打分 vs 十个 Barra 暴露的逐日截面 Spearman(打分面板与档位无关,算一次);
   持仓法暴露(复制引擎逐日记录持仓 h,5 相位平均,主动暴露 = Σ h_i X_i − 池等权暴露,
   贡献 = mean_t(主动暴露_t × 纯因子_t) × 242,行业含约束隐含的最后一个行业)。

只读研究期面板 Panel(),不触碰封存样本。不修改仓库既有文件。

## 产出
    research/ledger/barra_topfrac_summary.csv  长表:n_or_spec, term, beta, factor_ann, contrib_ann, t, se, note
    research/ledger/barra_topfrac_daily.csv    每档 strategy / excess / fitted / resid + bench + 排雷对照
    research/r_barra_topfrac.log               由调用方 tee
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from afactor.config import (LEDGER, COST_BPS_ONEWAY, STAMP_TAX_BPS_BEFORE, STAMP_TAX_BPS_AFTER,
                            STAMP_TAX_CHANGE_DATE, MAX_PARTICIPATION)
from afactor.panel import Panel
from afactor import factors as FA, strategy as ST, newdata as ND
from afactor.strategy import TRADING_DAYS, VERSIONS, EXEC_MODES, topn_weights, tilt_weights, exec_prices
from run_barra_attribution import nw_t, assert_aligned, build_exposures
from run_attribution import nw_se_ols

pd.set_option("display.width", 240)

YEARS = [2023, 2024, 2025]
VERSION, EXEC = "V1000", "open10"
REBAL = 5
SCORE_DIR = LEDGER / "scores"
TOL = 0.01
QS = (0, 5, 10, 20)
LIQ = "Liquidity 流动性"

# 阶梯:名称 -> (scheme, n, buffer)
LADDER = [
    ("top20",            "top",        20,  None),
    ("top100",           "top",        100, None),
    ("top200",           "top",        200, None),
    ("top400",           "top",        400, None),
    ("top_buffer_10_15", "top_buffer", 0,   (0.10, 0.15)),
]
WIDE = ["top400", "top_buffer_10_15"]        # 额外报持仓法暴露的两档(top20 也顺带算,用于对 §2.4 交叉核对)


# ---------------------------------------------------------------- 权重:选优缓冲带(镜像 backtest.weights_exclude_buffer)
def weights_top_buffer(F: np.ndarray, M: np.ndarray, sign: int = 1,
                       enter: float = 0.10, exit: float = 0.15) -> np.ndarray:
    """选优 + 缓冲带:从高往低的分位进入最好 enter 才买入,跌出最好 exit 才卖出,持有者等权。
    不对称阈值抑制在门槛附近来回进出造成的换手。与 weights_exclude_buffer 完全对称:
    那边 q=rankdata(F)/n 低分位=最差、q<=enter 进雷区、q>exit 出雷区;
    这边 q_top=rankdata(-F)/n 低分位=最好、q_top<=enter 入选、q_top>exit 剔出。"""
    from scipy.stats import rankdata
    T, N = F.shape
    W = np.zeros((T, N), np.float32)
    held = np.zeros(N, bool)
    for t in range(T):
        m = M[t] & np.isfinite(F[t])
        n = int(m.sum())
        if n < 50:
            W[t] = W[t - 1] if t else 0
            continue
        idx = np.flatnonzero(m)
        q_top = rankdata(-F[t, idx] * sign) / n     # 低分位 = 最好
        new = np.zeros(N, bool)
        new[idx[q_top <= enter]] = True             # 新进入最好 enter
        stay = held.copy()
        stay[idx[q_top > exit]] = False             # 跌出最好 exit
        stay[~m] = held[~m]                         # 当日无数据者维持原状态
        held = new | stay
        sel = idx[held[idx]]
        if sel.size:
            W[t, sel] = 1.0 / sel.size
    return W


# ---------------------------------------------------------------- 引擎复制(逐字复制 ST._run_one_phase,加 top_buffer 分支与持仓记录)
def run_one_phase_local(score, p, version, mode, scheme, n, rebalance, exclude_frac,
                        buffer, date_mask, sign, phase: int, universe_override=None,
                        record_holdings: bool = False) -> dict:
    cfg = VERSIONS[version]
    em = EXEC_MODES[mode]
    dates, codes = p.dates, p.codes
    U = (universe_override.reindex(index=dates, columns=codes).fillna(False) if universe_override is not None
         else ST.universe(p, version)).to_numpy(bool)
    if date_mask is not None:
        U = U & date_mask[:, None]
    S = (score.reindex(index=dates, columns=codes).to_numpy(np.float32) * sign)

    if scheme == "top":
        W = topn_weights(S, U, n)
    elif scheme == "top_buffer":                                    # ← 新增分支
        W = weights_top_buffer(S, U, 1, *buffer)
    elif scheme == "tilt":
        W = tilt_weights(S, U, power=float(n) if n else 1.0)
    elif scheme == "vw":
        W = np.where(U & np.isfinite(S), np.clip(S, 0, None), 0.0).astype(np.float32)
        rs = W.sum(axis=1, keepdims=True)
        W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    else:
        from afactor.backtest import weights_exclude_buffer, target_weights
        W = (weights_exclude_buffer(S, U, 1, *buffer) if buffer
             else target_weights(S, U, top_n=None, sign=1, scheme="exclude", top_frac=exclude_frac))

    px_df, amt_df = exec_prices(p, mode)
    px = px_df.to_numpy(np.float64)
    win_amt = amt_df.to_numpy(np.float64)
    dead = (p["one_word_up"] | p["one_word_down"]).to_numpy(bool)
    lag = em["lag"]

    T, N = W.shape
    h = np.zeros(N)
    rets, ridx, turns, costs, held, exps = [], [], [], [], [], []
    H = np.zeros((T, N), np.float32) if record_holdings else None   # ← 新增:成交后持仓,按信号日 t 索引
    stamp_switch = np.searchsorted(dates.values, np.datetime64(STAMP_TAX_CHANGE_DATE))
    t0 = int(np.flatnonzero(date_mask)[0]) if date_mask is not None and date_mask.any() else 0
    t0 += int(phase)
    for t in range(T - lag - 1):
        te = t + lag
        if te >= T - 1:
            break
        p_now, p_next = px[te], px[te + 1]
        ok_px = np.isfinite(p_now) & np.isfinite(p_next) & (p_now > 0)
        can = ok_px & ~dead[te] & (win_amt[te] > 0)
        if rebalance > 1 and ((t - t0) % rebalance):
            delta = np.zeros(N)
        else:
            stuck = (~can) & (h > 1e-12)
            budget = max(0.0, 1.0 - float(h[stuck].sum()))
            tgt = np.where(can, W[t].astype(np.float64), 0.0)
            s = tgt.sum()
            if s > 0:
                tgt = tgt / s * budget
            tgt = tgt + np.where(stuck, h, 0.0)
            delta = tgt - h
            cap_w = np.where(np.isfinite(win_amt[te]),
                             MAX_PARTICIPATION * win_amt[te] / cfg["capital"], 0.0)
            delta = np.clip(delta, -cap_w, cap_w)
            delta = np.where(can, delta, 0.0)
            over = float(np.clip(h + delta, 0, None).sum()) - 1.0
            if over > 1e-12:
                buys = np.clip(delta, 0, None); bs = buys.sum()
                if bs > 0:
                    delta = delta - buys * min(1.0, over / bs)
        newh = np.clip(h + delta, 0, None)
        buy = np.clip(delta, 0, None).sum(); sell = np.clip(-delta, 0, None).sum()
        stamp = (STAMP_TAX_BPS_BEFORE if te < stamp_switch else STAMP_TAX_BPS_AFTER) * 1e-4
        cost = (buy + sell) * COST_BPS_ONEWAY * 1e-4 + sell * stamp
        with np.errstate(invalid="ignore"):
            r = np.where(ok_px, p_next / p_now - 1.0, 0.0)
        port_r = float(np.nansum(newh * r)) - cost
        grown = newh * (1.0 + np.nan_to_num(r))
        h = grown / (1.0 + port_r) if (1.0 + port_r) > 1e-9 else grown
        hs = h.sum()
        if hs > 1.0 + 1e-12:
            h = h / hs
        if H is not None:
            H[t] = newh
        rets.append(port_r); ridx.append(dates[t])
        turns.append(np.abs(delta).sum() / 2); costs.append(cost)
        held.append(int((newh > 1e-8).sum())); exps.append(float(newh.sum()))
        assert newh.sum() <= 1.0 + 1e-9, f"exposure>1: {newh.sum():.6f} at t={t}"

    r_s = pd.Series(rets, index=pd.DatetimeIndex(ridx))
    turns, costs = np.asarray(turns), np.asarray(costs)
    held, exps = np.asarray(held), np.asarray(exps)
    if date_mask is not None:
        keep = pd.Series(date_mask[:len(dates)], index=dates).reindex(
            r_s.index).fillna(False).to_numpy(bool)
        if keep.sum() < 10:
            return dict(nav=(1.0 + r_s[keep]).cumprod(), n_days=int(keep.sum()))
        r_s = r_s[keep]
        turns, costs, held, exps = turns[keep], costs[keep], held[keep], exps[keep]
    nav = (1.0 + r_s).cumprod()
    yrs = len(r_s) / TRADING_DAYS
    out = dict(nav=nav, rets=r_s, ann_return=float(nav.iloc[-1] ** (1 / yrs) - 1),
               sharpe=float(r_s.mean() / r_s.std() * np.sqrt(TRADING_DAYS)) if r_s.std() > 0 else np.nan,
               max_dd=float((nav / nav.cummax() - 1).min()),
               turnover=float(turns.mean()), cost_bps=float(costs.mean() * 1e4),
               holdings=float(held.mean()), exposure=float(exps.mean()),
               n_days=len(r_s), vol=float(r_s.std() * np.sqrt(TRADING_DAYS)))
    if H is not None:
        pos = p.dates.get_indexer(r_s.index)
        out["H"] = pd.DataFrame(H[pos], index=r_s.index, columns=codes)
    return out


def run_spec_year(sc, p, spec, te, record_holdings=False) -> dict:
    """一档一年:5 相位、相位平均(与 ST.run_strategy(avg_phases=True) 同一聚合)。"""
    name, scheme, n, buf = spec
    subs = [run_one_phase_local(sc, p, VERSION, EXEC, scheme, n, REBAL, 0.3, buf, te, 1, ph,
                                record_holdings=record_holdings) for ph in range(REBAL)]
    subs = [x for x in subs if x.get("rets") is not None and len(x["rets"]) > 5]
    R = pd.concat([x["rets"] for x in subs], axis=1)
    r_s = R.mean(axis=1)
    nav = (1.0 + r_s).cumprod()
    yrs = len(r_s) / TRADING_DAYS
    agg = lambda key: float(np.mean([x[key] for x in subs]))
    out = dict(rets=r_s, phases=[x["rets"] for x in subs],
               ann_return=float(nav.iloc[-1] ** (1 / yrs) - 1),
               sharpe=float(r_s.mean() / r_s.std() * np.sqrt(TRADING_DAYS)),
               max_dd=float((nav / nav.cummax() - 1).min()),
               turnover=agg("turnover"), cost_bps=agg("cost_bps"), holdings=agg("holdings"),
               exposure=agg("exposure"), phase_ann=[x["ann_return"] for x in subs])
    if record_holdings:
        Hs = [x["H"] for x in subs]
        out["H"] = sum(h.reindex(r_s.index).fillna(0.0) for h in Hs) / len(Hs)
    return out


# ---------------------------------------------------------------- 归因(照 run_barra_attribution_top20.regress)
def geo_ann(r: pd.Series) -> float:
    nav = (1.0 + r).cumprod()
    return float(nav.iloc[-1] ** (TRADING_DAYS / len(r)) - 1.0)


def regress(y: pd.Series, F: pd.DataFrame, styles: list, indus: list) -> dict:
    terms = styles + indus
    A = np.column_stack([np.ones(len(y)), F[terms].to_numpy(float)])
    yv = y.to_numpy(float)
    b, *_ = np.linalg.lstsq(A, yv, rcond=None)
    fitted = A @ b
    resid = yv - fitted
    r2 = 1.0 - resid.var() / yv.var()
    fann = {t: float(F[t].mean() * TRADING_DAYS) for t in terms}
    contrib = {t: float(b[i + 1] * fann[t]) for i, t in enumerate(terms)}
    alpha_series = resid + b[0]
    t_nw = {q: nw_t(alpha_series, q) for q in QS}
    se_nw = {q: float(b[0] * TRADING_DAYS / t_nw[q]) if np.isfinite(t_nw[q]) and t_nw[q] != 0 else np.nan
             for q in QS}
    se_ols_raw = {q: nw_se_ols(A, resid, q)[0] for q in QS}
    t_ols = {q: float(b[0] / se_ols_raw[q]) for q in QS}
    se_ols = {q: float(se_ols_raw[q] * TRADING_DAYS) for q in QS}
    # 直接口径的 SE(不经 alpha/t 反推,避免 alpha≈0 时数值不稳):nw_t 的分母
    x = alpha_series; e = x - x.mean(); nn = len(x)
    se_direct = {}
    for q in QS:
        s = float((e ** 2).sum() / nn)
        for L in range(1, q + 1):
            s += 2.0 * (1.0 - L / (q + 1.0)) * float((e[L:] * e[:-L]).sum() / nn)
        se_direct[q] = float(np.sqrt(s / nn) * TRADING_DAYS)
    return dict(b=b, terms=terms, alpha=float(b[0] * TRADING_DAYS), r2=float(r2),
                tot=float(yv.mean() * TRADING_DAYS),
                style_sum=float(sum(contrib[t] for t in styles)),
                ind_sum=float(sum(contrib[t] for t in indus)),
                contrib=contrib, fann=fann, t_nw=t_nw, t_ols=t_ols,
                se_nw=se_direct, se_ols=se_ols,
                resid_vol=float(resid.std() * np.sqrt(TRADING_DAYS)),
                fitted=pd.Series(fitted, index=y.index), resid=pd.Series(resid, index=y.index),
                n=len(y))


def emit(rows: list, spec: str, section: str, res: dict, styles: list, indus: list, note: str = ""):
    for i, t in enumerate(res["terms"]):
        rows.append(dict(n_or_spec=spec, term=f"{section}|{t}", beta=res["b"][i + 1], factor_ann=res["fann"][t],
                         contrib_ann=res["contrib"][t], t=np.nan, se=np.nan,
                         note="风格" if t in styles else "行业"))
    def R(term, **kw):
        d = dict(n_or_spec=spec, term=f"{section}|{term}", beta=np.nan, factor_ann=np.nan,
                 contrib_ann=np.nan, t=np.nan, se=np.nan, note="")
        d.update(kw); rows.append(d)
    R("__总超额", contrib_ann=res["tot"], note=f"算术年化, n={res['n']}")
    R("__风格合计", contrib_ann=res["style_sum"])
    R("__行业合计", contrib_ann=res["ind_sum"])
    R("__alpha", contrib_ann=res["alpha"], t=res["t_nw"][5], se=res["se_nw"][5],
      note="t/se 为 NW q=5, nw_t(resid+alpha)" + (f"; {note}" if note else ""))
    for q in QS:
        R(f"__alpha_t_q{q}", t=res["t_nw"][q], se=res["se_nw"][q], note="nw_t(resid+alpha); se 年化")
        R(f"__alpha_t_ols_q{q}", t=res["t_ols"][q], se=res["se_ols"][q], note="nw_se_ols 截距; se 年化")
    R("__R2", contrib_ann=res["r2"])
    R("__残差年化波动", contrib_ann=res["resid_vol"])
    R("__Liquidity占比", contrib_ann=res["contrib"][LIQ] / res["tot"] if res["tot"] != 0 else np.nan,
      note="Liquidity 贡献 / 总超额")
    R("__alpha_1.96se", contrib_ann=1.96 * res["se_nw"][5], note="NW q=5 可检出门槛(|alpha| 达此值 t=1.96)")


def print_res(tag: str, res: dict):
    liq = res["contrib"][LIQ]
    print(f"  [{tag}] n={res['n']} 总超额 {res['tot']*100:+.2f}pp | 风格 {res['style_sum']*100:+.2f} "
          f"| 行业 {res['ind_sum']*100:+.2f} | alpha {res['alpha']*100:+.2f}pp | SE(q5) {res['se_nw'][5]*100:.2f}pp "
          f"| R2 {res['r2']:.3f} | 残差vol {res['resid_vol']*100:.1f}% "
          f"| Liquidity β {res['b'][res['terms'].index(LIQ)+1]:+.3f} 贡献 {liq*100:+.2f}pp "
          f"({liq/res['tot']*100 if res['tot'] else np.nan:+.0f}%)")
    print("         alpha t(nw_t) " + " ".join(f"q{q}={res['t_nw'][q]:+.2f}" for q in QS)
          + " | t(nw_se_ols) " + " ".join(f"q{q}={res['t_ols'][q]:+.2f}" for q in QS)
          + " | SE(nw_t)pp " + " ".join(f"q{q}={res['se_nw'][q]*100:.2f}" for q in QS)
          + " | SE(ols)pp " + " ".join(f"q{q}={res['se_ols'][q]*100:.2f}" for q in QS))


# ---------------------------------------------------------------- 主流程
def main():
    T0 = time.time()
    p = Panel(); FA.build_all()
    years = np.asarray(p.dates.year)
    oos = np.isin(years, YEARS)

    # ---- 0. 池子规模 ----
    print("=" * 100 + "\n0. V1000 决策池每日只数(OOS 2023–25)\n" + "=" * 100)
    U1000 = ST.universe(p, VERSION)
    pool_n = U1000.sum(axis=1)[oos]
    print(f"  中位 {int(pool_n.median())} | 均值 {pool_n.mean():.0f} | 最小 {int(pool_n.min())} | 最大 {int(pool_n.max())}")
    for y in YEARS:
        s = U1000.sum(axis=1)[years == y]
        print(f"    {y}: 中位 {int(s.median())} 最小 {int(s.min())} 最大 {int(s.max())}")
    print("  阶梯 n 占池中位: " + ", ".join(f"n={n}→{n/pool_n.median()*100:.1f}%" for n in (20, 100, 200, 400)))

    scores = {y: pd.read_parquet(SCORE_DIR / f"B_gbt_open_h5_{y}.parquet") for y in YEARS}
    def load_score(y):
        sc = scores[y]
        full = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
        idx = sc.index.intersection(p.dates)
        full.loc[idx, :] = sc.reindex(index=idx, columns=p.codes).to_numpy(np.float32)
        return full
    SC = {y: load_score(y) for y in YEARS}

    # ---- 1. 引擎复制一致性 ----
    print("\n" + "=" * 100 + "\n1. 复制引擎 vs ST._run_one_phase 逐相位逐日一致性(top n=20 与 n=400, 2023–25)\n" + "=" * 100)
    engine_max = 0.0
    for n in (20, 400):
        for y in YEARS:
            te = years == y
            for ph in range(REBAL):
                a = ST._run_one_phase(SC[y], p, VERSION, EXEC, "top", n, REBAL, 0.3, None, te, 1, ph)
                b = run_one_phase_local(SC[y], p, VERSION, EXEC, "top", n, REBAL, 0.3, None, te, 1, ph)
                assert a["rets"].index.equals(b["rets"].index)
                d = float((a["rets"] - b["rets"]).abs().max())
                engine_max = max(engine_max, d)
                for k in ("turnover", "cost_bps", "holdings", "exposure"):
                    assert abs(a[k] - b[k]) < 1e-9, (n, y, ph, k)
        print(f"  n={n}: 15 条相位序列 max|Δ| 累计 {engine_max:.2e}")
    assert engine_max < 1e-12, f"复制引擎与 ST._run_one_phase 不一致: {engine_max:.2e}"
    print(f"  引擎复制一致 (max|Δ| = {engine_max:.2e} < 1e-12)。")
    # 缓冲权重函数自检:enter==exit 时应与 top_frac 等价(非严格,只报重叠率)
    te = years == 2023
    Ub = (U1000.to_numpy(bool) & te[:, None])
    Sb = SC[2023].to_numpy(np.float32)
    Wb = weights_top_buffer(Sb, Ub, 1, 0.10, 0.15)
    Wt = weights_top_buffer(Sb, Ub, 1, 0.10, 0.10)
    ti = np.flatnonzero(te)[100]
    nb, nt = int((Wb[ti] > 0).sum()), int((Wt[ti] > 0).sum())
    ov = int(((Wb[ti] > 0) & (Wt[ti] > 0)).sum())
    print(f"  缓冲权重自检 2023 第 100 个信号日: 10/15 持 {nb} 只, 10/10 持 {nt} 只(=池 {int(Ub[ti].sum())} 的 10%), "
          f"10/10 ⊂ 10/15: {ov == nt}")
    assert ov == nt and nb >= nt

    # ---- 2. 逐档重建 ----
    print("\n" + "=" * 100 + "\n2. 阶梯回测(5 相位平均、周频、open10、V1000、OOS 逐年掩码后拼接)\n" + "=" * 100)
    ledger = pd.read_csv(LEDGER / "ml_results_open_h5.csv")
    ledger = ledger[(ledger.feature_set == "B") & (ledger.model == "gbt") &
                    (ledger.version == VERSION) & (ledger.plan == "Top20周频")].set_index("oos_year")
    bench_parts = [ST.benchmark(p, VERSION, EXEC, date_mask=(years == y))["rets"] for y in YEARS]
    bench = pd.concat(bench_parts).sort_index()
    assert bench.index.is_unique

    runs, rows = {}, []
    top20_ok = True
    for spec in LADDER:
        name = spec[0]
        rec = (name in WIDE) or (name == "top20")
        parts, phase_parts, H_parts, yr_rows = [], {ph: [] for ph in range(REBAL)}, [], []
        stats = []
        for y in YEARS:
            te = years == y
            r = run_spec_year(SC[y], p, spec, te, record_holdings=rec)
            bm = ST.benchmark(p, VERSION, EXEC, date_mask=te)
            ex = r["ann_return"] - bm["ann_return"]
            line = (f"    {name:18s} {y}: ann {r['ann_return']:+.4f} bench {bm['ann_return']:+.4f} excess {ex:+.4f} "
                    f"sharpe {r['sharpe']:.2f} 换手/日 {r['turnover']:.4f} 成本 {r['cost_bps']:.2f}bp/日 "
                    f"({r['cost_bps']*TRADING_DAYS/100:.2f}pp/年) 持仓 {r['holdings']:.0f} 仓位 {r['exposure']:.3f} "
                    f"相位 ann {[round(a, 4) for a in r['phase_ann']]}")
            if name == "top20":
                L = ledger.loc[y]
                d = (abs(r["ann_return"] - L.ann), abs(bm["ann_return"] - L.bench), abs(ex - L.excess))
                ok = all(x <= TOL for x in d)
                top20_ok &= ok
                line += f" | 账本 ann {L.ann:+.4f} excess {L.excess:+.4f} Δ {d[0]*100:+.2f}/{d[2]*100:+.2f}pp {'OK' if ok else 'MISMATCH'}"
            print(line)
            stats.append(dict(year=y, ann=r["ann_return"], bench=bm["ann_return"], excess=ex, sharpe=r["sharpe"],
                              turnover=r["turnover"], cost_bps=r["cost_bps"], holdings=r["holdings"],
                              exposure=r["exposure"], max_dd=r["max_dd"]))
            parts.append(r["rets"])
            for ph in range(REBAL):
                phase_parts[ph].append(r["phases"][ph])
            if rec:
                H_parts.append(r["H"])
        strat = pd.concat(parts).sort_index()
        assert strat.index.is_unique
        st = pd.DataFrame(stats).set_index("year")
        nav = (1 + strat).cumprod(); bnav = (1 + bench.reindex(strat.index)).cumprod()
        full_ann = float(nav.iloc[-1] ** (TRADING_DAYS / len(strat)) - 1)
        full_b = float(bnav.iloc[-1] ** (TRADING_DAYS / len(strat)) - 1)
        full_sh = float(strat.mean() / strat.std() * np.sqrt(TRADING_DAYS))
        ex_arith = float((strat - bench.reindex(strat.index)).mean() * TRADING_DAYS)
        print(f"  ▶ {name:18s} 全期: 几何 ann {full_ann:+.4f} bench {full_b:+.4f} 几何超额 {full_ann-full_b:+.4f} "
              f"算术超额 {ex_arith:+.4f} | 夏普 {full_sh:.2f} | 日换手 {st.turnover.mean():.4f} "
              f"| 成本 {st.cost_bps.mean():.2f}bp/日 ≈ {st.cost_bps.mean()*TRADING_DAYS/100:.2f}pp/年 "
              f"| 持仓 {st.holdings.mean():.0f} | 仓位 {st.exposure.mean():.3f} | maxDD(逐年均) {st.max_dd.mean():.3f}")
        runs[name] = dict(strat=strat, phases={ph: pd.concat(v).sort_index() for ph, v in phase_parts.items()},
                          stats=st, H=(pd.concat(H_parts).sort_index() if rec else None),
                          full=dict(ann=full_ann, bench=full_b, sharpe=full_sh, excess_geo=full_ann - full_b,
                                    excess_arith=ex_arith, turnover=float(st.turnover.mean()),
                                    cost_bps=float(st.cost_bps.mean()), holdings=float(st.holdings.mean()),
                                    exposure=float(st.exposure.mean())))
        ph_mean = pd.concat(runs[name]["phases"], axis=1).mean(axis=1).reindex(strat.index)
        assert float((ph_mean - strat).abs().max()) < 1e-10
        for y, s in st.iterrows():
            for k, v in s.items():
                rows.append(dict(n_or_spec=name, term=f"backtest_{y}|{k}", beta=np.nan, factor_ann=np.nan,
                                 contrib_ann=v, t=np.nan, se=np.nan, note="逐年回测统计"))
        for k, v in runs[name]["full"].items():
            rows.append(dict(n_or_spec=name, term=f"backtest_full|{k}", beta=np.nan, factor_ann=np.nan,
                             contrib_ann=v, t=np.nan, se=np.nan, note="全期 2023–25 回测统计(成本 bp/日)"))
    if not top20_ok:
        raise AssertionError("top20 重建与 ml_results 账本不符(>1pp)。")
    print("  top20 重建对账通过(逐年 ann / bench / excess 均在 ±1pp 内)。")

    ir = pd.read_parquet(LEDGER / "ir_daily_series.parquet")
    dmax = float((ir["open10|基准 池等权"].reindex(bench.index) - bench).abs().max())
    print(f"  ir_daily_series 池等权 vs ST.benchmark 同窗最大差 {dmax:.2e} ({'同源' if dmax < 1e-8 else '不同源!'})")

    # ---- 3. 因子收益与对齐 ----
    F = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0,
                    parse_dates=True)[EXEC].astype(float)
    styles = [c for c in F.columns if not c.startswith("行业:") and c != "Country"]
    indus = [c for c in F.columns if c.startswith("行业:")]
    idx = bench.index.intersection(F.dropna(how="any").index)
    for name in runs:
        idx = idx.intersection(runs[name]["strat"].index)
    Fi = F.reindex(idx)
    print(f"\n纯因子收益: {len(styles)} 风格 + {len(indus)} 行业; 回归样本 {len(idx)} 天 ({idx[0].date()} → {idx[-1].date()})")
    print("\n" + "=" * 100 + "\n3. 对齐断言\n" + "=" * 100)
    assert_aligned(bench.reindex(idx), Fi["Country"], "池等权 vs Country 纯因子")
    for name in runs:
        assert_aligned(runs[name]["strat"].reindex(idx), bench.reindex(idx), f"{name} vs 池等权")

    # ---- 4. 逐档归因 ----
    print("\n" + "=" * 100 + "\n4. 逐档 Barra 归因(全期 2023–25;分辨率阶梯)\n" + "=" * 100)
    daily = pd.DataFrame({"bench": bench.reindex(idx)})
    ladder_rows = []
    for name in runs:
        y_all = (runs[name]["strat"].reindex(idx) - bench.reindex(idx)).astype(float)
        res = regress(y_all, Fi, styles, indus)
        runs[name]["res"] = res
        print_res(f"{name} 全期", res)
        emit(rows, name, "full_2023_2025", res, styles, indus)
        daily[f"{name}|strategy"] = runs[name]["strat"].reindex(idx)
        daily[f"{name}|excess"] = y_all
        daily[f"{name}|fitted"] = res["fitted"]
        daily[f"{name}|resid"] = res["resid"]
        ladder_rows.append(dict(spec=name, holdings=runs[name]["full"]["holdings"],
                                excess_ann=res["tot"], alpha_ann=res["alpha"], alpha_se=res["se_nw"][5],
                                alpha_t_q5=res["t_nw"][5], r2=res["r2"],
                                liq_beta=res["b"][res["terms"].index(LIQ) + 1], liq_contrib=res["contrib"][LIQ],
                                style_total=res["style_sum"], sharpe=runs[name]["full"]["sharpe"],
                                turnover=runs[name]["full"]["turnover"]))
    print("\n  十风格逐项(β / 贡献 pp)按档:")
    hdr = f"    {'风格':22s}" + "".join(f"{n:>22s}" for n in runs)
    print(hdr)
    for t in styles:
        line = f"    {t:22s}"
        for name in runs:
            res = runs[name]["res"]; i = res["terms"].index(t)
            line += f"  {res['b'][i+1]:+7.3f} / {res['contrib'][t]*100:+6.2f}pp"
        print(line)

    # ---- 5. 逐年 ----
    print("\n" + "=" * 100 + "\n5. 逐年归因\n" + "=" * 100)
    for name in runs:
        y_all = (runs[name]["strat"].reindex(idx) - bench.reindex(idx)).astype(float)
        yl = []
        for y in YEARS:
            m = idx.year == y
            res = regress(y_all[m], Fi[m], styles, indus)
            print_res(f"{name} {y}", res)
            emit(rows, name, f"year_{y}", res, styles, indus)
            yl.append(f"{y}: {res['alpha']*100:+.1f}±{res['se_nw'][5]*100:.1f}pp (t {res['t_nw'][5]:+.2f})")
        for lr in ladder_rows:
            if lr["spec"] == name:
                lr["yearly_alpha"] = "; ".join(yl)

    # ---- 6. 逐相位(只报 alpha,量级参考)----
    print("\n" + "=" * 100 + "\n6. 逐相位 alpha(单相位序列)\n" + "=" * 100)
    for name in runs:
        al = []
        for ph in range(REBAL):
            yp = (runs[name]["phases"][ph].reindex(idx) - bench.reindex(idx)).astype(float)
            res = regress(yp, Fi, styles, indus)
            emit(rows, name, f"phase_{ph}", res, styles, indus)
            al.append(f"ph{ph} {res['alpha']*100:+.2f}pp(t{res['t_nw'][5]:+.2f})")
        print(f"  {name:18s} " + " | ".join(al))

    # ---- 7. 同窗排雷对照 ----
    print("\n" + "=" * 100 + "\n7. 同窗排雷对照 COMP_v5 排雷缓冲20/35(ir_daily_series, 2023–25)\n" + "=" * 100)
    ctl = ir["open10|排雷缓冲20/35"].reindex(idx).astype(float)
    ctl_b = ir["open10|基准 池等权"].reindex(idx).astype(float)
    ok_c = ctl.notna() & ctl_b.notna()
    print(f"  排雷序列有效 {int(ok_c.sum())} / {len(idx)} 天; 几何年化 策略 {geo_ann(ctl[ok_c]):+.4f} 池等权 {geo_ann(ctl_b[ok_c]):+.4f}")
    assert_aligned(ctl[ok_c], ctl_b[ok_c], "排雷 vs 池等权(同窗)")
    y_c = ctl[ok_c] - ctl_b[ok_c]
    ctl_res = regress(y_c, Fi[ok_c], styles, indus)
    print_res("排雷同窗", ctl_res)
    emit(rows, "control_exclude_20_35", "full_2023_2025", ctl_res, styles, indus, note="COMP_v5 排雷缓冲20/35")
    for y in YEARS:
        m = (y_c.index.year == y)
        res = regress(y_c[m], Fi[ok_c][m], styles, indus)
        print_res(f"排雷 {y}", res)
        emit(rows, "control_exclude_20_35", f"year_{y}", res, styles, indus)
    daily["control_exclude"] = ctl; daily["control_bench"] = ctl_b
    ctl_ladder = dict(spec="control_exclude_20_35", holdings=np.nan, excess_ann=ctl_res["tot"],
                      alpha_ann=ctl_res["alpha"], alpha_se=ctl_res["se_nw"][5], alpha_t_q5=ctl_res["t_nw"][5],
                      r2=ctl_res["r2"], liq_beta=ctl_res["b"][ctl_res["terms"].index(LIQ) + 1],
                      liq_contrib=ctl_res["contrib"][LIQ], style_total=ctl_res["style_sum"],
                      sharpe=np.nan, turnover=np.nan, yearly_alpha="")

    # ---- 8. 打分 vs Barra 暴露的截面 Spearman;持仓法暴露 ----
    print("\n" + "=" * 100 + "\n8. 打分 vs 十个 Barra 暴露:逐日截面 Spearman(V1000 决策池内;与档位无关)\n" + "=" * 100)
    u_all = p["universe"]
    X = build_exposures(p, u_all)
    score_full = pd.concat([SC[y].loc[years == y] for y in YEARS]).reindex(p.dates)
    S = score_full.to_numpy(np.float32)
    Um = U1000.to_numpy(bool) & np.isfinite(S)
    spear = {}
    for k, xdf in X.items():
        Xv = xdf.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
        cs = []
        for ti in np.flatnonzero(oos):
            m = Um[ti] & np.isfinite(Xv[ti])
            if m.sum() < 100:
                continue
            a = pd.Series(S[ti, m]).rank().to_numpy(); b = pd.Series(Xv[ti, m]).rank().to_numpy()
            cs.append(np.corrcoef(a, b)[0, 1])
        cs = np.asarray(cs); sp = float(np.nanmean(cs)); spear[k] = sp
        print(f"    {k:22s} Spearman 均值 {sp:+.4f}  (std {np.nanstd(cs):.3f}, n={len(cs)}, t={sp/np.nanstd(cs)*np.sqrt(len(cs)):+.1f})")
        rows.append(dict(n_or_spec="score_panel", term=f"score_exposure_spearman|{k}", beta=np.nan, factor_ann=np.nan,
                         contrib_ann=np.nan, t=sp, se=float(np.nanstd(cs) / np.sqrt(len(cs))),
                         note=f"逐日截面 Spearman 均值, n={len(cs)}; se=std/sqrt(n)"))

    print("\n" + "=" * 100 + "\n9. 持仓法暴露与归因(5 相位平均持仓 h_t × 暴露 X_t − 池等权暴露;贡献 = mean(主动暴露×纯因子)×242)\n" + "=" * 100)
    # 行业:纯因子收益里最后一个行业由约束隐含 f_last = -Σ_j cap_j/cap_last f_j(cap 为估计池市值和)
    Lab = ND.sw_industry_labels(p.dates, p.codes).reindex(index=p.dates, columns=p.codes)
    names = ND.sw_industry_names()
    inv = {v: k for k, v in names.items()}
    ind_codes_listed = [inv[t[3:]] for t in indus]
    u_arr = u_all.reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
    all_codes = sorted({int(c) for c in pd.unique(Lab.where(u_all).values.ravel()) if pd.notna(c) and int(c) >= 0})
    last_code = [c for c in all_codes if c not in ind_codes_listed]
    assert len(last_code) == 1, f"约束隐含的行业应恰有 1 个: {last_code}"
    last_code = last_code[0]
    print(f"  行业 {len(indus)} 个进回归 + 约束隐含 1 个({names.get(last_code, last_code)})")
    mv = FA._real(p, "mv").reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    Larr = Lab.to_numpy(np.float64)
    Xarr = {k: v.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64) for k, v in X.items()}
    Ub = U1000.to_numpy(bool)
    pos_idx = p.dates.get_indexer(idx)
    Fi_arr = Fi.to_numpy(float); fcols = list(Fi.columns)
    hold_rows = []
    for name in [n for n in runs if runs[n]["H"] is not None]:
        H = runs[name]["H"].reindex(index=idx, columns=p.codes).fillna(0.0).to_numpy(np.float64)
        act = {k: np.full(len(idx), np.nan) for k in X}
        act_ind = np.full((len(idx), len(indus)), np.nan)
        f_last = np.full(len(idx), np.nan)
        for j, ti in enumerate(pos_idx):
            h = H[j]; ub = Ub[ti]; nb_ = ub.sum()
            if nb_ < 100:
                continue
            bw = ub / nb_
            for k in X:
                x = Xarr[k][ti]; fin = np.isfinite(x)
                act[k][j] = np.nansum(h[fin] * x[fin]) - np.nansum(bw[fin] * x[fin])
            lab = Larr[ti]
            for jj, c in enumerate(ind_codes_listed):
                m = lab == c
                act_ind[j, jj] = h[m].sum() - bw[m].sum()
            # 隐含行业收益
            ue = u_arr[ti] & np.isfinite(lab) & np.isfinite(mv[ti])
            caps = np.array([mv[ti][ue & (lab == c)].sum() for c in ind_codes_listed])
            cap_last = mv[ti][ue & (lab == last_code)].sum()
            fj = np.array([Fi_arr[j, fcols.index(t)] for t in indus])
            f_last[j] = -(caps / cap_last * fj).sum() if cap_last > 0 else np.nan
        m_last = (Larr[pos_idx] == last_code)
        act_last = np.array([H[j][m_last[j]].sum() - Ub[ti][m_last[j]].sum() / Ub[ti].sum()
                             for j, ti in enumerate(pos_idx)])
        res = runs[name]["res"]
        tot = res["tot"]
        sty_c = {}
        print(f"  [{name}] 主动暴露均值 / 回归β / 持仓法贡献pp / 回归贡献pp:")
        for k in X:
            a = act[k]; f = Fi_arr[:, fcols.index(k)]
            c = float(np.nanmean(a * f) * TRADING_DAYS)
            sty_c[k] = c
            i = res["terms"].index(k)
            print(f"    {k:22s} 暴露 {np.nanmean(a):+.3f}  β {res['b'][i+1]:+.3f}  贡献 {c*100:+.2f} / {res['contrib'][k]*100:+.2f}")
            hold_rows.append(dict(n_or_spec=name, term=f"holdings_exposure|{k}", beta=float(np.nanmean(a)),
                                  factor_ann=res["fann"][k], contrib_ann=c, t=np.nan,
                                  se=float(np.nanstd(a)), note=f"beta=主动暴露均值(持仓法), se=暴露日 std; 回归β {res['b'][i+1]:+.3f}"))
        ind_c = 0.0
        for jj, t in enumerate(indus):
            ind_c += float(np.nanmean(act_ind[:, jj] * Fi_arr[:, fcols.index(t)]) * TRADING_DAYS)
        ind_c += float(np.nanmean(act_last * f_last) * TRADING_DAYS)
        sty_sum = sum(sty_c.values())
        resid_h = tot - sty_sum - ind_c
        print(f"    持仓法合计: 风格 {sty_sum*100:+.2f}pp 行业 {ind_c*100:+.2f}pp 残差 {resid_h*100:+.2f}pp "
              f"(总超额 {tot*100:+.2f}) | 回归法: 风格 {res['style_sum']*100:+.2f} 行业 {res['ind_sum']*100:+.2f} alpha {res['alpha']*100:+.2f}")
        print(f"    持仓法 Liquidity 贡献 {sty_c[LIQ]*100:+.2f}pp ({sty_c[LIQ]/tot*100 if tot else np.nan:+.0f}%)")
        for term, v in (("__风格合计", sty_sum), ("__行业合计", ind_c), ("__残差", resid_h), ("__总超额", tot)):
            hold_rows.append(dict(n_or_spec=name, term=f"holdings_exposure|{term}", beta=np.nan, factor_ann=np.nan,
                                  contrib_ann=v, t=np.nan, se=np.nan, note="持仓法"))
        for lr in ladder_rows:
            if lr["spec"] == name:
                lr["hold_liq_exposure"] = float(np.nanmean(act[LIQ]))
                lr["hold_resid"] = resid_h
    rows.extend(hold_rows)

    # ---- 10. 阶梯汇总 ----
    print("\n" + "=" * 100 + "\n10. 分辨率阶梯汇总(NW q=5)\n" + "=" * 100)
    lad = pd.DataFrame(ladder_rows + [ctl_ladder]).set_index("spec")
    show = lad.copy()
    for c in ("excess_ann", "alpha_ann", "alpha_se", "liq_contrib", "style_total"):
        show[c] = show[c] * 100
    print(show[["holdings", "excess_ann", "alpha_ann", "alpha_se", "alpha_t_q5", "r2", "liq_beta",
                "liq_contrib", "style_total", "sharpe", "turnover"]].round(3).to_string())
    print("  可检出门槛 1.96·SE(pp): " + ", ".join(f"{s}={v*196:.1f}" for s, v in lad.alpha_se.items()))
    for s, r in lad.iterrows():
        rows.append(dict(n_or_spec=s, term="ladder|alpha_se_q5", beta=np.nan, factor_ann=np.nan,
                         contrib_ann=r.alpha_ann, t=r.alpha_t_q5, se=r.alpha_se,
                         note=f"holdings={r.holdings:.0f}; excess={r.excess_ann*100:+.2f}pp; R2={r.r2:.3f}; 逐年 {r.get('yearly_alpha','')}"))

    # ---- 落盘 ----
    out = pd.DataFrame(rows, columns=["n_or_spec", "term", "beta", "factor_ann", "contrib_ann", "t", "se", "note"])
    out.to_csv(LEDGER / "barra_topfrac_summary.csv", index=False)
    daily.index.name = "date"
    daily.to_csv(LEDGER / "barra_topfrac_daily.csv", float_format="%.8g")
    lad.to_csv(LEDGER / "barra_topfrac_ladder.csv")
    print(f"\n已写 {LEDGER/'barra_topfrac_summary.csv'}, {LEDGER/'barra_topfrac_daily.csv'}, "
          f"{LEDGER/'barra_topfrac_ladder.csv'} | 用时 {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
