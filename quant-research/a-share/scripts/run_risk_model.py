"""Barra 式风险模型:搬迁核对 -> 构建缓存 -> 偏差检验(bias statistic)-> 事后归因示例。

## 三个阶段
  move   用库里的原版函数(src/afactor/barra.py: build_exposures + pure_factor_returns,逻辑未改)
         对 open10 口径重算纯因子收益,与 research/ledger/barra_factor_returns.csv 对账。
  build  RiskModel(**VARIANTS[v]).build(Panel(), "open10") 三版(v1 预注册原版暴露 / v2 稳健截尾 / v3 Size 不截尾):
         风险模型版回归(缺失填 0、银行回到样本、行业全还原)、
         EWMA+NW 因子协方差、收缩后的特质方差,缓存到 cache/_tmp/risk_open10_*;
         并把风险模型版因子收益与账本逐因子对比,现场复算几个决策日的协方差核对时序。
  bias   偏差统计量 b = std(z),z = 实现的 5 日收益 / 预测的 5 日波动,按年(2021–2025)报告:
           (i)   每个纯因子
           (ii)  200 个随机 100 只股票等权组合(V1000 决策池内每个决策日重抽)
           (iii) COMP_v5 标准排雷方案(缓冲 20/35)目标持仓 − 池等权 的主动组合
           (iv)  附加:B×GBT Top20(walk-forward 样本外打分,2023–25)目标持仓 − 池等权
           (v)   附加:池等权基准本身的总风险
         理想值 1 ± sqrt(2/T)(T = 每个相位的不重叠窗口数)。
         另做 COMP_v5 的持仓归因(attribute)示例与图。

## 偏差检验的口径(写死,不得按结果调)
- 决策日 t 的预测用 at(t):协方差只含 F[..t−2]、特质方差只含残差[..t−2]、暴露 X[t]。
- 实现收益 = R[t]+…+R[t+4],R = fwd_returns(open10) 即 twap_open10[s+2]/twap_open10[s+1]−1,
  缺失(停牌/一字板无价)记 0,与回测引擎"不可成交记 0"一致;用原始 twap(不剔一字板)。
- 不重叠窗口:同一年内按 (位置 − 当年首个决策日位置) mod 5 分 5 个相位,各相位各算一个 b,
  报告 5 相位平均与极差;相位之间共享行情、不独立,不能当 5 倍样本。
- 2021 年:纯因子收益要等 252 日动量窗口(2021-01 才有),协方差再要 60 个观测,
  所以 2021 年只覆盖约 4 月以后。
- COMP_v5 成员是在 2020–2025 研究期内选出的 -> (iii) 是**信号样本内**的组合;偏差检验测的是
  风险模型对这种主动组合的风险预测是否准,不涉及信号的样本外能力。B×GBT 打分在 2023–25 是样本外。
- 幸存者偏差:面板缺约 209 只已退市股,而退市股退市前恰是特质波动最大的一批 ->
  实现波动偏低、b 偏低的方向。

只读研究期面板 Panel(),不触碰封存样本(面板截至 2025-12-31,另加断言)。

## 产出
  research/ledger/risk_move_check.csv          搬迁核对 + 风险模型版 vs 账本
  research/ledger/risk_bias_stats.csv          偏差统计量主表
  research/ledger/risk_bias_random_detail.csv  200 个随机组合逐个的 b
  research/ledger/risk_bias_diagnostics.csv    偏差诊断:因子/特质拆分、单股特质校准、自相关、2024 事后剔除
  research/ledger/risk_bias_windows.csv        逐窗口 z(Country/池等权/COMP_v5 主动/Top20 主动)
  research/ledger/risk_model_summary.json      覆盖率、R²、特质波动、时序复算核对等
  research/ledger/risk_attribution_comp_v5.csv COMP_v5 持仓归因(按年)
  research/ledger/risk_attribution_comp_v5_factors.csv  逐因子贡献
  reports/risk_bias.png
"""
from __future__ import annotations
import os, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore", message=".*encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

from afactor.config import LEDGER, TMP, REPORTS, RESEARCH_END, assert_not_sealed
from afactor.panel import Panel
from afactor import factors as FA, strategy as ST, backtest as BT, barra as BR
from afactor.strategy import TRADING_DAYS

pd.set_option("display.width", 240)

MODE, VERSION = "open10", "V1000"
YEARS = [2021, 2022, 2023, 2024, 2025]
# 纪律 6:开发窗 2021–23 / 检验窗 2024–25 分开报;另报全期
PERIODS = {"2021-2023": (2021, 2023), "2024-2025": (2024, 2025), "2021-2025": (2021, 2025)}
ALL_P = YEARS + list(PERIODS)


def pmask(yr, y) -> np.ndarray:
    """yr 为年份数组;y 为单年或 PERIODS 的键。"""
    yr = np.asarray(yr)
    if y in PERIODS:
        lo, hi = PERIODS[y]
        return (yr >= lo) & (yr <= hi)
    return yr == y
H = 5                       # 持有期(与风险模型 horizon 一致)
N_RANDOM, RANDOM_SIZE, SEED = 200, 100, 20251010
ANN5 = np.sqrt(TRADING_DAYS / H)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def load_panel() -> Panel:
    p = Panel()
    assert_not_sealed(p.dates)
    assert p.dates.max() <= pd.Timestamp(RESEARCH_END), "面板越过研究期"
    return p


def read_ledger_F() -> pd.DataFrame:
    led = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0)
    led.index = pd.to_datetime(led.index)
    return led[MODE].dropna(how="all").astype(float)


# ================================================================ 阶段 1:搬迁核对
def stage_move(p) -> pd.DataFrame:
    FA.build_all()
    log("原版逻辑重算 open10 纯因子收益(库函数,cover=None)…")
    R = BR.fwd_returns(p, MODE)
    u = p["universe"].reindex_like(R).fillna(False)
    X0 = BR.build_exposures(p, u)
    F0 = BR.pure_factor_returns(R, X0, u, p)
    del X0
    F0.to_parquet(TMP / "risk_open10_F_orig.parquet")
    led = read_ledger_F()
    common = led.index.intersection(F0.index)
    cols_same = list(led.columns) == list(F0.columns)
    rows = []
    for c in led.columns:
        a, b = F0.loc[common, c].astype(float), led.loc[common, c]
        d = (a - b).abs()
        rows.append(dict(section="move_vs_ledger", factor=c, n=len(common), max_abs_diff=float(d.max()),
                         mean_abs_diff=float(d.mean()), corr=float(a.corr(b)),
                         std_ledger=float(b.std()), std_new=float(a.std()), note=""))
    dall = (F0.loc[common, led.columns].astype(float) - led.loc[common]).abs().to_numpy()
    rows.append(dict(section="move_vs_ledger", factor="__ALL", n=len(common),
                     max_abs_diff=float(np.nanmax(dall)), mean_abs_diff=float(np.nanmean(dall)),
                     corr=np.nan, std_ledger=np.nan, std_new=np.nan,
                     note=f"列名与顺序一致={cols_same}; 新算 {len(F0)} 天 [{F0.index[0].date()}~{F0.index[-1].date()}],"
                          f" 账本 {len(led)} 天, 交集 {len(common)}, 只在新算 {len(F0.index.difference(led.index))},"
                          f" 只在账本 {len(led.index.difference(F0.index))}"))
    out = pd.DataFrame(rows)
    log(f"搬迁核对:交集 {len(common)} 天, 全体最大绝对差 {np.nanmax(dall):.3e}, 列一致 {cols_same}")
    return out


# ================================================================ 阶段 2:构建
def stage_build(p, force: bool, ver: str) -> tuple[BR.RiskModel, pd.DataFrame, dict]:
    rm = BR.RiskModel(**BR.RiskModel.VARIANTS[ver]).build(p, MODE, force=force)
    led = read_ledger_F()
    F = rm.F
    common = led.index.intersection(F.index)
    rows = []
    for c in led.columns:
        a, b = F.loc[common, c], led.loc[common, c]
        rows.append(dict(section=f"riskreg_{ver}_vs_ledger", factor=c, n=len(common),
                         max_abs_diff=float((a - b).abs().max()), mean_abs_diff=float((a - b).abs().mean()),
                         corr=float(a.corr(b)) if b.std() > 0 else np.nan,
                         std_ledger=float(b.std()), std_new=float(a.std()),
                         note="账本此列 99.9% 为 0(银行被完整样本回归剔除)" if b.std() < 1e-6 or (b == 0).mean() > 0.5 else ""))
    extra = [c for c in F.columns if c not in led.columns]
    for c in extra:
        rows.append(dict(section=f"riskreg_{ver}_vs_ledger", factor=c, n=len(F), max_abs_diff=np.nan,
                         mean_abs_diff=np.nan, corr=np.nan, std_ledger=np.nan, std_new=float(F[c].std()),
                         note="账本无此列:原版把最后一个行业用约束表出;风险模型版还原了它"))

    # ---- 现场复算:若干决策日的协方差 = 只用 pos ≤ t−2 的 F 从头估计
    dates = rm.dates
    pos_F = dates.get_indexer(F.index)
    checks = []
    avail_i = np.flatnonzero(np.isfinite(rm.cov[:, 0, 0]))
    for i in avail_i[[0, len(avail_i) // 3, 2 * len(avail_i) // 3, -1]]:
        keep = pos_F <= i - BR.INFO_LAG
        C = BR.factor_cov_series(F.to_numpy()[keep], rm.params["half_life_corr"],
                                 rm.params["half_life_vol"], rm.params["nw_lags"], rm.params["horizon"])[-1]
        Cp, _ = BR.psd_sqrt(C)
        last_used = F.index[keep][-1]
        checks.append(dict(decision=str(dates[i].date()), last_factor_date_used=str(last_used.date()),
                           trading_days_gap=int(i - pos_F[keep][-1]),
                           max_abs_diff=float(np.abs(Cp - rm.cov[i]).max())))
    assert all(c["max_abs_diff"] < 1e-12 and c["trading_days_gap"] >= 2 for c in checks), checks

    # ---- 诊断摘要
    mask = np.asarray(rm.mask)
    U = ST.universe(p, VERSION).reindex(index=dates, columns=rm.codes).to_numpy(bool)
    yr = dates.year
    diag = rm.diag
    spec = np.asarray(rm.spec)
    summ = dict(
        params=rm.params, factor_names=rm.factor_names, n_styles=rm.K, n_industries=rm.J,
        F_first=rm.meta["F_first"], F_last=rm.meta["F_last"], n_F_days=int(len(F)),
        first_forecast=rm.meta["first_forecast"], cov_time_checks=checks,
        by_year={})
    chol_fail = 0
    for i in avail_i:
        try:
            np.linalg.cholesky(rm.cov[i])
        except np.linalg.LinAlgError:
            chol_fail += 1
    summ["cholesky_fallback_days"] = chol_fail
    # 风格暴露在估计池内的截面标准差(按年平均):原版 zscore 下应≈1,异常值会把它压扁
    sd_rows, lo_rows, hi_rows, le_rows, sq_rows = {}, {}, {}, {}, {}
    for i in avail_i:
        m = U[i] & mask[i]
        xi = np.asarray(rm.X[i])[m]
        sd_rows[dates[i]] = np.nanstd(xi, axis=0)
        lo_rows[dates[i]] = np.isclose(xi, -BR.WINSOR, atol=1e-6).mean(axis=0)   # 恰好压在 −3 上的比例
        hi_rows[dates[i]] = np.isclose(xi, BR.WINSOR, atol=1e-6).mean(axis=0)
        le_rows[dates[i]] = (xi <= -BR.WINSOR + 1e-6).mean(axis=0)              # ≤ −3(含越界)的比例
        sq_rows[dates[i]] = np.nanpercentile(xi[:, 0], [0, 1, 5, 25, 50, 75, 95, 99, 100])
    sq = pd.DataFrame(sq_rows, index=["min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"]).T
    summ["size_exposure_quantiles_by_year"] = {str(y): {k: round(float(v), 3) for k, v in r.items()}
                                               for y, r in sq.groupby(sq.index.year).mean().iterrows()}
    for key, rows_ in [("style_frac_pinned_minus3_by_year", lo_rows), ("style_frac_pinned_plus3_by_year", hi_rows),
                       ("style_frac_le_minus3_by_year", le_rows)]:
        fr = pd.DataFrame(rows_, index=rm.style_names).T
        summ[key] = {str(y): {k: round(float(v), 4) for k, v in r.items()}
                     for y, r in fr.groupby(fr.index.year).mean().iterrows()}
    sd = pd.DataFrame(sd_rows, index=rm.style_names).T
    g = sd.groupby(sd.index.year)
    summ["style_xs_std_mean_by_year"] = {str(y): {k: round(float(v), 4) for k, v in r.items()}
                                         for y, r in g.mean().iterrows()}
    summ["style_xs_std_min_by_year"] = {str(y): {k: round(float(v), 4) for k, v in r.items()}
                                        for y, r in g.min().iterrows()}
    summ["n_forecast_days"] = int(len(avail_i))
    cond = [float(np.linalg.cond(rm.cov[i])) for i in avail_i[::20]]
    summ["cov_condition_number_median"] = float(np.median(cond))
    for y in sorted(set(yr)):
        sel = (yr == y)
        av = sel & np.isfinite(rm.cov[:, 0, 0])
        dsel = diag[diag.index.year == y]
        row = dict(
            n_F_days=int((F.index.year == y).sum()),
            n_forecast_days=int(av.sum()),
            n_est_mean=float(dsel["n_est"].mean()) if len(dsel) else None,
            n_cover_mean=float(dsel["n_cover"].mean()) if len(dsel) else None,
            wls_r2_mean=float(dsel["wls_r2"].mean()) if len(dsel) else None)
        if av.any():
            m_av = mask[av]
            row["n_mask_mean"] = float(m_av.sum(1).mean())
            row["univ_not_covered_frac"] = float((U[av] & ~m_av).sum() / max(U[av].sum(), 1))
            sv = np.sqrt(spec[av]) * ANN5
            row["spec_vol_ann_median_univ"] = float(np.nanmedian(np.where(U[av], sv, np.nan)))
            fv = np.sqrt(np.einsum("tii->ti", rm.cov[av])) * ANN5
            row["country_vol_ann_mean"] = float(fv[:, 0].mean())
            row["style_vol_ann_mean"] = float(fv[:, 1:1 + rm.K].mean())
            row["industry_vol_ann_mean"] = float(fv[:, 1 + rm.K:].mean())
        summ["by_year"][str(y)] = row
    log(f"构建完成:F {summ['F_first']}~{summ['F_last']},首个预测日 {summ['first_forecast']},"
        f" Cholesky 退回特征分解 {chol_fail}/{len(avail_i)} 天")
    return rm, pd.DataFrame(rows), summ


# ================================================================ 阶段 3:偏差检验
def five_day_sum(A: np.ndarray, require_all: bool) -> np.ndarray:
    """A[t..t+4] 之和,[T−4 x ...]。require_all=True 时窗口内有 NaN 则 NaN,否则 NaN 记 0。"""
    if require_all:
        out = sum(A[k:len(A) - H + 1 + k] for k in range(H))
        return out
    A0 = np.nan_to_num(A)
    cs = np.concatenate([np.zeros((1,) + A.shape[1:]), np.cumsum(A0, axis=0)])
    return cs[H:] - cs[:-H]


def bias_table(test: str, name: str, pos: np.ndarray, dates: pd.DatetimeIndex,
               pred: np.ndarray, real: np.ndarray, note: str = "") -> list[dict]:
    """按年 × 5 相位算 b = std(z)。pos 是决策日在面板中的位置,pred 是预测的 5 日波动。"""
    ok = np.isfinite(pred) & np.isfinite(real) & (pred > 0)
    pos, pred, real = pos[ok], pred[ok], real[ok]
    z = real / pred
    yr = dates[pos].year
    rows = []
    for y in ALL_P:
        sel = pmask(yr, y)
        if sel.sum() < 2 * H:
            continue
        p_sel = pos[sel]
        base = p_sel.min()
        ph = (p_sel - base) % H
        bs, ns = [], []
        for k in range(H):
            zk = z[sel][ph == k]
            if len(zk) >= 5:
                bs.append(float(np.std(zk, ddof=1))); ns.append(len(zk))
        if not bs:
            continue
        n = float(np.mean(ns))
        b = float(np.mean(bs))
        band = np.sqrt(2.0 / n)
        rows.append(dict(test=test, portfolio=name, year=str(y), n_windows_per_phase=n,
                         bias=b, bias_phase_min=min(bs), bias_phase_max=max(bs),
                         band_lo=1 - band, band_hi=1 + band, in_band=bool(abs(b - 1) <= band),
                         mean_z=float(z[sel].mean()),
                         pred_vol_ann=float(pred[sel].mean() * ANN5),
                         real_vol_ann=float(real[sel].std(ddof=1) * ANN5),
                         n_decisions=int(sel.sum()), note=note))
    return rows


def stage_bias(p, rm: BR.RiskModel):
    dates, codes = rm.dates, rm.codes
    assert dates.equals(p.dates) and codes.equals(p.codes)
    R = BR.fwd_returns(p, MODE)
    T_R = len(R)
    Rv = R.to_numpy(np.float64)
    R5 = five_day_sum(Rv, require_all=False)                 # [T_R−4 x N],缺失记 0
    U = ST.universe(p, VERSION).reindex(index=dates, columns=codes).to_numpy(bool)

    # ---- 对齐核对:R 在池上的等权 = 引擎基准(同索引、同数值)
    pool_r = pd.Series(np.nanmean(np.where(U[:T_R], Rv, np.nan), axis=1), index=R.index).fillna(0.0)
    bench = ST.benchmark(p, VERSION, MODE)["rets"]
    common = pool_r.index.intersection(bench.index)
    bdiff = float((pool_r[common] - bench[common]).abs().max())
    BR.assert_aligned(pool_r[common], bench[common], "实现收益 vs 引擎基准")
    log(f"实现收益与引擎池等权基准逐日最大差 {bdiff:.2e}")
    assert bdiff < 1e-6

    avail = np.isfinite(rm.cov[:, 0, 0])
    test_pos = np.flatnonzero(avail & (np.arange(len(dates)) <= T_R - H)
                              & (dates.year >= 2021) & (dates.year <= 2025))
    log(f"检验决策日 {len(test_pos)} 个 [{dates[test_pos[0]].date()} ~ {dates[test_pos[-1]].date()}]")
    rows = []

    # ---------------- (i) 纯因子
    Ff = rm.F.reindex(dates[:T_R]).to_numpy(np.float64)
    F5 = five_day_sum(Ff, require_all=True)
    for k, nm in enumerate(rm.factor_names):
        pred = np.sqrt(rm.cov[test_pos, k, k])
        real = F5[test_pos, k]
        kind = "country" if k == 0 else ("style" if k <= rm.K else "industry")
        rows += bias_table("factor", nm, test_pos, dates, pred, real, note=kind)
    log("(i) 纯因子完成")

    # ---------------- 公用:全部股票的 one-hot 行业(快照,不随时间变)
    Hall = rm._onehot(rm.ind)

    def port_pred(i, a):
        """a 为对 codes 的主动/总权重向量;返回 (预测 5 日波动, 覆盖池外权重)。"""
        m = np.asarray(rm.mask[i])
        X = np.nan_to_num(np.asarray(rm.X[i], np.float64))
        d = np.nan_to_num(np.asarray(rm.spec[i], np.float64))
        am = np.where(m, a, 0.0)
        e = np.concatenate([[am.sum()], X.T @ am, Hall.T @ am])
        v = float(e @ rm.cov[i] @ e + np.sum(d * am * am))
        return np.sqrt(max(v, 0.0)), float(np.abs(a[~m]).sum())

    # ---------------- (ii) 200 个随机 100 只等权
    rng = np.random.default_rng(SEED)
    predR = np.full((len(test_pos), N_RANDOM), np.nan)
    realR = np.full((len(test_pos), N_RANDOM), np.nan)
    for j, i in enumerate(test_pos):
        m = np.asarray(rm.mask[i])
        cand = np.flatnonzero(U[i] & m)
        pick = cand[np.argpartition(rng.random((N_RANDOM, len(cand))), RANDOM_SIZE, axis=1)[:, :RANDOM_SIZE]]
        X = np.nan_to_num(np.asarray(rm.X[i], np.float64))
        d = np.asarray(rm.spec[i], np.float64)
        e = np.concatenate([np.ones((N_RANDOM, 1)), X[pick].mean(1), Hall[pick].mean(1)], axis=1)
        vf = np.einsum("pm,mn,pn->p", e, rm.cov[i], e)
        vs = d[pick].sum(1) / RANDOM_SIZE ** 2
        predR[j] = np.sqrt(vf + vs)
        realR[j] = R5[i][pick].mean(1)
    det_rows = []
    for q in range(N_RANDOM):
        det_rows += bias_table("random100", f"rp{q:03d}", test_pos, dates, predR[:, q], realR[:, q])
    det = pd.DataFrame(det_rows)
    for y, g in det.groupby("year", sort=False):
        b = g["bias"]
        base = dict(test="random100", year=y, n_windows_per_phase=g["n_windows_per_phase"].mean(),
                    band_lo=g["band_lo"].mean(), band_hi=g["band_hi"].mean(),
                    mean_z=g["mean_z"].mean(), pred_vol_ann=g["pred_vol_ann"].mean(),
                    real_vol_ann=g["real_vol_ann"].mean(), n_decisions=int(g["n_decisions"].mean()),
                    bias_phase_min=g["bias_phase_min"].mean(), bias_phase_max=g["bias_phase_max"].mean())
        for stat, val in [("mean_of_200", b.mean()), ("median_of_200", b.median()),
                          ("p05_of_200", b.quantile(0.05)), ("p95_of_200", b.quantile(0.95))]:
            rows.append(dict(base, portfolio=stat, bias=float(val),
                             in_band=bool(abs(val - 1) <= (base["band_hi"] - 1)),
                             note=f"200 个组合在同一批日期上高度相关,不是 200 个独立样本;in_band 比例 {g['in_band'].mean():.2f}"))
    log("(ii) 随机组合完成")

    # ---------------- (iii) COMP_v5 标准排雷目标持仓 − 池等权
    score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    S = score.reindex(index=dates, columns=codes).to_numpy(np.float32)
    W = BT.weights_exclude_buffer(S, U, 1, 0.20, 0.35)       # 与 run_strategy 的 scheme=exclude, buffer 同一调用
    pool = U / np.maximum(U.sum(1, keepdims=True), 1)
    predC, realC, outC = [], [], []
    for i in test_pos:
        a = W[i].astype(np.float64) - pool[i]
        pv, ow = port_pred(i, a)
        predC.append(pv); outC.append(ow); realC.append(float(a @ R5[i]))
    predC, realC = np.array(predC), np.array(realC)
    rows += bias_table("comp_v5_active", "COMP_v5 排雷20/35 目标 − 池等权", test_pos, dates, predC, realC,
                       note=f"信号样本内(成员在 2020–25 选出);覆盖池外主动权重均值 {np.mean(outC):.2e}")
    log(f"(iii) COMP_v5 完成,预测年化主动风险均值 {predC.mean() * ANN5:.4f}")

    # ---------------- (iv) B×GBT Top20 OOS − 池等权
    predT, realT, posT, aT = [], [], [], {}
    for y in (2023, 2024, 2025):
        sc = pd.read_parquet(LEDGER / "scores" / f"B_gbt_open_h5_{y}.parquet")
        assert_not_sealed(sc.index)
        full = pd.DataFrame(np.nan, index=dates, columns=codes, dtype=np.float32)
        idx = sc.index.intersection(dates)
        full.loc[idx, :] = sc.reindex(index=idx, columns=codes).to_numpy(np.float32)
        Sy = full.to_numpy(np.float32)
        Wy = ST.topn_weights(Sy, U & (dates.year == y)[:, None], 20)
        for i in test_pos[dates[test_pos].year == y]:
            if Wy[i].sum() <= 0:
                continue
            a = Wy[i].astype(np.float64) - pool[i]
            pv, _ = port_pred(i, a)
            predT.append(pv); realT.append(float(a @ R5[i])); posT.append(i); aT[i] = a
    rows += bias_table("gbt_top20_active", "B×GBT Top20 目标 − 池等权", np.array(posT), dates,
                       np.array(predT), np.array(realT), note="walk-forward 样本外打分 2023–25")
    log("(iv) Top20 完成")

    # ---------------- (v) 池等权总风险
    predP, realP = [], []
    for i in test_pos:
        pv, _ = port_pred(i, pool[i])
        predP.append(pv); realP.append(float(pool[i] @ R5[i]))
    predP, realP = np.array(predP), np.array(realP)
    rows += bias_table("pool_total", "V1000 池等权(总风险)", test_pos, dates, predP, realP)
    log("(v) 池等权完成")

    # ---------------- 汇总行:风格 / 行业平均
    out = pd.DataFrame(rows)
    fac = out[out.test == "factor"]
    for kind, lab in [("style", "__风格平均"), ("industry", "__行业平均")]:
        g = fac[fac.note == kind].groupby("year", sort=False)
        for y, gg in g:
            out = pd.concat([out, pd.DataFrame([dict(
                test="factor", portfolio=lab, year=y, n_windows_per_phase=gg["n_windows_per_phase"].mean(),
                bias=gg["bias"].mean(), bias_phase_min=gg["bias"].min(), bias_phase_max=gg["bias"].max(),
                band_lo=gg["band_lo"].mean(), band_hi=gg["band_hi"].mean(), in_band=np.nan,
                mean_z=gg["mean_z"].mean(), pred_vol_ann=gg["pred_vol_ann"].mean(),
                real_vol_ann=gg["real_vol_ann"].mean(), n_decisions=int(gg["n_decisions"].mean()),
                note=f"{kind} 平均;bias_phase_min/max 此处为因子间最小/最大;in_band 比例 {gg['in_band'].mean():.2f}")])],
                ignore_index=True)

    series = dict(test_pos=test_pos, predC=predC, realC=realC, predP=predP, realP=realP,
                  F5c=F5[test_pos, 0], predF0=np.sqrt(rm.cov[test_pos, 0, 0]))
    ctx = dict(test_pos=test_pos, W=W, pool=pool, R=R, U=U, R5=R5, F5=F5, aT=aT, Hall=Hall,
               predC=predC, realC=realC, predP=predP, realP=realP,
               posT=np.array(posT), predT=np.array(predT), realT=np.array(realT))
    return out, det, series, ctx



# ================================================================ 诊断(解释偏差统计量为什么偏离 1)
JANFEB24 = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-29"))


def stage_diag(p, rm, ctx) -> pd.DataFrame:
    """五组诊断,全部写进 risk_bias_diagnostics.csv:
    (a) 主动组合的偏差拆成因子部分与特质部分:因子部分 = e_t·ΣF[t..t+4](e_t = B_t^T a_t),
        特质部分 = 实现主动收益 − 因子部分(含 5 日内暴露漂移),分别对预测的因子/特质波动标准化;
    (b) 单只股票特质风险的校准:z = Σresid[t..t+4] / sqrt(d_t),池内全部股票,按年与按市值十分位;
        同时报稳健尺度 1.4826·MAD(厚尾下 std 会被少数极端值拉高);
    (c) 特质收益 ×5 换算的前提(无自相关)检验:Σ(5 日特质和)² / Σ(5 个日特质²),=1 为无自相关;
    (d) 事后诊断:2024 年剔除 1–2 月(微盘股流动性危机)决策日后的偏差 —— 标注为事后,不是模型选择依据。
    (e) 缺失因子检验:COMP_v5 主动特质收益能被 5/20 日反转解释多少。
    """
    dates = rm.dates
    test_pos, W, pool, U = ctx["test_pos"], ctx["W"], ctx["pool"], ctx["U"]
    R5, F5, Hall = ctx["R5"], ctx["F5"], ctx["Hall"]
    rows = []

    def split(i, a):
        m = np.asarray(rm.mask[i])
        X = np.nan_to_num(np.asarray(rm.X[i], np.float64))
        d = np.nan_to_num(np.asarray(rm.spec[i], np.float64))
        am = np.where(m, a, 0.0)
        e = np.concatenate([[am.sum()], X.T @ am, Hall.T @ am])
        vf = float(e @ rm.cov[i] @ e); vs = float(np.sum(d * am * am))
        rf = float(e @ F5[i]); rt = float(a @ R5[i])
        return np.sqrt(vf), np.sqrt(vs), rf, rt - rf, vs / (vf + vs)

    # ---- (a)
    for test, pos, getw in [("comp_v5_active", test_pos, lambda i: W[i].astype(np.float64) - pool[i]),
                            ("gbt_top20_active", ctx["posT"], lambda i: ctx["aT"][i])]:
        res = np.array([split(i, getw(i)) for i in pos])
        rows += bias_table(f"{test}|因子部分", "", pos, dates, res[:, 0], res[:, 2])
        rows += bias_table(f"{test}|特质部分", "", pos, dates, res[:, 1], res[:, 3])
        yr = dates[pos].year
        for y in sorted(set(yr)):
            rows.append(dict(test=f"{test}|预测特质方差占比", portfolio="", year=str(y),
                             bias=np.nan, mean_z=np.nan, pred_vol_ann=np.nan, real_vol_ann=np.nan,
                             value=float(res[yr == y, 4].mean()), n_decisions=int((yr == y).sum()),
                             note="mean(d·a² / 总预测方差)"))

    # ---- (b)(c) 单只股票特质风险
    resid = rm.resid()
    T_R = R5.shape[0] + H - 1
    Rs = np.asarray(resid[:T_R], np.float64)
    fin = np.isfinite(Rs)
    S5 = five_day_sum(Rs, require_all=True)                      # 5 日都有特质收益才算
    SQ = five_day_sum(np.where(fin, Rs ** 2, np.nan), require_all=True)
    mvv = FA._real(p, "mv").reindex(index=dates, columns=rm.codes).to_numpy(np.float64)   # t 日市值分桶
    yr_all = dates[test_pos].year
    zs, ys, ds, ph, num, den = [], [], [], [], [], []
    for i in test_pos:
        m = U[i] & np.asarray(rm.mask[i]) & np.isfinite(S5[i])
        idx = np.flatnonzero(m)
        z = S5[i, idx] / np.sqrt(np.asarray(rm.spec[i], np.float64)[idx])
        zs.append(z); ys.append(np.full(len(idx), dates[i].year))
        rk = pd.Series(mvv[i, idx]).rank(pct=True).to_numpy()
        ds.append(np.minimum((rk * 10).astype(int), 9))
        ph.append(np.full(len(idx), i))
        num.append(S5[i, idx] ** 2); den.append(SQ[i, idx])
    zs, ys, ph = np.concatenate(zs), np.concatenate(ys), np.concatenate(ph)
    num, den = np.concatenate(num), np.concatenate(den)
    ds = np.concatenate(ds)

    def zrow(test, port, year, sel):
        z, pos = zs[sel], ph[sel]
        base = pos.min()
        bs, rb, ns = [], [], []
        for k in range(H):
            zk = z[(pos - base) % H == k]
            if len(zk) > 50:
                bs.append(np.std(zk, ddof=1))
                rb.append(1.4826 * np.median(np.abs(zk - np.median(zk))))
                ns.append(len(zk))
        ratio = float(num[sel].sum() / den[sel].sum())
        return dict(test=test, portfolio=port, year=str(year), n_windows_per_phase=float(np.mean(ns)),
                    bias=float(np.mean(bs)), bias_phase_min=float(min(bs)), bias_phase_max=float(max(bs)),
                    value=float(np.mean(rb)), mean_z=float(z.mean()),
                    kurtosis=float(pd.Series(z).kurt()), autocorr_ratio=ratio,
                    note="value=稳健尺度 1.4826·MAD;autocorr_ratio=Σ(5日特质和)²/Σ5个日特质², 1=无自相关;"
                         "n 是股票×窗口,同一窗口内股票不独立,不给 sqrt(2/T) 带")
    for y in ALL_P:
        sel = pmask(ys, y)
        if sel.sum() > 1000:
            rows.append(zrow("specific_single_stock", "V1000 池全部", y, sel))
    for k in range(10):
        rows.append(zrow("specific_single_stock", f"市值十分位 D{k + 1}", "2021-2025", (ds == k) & (ys >= 2021)))

    # ---- (e) 缺失因子检验:COMP_v5 主动组合的"特质"收益是不是没建模的短期反转?
    # 候选在看结果前定为两个:5 日与 20 日反转(Barra CNE6 的 Short-Term Reversal;CNE5 十风格里没有)。
    # 做法:每日把风险模型残差对候选暴露(t 日收盘已知,稳健截尾 + 同口径标准化)做 sqrt(市值) 加权回归,
    # 斜率 g_t = "缺失因子收益";再看 COMP 主动特质收益 s_t = a_t·resid_t 能被 (a_t·x_t)·g_t 解释多少(R²)。
    close = p["close_px"].reindex(index=dates, columns=rm.codes)
    uu = p["universe"].reindex(index=dates, columns=rm.codes).fillna(False)
    mvdf = pd.DataFrame(mvv, index=dates, columns=rm.codes)
    a_all = W[:T_R].astype(np.float64) - pool[:T_R]
    s_t = np.nansum(np.where(np.isfinite(Rs), Rs, 0.0) * np.where(np.asarray(rm.mask[:T_R]), a_all, 0.0), axis=1)
    yr_R = dates[:T_R].year
    keep_R = (yr_R >= 2021) & np.isfinite(rm.cov[:T_R, 0, 0])
    for nm, h in [("rev5", 5), ("rev20", 20)]:
        x = -(close / close.shift(h) - 1.0)
        z = BR.zscore(BR.robust_clip(x, uu, 5.0), uu, mvdf.where(uu)).to_numpy(np.float64)
        g = np.full(T_R, np.nan)
        for t in np.flatnonzero(keep_R):
            m = U[t] & np.isfinite(Rs[t]) & np.isfinite(z[t]) & np.isfinite(mvv[t]) & (mvv[t] > 0)
            if m.sum() < 200:
                continue
            ww = np.sqrt(mvv[t, m]); A_ = np.column_stack([np.ones(m.sum()), z[t, m]])
            g[t] = np.linalg.lstsq(A_ * np.sqrt(ww)[:, None], Rs[t, m] * np.sqrt(ww), rcond=None)[0][1]
        e = np.nansum(np.nan_to_num(z[:T_R]) * a_all, axis=1)             # 主动暴露(标准差单位)
        f = e * g
        ok = keep_R & np.isfinite(f)
        for y in ALL_P:
            sel = ok & pmask(yr_R, y)
            if sel.sum() < 30:
                continue
            c = float(np.corrcoef(s_t[sel], f[sel])[0, 1])
            rows.append(dict(test="missing_factor|COMP_v5 主动特质收益", portfolio=nm, year=str(y),
                             value=c ** 2, mean_z=np.nan, bias=np.nan,
                             pred_vol_ann=np.nan, real_vol_ann=float(np.std(g[sel]) * np.sqrt(TRADING_DAYS)),
                             n_decisions=int(sel.sum()),
                             contrib_ann=float(np.mean(f[sel]) * TRADING_DAYS),
                             spec_ann=float(np.mean(s_t[sel]) * TRADING_DAYS),
                             note=f"value=R²(s_t ~ e_t·g_t);corr={c:+.3f};主动暴露均值 e={np.mean(e[sel]):+.3f};"
                                  f"contrib_ann=Σ(e·g) 的算术年化;spec_ann=主动特质收益 s_t 的算术年化;"
                                  f"real_vol_ann=缺失因子收益 g 的年化波动(每单位暴露)"))

    # ---- (d) 事后:2024 剔除 1–2 月
    lo, hi = JANFEB24
    keep = ~((dates[test_pos] >= lo) & (dates[test_pos] <= hi))
    F5c = F5[test_pos, 0]; predF0 = np.sqrt(rm.cov[test_pos, 0, 0])
    for test, pos, pred, real in [
            ("Country", test_pos, predF0, F5c),
            ("pool_total", test_pos, ctx["predP"], ctx["realP"]),
            ("comp_v5_active", test_pos, ctx["predC"], ctx["realC"])]:
        r = [x for x in bias_table(f"{test}|事后:剔除2024年1–2月", "", pos[keep], dates, pred[keep], real[keep])
             if x["year"] == "2024"]
        rows += r
    kT = ~((dates[ctx["posT"]] >= lo) & (dates[ctx["posT"]] <= hi))
    rows += [x for x in bias_table("gbt_top20_active|事后:剔除2024年1–2月", "", ctx["posT"][kT], dates,
                                   ctx["predT"][kT], ctx["realT"][kT]) if x["year"] == "2024"]
    return pd.DataFrame(rows)

# ================================================================ 归因示例
def stage_attr(rm, W, pool, R, U):
    """COMP_v5 持仓归因。持仓代理 = 最近 5 个决策日目标权重的平均(5 相位重叠组合的稳态持仓,
    不含成交约束、成本与持仓漂移)− 当日池等权。逐日 a_t·R[t] 拆成 Country/风格/行业/特质。"""
    dates, codes = rm.dates, rm.codes
    T_R = len(R)
    Wavg = pd.DataFrame(W[:T_R], index=dates[:T_R]).rolling(H, min_periods=H).mean().to_numpy()
    A = pd.DataFrame(Wavg - pool[:T_R], index=dates[:T_R], columns=codes)
    A = A[(A.index.year >= 2021) & (A.index.year <= 2025)].dropna(how="all")
    daily, fac = rm.attribute(A, R, by_factor=True)
    irs = pd.read_parquet(LEDGER / "ir_daily_series.parquet")
    assert_not_sealed(irs.index)
    eng = (irs["open10|排雷缓冲20/35"] - irs["open10|基准 池等权"]).dropna()
    rows = []
    for y in ALL_P:
        sel = pmask(daily.index.year, y)
        d = daily[sel]
        r = dict(year=str(y), n_days=int(sel.sum()))
        for c in ["total", "country", "style", "industry", "specific"]:
            r[f"{c}_ann"] = float(d[c].mean() * TRADING_DAYS)
        r["specific_t_nw5"] = BR.nw_t(d["specific"].to_numpy(), 5)
        r["total_t_nw5"] = BR.nw_t(d["total"].to_numpy(), 5)
        r["uncovered_w_mean"] = float(d["uncovered_w"].mean())
        # 对照:引擎实跑的排雷缓冲 20/35(净成本、含成交约束)相对池等权的算术年化超额
        es = eng.reindex(d.index).dropna()
        r["engine_excess_ann"] = float(es.mean() * TRADING_DAYS) if len(es) else np.nan
        rows.append(r)
    summ = pd.DataFrame(rows)
    fac_ann = []
    for y in ALL_P:
        sel = pmask(fac.index.year, y)
        s = fac[sel].mean() * TRADING_DAYS
        fac_ann.append(s.rename(str(y)))
    fac_ann = pd.concat(fac_ann, axis=1)
    fac_ann.index.name = "factor"
    return summ, fac_ann


# ================================================================ 图
def plot(out: pd.DataFrame, series: dict, dates: pd.DatetimeIndex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))

    ax = axes[0]
    groups = [("factor", "Country", "Country 因子"),
              ("factor", "__风格平均", "风格因子平均"),
              ("factor", "__行业平均", "行业因子平均"),
              ("random100", "median_of_200", "随机 100 只(中位)"),
              ("pool_total", None, "池等权总风险"),
              ("comp_v5_active", None, "COMP_v5 主动"),
              ("gbt_top20_active", None, "Top20 主动")]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]   # 参考调色板固定顺序 1–7
    yrs = [str(y) for y in YEARS]
    wbar = 0.8 / len(groups)
    for g, (test, port, lab) in enumerate(groups):
        d = out[(out.test == test) & ((out.portfolio == port) if port else True)].set_index("year")
        vals = [d["bias"].get(y, np.nan) for y in yrs]
        ax.bar(np.arange(len(yrs)) + (g - len(groups) / 2 + 0.5) * wbar, vals, wbar, label=lab, color=colors[g])
    band = out[(out.test == "comp_v5_active")].set_index("year")
    for k, y in enumerate(yrs):
        if y in band.index:
            ax.plot([k - 0.42, k + 0.42], [band.loc[y, "band_hi"]] * 2, color="grey", lw=0.8, ls="--")
            ax.plot([k - 0.42, k + 0.42], [band.loc[y, "band_lo"]] * 2, color="grey", lw=0.8, ls="--")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs)
    ax.set_ylabel("偏差统计量 b = std(实现/预测)")
    ax.set_title(f"偏差统计量(模型 {PRIMARY};5 相位平均;虚线 = 1 ± sqrt(2/T))", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    pos = series["test_pos"]
    idx = dates[pos]
    pred = pd.Series(series["predC"] * ANN5, index=idx)
    real = pd.Series(series["realC"], index=idx)
    roll = real.rolling(60, min_periods=40).std() * ANN5
    ax.plot(idx, pred, color="#2a78d6", lw=2, label="预测年化主动风险(事前)")
    ax.plot(idx, roll, color="#eb6834", lw=2, label="实现(5 日主动收益滚动 60 窗口 std,年化)")
    ax.set_title(f"COMP_v5 排雷目标 − 池等权:预测 vs 实现(模型 {PRIMARY},信号样本内)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    f = REPORTS / "risk_bias.png"
    fig.savefig(f, dpi=140)
    log(f"图已写 {f}")


# ================================================================ main
def write_check(parts: list[pd.DataFrame], drop: tuple = ("riskreg_vs_ledger",)):
    """risk_move_check.csv 按 section 增量写:本次跑到的 section 覆盖,其余保留(drop 为废弃的旧 section)。"""
    f = LEDGER / "risk_move_check.csv"
    new = pd.concat(parts, ignore_index=True)
    if f.exists():
        old = pd.read_csv(f)
        old = old[~old.section.isin(set(new.section.unique()) | set(drop))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(f, index=False)
    log(f"已写 {f}")


def raw_outlier_evidence(p, day: str = "2021-05-06") -> dict:
    """v2 修复的留证:该日估计池内原始 fu_leverage 的极值、中位数与稳健标准差。"""
    FA.build_all()
    u = p["universe"].loc[pd.Timestamp(day)]
    x = FA.REGISTRY["fu_leverage"][1](p).reindex(index=p.dates, columns=p.codes) \
        .loc[pd.Timestamp(day)].where(u).dropna()
    med = float(x.median()); s_rob = float(1.4826 * (x - med).abs().median())
    xs = x.sort_values()
    return dict(day=day, descriptor="fu_leverage(= −负债/资产,非金融)", n=int(len(x)),
                min_code=str(xs.index[0]), min_value=float(xs.iloc[0]),
                second_min_code=str(xs.index[1]), second_min_value=float(xs.iloc[1]),
                median=med, robust_std=s_rob, raw_std=float(x.std()),
                clip_bound_k5=med - 5 * s_rob)


# 风险模型版本(见 barra.py 模块说明与 RiskModel.VARIANTS):v1 预注册原版暴露 / v2 稳健截尾 / v3 再加 Size 不截尾
RM_VERSIONS = ["v1", "v2", "v3"]
PRIMARY = "v3"          # 看 v2/v3 偏差结果之前按暴露度数据质量判据定下的默认版本


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "move", "build", "bias"])
    ap.add_argument("--force", action="store_true", help="忽略缓存重建风险模型")
    args = ap.parse_args()
    p = load_panel()
    if args.stage in ("all", "move"):
        write_check([stage_move(p)])
        p = load_panel()                    # 丢掉原版暴露度留在面板里的派生缓存
    if args.stage in ("all", "build", "bias"):
        models, summ_all, parts = {}, {}, []
        for ver in RM_VERSIONS:
            log(f"==== 风险模型 {ver} {BR.RiskModel.VARIANTS[ver]}")
            rm, cmp_df, summ = stage_build(p, args.force, ver)
            models[ver], summ_all[ver] = rm, summ
            parts.append(cmp_df)
        write_check(parts)
        summ_all["raw_outlier_evidence"] = raw_outlier_evidence(p)
        json.dump(summ_all, open(LEDGER / "risk_model_summary.json", "w"), ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if args.stage in ("all", "bias"):
        outs, dets, dgs = [], [], []
        for ver, rm_v in models.items():
            log(f"==== 偏差检验 {ver}")
            out_v, det_v, series_v, ctx_v = stage_bias(p, rm_v)
            out_v.insert(0, "model", ver); det_v.insert(0, "model", ver)
            outs.append(out_v); dets.append(det_v)
            dg_v = stage_diag(p, rm_v, ctx_v)
            dg_v.insert(0, "model", ver)
            dgs.append(dg_v)
            if ver == PRIMARY:
                rm, out, series, ctx = rm_v, out_v, series_v, ctx_v
            else:
                del ctx_v
        pd.concat(outs, ignore_index=True).to_csv(LEDGER / "risk_bias_stats.csv", index=False)
        pd.concat(dets, ignore_index=True).to_csv(LEDGER / "risk_bias_random_detail.csv", index=False)
        log("已写 risk_bias_stats.csv / risk_bias_random_detail.csv(v1/v2/v3)")
        # 以下只对主版本(PRIMARY):逐窗口明细、归因、图(诊断三版都跑)
        tp, dts = ctx["test_pos"], rm.dates
        wins = [pd.DataFrame(dict(test=t, decision=dts[pos], pred_vol5=pr, real5=re, z=re / pr))
                for t, pos, pr, re in [("Country", tp, np.sqrt(rm.cov[tp, 0, 0]), ctx["F5"][tp, 0]),
                                       ("pool_total", tp, ctx["predP"], ctx["realP"]),
                                       ("comp_v5_active", tp, ctx["predC"], ctx["realC"]),
                                       ("gbt_top20_active", ctx["posT"], ctx["predT"], ctx["realT"])]]
        pd.concat(wins, ignore_index=True).assign(model=PRIMARY).to_csv(
            LEDGER / "risk_bias_windows.csv", index=False)
        dg = pd.concat(dgs, ignore_index=True)
        dg.to_csv(LEDGER / "risk_bias_diagnostics.csv", index=False)
        log("已写 risk_bias_diagnostics.csv(v1/v2/v3)")
        print(dg[dg.model == PRIMARY][["test", "portfolio", "year", "bias", "value", "autocorr_ratio", "kurtosis"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        summ_attr, fac_ann = stage_attr(rm, ctx["W"], ctx["pool"], ctx["R"], ctx["U"])
        summ_attr.insert(0, "model", PRIMARY)
        summ_attr.to_csv(LEDGER / "risk_attribution_comp_v5.csv", index=False)
        fac_ann.to_csv(LEDGER / "risk_attribution_comp_v5_factors.csv")
        log("已写 risk_attribution_comp_v5*.csv")
        plot(out, series, rm.dates)
        allb = pd.concat(outs, ignore_index=True)
        show = allb[allb.portfolio.isin(["Country", "__风格平均", "__行业平均", "median_of_200", "Leverage 杠杆",
                                         "EarningsYield 盈利", "Size 市值", "NonlinSize 非线性市值"])
                    | allb.test.isin(["comp_v5_active", "gbt_top20_active", "pool_total"])]
        print(show.pivot_table(index=["test", "portfolio"], columns=["model", "year"], values="bias")
              .round(3).to_string())
        print(summ_attr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log("完成")


if __name__ == "__main__":
    main()
