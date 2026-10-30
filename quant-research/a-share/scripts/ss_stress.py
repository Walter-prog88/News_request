"""反向压力测试与假设情景(前缀 ss;2026-10-11)。报告 research/SCENARIO_STRESS_TEST.md。

这是**风险刻画**,不做收益预测,不改任何策略参数。情景与参数网格由主控预先指定,全部报告,不增删、不按结果挑选。
只用 <= 2025-12-31 的数据(长历史面板与第一步每日序列本身截至 2025-12-31)。

阶段(默认全跑;python3 scripts/ss_stress.py [anchors reverse scale scenarios checks figure]):
  anchors    情景 B 的历史锚:换手率五分位溢价与排雷暴露、20 日块回归、2015-07 锁仓比例、最小市值档权重、
             2016-01 熔断、924 / 2015H1 / 2019 初反弹的相对表现 -> research/ledger/ss_anchors.csv
  reverse    A.1 合成危机路径的反向压力测试(确定性路径 + 波动率敏感性)      -> research/ledger/ss_reverse.csv
  scale      A.2 2008 / 2015 实际路径放大 k 倍                                -> research/ledger/ss_scale.csv
  scenarios  B 假设情景 S1-S6                                                  -> research/ledger/ss_scenarios.csv
  checks     机械复核(f = 1 恒等、跳空形状 = f·D、与第二步引擎 O1 静态降仓对账)  -> research/ledger/ss_checks.csv
  figure     reports/ss_reverse_frontier.png

账户记账(A、B 共用,机械):账户 = 股票部分 + 现金,起点净值 1 = 峰值,股票部分初始占比 f。
  rb5 = 每 5 个交易日(第 5、10、…… 天收盘,phase 0;敏感性 phase 0..4)把股票部分再平衡回 f;
        当天"股票部分不可成交"时顺延到下一个可成交日。
  bh  = 买入持有,仓位随涨跌漂移,从不交易。
  现金收益 0%(主)/ 2% 年化(按 1/242 逐日复利)。回撤 = 账户净值相对运行峰值的最大跌幅。
  不计账户层再平衡的交易成本(第二步引擎版 O1 计了,对账见 ss_checks.csv)。

审查后补充(2026-10-11,均为已有结论的口径敏感性,不是新情景,不改策略参数):
  P0  ss_scale.csv kind=bh_start:买入持有对起点的敏感性(2006-01 ~ 2007-10 每月起点,全部报告)
  P1  ss_anchors.csv S3:主锚改为"停牌或开盘跌停"(lock_sell_*),原"含开盘涨停"口径改名 lock_anylimit_* 作敏感性;
      情景 S3 的池子换算比改用 lock_ratio_sell_*
  P2  ss_reverse.csv kind=dstar_buyable:跌停日仍可买入补仓的约定敏感性;
      ss_scale.csv margin:引擎谷底实际仓位、20 日 rb5 余量、k 比例放大的引擎修正、引擎口径的排雷峰谷回撤;
      ss_anchors.csv S1:平均超额与 β·LS 的分解(截距证据)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor import lh_engine as LE
from afactor.config import LEDGER, REPORTS, RESEARCH_END, TMP

TD = 242
DAILY1 = TMP / "lh_step1_daily.parquet"
SSC = TMP / "ss_cache"
LOG = ROOT / "logs" / "ss_stress.log"

F_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00)
F_B = (0.50, 0.70, 1.00)
CASH_GRID = (0.0, 0.02)
D_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
T_GRID = (20, 60, 250)
SHAPES = ("linear", "front", "gap")
MODES = ("rb5", "bh")
K_GRID = (1.0, 1.2, 1.5, 2.0)
DD_LIMIT = 0.50
CAP = float(np.log(0.9))            # 组合层面单日最多 -10%(全部跌停),跌到这里即"不可成交"
FRONT_DAYS_FRAC, FRONT_LOSS_FRAC = 0.25, 0.60   # 前段急跌:前 25% 的交易日承担 60% 的对数跌幅(超过 -10% 的部分顺延)
D_FINE = np.round(np.arange(0.001, 0.9995, 0.001), 3)
K_FINE = np.round(np.arange(0.50, 6.0001, 0.005), 3)
HIST_WINDOWS = {"2008": ("2007-10-01", "2008-11-30"), "2015": ("2015-06-01", "2016-01-31")}  # 成交日 te 口径
VOL_PATHS = 200
VOL_D = np.round(np.arange(0.005, 0.9951, 0.005), 3)
SEED = 20261011
# 审查补充(2026-10-11):买入持有对起点的敏感性。起点 = 2006-01 ~ 2007-10 每月第一个成交日(22 个,全部报告),终点同 2008 窗口
BH_START_MONTHS = [str(p) for p in pd.period_range("2006-01", "2007-10", freq="M")]
STEP2_RUNS = TMP / "lh_step2_runs"

SHAPE_CN = {"linear": "线性阴跌", "front": "前段急跌后段阴跌", "gap": "连续跌停跳空"}
MODE_CN = {"rb5": "每5日再平衡", "bh": "买入持有"}


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(s + "\n")


# ====================================================================== 账户记账
def simulate(R, f: float, cash: float = 0.0, mode: str = "rb5", phase: int = 0, untradable=None):
    """R: (B, T) 股票部分日收益。返回 (最大回撤[B], 期末净值[B])。"""
    R = np.atleast_2d(np.asarray(R, np.float64))
    B, T = R.shape
    U = np.zeros((B, T), bool) if untradable is None else np.broadcast_to(np.atleast_2d(untradable), (B, T))
    cd = cash / TD
    S = np.full(B, f, np.float64)
    C = np.full(B, 1.0 - f, np.float64)
    peak = np.ones(B)
    mdd = np.zeros(B)
    pend = np.zeros(B, bool)
    for i in range(T):
        S = S * (1.0 + R[:, i])
        C = C * (1.0 + cd)
        nav = S + C
        np.maximum(peak, nav, out=peak)
        np.maximum(mdd, 1.0 - nav / peak, out=mdd)
        if mode == "rb5":
            if (i + 1 - phase) % 5 == 0:
                pend[:] = True
            do = pend & ~U[:, i]
            if do.any():
                S = np.where(do, f * nav, S)
                C = np.where(do, (1.0 - f) * nav, C)
                pend &= ~do
    return mdd, S + C


def crossing(x, y, level: float = DD_LIMIT):
    """y 首次 >= level 时的 x(线性插值)。返回 (x*, 状态, y 的最大值)。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) == 0:
        return np.nan, "无可行点", np.nan
    idx = np.flatnonzero(y >= level - 1e-12)
    if len(idx) == 0:
        return np.nan, "未达到", float(y.max())
    j = int(idx[0])
    if j == 0:
        return float(x[0]), "下界即达到", float(y.max())
    x0, x1, y0, y1 = x[j - 1], x[j], y[j - 1], y[j]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0)), "达到", float(y.max())


# ====================================================================== A.1 合成路径
def build_path(shape: str, D: float, T: int):
    """股票部分 T 天累计下跌 D 的确定性路径。返回 (日收益[T], 不可成交[T]) 或 None(不可行:要求日跌幅超过 -10%)。"""
    L = float(np.log(1.0 - D))
    if shape == "linear":
        if L / T < CAP - 1e-12:
            return None
        l = np.full(T, L / T)
        return np.expm1(l), l <= CAP + 1e-9
    if shape == "front":
        nf = int(np.ceil(FRONT_DAYS_FRAC * T))
        l = np.empty(T)
        l[:nf] = FRONT_LOSS_FRAC * L / nf
        l[nf:] = (1.0 - FRONT_LOSS_FRAC) * L / (T - nf)
        carry = 0.0
        for i in range(T):
            x = l[i] + carry
            if x < CAP:
                carry, x = x - CAP, CAP
            else:
                carry = 0.0
            l[i] = x
        if carry < -1e-12:
            return None
        return np.expm1(l), l <= CAP + 1e-9
    if shape == "gap":
        nfull = int(np.floor(L / CAP + 1e-9))
        resid = L - nfull * CAP
        m = nfull + (1 if resid < -1e-12 else 0)
        if m > T:
            return None
        l = np.zeros(T)
        l[:nfull] = CAP
        if m > nfull:
            l[nfull] = resid
        u = np.zeros(T, bool)
        u[:m] = True               # 跳空日全部不可成交(含最后一个不足 -10% 的跳空日);之后持平、可成交
        return np.expm1(l), u
    raise ValueError(shape)


def path_matrix(shape: str, T: int, Ds):
    R = np.zeros((len(Ds), T))
    U = np.zeros((len(Ds), T), bool)
    ok = np.zeros(len(Ds), bool)
    for i, D in enumerate(Ds):
        p = build_path(shape, float(D), T)
        if p is not None:
            R[i], U[i] = p
            ok[i] = True
    return R, U, ok


def max_feasible_D(shape: str, T: int) -> float:
    return float(1.0 - 0.9 ** T)   # 三种形状的可行上限都是 T 天每天 -10%


def stage_reverse():
    t0 = time.time()
    rows = []
    for shape in SHAPES:
        for T in T_GRID:
            Rg, Ug, okg = path_matrix(shape, T, D_GRID)
            Rf, Uf, okf = path_matrix(shape, T, D_FINE)
            n_untr = Ug.sum(axis=1)
            for f in F_GRID:
                for cash in CASH_GRID:
                    for mode in MODES:
                        phases = range(5) if mode == "rb5" else (0,)
                        # ---- 网格点(phase 0)----
                        ddg = np.full(len(D_GRID), np.nan)
                        if okg.any():
                            ddg[okg] = simulate(Rg[okg], f, cash, mode, 0, Ug[okg])[0]
                        for i, D in enumerate(D_GRID):
                            rows.append(dict(kind="grid", shape=shape, shape_cn=SHAPE_CN[shape], T=T, f=f, cash=cash,
                                             mode=mode, mode_cn=MODE_CN[mode], D=D, feasible=bool(okg[i]),
                                             untradable_days=int(n_untr[i]) if okg[i] else np.nan,
                                             account_dd=ddg[i], over_50=bool(ddg[i] >= DD_LIMIT) if okg[i] else np.nan,
                                             bh_formula_fD=f * D if okg[i] else np.nan))
                        # ---- D*:细网格(0.1pp)+ 7 点网格插值 ----
                        ds_ph = []
                        for ph in phases:
                            ddf = np.full(len(D_FINE), np.nan)
                            ddf[okf] = simulate(Rf[okf], f, cash, mode, ph, Uf[okf])[0]
                            x, st, ymax = crossing(D_FINE, ddf)
                            ds_ph.append((x, st, ymax))
                        x0, st0, ymax0 = ds_ph[0]
                        xg, stg, _ = crossing(np.array(D_GRID), ddg)
                        mf = max_feasible_D(shape, T)
                        if st0 == "达到":
                            status = "可达"
                        elif mf < 0.999:
                            status = f"可行域内不可达(D≤{mf:.1%},账户最多{ymax0:.1%})"
                        else:
                            status = f"不可达(D→100%,账户最多{ymax0:.1%})"
                        xs = np.array([d[0] for d in ds_ph], float)
                        rows.append(dict(kind="dstar", shape=shape, shape_cn=SHAPE_CN[shape], T=T, f=f, cash=cash,
                                         mode=mode, mode_cn=MODE_CN[mode], dstar=x0, dstar_status=status,
                                         dstar_grid_interp=xg,
                                         dstar_grid_status=("<30%" if stg == "下界即达到" else
                                                            (">90%或不可行" if stg != "达到" else "插值")),
                                         dstar_phase_min=float(np.nanmin(xs)) if np.isfinite(xs).any() else np.nan,
                                         dstar_phase_max=float(np.nanmax(xs)) if np.isfinite(xs).any() else np.nan,
                                         n_phase_reached=int(np.isfinite(xs).sum()), n_phases=len(xs),
                                         max_feasible_D=mf,
                                         dstar_cont_rebal_formula=(1 - 0.5 ** (1 / f)),
                                         dstar_bh_formula=(0.5 / f if f > 0.5 else np.nan)))
        log(f"[reverse] {shape} 完成 {time.time() - t0:.1f}s")
    rows += reverse_buyable()
    rows += reverse_vol()
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "ss_reverse.csv", index=False)
    log(f"[reverse] ss_reverse.csv {len(df)} 行,{time.time() - t0:.1f}s")
    return df


def reverse_buyable():
    """审查补充(P2):约定敏感性,不是新情景。主口径把 -10% 日按"双向不可成交"处理(账户不在跌停日补仓);
    A 股跌停股可买不可卖,账户层"调回 f"可以挂跌停价买入。这里令所有日子都可再平衡(untradable = None),
    只报 rb5、现金 0 的 D*(相位 0 与相位 0..4 范围)。买入持有不受这个约定影响。"""
    rows = []
    for shape in SHAPES:
        for T in T_GRID:
            Rf, _, okf = path_matrix(shape, T, D_FINE)
            for f in F_GRID:
                xs = []
                ymax0 = np.nan
                for ph in range(5):
                    ddf = np.full(len(D_FINE), np.nan)
                    ddf[okf] = simulate(Rf[okf], f, 0.0, "rb5", ph, None)[0]
                    x, st, ymax = crossing(D_FINE, ddf)
                    xs.append(x)
                    if ph == 0:
                        ymax0 = ymax
                xs = np.array(xs, float)
                mf = max_feasible_D(shape, T)
                status = "可达" if np.isfinite(xs[0]) else (f"可行域内不可达(D≤{mf:.1%},账户最多{ymax0:.1%})"
                                                          if mf < 0.999 else f"不可达(账户最多{ymax0:.1%})")
                rows.append(dict(kind="dstar_buyable", shape=shape, shape_cn=SHAPE_CN[shape], T=T, f=f, cash=0.0,
                                 mode="rb5", mode_cn="每5日再平衡(跌停日也补仓)", dstar=xs[0], dstar_status=status,
                                 dstar_phase_min=float(np.nanmin(xs)) if np.isfinite(xs).any() else np.nan,
                                 dstar_phase_max=float(np.nanmax(xs)) if np.isfinite(xs).any() else np.nan,
                                 n_phase_reached=int(np.isfinite(xs).sum()), n_phases=5, max_feasible_D=mf,
                                 dstar_cont_rebal_formula=(1 - 0.5 ** (1 / f))))
    log(f"[reverse] 跌停日可补仓敏感性 {len(rows)} 行")
    return rows


def crisis_sigma() -> float:
    d = pd.read_parquet(DAILY1)
    te = te_index(d.index)
    a, b = HIST_WINDOWS["2008"]
    m = (te >= pd.Timestamp(a)) & (te <= pd.Timestamp(b))
    return float(d.loc[m, "strat_main"].std())


def reverse_vol():
    """波动率敏感性(只用 cash 0、phase 0、linear / front):在确定性对数路径上叠加日波动 σ(= 排雷 2008 窗口实测),
    噪声做成离散布朗桥(逐路径去均值),终点累计跌幅仍恰好是 D。不施加 -10% 上限、全部可成交。
    报告:账户回撤中位数 = 50% 与 90 分位 = 50% 时的 D*。共同随机数(同一组噪声用于所有 D)。"""
    t0 = time.time()
    sig = crisis_sigma()
    rng = np.random.default_rng(SEED)
    rows = []
    for T in T_GRID:
        E = rng.standard_normal((VOL_PATHS, T))
        E = (E - E.mean(axis=1, keepdims=True)) * sig
        for shape in ("linear", "front"):
            base = []
            for D in VOL_D:
                L = np.log(1.0 - D)
                if shape == "linear":
                    l = np.full(T, L / T)
                else:
                    nf = int(np.ceil(FRONT_DAYS_FRAC * T))
                    l = np.empty(T)
                    l[:nf] = FRONT_LOSS_FRAC * L / nf
                    l[nf:] = (1.0 - FRONT_LOSS_FRAC) * L / (T - nf)
                base.append(l)
            base = np.array(base)                                   # (nD, T)
            Lmat = base[:, None, :] + E[None, :, :]                 # (nD, P, T)
            R = np.expm1(Lmat).reshape(-1, T)
            stock_nav = np.exp(np.cumsum(Lmat, axis=2))
            stock_dd = 1 - (stock_nav / np.maximum.accumulate(np.concatenate(
                [np.ones(stock_nav.shape[:2] + (1,)), stock_nav], axis=2), axis=2)[:, :, 1:]).min(axis=2)
            for f in F_GRID:
                for mode in MODES:
                    dd = simulate(R, f, 0.0, mode, 0)[0].reshape(len(VOL_D), VOL_PATHS)
                    med = np.median(dd, axis=1)
                    p90 = np.quantile(dd, 0.9, axis=1)
                    x50, s50, _ = crossing(VOL_D, med)
                    x90, s90, _ = crossing(VOL_D, p90)
                    j = int(np.argmin(np.abs(VOL_D - x50))) if np.isfinite(x50) else None
                    rows.append(dict(kind="dstar_vol", shape=shape, shape_cn=SHAPE_CN[shape], T=T, f=f, cash=0.0,
                                     mode=mode, mode_cn=MODE_CN[mode], vol_sigma_daily=sig, vol_paths=VOL_PATHS,
                                     dstar_vol_median=x50, dstar_vol_p90=x90,
                                     dstar_status=("可达" if s50 == "达到" else s50),
                                     stock_dd_median_at_dstar=(float(np.median(stock_dd[j])) if j is not None else np.nan)))
    log(f"[reverse] 波动率敏感性 σ={sig:.4f} 完成 {time.time() - t0:.1f}s")
    return rows


# ====================================================================== A.2 历史路径放大
_DATES = None


def te_index(idx: pd.DatetimeIndex) -> pd.Series:
    """信号日 t -> 成交日 te = 下一个交易日(长历史面板日历)。"""
    global _DATES
    if _DATES is None:
        _DATES = LE.load_inputs().dates
    pos = _DATES.get_indexer(idx)
    assert (pos >= 0).all()
    return pd.Series(_DATES[np.minimum(pos + 1, len(_DATES) - 1)], index=idx)


def window_slice(d: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    te = te_index(d.index)
    m = (te >= pd.Timestamp(a)) & (te <= pd.Timestamp(b))
    return d.loc[m.to_numpy()]


def scale_ret(r: np.ndarray, k) -> np.ndarray:
    return np.expm1(np.multiply.outer(np.atleast_1d(k), np.log1p(r)))


def stock_stats(r: np.ndarray) -> tuple[float, float]:
    nav = np.cumprod(1 + r)
    dd = 1 - nav / np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return float(dd.max()), float(nav[-1] - 1)


def simulate_trace(r, f: float, cash: float = 0.0, mode: str = "rb5", phase: int = 0):
    """单条路径的逐日净值与(当日收盘、再平衡之后的)股票占比。"""
    S, C, cd = f, 1.0 - f, cash / TD
    nav = np.empty(len(r))
    share = np.empty(len(r))
    for i, x in enumerate(r):
        S *= 1 + x
        C *= 1 + cd
        v = S + C
        if mode == "rb5" and (i + 1 - phase) % 5 == 0:
            S, C = f * v, (1 - f) * v
        nav[i] = v
        share[i] = S / v
    return nav, share


def engine_gap() -> dict:
    """第二步引擎版静态降仓 O1 的全期最大回撤(含账户层调仓成本、残仓)与本脚本叠加口径(rb5,5 个账户相位平均)之差。
    全期最大回撤都落在 2008 窗口,所以只用于 2008 窗口的保守修正。"""
    d = pd.read_parquet(DAILY1).loc[LE.EVAL_START:]
    s2 = pd.read_csv(LEDGER / "lh_step2_summary.csv")
    s2 = s2[(s2.period == "full") & (s2.cash == 0) & (s2.valuation == "close")].set_index("rule")
    out = {}
    for f in F_GRID:
        rule = f"O1_f{int(round(f * 1000)):04d}"
        if rule not in s2.index:
            continue
        ov = float(np.mean([simulate(d["strat_main"].to_numpy(), f, 0.0, "rb5", ph)[0][0] for ph in range(5)]))
        e = float(-s2.loc[rule, "mdd"])
        # 审查补充(P2):引擎在其谷底(信号日 mdd_trough)的实际股票仓位(5 相位平均),
        # 读第二步运行缓存 lh_step2_runs/O1_fXXXX.pkl;缺失时退回全期平均 net_exposure_avg
        trough = pd.Timestamp(s2.loc[rule, "mdd_trough"])
        expo, src = float(s2.loc[rule, "net_exposure_avg"]), "lh_step2_summary.csv net_exposure_avg(全期平均)"
        pk = STEP2_RUNS / f"{rule}.pkl"
        if pk.exists():
            import pickle
            with open(pk, "rb") as fh:
                run = pickle.load(fh)
            ex = pd.concat([p_["exposure"] for p_ in run["phases"]], axis=1).mean(axis=1)
            if trough in ex.index:
                expo, src = float(ex.loc[trough]), f"lh_step2_runs/{rule}.pkl 5 相位平均 exposure @ {trough.date()}"
        out[f] = (e, e - ov, expo, src)
    return out


def margin_path(dd0: float, s0: float, f: float, x: float, n: int = 20, phase: int = 0) -> float:
    """派生指标:账户已在回撤 dd0(峰值 = 1)、股票占比 s0 时,股票部分在 n 日内按等比路径再跌 x,
    期间每 5 日调回 f(phase 0 = 第 5、10、... 日收盘)。返回期间账户相对原峰值的最大回撤。"""
    x = np.asarray(x, np.float64)
    nav = 1.0 - dd0
    S = np.full(x.shape, s0 * nav)
    C = np.full(x.shape, (1.0 - s0) * nav)
    r = np.expm1(np.log1p(-x) / n)
    worst = np.full(x.shape, dd0)
    for i in range(n):
        S = S * (1 + r)
        v = S + C
        worst = np.maximum(worst, 1.0 - v)
        if (i + 1 - phase) % 5 == 0:
            S, C = f * v, (1 - f) * v
    return worst


def margin_20d(dd0: float, s0: float, f: float) -> tuple[float, float, float]:
    """margin_path 使账户恰好到 50% 的 x(相位 0、相位 0..4 的最小 / 最大)。账户已 >= 50% 时返回 nan。"""
    if dd0 >= DD_LIMIT:
        return np.nan, np.nan, np.nan
    xs = np.round(np.arange(0.0005, 0.9995, 0.0005), 4)
    out = []
    for ph in range(5):
        out.append(crossing(xs, margin_path(dd0, s0, f, xs, 20, ph))[0])
    out = np.array(out, float)
    return out[0], float(np.nanmin(out)) if np.isfinite(out).any() else np.nan, \
        float(np.nanmax(out)) if np.isfinite(out).any() else np.nan


def bh_start_rows(d: pd.DataFrame, end: str) -> list:
    """审查补充(P0):'不再平衡 f ≤ 50% 不破线'依赖'危机开始时股票占比恰为 f'。真实的买入持有在牛市里占比会漂高。
    起点 = 2006-01 ~ 2007-10 每月第一个成交日(全部 22 个),建仓占比 f,一直持有到 2008 窗口终点;
    同时报 rb5(相位 0..4)作对照。不是新情景,是 A.2 买入持有结论的起点敏感性。"""
    te_all = te_index(d.index)
    rows = []
    for mon in BH_START_MONTHS:
        a = pd.Period(mon).start_time
        m = ((te_all >= a) & (te_all <= pd.Timestamp(end))).to_numpy()
        r = d.loc[m, "strat_main"].to_numpy(np.float64)
        te = te_all[m]
        sdd, scum = stock_stats(r)
        for f in F_GRID:
            nav, share = simulate_trace(r, f, 0.0, "bh", 0)
            pk = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
            dd = 1 - nav / pk
            j = int(np.argmax(dd))
            jp = int(np.argmax(nav[: j + 1])) if nav[: j + 1].max() > 1.0 else -1
            rb = [float(simulate(r, f, 0.0, "rb5", ph)[0][0]) for ph in range(5)]
            rows.append(dict(kind="bh_start", window="2008", te_start=str(te.iloc[0].date()), te_end=str(te.iloc[-1].date()),
                             n_days=len(r), series="strat_main", f=f, cash=0.0, mode="bh", mode_cn=MODE_CN["bh"],
                             stock_mdd=sdd, stock_cum=scum, account_mdd=float(dd[j]), over_50=bool(dd[j] >= DD_LIMIT),
                             trough_te=str(te.iloc[j].date()),
                             peak_te=(str(te.iloc[jp].date()) if jp >= 0 else str(te.iloc[0].date()) + "(起点)"),
                             stock_share_at_peak=(float(share[jp]) if jp >= 0 else f),
                             stock_share_max=float(share.max()), stock_share_at_trough=float(share[j]),
                             rb5_account_mdd=rb[0], rb5_account_mdd_phase_min=min(rb), rb5_account_mdd_phase_max=max(rb)))
    log(f"[scale] 买入持有起点敏感性 {len(rows)} 行")
    return rows


def stage_scale():
    t0 = time.time()
    d = pd.read_parquet(DAILY1)
    series = ["strat_main"] + [f"strat_main_ph{i}" for i in range(5)]
    rows = []
    for wname, (a, b) in HIST_WINDOWS.items():
        w = window_slice(d, a, b)
        te = te_index(w.index)
        base = {s: w[s].to_numpy(np.float64) for s in series}
        sdd1, scum1 = stock_stats(base["strat_main"])
        # ---- 网格(主序列 strat_main,phase 0)----
        for k in K_GRID:
            r = scale_ret(base["strat_main"], k)[0]
            sdd, scum = stock_stats(r)
            for f in F_GRID:
                for cash in CASH_GRID:
                    for mode in MODES:
                        dd = simulate(r, f, cash, mode, 0)[0][0]
                        rows.append(dict(kind="grid", window=wname, te_start=str(te.iloc[0].date()),
                                         te_end=str(te.iloc[-1].date()), n_days=len(w), series="strat_main",
                                         k=k, f=f, cash=cash, mode=mode, mode_cn=MODE_CN[mode],
                                         stock_mdd=sdd, stock_cum=scum, account_mdd=dd,
                                         over_50=bool(dd >= DD_LIMIT)))
        # ---- k*:细网格,主 = strat_main + 账户 phase 0;最差 = 6 条序列 x 5 个账户相位 ----
        Rk = {s: scale_ret(base[s], K_FINE) for s in series}
        for f in F_GRID:
            for cash in CASH_GRID:
                for mode in MODES:
                    phases = range(5) if mode == "rb5" else (0,)
                    res = {}
                    for s in series:
                        for ph in phases:
                            dd = simulate(Rk[s], f, cash, mode, ph)[0]
                            res[(s, ph)] = crossing(K_FINE, dd)
                    x0, st0, ymax0 = res[("strat_main", 0)]
                    xs = {key: v[0] for key, v in res.items()}
                    fin = {key: v for key, v in xs.items() if np.isfinite(v)}
                    worst_key = min(fin, key=fin.get) if fin else None
                    mph = [xs[("strat_main", ph)] for ph in phases]
                    sdd_k = 1 - (1 - sdd1) ** x0 if np.isfinite(x0) else np.nan
                    kg = crossing(np.array(K_GRID), np.array(
                        [r_["account_mdd"] for r_ in rows if r_["kind"] == "grid" and r_["window"] == wname
                         and r_["f"] == f and r_["cash"] == cash and r_["mode"] == mode]))
                    rows.append(dict(kind="kstar", window=wname, te_start=str(te.iloc[0].date()),
                                     te_end=str(te.iloc[-1].date()), n_days=len(w), series="strat_main",
                                     f=f, cash=cash, mode=mode, mode_cn=MODE_CN[mode],
                                     kstar=x0, kstar_status=("可达" if st0 == "达到" else
                                                             (f"k≤{K_FINE[-1]:.0f} 内未达到(账户最多 {ymax0:.1%})"
                                                              if st0 == "未达到" else st0)),
                                     kstar_grid_interp=kg[0] if kg[1] == "达到" else np.nan,
                                     kstar_grid_status={"达到": "插值", "下界即达到": "k=1 已触及(<1.0)"}.get(kg[1], ">2.0"),
                                     kstar_main_phase_min=float(np.nanmin(mph)) if np.isfinite(mph).any() else np.nan,
                                     kstar_main_phase_max=float(np.nanmax(mph)) if np.isfinite(mph).any() else np.nan,
                                     kstar_worst=(fin[worst_key] if worst_key else np.nan),
                                     kstar_worst_where=(f"{worst_key[0]}|账户phase{worst_key[1]}" if worst_key else ""),
                                     stock_mdd_k1=sdd1, stock_cum_k1=scum1,
                                     stock_mdd_at_kstar=sdd_k,
                                     extra_stock_loss_vs_k1=(1 - (1 - sdd_k) / (1 - sdd1)) if np.isfinite(sdd_k) else np.nan))
        # ---- 谷底一次性冲击余量(派生指标,不是新情景):历史路径走到账户谷底时,股票部分再一次性跌 x 使账户回撤恰好 50% ----
        eng = engine_gap() if wname == "2008" else {}
        for f in F_GRID:
            for cash in CASH_GRID:
                for mode in MODES:
                    nav, share = simulate_trace(base["strat_main"], f, cash, mode, 0)
                    pk = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
                    dd = 1 - nav / pk
                    j = int(np.argmax(dd))
                    x = (1 - (1 - DD_LIMIT) / (1 - dd[j])) / share[j] if dd[j] < DD_LIMIT else np.nan
                    rec = dict(kind="margin", window=wname, te_start=str(te.iloc[0].date()), te_end=str(te.iloc[-1].date()),
                               n_days=len(w), series="strat_main", f=f, cash=cash, mode=mode, mode_cn=MODE_CN[mode],
                               account_mdd=float(dd[j]), trough_te=str(te_index(w.index[[j]]).iloc[0].date()),
                               stock_share_at_trough=float(share[j]),
                               lump_margin=float(x) if np.isfinite(x) else np.nan,
                               lump_status=("可承受" if np.isfinite(x) and x <= 1 else
                                            ("k=1 已破线" if dd[j] >= DD_LIMIT else "股票清零也不破线")))
                    if mode == "rb5":
                        # 审查补充(P2):谷底之后股票部分在 20 日内等比再跌 x、期间每 5 日调回 f,账户恰好到 50% 的 x。
                        # 与 S2(20 日、rb5)同口径;瞬时口径 lump_margin 对 S2 偏乐观
                        m0, mlo, mhi = margin_20d(float(dd[j]), float(share[j]), f)
                        rec.update(margin_20d_rb5=m0, margin_20d_rb5_phase_min=mlo, margin_20d_rb5_phase_max=mhi)
                    if eng and mode == "rb5" and cash == 0.0 and f in eng:
                        e_mdd, gap, e_expo, e_src = eng[f]
                        rec["engine_O1_mdd"] = e_mdd
                        rec["engine_minus_overlay_pp"] = gap * 100
                        rec["engine_exposure_at_trough"] = e_expo
                        rec["engine_exposure_src"] = e_src
                        # 审查修正(P2):谷底余量按引擎谷底的实际仓位(而不是名义 f)计算;按 f 的旧值保留为 _at_f
                        ok_e = e_mdd < DD_LIMIT
                        rec["lump_margin_engine"] = ((1 - (1 - DD_LIMIT) / (1 - e_mdd)) / e_expo) if ok_e else np.nan
                        rec["lump_margin_engine_at_f"] = ((1 - (1 - DD_LIMIT) / (1 - e_mdd)) / f) if ok_e else np.nan
                        m0, mlo, mhi = margin_20d(e_mdd, e_expo, f) if ok_e else (np.nan,) * 3
                        rec.update(margin_20d_rb5_engine=m0, margin_20d_rb5_engine_phase_min=mlo,
                                   margin_20d_rb5_engine_phase_max=mhi)
                        ddk = simulate(Rk["strat_main"], f, 0.0, "rb5", 0)[0]
                        # 主修正:k = 1 时的常数差。k 增大时调仓量与残仓可能更多,所以它偏乐观(不是严格上界);
                        # 敏感性:差距随 k 线性放大(gap·k)
                        ka = crossing(K_FINE, ddk, DD_LIMIT - gap)[0]
                        rec["kstar_engine_adj"] = ka
                        rec["kstar_engine_adj_gap_prop_k"] = crossing(K_FINE, ddk + gap * K_FINE, DD_LIMIT)[0]
                        rec["stock_mdd_at_kstar_engine"] = (1 - (1 - sdd1) ** ka) if np.isfinite(ka) else np.nan
                    rows.append(rec)
        # ---- 审查补充(P0):买入持有对起点的敏感性(只 2008;起点 = 每月第一个成交日,终点 = 2008 窗口终点)----
        if wname == "2008":
            rows += bh_start_rows(d, b)
        log(f"[scale] {wname} 完成 {time.time() - t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "ss_scale.csv", index=False)
    log(f"[scale] ss_scale.csv {len(df)} 行")
    return df


# ====================================================================== B 锚
def _cache(name, fn):
    SSC.mkdir(parents=True, exist_ok=True)
    p = SSC / name
    if p.exists():
        return np.load(p) if p.suffix == ".npy" else pd.read_parquet(p)
    out = fn()
    if p.suffix == ".npy":
        np.save(p, out)
    else:
        out.to_parquet(p)
    return out


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    b = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]
    res = np.asarray(y, float) - X @ b
    r2 = 1 - res.var() / np.asarray(y, float).var()
    return b, r2


def stage_anchors():
    t0 = time.time()
    X = LE.load_inputs()
    ad = LE.adapter(X)
    U = LE.pool(X)
    Uv = U.to_numpy()
    W = _cache("W_S1.npy", lambda: np.asarray(LE.weights(X, "S1"), np.float32))
    d = pd.read_parquet(DAILY1)
    rows = []

    def add(scen, item, value, unit="", note=""):
        rows.append(dict(scenario=scen, item=item, value=value, unit=unit, note=note))

    ev = d.loc[LE.EVAL_START:]
    ex_b1 = ev["strat_main"] - ev["bench_b1"]
    ex_fpg = ev["strat_main"] - ev["strat_fullpool_ew_gross"]

    # ---------------- S1:换手率五分位(池内,按 S1 = -turn20 排序;Q1 = 换手最低)----------------
    def quint():
        S = X.arr("score_S1")
        sc = np.where(Uv, S, np.nan)
        rk = pd.DataFrame(sc).rank(axis=1, pct=True, ascending=False).to_numpy()
        out = {}
        for q in range(5):
            m = (rk > q / 5) & (rk <= (q + 1) / 5) & Uv
            out[f"Q{q + 1}"] = LE.bench(ad, pd.DataFrame(m, index=U.index, columns=U.columns))
        return pd.DataFrame(out)
    Q = _cache("quintiles_S1.parquet", quint).reindex(d.index).loc[LE.EVAL_START:]
    ls = Q["Q1"] - Q["Q5"]
    P = float(ls.mean() * TD)
    for c in Q:
        add("S1", f"{c}_ann_geo", LE.ann_geo(Q[c]), "年化", "池内换手率五分位等权,5 日同频再平衡,不计成本(与 B1 同引擎)")
    add("S1", "LS_Q1minusQ5_arith_ann", P, "年化", "低换手 Q1 - 高换手 Q5 日差 x 242(线性近似用这个口径)")
    add("S1", "LS_geo_diff", LE.ann_geo(Q["Q1"]) - LE.ann_geo(Q["Q5"]), "年化", "几何年化之差(含 Q5 波动拖累,偏大)")
    add("S1", "LS_nw_t5", LE.nw_t(ls.to_numpy(), 5), "t")
    lsy = ls.groupby(ls.index.year).sum()
    add("S1", "LS_worst_calendar_year", float(lsy.min()), "年和", f"最差年份 {int(lsy.idxmin())}")
    add("S1", "LS_n_negative_years", int((lsy < 0).sum()), "年", ",".join(str(y) for y in lsy.index[lsy < 0]))
    for n in (2, 3):
        rs = lsy.rolling(n).sum()
        add("S1", f"LS_worst_rolling_{n}y_sum", float(rs.min()), f"{n}年和", f"截至 {int(rs.idxmin())}")
    lsr = ls.rolling(TD).sum()
    add("S1", "LS_worst_rolling_242d_sum", float(lsr.min()), "242日和", f"截至 {lsr.idxmin().date()}")
    for tag, ex in (("b1", ex_b1), ("fpg", ex_fpg)):
        b, r2 = ols(ex.to_numpy(), [ls.to_numpy()])
        add("S1", f"beta_excess_{tag}_on_LS", float(b[1]), "", f"日超额(对 {tag})对 LS 日收益回归,R² {r2:.3f}")
        add("S1", f"alpha_excess_{tag}_ann", float(b[0] * TD), "年化", "截距 x 242")
        # 审查补充(P2):"唯一收益来源"的证据是截距 ≈ 0 或为负(β·LS 已解释全部平均超额),不是 R²
        mx = float(ex.mean() * TD)
        add("S1", f"mean_excess_{tag}_arith_ann", mx, "年化", f"排雷日超额(对 {tag})均值 x 242(= β·LS 年均 + 截距)")
        add("S1", f"beta_x_LS_{tag}_ann", float(b[1] * P), "年化", "β x LS 年均:LS 暴露贡献的平均超额")
        add("S1", f"share_mean_excess_from_LS_{tag}", float(b[1] * P / mx) if mx != 0 else np.nan, "",
            "β·LS 年均 / 平均超额;> 1 表示截距为负,LS 暴露之外没有正的平均超额")
    s1 = pd.read_csv(LEDGER / "lh_step1_summary.csv")
    e_geo = float(s1[(s1.config == "main") & (s1.period == "full")]["excess_geo"].iloc[0])
    add("S1", "excl_excess_geo_vs_b1_full", e_geo, "年化", "lh_step1_summary.csv main/full(排雷自身历史超额)")

    # ---------------- S2:20 日非重叠块回归 + W1 锚 ----------------
    g = np.arange(len(ev)) // 20
    blk = lambda s: (1 + ev[s]).groupby(g).prod() - 1
    Bk = pd.DataFrame({"s": blk("strat_main"), "b": blk("bench_b1"), "fpg": blk("strat_fullpool_ew_gross"),
                       "c": blk("idx_CSI1000_oo")})
    Bk = Bk.iloc[:-1] if (len(ev) % 20) else Bk
    for tag, col in (("b1", "b"), ("fpg", "fpg")):
        b, r2 = ols((Bk.s - Bk[col]).to_numpy(), [Bk.c.to_numpy(), (Bk[col] - Bk.c).to_numpy()])
        add("S2", f"blk20_{tag}_intercept", float(b[0]), "每20日", f"排雷 20 日超额 = a + bm·中证1000 + bs·(池 - 中证1000),R² {r2:.3f},n={len(Bk)}")
        add("S2", f"blk20_{tag}_b_mkt", float(b[1]), "")
        add("S2", f"blk20_{tag}_b_spread", float(b[2]), "")
    # 历史上最差的 20 日:池等权绝对收益、池等权相对中证1000(滚动 20 个信号日复利,终点 = 窗口最后一个信号日)
    lr = lambda s: np.log1p(ev[s]).rolling(20).sum()
    r20b = np.expm1(lr("bench_b1"))
    r20s = np.expm1(lr("bench_b1")) - np.expm1(lr("idx_CSI1000_oo"))
    r20x = np.expm1(lr("strat_main")) - np.expm1(lr("bench_b1"))
    add("S2", "worst20d_b1", float(r20b.min()), "20日收益", f"池等权 B1 滚动 20 日最差,终点信号日 {r20b.idxmin().date()}")
    add("S2", "worst20d_b1_minus_csi1000", float(r20s.min()), "20日差", f"池等权 - 中证1000 滚动 20 日最差,终点信号日 {r20s.idxmin().date()}(B1 停牌估值与中证1000 钟点差会放大单个窗口)")
    add("S2", "excess_b1_at_worst20d_b1", float(r20x.loc[r20b.idxmin()]), "20日差", "池等权最差 20 日里排雷的超额")
    wins = {"W1": ("2024-01-02", "2024-02-07"), "W3": ("2024-09-24", "2024-10-08"), "W3_plus1m": ("2024-09-24", "2024-10-31"),
            "H1_2015": ("2015-01-05", "2015-06-12"), "BULL19": ("2019-01-04", "2019-04-22"),
            "CB2016": ("2016-01-04", "2016-01-07"), "JUL2015": ("2015-07-01", "2015-07-09"), "Y2017": ("2017-01-03", "2017-12-29")}
    win_vals = {}
    for wn, (a, b_) in wins.items():
        w = window_slice(d, a, b_)
        c = {s: float((1 + w[s]).prod() - 1) for s in ("strat_main", "bench_b1", "strat_fullpool_ew_gross", "idx_CSI1000_oo")}
        win_vals[wn] = c
        sc = {"W1": "S2", "W3": "S6", "W3_plus1m": "S6", "H1_2015": "S6", "BULL19": "S6", "CB2016": "S5",
              "JUL2015": "S3", "Y2017": "S4"}[wn]
        note = f"成交日 te ∈ [{a}, {b_}],{len(w)} 个交易日,窗口复利"
        add(sc, f"{wn}_strat", c["strat_main"], "窗口收益", note)
        add(sc, f"{wn}_b1", c["bench_b1"], "窗口收益", note)
        add(sc, f"{wn}_fpg", c["strat_fullpool_ew_gross"], "窗口收益", note)
        add(sc, f"{wn}_csi1000", c["idx_CSI1000_oo"], "窗口收益", note)
        add(sc, f"{wn}_excess_b1", c["strat_main"] - c["bench_b1"], "pp")
        add(sc, f"{wn}_excess_fpg", c["strat_main"] - c["strat_fullpool_ew_gross"], "pp")
        add(sc, f"{wn}_ratio_excess_over_b1", (c["strat_main"] - c["bench_b1"]) / c["bench_b1"], "",
            "超额 / 池等权窗口收益(线性外推用)")
        add(sc, f"{wn}_ratio_excess_over_fpg", (c["strat_main"] - c["strat_fullpool_ew_gross"]) / c["strat_fullpool_ew_gross"], "")

    # ---------------- S3:2015-07 锁仓锚(固定 t0 持仓,其后 25 个交易日内最长连续不可成交天数)----------------
    # 审查修正(P1):情景 S3 讲"卖不出去"。主口径 = 停牌或开盘跌停(lock_sell_*);
    # 原口径"停牌或开盘触板(含涨停,= 引擎 dead_open)"改名 lock_anylimit_*,留作敏感性。
    # 开盘涨停可以卖出;2015-07-09/10 的反弹涨停会把跌停段与涨停段连成 ≥5 日的连续段,高估卖出侧锁仓。
    amt = X.arr("amt_r114")
    dead = X.arr("dead_open")
    fr = X.arr("frozen")
    from afactor.lh_panel import LongPanel
    _odn = LongPanel().raw("open_at_dn")
    assert _odn.shape == dead.shape and (_odn.index == X.dates).all() and (_odn.columns == X.codes).all()
    dn = _odn.to_numpy(bool) & ~fr
    Uw = Uv / np.maximum(Uv.sum(axis=1, keepdims=True), 1)
    H = 25
    dates = X.dates
    defs = {"sell": ("停牌或开盘跌停(卖出侧,主口径)", lambda s: (~(np.nan_to_num(amt[s]) > 0) | dn[s]) & ~fr[s]),
            "anylimit": ("停牌或开盘触板(含开盘涨停,= 引擎 dead_open;敏感性)",
                         lambda s: (~(np.nan_to_num(amt[s]) > 0) | dead[s]) & ~fr[s])}
    lock = []
    for t0d in pd.bdate_range("2015-06-12", "2015-07-07"):
        if t0d not in dates:
            continue
        t = int(dates.get_loc(t0d))
        te = t + 1
        rec = dict(t0=str(t0d.date()))
        for tag, (_, fn) in defs.items():
            un = fn(slice(te, te + H))
            best = np.zeros(un.shape[1], int)
            cur = np.zeros(un.shape[1], int)
            for r_ in un:
                cur = np.where(r_, cur + 1, 0)
                best = np.maximum(best, cur)
            for n in (1, 5, 10):
                rec[f"{tag}_ex_{n}"] = float(np.nansum(W[t] * (best >= n)))
                rec[f"{tag}_pool_{n}"] = float(np.nansum(Uw[t] * (best >= n)))
        lock.append(rec)
    lock = pd.DataFrame(lock)
    for tag, (desc, fn) in defs.items():
        for n in (5, 10):
            ex_c, po_c = f"{tag}_ex_{n}", f"{tag}_pool_{n}"
            add("S3", f"lock_{tag}_ge{n}d_excl_max", float(lock[ex_c].max()), "持仓权重",
                f"t0 ∈ [2015-06-12, 2015-07-07] 的排雷目标持仓中,其后 {H} 日内曾连续 ≥{n} 日不可成交({desc})的权重,取最大 t0={lock.loc[lock[ex_c].idxmax(), 't0']}")
            add("S3", f"lock_{tag}_ge{n}d_excl_mean", float(lock[ex_c].mean()), "持仓权重", "同上,各 t0 平均")
            add("S3", f"lock_{tag}_ge{n}d_pool_max", float(lock[po_c].max()), "池等权权重")
            add("S3", f"lock_ratio_{tag}_pool_over_excl_ge{n}d", float((lock[po_c] / lock[ex_c]).mean()), "",
                "池等权锁仓比例 / 排雷锁仓比例(各 t0 平均)" + (",情景里用来把排雷的 q 换算成池子的 q" if tag == "sell" else ""))
        # 单日峰值(当日目标持仓口径)
        daily = []
        for t in range(int(dates.get_loc(pd.Timestamp("2015-06-12"))), int(dates.get_loc(pd.Timestamp("2015-07-31")))):
            te = t + 1
            u1 = fn(te)
            daily.append((str(dates[te].date()), float(np.nansum(W[t] * u1)), float(np.nansum(Uw[t] * u1))))
        dmax = max(daily, key=lambda z: z[1])
        add("S3", f"daily_untradable_{tag}_peak_excl", dmax[1], "持仓权重",
            f"单日不可成交({desc})权重峰值,成交日 {dmax[0]}(当日目标持仓口径)")
        add("S3", f"daily_untradable_{tag}_peak_pool", dmax[2], "池等权权重", f"同日 {dmax[0]}")

    # ---------------- S4:最小市值 20% 档的权重 ----------------
    # 合格股:当日有成交且上市满 60 个交易日(排除新股连板:新股流通市值极小、上市初期一字连板不可成交,
    # 不剔除会让"最小 20% 档"的历史收益被新股连板污染,例如 2017 年)
    def listed60():
        from afactor.lh_panel import LongPanel
        p = LongPanel()
        dl = p.raw("days_listed").to_numpy(np.float32)
        return (np.nan_to_num(np.asarray(amt)) > 0) & (np.nan_to_num(dl) >= 60)
    L60 = _cache("elig_listed60.npy", listed60).astype(bool)

    def small_masks():
        from afactor.lh_panel import LongPanel
        p = LongPanel()
        mv = p["float_mv"].to_numpy(np.float64)
        elig = L60 & np.isfinite(mv) & (mv > 0)
        rk = pd.DataFrame(np.where(elig, mv, np.nan)).rank(axis=1, pct=True).to_numpy()
        allA = (rk <= 0.2) & elig
        rkp = pd.DataFrame(np.where(Uv & elig, mv, np.nan)).rank(axis=1, pct=True).to_numpy()
        inpool = rkp <= 0.2
        rest = elig & ~allA
        return np.stack([allA, inpool, rest])
    SM = _cache("small20_masks.npy", small_masks)
    i23 = int(dates.searchsorted(pd.Timestamp("2023-01-03")))
    wex = (W * SM[0]).sum(axis=1)
    wpo = (Uw * SM[0]).sum(axis=1)
    wexp = (W * SM[1]).sum(axis=1)
    wpop = (Uw * SM[1]).sum(axis=1)
    add("S4", "w_small20_allA_floatmv_excl_2023_2025", float(wex[i23:].mean()), "权重", "全 A 有成交、上市满 60 日的股票按流通市值估计(LongPanel float_mv)最小 20%;排雷目标权重,2023-2025 日均")
    add("S4", "w_small20_allA_floatmv_pool_2023_2025", float(wpo[i23:].mean()), "权重", "同上,池等权")
    add("S4", "w_small20_inpool_floatmv_excl_2023_2025", float(wexp[i23:].mean()), "权重", "池内最小 20%")
    add("S4", "w_small20_inpool_floatmv_pool_2023_2025", float(wpop[i23:].mean()), "权重", "池内最小 20%,池等权(按构造约 20%)")
    add("S4", "w_small20_allA_floatmv_excl_2006_2025", float(wex[int(dates.searchsorted(pd.Timestamp(LE.EVAL_START))):].mean()), "权重", "全期参考")
    add("S4", "w_small20_allA_floatmv_pool_2006_2025", float(wpo[int(dates.searchsorted(pd.Timestamp(LE.EVAL_START))):].mean()), "权重", "全期参考")

    def tmv_mask():
        from afactor import newdata as ND
        dd_ = dates[i23:]
        tm = ND.total_mv(dd_, X.codes).to_numpy(np.float64)
        valid = L60[i23:]
        x = np.where(valid & np.isfinite(tm) & (tm > 0), tm, np.nan)
        cov = float((np.isfinite(x).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)).mean())
        m = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy() <= 0.2
        return np.concatenate([m, np.full((1, m.shape[1]), cov)], axis=0)
    TM = _cache("small20_totalmv_2023.npy", tmv_mask)
    tmcov = float(TM[-1, 0])
    TMm = TM[:-1].astype(bool)
    add("S4", "w_small20_allA_totalmv_excl_2023_2025", float((W[i23:] * TMm).sum(axis=1).mean()), "权重",
        f"交叉核对:研究数据 newdata.total_mv(真实总市值,mvfix 口径)最小 20%,覆盖合格股 {tmcov:.1%}")
    add("S4", "w_small20_allA_totalmv_pool_2023_2025", float((Uw[i23:] * TMm).sum(axis=1).mean()), "权重", "同上,池等权")
    # 历史:最小 20% 档相对其余的滚动 242 日相对收益
    def small_rel():
        sm = LE.bench(ad, pd.DataFrame(SM[0].astype(bool), index=U.index, columns=U.columns))
        rest = LE.bench(ad, pd.DataFrame(SM[2].astype(bool), index=U.index, columns=U.columns))
        return pd.DataFrame({"small": sm, "rest": rest})
    SR = _cache("small20_rel.parquet", small_rel).reindex(d.index).loc[LE.EVAL_START:]
    rel = np.log1p(SR["small"]) - np.log1p(SR["rest"])
    rr = np.expm1(rel.rolling(TD).sum())
    add("S4", "small20_vs_rest_worst_242d", float(rr.min()), "相对收益", f"全 A 最小 20%(流通市值)等权 vs 其余等权,滚动 242 日最差,截至 {rr.idxmin().date()}")
    yy = np.expm1(rel.groupby(rel.index.year).sum())
    add("S4", "small20_vs_rest_worst_year", float(yy.min()), "相对收益", f"最差日历年 {int(yy.idxmin())}")
    add("S4", "small20_vs_rest_2017", float(yy.loc[2017]), "相对收益", "2017 年(IPO 提速、壳价值下行)")
    add("S4", "small20_vs_rest_2024", float(yy.loc[2024]), "相对收益", "2024 年(微盘踩踏 + 新国九条)")

    # ---------------- S5:大跌日 β ----------------
    for tag, col in (("b1", "bench_b1"), ("fpg", "strat_fullpool_ew_gross")):
        m = ev[col] < -0.05
        sl = np.polyfit(ev.loc[m, col], ev.loc[m, "strat_main"], 1)[0]
        add("S5", f"downday_beta_{tag}", float(sl), "", f"{tag} 日收益 < -5% 的 {int(m.sum())} 天,排雷对它的斜率")
        add("S5", f"downday_mean_excess_{tag}", float((ev.loc[m, "strat_main"] - ev.loc[m, col]).mean()), "日均")
    # ---------------- S6:大涨日 β ----------------
    for tag, col in (("b1", "bench_b1"), ("fpg", "strat_fullpool_ew_gross")):
        m = ev[col] > 0.05
        sl = np.polyfit(ev.loc[m, col], ev.loc[m, "strat_main"], 1)[0]
        add("S6", f"upday_beta_{tag}", float(sl), "", f"{tag} 日收益 > +5% 的 {int(m.sum())} 天,排雷对它的斜率")

    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "ss_anchors.csv", index=False)
    lock.to_csv(SSC / "s3_lock_by_t0.csv", index=False)
    log(f"[anchors] ss_anchors.csv {len(df)} 行,{time.time() - t0:.1f}s")
    return df


# ====================================================================== B 情景
def linear_path(total: float, n: int) -> np.ndarray:
    return np.full(n, np.expm1(np.log1p(total) / n))


def account_block(stock_total: float, n_days: int, untradable_all: bool = False, cash: float = 0.0) -> dict:
    """股票部分在 n_days 内按等比路径累计 stock_total。返回 f ∈ {50,70,100} x {rb5, bh} 的账户回撤与收益。"""
    r = linear_path(stock_total, n_days)
    u = np.full(n_days, untradable_all)
    out = {}
    for f in F_B:
        for mode in MODES:
            dd, nav = simulate(r, f, cash, mode, 0, u)
            tag = f"f{int(round(f * 100))}_{mode}"
            out[f"acct_dd_{tag}"] = float(dd[0])
            out[f"acct_ret_{tag}"] = float(nav[0] - 1)
    return out


def stage_scenarios():
    t0 = time.time()
    A = pd.read_csv(LEDGER / "ss_anchors.csv")
    a = dict(zip(A["item"], A["value"]))
    rows = []

    def add(scen, pa, pb, method, mtype, excess, excess_int, pool_ret, stock_ret, n_days, background, breaks,
            acct=None, note=""):
        rec = dict(scenario=scen, param_a=pa, param_b=pb, method=method, type=mtype,
                   excess_pp=excess * 100, excess_with_intercept_pp=(excess_int * 100 if excess_int is not None else np.nan),
                   pool_ret=pool_ret, stock_ret=stock_ret, path_days=n_days, background=background,
                   breaks=breaks, note=note)
        if acct:
            rec.update(acct)
            for f in F_B:
                for mode in MODES:
                    tag = f"f{int(round(f * 100))}_{mode}"
                    rec[f"acct_vs_pool_pp_{tag}"] = (rec[f"acct_ret_{tag}"] - pool_ret) * 100
        rows.append(rec)

    # ---------------- S1 流动性溢价反转 ----------------
    P = a["LS_Q1minusQ5_arith_ann"]
    br1 = "低换手跑赢高换手(流动性溢价)长期为正——排雷唯一的收益来源;同时打破'熊市里相对收益守得住'"
    for m in (0.5, 1.0, 1.5):
        for n in (1, 2, 3):
            meths = [("LS五分位×β(对B1)", a["beta_excess_b1_on_LS"] * (-m * P), a["alpha_excess_b1_ann"]),
                     ("LS五分位×β(对fp毛)", a["beta_excess_fpg_on_LS"] * (-m * P), a["alpha_excess_fpg_ann"]),
                     ("排雷自身超额反号", -m * a["excl_excess_geo_vs_b1_full"], None)]
            for name, imp_y, alpha in meths:
                ex = (1 + imp_y) ** n - 1
                exi = (1 + imp_y + alpha) ** n - 1 if alpha is not None else None
                worst = min(ex, exi) if exi is not None else ex
                acct = account_block(worst, TD * n)
                add("S1", f"幅度={m}x", f"持续={n}年", name, "假设:线性暴露×历史溢价外推", ex, exi, 0.0, worst,
                    TD * n, "池等权持平(只看情景本身);账户回撤按'含/不含截距'中更不利者", br1, acct,
                    note=f"历史 LS 年均 {P:.1%}(算术);每年冲击 {imp_y:+.2%}")

    # ---------------- S2 小微盘踩踏放大版 ----------------
    br2 = "'池等权 ≈ 小微盘 beta 可以承受'与'1 个月内相对中证1000 最多落后约 10pp'(2024-01 实测);排雷相对池子不受伤,伤在绝对回撤"
    mkt = -0.20
    for u in (0.15, 0.20, 0.30):
        pool = mkt - u
        meths = [("20日块回归(对B1)", a["blk20_b1_b_mkt"] * mkt + a["blk20_b1_b_spread"] * (-u), a["blk20_b1_intercept"]),
                 ("20日块回归(对fp毛)", a["blk20_fpg_b_mkt"] * mkt + a["blk20_fpg_b_spread"] * (-u), a["blk20_fpg_intercept"]),
                 ("W1比例锚(对B1)", a["W1_ratio_excess_over_b1"] * pool, None)]
        for name, ex, icpt in meths:
            exi = ex + icpt if icpt is not None else None
            worst = min(ex, exi) if exi is not None else ex
            stock = pool + worst
            add("S2", f"池等权跑输中证1000={u:.0%}", "中证1000=-20%,1个月(20日)", name,
                "假设:20 日回归 / W1 比例线性外推(超出历史区间)", ex, exi, pool, stock, 20,
                "中证1000 -20%,池等权 = -20% - 跑输幅度", br2, account_block(stock, 20),
                note=f"W1 实测:池等权 {a['W1_b1']:+.1%}、中证1000 {a['W1_csi1000']:+.1%}、排雷超额 {a['W1_excess_b1'] * 100:+.2f}pp")

    # ---------------- S3 流动性枯竭、卖不出去 ----------------
    br3 = "'5 日调仓、随时能按开盘价减仓'的可成交假设;任何依赖'跌了就减仓'的风控在锁仓期内都无法执行"
    for q in (0.30, 0.50, 0.70):
        for n in (5, 10):
            loss = 1 - 0.9 ** n
            ratio = a[f"lock_ratio_sell_pool_over_excl_ge{n}d"]
            stock = -q * loss
            pool = -q * ratio * loss
            ex = stock - pool
            acct = {}
            for f in F_B:
                for mode in MODES:
                    tag = f"f{int(round(f * 100))}_{mode}"
                    acct[f"acct_dd_{tag}"] = f * q * loss
                    acct[f"acct_ret_{tag}"] = -f * q * loss
            add("S3", f"锁定比例={q:.0%}", f"连续={n}日", "机械:锁定部分每日-10%,可成交部分持平;池子锁定比例按2015-07锚换算",
                "机械(给定锁定比例与跌停幅度)", ex, None, pool, stock, n,
                "可成交部分持平(只看锁仓本身)", br3, acct,
                note=(f"2015-07 锚(停牌或开盘跌停,主):排雷持仓曾连续≥{n}日卖不出的权重最高 {a[f'lock_sell_ge{n}d_excl_max']:.1%}"
                      f"(池等权 {a[f'lock_sell_ge{n}d_pool_max']:.1%};含开盘涨停的敏感性口径 {a[f'lock_anylimit_ge{n}d_excl_max']:.1%});"
                      f"q / 锚 = {q / a[f'lock_sell_ge{n}d_excl_max']:.1f} 倍;锁定期跌幅 {loss:.1%};"
                      "再平衡只买可成交(持平)的股票、不在跌停股上补仓,所以两种再平衡方式回撤相同"))

    # ---------------- S4 壳价值消失 ----------------
    br4 = "'小市值估值溢价(壳价值)稳定';排雷对最小市值档是低配,这条假设破了对相对收益有利、对绝对收益有小幅损失"
    ws = [("流通市值·全A最小20%(主)", a["w_small20_allA_floatmv_excl_2023_2025"], a["w_small20_allA_floatmv_pool_2023_2025"]),
          ("流通市值·池内最小20%", a["w_small20_inpool_floatmv_excl_2023_2025"], a["w_small20_inpool_floatmv_pool_2023_2025"]),
          ("总市值·全A最小20%(交叉核对)", a["w_small20_allA_totalmv_excl_2023_2025"], a["w_small20_allA_totalmv_pool_2023_2025"])]
    for c in (0.20, 0.30, 0.40):
        for name, wx, wp in ws:
            stock = -wx * c
            pool = -wp * c
            add("S4", f"估值下调={c:.0%}", "1年(242日)", name, "机械:权重×冲击(其余股票持平)", stock - pool, None,
                pool, stock, TD, "其余股票持平", br4, account_block(stock, TD),
                note=f"2023-2025 日均权重:排雷 {wx:.1%},池等权 {wp:.1%};历史最差滚动 1 年 {a['small20_vs_rest_worst_242d']:+.1%}")

    # ---------------- S5 熔断 / 休市跳空 ----------------
    br5 = "'每天都能交易、5 日再平衡与风控信号能执行';固定仓位上限不依赖交易,所以不受影响,择时/止损类规则失效"
    for n in (2, 4):
        pool = 0.93 ** n - 1
        meths = [("2016-01熔断锚(对fp毛)", a["CB2016_strat"] / a["CB2016_fpg"]),
                 ("2016-01熔断锚(对B1)", a["CB2016_strat"] / a["CB2016_b1"]),
                 ("大跌日β(对B1)", a["downday_beta_b1"]),
                 ("机械β=1", 1.0)]
        for name, ratio in meths:
            stock = (1 - 0.07 * ratio) ** n - 1
            acct = account_block(stock, n, untradable_all=True)
            add("S5", f"连续={n}日", "每日开盘-7%即休市", name, "机械(给定跳空)+ 锚比例", stock - pool, None, pool,
                stock, n, "池等权每日 -7%", br5, acct, note=f"排雷/池等权 跌幅比 {ratio:.3f};期间不可交易,两种再平衡方式相同")

    # ---------------- S6 政策急涨中的相对落后 ----------------
    br6 = "'相对收益在急涨里不落后'——排雷本身落后有限;真正打破的是仓位上限 f < 1 时'相对池等权'的目标"
    R0 = a["W3_b1"]
    anchors6 = [("W3锚(924,6日)", a["W3_ratio_excess_over_b1"]),
                ("W3+1月锚(至10-31)", a["W3_plus1m_ratio_excess_over_b1"]),
                ("2015H1锚", a["H1_2015_ratio_excess_over_b1"]),
                ("2019初锚", a["BULL19_ratio_excess_over_b1"]),
                ("大涨日β(对B1)", a["upday_beta_b1"] - 1.0)]
    for m in (1.0, 1.5, 2.0):
        for s in (1.0, 1.5, 2.0):
            n = max(1, int(round(6 / s)))
            pool = m * R0
            for name, ratio in anchors6:
                ex = ratio * pool
                stock = pool + ex
                add("S6", f"幅度={m}x", f"速度={s}x({n}日)", name, "假设:锚比例线性外推", ex, None, pool, stock, n,
                    f"池等权 {n} 日内 +{pool:.1%}", br6, account_block(stock, n),
                    note=f"W3 实测池等权 {R0:+.1%}(6 日);落后比例 {ratio:+.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "ss_scenarios.csv", index=False)
    log(f"[scenarios] ss_scenarios.csv {len(df)} 行,{time.time() - t0:.1f}s")
    return df


# ====================================================================== 机械复核
def stage_checks():
    rows = []
    d = pd.read_parquet(DAILY1)
    ev = d.loc[LE.EVAL_START:]
    s2 = pd.read_csv(LEDGER / "lh_step2_summary.csv")
    s2 = s2[(s2.period == "full") & (s2.cash == 0) & (s2.valuation == "close")].set_index("rule")
    r = ev["strat_main"].to_numpy()
    for f in F_GRID:
        mdds = [simulate(r, f, 0.0, "rb5", ph)[0][0] for ph in range(5)]
        rule = f"O1_f{int(round(f * 1000)):04d}"
        eng = float(-s2.loc[rule, "mdd"]) if rule in s2.index else np.nan
        rows.append(dict(check="overlay_vs_engine_O1_full", f=f, value=float(np.mean(mdds)),
                         value_min=float(np.min(mdds)), value_max=float(np.max(mdds)), reference=eng,
                         diff_pp=(float(np.mean(mdds)) - eng) * 100,
                         note="本脚本账户叠加(strat_main 5 相位平均 x 账户 rb5 相位 0-4 平均)全期 2006-2025 最大回撤 vs 第二步引擎 O1(lh_step2_summary.csv)"))
    for wname, (a, b) in HIST_WINDOWS.items():
        w = window_slice(d, a, b)
        sdd, _ = stock_stats(w["strat_main"].to_numpy())
        acc = simulate(w["strat_main"].to_numpy(), 1.0, 0.0, "rb5", 0)[0][0]
        rows.append(dict(check=f"identity_f1_k1_{wname}", f=1.0, value=acc, reference=sdd, diff_pp=(acc - sdd) * 100,
                         note="f = 100%、k = 1 时账户回撤 = 排雷窗口回撤"))
    sc = pd.read_csv(LEDGER / "ss_reverse.csv")
    g = sc[(sc.kind == "grid") & (sc["shape"] == "gap") & (sc.cash == 0) & sc.feasible.astype(bool)]
    err = float((g.account_dd - g.bh_formula_fD).abs().max())
    rows.append(dict(check="gap_shape_equals_fD", f=np.nan, value=err, reference=0.0, diff_pp=err * 100,
                     note="连续跌停跳空、现金 0:两种再平衡方式的账户回撤都应恰好 = f·D"))
    g = sc[(sc.kind == "grid") & (sc["shape"] != "gap") & (sc["mode"] == "bh") & (sc.cash == 0) & sc.feasible.astype(bool)]
    err = float((g.account_dd - g.bh_formula_fD).abs().max())
    rows.append(dict(check="bh_equals_fD_deterministic", f=np.nan, value=err, reference=0.0, diff_pp=err * 100,
                     note="确定性单调下跌、买入持有、现金 0:账户回撤 = f·D"))
    k = sc[(sc.kind == "dstar") & (sc["shape"] == "linear") & (sc["mode"] == "rb5") & (sc.cash == 0) & (sc["T"] == 250)]
    for _, rr in k.iterrows():
        rows.append(dict(check="rb5_vs_continuous_rebal_formula_T250", f=rr.f, value=rr.dstar,
                         reference=rr.dstar_cont_rebal_formula, diff_pp=(rr.dstar - rr.dstar_cont_rebal_formula) * 100,
                         note="线性阴跌 250 日、每 5 日再平衡的 D* vs 连续再平衡解析式 1 - 0.5^(1/f)"))
    sc2 = pd.read_csv(LEDGER / "ss_scale.csv")
    for wname in HIST_WINDOWS:
        g = sc2[(sc2.kind == "grid") & (sc2.window.astype(str) == wname)]
        x = g.groupby(["k"]).stock_mdd.first()
        k1 = float(x.loc[1.0])
        err = max(abs(float(x.loc[kk]) - (1 - (1 - k1) ** kk)) for kk in K_GRID)
        rows.append(dict(check=f"scale_stock_dd_identity_{wname}", f=np.nan, value=err, reference=0.0, diff_pp=err * 100,
                         note="对数收益乘 k 后,股票部分窗口回撤 = 1 - (1 - DD₁)^k(峰谷位置不变)"))
    # ---- 审查补充的复核 ----
    bs = sc2[sc2.kind == "bh_start"]
    gr = sc2[(sc2.kind == "grid") & (sc2.window.astype(str) == "2008") & (sc2.k == 1.0) & (sc2.cash == 0)]
    w08 = str(sc2[(sc2.kind == "grid") & (sc2.window.astype(str) == "2008")].te_start.iloc[0])
    for f in F_GRID:
        b_ = bs[(bs.f == f) & (bs.te_start == w08)]
        g_ = gr[(gr.f == f) & (gr["mode"] == "bh")]
        if len(b_) and len(g_):
            rows.append(dict(check="bh_start_at_window_equals_grid", f=f, value=float(b_.account_mdd.iloc[0]),
                             reference=float(g_.account_mdd.iloc[0]),
                             diff_pp=(float(b_.account_mdd.iloc[0]) - float(g_.account_mdd.iloc[0])) * 100,
                             note="买入持有起点敏感性中起点 = 2008 窗口起点的那一行 = A.2 网格 k=1 bh"))
        bf = bs[bs.f == f]
        rows.append(dict(check="bh_start_range", f=f, value=float(bf.account_mdd.max()), value_min=float(bf.account_mdd.min()),
                         value_max=float(bf.account_mdd.max()), reference=float(bf.rb5_account_mdd_phase_max.max()),
                         diff_pp=(float(bf.account_mdd.max()) - float(bf.rb5_account_mdd_phase_max.max())) * 100,
                         note=f"{len(bf)} 个起点:买入持有回撤范围 [value_min, value_max];reference = 同起点 rb5(相位 0..4)回撤的最大值"))
    mg = sc2[(sc2.kind == "margin") & (sc2["mode"] == "rb5") & sc2.lump_margin.notna() & sc2.margin_20d_rb5.notna()]
    rows.append(dict(check="margin_20d_rb5_le_lump", f=np.nan, value=float((mg.margin_20d_rb5 - mg.lump_margin).max()),
                     reference=0.0, diff_pp=float((mg.margin_20d_rb5 - mg.lump_margin).max()) * 100,
                     note="20 日 rb5 余量应 ≤ 瞬时余量(再平衡在下跌中放大账户损失);value = max(差) 应 ≤ 0"))
    an = pd.read_csv(LEDGER / "ss_anchors.csv").set_index("item")["value"].astype(float)
    for n in (5, 10):
        rows.append(dict(check=f"s3_sell_le_anylimit_ge{n}d", f=np.nan, value=float(an[f"lock_sell_ge{n}d_excl_max"]),
                         reference=float(an[f"lock_anylimit_ge{n}d_excl_max"]),
                         diff_pp=float(an[f"lock_sell_ge{n}d_excl_max"] - an[f"lock_anylimit_ge{n}d_excl_max"]) * 100,
                         note="卖出侧锁仓(停牌或开盘跌停)应 ≤ 含开盘涨停的口径"))
    bu = sc[(sc.kind == "dstar_buyable") & (sc["shape"] == "linear")]
    li = sc[(sc.kind == "dstar") & (sc["shape"] == "linear") & (sc["mode"] == "rb5") & (sc.cash == 0)]
    mm = bu.merge(li, on=["T", "f"], suffixes=("_b", "_m"))
    dif = (mm.dstar_b - mm.dstar_m).abs()
    rows.append(dict(check="buyable_linear_equals_main", f=np.nan, value=float(dif.max()) if dif.notna().any() else 0.0,
                     reference=0.0, diff_pp=float(dif.max()) * 100 if dif.notna().any() else 0.0,
                     note="线性阴跌在可行域内(除 T=20、D=87.8% 端点)没有 -10% 日,跌停日可补仓的约定不改变 D*"))
    rows.append(dict(check="data_cutoff", f=np.nan, value=str(d.index.max().date()), reference=RESEARCH_END,
                     diff_pp=np.nan, note="第一步每日序列最后一个信号日;长历史面板截至 2025-12-31,未读 2026 数据"))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "ss_checks.csv", index=False)
    log(f"[checks] ss_checks.csv {len(df)} 行")
    print(df.to_string())
    return df


# ====================================================================== 图
def stage_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    for fn in ("PingFang SC", "Heiti SC", "Arial Unicode MS"):
        if any(fn in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [fn]
            break
    plt.rcParams["axes.unicode_minus"] = False
    sc = pd.read_csv(LEDGER / "ss_reverse.csv")
    ds = sc[(sc.kind == "dstar") & (sc.cash == 0)]
    # 买入持有(三种形状)与连续跌停跳空(两种方式)在确定性路径下恒等于 D* = 50%/f,先核对再合并成一条线
    same = ds[(ds["mode"] == "bh") | (ds["shape"] == "gap")]
    ref = np.where(same.f > 0.5, 0.5 / same.f, np.nan)
    ok_same = np.isfinite(same.dstar) == np.isfinite(ref)
    err = np.nanmax(np.abs(same.dstar.to_numpy() - ref)) if np.isfinite(ref).any() else 0.0
    assert ok_same.all() and err < 2e-3, f"买入持有/跳空 与 50%/f 不一致: {err}"
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    dsb = sc[sc.kind == "dstar_buyable"]
    series = [("linear", "rb5", "#2a78d6", "-", "o", "线性阴跌 · 每 5 日再平衡", -0.9, ds),
              ("front", "rb5", "#eb6834", "-", "o", "前段急跌后段阴跌 · 每 5 日再平衡(跌停日不补仓)", 0.0, ds),
              ("gap", "bh", "#1baf7a", (0, (4, 3)), "s",
               "买入持有 = 连续跌停跳空且跌停日不补仓:D* = 50% ÷ f(仅当危机开始时股票占比恰为 f)", 0.9, ds),
              ("gap", "rb5", "#7a5bd6", (0, (1, 1.6)), "D", "连续跌停跳空 · 跌停日仍挂跌停价补回 f(审查补充的约定敏感性)", 1.8, dsb)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.3), sharey=True, facecolor=surf)
    for ax, T in zip(axes, T_GRID):
        ax.set_facecolor(surf)
        for shape, mode, c, ls, mk, lab, dx, src in series:
            g = src[(src["shape"] == shape) & (src["mode"] == mode) & (src["T"] == T)].sort_values("f")
            x = g.f.to_numpy() * 100
            y = g.dstar.to_numpy() * 100
            ok = np.isfinite(y)
            ax.plot(x[ok], y[ok], color=c, ls=ls, lw=2.0, marker=mk, ms=6, mfc=c if mode == "rb5" else surf,
                    mec=c, mew=1.6, zorder=3)
            if (~ok).any():
                ax.scatter(x[~ok] + dx, np.full((~ok).sum(), 101.8), marker="^", s=40, facecolor=surf, edgecolor=c,
                           lw=1.5, zorder=4, clip_on=False)
        ax.axhline(69.6, color=ink2, lw=1, ls=":", zorder=1)
        ax.text(101.5, 70.4, "2008 排雷实际回撤 69.6%", color=ink2, fontsize=8.5, ha="right", va="bottom")
        ax.axvline(50, color=grid, lw=1.2, zorder=0)
        ax.set_title(f"危机持续 T = {T} 个交易日", color=ink, fontsize=11)
        ax.set_xlim(27, 103)
        ax.set_ylim(25, 105)
        ax.set_xticks([30, 40, 50, 60, 70, 80, 100])
        ax.set_xlabel("股票仓位上限 f(%)", color=ink2)
        ax.grid(axis="y", color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=ink2)
        mf = (1 - 0.9 ** T) * 100
        if mf < 99.9:
            ax.axhline(mf, color=grid, lw=1, ls="--", zorder=0)
            ax.text(101.5, mf - 1.0, f"可行上限 {mf:.1f}%(每天 -10%)", color=ink2, fontsize=7.5, ha="right", va="top")
    axes[0].set_ylabel("账户回撤恰好 50% 时股票部分的累计跌幅 D*(%)", color=ink2)
    handles = [Line2D([], [], color=c, ls=ls, lw=2, marker=mk, ms=6, mfc=c if mode == "rb5" else surf, mec=c, mew=1.6)
               for _, mode, c, ls, mk, _, _, _ in series]
    handles.append(Line2D([], [], ls="none", marker="^", ms=7, mfc=surf, mec=ink2, mew=1.4))
    labels = [s_[5] for s_ in series] + ["顶端三角:股票部分跌到 100%(或可行上限)账户也不触及 50%"]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("反向压力测试:账户回撤触及 50% 需要股票部分跌多少(确定性路径,现金收益 0%)", color=ink, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "ss_reverse_frontier.png"
    fig.savefig(out, dpi=160, facecolor=surf)
    log(f"[figure] {out}")


STAGES = {"anchors": stage_anchors, "reverse": stage_reverse, "scale": stage_scale, "scenarios": stage_scenarios,
          "checks": stage_checks, "figure": stage_figure}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=list(STAGES))
    args = ap.parse_args()
    t_all = time.time()
    for st in args.stages:
        t = time.time()
        STAGES[st]()
        log(f"== {st} {time.time() - t:.1f}s")
    log(f"== 全部 {time.time() - t_all:.1f}s")
