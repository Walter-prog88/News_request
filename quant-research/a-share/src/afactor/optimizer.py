"""中证 1000 指数增强组合优化器(2026-10-10,前缀 ie)。

## 为什么要有它
项目此前的持仓只有两种:Top-N 等权(选优)与排雷缓冲(剔最差 20/35、其余等权)。二者都没有基准约束,
相对任何真实指数的跟踪误差、风格偏离都是"顺便得到"的。业内最主要的量化产品是**约束下的指数增强**:
给定基准(这里是中证 1000 的 PIT 代理)、给定风险模型(afactor.barra.RiskModel)与成本模型
(afactor.impact.CostModel),在跟踪误差、行业/风格偏离、个股偏离的约束下最大化"预期 alpha − 交易成本"。
本模块实现这个优化器,并通过引擎的 callable 接口(strategy.simulate 的 target_fn)接入逐日成交模拟。

## 问题形式(预注册,**不得按回测结果调整**;敏感性另跑并全部报告)
    max_w   α'w − κ·Cost(w − h) − λ·Var5(w − wb)
    s.t.    w ≥ 0,Σw = 1                                  (多头、满仓)
            w_i ≤ wb_i + 1%                               (个股主动上限)
            |H'(w − wb)| ≤ 2%                             (申万一级 30 行业,逐个)
            |X_k'(w − wb)| ≤ 0.1   k ∈ {Size, NonlinSize} (z 单位)
            |X_k'(w − wb)| ≤ 0.5   其余 8 个 CNE5 风格
            sqrt(Var5(w − wb) · 242/5) ≤ 5%                (事前年化跟踪误差,风险模型因子形式)
            w_i = h_i                                     (te 日不可成交:停牌/一字板;未持有的即 0)
            |w_i − h_i| ≤ cap_i                           (参与率上限的事前估计,见下"偏离任务书之处")
            w_i = 0  对候选集之外的股票
  - α_i = IC_t · ω_i · z_i(Grinold):z = 打分在可投资池(V1000 决策池 ∩ 风险模型覆盖)内的截面 z-score,
    截尾 ±3;ω_i = 风险模型 5 日特质波动 sqrt(d_i);IC_t = 该打分 5 日 rank IC 的 250 日滚动均值,
    只用到 t−6 已实现的 IC(信号日 s 的 5 日前向收益 twap_open10[s+6]/twap_open10[s+1]−1 在 s+6 开盘才实现),
    下限 0;不足 20 个观测时视为 0(只影响 V7 的样本外打分起点)。可投资池外的基准成分 α=0。
  - Cost = 引擎同一成本模型(impact.CostModel.cost_coefficients):
        Σ lin_buy_i·buy_i + Σ lin_sell_i·sell_i + Σ c15_i·|Δw_i|^1.5,κ = 成本乘数 1
    (整笔成本对 5 日 α 一次性扣除:5 日调仓,α 也是 5 日预期)。
  - Var5(a) = ||Lᵀ Bᵀ a||² + Σ d_i a_i²,B = [1, X, H],L Lᵀ = 5 日因子协方差(RiskModel.at)。不构造 N×N。
  - 候选集 = 基准成分 ∪ 可投资池中 α 前 30% ∪ 当前持仓(业内常用的候选集裁剪:否则 4300 只全进优化器,
    其中 α 低于均值的非成分股在多头约束下最优权重必为 0,只拖慢求解)。

## 偏离任务书之处(全部在看任何回测结果之前定死)
1. **λ·Var5 平局项**(λ=1,5 日方差单位)。任务书的目标里只有 α − 成本、风险作为约束;但 α=0 且成本=0 时
   目标恒为 0,任何可行点都是最优解,内点法会返回最优面的"解析中心"而不是基准,"零 α 持仓 = 基准"不成立。
   λ=1 时,事前 TE 顶在 5% 上限处的惩罚 = 5.2e-5(每 5 日 0.52bp,年化约 0.25pp),比典型的 5 日事前 α
   小两个数量级,几乎不改变有 α 时的解,只在 α≈0 时把解唯一地拉回基准。
2. **参与率上限进优化器**(|Δw_i| ≤ ctx['cap_w'],截至 te−1 的 20 日窗口成交额中位数 × 10% / 资金)。
   引擎无论如何都会按 te 实际窗口成交额截量;优化器不知道这一点时,大资金下目标会被截成违反行业/风格约束的
   残缺组合。1000 万 open10 下几乎不绑定(CSI1000 成分开盘 10 分钟成交额的 10% 通常是数个百分点的权重)。
   若因此不可行(大资金首次建仓时 Σ min(wb+1%, h+cap) < 1),按下面的求解级联去掉这一条重解。
3. 基准成分中**风险模型覆盖池之外**的股票(无真实市值,平均 0.34%、最大 0.88% 的基准权重,约 3 只)
   不进优化器;优化用的基准权重在覆盖池内重新归一。实际组合因此对这几只有隐含的负主动权重,
   事后归因里记为 uncovered。
4. 求解器:CLARABEL 为主,SCS 兜底。**OSQP 不适用**:TE 是二阶锥约束、冲击是 3/2 次幂,OSQP 只解
   线性约束的 QP;要用它得把 TE 改成罚项、把 3/2 次幂线性化,等于换了一个问题。

## 求解级联(失败处理)
  CLARABEL → 状态既非 optimal 也非 optimal_inaccurate、或事后校验违约 → SCS → 仍失败且启用了参与率约束
  → 去掉参与率约束重来 → 仍失败 → **保持当前持仓不交易**(target_fn 返回 None),记入 solver log 并在报告里计数。
  事后校验(check_solution):边界 1e-6、行业/风格 1e-5、TE 相对 1e-3、Σw 1e-6。
  optimal_inaccurate(CLARABEL 的 AlmostSolved:间隙已到 ~1e-7 但最后几步停滞)**通过事后校验才接受**;
  冒烟测试里它与收紧容差重解的目标值差 < 1e-9,大资金(冲击系数大)时较常见,报告里单独计数。

## 与引擎的接口(strategy.simulate 的 callable 约定)
  IndexEnhancer(...) 本身就是 target_fn(t, h, ctx):
  - 只用决策时点可得的 ctx 字段:h_dec / nav_dec / can / stuck / cap_w / te / cost_model;
  - 位置参数 h(按 px[te] 计价)不用于任何判断;
  - 返回目标里:不可成交的持仓、以及优化后 |w − h_dec| < trade_tol 的股票给 NaN(= 保持实际持仓不交易),
    否则 h_dec 与 h 之间一天的价格漂移会让引擎对上千只股票做"再平衡回目标"的微小交易。

## 评价基准(2026-10-10 审查 P0 修复,只用于事后评价,不进优化)
  exec_time_bench_returns(p, mode, W):代理权重先从收盘漂移到成交时点(W[t]·px[t+1]/close[t] 归一)再乘引擎成交价收益,
  = "在引擎成交价上计价的同一只指数"。旧的 IP.engine_consistent_bench_returns 直接用收盘权重 W[t],
  每天重置回过时一个隔夜的权重,open10 下偏低约 0.75pp/年、vwap1 下偏低约 2.4pp/年(ie_bench_compare.csv)。
  优化器本身仍用 W[t](决策时点只知道收盘权重,px[t+1] 不可知)。
  (2026-10-10 第二轮修订:实现搬到 afactor.index_proxy.exec_time_bench_returns,作为全项目推荐的代理基准;
   本模块保留同名再导出,调用方式与数值不变。)

## 已知局限
  - 风险模型(RISK_MODEL.md)对排雷/反转型主动组合的 TE 低估约 29%(缺短期反转因子);
    事前 5% 的 TE 约束对应的事后 TE 预计更高。本模块不做经验放大(放大系数是样本内估计)。
  - 行业是当前快照(无 in/out 日期);Size 暴露是 v3 不截尾口径。
  - vwap1 口径(容量曲线)沿用 open10 的风险模型(5 日开盘到开盘的协方差),平均价口径下 5 日方差近似相同。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

SIZE_STYLES = ("Size 市值", "NonlinSize 非线性市值")


# ================================================================ 配置
@dataclass(frozen=True)
class OptConfig:
    """优化器参数。默认值 = 预注册的基准变体 V1。"""
    te_ann: float | None = 0.05           # 事前年化跟踪误差上限;None = 不约束(仅测试用)
    horizon: int = 5                      # 风险模型预测期(日),与调仓频率一致
    ann_days: int = 242
    stock_active_ub: float = 0.01         # w_i <= wb_i + 1%
    ind_bound: float | None = 0.02        # 申万一级行业主动暴露
    size_bound: float | None = 0.10       # Size / NonlinSize
    style_bound: float | None = 0.50      # 其余 8 个风格
    min_bench_weight: float | None = None  # V5 公募口径:成分股权重 >= 80%
    cost_mult: float = 1.0                # 成本乘数 κ;V6 = 0(优化器忽略成本,引擎照收)
    risk_aversion: float = 1.0            # λ:平局项(见模块文档"偏离任务书之处 1")
    cand_top_frac: float = 0.30           # 候选集:可投资池 α 前 30%
    z_clip: float = 3.0                   # z-score 截尾
    trade_tol: float = 2e-5               # |w-h_dec| 小于它视为不交易(0.2bp 净值)
    trade_bounds: bool = True             # 参与率上限进优化器
    solvers: tuple = ("CLARABEL", "SCS")
    obj_scale: float = 1.0                # 目标函数放大倍数(数值用;实测 1e4 会让 CLARABEL 停在 AlmostSolved,1.0 正常收敛,解相同到 1e-9)
    ic_min_obs: int = 20                  # IC 滚动均值最少观测
    ic_window: int = 250
    ic_lag: int = 6                       # 5 日前向收益在 s+6 开盘实现 -> 决策日 t 只能用 s <= t-6

    @property
    def te5(self) -> float | None:
        """5 日主动波动上限(收益单位)。"""
        if self.te_ann is None:
            return None
        return float(self.te_ann * np.sqrt(self.horizon / self.ann_days))

    def describe(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


# ================================================================ 核心问题(与引擎无关,测试直接用)
@dataclass
class ActiveProblem:
    """候选集上的优化问题数据(全部按候选集顺序,长度 n)。

    alpha   n       5 日预期 α(收益单位)
    wb      n       基准权重(已在覆盖池内归一;非成分为 0)
    h       n       当前持仓(决策口径 h_dec)
    X       n×K     风格暴露(缺失已填 0)
    H       n×J     行业 one-hot(无行业为全 0 行)
    L       M×M     L Lᵀ = 5 日因子协方差,M = 1 + K + J(顺序 [Country, 风格, 行业])
    d       n       5 日特质方差
    lin_buy/lin_sell/c15  n  成本系数(占净值比例;冲击 = c15·|Δw|^1.5)
    fixed   n bool  权重固定为 h(te 不可成交)
    cap     n|None  单日最大 |Δw|(参与率上限的事前估计);None = 不约束
    bench   n bool  是否基准成分(V5 用)
    style_names     K 个风格名(决定哪两个用 size_bound)
    """
    alpha: np.ndarray
    wb: np.ndarray
    h: np.ndarray
    X: np.ndarray
    H: np.ndarray
    L: np.ndarray
    d: np.ndarray
    lin_buy: np.ndarray
    lin_sell: np.ndarray
    c15: np.ndarray
    fixed: np.ndarray
    cap: np.ndarray | None = None
    bench: np.ndarray | None = None
    style_names: list = field(default_factory=list)
    total: float = 1.0          # Σw;覆盖池外有卡住的持仓时 = 1 − 其权重(基准同比例缩放)
    Xe: np.ndarray | None = None   # n×E 额外暴露(不在风险模型里,只做约束;如 20 日反转,见 V8)
    be: np.ndarray | None = None   # E 个上界(|Xe'(w−wb)| ≤ be)

    @property
    def n(self) -> int:
        return len(self.alpha)

    def B(self) -> np.ndarray:
        return np.column_stack([np.ones(self.n), self.X, self.H])

    def style_bounds(self, cfg: OptConfig) -> np.ndarray | None:
        if cfg.size_bound is None and cfg.style_bound is None:
            return None
        K = self.X.shape[1]
        names = self.style_names or [f"s{k}" for k in range(K)]
        b = np.full(K, np.inf)
        for k, nm in enumerate(names):
            v = cfg.size_bound if nm in SIZE_STYLES else cfg.style_bound
            if v is not None:
                b[k] = float(v)
        return b

    def bounds(self, cfg: OptConfig, use_cap: bool) -> tuple[np.ndarray, np.ndarray]:
        """逐股上下界。固定的 = h;参与率约束与个股上限冲突时(h 已高于 wb+1% 且卖不完)参与率优先。"""
        lb = np.zeros(self.n)
        ub = self.wb + cfg.stock_active_ub
        if use_cap and self.cap is not None:
            cap = np.where(np.isfinite(self.cap), np.maximum(self.cap, 0.0), 0.0)
            lo = np.maximum(self.h - cap, 0.0)
            hi = self.h + cap
            lb = np.maximum(lb, lo)
            ub = np.minimum(ub, hi)
            ub = np.maximum(ub, lb)                 # 冲突:卖到参与率允许的最低处
        lb = np.where(self.fixed, self.h, lb)
        ub = np.where(self.fixed, self.h, ub)
        return lb, ub

    def risk_parts(self, w: np.ndarray) -> tuple[float, float]:
        """(因子方差, 特质方差),5 日单位。"""
        a = w - self.wb
        with np.errstate(all="ignore"):
            e = np.concatenate([[a.sum()], np.dot(self.X.T, a), np.dot(self.H.T, a)])
            y = np.dot(self.L.T, e)
        return float(np.sum(y * y)), float(np.sum(self.d * a * a))

    def te_ann(self, w: np.ndarray, cfg: OptConfig) -> float:
        vf, vs = self.risk_parts(w)
        return float(np.sqrt(max(vf + vs, 0.0)) * np.sqrt(cfg.ann_days / cfg.horizon))

    def cost(self, w: np.ndarray) -> float:
        dw = w - self.h
        return float(np.sum(self.lin_buy * np.clip(dw, 0, None)) + np.sum(self.lin_sell * np.clip(-dw, 0, None))
                     + np.sum(self.c15 * np.abs(dw) ** 1.5))


def check_solution(prob: ActiveProblem, w: np.ndarray, cfg: OptConfig, use_cap: bool) -> dict:
    """事后校验所有约束,返回各项最大违约量(>0 为违约)。"""
    lb, ub = prob.bounds(cfg, use_cap)
    a = w - prob.wb
    out = dict(sum=abs(float(w.sum()) - prob.total),
               lb=float(np.max(lb - w, initial=0.0)), ub=float(np.max(w - ub, initial=0.0)))
    with np.errstate(all="ignore"):
        if cfg.ind_bound is not None and prob.H.shape[1]:
            out["ind"] = float(np.max(np.abs(np.dot(prob.H.T, a)) - cfg.ind_bound, initial=0.0))
        sb = prob.style_bounds(cfg)
        if sb is not None:
            out["style"] = float(np.max(np.abs(np.dot(prob.X.T, a)) - sb, initial=0.0))
        if prob.Xe is not None:
            out["extra"] = float(np.max(np.abs(np.dot(prob.Xe.T, a)) - prob.be, initial=0.0))
    if cfg.te5 is not None:
        vf, vs = prob.risk_parts(w)
        out["te_rel"] = float(np.sqrt(max(vf + vs, 0.0)) / cfg.te5 - 1.0)
    if cfg.min_bench_weight is not None and prob.bench is not None:
        out["bench"] = float(cfg.min_bench_weight - w[prob.bench].sum())
    return out


_TOL = dict(sum=1e-6, lb=1e-6, ub=1e-6, ind=1e-5, style=1e-5, extra=1e-5, te_rel=1e-3, bench=1e-6)


def violations_ok(v: dict) -> bool:
    return all(v.get(k, 0.0) <= tol for k, tol in _TOL.items())


def _build_cvx(prob: ActiveProblem, cfg: OptConfig, use_cap: bool):
    import cvxpy as cp
    n = prob.n
    w = cp.Variable(n)
    a = w - prob.wb
    dw = w - prob.h
    lb, ub = prob.bounds(cfg, use_cap)
    cons = [cp.sum(w) == prob.total, w >= lb, w <= ub]
    with np.errstate(all="ignore"):
        G = np.dot(prob.B(), prob.L)             # n×M:因子风险 y = Gᵀ a
    sd = np.sqrt(np.clip(prob.d, 0.0, None))
    y = G.T @ a
    if cfg.ind_bound is not None and prob.H.shape[1]:
        hm = prob.H.sum(axis=0) > 0              # 候选集里没有成员的行业:暴露 = −wb 合计 = 0(wb 也在候选集内)
        if hm.any():
            ea = prob.H[:, hm].T @ a
            cons += [ea <= cfg.ind_bound, ea >= -cfg.ind_bound]
    sb = prob.style_bounds(cfg)
    if sb is not None:
        k = np.flatnonzero(np.isfinite(sb))
        if len(k):
            es = prob.X[:, k].T @ a
            cons += [es <= sb[k], es >= -sb[k]]
    if prob.Xe is not None and prob.Xe.shape[1]:
        ee = prob.Xe.T @ a
        cons += [ee <= prob.be, ee >= -prob.be]
    if cfg.te5 is not None:
        cons.append(cp.norm(cp.hstack([y, cp.multiply(sd, a)]), 2) <= cfg.te5)
    if cfg.min_bench_weight is not None and prob.bench is not None and prob.bench.any():
        cons.append(cp.sum(w[np.flatnonzero(prob.bench)]) >= cfg.min_bench_weight)
    obj = prob.alpha @ w
    if cfg.cost_mult > 0:
        obj = obj - cfg.cost_mult * (prob.lin_buy @ cp.pos(dw) + prob.lin_sell @ cp.neg(dw)
                                     + prob.c15 @ cp.power(cp.abs(dw), 1.5))
    if cfg.risk_aversion > 0:
        obj = obj - cfg.risk_aversion * (cp.sum_squares(y) + cp.sum_squares(cp.multiply(sd, a)))
    return cp.Problem(cp.Maximize(cfg.obj_scale * obj), cons), w


def solve(prob: ActiveProblem, cfg: OptConfig) -> dict:
    """按级联求解。返回 dict(w|None, status, solver, cap_relaxed, attempts, solve_s, viol, obj, te_ann, cost)。"""
    import cvxpy as cp
    t0 = time.time()
    attempts = []
    caps = [True, False] if (cfg.trade_bounds and prob.cap is not None) else [False]
    for use_cap in caps:
        try:
            cp_prob, w = _build_cvx(prob, cfg, use_cap)
        except Exception as e:                          # 构建失败(数据问题)直接记失败
            attempts.append(("build", use_cap, f"error:{type(e).__name__}"))
            continue
        for sv in cfg.solvers:
            try:
                kw = dict(max_iters=20000, eps_abs=1e-7, eps_rel=1e-7) if sv == "SCS" else {}
                cp_prob.solve(solver=sv, **kw)
                st = cp_prob.status
            except Exception as e:
                attempts.append((sv, use_cap, f"error:{type(e).__name__}"))
                continue
            if st in ("optimal", "optimal_inaccurate") and w.value is not None:
                wv = np.asarray(w.value, np.float64)
                wv = np.where(np.abs(wv) < 1e-12, 0.0, wv)
                v = check_solution(prob, wv, cfg, use_cap)
                attempts.append((sv, use_cap, st if violations_ok(v) else st + "|viol"))
                if violations_ok(v):
                    return dict(w=wv, status=st, solver=sv, cap_relaxed=(cfg.trade_bounds and not use_cap
                                                                         and prob.cap is not None),
                                attempts=attempts, solve_s=time.time() - t0, viol=v,
                                obj=float(np.dot(prob.alpha, wv)) - cfg.cost_mult * prob.cost(wv),
                                te_ann=prob.te_ann(wv, cfg), cost=prob.cost(wv))
            else:
                attempts.append((sv, use_cap, st))
            if st in ("infeasible", "infeasible_inaccurate"):
                break                                    # 不可行不必换求解器,直接放松参与率
    return dict(w=None, status="failed", solver=None, cap_relaxed=False, attempts=attempts,
                solve_s=time.time() - t0, viol=None, obj=np.nan, te_ann=np.nan, cost=np.nan)


# ================================================================ α:Grinold + 滚动 IC
def rank_ic_series(score: pd.DataFrame, fwd: pd.DataFrame, universe: pd.DataFrame,
                   min_stocks: int = 200) -> pd.Series:
    """逐日截面 Spearman rank IC(决策池内、两边有限)。索引 = 信号日 s(IC[s] 在 fwd 实现后才可知)。"""
    from scipy.stats import rankdata
    S = score.reindex(index=fwd.index, columns=fwd.columns).to_numpy(np.float64)
    R = fwd.to_numpy(np.float64)
    U = universe.reindex(index=fwd.index, columns=fwd.columns).fillna(False).to_numpy(bool)
    out = np.full(len(fwd.index), np.nan)
    for t in range(len(out)):
        m = U[t] & np.isfinite(S[t]) & np.isfinite(R[t])
        if m.sum() < min_stocks:
            continue
        a = rankdata(S[t, m]); b = rankdata(R[t, m])
        a -= a.mean(); b -= b.mean()
        den = np.sqrt(np.sum(a * a) * np.sum(b * b))
        out[t] = float(np.sum(a * b) / den) if den > 0 else np.nan
    return pd.Series(out, index=fwd.index, name="rank_ic")


def ic_for_decision(ic: pd.Series, window: int = 250, lag: int = 6, min_obs: int = 20,
                    floor: float = 0.0) -> pd.Series:
    """决策日 t 可用的 IC 预测 = IC[t−lag−window+1 .. t−lag] 的均值(下限 floor;观测不足为 NaN)。

    lag=6:open10 下信号日 s 的 5 日前向收益 = twap_open10[s+6]/twap_open10[s+1]−1,在 s+6 的 09:40 实现,
    所以 t 日收盘只知道 s ≤ t−6 的 IC。"""
    m = ic.rolling(window, min_periods=min_obs).mean().shift(lag)
    return m.clip(lower=floor)


def grinold_alpha(score_row: np.ndarray, inv: np.ndarray, omega: np.ndarray, ic: float,
                  z_clip: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """α = IC·ω·z。score_row/inv/omega 等长;z 只在 inv 内算(均值 0、标准差 1,再截尾 ±z_clip)。
    返回 (alpha, z),inv 外为 0。IC 非有限或 ≤0 时 α 全 0。"""
    n = len(score_row)
    z = np.zeros(n)
    m = inv & np.isfinite(score_row)
    if m.sum() >= 20:
        s = score_row[m].astype(np.float64)
        sd = s.std()
        if sd > 0:
            z[m] = np.clip((s - s.mean()) / sd, -z_clip, z_clip)
    if not np.isfinite(ic) or ic <= 0:
        return np.zeros(n), z
    return ic * np.nan_to_num(omega) * z, z


# ================================================================ 引擎接入
class IndexEnhancer:
    """strategy.simulate / run_weights 的 target_fn。每次调仓日:
    风险模型快照 -> Grinold α -> 候选集 -> 成本系数 -> 求解 -> 目标(NaN = 不交易)。

    p        主口径 Panel
    rm       RiskModel(open10)
    bench_w  基准权重 DataFrame [dates × codes](IP.proxy_weights,W[t] = 信号日 t 收盘权重)
    score    打分 DataFrame(信号日 t 收盘已知)
    ic_dec   决策日可用的 IC 预测 Series(ic_for_decision 的结果,已下限 0)
    universe V1000 决策池 DataFrame(可投资池 = 决策池 ∩ 风险模型覆盖)
    """

    def __init__(self, p, rm, bench_w: pd.DataFrame, score: pd.DataFrame, ic_dec: pd.Series,
                 universe: pd.DataFrame, cfg: OptConfig = OptConfig(), tag: str = "",
                 keep_targets: bool = True, extra_exposures: dict | None = None):
        self.cfg = cfg
        self.tag = tag
        self.rm = rm
        self.dates, self.codes = p.dates, p.codes
        assert rm.dates.equals(p.dates) and rm.codes.equals(p.codes), "风险模型与面板不对齐"
        self.WB = bench_w.reindex(index=p.dates, columns=p.codes).fillna(0.0).to_numpy(np.float32)
        self.S = score.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32)
        self.U = universe.reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
        self.ic = ic_dec.reindex(p.dates).to_numpy(np.float64)
        self.style_names = list(rm.style_names)
        self.log: list[dict] = []
        self.targets: dict = {}                 # (phase, t) -> 稀疏目标(诊断:事前暴露)
        self.keep_targets = keep_targets
        # 额外暴露约束 {名字: (DataFrame[dates×codes], 上界)};值须在信号日 t 收盘已知,缺失按 0
        self.extra = {k: (v.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32), float(b))
                      for k, (v, b) in (extra_exposures or {}).items()}

    # -------------------------------------------------- 单次调仓
    def __call__(self, t: int, h, ctx: dict):
        from afactor.barra import psd_sqrt
        cfg = self.cfg
        rec = dict(tag=self.tag, phase=ctx["phase"], date=self.dates[t], trade_date=ctx["trade_date"])
        t_start = time.time()
        if not self.rm.available(self.dates[t]):
            rec.update(status="no_risk_model")
            self.log.append(rec)
            return None
        mask, X, Hm, cov, d = self.rm.at(self.dates[t])
        L = psd_sqrt(cov)[1]
        idx_m = np.flatnonzero(mask)
        pos_m = np.full(len(self.codes), -1)
        pos_m[idx_m] = np.arange(len(idx_m))

        h_dec = np.asarray(ctx["h_dec"], np.float64)
        can = np.asarray(ctx["can"], bool)
        held = h_dec > 1e-10

        # ---- 基准:覆盖池内重新归一(覆盖池外的成分记 dropped)
        wb_full = self.WB[t].astype(np.float64)
        wb_tot = wb_full.sum()
        wb_in = np.where(mask, wb_full, 0.0)
        dropped = float(wb_tot - wb_in.sum())
        if wb_in.sum() <= 0:
            rec.update(status="no_benchmark")
            self.log.append(rec)
            return None
        wb_in = wb_in / wb_in.sum()

        # ---- α
        inv = self.U[t] & mask
        omega = np.zeros(len(self.codes))
        omega[idx_m] = np.sqrt(np.clip(d, 0.0, None))
        ic_t = self.ic[t] if np.isfinite(self.ic[t]) else 0.0
        alpha, z = grinold_alpha(self.S[t].astype(np.float64), inv, omega, ic_t, cfg.z_clip)

        # ---- 候选集 = 成分 ∪ α 前 30% ∪ 当前持仓(覆盖池内)
        cand = wb_in > 0
        inv_idx = np.flatnonzero(inv & np.isfinite(self.S[t]))
        if len(inv_idx):
            k = max(1, int(np.ceil(cfg.cand_top_frac * len(inv_idx))))
            top = inv_idx[np.argsort(-alpha[inv_idx], kind="stable")[:k]] if ic_t > 0 else \
                inv_idx[np.argsort(-z[inv_idx], kind="stable")[:k]]
            cand[top] = True
        cand |= held & mask
        # 覆盖池外的持仓:不可成交的只能拿着(占预算),可成交的清掉(无暴露/风险可建模)
        out_held = held & ~mask
        out_stuck = out_held & ~can
        ci = np.flatnonzero(cand)
        fixed = ~can[ci]
        budget_out = float(h_dec[out_stuck].sum())

        # ---- 成本
        cm = ctx["cost_model"]
        te = int(ctx["te"])
        if cm is not None:
            co = cm.cost_coefficients(te, ctx["capital"], ctx["nav_dec"])
            lin_b, lin_s, c15 = co["lin_buy"][ci], co["lin_sell"][ci], co["c15"][ci]
        else:                                                  # 无成本模型时:佣金 + 印花税(与旧引擎同)
            from afactor.config import COST_BPS_ONEWAY, STAMP_TAX_BPS_BEFORE, STAMP_TAX_BPS_AFTER, \
                STAMP_TAX_CHANGE_DATE
            stp = (STAMP_TAX_BPS_BEFORE if self.dates[te] < pd.Timestamp(STAMP_TAX_CHANGE_DATE)
                   else STAMP_TAX_BPS_AFTER) * 1e-4
            lin_b = np.full(len(ci), COST_BPS_ONEWAY * 1e-4); lin_s = lin_b + stp; c15 = np.zeros(len(ci))
        cap = np.asarray(ctx["cap_w"], np.float64)[ci] if cfg.trade_bounds else None

        pm = pos_m[ci]
        scale = 1.0 - budget_out                                # 覆盖池外卡住的仓位先占预算
        prob = ActiveProblem(alpha=alpha[ci], wb=wb_in[ci] * scale, h=h_dec[ci], X=X[pm], H=Hm[pm], L=L,
                             d=d[pm], lin_buy=lin_b, lin_sell=lin_s, c15=c15, fixed=fixed, cap=cap,
                             bench=(wb_in[ci] > 0), style_names=self.style_names, total=scale)
        if self.extra:
            prob.Xe = np.column_stack([np.nan_to_num(v[t, ci].astype(np.float64)) for v, _ in self.extra.values()])
            prob.be = np.array([b for _, b in self.extra.values()])
        res = solve(prob, cfg)
        rec.update(status=res["status"], solver=res["solver"], cap_relaxed=res["cap_relaxed"],
                   attempts="; ".join(f"{a}/{'cap' if b else 'nocap'}/{c}" for a, b, c in res["attempts"]),
                   solve_s=res["solve_s"], n_cand=len(ci), n_fixed=int(fixed.sum()),
                   n_inv=int(inv.sum()), ic=float(ic_t), wb_dropped=dropped, out_held=float(h_dec[out_held].sum()),
                   h_sum=float(h_dec.sum()))
        if res["w"] is None:
            rec.update(total_s=time.time() - t_start)
            self.log.append(rec)
            return None                                          # 失败:保持当前持仓
        w_c = np.clip(res["w"], 0.0, None)
        # 事前诊断
        wb_c = wb_in[ci] * scale
        a = w_c - wb_c
        with np.errstate(all="ignore"):
            e_sty = np.dot(X[pm].T, a)
            e_ind = np.dot(Hm[pm].T, a)
        dw = w_c - h_dec[ci]
        trade = (np.abs(dw) >= cfg.trade_tol) & ~fixed
        rec.update(te_ex_ante=res["te_ann"], alpha_ex_ante=float(np.dot(alpha[ci], a)),
                   cost_ex_ante=res["cost"], obj=res["obj"], turnover_plan=float(np.abs(dw[trade]).sum() / 2),
                   n_trades=int(trade.sum()), n_hold=int((w_c > 1e-6).sum()),
                   bench_w_in=float(w_c[wb_c > 0].sum()), max_abs_ind=float(np.max(np.abs(e_ind), initial=0.0)),
                   **{f"exp:{nm}": float(e_sty[k]) for k, nm in enumerate(self.style_names)},
                   **{f"exp:{nm}": float(np.dot(np.nan_to_num(v[t, ci].astype(np.float64)), a))
                      for nm, (v, _) in self.extra.items()},
                   **{f"viol:{k}": v for k, v in (res["viol"] or {}).items()},
                   total_s=time.time() - t_start)
        self.log.append(rec)

        # ---- 目标:不交易的给 NaN(保持实际持仓),其余给优化权重
        tgt = np.zeros(len(self.codes))
        tgt[ci] = w_c
        no_trade = np.zeros(len(self.codes), bool)
        no_trade[ci] = ~trade
        tgt[no_trade] = np.nan
        tgt[out_stuck] = np.nan
        if self.keep_targets:
            self.targets[(int(ctx["phase"]), int(t))] = (ci.astype(np.int32), w_c.astype(np.float32))
        return tgt

    def log_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.log)
        if "cap_relaxed" in df:                          # bool 与缺失混在 object 列里,parquet 不收
            df["cap_relaxed"] = df["cap_relaxed"].astype(float)
        return df



# ================================================================ 评价:成交时点权重的代理基准(2026-10-10 审查 P0 修复)
# 2026-10-10 第二轮修订(前缀 fixa):三个函数搬到 afactor.index_proxy(推荐基准的归属地),这里保留旧名
# 的再导出,OPT.exec_time_bench_returns(p, mode, W, return_weights=...) 等调用逐位不变(同一份代码)。
from afactor.index_proxy import (exec_time_weights, exec_time_bench_arrays,  # noqa: E402,F401
                                 exec_time_bench_returns)
