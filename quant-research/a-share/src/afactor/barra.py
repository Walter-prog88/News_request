"""Barra 式风险模型:暴露度、纯因子收益、因子协方差、特质风险、组合风险与事后归因。

## 来历
2026-09-02 的 `scripts/run_barra_attribution.py` 把 zscore / build_exposures / pure_factor_returns
写在脚本里。下游的指增优化器需要同一套暴露度与风险预测,故 2026-10 搬进库里。
**搬迁逻辑不变**:原函数的签名与默认行为逐位一致,新增参数(`cover`)默认关闭;
核对见 `scripts/run_risk_model.py` 与 `research/ledger/risk_move_check.csv`。
脚本仍从这里 re-export 这些名字,`from run_barra_attribution import build_exposures` 照常可用。

## 两套截面回归,用途不同,不要混用
1. `pure_factor_returns`(原版,归因用,账本 barra_factor_returns.csv 由它产出):
   **完整样本(complete case)**——任一风格暴露缺失的股票整行剔除。副作用:`fu_leverage`
   对金融股(comp_type≠1)恒为 NaN,于是**银行股全部被剔出回归,"行业:银行"纯因子收益恒为 0**
   (账本里 99.9% 的天为 0)。对归因影响小(排雷组合相对池等权的银行暴露很小),
   但对风险模型是致命的:沪深 300 里银行权重 10%+,风险模型会说银行没有行业风险。
2. `risk_factor_regression`(风险模型版,本模块新增,预注册):
   - 估计池 = 决策池 u(V1000,ADV≥2000 万);风格暴露缺失**填 0**(= 市值加权截面均值),
     银行/新股因此留在回归里;某天任一风格在估计池里的非缺失率 < 50% 则当天不出因子收益
     (2020 年动量需 252 日窗口,全体缺失 -> 2020 只作预热,与原版起点一致)。
   - 行业仍用"市值加权和为零"约束识别,回归后把被约束掉的最后一个行业**还原**出来,
     输出全部 J 个行业的纯因子收益,风险模型因此可以用 one-hot 行业暴露。
   - 特质收益对**覆盖池**(上市满 N 日、有成交、有市值,不设 ADV 门槛)的全部股票计算
     = r_i − B_i f,不只是估计池;低流动性股也因此有自己的特质波动历史。

## 时序(两个口径信息滞后都是 2 天)
- open10:信号日 s 的收益 R[s] = twap_open10[s+2]/twap_open10[s+1]−1,在 s+2 的 09:40 实现。
  决策日 t(收盘后定权重,t+1 开盘成交)只能用 s ≤ t−2 的纯因子收益与特质收益。
- tail1450:R[s] = twap_tail8[s+1]/twap_tail8[s]−1,在 s+1 的 14:57 实现;决策在 t 日 14:50,
  s+1 ≤ t−1 -> 同样 s ≤ t−2。
- 暴露度 X[t] 用 t 日收盘(尾盘口径由 TailPanel 自动换成 14:49 前)已知的数据,决策时点可得。
- **风险预测的对象**:R[t] + R[t+1] + … + R[t+4] 之和(t+1 开盘买、t+6 开盘卖的 5 日持有),
  单位是"5 日收益方差"。

## v1 / v2 / v3(2026-10 首轮偏差检验后发现的两处暴露度构造缺陷,三版都留档)
- v1 = 预注册版本:暴露度沿用原版 zscore(先算标准差、再 ±3 截尾)。
- v2 = v1 + 原始描述量稳健截尾(median ± 5·1.4826·MAD,Size/NonlinSize 除外),见 robust_clip。
  起因:Leverage 因子 2021 年 b=2.47、2022 年 b=0.24 —— 一只股票(000820.SZ)资产负债率
  191 倍,把全体 Leverage 暴露压到截面标准差 0.08。
- v3 = v2 + Size 不做 ±3 截尾(**默认**)。起因:Size 用市值加权均值去中心,A 股市值极度右偏,
  ±3 截尾把 V1000 池 15%(2021)→51%(2025)的股票压在同一个 Size=−3 上。
两处都是用"暴露截面标准差≈1、截尾边界上不应有质量点"这条数据质量判据定位的构造缺陷,
修法与倍数在看修后偏差统计量之前定死;协方差/特质风险的预注册参数三版完全相同。
但修复是在看过 v1 的偏差统计量之后做的,所以 v2/v3 的偏差统计量不是纯粹的事前验证,报告里如实标注。
RiskModel() 默认 v3;RiskModel(**RiskModel.VARIANTS["v1"]) 或 RiskModel.load(version="v1") 取 v1。

## 风险模型参数(预注册,不得按回测结果调)
暴露 = Country + 10 个 Barra CNE 风格 + 申万一级行业 one-hot;
因子协方差 = 指数加权(相关性半衰期 90 日、波动半衰期 90 日,零均值假设)+ Newey-West 2 阶
(Bartlett 权重 1−l/3),×5 得 5 日持有期协方差;最少 60 个因子收益观测才出预测。
特质方差 = 特质收益平方的 EWMA(半衰期 90)开方得 σ_i,向同日市值十分位的市值加权平均 σ
做固定强度收缩 σ* = 0.3·σ̄_decile + 0.7·σ_i;观测不足 20 个的股票直接取 σ̄_decile;×5 得 5 日。

## 给优化器的因子形式(不构造 N×N 稠密矩阵)
    Var(a) = || L^T B^T a ||^2 + Σ_i d_i a_i^2,   B = [1, X, H],  L L^T = Σ_factor
`RiskModel.factor_sqrt(t)` 返回 L(优先 Cholesky;行业还原使协方差近奇异时退回特征分解,
二者都满足 L L^T = Σ,优化器只需要这一条性质)。
"""
from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from afactor.config import TMP, MIN_LIST_DAYS, RESEARCH_END, assert_not_sealed
from afactor import factors as FA, ops, newdata as ND, strategy as ST

WINSOR = 3.0        # 暴露度截尾(标准差倍数)


# ================================================================ 搬迁自脚本(逻辑不变)
def nw_t(x, q: int) -> float:
    """Newey-West t 值(Bartlett 权重,q 阶)。样本 < 30 返回 NaN。"""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = float((e ** 2).sum() / n)
    for L in range(1, q + 1):
        s += 2.0 * (1.0 - L / (q + 1.0)) * float((e[L:] * e[:-L]).sum() / n)
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def zscore(x: pd.DataFrame, u: pd.DataFrame, w: pd.DataFrame | None = None,
           cover: pd.DataFrame | None = None, clip: bool = True) -> pd.DataFrame:
    """截面标准化:去均值(可市值加权)、除标准差、按 WINSOR 截尾。

    均值与标准差**只在 u(估计池)上算**。cover=None(默认,原脚本行为)时输出也只留 u 内;
    给定 cover 时,用 u 的矩把 cover 内全部股票标准化(风险模型给池外股票也要暴露度)。
    cover 内、u 内的数值与 cover=None 时逐位相同。
    clip=False 时不做 ±WINSOR 截尾(风险模型 v3 的 Size 用,见 build_exposures)。
    """
    xu = x.where(u)
    mu = ((xu * w).sum(axis=1) / w.where(xu.notna()).sum(axis=1)) if w is not None else xu.mean(axis=1)
    sd = xu.std(axis=1).replace(0, np.nan)
    src = xu if cover is None else x.where(cover)
    z = src.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-WINSOR, WINSOR) if clip else z


def robust_clip(x: pd.DataFrame, u: pd.DataFrame, k: float) -> pd.DataFrame:
    """原始描述量的稳健截尾:逐日截到 u 内 median ± k·1.4826·MAD。MAD=0 的日子不截。

    为什么需要(2026-10 风险模型偏差检验发现):原版 zscore 先算标准差再 ±3 截尾,
    一个数据异常值就能把全体暴露压扁。实例:000820.SZ 的资产负债率 191 倍,
    使 2021-05~08 估计池内 Leverage 暴露的截面标准差只剩 0.08(应≈1),
    该因子收益的量纲随异常值进出跳 10 倍,EWMA 协方差在两段之间严重错配。
    """
    xu = x.where(u)
    med = xu.median(axis=1)
    s = 1.4826 * xu.sub(med, axis=0).abs().median(axis=1)
    s = s.where(s > 0)                                   # NaN 界 = 不截
    return x.clip(lower=med - k * s, upper=med + k * s, axis=0)


def fwd_returns(p, mode) -> pd.DataFrame:
    """按信号日索引的 1 日成交收益:open10 下 R[s] = px[s+2]/px[s+1]−1(s+2 才实现)。"""
    px, _ = ST.exec_prices(p, mode)
    lag = ST.EXEC_MODES[mode]["lag"]
    g = px.shift(-1) / px - 1.0
    return g.shift(-lag).iloc[:len(p.dates) - 1 - lag]


def assert_aligned(rs, rb, tag):
    """引擎回归断言②:corr(策略, 基准.shift(k)) 峰值必须在 k=0。"""
    ks = (-2, -1, 0, 1, 2)
    cs = {k: float(rs.corr(rb.shift(k))) for k in ks}
    peak = max(ks, key=lambda k: cs[k])
    print("  [%s] %-18s %s 峰值 k=%+d" % ("OK" if peak == 0 else "MISALIGNED", tag,
          " ".join(f"{k:+d}:{cs[k]:+.3f}" for k in ks), peak))
    if peak != 0:
        raise AssertionError(f"{tag} 错位 {peak} 天")


def build_exposures(p, u, cover: pd.DataFrame | None = None,
                    raw_clip: float | None = None, size_clip: bool = True) -> dict[str, pd.DataFrame]:
    """Barra CNE 风格暴露度。全部只走 _real/_fu 合规访问器。

    cover=None 为原脚本行为(只输出 u 内)。给定 cover 时标准化的矩与非线性市值的正交化
    仍只在 u 上算,只是把输出扩到 cover —— u 内的数值不变。
    raw_clip=None 为原脚本行为;给定 k 时,除 Size / NonlinSize 外的原始描述量先做
    robust_clip(k) 再标准化(风险模型 v2)。Size 系不截:对数市值没有数据错误式的尾部,
    截了会让权重最大的几只巨盘股暴露并列,而它们正是市值加权基准的主体。
    size_clip=True 为原脚本行为;False 时 Size(及据以构造 NonlinSize 的 z_lncap)不做 ±3 截尾
    (风险模型 v3)。原因:市值加权均值远在等权均值右侧,±3 截尾把 V1000 池里 15%(2021)→51%(2025)
    的股票压在同一个 Size=−3 上,模型看不见小盘与微盘之间的市值差异(见 RISK_MODEL.md)。
    """
    _NO_CLIP = {"Size 市值", "NonlinSize 非线性市值"}
    close = p["close_px"]
    ret = close / close.shift(1) - 1.0
    mv = FA._real(p, "mv")
    lncap = ops.signed_log(mv)
    w = mv.where(u)

    # 市场收益(市值加权)——算 Beta 与 Residual Vol 用
    mkt = (ret * w).sum(axis=1) / w.sum(axis=1)

    # Beta / Residual Volatility:252 日滚动,半衰期加权简化为等权
    cov = ret.rolling(252, min_periods=120).cov(mkt)
    var = mkt.rolling(252, min_periods=120).var()
    beta = cov.div(var, axis=0)
    resid = ret.sub(beta.mul(mkt, axis=0))
    rvol = resid.rolling(252, min_periods=120).std()

    # Momentum:Barra 口径 = 过去 252 日剔除最近 21 日
    mom = close.shift(21) / close.shift(252) - 1.0

    # Non-linear Size:lncap^3 对 lncap 正交化后的残差(正交化的内积只在 u 上算)
    z_lncap = zscore(lncap, u, w, cover=cover, clip=size_clip)
    cube = z_lncap ** 3
    zu = z_lncap if cover is None else z_lncap.where(u)
    cu = cube if cover is None else cube.where(u)
    nls = cube - z_lncap.mul((cu * zu).sum(axis=1) /
                             (zu ** 2).sum(axis=1).replace(0, np.nan), axis=0)

    # Liquidity:Barra 用换手对数,这里用真实换手 21 日均的对数
    turn = FA._real(p, "turn")
    liq = ops.signed_log(turn.rolling(21, min_periods=10).mean())

    raw = {
        "Size 市值":            lncap,
        "Beta 市场敏感":         beta,
        "Momentum 动量":        mom,
        "ResidVol 残差波动":     rvol,
        "NonlinSize 非线性市值": nls,
        "BookToPrice 价值":     _reg(p, "fu_bp"),
        "Liquidity 流动性":      liq,
        "EarningsYield 盈利":    _reg(p, "fu_ep"),
        "Growth 成长":          _reg(p, "fu_rev_yoy"),
        "Leverage 杠杆":        _reg(p, "fu_leverage"),
    }
    out = {}
    for k, v in raw.items():
        if v is None:
            print(f"  ⚠ 跳过 {k}(字段不可用)")
            continue
        v = v.reindex_like(close)
        if raw_clip is not None and k not in _NO_CLIP:
            v = robust_clip(v, u, raw_clip)
        out[k] = zscore(v, u, w, cover=cover, clip=size_clip or k != "Size 市值")
    return out


def _reg(p, name):
    """基本面风格走因子注册表(fu_bp/fu_ep/... 是因子名,不是报表列名)。"""
    if name not in FA.REGISTRY:
        print(f"  ⚠ 注册表里没有 {name}")
        return None
    try:
        return FA.REGISTRY[name][1](p).reindex(index=p.dates, columns=p.codes)\
                 .replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        print(f"  ⚠ 构建 {name} 失败: {type(e).__name__}: {e}")
        return None


def pure_factor_returns(R, X: dict, u, p) -> pd.DataFrame:
    """每日截面 WLS:r = country + styles + industries,权重 sqrt(市值)。

    行业用市值加权和为零的约束识别(把最后一个行业用其余行业线性表出)。
    返回 T x K 的纯因子收益。

    ⚠ 完整样本回归:任一风格缺失的股票整行剔除 -> 金融股(fu_leverage 恒 NaN)全部出局,
    "行业:银行"恒为 0。原样保留以复现账本;风险模型用 risk_factor_regression。
    """
    mv = FA._real(p, "mv")
    L = ND.sw_industry_labels(p.dates, p.codes).reindex_like(R)
    inds = sorted({int(c) for c in pd.unique(L.where(u).values.ravel())
                   if pd.notna(c) and int(c) >= 0})
    names = ND.sw_industry_names()
    keys = list(X.keys())
    cols = ["Country"] + keys + [f"行业:{names.get(c, c)}" for c in inds[:-1]]
    out = pd.DataFrame(index=R.index, columns=cols, dtype=float)

    Xarr = {k: v.reindex_like(R).to_numpy() for k, v in X.items()}
    Larr, Rarr, Uarr = L.to_numpy(), R.to_numpy(), u.reindex_like(R).to_numpy(bool)
    Warr = np.sqrt(np.maximum(mv.reindex_like(R).to_numpy(), 0.0))
    mvarr = mv.reindex_like(R).to_numpy()

    for ti in range(len(R.index)):
        m = Uarr[ti] & np.isfinite(Rarr[ti]) & np.isfinite(Warr[ti]) & (Warr[ti] > 0)
        for k in keys:
            m &= np.isfinite(Xarr[k][ti])
        m &= np.isfinite(Larr[ti])
        if m.sum() < 200:
            continue
        idx = np.flatnonzero(m)
        S = np.column_stack([Xarr[k][ti][idx] for k in keys])
        lab = Larr[ti][idx].astype(int)
        cap = mvarr[ti][idx]
        # 行业哑变量 + 市值加权和为零约束:最后一个行业用 -(cap_j/cap_last) 表出
        last = inds[-1]
        D = np.zeros((len(idx), len(inds) - 1))
        capsum = np.array([cap[lab == c].sum() for c in inds])
        for j, c in enumerate(inds[:-1]):
            D[lab == c, j] = 1.0
            if capsum[-1] > 0:
                D[lab == last, j] = -capsum[j] / capsum[-1]
        A = np.column_stack([np.ones(len(idx)), S, D])
        sw = np.sqrt(Warr[ti][idx])
        Aw, yw = A * sw[:, None], Rarr[ti][idx] * sw
        try:
            f, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out.iloc[ti, :len(f)] = f
    return out.dropna(how="all")


# ================================================================ 风险模型版回归
INFO_LAG = 2            # 决策日 t 只能用 s ≤ t−2 的因子收益/特质收益(两个口径都是 2,见模块 docstring)
MIN_EST = 200           # 估计池最少股票数(与原版一致)
MIN_STYLE_COVER = 0.5   # 某风格在估计池内非缺失率低于此值 -> 当天不出因子收益


def coverage_mask(p) -> pd.DataFrame:
    """风险模型覆盖池:当日有成交、上市满 MIN_LIST_DAYS、真实总市值 > 0。不设 ADV 门槛。
    只用 t 日收盘已知信息(与决策池同口径,只是去掉了流动性门槛)。"""
    mv = FA._real(p, "mv")
    return (p["valid"] & (p["days_listed"] >= MIN_LIST_DAYS) & (mv > 0)).fillna(False)


def industry_index(p, u) -> tuple[np.ndarray, list[str]]:
    """申万一级行业:返回每只股票的行业列号(-1 = 无行业或该行业从未进过估计池)与行业名。

    行业集合 = 研究期内在估计池 u 里出现过的申万一级行业(与原版 inds 相同口径)。
    ⚠ stock_sw_industry.csv 是**当前快照**(无 in/out 日期),给 2021 年的股票贴 2026 年的行业。
    """
    L = ND.sw_industry_labels(pd.DatetimeIndex(p.dates[:1]), p.codes).iloc[0].to_numpy().astype(int)
    U = u.to_numpy(bool)
    present = np.unique(L[U.any(axis=0)])
    inds = sorted(int(c) for c in present if c >= 0)
    names = ND.sw_industry_names()
    col = {c: j for j, c in enumerate(inds)}
    ind_idx = np.array([col.get(int(c), -1) for c in L], dtype=np.int16)
    return ind_idx, [f"行业:{names.get(c, c)}" for c in inds]


def risk_factor_regression(R: pd.DataFrame, Xs: np.ndarray, ind_idx: np.ndarray, J: int,
                           u: pd.DataFrame, cover: pd.DataFrame, mv: pd.DataFrame,
                           style_names: list[str], ind_names: list[str]):
    # numpy 2.0 + macOS Accelerate 的 matmul 会对正常有限输入误报 divide/overflow/invalid 警告
    # (随机正态矩阵相乘即可复现)。这里静音,正确性由 isfinite 检查与单测保证。
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _risk_factor_regression(R, Xs, ind_idx, J, u, cover, mv, style_names, ind_names)


def _risk_factor_regression(R, Xs, ind_idx, J, u, cover, mv, style_names, ind_names):
    """风险模型版逐日截面 WLS(见模块 docstring 第 2 条)。

    参数
      R       信号日索引的 1 日收益 [T_R x N]
      Xs      风格暴露 [T_P x N x K](T_P = 面板日期数,前 T_R 行与 R 对齐),cover 外为 NaN
      ind_idx 行业列号 [N],-1 = 无行业
    返回
      F     DataFrame [日期 x (1+K+J)]:Country、K 风格、J 个行业(含还原出的最后一个)
      resid np.ndarray float32 [T_R x N]:覆盖池内 r − B f(其余 NaN)
      diag  DataFrame:每日估计池股数、覆盖池股数、WLS R²
    """
    T, N = R.shape
    K = Xs.shape[2]
    Rarr = R.to_numpy(np.float64)
    Uarr = u.reindex_like(R).fillna(False).to_numpy(bool)
    Carr = cover.reindex_like(R).fillna(False).to_numpy(bool)
    mvarr = mv.reindex_like(R).to_numpy(np.float64)
    has_ind = ind_idx >= 0
    cols = ["Country"] + list(style_names) + list(ind_names)
    F = np.full((T, 1 + K + J), np.nan)
    resid = np.full((T, N), np.nan, np.float32)
    diag = np.full((T, 3), np.nan)

    for ti in range(T):
        r = Rarr[ti]
        Xi = Xs[ti].astype(np.float64)
        m = Uarr[ti] & Carr[ti] & np.isfinite(r) & np.isfinite(mvarr[ti]) & (mvarr[ti] > 0)
        if m.sum() < MIN_EST:
            continue
        fin = np.isfinite(Xi[m])
        if (fin.mean(axis=0) < MIN_STYLE_COVER).any():
            continue                                    # 某风格整体缺失(如 2020 年的动量)
        idx = np.flatnonzero(m)
        S = np.nan_to_num(Xi[idx], nan=0.0)            # 缺失 = 截面(市值加权)均值
        lab = ind_idx[idx]
        cap = mvarr[ti][idx]
        capsum = np.array([cap[lab == j].sum() for j in range(J)])
        # 约束:Σ_j capsum_j f_j = 0,最后一个行业 f_{J-1} = −Σ_{j<J-1} (capsum_j/capsum_{J-1}) f_j
        D = np.zeros((len(idx), J - 1))
        for j in range(J - 1):
            D[lab == j, j] = 1.0
            if capsum[-1] > 0:
                D[lab == J - 1, j] = -capsum[j] / capsum[-1]
        A = np.column_stack([np.ones(len(idx)), S, D])
        sw = cap ** 0.25                                  # 行乘 cap^(1/4) = WLS 权重 sqrt(cap)
        try:
            f, *_ = np.linalg.lstsq(A * sw[:, None], r[idx] * sw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        f_ind = np.zeros(J)
        f_ind[:J - 1] = f[1 + K:]
        if capsum[-1] > 0:
            f_ind[J - 1] = -float(capsum[:J - 1] @ f[1 + K:]) / capsum[-1]
        F[ti] = np.concatenate([[f[0]], f[1:1 + K], f_ind])

        # 估计池 WLS R²(诊断)
        fit = A @ f
        wt = cap ** 0.5
        ybar = np.average(r[idx], weights=wt)
        diag[ti, 2] = 1.0 - np.sum(wt * (r[idx] - fit) ** 2) / np.sum(wt * (r[idx] - ybar) ** 2)

        # 覆盖池特质收益:r − (f_c + X f_s + f_ind)
        c = Carr[ti] & np.isfinite(r)
        ci = np.flatnonzero(c)
        Xc = np.nan_to_num(Xi[ci], nan=0.0)
        pred = f[0] + Xc @ f[1:1 + K]
        hi = has_ind[ci]
        pred[hi] += f_ind[ind_idx[ci][hi]]
        resid[ti, ci] = (r[ci] - pred).astype(np.float32)
        diag[ti, 0], diag[ti, 1] = len(idx), len(ci)

    Fdf = pd.DataFrame(F, index=R.index, columns=cols)
    ok = np.isfinite(F).all(axis=1)
    diag = pd.DataFrame(diag, index=R.index, columns=["n_est", "n_cover", "wls_r2"])
    return Fdf[ok], resid, diag[ok], ok


# ================================================================ 协方差与特质方差(可独立测试)
def _ewma_nw(F: np.ndarray, half_life: float, nw_lags: int) -> np.ndarray:
    """逐行递推的指数加权 + Newey-West 协方差(日频,零均值)。

    第 s 行输出 **只用 F[0..s]**;权重 λ^(s−τ),λ = 0.5^(1/half_life);
    Σ_NW = Γ0 + Σ_{l=1..q} (1 − l/(q+1)) (Γ_l + Γ_l^T),Γ_l = Σ_τ w_τ f_τ f_{τ−l}^T / Σ_τ w_τ。
    """
    T, M = F.shape
    lam = 0.5 ** (1.0 / half_life)
    S0 = np.zeros((M, M)); Sl = [np.zeros((M, M)) for _ in range(nw_lags)]
    W = 0.0
    out = np.empty((T, M, M))
    for s in range(T):
        f = F[s]
        S0 = lam * S0 + np.outer(f, f)
        for l in range(1, nw_lags + 1):
            Sl[l - 1] *= lam
            if s - l >= 0:
                Sl[l - 1] += np.outer(f, F[s - l])
        W = lam * W + 1.0
        C = S0 / W
        for l in range(1, nw_lags + 1):
            G = Sl[l - 1] / W
            C = C + (1.0 - l / (nw_lags + 1.0)) * (G + G.T)
        out[s] = C
    return out


def factor_cov_series(F: np.ndarray, half_life_corr: float = 90, half_life_vol: float = 90,
                      nw_lags: int = 2, horizon: int = 5) -> np.ndarray:
    """因子协方差序列 [T_F x M x M]:第 s 行 = 只用 F[0..s] 估计的 horizon 日协方差。

    波动与相关可用不同半衰期(Barra USE4 做法):Σ = D_vol · Corr_corr · D_vol。
    半衰期相等时退化为单一 EWMA。最后做对称化。PSD 修复在 RiskModel 里按决策日做。
    """
    Cv = _ewma_nw(F, half_life_vol, nw_lags)
    if half_life_corr == half_life_vol:
        C = Cv
    else:
        Cc = _ewma_nw(F, half_life_corr, nw_lags)
        dv = np.sqrt(np.clip(np.einsum("tii->ti", Cv), 0, None))
        dc = np.sqrt(np.clip(np.einsum("tii->ti", Cc), 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = Cc / (dc[:, :, None] * dc[:, None, :])
        corr = np.nan_to_num(corr)
        C = corr * dv[:, :, None] * dv[:, None, :]
    C = 0.5 * (C + np.transpose(C, (0, 2, 1)))
    return C * horizon


def ewma_spec_var(resid: np.ndarray, half_life: float = 90) -> tuple[np.ndarray, np.ndarray]:
    """特质收益平方的 EWMA(零均值)。第 s 行只用 resid[0..s]。

    缺失日(停牌/不在覆盖池)不入样,但权重按日历(行)衰减:S、W 每行都乘 λ,
    只有有观测时才加 —— 比值 = 有观测日的按龄加权均值。返回 (日方差, 累计观测数)。
    """
    T, N = resid.shape
    lam = 0.5 ** (1.0 / half_life)
    S = np.zeros(N); W = np.zeros(N); n = np.zeros(N, np.int32)
    var = np.full((T, N), np.nan, np.float32)
    nobs = np.zeros((T, N), np.int32)
    for s in range(T):
        x = resid[s].astype(np.float64)
        ok = np.isfinite(x)
        S *= lam; W *= lam
        S[ok] += x[ok] ** 2; W[ok] += 1.0; n[ok] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            var[s] = np.where(W > 0, S / W, np.nan)
        nobs[s] = n
    return var, nobs


def shrink_spec_vol(sig: np.ndarray, nobs: np.ndarray, mv: np.ndarray, mask: np.ndarray,
                    shrink: float = 0.3, n_bucket: int = 10, min_obs: int = 20) -> np.ndarray:
    """同市值十分位贝叶斯收缩(固定强度):σ* = shrink·σ̄_bucket + (1−shrink)·σ_i。

    十分位按决策日 t 覆盖池内的真实总市值切;σ̄_bucket = 桶内有效 σ 的市值加权均值;
    观测数 < min_obs 的股票不参与 σ̄ 计算,直接取 σ̄_bucket(其自身 σ 不可信)。
    返回长度 N 的 σ*(mask 外 NaN)。
    """
    out = np.full(sig.shape, np.nan)
    mk = mask & np.isfinite(mv) & (mv > 0)
    if mk.sum() < n_bucket:
        return out
    idx = np.flatnonzero(mk)
    rk = pd.Series(mv[idx]).rank(method="first").to_numpy()
    b = np.minimum((rk - 1) * n_bucket // len(idx), n_bucket - 1).astype(int)
    good = np.isfinite(sig[idx]) & (nobs[idx] >= min_obs)
    cap = mv[idx]
    gmean = np.average(sig[idx][good], weights=cap[good]) if good.any() else np.nan
    for k in range(n_bucket):
        inb = b == k
        g = inb & good
        mbar = np.average(sig[idx][g], weights=cap[g]) if g.any() else gmean
        v = np.where(good[inb], shrink * mbar + (1.0 - shrink) * sig[idx][inb], mbar)
        out[idx[inb]] = v
    return out


def decision_source_rows(pos_F: np.ndarray, T: int, info_lag: int = INFO_LAG) -> np.ndarray:
    """决策日 i(面板位置)可用的最后一行因子收益的**行号**(在 F 中);−1 = 尚无可用行。

    pos_F 是 F 各行在面板中的位置(递增,可有缺口)。可用条件 pos_F ≤ i − info_lag:
    open10 下 s 日的因子收益在 s+2 开盘才实现,决策日 t 收盘时最多用到 s = t−2。
    """
    pos_F = np.asarray(pos_F)
    assert np.all(np.diff(pos_F) > 0)
    return np.searchsorted(pos_F, np.arange(T) - info_lag, side="right") - 1


def psd_sqrt(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (C_psd, L),L L^T = C_psd。先试 Cholesky,失败则特征分解并把负特征值截到 0。"""
    C = 0.5 * (C + C.T)
    try:
        L = np.linalg.cholesky(C)
        return C, L
    except np.linalg.LinAlgError:
        ev, V = np.linalg.eigh(C)
        ev = np.clip(ev, 0.0, None)
        L = V * np.sqrt(ev)[None, :]
        return L @ L.T, L


# ================================================================ RiskModel
@dataclass
class RiskSnapshot:
    """某决策日的风险模型切片(只含覆盖池内股票,行顺序 = codes)。"""
    date: pd.Timestamp
    mask: np.ndarray          # [N_all] bool,对 p.codes
    codes: pd.Index           # 覆盖池股票代码
    X: np.ndarray             # [n x K] 风格暴露(缺失已填 0)
    H: np.ndarray             # [n x J] 行业 one-hot(无行业为全 0 行)
    cov: np.ndarray           # [(1+K+J) x (1+K+J)] 5 日因子协方差,顺序 Country, 风格, 行业
    d: np.ndarray             # [n] 5 日特质方差
    L: np.ndarray             # L L^T = cov
    factor_names: list

    @property
    def B(self) -> np.ndarray:
        """完整暴露矩阵 [1, X, H]。"""
        return np.column_stack([np.ones(len(self.codes)), self.X, self.H])


class RiskModel:
    """Barra 式风险模型。用法:

        rm = RiskModel().build(Panel(), mode="open10")   # 首次计算并缓存到 cache/_tmp/risk_open10_*
        rm = RiskModel.load("open10")                    # 之后直接读缓存(memmap)
        mask, X, H, cov, d = rm.at("2024-03-01")
        rm.active_risk(w, wb, "2024-03-01")              # 5 日主动风险(标准差)

    at(t) 的 t 是**决策日**(t 日收盘后定权重、t+1 开盘成交),预测对象是 R[t..t+4] 之和。
    """

    VERSION = 1

    def __init__(self, half_life_corr: float = 90, half_life_vol: float = 90, nw_lags: int = 2,
                 horizon: int = 5, spec_half_life: float = 90, shrink: float = 0.3,
                 n_bucket: int = 10, min_cov_obs: int = 60, min_spec_obs: int = 20,
                 raw_clip: float | None = 5.0, size_clip: bool = False, cache_dir: Path = TMP):
        self.params = dict(half_life_corr=half_life_corr, half_life_vol=half_life_vol,
                           nw_lags=nw_lags, horizon=horizon, spec_half_life=spec_half_life,
                           shrink=shrink, n_bucket=n_bucket, min_cov_obs=min_cov_obs,
                           min_spec_obs=min_spec_obs, info_lag=INFO_LAG,
                           min_style_cover=MIN_STYLE_COVER, winsor=WINSOR, raw_clip=raw_clip,
                           size_clip=size_clip, version=self.VERSION)
        self.cache_dir = Path(cache_dir)
        self.mode = None

    # ------------------------------------------------------------ 缓存路径
    VARIANTS = {"v1": dict(raw_clip=None, size_clip=True),     # 预注册:原版暴露
                "v2": dict(raw_clip=5.0, size_clip=True),      # + 原始描述量稳健截尾
                "v3": dict(raw_clip=5.0, size_clip=False)}     # + Size 不做 ±3 截尾(默认)

    @property
    def variant(self) -> str:
        for k, v in self.VARIANTS.items():
            if self.params.get("raw_clip") == v["raw_clip"] and \
                    self.params.get("size_clip", True) == v["size_clip"]:
                return k
        return "custom"

    @property
    def tag(self) -> str:
        """缓存标签:默认 v3 无后缀,其余加 _v1/_v2。"""
        return "" if self.variant == "v3" else f"_{self.variant}"

    def _f(self, mode: str, what: str) -> Path:
        return self.cache_dir / f"risk_{mode}{self.tag}_{what}"

    # ------------------------------------------------------------ 构建
    def build(self, p, mode: str = "open10", force: bool = False, verbose: bool = True) -> "RiskModel":
        """一次性计算暴露度 / 纯因子收益 / 协方差 / 特质方差并缓存。缓存参数一致时直接读。"""
        if ST.EXEC_MODES[mode]["tail"] != bool(getattr(p, "_is_tail", False)):
            raise ValueError(f"mode={mode} 需要 {'TailPanel' if ST.EXEC_MODES[mode]['tail'] else 'Panel'}")
        if getattr(p, "sealed", False):
            raise PermissionError("风险模型只在研究期面板上构建,不得读封存样本。")
        assert_not_sealed(p.dates)
        assert p.dates.max() <= pd.Timestamp(RESEARCH_END)
        meta_f = self._f(mode, "meta.json")
        if meta_f.exists() and not force:
            meta = json.load(open(meta_f))
            if meta.get("params") == self.params and meta.get("n_dates") == len(p.dates) \
                    and meta.get("n_codes") == len(p.codes):
                if verbose:
                    print(f"  [RiskModel] 读缓存 {meta_f.name}")
                return self._load(mode)
        import time
        t0 = time.time()
        log = (lambda *a: print("  [RiskModel]", *a, f"({time.time() - t0:.0f}s)", flush=True)) \
            if verbose else (lambda *a: None)
        FA.build_all()
        pr = self.params
        dates, codes = p.dates, p.codes
        T, N = len(dates), len(codes)

        R = fwd_returns(p, mode)
        u = p["universe"].reindex(index=dates, columns=codes).fillna(False)
        cover = coverage_mask(p).reindex(index=dates, columns=codes).fillna(False)
        mv = FA._real(p, "mv").reindex(index=dates, columns=codes)
        log("构建暴露度(cover=覆盖池)…")
        Xd = build_exposures(p, u, cover=cover, raw_clip=pr["raw_clip"], size_clip=pr["size_clip"])
        style_names = list(Xd.keys())
        K = len(style_names)
        Xs = np.full((T, N, K), np.nan, np.float32)
        for k, nm in enumerate(style_names):
            Xs[:, :, k] = Xd[nm].reindex(index=dates, columns=codes).to_numpy(np.float32)
        del Xd
        Cv = cover.to_numpy(bool)
        Xs[~Cv] = np.nan
        ind_idx, ind_names = industry_index(p, u)
        J = len(ind_names)
        log(f"风格 {K} 个、行业 {J} 个;逐日 WLS…")

        # ⚠ 回归必须拿**未填充**的暴露:MIN_STYLE_COVER 门槛靠 NaN 判断某风格当天是否整体缺失
        # (2020 年动量全缺)。先填 0 再回归会让门槛失效,2020 年会产出动量列恒 0 的退化因子收益。
        F, resid, diag, okrow = risk_factor_regression(R, Xs, ind_idx, J, u, cover, mv,
                                                       style_names, ind_names)
        Xs = np.where(Cv[:, :, None] & ~np.isfinite(Xs), np.float32(0.0), Xs)   # 覆盖池内缺失填 0
        factor_names = list(F.columns)
        M = len(factor_names)
        log(f"纯因子收益 {len(F)} 天 [{F.index[0].date()} ~ {F.index[-1].date()}]")

        # ---- 因子协方差:按 F 的行递推,再映射到决策日(只用 s ≤ t−INFO_LAG)
        Fv = F.to_numpy(np.float64)
        Cs = factor_cov_series(Fv, pr["half_life_corr"], pr["half_life_vol"],
                               pr["nw_lags"], pr["horizon"])
        pos_F = dates.get_indexer(F.index)                # F 行在面板中的位置
        assert (pos_F >= 0).all()
        src = decision_source_rows(pos_F, T, INFO_LAG)
        cov = np.full((T, M, M), np.nan)
        for i in range(T):
            s = src[i]
            if s < 0 or s + 1 < pr["min_cov_obs"]:
                continue
            cov[i], _ = psd_sqrt(Cs[s])
        log("因子协方差完成")

        # ---- 特质方差:对**面板的每个 R 行**递推(R 行与面板前 T_R 行对齐,缺失日自然跳过)
        resid_full = np.full((T, N), np.nan, np.float32)
        resid_full[np.flatnonzero(okrow)] = resid[okrow]
        var_d, nobs = ewma_spec_var(resid_full[:len(R)], pr["spec_half_life"])
        spec = np.full((T, N), np.nan, np.float32)
        mvv = mv.to_numpy(np.float64)
        avail = np.isfinite(cov[:, 0, 0])
        mask = Cv & avail[:, None] & np.isfinite(Xs[:, :, 0])
        for i in range(T):
            j = i - INFO_LAG                                 # 特质收益只能用到 t−2
            if not avail[i] or j < 0 or j >= var_d.shape[0]:
                mask[i] = False
                continue
            sig = np.sqrt(var_d[j].astype(np.float64))
            ss = shrink_spec_vol(sig, nobs[j], mvv[i], mask[i], pr["shrink"],
                                 pr["n_bucket"], pr["min_spec_obs"])
            spec[i] = (ss ** 2 * pr["horizon"]).astype(np.float32)
        mask &= np.isfinite(spec)
        log("特质方差完成")

        # ---- 落盘
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._f(mode, "X.npy"), Xs)
        np.save(self._f(mode, "ind.npy"), ind_idx)
        np.save(self._f(mode, "cov.npy"), cov)
        np.save(self._f(mode, "spec.npy"), spec)
        np.save(self._f(mode, "mask.npy"), mask)
        np.save(self._f(mode, "resid.npy"), resid_full)
        F.to_parquet(self._f(mode, "F.parquet"))
        diag.to_parquet(self._f(mode, "diag.parquet"))
        pd.DataFrame({"date": dates}).to_parquet(self._f(mode, "dates.parquet"))
        pd.DataFrame({"code": codes}).to_parquet(self._f(mode, "codes.parquet"))
        meta = dict(params=self.params, mode=mode, n_dates=T, n_codes=N,
                    style_names=style_names, ind_names=ind_names, factor_names=factor_names,
                    F_first=str(F.index[0].date()), F_last=str(F.index[-1].date()),
                    first_forecast=str(dates[np.flatnonzero(avail)[0]].date()) if avail.any() else None,
                    built_at=pd.Timestamp.now().isoformat(timespec="seconds"))
        json.dump(meta, open(meta_f, "w"), ensure_ascii=False, indent=1)
        log("缓存完成")
        return self._load(mode)

    @classmethod
    def load(cls, mode: str = "open10", version: str = "v3", cache_dir: Path = TMP) -> "RiskModel":
        """version ∈ VARIANTS:"v3"(默认)/"v2"/"v1"(预注册原版暴露)。"""
        rm = cls(cache_dir=cache_dir, **cls.VARIANTS[version])
        meta = json.load(open(rm._f(mode, "meta.json")))
        rm.params = meta["params"]
        return rm._load(mode)

    def _load(self, mode: str) -> "RiskModel":
        meta = json.load(open(self._f(mode, "meta.json")))
        self.mode = mode
        self.meta = meta
        self.style_names = meta["style_names"]
        self.ind_names = meta["ind_names"]
        self.factor_names = meta["factor_names"]
        self.dates = pd.DatetimeIndex(pd.read_parquet(self._f(mode, "dates.parquet"))["date"])
        self.codes = pd.Index(pd.read_parquet(self._f(mode, "codes.parquet"))["code"])
        assert_not_sealed(self.dates)
        self.X = np.load(self._f(mode, "X.npy"), mmap_mode="r")
        self.ind = np.load(self._f(mode, "ind.npy"))
        self.cov = np.load(self._f(mode, "cov.npy"))
        self.spec = np.load(self._f(mode, "spec.npy"), mmap_mode="r")
        self.mask = np.load(self._f(mode, "mask.npy"), mmap_mode="r")
        self.F = pd.read_parquet(self._f(mode, "F.parquet"))
        self.diag = pd.read_parquet(self._f(mode, "diag.parquet"))
        self.K, self.J = len(self.style_names), len(self.ind_names)
        return self

    def resid(self) -> np.ndarray:
        """覆盖池特质收益 [T x N](信号日索引,s 日的值在 s+2 才实现)。诊断用。"""
        return np.load(self._f(self.mode, "resid.npy"), mmap_mode="r")

    # ------------------------------------------------------------ 查询
    def _pos(self, t) -> int:
        i = int(self.dates.searchsorted(pd.Timestamp(t), side="right") - 1)
        if i < 0:
            raise KeyError(f"{t} 早于面板起点")
        return i

    def available(self, t) -> bool:
        return bool(np.isfinite(self.cov[self._pos(t), 0, 0]))

    def _onehot(self, ind: np.ndarray) -> np.ndarray:
        H = np.zeros((len(ind), self.J))
        ok = ind >= 0
        H[np.flatnonzero(ok), ind[ok]] = 1.0
        return H

    def at(self, t):
        """返回 (codes_mask, X[n×K], H[n×J], cov[(1+K+J)²], d[n]),n = codes_mask.sum()。

        codes_mask 是对 self.codes(= p.codes)的布尔向量;X/H/d 只含 mask 内股票、顺序同 codes。
        cov 是 5 日因子协方差,因子顺序 [Country] + 风格 + 行业(= self.factor_names);
        完整暴露 B = [1, X, H]。d 是 5 日特质方差。t 不是交易日时取之前最近的交易日。
        """
        i = self._pos(t)
        if not np.isfinite(self.cov[i, 0, 0]):
            raise KeyError(f"{t}: 风险模型尚无预测(因子收益观测 < {self.params['min_cov_obs']})")
        m = np.asarray(self.mask[i]).copy()
        X = np.asarray(self.X[i])[m].astype(np.float64)
        H = self._onehot(self.ind[m])
        d = np.asarray(self.spec[i])[m].astype(np.float64)
        return m, X, H, self.cov[i].copy(), d

    def factor_sqrt(self, t) -> np.ndarray:
        """L 使 L L^T = 因子协方差(at(t) 的 cov)。"""
        return psd_sqrt(self.cov[self._pos(t)])[1]

    def snapshot(self, t) -> RiskSnapshot:
        m, X, H, cov, d = self.at(t)
        return RiskSnapshot(date=self.dates[self._pos(t)], mask=m, codes=self.codes[m], X=X, H=H,
                            cov=cov, d=d, L=psd_sqrt(cov)[1], factor_names=list(self.factor_names))

    # ------------------------------------------------------------ 组合风险
    def _vec(self, w) -> np.ndarray:
        """权重统一成对 self.codes 的 numpy 向量(Series 按代码对齐,缺失为 0)。"""
        if w is None:
            return np.zeros(len(self.codes))
        if isinstance(w, pd.Series):
            return w.reindex(self.codes).fillna(0.0).to_numpy(np.float64)
        w = np.asarray(w, np.float64)
        if w.shape != (len(self.codes),):
            raise ValueError(f"权重长度 {w.shape} ≠ 股票数 {len(self.codes)};请传按代码索引的 Series")
        return np.nan_to_num(w)

    def _active(self, w, wb, t, strict: bool):
        a = self._vec(w) - self._vec(wb)
        m, X, H, cov, d = self.at(t)
        out = float(np.abs(a[~m]).sum())
        if out > 1e-6:
            msg = f"{t}: 覆盖池外有 {out:.4%} 的主动权重(无暴露度/特质风险)"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg + ",已忽略")
        return a[m], X, H, cov, d, out

    def active_exposure(self, w, wb, t, strict: bool = True) -> pd.Series:
        """主动暴露 B^T (w − wb),index = factor_names(Country 暴露 = 主动权重之和)。"""
        a, X, H, _, _, _ = self._active(w, wb, t, strict)
        e = np.concatenate([[a.sum()], X.T @ a, H.T @ a])
        return pd.Series(e, index=self.factor_names)

    def active_risk(self, w, wb, t, detail: bool = False, strict: bool = True):
        """5 日主动风险(标准差,收益单位)。年化 ≈ × sqrt(242/5)。

        因子形式:Var = ||L^T B^T a||² + Σ d_i a_i²,不构造 N×N 矩阵。
        detail=True 返回 dict:total/factor/specific/style/industry/country 的 5 日方差拆分与年化波动。
        """
        a, X, H, cov, d, out = self._active(w, wb, t, strict)
        e = np.concatenate([[a.sum()], X.T @ a, H.T @ a])
        L = psd_sqrt(cov)[1]
        vf = float(np.sum((L.T @ e) ** 2))
        vs = float(np.sum(d * a * a))
        if not detail:
            return float(np.sqrt(vf + vs))
        K = self.K
        blk = {"country": slice(0, 1), "style": slice(1, 1 + K), "industry": slice(1 + K, None)}
        # 分块方差:各块自身方差(不含块间协方差;块间项 = factor − Σ 块)
        parts = {k: float(e[s] @ cov[s, s] @ e[s]) for k, s in blk.items()}
        ann = np.sqrt(242.0 / self.params["horizon"])
        return dict(var_total=vf + vs, var_factor=vf, var_specific=vs, **{f"var_{k}": v for k, v in parts.items()},
                    var_cross=vf - sum(parts.values()), vol5_total=np.sqrt(vf + vs),
                    vol_ann_total=np.sqrt(vf + vs) * ann, uncovered_weight=out)

    # ------------------------------------------------------------ 事后归因
    def attribute(self, active_weights_daily: pd.DataFrame, rets: pd.DataFrame,
                  by_factor: bool = False):
        """事后归因:把逐日主动收益拆成 Country / 风格 / 行业 / 特质。

        口径:active_weights_daily 的第 t 行是**决策日 t** 定下的主动权重;rets 的第 t 行必须是
        同一权重实际吃到的收益 —— 即本模块 fwd_returns 的信号日索引(open10:px[t+2]/px[t+1]−1)。
        暴露用 t 日的 B_t,因子收益用回归得到的 f_t(t+2 才实现,事后归因可用)。
        total = Σ a_i r_i(r 缺失记 0,与回测引擎"不可成交记 0"一致);
        specific = total − country − style − industry(含覆盖池外权重的全部收益,单列 uncovered_w)。
        返回逐日 DataFrame;by_factor=True 时另返回逐日逐因子贡献。
        """
        A = active_weights_daily.reindex(columns=self.codes).fillna(0.0)
        Rr = rets.reindex(index=A.index, columns=self.codes)
        rows, fac_rows, idx = [], [], []
        K = self.K
        for t in A.index:
            if t not in self.F.index:
                continue
            i = self._pos(t)
            if self.dates[i] != pd.Timestamp(t):
                continue
            a = A.loc[t].to_numpy(np.float64)
            r = np.nan_to_num(Rr.loc[t].to_numpy(np.float64))
            m = np.asarray(self.mask[i]) if np.isfinite(self.cov[i, 0, 0]) else \
                np.isfinite(np.asarray(self.X[i])[:, 0])        # 风险预测尚未起步的早期日也能归因
            X = np.nan_to_num(np.asarray(self.X[i])[m].astype(np.float64))
            H = self._onehot(self.ind[m])
            am = a[m]
            e = np.concatenate([[am.sum()], X.T @ am, H.T @ am])
            f = self.F.loc[t].to_numpy(np.float64)
            c = e * f
            tot = float(a @ r)
            rows.append([tot, c[0], c[1:1 + K].sum(), c[1 + K:].sum(),
                         tot - c.sum(), float(np.abs(a[~m]).sum())])
            fac_rows.append(c)
            idx.append(pd.Timestamp(t))
        out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx),
                           columns=["total", "country", "style", "industry", "specific", "uncovered_w"])
        if by_factor:
            return out, pd.DataFrame(fac_rows, index=out.index, columns=self.factor_names)
        return out
