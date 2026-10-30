"""回测引擎回归测试(前缀 eng,2026-10-10)。读真实研究期面板,慢(约 3~6 分钟)。

运行:  python3 -m pytest tests/test_engine_regression.py -q

金标准 tests/golden_engine.json 由 scripts/eng_make_golden.py 在**改造前**的 strategy.py 上生成
(文件里记录了改造前 strategy.py 的 sha256),本测试逐日比对改造后的数值。

覆盖:
  1 金标准逐位一致(容差 1e-10)                    6 冲击成本随 capital 单调不减
  2 exposure <= 1(含冲击模型、callable、vwap1)      7 run_weights(topn_weights) == run_strategy(top)
  3 corr(策略, 基准.shift(k)) 峰值在 k=0             8 callable 返回固定 W == ndarray
  4 全池等权 vs 池基准:超额 ≈ -成本                  9 backtest.py 与 strategy.py 同配置逐位一致
  5 CostModel(Y=0, 价差=0) 与旧引擎逐位一致         另:成本系数与引擎同一模型、估计量无前视、vwap1 口径
  4b(2026-10-10 第二轮修订 fixa)全池等权 rebalance=k vs **同频**池等权基准 benchmark(rebalance=k):
     超额 ≈ -成本(open10 / vwap1,k ∈ {1,5});同频基准的实现核对(k=1 特例 = 旧路径、买入持有恒等式、锚点无关)
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.panel import Panel                      # noqa: E402
from afactor import strategy as ST, backtest as BT, impact as IM   # noqa: E402
from afactor.config import TMP                       # noqa: E402

TOL = 1e-10
GOLD_PATH = ROOT / "tests" / "golden_engine.json"
SCORE_PATH = TMP / "comp_v5_score_open10.parquet"

pytestmark = pytest.mark.skipif(not (GOLD_PATH.exists() and SCORE_PATH.exists()),
                                reason="需要真实数据面板与金标准文件")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def p():
    return Panel()


@pytest.fixture(scope="module")
def score():
    return pd.read_parquet(SCORE_PATH)


@pytest.fixture(scope="module")
def gold():
    return json.loads(GOLD_PATH.read_text())


@pytest.fixture(scope="module")
def m2024(p):
    return np.asarray(p.dates.year == 2024)


@pytest.fixture(scope="module")
def cm0(p):
    return IM.CostModel(p, impact_y=0.0, spread_mult=0.0)


@pytest.fixture(scope="module")
def cm1(p):
    return IM.CostModel(p, impact_y=1.0)


def _run_gold_config(cfg: dict, score, p, mask, **extra):
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind == "strategy":
        version, mode = cfg.pop("version"), cfg.pop("mode")
        if "buffer" in cfg:
            cfg["buffer"] = tuple(cfg["buffer"])
        return ST.run_strategy(score, p, version, mode, date_mask=mask, **cfg, **extra)
    if kind == "benchmark":
        return ST.benchmark(p, cfg["version"], cfg["mode"], date_mask=mask)
    if kind == "backtest_standard":
        return BT.run_standard(score, p, date_mask=mask)
    raise ValueError(kind)


def _mask(p, wname):
    return None if wname == "full" else np.asarray(p.dates.year == 2024)


def _strategy_keys(gold):
    return [k for k in gold["results"] if gold["configs"][k.split("|")[0]]["kind"] == "strategy"]


# ------------------------------------------------------------------ 1 金标准
def test_golden_bitwise(p, score, gold):
    """改造后全部配置的逐日收益与标量字段与改造前一致(容差 1e-10,实测为逐位相等)。"""
    assert gold["meta"]["strategy_sha256"] == "56d74351ea1bf1e34cbc9f7f39ab2abda363634083b17421657baed01154e4b0"
    worst = 0.0
    for key, g in gold["results"].items():
        name, wname = key.split("|")
        res = _run_gold_config(gold["configs"][name], score, p, _mask(p, wname))
        r = res["rets"].to_numpy(np.float64)
        gr = np.asarray(g["rets"], np.float64)
        assert len(r) == len(gr), key
        assert str(res["rets"].index[0].date()) == g["rets_index_first"], key
        assert str(res["rets"].index[-1].date()) == g["rets_index_last"], key
        d = float(np.max(np.abs(r - gr)))
        worst = max(worst, d)
        assert d <= TOL, f"{key}: 逐日收益偏离 {d:.3e}"
        for f, v in g.items():
            if f in ("rets", "rets_index_first", "rets_index_last", "rets_sha256"):
                continue
            if f == "phase_ann":
                assert np.allclose(res[f], v, rtol=0, atol=TOL), key
                continue
            if f in res and np.isfinite(v):
                assert abs(float(res[f]) - v) <= TOL, f"{key}.{f}: {res[f]} vs {v}"
    print(f"golden worst |diff| = {worst:.3e}")


# ------------------------------------------------------------------ 2 exposure
def test_exposure_le_one(p, score, cm1, m2024):
    """所有新路径的每日仓位 <= 1(引擎循环内还有硬断言)。"""
    W = ST.build_weights(score, p, "V1000", "top", 20)
    runs = [
        ST.run_weights(W, p, "open10", 1e7, rebalance=5, record=True),
        ST.run_weights(W, p, "open10", 1e8, rebalance=5, cost_model=cm1, record=True),
        ST.run_weights(W, p, "vwap1", 1e7, rebalance=5, cost_model=cm1, record=True, date_mask=m2024),
        ST.run_weights(lambda t, h, ctx: W[t] * 0.7, p, "open10", 1e7, rebalance=5, normalize=False,
                       record=True, date_mask=m2024),
        ST.run_weights(W, p, "tail1450", 1e9, rebalance=1, cost_model=cm1, record=True, date_mask=m2024,
                       participation_on_nav=True),
        ST.run_strategy(score, p, "V1000", "open10", scheme="exclude", buffer=(0.20, 0.35), rebalance=5,
                        record=True),
    ]
    for res in runs:
        by = res["record"]["daily_by_phase"]
        assert by["exposure"].max() <= 1.0 + 1e-9
        assert res["exposure"] <= 1.0 + 1e-9
        H = res["record"]["holdings"]
        # holdings 是 float32;在 float32 里对 ~3500 只求和会累积 ~1e-6 舍入误差,故转 float64 再加
        assert float(H.to_numpy(np.float64).sum(axis=1).max()) <= 1.0 + 1e-6
        assert (H.to_numpy() >= 0).all()
    # normalize=False 时 0.7 倍目标应留约 30% 现金
    assert runs[3]["exposure"] < 0.75


# ------------------------------------------------------------------ 3 对齐
@pytest.mark.parametrize("mode", ["open10", "tail1450", "vwap1"])
def test_alignment_peak_at_zero(p, score, mode):
    """策略与池基准的日收益同索引、互相关峰值在 k=0(否则逐年超额带 1~10pp 伪信号)。"""
    for plan in (dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5), dict(scheme="top", n=20, rebalance=5)):
        s = ST.run_strategy(score, p, "V1000", mode, **plan)
        b = ST.benchmark(p, "V1000", mode)
        assert s["rets"].index.equals(b["rets"].index)
        cs = {k: s["rets"].corr(b["rets"].shift(k)) for k in (-2, -1, 0, 1, 2)}
        assert max(cs, key=cs.get) == 0, (mode, plan, cs)


# ------------------------------------------------------------------ 4 全池等权
@pytest.mark.parametrize("mode", ["open10", "tail1450", "vwap1"])
def test_fullpool_excess_equals_minus_cost(p, mode):
    """全池等权(日调仓)vs 池等权基准:超额 + 成本 ≈ 0。

    已知残差(eng_engine_checks.csv):三口径两版本逐年 0.10~0.91pp、全期 0.27~0.63pp,
    全部为正 —— 一字板可成交性差异(策略买不进/卖不掉一字板,持有中的涨停一字板因漂移超配),
    已定位非 bug。容差取 [-0.5, +1.5]pp。
    ⚠ 必须用常数打分:COMP_v5 在 2020-02-06 前缺失,策略空仓而基准满仓经历 2020-02-03 暴跌,
    会给 2020 年带来 +10pp 的伪残差(那是打分预热,不是引擎问题)。"""
    ones = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype=np.float32)
    for y in (None, 2021, 2024):
        m = None if y is None else np.asarray(p.dates.year == y)
        s = ST.run_strategy(ones, p, "V1000", mode, scheme="exclude", exclude_frac=0.0, rebalance=1, date_mask=m)
        b = ST.benchmark(p, "V1000", mode, date_mask=m)
        resid_pp = (s["ann_return"] - b["ann_return"] + s["cost_bps"] * 1e-4 * ST.TRADING_DAYS) * 100
        assert -0.5 < resid_pp < 1.5, (mode, y, resid_pp)


# ------------------------------------------------------------------ 4b 同频基准(2026-10-10 第二轮修订,前缀 fixa)
@pytest.mark.parametrize("mode", ["open10", "vwap1"])
def test_fullpool_samefreq_excess_equals_minus_cost(p, mode):
    """纪律 4:全池等权(rebalance=k)vs **同频再平衡**的池等权基准(benchmark(rebalance=k)),超额 + 成本 ≈ 0。

    为什么要同频:k 日调仓的组合其间让权重漂移,k=1 基准每天重置等权;个股相对池均值的收益有正的截面序列相关时
    (vwap1 平均价口径滞后 1 期截面相关 0.175,open10 0.037,research/ledger/fixa_r2_mech.csv;池等权序列本身的
    自相关只是旁证)两者之差是与选股无关的再平衡频率差。只对 k=1 基准做这条判据,vwap1 的 5 日调仓会被
    "超额 +2pp"误判为引擎问题,或者反过来,把这段差当成策略超额。
    口径:常数打分、旧成本、1000 万、V1000;**一条连续路径**(无 mask)按年切片,不在窗口起点冷启动 —— 冷启动时
    相位 k 前 k 天空仓,2021-2025 掩码下多出约 +0.5pp(2021 年 1 月的空仓伪超额,与基准口径无关;
    research/ledger/fixa_fullpool_checks.csv 的 path=audit_mask2021)。残差 = mean(策略 − 基准 + 成本) × 242(算术)。
    容差 [-0.5, +1.5]pp 沿用原判据(可成交性残差:策略跳一字板、基准不跳,已定位非 bug)。
    实测(fixa_fullpool_checks.csv,path=continuous):
      k=1:open10 2021-2025 +0.52pp、逐年 +0.26~+0.74;vwap1 +0.31、逐年 +0.13~+0.50
      k=5 对同频基准:open10 +0.23、逐年 +0.11~+0.34;vwap1 +0.05、逐年 −0.06~+0.12
    反例(判据能分辨):k=5 对 k=1 基准,vwap1 2021-2025 +1.95pp、2024 年 +2.56pp,open10 +0.67pp;
    所以另断言 vwap1 下这一对照的残差 > 1.5pp(偏差真实存在,不是靠容差放宽掩盖的),
    以及同频残差不超过日调仓残差 + 0.3pp(5 日调仓交易更少,可成交性残差只会更小)。"""
    ones = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype=np.float32)
    windows = {"2021-2025": lambda ix: ix.year >= 2021, "2021": lambda ix: ix.year == 2021,
               "2024": lambda ix: ix.year == 2024}
    resid, s5 = {}, None
    for k in (1, 5):
        s = ST.run_strategy(ones, p, "V1000", mode, scheme="exclude", exclude_frac=0.0, rebalance=k, record=True)
        b = ST.benchmark(p, "V1000", mode, rebalance=k)
        assert s["rets"].index.equals(b["rets"].index)
        cs = {kk: s["rets"].corr(b["rets"].shift(kk)) for kk in (-2, -1, 0, 1, 2)}
        assert max(cs, key=cs.get) == 0, (mode, k, cs)
        assert s["exposure"] <= 1.0 + 1e-9
        d = s["rets"] - b["rets"] + s["record"]["daily"]["cost_total"]
        for w, f in windows.items():
            resid[(k, w)] = float(d[f(d.index)].mean() * ST.TRADING_DAYS * 100)
            assert -0.5 < resid[(k, w)] < 1.5, (mode, k, w, resid[(k, w)])
        if k == 5:
            s5 = s
    for w in windows:
        assert resid[(5, w)] <= resid[(1, w)] + 0.3, (mode, w, resid)
    if mode == "vwap1":
        b1 = ST.benchmark(p, "V1000", mode)["rets"]
        d = s5["rets"] - b1 + s5["record"]["daily"]["cost_total"]
        biased = float(d[d.index.year >= 2021].mean() * ST.TRADING_DAYS * 100)
        assert biased > 1.5, biased


def test_samefreq_benchmark_implementation(p):
    """同频基准的实现核对:
    (a) rebalance=1 显式传入与默认逐位相同(默认路径不变,金标准见 test_golden_bitwise);
    (b) 同频实现的 k=1 特例 = 旧路径(同一组股票、同一均值;差只来自 float32 原料的求和顺序,实测 8.4e-9);
    (c) 买入持有恒等式:任一相位在两个调仓日之间的累计收益 = 调仓日池内(当日有持有收益的)股票各自买入持有的
        等权平均,缺价日收益记 0(与引擎对持仓的处理一致);
    (d) 相位平均与锚点无关:date_mask 截断后的序列 = 全样本序列的切片(k 个相位覆盖全部调仓日历)。"""
    for mode in ("open10", "vwap1"):
        a = ST.benchmark(p, "V1000", mode)
        b = ST.benchmark(p, "V1000", mode, rebalance=1)
        assert np.array_equal(a["rets"].to_numpy(), b["rets"].to_numpy()) and a["ann_return"] == b["ann_return"]
        R1 = ST._pool_ew_drift_returns(p, "V1000", mode, 1)
        assert float(np.max(np.abs(R1[0] - a["rets"].to_numpy()))) < 1e-7
    mode, k, ph = "vwap1", 5, 2
    R = ST._pool_ew_drift_returns(p, "V1000", mode, k)
    px = ST.exec_prices(p, mode)[0].to_numpy(np.float64)
    U = ST.universe(p, "V1000").to_numpy(bool)
    lag = ST.EXEC_MODES[mode]["lag"]
    n_sig = len(p.dates) - 1 - lag
    with np.errstate(invalid="ignore", divide="ignore"):
        G = px[1 + lag:1 + lag + n_sig] / px[lag:lag + n_sig] - 1.0
    starts = [t for t in np.flatnonzero(np.asarray(p.dates[:n_sig].year == 2024)) if (t - ph) % k == 0][:6]
    for t in starts:
        mem = U[t] & np.isfinite(G[t])
        gz = np.where(np.isfinite(G[t:t + k]), G[t:t + k], 0.0)[:, mem]
        bh = float(np.prod(1.0 + gz, axis=0).mean())
        assert abs(float(np.prod(1.0 + R[ph, t:t + k])) - bh) < 1e-12, (t, bh)
    full = ST.benchmark(p, "V1000", mode, rebalance=k)["rets"]
    m = np.asarray(p.dates.year == 2024)
    cut = ST.benchmark(p, "V1000", mode, rebalance=k, date_mask=m)["rets"]
    assert np.allclose(cut.to_numpy(), full.reindex(cut.index).to_numpy(), rtol=0, atol=1e-14)


# ------------------------------------------------------------------ 5 零成本模型
def test_cost_model_zero_is_legacy(p, score, gold, cm0):
    """CostModel(impact_y=0, spread_mult=0) 与 cost_model=None(=改造前)逐位一致。"""
    for key in _strategy_keys(gold):
        name, wname = key.split("|")
        res = _run_gold_config(gold["configs"][name], score, p, _mask(p, wname), cost_model=cm0)
        assert np.array_equal(res["rets"].to_numpy(np.float64), np.asarray(gold["results"][key]["rets"])), key


# ------------------------------------------------------------------ 6 单调
def test_impact_monotone_in_capital(p, score, cm1, m2024):
    """(a) 函数层:固定 Δw,冲击成本随 capital 严格增加;
    (b) 引擎层:固定目标 W、参与率不设限(否则大资金被截量、成交变少,冲击反而可能下降),
        日均冲击成本随 capital 单调不减。"""
    rng = np.random.default_rng(0)
    for te in (300, 800, 1300):
        delta = np.zeros(len(p.codes))
        idx = rng.choice(np.flatnonzero(np.isfinite(cm1.adv(te))), 50, replace=False)
        delta[idx] = rng.normal(0, 0.01, 50)
        imp = [cm1.trade_cost(te, delta, cap)["impact"] for cap in (1e6, 1e7, 1e8, 1e9)]
        assert all(b > a for a, b in zip(imp, imp[1:])), imp
    W = ST.build_weights(score, p, "V1000", "top", 20)
    imp = []
    for cap in (1e6, 1e7, 1e8, 1e9):
        r = ST.run_weights(W, p, "open10", cap, rebalance=5, date_mask=m2024, cost_model=cm1,
                           max_participation=1e6, record=True, phases=[0, 2])
        imp.append(float(r["record"]["daily"]["cost_impact"].mean()))
    assert all(b >= a for a, b in zip(imp, imp[1:])), imp


# ------------------------------------------------------------------ 7 run_weights == run_strategy
def test_run_weights_matches_run_strategy(p, score, m2024):
    U = ST.universe(p, "V1000").to_numpy(bool)
    S = score.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32)
    for mask in (None, m2024):
        a = ST.run_strategy(score, p, "V1000", "open10", scheme="top", n=20, rebalance=5, date_mask=mask)
        W = ST.topn_weights(S, U if mask is None else U & mask[:, None], 20)
        b = ST.run_weights(W, p, "open10", 1e7, rebalance=5, date_mask=mask)
        assert np.array_equal(a["rets"].to_numpy(), b["rets"].to_numpy())
        assert a["ann_return"] == b["ann_return"] and a["turnover"] == b["turnover"]
        # 未按 mask 截断的 W 由 run_weights 内部截断,结果相同
        c = ST.run_weights(ST.topn_weights(S, U, 20), p, "open10", 1e7, rebalance=5, date_mask=mask)
        assert np.array_equal(a["rets"].to_numpy(), c["rets"].to_numpy())
    # 单相位同样一致
    a = ST.run_strategy(score, p, "V1000", "open10", scheme="top", n=20, rebalance=5, avg_phases=False)
    b = ST.run_weights(ST.topn_weights(S, U, 20), p, "open10", 1e7, rebalance=5, avg_phases=False)
    assert np.array_equal(a["rets"].to_numpy(), b["rets"].to_numpy())


# ------------------------------------------------------------------ 8 callable == ndarray
def test_callable_fixed_matches_ndarray(p, score, m2024, cm1):
    W = ST.build_weights(score, p, "V1000", "exclude", buffer=(0.20, 0.35))
    seen = []

    def fn(t, h, ctx):
        seen.append((ctx["te"] - t, ctx["phase"]))
        assert ctx["budget"] <= 1.0 + 1e-12 and ctx["can"].dtype == bool and ctx["capital"] == 1e7
        return W[t]

    for mask in (None, m2024):
        for cm in (None, cm1):
            a = ST.run_weights(W, p, "open10", 1e7, rebalance=5, date_mask=mask, cost_model=cm)
            b = ST.run_weights(fn, p, "open10", 1e7, rebalance=5, date_mask=mask, cost_model=cm)
            assert np.array_equal(a["rets"].to_numpy(), b["rets"].to_numpy()), (mask is None, cm)
    assert {d for d, _ in seen} == {1} and {ph for _, ph in seen} == set(range(5))
    # 返回 None = 当天不交易:永远 None -> 永远空仓
    z = ST.run_weights(lambda t, h, ctx: None, p, "open10", 1e7, rebalance=5, date_mask=m2024)
    assert z["exposure"] == 0.0 and z["turnover"] == 0.0


# ------------------------------------------------------------------ 9 backtest vs strategy
def test_backtest_matches_strategy(p, score):
    """backtest.run_standard 与 strategy 单相位同配置逐位一致(README 判据 ③)。"""
    a = BT.run_standard(score, p)
    b = ST.run_strategy(score, p, "V1000", "open10", scheme="exclude", buffer=(0.20, 0.35), rebalance=5,
                        avg_phases=False)
    assert np.array_equal(a["rets"].to_numpy(), b["rets"].to_numpy())
    a = BT.run_backtest(score, p, top_n=100)
    b = ST.run_strategy(score, p, "V1000", "open10", scheme="top", n=100, rebalance=1)
    assert abs(a["ann_return"] - b["ann_return"]) < 1e-12


# ------------------------------------------------------------------ 其他:成本模型一致性
def test_cost_coefficients_match_engine(p, cm1):
    """优化器用的系数与引擎扣的成本是同一个模型:Σ lin*|Δw| + Σ c15*|Δw|^1.5 == trade_cost。"""
    rng = np.random.default_rng(1)
    for te in (100, 900, 1400):
        for cap, nav in ((1e7, 1.0), (1e9, 1.7)):
            delta = np.zeros(len(p.codes))
            idx = rng.choice(len(p.codes), 80, replace=False)
            delta[idx] = rng.normal(0, 0.01, 80)
            co = cm1.cost_coefficients(te, cap, nav)
            buy, sell = np.clip(delta, 0, None), np.clip(-delta, 0, None)
            via_coef = float((co["lin_buy"] * buy).sum() + (co["lin_sell"] * sell).sum()
                             + (co["c15"] * np.abs(delta) ** 1.5).sum())
            tc = cm1.trade_cost(te, delta, cap, nav)
            assert abs(via_coef - tc["total"]) <= 1e-12 * max(1.0, abs(tc["total"]))
            # 引擎的旧成本部分(佣金+印花税)与 config 常数同一表达式
            stamp = cm1.stamp(te)
            legacy = (buy.sum() + sell.sum()) * 1e-4 + sell.sum() * stamp
            assert abs(tc["commission"] + tc["stamp"] - legacy) < 1e-15
    # 引擎记录的成本分解加总 = 每日成本
    W = np.zeros((len(p.dates), len(p.codes)), np.float32)
    W[:, :50] = 1 / 50
    r = ST.run_weights(W, p, "open10", 1e8, rebalance=5, cost_model=cm1, record=True, phases=[0],
                       date_mask=np.asarray(p.dates.year == 2023))
    d = r["record"]["daily"]
    assert np.allclose(d[["cost_commission", "cost_stamp", "cost_spread", "cost_impact"]].sum(axis=1),
                       d["cost_total"], rtol=1e-12, atol=1e-15)
    assert np.allclose(d["ret"], d["gross_ret"] - d["cost_total"], rtol=0, atol=1e-15)


# ------------------------------------------------------------------ 其他:无前视
class _TruncPanel(Panel):
    """把第 k 行之后的原料全部置 NaN 的面板:截断后估计量在 te<=k+1 行必须不变。"""

    def __init__(self, k: int):
        super().__init__()
        self._k = k

    def raw(self, name):
        key = "__trunc__" + name                # 不与父类缓存键冲突(父类按 name 缓存未截断原料)
        if key not in self._cache:
            df = super().raw(name).copy()
            df.iloc[self._k + 1:] = np.nan
            self._cache[key] = df
        return self._cache[key]


@pytest.mark.parametrize("method", ["tick", "ar_window"])
def test_cost_estimates_no_lookahead(p, method, monkeypatch):
    """成交日 te 的 σ/ADV/价差只用到 te-1:把 te 及以后的数据抹掉,te 行的估计不变。"""
    from afactor import newdata as ND
    k = 1000                                    # 保留到第 k 行(含),检查 te = k+1
    full = IM.CostModel(p, impact_y=1.0, spread_method=method)     # 先用完整数据建(含未复权价)
    orig = ND.raw_close

    def trunc_raw_close(d, c):
        df = orig(d, c).copy()
        df.iloc[k + 1:] = np.nan
        return df

    monkeypatch.setattr(ND, "raw_close", trunc_raw_close)
    tp = _TruncPanel(k)
    tr = IM.CostModel(tp, impact_y=1.0, spread_method=method)
    te = k + 1
    for f in ("half_spread", "sigma", "adv"):
        a, b = getattr(full, f)(te), getattr(tr, f)(te)
        ok = np.isfinite(a)
        assert np.array_equal(a[ok], b[ok]), f


# ------------------------------------------------------------------ 其他:执行口径
def test_exec_modes_unchanged_and_vwap1(p):
    expect = {
        "open10": dict(win=(1, 11), lag=1, tail=False), "open5": dict(win=(1, 6), lag=1, tail=False),
        "tail1450": dict(win=(230, 238), lag=0, tail=True), "tail1445": dict(win=(225, 238), lag=0, tail=True),
        "close": dict(win=(240, 241), lag=0, tail=True), "vwap1": dict(win=(0, 241), lag=1, tail=False)}
    for k, v in expect.items():
        assert {kk: ST.EXEC_MODES[k][kk] for kk in v} == v, k
    px, amt = ST.exec_prices(p, "vwap1")
    assert px is p.raw("vwap") and amt is p.raw("amount")
    for m, (a, b) in {"open10": ("twap_open10", "amt_open10"), "open5": ("twap_open10", "amt_open10"),
                      "tail1450": ("twap_tail8", "amt_tail8"), "close": ("close_px", "amt_close_auction")}.items():
        px, amt = ST.exec_prices(p, m)
        assert px is p.raw(a) and amt is p.raw(b), m
    # vwap 落在 [low, high] 内。⚠ 这是**按构造恒成立**的健全性检查(vwap 是各 bar 中价的成交额加权
    # 调和平均,high/low 是同一套 bar 的 max/min(open,close)),只防原料文件错位,不能作为 VWAP 口径的证据;
    # 外部佐证见 eng_vwap_checks.csv 的 ext_* 行(新浪成交量反推的 VWAP)。
    v, hi, lo = p.raw("vwap"), p.raw("high_px"), p.raw("low_px")
    ok = v.notna() & hi.notna() & lo.notna()
    inside = ((v >= lo * (1 - 1e-4)) & (v <= hi * (1 + 1e-4))) | ~ok
    assert bool(inside.to_numpy().all())


def test_record_consistency(p, score):
    """record 的 daily['ret'] 相位平均 == 返回的 rets;holdings 行和 == exposure。"""
    r = ST.run_strategy(score, p, "V1000", "open10", scheme="top", n=20, rebalance=5, record=True)
    d = r["record"]["daily"]
    assert np.allclose(d["ret"].to_numpy(), r["rets"].to_numpy(), rtol=0, atol=1e-15)
    H = r["record"]["holdings"]
    assert np.allclose(H.to_numpy(np.float64).sum(axis=1), d["exposure"].to_numpy(), atol=1e-6)
    assert set(r["record"]["daily_by_phase"]["phase"].unique()) == set(range(5))


# ------------------------------------------------------------------ 审查修复(2026-10-10)
class _Stop(Exception):
    pass


class _PerturbPanel(Panel):
    """rows >= k 的指定原料乘 U(0.5, 2)(保持有限且为正,不改变可成交性判定)。一字板判定借用原面板。"""

    def __init__(self, base, k, fields, seed=0):
        super().__init__()
        self._k, self._f, self._rng = k, set(fields), np.random.default_rng(seed)
        for f in ("one_word_up", "one_word_down"):
            self._derived[f] = base[f]

    def raw(self, name):
        if name not in self._f:
            return super().raw(name)
        key = "__pert__" + name
        if key not in self._cache:
            df = super().raw(name)
            a = df.to_numpy(np.float32, copy=True)
            a[self._k:] *= self._rng.uniform(0.5, 2.0, size=a[self._k:].shape).astype(np.float32)
            self._cache[key] = pd.DataFrame(a, index=df.index, columns=df.columns)
        return self._cache[key]


def _capture(p, W, mode, t_star, mask, capital=1e8, on_nav=True):
    got = {}

    def fn(t, h, ctx):
        if t == t_star:
            got.update(ctx=ctx, h=h)
            raise _Stop
        return W[t]

    with pytest.raises(_Stop):
        ST.simulate(fn, p, mode, capital, rebalance=1, date_mask=mask, participation_on_nav=on_nav)
    return got


@pytest.mark.parametrize("mode", ["open10", "vwap1"])
def test_ctx_is_decision_time_only(p, score, m2024, mode):
    """审查 P1/P2:ctx 的 cap_w / h_dec / nav_dec 只用决策时点信息。
    把 te 及以后的成交价、成交额乘随机扰动,这些字段逐位不变;按 px[te] 计价的 h / nav 则会变(对照)。
    另:cap_w == 0.1 × 截至 te-1 的 20 日窗口成交额中位数 / capital(独立重算)。"""
    lag = ST.EXEC_MODES[mode]["lag"]
    px_f, amt_f = ("vwap", "amount") if mode == "vwap1" else ("twap_open10", "amt_open10")
    te_star = int(np.flatnonzero(m2024)[0]) + 40
    W = ST.build_weights(score, p, "V1000", "top", 50)
    a = _capture(p, W, mode, te_star - lag, m2024)
    tp = _PerturbPanel(p, te_star, (px_f, amt_f), seed=3)
    b = _capture(tp, W, mode, te_star - lag, m2024)
    ca, cb = a["ctx"], b["ctx"]
    assert ca["te"] == te_star
    for f in ("cap_w", "h_dec", "nav_dec", "budget_dec", "can", "stuck"):
        assert np.array_equal(np.asarray(ca[f]), np.asarray(cb[f])), f
    held = a["h"] > 1e-12
    assert held.sum() > 20 and np.mean(a["h"][held] != b["h"][held]) > 0.9      # 对照:h 含 te 价
    assert ca["nav"] != cb["nav"]
    # cap_w 的定义(participation_on_nav=False 时分母为 capital)
    c = _capture(p, W, mode, te_star - lag, m2024, on_nav=False)["ctx"]
    amt = p.raw(amt_f).astype(np.float64)
    med = amt.rolling(20, min_periods=10).median().to_numpy()[te_star - 1]
    assert np.array_equal(c["cap_w"], np.where(np.isfinite(med), 0.10 * med / 1e8, 0.0))
    realized = amt.to_numpy()[te_star]
    assert not np.array_equal(c["cap_w"], np.where(np.isfinite(realized), 0.10 * realized / 1e8, 0.0))


@pytest.mark.parametrize("mode", ["vwap1", "tail1450"])
def test_hdec_is_exact_rollback(p, score, mode):
    """ctx['h_dec'] == h 按 px[te-1]/px[te] 回滚、含现金腿归一的精确值(= 上一笔成交后的持仓)。"""
    px = ST.exec_prices(p, mode)[0].to_numpy(np.float64)
    W = ST.build_weights(score, p, "V1000", "top", 50)
    m = np.asarray((p.dates >= "2024-01-01") & (p.dates <= "2024-04-30"))
    worst = [0.0, 0.0]

    def fn(t, h, ctx):
        te = ctx["te"]
        ok = np.isfinite(px[te - 1]) & np.isfinite(px[te]) & (px[te - 1] > 0)
        x = h / np.where(ok, px[te] / px[te - 1], 1.0)
        worst[0] = max(worst[0], float(np.max(np.abs(x / (x.sum() + 1.0 - h.sum()) - ctx["h_dec"]))))
        worst[1] = max(worst[1], float(np.max(np.abs(h - ctx["h_dec"]))))
        return W[t]

    ST.simulate(fn, p, mode, 1e7, rebalance=1, date_mask=m)
    assert worst[0] < 1e-12 and worst[1] > 1e-4, worst


def test_target_validation_and_ctx_copies(p, score, m2024):
    """审查 P2:负权重报错、求解器噪声级负数截零、inf 报错;ctx 数组是副本(原地改不影响引擎)。"""
    W = ST.build_weights(score, p, "V1000", "top", 20).astype(np.float64)
    base = ST.simulate(lambda t, h, ctx: W[t], p, "open10", 1e7, rebalance=5, date_mask=m2024)
    with pytest.raises(ValueError):
        ST.simulate(lambda t, h, ctx: np.where(ctx["can"] & (W[t] == 0), -0.01, W[t]), p, "open10", 1e7,
                    rebalance=5, date_mask=m2024)
    with pytest.raises(ValueError):
        Wn = W.copy(); Wn[:, 0] = -0.5
        ST.run_weights(Wn, p, "open10", 1e7, rebalance=5, date_mask=m2024, phases=[0])
    with pytest.raises(ValueError):
        ST.simulate(lambda t, h, ctx: np.where(W[t] > 0, np.inf, 0.0), p, "open10", 1e7, rebalance=5, date_mask=m2024)
    tiny = ST.simulate(lambda t, h, ctx: np.where(W[t] == 0, -1e-8, W[t]), p, "open10", 1e7, rebalance=5,
                       date_mask=m2024)
    assert np.array_equal(tiny["rets"].to_numpy(), base["rets"].to_numpy())

    def mut(t, h, ctx):
        ctx["can"][:] = True; ctx["stuck"][:] = False
        return W[t]

    r = ST.simulate(mut, p, "open10", 1e7, rebalance=5, date_mask=m2024)
    assert np.array_equal(r["rets"].to_numpy(), base["rets"].to_numpy())


def test_nan_means_hold(p, score, m2024):
    """NaN = 保持实际持仓不交易:全 NaN 与返回 None 逐位一致;部分 NaN 与"回传 h 的数值"逐位一致
    (normalize=False 且不超预算时两者动作相同)。"""
    W = ST.build_weights(score, p, "V1000", "top", 30).astype(np.float64)
    first = {}

    def nan_all(t, h, ctx):
        if not first:
            first["t"] = t
            return W[t]
        return np.full(len(W[t]), np.nan)

    def none_after(t, h, ctx):
        if t == first["t"]:
            return W[t]
        return None

    a = ST.simulate(nan_all, p, "vwap1", 1e7, rebalance=5, date_mask=m2024)
    b = ST.simulate(none_after, p, "vwap1", 1e7, rebalance=5, date_mask=m2024)
    assert np.array_equal(a["rets"].to_numpy(), b["rets"].to_numpy())

    def rule(use_nan):
        def fn(t, h, ctx):
            w = 0.5 * W[t]
            keep = (ctx["h_dec"] > 1e-12) & (W[t] > 0)
            return np.where(keep, np.nan if use_nan else h, w)
        return fn

    c = ST.simulate(rule(True), p, "vwap1", 1e7, rebalance=5, date_mask=m2024, normalize=False)
    d = ST.simulate(rule(False), p, "vwap1", 1e7, rebalance=5, date_mask=m2024, normalize=False)
    assert np.array_equal(c["rets"].to_numpy(), d["rets"].to_numpy())
    assert c["exposure"] < 0.9          # 0.5 倍新目标 + 保持仓 -> 留有现金


def test_lr_fields_and_excess_stats():
    """自相关稳健字段:白噪声 vr≈1;AR(1) ρ=0.3 时 vr 明显 >1;vol_adj/sharpe_adj 与 vr 一致;IR = 超额/TE。"""
    rng = np.random.default_rng(0)
    e = rng.normal(0, 0.01, 6000)
    assert abs(ST.lr_variance_ratio(e) - 1.0) < 0.15
    x = np.zeros_like(e)
    for i in range(1, len(e)):
        x[i] = 0.3 * x[i - 1] + e[i]
    vr = ST.lr_variance_ratio(x)
    assert 1.5 < vr < (1.3 / 0.7) + 0.1                     # Bartlett(20) 略低于理论值 1.857
    s = pd.Series(x + 0.0005, index=pd.bdate_range("2010-01-01", periods=len(x)))
    f = ST._lr_fields(s, 0.2, 1.0)
    assert np.isclose(f["vol_adj"], 0.2 * np.sqrt(f["vr_lr"])) and np.isclose(f["sharpe_adj"], 1.0 / np.sqrt(f["vr_lr"]))
    b = pd.Series(e, index=s.index)
    es = ST.excess_stats(s, b)
    assert np.isclose(es["ir"], es["excess_arith"] / es["te"])
    assert es["te_adj"] > es["te"] and es["ir_adj"] < es["ir"]
