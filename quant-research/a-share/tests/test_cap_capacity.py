"""容量曲线脚本的回归测试(前缀 cap,2026-10-10 审查修复后新增)。读真实研究期面板,约 1 分钟。

运行:  python3 -m pytest tests/test_cap_capacity.py -q

覆盖:
  1 exante_sizing 上下文:退出(含异常)后 ST.exec_prices 恢复为原函数;块内价格不变
  2 exante 成交额 = 截至 te-1 的 20 日窗口成交额中位数(独立重算),且不改变可成交集合(>0 的模式)
  3 ConstAUMCost:冲击 Q 与净值无关(nav 固定为 1)
  4 pov 主网格可复现:重跑一格,日收益与缓存逐位一致,窗口超额与 capacity_grid.csv 一致
  5 exante 在 1e6(参与率不约束)与 pov 逐位相同;在 1e10 排雷 open10 低于 pov(审查探针 2.44 vs 3.10pp)
  6 现金补仓列 = ret + cash × 基准 的几何年化差(独立重算)
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from afactor.config import TMP, LEDGER                    # noqa: E402

DAILY = TMP / "cap_daily_open10.parquet"
GRID = LEDGER / "capacity_grid.csv"
pytestmark = pytest.mark.skipif(not (DAILY.exists() and GRID.exists() and (TMP / "comp_v5_score_open10.parquet").exists()),
                                reason="需要真实数据面板与容量缓存")

import cap_capacity_curve as C                            # noqa: E402
from afactor.panel import Panel                           # noqa: E402
from afactor import strategy as ST, impact as IM          # noqa: E402


@pytest.fixture(scope="module")
def p():
    return Panel()


@pytest.fixture(scope="module")
def scores(p):
    return C.load_scores(p)


def test_context_restores_exec_prices(p):
    orig = ST.exec_prices
    with C.exante_sizing(p):
        assert ST.exec_prices is not orig
        px_a, amt_a = ST.exec_prices(p, "open10")
    assert ST.exec_prices is orig
    px_b, amt_b = ST.exec_prices(p, "open10")
    assert px_a.equals(px_b)                                  # 价格不变
    assert not amt_a.equals(amt_b)                            # 成交额换成了估计
    with pytest.raises(ZeroDivisionError):
        with C.exante_sizing(p):
            1 / 0
    assert ST.exec_prices is orig
    with C.exante_sizing(p, on=False):
        assert ST.exec_prices is orig


@pytest.mark.parametrize("mode", ["open10", "vwap1"])
def test_exante_amount_is_lagged_median(p, mode):
    used, st = C.exante_amount(p, mode)
    _, act = ST.exec_prices(p, mode)
    a, u = act.to_numpy(np.float64), used.to_numpy(np.float64)
    assert np.array_equal(a > 0, u > 0), "可成交集合 can 被改变"
    rng = np.random.default_rng(0)
    T, N = a.shape
    checked = 0
    for _ in range(4000):
        te, j = int(rng.integers(30, T)), int(rng.integers(0, N))
        if not a[te, j] > 0:
            continue
        hist = a[te - 20:te, j]                               # te-20 .. te-1,不含 te
        hist = hist[np.isfinite(hist)]
        if len(hist) < 10 or np.median(hist) <= 0:
            assert u[te, j] == a[te, j]                       # 退回实际值
            continue
        assert np.isclose(u[te, j], np.median(hist), rtol=1e-6), (te, j)
        checked += 1
    assert checked > 1000
    assert st["fallback_n"] / st["actual_pos_n"] < 0.02


def test_const_aum_cost_ignores_nav(p):
    cm = IM.CostModel(p, impact_y=1.0)
    w = C.ConstAUMCost(cm)
    te = int(np.flatnonzero(p.dates >= pd.Timestamp("2024-03-01"))[0])
    d = np.zeros(len(p.codes)); d[:50] = 0.01
    assert w.extra_cost(te, d, 1e9, 2.5) == cm.extra_cost(te, d, 1e9, 1.0)
    assert w.extra_cost(te, d, 1e9, 2.5) != cm.extra_cost(te, d, 1e9, 2.5)


def test_pov_cell_reproduces_cache_and_grid(p, scores):
    strat, aum = "COMP_v5_exclude", 1e10
    m = C.sim_mask(p, strat)
    W = C.strategy_weights(p, scores, strat)
    res, daily, _ = C.run_cell(p, W, "open10", aum, "Y1.0", m, {}, "pov")
    D = pd.read_parquet(DAILY, columns=["cell", "date", "ret"])
    old = D[D.cell == f"{strat}|open10|Y1.0|{aum:.0e}"].set_index("date")["ret"]
    assert float(np.abs(res["rets"].reindex(old.index) - old).max()) == 0.0
    B = pd.read_parquet(TMP / "cap_bench_open10.parquet")
    G = pd.read_csv(GRID)
    for role, (y0, y1) in (("full", (2021, 2025)), ("test", (2024, 2025))):
        rs = res["rets"][(res["rets"].index.year >= y0) & (res["rets"].index.year <= y1)]
        ex = C.geo_ann(rs) - C.geo_ann(B["pool"].reindex(rs.index))
        g = G.query("strategy==@strat and mode=='open10' and variant=='Y1.0' and sizing=='pov' "
                    "and aum==@aum and window_role==@role")
        assert len(g) == 1 and abs(float(g.excess_pool.iloc[0]) - ex) < 1e-12


def test_exante_vs_pov(p, scores):
    strat = "COMP_v5_exclude"
    m = C.sim_mask(p, strat)
    W = C.strategy_weights(p, scores, strat)
    r_lo = {sz: C.run_cell(p, W, "open10", 1e6, "Y1.0", m, {}, sz)[0]["rets"] for sz in C.SIZINGS}
    assert r_lo["pov"].equals(r_lo["exante"])                 # 1e6 不约束:逐位相同
    G = pd.read_csv(GRID)
    g = G.query("strategy==@strat and mode=='open10' and variant=='Y1.0' and aum==1e10 and window_role=='full'")
    po = float(g[g.sizing == "pov"].excess_pool.iloc[0]); ea = float(g[g.sizing == "exante"].excess_pool.iloc[0])
    assert ea < po - 0.003                                    # 审查探针:2.44 vs 3.10pp


def test_cash_adjusted_columns():
    G = pd.read_csv(GRID)
    D = pd.read_parquet(DAILY, columns=["cell", "date", "ret", "cash"])
    B = pd.read_parquet(TMP / "cap_bench_open10.parquet")
    cell = "COMP_v5_top20|open10|legacy|1e+08"
    d = D[D.cell == cell].set_index("date").sort_index()
    d = d[(d.index.year >= 2021) & (d.index.year <= 2025)]
    for col, bn in (("excess_pool_cashpool", "pool"), ("excess_pool_cashcsi", "csi1000p")):
        r = d["ret"] + d["cash"] * B[bn].reindex(d.index)
        ex = C.geo_ann(r) - C.geo_ann(B["pool"].reindex(d.index))
        g = G.query("strategy=='COMP_v5_top20' and mode=='open10' and variant=='legacy' and sizing=='pov' "
                    "and aum==1e8 and window_role=='full'")
        assert abs(float(g[col].iloc[0]) - ex) < 1e-12
