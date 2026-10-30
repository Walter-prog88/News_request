"""INC-T6 工具函数的单元测试(合成数据,不读面板、不算任何策略收益)。

运行:PYTHONPATH=src python3 -m pytest tests/test_inc_t6.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import inc_t6_common as C  # noqa: E402


def _toy(T=4, N=9, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=T)
    codes = pd.Index([f"{i:06d}.SZ" for i in range(N)])
    x = pd.DataFrame(rng.normal(size=(T, N)), index=dates, columns=codes)
    U = pd.DataFrame(True, index=dates, columns=codes)
    return x, U


def test_tercile_boundaries_and_nan():
    x, U = _toy(T=1, N=9)
    x.iloc[0] = np.arange(9, dtype=float)            # 秩 1..9 -> pct 1/9..1
    x.iloc[0, 4] = np.nan                            # NaN 不进任何场地
    U.iloc[0, 8] = False                             # 池外不进任何场地
    code = C.tercile_code(x, U)[0]
    assert code[4] == -1 and code[8] == -1
    fin = [0, 1, 2, 3, 5, 6, 7]                      # 7 只:pct = 1/7..7/7
    assert list(code[fin]) == [1, 1, 2, 2, 3, 3, 3]  # <=1/3 低;(1/3,2/3] 中;>2/3 高
    # 每只有效股票恰好落在一个档
    assert set(code[fin]) <= {1, 2, 3}


def test_tercile_is_row_local_no_lookahead():
    """分档只用当日截面:截掉未来行不改变过去行的编码。"""
    x, U = _toy(T=6, N=30, seed=1)
    full = C.tercile_code(x, U)
    part = C.tercile_code(x.iloc[:3], U.iloc[:3])
    assert np.array_equal(full[:3], part)


def test_cell_masks_partition_and_cross():
    x, U = _toy(T=3, N=60, seed=2)
    y, _ = _toy(T=3, N=60, seed=3)
    codes = {"D1": np.where(x.to_numpy() < 0, 0, 1).astype(np.int8),
             "D2": C.tercile_code(x, U), "D3": C.tercile_code(y, U), "D4": C.tercile_code(-x, U)}
    m = C.cell_masks_from_codes(codes, "main")
    for d in ("D2", "D3", "D4"):
        s = m[f"{d}_low"].astype(int) + m[f"{d}_mid"].astype(int) + m[f"{d}_high"].astype(int)
        assert (s == 1).all()                        # 三档互斥且覆盖全部有效股票
    assert np.array_equal(m["Q"], m["D3_high"] & m["D2_high"])
    assert np.array_equal(m["Qm"], m["D3_low"] & m["D2_low"])
    assert not (m["D1_loss"] & m["D1_profit"]).any()


def test_merge_groups_equal_weight():
    a = np.array([[0.5, 0.5, 0, 0, 0], [0, 0, 0, 0, 0]], np.float32)
    b = np.array([[0, 0, 0, 1.0, 0], [0, 0, 1 / 3, 1 / 3, 1 / 3]], np.float32)
    W = C.w_merge_groups([a, b])
    assert np.allclose(W.sum(1), 1.0)
    assert np.allclose(W[0], [1 / 3, 1 / 3, 0, 1 / 3, 0])
    assert np.allclose(W[1], [0, 0, 1 / 3, 1 / 3, 1 / 3])


def test_const_weights_cover_pool():
    U = np.zeros((2, 120), bool)
    U[0, :100] = True
    U[1, 10:70] = True
    W = C.w_const(None, U)
    assert np.allclose(W.sum(1), 1.0)
    assert np.allclose(W[0, :100], 0.01) and (W[0, 100:] == 0).all()
    assert np.allclose(W[1, 10:70], 1 / 60)


def test_exclude_and_top_buffer_within_cell():
    """排雷 / 选优只在场地掩码内排序:场地外股票永远是 0 权重。"""
    rng = np.random.default_rng(4)
    T, N = 5, 400
    S = rng.normal(size=(T, N)).astype(np.float32)
    M = np.zeros((T, N), bool)
    M[:, :300] = True
    W1 = C.w_exclude(S, M)
    W2 = C.w_top(S, M)
    for W in (W1, W2):
        assert (W[:, 300:] == 0).all()
        assert np.allclose(W.sum(1), 1.0)
    assert ((W1 > 0).sum(1) >= 300 * 0.64).all()      # 第一天剔最差 20%,之后缓冲 35%
    assert ((W2 > 0).sum(1) <= 300 * 0.15 + 1).all()  # 选优持仓不超过 15% 缓冲上限


def test_nw_se_matches_iid_and_window_guard(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    x = rng.normal(size=5000)
    se_nw = C.nw_se_mean(x, 10)
    se_iid = x.std() / np.sqrt(len(x))
    assert abs(se_nw / se_iid - 1) < 0.1
    # 阶段 A 名单落盘前,任何含检验窗的窗口切片都要报错
    monkeypatch.setattr(C, "STAGEA_FILE", tmp_path / "nope.csv")
    s = pd.Series(rng.normal(size=1500), index=pd.bdate_range("2020-06-01", periods=1500))
    C.wslice(s, "dev")                                # 开发窗允许
    for w in ("test", "full", "y2024", "stress"):
        with pytest.raises(RuntimeError):
            C.wslice(s, w)


def test_diff_stats_ci_and_quarter_t(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "STAGEA_FILE", tmp_path / "ok.csv")
    (tmp_path / "ok.csv").write_text("x\n")
    idx = pd.bdate_range("2024-01-02", "2025-12-31")
    d = pd.Series(0.0002 + np.random.default_rng(6).normal(0, 0.002, len(idx)), index=idx)
    st = C.diff_stats(d, "test")
    assert st["lo95"] < st["ann"] < st["hi95"]
    assert abs((st["hi95"] - st["ann"]) - 1.96 * st["se_ann"]) < 1e-12
    assert st["n_quarters"] == 8


def test_align_peak_zero_for_identical():
    s = pd.Series(np.random.default_rng(7).normal(size=300), index=pd.bdate_range("2021-01-04", periods=300))
    k, _ = C.align_peak(s, s)
    assert k == 0


def test_clone_sign_top_picks_smallest_size():
    """C2 = −log(MV):选优端(最好 10%)应当是市值最小的 10%。"""
    rng = np.random.default_rng(8)
    T, N = 3, 500
    mv = np.exp(rng.normal(22, 1, size=(T, N)))
    S = (-np.log(mv)).astype(np.float32)
    M = np.ones((T, N), bool)
    W = C.w_top(S, M)
    held = W[0] > 0
    assert held.sum() == 50
    assert mv[0, held].max() <= np.sort(mv[0])[49] + 1e-6


def test_exclude_buffer_keeps_state_outside_cell():
    """场地成员变化:离开场地的股票不得持有(权重只落在当日场地内)。"""
    rng = np.random.default_rng(9)
    T, N = 6, 300
    S = rng.normal(size=(T, N)).astype(np.float32)
    M = np.ones((T, N), bool)
    M[3:, :100] = False                               # 第 3 天起前 100 只离开场地
    W = C.w_exclude(S, M)
    assert (W[3:, :100] == 0).all()
    assert np.allclose(W.sum(1), 1.0)


def test_grid_cell_name_parsing():
    """gridtab 从基准文件名取场地名(曾误用 name 带出 .parquet 后缀)。"""
    from pathlib import Path
    f = Path("cache/_tmp/inc_t6_arms/inc_t6_bench_q5__D2q5_1.parquet")
    assert f.stem.split("__")[1] == "D2q5_1"


class _StubPanel:
    """最小面板桩:只提供 ST.exec_prices / _pool_ew_drift_returns 用到的 dates / codes / raw。"""

    def __init__(self, P: pd.DataFrame):
        self.dates, self.codes, self._P = P.index, P.columns, P

    def raw(self, name):
        return self._P if name == "twap_open10" else self._P * 0 + 1.0


def test_bench_engine_like_reproduces_official_without_notrade():
    """审查 P1 修复:notrade 为空(或 engine_like=False)时,与 ST._pool_ew_drift_returns 的相位平均逐位相同。"""
    from afactor import strategy as ST
    rng = np.random.default_rng(10)
    T, N = 60, 25
    dates = pd.bdate_range("2021-01-04", periods=T)
    P = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0)), index=dates,
                     columns=[f"{i:06d}.SZ" for i in range(N)])
    P.iloc[10:14, 3] = np.nan                                   # 缺价
    U = rng.random((T, N)) > 0.2
    stub = _StubPanel(P)
    ref = ST._pool_ew_drift_returns(stub, "V1000", "open10", 5, 0, pd.DataFrame(U, index=dates, columns=P.columns)).mean(axis=0)
    mine = C.bench_engine_like(P.to_numpy(), U, None, k=5, lag=1, engine_like=False)
    assert np.max(np.abs(ref - mine)) < 1e-12
    mine2 = C.bench_engine_like(P.to_numpy(), U, np.zeros((T, N), bool), k=5, lag=1, engine_like=False)
    assert np.max(np.abs(ref - mine2)) < 1e-12


def test_bench_engine_like_keeps_drifted_weight_of_untradable_stock():
    """调仓日 te 不可交易的持仓保留漂移权重(引擎 stuck 规则),而不是被重置回等权。"""
    T, N = 6, 3
    P = np.full((T, N), 100.0)
    P[:, 0] = 100.0 * 1.1 ** np.arange(T)                       # 股票 0 每天涨 10%
    U = np.ones((T, N), bool)
    nt = np.zeros((T, N), bool)
    nt[2, 0] = True                                             # 信号日 t=1 的成交日 te=2,股票 0 不可交易
    r_like = C.bench_engine_like(P, U, nt, k=1, lag=1, engine_like=True)
    r_reset = C.bench_engine_like(P, U, nt, k=1, lag=1, engine_like=False)
    w0_drift = (1 / 3 * 1.1) / (1 + 0.1 / 3)                    # t=0 满仓等权后漂移
    assert abs(r_like[0] - 0.1 / 3) < 1e-12 and abs(r_reset[0] - 0.1 / 3) < 1e-12
    assert abs(r_like[1] - w0_drift * 0.1) < 1e-12              # 保留漂移权重,吃到股票 0 的 10%
    assert abs(r_reset[1] - 0.1 / 3) < 1e-12                    # 重置等权
    assert r_like[1] > r_reset[1]
    assert abs(r_like[2] - r_reset[2]) < 1e-3                   # 之后各自继续(权重几乎相同)


def test_bench_engine_like_untradable_not_bought_and_budget_sums_to_one():
    """不可交易的非持仓不买入;可交易股票在 1 - 残仓权重 的预算内等权(权重和恒为 1)。"""
    T, N = 5, 4
    P = np.full((T, N), 100.0)
    P[2:, 1] = 100.0 * np.cumprod(np.full(T - 2, 0.9))          # 股票 1 在 te>=2 起每天跌 10%
    U = np.ones((T, N), bool)
    nt = np.zeros((T, N), bool)
    nt[1, 1] = True                                             # 首日(t=0,te=1)股票 1 不可交易 -> 不买
    r = C.bench_engine_like(P, U, nt, k=1, lag=1, engine_like=True)
    # t=0:只买 0/2/3 各 1/3;股票 1 在 te=1 -> 2 下跌 10% 但没被买入,所以组合收益 0
    assert abs(r[0]) < 1e-12
    # t=1(te=2):股票 1 可交易,四只等权;te=2 -> 3 股票 1 下跌 10%
    assert abs(r[1] - (-0.1 / 4)) < 1e-12
