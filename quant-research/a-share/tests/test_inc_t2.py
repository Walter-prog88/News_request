"""INC_T2(指数调样事件)回归测试:解析、分类、叠加目标、统计工具与账本一致性。

运行:PYTHONPATH=src python3 -m pytest tests/test_inc_t2.py -q
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

LEDGER = ROOT / "research" / "ledger"
TMP = ROOT / "cache" / "_tmp"


# ------------------------------------------------------------------ 解析(纯标准库)
def test_parse_counts_and_tables():
    import inc_t2_parse_official as PO
    txt = "其中沪深300指数更换25只样本，中证500指数更换50只样本，中证1 000指数更换100只样本"
    c = PO.parse_counts(txt)
    assert c["000300"] == 25 and c["000905"] == 50 and c["000852"] == 100
    html = ("<p>沪深300指数样本调整名单：</p><table><tr><td>调出名单</td><td>调入名单</td></tr>"
            "<tr><td>证券代码</td><td>证券名称</td><td>证券代码</td><td>证券名称</td></tr>"
            "<tr><td>000627</td><td>天茂集团</td><td>000800</td><td>一汽解放</td></tr></table>"
            "<p>中证500指数备选名单：</p><table><tr><td>1</td><td>600297</td><td>广汇汽车</td></tr></table>")
    p = PO.TableText(); p.feed(html); p.close()
    secs = [PO.section_of(v) for k, v in p.events if k == "text"]
    assert ("000300", "change") in secs and ("000905", "reserve") in secs
    rows = [v for k, v in p.events if k == "row"]
    assert ["000627", "天茂集团", "000800", "一汽解放"] in rows


def test_code_conversion():
    from inc_t2_build_official import code6_to_ts, bs_to_ts
    assert code6_to_ts("600000") == "600000.SH"
    assert code6_to_ts("300750") == "300750.SZ"
    assert code6_to_ts("688111") == "688111.SH"
    assert bs_to_ts("sz.000001") == "000001.SZ"


# ------------------------------------------------------------------ 分类
def test_classify_chain():
    from inc_t2_build_events import classify_chain
    before = {"000300": {"A"}, "000905": {"B"}, "000852": {"C"}}
    after = {"000300": {"B"}, "000905": {"C"}, "000852": {"D"}}
    assert classify_chain(before, after, "B", "000300", "in") == "MIG_up"      # 500 -> 300
    assert classify_chain(before, after, "C", "000905", "in") == "MIG_up"      # 1000 -> 500
    assert classify_chain(before, after, "D", "000852", "in") == "ADD_pure"
    assert classify_chain(before, after, "A", "000300", "out") == "DEL_pure"   # 300 -> none
    before2 = {"000300": {"A"}, "000905": set(), "000852": set()}
    after2 = {"000300": set(), "000905": {"A"}, "000852": set()}
    assert classify_chain(before2, after2, "A", "000300", "out") == "MIG_down"
    assert classify_chain(before2, after2, "A", "000905", "in") == "MIG_down"


# ------------------------------------------------------------------ 统计
def test_cluster_mean_se_matches_formula():
    from inc_t2_common import cluster_mean_se, period_t
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    g = np.repeat(np.arange(6), 10)
    mu, se, nG = cluster_mean_se(x, g)
    e = x - x.mean()
    S = pd.Series(e).groupby(g).sum().to_numpy()
    assert nG == 6
    assert np.isclose(se, np.sqrt((S ** 2).sum() * 6 / 5) / 60)
    m, t, n, pos = period_t(x, g)
    pm = pd.Series(x).groupby(g).mean()
    assert n == 6 and np.isclose(m, pm.mean()) and pos == int((pm > 0).sum())


# ------------------------------------------------------------------ 叠加目标
def _toy_W(T=40, N=10):
    W = np.zeros((T, N))
    W[:, :8] = 1.0 / 8
    return W


def test_make_target_baseline_only_trades_on_schedule():
    from inc_t2_overlay import make_target, K
    W = _toy_W()
    wins = [dict(A=12, Em1=20, add=np.array([8, 9]), dele=np.array([0]))]
    fn = make_target(W, t_first=5, phase=2, wins=wins, variant="none", x=0.1)
    calls = [t for t in range(5, 40) if fn(t, None, {}) is not None]
    assert calls == [t for t in range(5, 40) if (t - 7) % K == 0]


def test_make_target_overlay_weights():
    from inc_t2_overlay import make_target
    W = _toy_W()
    wins = [dict(A=12, Em1=20, add=np.array([8, 9]), dele=np.array([0]))]
    fA = make_target(W, 5, 0, wins, "A", 0.10)
    for t in range(5, 12):
        fA(t, None, {})
    w = fA(12, None, {})                       # 入场日(非基线调仓日也交易)
    assert w is not None and np.isclose(w.sum(), 1.0)
    assert np.isclose(w[8], 0.05) and np.isclose(w[9], 0.05)
    assert np.isclose(w[:8].sum(), 0.90)
    assert fA(13, None, {}) is None            # 窗口内非调仓日、非事件日:不交易
    w15 = fA(15, None, {})                     # 窗口内基线调仓日:仍是叠加目标
    assert np.isclose(w15[8], 0.05)
    wx = fA(20, None, {})                      # E−1 信号:回到基线
    assert np.isclose(wx[8:].sum(), 0.0) and np.isclose(wx.sum(), 1.0)
    fB = make_target(W, 5, 0, wins, "B", 0.0)
    for t in range(5, 12):
        fB(t, None, {})
    wb = fB(12, None, {})
    assert wb[0] == 0.0 and np.isclose(wb.sum(), 1.0) and np.isclose(wb[1], 1.0 / 7)


# ------------------------------------------------------------------ 账本一致性(产出存在时)
@pytest.mark.skipif(not (LEDGER / "inc_t2_official_changes.csv").exists(), reason="未生成官方名单")
def test_official_counts_match_gate0():
    O = pd.read_csv(LEDGER / "inc_t2_official_changes.csv", dtype={"index": str})
    G = pd.read_csv(LEDGER / "inc_t2_gate0.csv", dtype={"index": str})
    O["index"] = O["index"].str.zfill(6); G["index"] = G["index"].str.zfill(6)
    c = O.groupby(["review", "index", "direction"]).size().unstack()
    for r in G[G["index"] != "000852"].itertuples():
        assert c.loc[(r.review, r.index), "in"] == r.n_in == c.loc[(r.review, r.index), "out"]
        if np.isfinite(r.n_text):
            assert r.n_in == r.n_text
    assert not G.loc[G["index"] == "000852", "gate0_pass"].any()


@pytest.mark.skipif(not (TMP / "inc_t2_events.parquet").exists(), reason="未生成事件表")
def test_events_sealed_and_calendar():
    from afactor.panel import Panel
    from inc_t2_common import calendar
    E = pd.read_parquet(TMP / "inc_t2_events.parquet")
    p = Panel()
    cal = calendar(p.dates)
    assert p.dates[E["E_i"].max()] <= pd.Timestamp("2025-12-31")
    assert (cal["A1_i"] == cal["A_i"] + 1).all() and (cal["E_i"] - cal["A_i"] >= 9).all()
    assert (cal["C"] < cal["A"]).all()


# ------------------------------------------------------------------ 审查修复回归(F2/F3/F4/F6)
@pytest.mark.skipif(not (LEDGER / "inc_t2_pa_st_diff.csv").exists(), reason="未生成 P^A(ST@A) 对账")
def test_pa_st_at_A_membership_identical_to_cache():
    """F4:P^A 的 ST 状态取 A 日 vs 取 E−10(7/10 期晚一天):36 个(期, 指数)的成分逐只一致。"""
    D = pd.read_csv(LEDGER / "inc_t2_pa_st_diff.csv", dtype={"index": str})
    assert len(D) == 36 and (D["only_cache"] == 0).all() and (D["only_stA"] == 0).all()
    assert (D["ann_idx_stA"] <= D["ann_idx_cache"]).all() and (D["ann_idx_cache"] - D["ann_idx_stA"]).max() == 1


@pytest.mark.skipif(not (TMP / "inc_t2_controls.parquet").exists(), reason="未生成对照")
def test_c_day_controls_exclude_only_pc_known_and_own_group():
    """F3:t0 = C 的对照候选只剔除 P^C 预测变动股与本组处理股;A 日才公开的官方变动股可以做对照。
    t0 = A、E 仍剔除本期全部来源的变动股(官方名单 A 盘后已公开)。"""
    E = pd.read_parquet(TMP / "inc_t2_events.parquet")
    Cn = pd.read_parquet(TMP / "inc_t2_controls.parquet")
    Cn = Cn[(Cn["tol"] == "tol10") & (Cn["ctrl_j"] >= 0)].merge(
        E[["ev_id", "source", "review", "index", "direction"]], on="ev_id")
    pc = E[E["source"] == "PC"].groupby("review")["j"].apply(lambda s: set(s[s >= 0])).to_dict()
    allc = E.groupby("review")["j"].apply(lambda s: set(s[s >= 0])).to_dict()
    own = E[E["j"] >= 0].groupby(["source", "review", "index", "direction"])["j"].apply(set).to_dict()
    cC = Cn[Cn["t0_kind"] == "C"]
    assert not any(j in pc.get(rv, set()) for rv, j in zip(cC["review"], cC["ctrl_j"]))
    assert not any(j in own[(s, rv, i, d)] for s, rv, i, d, j in
                   zip(cC["source"], cC["review"], cC["index"], cC["direction"], cC["ctrl_j"]))
    # 只被官方名单(不被任何预测)变动的股票,确实出现在 C 日对照里 -> 修复生效
    off_only = {rv: allc[rv] - pc.get(rv, set()) for rv in allc}
    assert any(j in off_only[rv] for rv, j in zip(cC["review"], cC["ctrl_j"]))
    for k in ("A", "E"):
        ck = Cn[Cn["t0_kind"] == k]
        assert not any(j in allc[rv] for rv, j in zip(ck["review"], ck["ctrl_j"]))


@pytest.mark.skipif(not (TMP / "inc_t2_event_legs.parquet").exists(), reason="未生成事件-腿表")
def test_cost_rt_column_is_the_deducted_cost():
    """F2:cost_rt_Y1_1e7 = 净值里实际扣除的往返成本。L1 完全按竞价卖出且全部成交时,毛超额 − 净超额 = cost_rt。"""
    L = pd.read_parquet(TMP / "inc_t2_event_legs.parquet")
    x = L[(L["leg"] == "L1") & L["entry_ok"] & (L["fill_1e7"] == 1.0) & (L["sell_auc_frac_1e7"] == 1.0)
          & np.isfinite(L["net_idx_Y1_10000000"]) & np.isfinite(L["ar_matched"])]
    assert len(x) > 1000
    assert np.allclose(x["ar_matched"] - x["net_idx_Y1_10000000"], x["cost_rt_Y1_1e7"], atol=1e-9)
    # 旧口径(冲击分母 ADV20)保留为 cost_rt_adv20ref;L1 的实际扣除高于它
    o5 = L[(L["leg"] == "L1") & (L["source"] == "O") & (L["index"] == "000905") & (L["etype"] == "ADD_800")
           & L["entry_ok"] & L["window"].isin(["dev", "test"])]
    assert o5["cost_rt_Y1_1e7"].mean() > o5["cost_rt_adv20ref_Y1_1e7"].mean() + 0.001
    assert 0.0040 < o5["cost_rt_Y1_1e7"].mean() < 0.0050


@pytest.mark.skipif(not (LEDGER / "inc_t2_fill_by_capital.csv").exists(), reason="未生成成交比例账本")
def test_fill_ratio_falls_with_capital():
    """F6:10 亿篮子的买入受 10% × 开盘窗口成交额限制,成交比例随资金单调下降。"""
    f = pd.read_csv(LEDGER / "inc_t2_fill_by_capital.csv", dtype={"index": str})
    x = f[(f["source"] == "O") & (f["index"] == "000905") & (f["etype"] == "ADD_800") & (f["leg"] == "L1")
          ].sort_values("capital")
    assert x["fill_mean"].is_monotonic_decreasing and x["fill_mean"].iloc[-1] < 0.3
