"""INC-T3(限售解禁)回归测试:口径函数、剔除窗口时序、安慰剂构造、统计工具、产出账本的一致性。

运行:PYTHONPATH=src python3 -m pytest tests/test_inc_t3.py -q
不读取任何 2025-12-31 之后的数据;依赖已落盘的 research/ledger/inc_t3_* 的测试在文件缺失时跳过。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import inc_t3_common as C          # noqa: E402
import inc_t3_overlay as OV        # noqa: E402


# ---------------------------------------------------------------- 类型 / 层
@pytest.mark.parametrize("s,tier,tg", [
    ("首发原股东限售股份", "A", "首发"), ("首发机构配售股份", "A", "首发"), ("公开发行原股东限售股份", "A", "首发"),
    ("定向增发机构配售股份", "A", "定增"), ("追加承诺限售股份上市流通", "A", "追加承诺"),
    ("股权分置限售股份", "A", "其他"), ("股权激励限售股份", "B", "股权激励"),
    ("股权激励一般股份", "B", "股权激励"), ("其他类型", "B", "其他"),
    ("定向增发机构配售股份,首发原股东限售股份", "A", "定增"),       # 逗号取第一个
])
def test_tier_and_type(s, tier, tg):
    assert C.tier_of(s) == tier
    assert C.type_group(s) == tg


def test_ts_code():
    assert C.ts_code("1", "szzb") == "000001.SZ"
    assert C.ts_code("688001", "kcb") == "688001.SH"
    assert C.ts_code("600000", "sh") == "600000.SH"
    assert C.ts_code("300750", "cyb") == "300750.SZ"


def test_age_and_size_buckets():
    ab = C.age_bucket(np.array([0, 249, 250, 750, 751, np.nan]))
    assert ab.tolist() == [0, 0, 1, 1, 2, -1]
    sb = C.size_bucket(pd.Series([0.04, 0.05, 0.0999, 0.10, 0.2, 0.5, 3.0]))
    assert sb.tolist()[1:] == ["[5,10)", "[5,10)", "[10,20)", "[20,50)", ">=50", ">=50"]
    assert pd.isna(sb.iloc[0])


# ---------------------------------------------------------------- 剔除窗口(时序)
def _ev(rows):
    return pd.DataFrame(rows, columns=["j", "d_pos", "tier", "ratio_plan"])


def test_exclusion_window_tiers_and_M():
    ev = _ev([(0, 100, "A", 0.30), (1, 100, "B", 0.30), (2, 100, "A", 0.05)])
    w = OV.exclusion_windows(ev, 0.10, 10, 0, T=300).set_index("j")
    assert 2 not in w.index                                     # 规模不够不剔
    assert (w.loc[0, "start"], w.loc[0, "end"]) == (90, 99)     # A 层:D-10 .. D-1(M=0 不吃 D)
    assert (w.loc[1, "start"], w.loc[1, "end"]) == (97, 99)     # B 层:已知日 D-3 起
    w10 = OV.exclusion_windows(ev, 0.10, 10, 10, T=300).set_index("j")
    assert w10.loc[0, "end"] == 109                             # M=10:到 D+9
    wb = OV.exclusion_windows(ev, 0.10, 20, 0, T=300, tier_rule="allB").set_index("j")
    assert wb.loc[0, "start"] == 97                             # 全 B 层
    wn = OV.exclusion_windows(ev, 0.10, 20, 0, T=300, tier_rule="none").set_index("j")
    assert wn.loc[1, "start"] == 80                             # C 层不设已知日


def test_exclusion_window_A_known_horizon():
    """A 层已知日 D-60 >= 任何 N_pre(<=20),所以 A 层窗口起点永远是 D - N_pre。"""
    ev = _ev([(0, 500, "A", 0.5)])
    for n in OV.NPRES:
        w = OV.exclusion_windows(ev, 0.1, n, 0, T=1000)
        assert w["start"].iloc[0] == 500 - n


def test_exclusion_window_clipping_and_panel():
    ev = _ev([(0, 2, "A", 0.5), (1, 299, "A", 0.5), (2, 400, "A", 0.5)])
    w = OV.exclusion_windows(ev, 0.1, 10, 10, T=300)
    assert set(w["j"]) == {0, 1}                                # D 在面板外且窗口整体在外 -> 丢弃
    E = OV.panel_from_windows(w, 300, 3)
    assert E[:12, 0].all() and not E[12:, 0].any()              # [0, 11]
    assert E[289:300, 1].all() and not E[:289, 1].any()


def test_signal_to_trade_alignment_M0():
    """M=0:信号日 D-1 仍被剔(成交日 D 开盘不持有,不吃解禁日 D 的开盘->次日开盘收益);信号日 D 重新纳入。"""
    ev = _ev([(0, 50, "A", 0.5)])
    E = OV.panel_from_windows(OV.exclusion_windows(ev, 0.1, 5, 0, T=100), 100, 1)
    assert E[49, 0] and not E[50, 0] and E[45, 0] and not E[44, 0]


# ---------------------------------------------------------------- 安慰剂
def test_placebo_windows_matching_and_disjoint():
    rng = np.random.default_rng(0)
    T, N = 120, 400
    U = np.ones((T, N), bool)
    U[:, 390:] = False
    S = np.tile((np.arange(N) % 15).astype(np.int16), (T, 1))
    S[~U] = -1
    ev = _ev([(j, int(d), "A", 0.5) for j, d in zip(rng.choice(390, 40, replace=False), rng.integers(30, 110, 40))])
    win = OV.exclusion_windows(ev, 0.1, 10, 0, T)
    E = OV.panel_from_windows(win, T, N)
    pw, st = OV.placebo_windows(win, E, U, S, [1, 2, 3])
    assert st["n_placebo"] == len(win) and st["n_fail"] == 0
    Ep = OV.panel_from_windows(pw, T, N)
    assert not (Ep & E).any()                                   # 伪剔除与真实剔除不重叠
    assert (Ep.sum(axis=1) == E.sum(axis=1)).all()              # 每日数量一致
    wj = win.sort_values(["start", "j"]).reset_index(drop=True)
    assert (S[wj["start"], wj["j"]] == S[pw["start"], pw["j"]]).all()   # 同层
    assert not U[:, pw["j"]][0].__invert__().any()              # 伪股票在池
    pw2, _ = OV.placebo_windows(win, E, U, S, [1, 2, 3])
    assert pw.equals(pw2)                                       # 种子确定性


# ---------------------------------------------------------------- 统计
def test_cluster_se_singletons_equals_iid_cr1():
    x = np.random.default_rng(1).normal(size=500)
    mu, se, nG = C.cluster_mean_se(x, np.arange(500))
    iid = x.std(ddof=0) / np.sqrt(500) * np.sqrt(500 / 499)
    assert nG == 500 and np.isclose(se[0], iid)


def test_cluster_se_grows_with_common_shock():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(50), 20)
    x = rng.normal(size=1000) + rng.normal(size=50)[g]
    _, se_c, _ = C.cluster_mean_se(x, g)
    _, se_i, _ = C.cluster_mean_se(x, np.arange(1000))
    assert se_c[0] > 2 * se_i[0]


def test_nw_se_L0_is_iid():
    x = np.random.default_rng(3).normal(size=300)
    assert np.isclose(C.nw_se(x, 0), x.std(ddof=0) / np.sqrt(300))
    y = np.convolve(np.random.default_rng(4).normal(size=400), np.ones(5), "valid")
    assert C.nw_se(y, 10) > C.nw_se(y, 0)


# ---------------------------------------------------------------- 账本一致性(产出存在时)
LED = C.LEDGER


@pytest.mark.skipif(not (LED / "inc_t3_unlock_events.csv").exists(), reason="事件表未生成")
def test_events_table_pit_and_definitions():
    ev = C.load_events()
    assert ev["FREE_DATE"].max() <= pd.Timestamp("2025-12-31")
    assert not ev.duplicated(["code", "D"]).any()
    assert (ev["D"] >= ev["FREE_DATE"]).all()
    kA = ev["tier"] == "A"
    assert ((ev.loc[kA, "d_pos"] - ev.loc[kA, "K_pos"]) == C.K_A).all()
    assert ((ev.loc[~kA, "d_pos"] - ev.loc[~kA, "K_pos"]) == C.K_B).all()
    assert ((ev["d_pos"] - ev["t_ref_pos"]) == 21).all()
    ok = ev["ratio_actual_em"] < 1
    assert np.allclose(ev.loc[ok, "ratio_actual"], ev.loc[ok, "ratio_actual_em"] / (1 - ev.loc[ok, "ratio_actual_em"]))
    assert {"B20_ADJCHRATE", "A20_ADJCHRATE"}.isdisjoint(ev.columns)        # 东财事后收益列不得进入


@pytest.mark.skipif(not C.EM_PARQUET.exists(), reason="快照缺失")
def test_load_em_drops_post_event_returns_and_cut():
    d = C.load_em()
    assert {"B20_ADJCHRATE", "A20_ADJCHRATE"}.isdisjoint(d.columns)
    assert d["FREE_DATE"].max() <= pd.Timestamp("2025-12-31")


def test_prereg_unchanged():
    assert C.md5(C.PREREG) == C.PREREG_MD5


@pytest.mark.skipif(not (LED / "inc_t3_acceptance.csv").exists(), reason="验收表未生成")
def test_gates_g1_to_g4_passed():
    a = pd.read_csv(LED / "inc_t3_acceptance.csv")
    g = a[a["gate"].isin(["G1", "G2", "G3", "G4"])]
    assert set(g["gate"]) == {"G1", "G2", "G3", "G4"}
    assert g["passed"].astype(bool).all()


@pytest.mark.skipif(not (LED / "inc_t3_checks.csv").exists(), reason="引擎检查未生成")
def test_g5_engine_checks_passed():
    c = pd.read_csv(LED / "inc_t3_checks.csv")
    g5 = c[c["check"].str.startswith("G5")]
    assert len(g5) >= 6 and g5["passed"].astype(bool).all()
    assert c[c["check"] == "engine_fullpool_ew_vs_samefreq"]["passed"].astype(bool).all()


@pytest.mark.skipif(not (LED / "inc_t3_balance.csv").exists(), reason="平衡表未生成")
def test_g6_balance_passed():
    b = pd.read_csv(LED / "inc_t3_balance.csv")
    g6 = b[b["gate"] == "G6"]
    assert len(g6) == 4 and g6["pass"].astype(bool).all()


# ---------------------------------------------------------------- 审查修复:规则日分类与敏感性情景(inc_t3_review_sens)
import inc_t3_review_sens as RS    # noqa: E402


def test_rule_gap_nearest_anniversary():
    ld = pd.Timestamp("2021-04-28")
    g, k, r = RS.nearest_rule_gap(ld, pd.Timestamp("2022-04-28"))
    assert (g, k, r) == (0, 12, pd.Timestamp("2022-04-28"))
    g, k, _ = RS.nearest_rule_gap(ld, pd.Timestamp("2022-05-25"))       # 审查例:001201 上市 2021-04-28,解禁 2022-05-25
    assert (g, k) == (27, 12) and RS.gap_bin(g) == "晚8-30日" and not RS.is_regular(g)
    g, k, _ = RS.nearest_rule_gap(ld, pd.Timestamp("2024-04-30"))
    assert (g, k) == (2, 36) and RS.is_regular(g)
    assert np.isnan(RS.nearest_rule_gap(pd.NaT, pd.Timestamp("2022-04-28"))[0])
    assert RS.gap_bin(np.nan) == "无锚点日期"
    assert [RS.gap_bin(x) for x in (-20, -3, 0, 7, 8, 30, 31)] == ["提前>7日", "提前1-7日", "规则日后0-7日", "规则日后0-7日",
                                                                "晚8-30日", "晚8-30日", "晚>30日"]


def _scen_inputs():
    ev = pd.DataFrame(dict(event_idx=[0, 1, 2, 3, 4], j=[0, 1, 2, 3, 4], d_pos=[100] * 5,
                           tier=["A"] * 5, ratio_plan=[0.3] * 5, tg=["首发", "首发", "定增", "追加承诺", "首发"]))
    ipo = pd.DataFrame(dict(event_idx=[0, 1, 4], regular=[True, False, False],
                            R_date=pd.to_datetime(["2021-01-01", "2021-01-04", "2021-01-04"])))
    dates = pd.bdate_range("2020-11-02", periods=200)
    return ev, ipo, dates


def test_scenarios_tiering_and_ruleD():
    ev, ipo, dates = _scen_inputs()
    T = 200
    w = {n: RS.scenario_windows(OV, ev, T, n, ipo, dates=dates).set_index("j") for n in ("main", "irrB", "strictIPO")}
    assert (w["main"]["start"] == 90).all() and (w["main"]["end"] == 99).all()          # 全部 A:D-10..D-1
    assert w["irrB"].loc[1, "start"] == 97 and w["irrB"].loc[0, "start"] == 90          # 不规则首发降 B(D-3)
    assert w["irrB"].loc[2, "start"] == 90 and w["irrB"].loc[3, "start"] == 90          # 定增 / 追加承诺保留 A
    assert w["strictIPO"].loc[0, "start"] == 90                                         # 规则首发保留 A
    assert all(w["strictIPO"].loc[j, "start"] == 97 for j in (1, 2, 3, 4))              # 其余全降 B
    wr = RS.scenario_windows(OV, ev, T, "ruleD", ipo, dates=dates)
    rpos = int(np.searchsorted(dates.to_numpy(), np.datetime64("2021-01-04")))
    extra = wr[(wr["j"] == 1) & (wr["start"] == rpos - 10)]
    assert len(extra) == 1 and extra["end"].iloc[0] == rpos - 1                          # 额外的 [R-10, R-1]
    assert len(wr[wr["j"] == 0]) == 1                                                    # 规则首发没有额外窗口


def test_scenarios_source_filters_and_calib_demotion():
    ev, ipo, dates = _scen_inputs()
    T = 200
    sets = {n: set(RS.scenario_windows(OV, ev, T, n, ipo, dates=dates)["j"]) for n in
            ("noZhuijia", "onlyIPO", "onlyIPOreg", "onlyIPOirr", "onlyDZ", "onlyZJ", "onlyOther")}
    assert sets["noZhuijia"] == {0, 1, 2, 4} and sets["onlyIPO"] == {0, 1, 4} and sets["onlyIPOreg"] == {0}
    assert sets["onlyIPOirr"] == {1, 4} and sets["onlyDZ"] == {2} and sets["onlyZJ"] == {3} and sets["onlyOther"] == set()
    ev2 = pd.concat([ev] * 400, ignore_index=True)
    ev2["event_idx"] = np.tile(np.arange(5), 400)
    rng = np.random.default_rng(7)
    wc = RS.scenario_windows(OV, ev2, T, "calib", ipo, rng=rng, rates={"定增": 1.0, "追加承诺": 0.0}, dates=dates)
    m = ev2["tg"].to_numpy()
    assert (wc["start"].to_numpy()[m == "定增"] == 97).all() and (wc["start"].to_numpy()[m == "追加承诺"] == 90).all()


@pytest.mark.skipif(not (LED / "inc_t3_review_sens.csv").exists(), reason="审查敏感性账本未生成")
def test_review_sens_main_reproduces_grid_and_rates():
    S = pd.read_csv(LED / "inc_t3_review_sens.csv")
    g = pd.read_csv(LED / "inc_t3_overlay_grid.csv")
    gm = g[(g["cell"] == "X10_N10_M0") & (g["variant"] == "Y1.0") & (g["aum"] == 1e7)].iloc[0]
    m = S[S["scenario"] == "main"].iloc[0]
    assert abs(m["ann_2021-2025"] - gm["ann_2021-2025"]) < 1e-9
    assert abs(m["ann_2024-2025"] - gm["ann_2024-2025"]) < 1e-9
    assert (S[S["kind"] == "draw"].shape[0]) == RS.R_CALIB
    r = RS.calib_rates()
    assert 0.0 < r["定增"] < 0.5 and 0.0 < r["追加承诺"] < 0.5


@pytest.mark.skipif(not (LED / "inc_t3_review_ipo_ruledate.csv.gz").exists(), reason="规则日分类未生成")
def test_review_ipo_ruledate_shares():
    e = pd.read_csv(LED / "inc_t3_review_ipo_ruledate.csv.gz")
    assert (e["ratio_plan"] >= 0.10).all() and e["event_idx"].is_unique
    assert 0.80 < e["regular"].mean() < 0.95                                # 审查口径约 88%
    assert (e.loc[e["regular"], "gap_cal"].between(0, 7)).all()
