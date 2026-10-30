"""PEAD(前缀 pead)执行外挂与下游信号面板的回归检查。

运行:  python3 -m pytest tests/test_pead_engine.py -q

覆盖(2026-10-10 审查修复后补):
  1 整手:主板/创业板 100 股向下取整;科创板最低 200 股、之后 1 股递增;价格无效 -> 0 股
  2 成本钩子:每笔最低 5 元佣金补差、< 1 元的浮点残差不收、+bp 压力按 |Δw| 计
  3 信号面板只能直接用于 lag>=1 口径:lag=0 默认报错;shift / mask 两种处理的数值
  4 信号面板伴随文件:known_at_t_close <=> V <= t;P 早于成交日;不含封存期;与 ledger 计数一致
  5 封存核对与事件层/账户层对账的 ledger 不变式
数据类检查在 cache/_tmp、research/ledger 产物缺失时跳过(需先跑 scripts/pead_build_batches.py 等)。
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor import pead_engine as PE          # noqa: E402
from afactor.config import LEDGER, TMP, RESEARCH_END, COST_BPS_ONEWAY   # noqa: E402

SIG = TMP / "pead_signal_open10.parquet"
SIG_EV = TMP / "pead_signal_open10_events.parquet"
need_signal = pytest.mark.skipif(not (SIG.exists() and SIG_EV.exists()
                                      and (LEDGER / "pead_signal_panel_meta.csv").exists()),
                                 reason="需要先跑 scripts/pead_build_batches.py --signal-only")


# ------------------------------------------------------------------ 1 整手
def test_lot_shares_main_and_star():
    px = np.array([10.0, 10.0, 10.0, 10.0, np.nan, 0.0])
    val = np.array([1999.0, 2000.0, 1500.0, 2510.0, 5000.0, 5000.0])
    star = np.array([False, False, True, True, False, False])
    sh = PE.lot_shares(val, px, star)
    # 主板 199.9 股 -> 100;200 股 -> 200;科创 150 股 < 200 -> 0;科创 251 股 -> 251;无效价格 -> 0
    assert sh.tolist() == [100.0, 200.0, 0.0, 251.0, 0.0, 0.0]


def test_star_mask():
    m = PE.star_mask(["688001.SH", "689009.SH", "600000.SH", "300750.SZ"])
    assert m.tolist() == [True, True, False, False]


# ------------------------------------------------------------------ 2 成本钩子
def test_event_fees_min_commission_and_stress():
    fees = PE.EventFees(min_commission=5.0, stress_bps=5.0)
    capital, nav = 1e6, 1.0
    # 三笔:1 万元(佣金 1 元 -> 补 4 元)、10 万元(佣金 10 元 -> 不补)、0.5 元(浮点残差,不收)
    dw = np.array([1e4, 1e5, 0.5]) / capital
    a, b = fees.extra_cost(0, dw, capital, nav)
    exp_min = max(0.0, 5.0 - 1e4 * COST_BPS_ONEWAY * 1e-4) / capital
    assert np.isclose(a, exp_min + 5e-4 * dw.sum())
    assert b == 0.0
    te, n_tr, n_min, mc, stv, sp, im = fees.log[0]
    assert (n_tr, n_min) == (2, 1)
    assert np.isclose(mc, exp_min) and np.isclose(stv, 5e-4 * dw.sum())


def test_event_fees_zero_when_disabled():
    fees = PE.EventFees(min_commission=0.0, stress_bps=0.0)
    a, b = fees.extra_cost(0, np.array([1e-3, 2e-3]), 1e6, 1.0)
    assert a == 0.0 and b == 0.0


# ------------------------------------------------------------------ 3/4 信号面板
def _fake_panel():
    S = pd.read_parquet(SIG)
    return SimpleNamespace(dates=S.index, codes=S.columns), S


@need_signal
def test_signal_panel_lag_guard():
    p, S = _fake_panel()
    for mode in ("open10", "open5", "vwap1"):
        out = PE.load_signal_panel(p, mode)
        assert out.equals(S)
    for mode in ("tail1450", "tail1445", "close"):
        with pytest.raises(ValueError):
            PE.load_signal_panel(p, mode)
    sh = PE.load_signal_panel(p, "tail1450", lag0="shift")
    assert sh.iloc[1:].equals(S.iloc[:-1].set_axis(S.index[1:]))
    assert sh.iloc[0].isna().all()
    mk = PE.load_signal_panel(p, "close", lag0="mask")
    meta = pd.read_csv(LEDGER / "pead_signal_panel_meta.csv").set_index("item")
    assert int(mk.notna().sum().sum()) == int(meta.loc["P_rel_t:P<t", "n"])
    # mask 只删格子,不改值
    m = mk.notna().to_numpy()
    assert np.array_equal(mk.to_numpy()[m], S.to_numpy()[m])


@need_signal
def test_signal_panel_attrs_and_companion():
    S = pd.read_parquet(SIG)
    assert "lag" in S.attrs.get("doc", "") and S.attrs.get("min_lag") == "1"
    assert S.index.max() <= pd.Timestamp(RESEARCH_END)
    L = pd.read_parquet(SIG_EV)
    assert len(L) == int(S.notna().sum().sum())
    v = S.to_numpy()[L["t_pos"].to_numpy(), L["code_pos"].to_numpy()]
    assert np.array_equal(v, L["value"].to_numpy())
    assert (L["known_at_t_close"] == (L["V"] <= L["t_signal"])).all()
    assert (L["known_at_t_close"] == (L["P"] < L["t_signal"])).all()
    assert (L["P"] < L["T"]).all() and (L["V"] <= L["T"]).all()
    assert L["T"].max() <= pd.Timestamp(RESEARCH_END)
    meta = pd.read_csv(LEDGER / "pead_signal_panel_meta.csv").set_index("item")
    for k in ("P<t", "P==t", "P>t"):
        assert int((L["P_rel_t"] == k).sum()) == int(meta.loc[f"P_rel_t:{k}", "n"])


# ------------------------------------------------------------------ 5 ledger 不变式
@pytest.mark.skipif(not (LEDGER / "pead_seal_audit.csv").exists(), reason="需要 pead_seal_audit.csv")
def test_seal_audit():
    A = pd.read_csv(LEDGER / "pead_seal_audit.csv").set_index("item")
    assert pd.Timestamp(A.loc["daily_basic_rows_after_truncation", "value"]) <= pd.Timestamp(RESEARCH_END)
    # 与数据验收的冲突表同一口径
    C = pd.read_csv(LEDGER / "pead_event_conflicts.csv")
    row = C[(C["section"] == "主指标 v1") & (C["item"] == "value_status=own_revised_sealed")]
    assert len(row) == 1
    assert int(row["n"].iloc[0]) == int(A.loc["own_revised_sealed_rows", "n"])
    assert int(row["denom"].iloc[0]) == int(A.loc["main_indicator_first_disclosure_rows", "n"])


@pytest.mark.skipif(not (LEDGER / "pead_event_account_recon.csv").exists(),
                    reason="需要 pead_event_account_recon.csv")
def test_event_account_recon_signs():
    """按 E 目标权重加权的事件层 E−B 与账户层(无整手无最低佣金)同号;n_sel 拆分的加权年化相加 = 全部。"""
    R = pd.read_csv(LEDGER / "pead_event_account_recon.csv")
    for w in ("dev", "test"):
        x = R[(R.control == "B") & (R.window == w)]
        wa = x[(x.kind == "weight_ann_pp") & (x.subset == "all")]["est"].iloc[0]
        big = x[(x.kind == "weight_ann_pp") & (x.subset == "n_sel>=20")]["est"].iloc[0]
        small = x[(x.kind == "weight_ann_pp") & (x.subset == "n_sel<20")]["est"].iloc[0]
        acct = x[x.kind == "account_ann_pp:nolot_1m"]["est"].iloc[0]
        assert np.sign(wa) == np.sign(acct)
        assert np.isclose(big + small, wa, atol=1e-9)
        ws = x[(x.kind == "weight_ann_pp") & (x.subset != "all")]["weight_share"].sum()
        assert np.isclose(ws, 1.0)
