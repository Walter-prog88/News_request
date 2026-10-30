"""压力测试(前缀 stress)的回归检查。

运行:  python3 -m pytest tests/test_stress_checks.py -q

前提:已跑过 `python3 scripts/run_stress_test.py all`(engine/dims/analyze/figure)。
覆盖(2026-10-10 审查修复后补):
  1 复刻引擎 == ST.run_strategy(ledger 记录 + 现场重跑一个方案逐位核对),exposure <= 1,相关峰值 k=0
  2 "池等权 vs 中证1000":钟点错位的行 sigma 必须为 NaN、同钟行 sigma 有限;同钟日超额 sd 小于错位口径
  3 特质项匹配拆分:档位效应 + 档内选股 == 归因的特质合计
  4 开盘钉跌停口径是一字跌停的超集
  5 stress_survivorship.json 能由脚本函数原样重建(此前是临时脚本算的,不可复现)
  6 封存:所有 stress 收益序列不晚于 2025-12-31
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from afactor.config import LEDGER, TMP, RESEARCH_END   # noqa: E402

RETS = TMP / "stress_rets.parquet"
pytestmark = pytest.mark.skipif(not (RETS.exists() and (LEDGER / "stress_windows.csv").exists()),
                                reason="需要先跑 scripts/run_stress_test.py all")


@pytest.fixture(scope="module")
def p():
    from afactor.panel import Panel
    return Panel()


def test_engine_checks_ledger():
    c = pd.read_csv(LEDGER / "stress_engine_checks.csv")
    d = c[c.check == "replica_vs_ST_max_abs_diff"]
    assert len(d) == 21 and float(d.value.max()) == 0.0
    assert float(c[c.check == "max_exposure"].value.max()) <= 1 + 1e-9
    assert (c[c.check.str.startswith("corr_peak_k")].value == 0).all()


def test_replica_matches_engine_live(p):
    import run_stress_test as RS
    comp = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    res = RS.run_rec(comp, p, "top", n=20, k=5, check=True, tag="pytest")
    assert res["max_abs_diff"] == 0.0
    ref = pd.read_parquet(RETS)["COMP_TOP20"].dropna()
    assert float((res["rets"].reindex(ref.index) - ref).abs().max()) == 0.0


def test_pool_vs_csi_clock():
    w = pd.read_csv(LEDGER / "stress_windows.csv")
    mis = w[w.object.isin(["POOL_EW", "POOL_EW|CSI_te+1"])]
    assert mis.clock_misaligned.all() and mis.excess_sigma.isna().all()
    assert not mis.window.str.startswith("H").any(), "钟点错位口径上不应再找事后最差窗口"
    same = w[w.object == "POOL_C2C"]
    assert np.isfinite(same.excess_sigma).all()
    c = pd.read_csv(LEDGER / "stress_engine_checks.csv").set_index("check")["value"]
    assert c["daily_excess_sd[sameclock_c2c_vs_csi_te]"] < 0.6 * c["daily_excess_sd[open10_vs_csi_te]"]
    # 派生窗口:相对净值列存在且与复利收益一致
    x = w[w.window == "W1+W1b"]
    assert len(x) > 0
    np.testing.assert_allclose(x.rel_nav_change, (1 + x.ret) / (1 + x.bench_ret) - 1, atol=1e-12)


def test_matched_split_reconciles():
    m = pd.read_csv(LEDGER / "stress_specific_matched.csv")
    s = m[m.kind == "匹配拆分"]
    assert len(s) > 0
    np.testing.assert_allclose(s.bucket_part + s.selection_part, s.specific_total, atol=1e-9)
    assert float(s.recon_err.abs().max()) < 1e-6
    # 与归因长表的特质合计一致
    a = pd.read_csv(LEDGER / "stress_attribution.csv")
    a = a[a.component == "specific"].set_index(["object", "window"])["value"]
    for _, r in s.iterrows():
        assert abs(a[(r.object, r.window)] - r.specific_total) < 1e-6


def test_pinned_is_superset_of_one_word(p):
    import run_stress_test as RS
    pin = RS.pinned_down(p).loc["2024-01-02":"2024-02-08"]
    ow = p["one_word_down"].loc["2024-01-02":"2024-02-08"].reindex_like(pin).fillna(False)
    assert not (ow & ~pin).to_numpy().any()
    assert int(pin.to_numpy().sum()) > int(ow.to_numpy().sum())


def test_survivorship_reproducible(p):
    import stress_analyze as SA
    old = json.load(open(LEDGER / "stress_survivorship.json"))
    new = SA.write_survivorship(p)
    assert new == old
    assert new["delisted_2020_2025_missing"] == 204 and new["missing_listed_through_W1"] == 77


def test_seal():
    r = pd.read_parquet(RETS)
    assert r.index.max() <= pd.Timestamp(RESEARCH_END)
    for f in ("rets_navcap", "cash_navcap", "dims"):
        x = pd.read_parquet(TMP / f"stress_{f}.parquet")
        assert x.index.max() <= pd.Timestamp(RESEARCH_END)
