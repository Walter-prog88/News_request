"""附加 α 接口(afactor.optimizer_pead.AdditiveAlphaEnhancer)单测(合成数据,不读面板)。

覆盖:附加 IC ≡ 0 时与 IndexEnhancer 逐位相同;α = α_主 + α_附加(各自 Grinold);附加打分非缺失 < 20 只时不生效;
monkeypatch 在正常返回和异常时都还原;诊断字段与事前 α 对账。
运行:python3 -m pytest tests/test_optimizer_pead.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from afactor import optimizer as OPT  # noqa: E402
from afactor import optimizer_pead as OPTP  # noqa: E402
import test_optimizer as TO  # noqa: E402  (复用合成问题与风险模型替身)


def _setup(seed=21, n_add=None, add_ic=0.04, base_ic=0.05, cfg=None):
    prob = TO.make_problem(seed)
    n = prob.n
    dates = pd.bdate_range("2024-01-01", periods=30)
    codes = pd.Index([f"{i:06d}.SZ" for i in range(n)])
    rng = np.random.default_rng(seed + 100)
    bw = pd.DataFrame(np.tile(prob.wb, (30, 1)), index=dates, columns=codes)
    score = pd.DataFrame(np.tile(rng.normal(size=n), (30, 1)), index=dates, columns=codes)
    add = rng.normal(size=n)
    if n_add is not None:                                   # 只有 n_add 只股票有附加打分(事件池)
        add[n_add:] = np.nan
    add_df = pd.DataFrame(np.tile(add, (30, 1)), index=dates, columns=codes)
    U = pd.DataFrame(True, index=dates, columns=codes)

    class _P:
        pass
    pp = _P(); pp.dates, pp.codes = dates, codes
    cfg = cfg or OPT.OptConfig(te_ann=0.10)
    rm = TO._FakeRM(dates, codes, prob)
    icb = pd.Series(base_ic, index=dates)
    ica = pd.Series(add_ic, index=dates)
    base = OPT.IndexEnhancer(pp, rm, bw, score, icb, U, cfg, tag="base")
    addv = OPTP.AdditiveAlphaEnhancer(pp, rm, bw, score, icb, U, cfg, tag="add", add_scores={"X": (add_df, ica)})
    h = np.zeros(n)
    can = np.ones(n, bool)
    ctx = dict(t=5, te=6, date=dates[5], trade_date=dates[6], phase=0, mode="open10", capital=1e7,
               cost_model=None, h_dec=h, nav_dec=1.0, aum_dec=1e7, can=can, stuck=np.zeros(n, bool),
               cap_w=np.full(n, 1.0))
    return prob, base, addv, ctx, score, add_df, h


def test_zero_add_ic_identical():
    prob, base, addv, ctx, *_ , h = _setup(add_ic=0.0)
    t1 = base(5, h, dict(ctx))
    t2 = addv(5, h, dict(ctx))
    assert np.array_equal(t1, t2, equal_nan=True)
    assert base.log[-1]["obj"] == addv.log[-1]["obj"]
    assert addv.log[-1]["ic:X"] == 0.0


def test_alpha_is_sum_of_grinold(monkeypatch):
    cfg = OPT.OptConfig(te_ann=0.10, cand_top_frac=1.0)     # 候选集 = 全部,便于逐位比较 α 向量
    prob, base, addv, ctx, score, add_df, h = _setup(cfg=cfg)
    cap = {}

    def fake_solve(pr, c):
        cap["alpha"] = pr.alpha.copy()
        return dict(w=None, status="failed", solver=None, cap_relaxed=False, attempts=[], solve_s=0.0,
                    viol=None, obj=np.nan, te_ann=np.nan, cost=np.nan)
    monkeypatch.setattr(OPT, "solve", fake_solve)
    assert addv(5, h, dict(ctx)) is None
    om = np.sqrt(prob.d)
    inv = np.ones(prob.n, bool)
    a0, _ = OPT.grinold_alpha(score.iloc[5].to_numpy(np.float32).astype(np.float64), inv, om, 0.05)  # 引擎内打分存 float32
    a1, _ = OPT.grinold_alpha(add_df.iloc[5].to_numpy(np.float32).astype(np.float64), inv, om, 0.04)
    assert np.allclose(cap["alpha"], a0 + a1, rtol=0, atol=1e-15)
    assert not np.allclose(a1, 0)


def test_add_needs_20_names():
    prob, base, addv, ctx, *_, h = _setup(n_add=15)         # 事件池只有 15 只:z = 0,α_附加 = 0
    t1 = base(5, h, dict(ctx))
    t2 = addv(5, h, dict(ctx))
    assert np.array_equal(t1, t2, equal_nan=True)
    assert addv.log[-1]["n_z:X"] == 0


def test_patch_restored_on_exception(monkeypatch):
    prob, base, addv, ctx, *_, h = _setup()
    orig = OPT.grinold_alpha

    def boom(pr, c):
        raise RuntimeError("x")
    monkeypatch.setattr(OPT, "solve", boom)
    with pytest.raises(RuntimeError):
        addv(5, h, dict(ctx))
    assert OPT.grinold_alpha is orig


def test_zero_ic_candidates_follow_main_score_z():
    """审查后追加(P0 机制):IC ≡ 0 时 α ≡ 0,但上游候选集仍按主打分的 z 取前 30%,
    所以"打分 = COMP、IC = 0"的 P0 非成分候选来自 COMP;打分全缺失(P0c)时候选集只剩成分 ∪ 持仓。"""
    prob, base, addv, ctx, score, add_df, h = _setup()
    n, nb = prob.n, int(prob.bench.sum())
    z0 = OPT.IndexEnhancer(base_pp(base), base.rm, _bw(base), score, pd.Series(0.0, index=base.dates),
                           _U(base), base.cfg, tag="P0")
    blank = score * np.nan
    zc = OPT.IndexEnhancer(base_pp(base), base.rm, _bw(base), blank, pd.Series(0.0, index=base.dates),
                           _U(base), base.cfg, tag="P0c")
    t0 = z0(5, h, dict(ctx))
    t1 = zc(5, h, dict(ctx))
    assert t0 is not None and t1 is not None
    s = score.iloc[5].to_numpy(np.float32).astype(np.float64)
    k = int(np.ceil(base.cfg.cand_top_frac * n))
    top = np.argsort(-((s - s.mean()) / s.std()), kind="stable")[:k]
    extra = np.setdiff1d(top, np.flatnonzero(prob.bench))
    assert len(extra) > 0
    assert z0.log[-1]["n_cand"] == nb + len(extra)          # 成分 ∪ 主打分 z 前 30%
    assert zc.log[-1]["n_cand"] == nb                       # 只剩成分(h = 0,无持仓)
    assert z0.log[-1]["alpha_ex_ante"] == 0.0 and zc.log[-1]["alpha_ex_ante"] == 0.0
    nonb = np.ones(n, bool); nonb[np.flatnonzero(prob.bench)] = False
    assert np.nansum(t1[nonb]) == 0.0                       # P0c 不持有非成分


def base_pp(enh):
    class _P:
        pass
    pp = _P(); pp.dates, pp.codes = enh.dates, enh.codes
    return pp


def _bw(enh):
    return pd.DataFrame(enh.WB, index=enh.dates, columns=enh.codes)


def _U(enh):
    return pd.DataFrame(enh.U, index=enh.dates, columns=enh.codes)


def test_diag_matches_ex_ante_alpha():
    # 主 IC = 0:总 α 全部来自附加信号,事前 α 的两种算法应一致
    prob, base, addv, ctx, *_, h = _setup(base_ic=0.0, add_ic=0.05)
    tgt = addv(5, h, dict(ctx))
    assert tgt is not None
    r = addv.log[-1]
    assert OPT.grinold_alpha is not None and r["status"] in ("optimal", "optimal_inaccurate")
    assert abs(r["alpha_ex_ante"] - r["alpha_ex_ante:X"]) < 1e-10
    assert r["n_z:X"] == prob.n and r["aw:X"] == pytest.approx(0.0, abs=1e-8)   # 全部股票有打分:Σ(w−wb)=0
    assert r["exp:X"] > 0                                    # 正 α 应带来正的 z 暴露
