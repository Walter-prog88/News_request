"""T5 多风格篮子的合成数据测试(前缀 inc_t5;不读真实面板,不算任何真实收益)。

PYTHONPATH=src python3 -m pytest -q tests/test_inc_t5.py
"""
import numpy as np
import pandas as pd

from afactor import inc_t5_signals as SG
from afactor import inc_t5_stats as S5
from afactor import backtest as BT


# ------------------------------------------------------------------ 信号
def test_lowvol_masks_suspension_and_min_obs():
    T, N = 200, 3
    rng = np.random.default_rng(0)
    close = np.cumprod(1 + rng.normal(0, [0.01, 0.03, 0.02], (T, N)), axis=0)
    valid = np.ones((T, N), bool)
    valid[50:60, 2] = False                    # 停牌 10 天
    lv = SG.lowvol(close, valid, 120, 60)
    assert np.isnan(lv[:60]).all()              # 观测不足 60
    assert lv[150, 0] > lv[150, 1]              # 波动小 -> 打分高(负的 std 更接近 0)
    # 复牌日收益(跨停牌)不计:窗口内收益个数 = 119 - 11(停牌 10 天 + 复牌日)
    r = close[1:] / close[:-1] - 1
    keep = np.ones(T - 1, bool); keep[49:60] = False           # r 下标 i 对应 t=i+1;t=50..60 的收益均剔除
    win = np.arange(T - 1)
    exp = -np.std(r[(win >= 150 - 120) & (win < 150) & keep, 2], ddof=1)
    assert abs(lv[150, 2] - exp) < 1e-9


def test_reversal_requires_both_ends_valid():
    close = np.tile(np.arange(1, 31, dtype=float)[:, None], (1, 2)) * np.array([1.0, 2.0])
    valid = np.ones((30, 2), bool)
    valid[10, 1] = False
    rv = SG.reversal(close, valid, 20)
    assert abs(rv[25, 0] - (-(26 / 6 - 1))) < 1e-9
    assert np.isfinite(rv[29, 1])                       # t-20 = 9 有效
    assert np.isnan(rv[:20]).all()                      # 前 20 行没有 t-k
    rv10 = SG.reversal(close, valid, 10)
    assert np.isnan(rv10[20, 1]) and np.isfinite(rv10[20, 0])       # t-10 = 10 在列 1 无效


def test_cs_rank_pool_only_ranks_pool():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, np.nan, 1.0]])
    pool = np.array([[True, True, True, False], [True, True, True, True]])
    r = SG.cs_rank_pool(x, pool)
    assert np.allclose(r[0, :3], [1 / 3, 2 / 3, 1.0]) and np.isnan(r[0, 3])
    assert np.isnan(r[1, 2]) and np.allclose(r[1, [3, 1, 0]], [1 / 3, 2 / 3, 1.0])


def test_per_share_conversion_after_split_and_dividend_ratio():
    """1 拆 2 送转后,每股量按 f_x/adj_t 换算:BPS_e = 10,e 时 adj=raw=20(f=1);t 时后复权 adj=40、raw=20(f=2)...
    这里用 hfq 定义:adj/raw 在 e 为 1,t 为 2 -> BP_t = BPS_e*f_e/adj_t = 10*1/40 = 0.25 = (10/2)/20。"""
    T, N = 10, 1
    dates = pd.bdate_range("2020-01-01", periods=T)
    raw = np.array([20.0] * 5 + [10.0] * 5)[:, None]            # 第 5 日 1 拆 2,价格减半
    adj = np.array([20.0] * 5 + [20.0] * 5)[:, None]            # 后复权价连续
    f = SG.adj_ratio_ffill(adj, raw)
    assert np.allclose(f[:5, 0], 1.0) and np.allclose(f[5:, 0], 2.0)
    bps_e, f_e = 10.0, f[2, 0]
    bp_t = bps_e * f_e / adj[7, 0]
    assert abs(bp_t - (bps_e / 2) / raw[7, 0]) < 1e-12


def test_dy_numerator_window_and_hfq():
    dates = pd.bdate_range("2020-01-01", periods=800)
    codes = pd.Index(["A.SZ", "B.SZ"])
    f_raw = np.ones((800, 2)); f_raw[300:, 0] = 2.0                # A 在第 300 日后 f 翻倍
    ev = pd.DataFrame({"lp_code": ["A.SZ", "A.SZ", "B.SZ"],
                       "ann": [dates[100], dates[350], dates[10]], "cash_ps": [0.5, 0.4, 0.0],
                       "impl": [True, False, True], "src": "em"})
    num = SG.dy_numerator(ev, dates, codes, f_raw, 365, False)
    end1 = int(dates.searchsorted(dates[100] + pd.Timedelta(days=365), side="left"))
    end2 = int(dates.searchsorted(dates[350] + pd.Timedelta(days=365), side="left"))
    assert end1 > 350 and end2 < 800
    assert num[99, 0] == 0 and abs(num[100, 0] - 0.5) < 1e-6                 # 公告日当日生效
    assert abs(num[end1 - 1, 0] - (0.5 + 0.4 * 2.0)) < 1e-6                  # 第二笔按公告日 f_a = 2 换算
    assert abs(num[end1, 0] - 0.4 * 2.0) < 1e-6                              # 第一笔 365 自然日后失效
    assert abs(num[end2, 0]) < 1e-6
    assert (num[:, 1] == 0).all()                                            # 0 现金不计
    num_i = SG.dy_numerator(ev, dates, codes, f_raw, 365, True)
    assert abs(num_i[end1 - 1, 0] - 0.5) < 1e-6                              # 只计实施:第二笔 impl=False 被剔除
    num_730 = SG.dy_numerator(ev, dates, codes, f_raw, 730, False)
    assert abs(num_730[end1, 0] - (0.5 + 0.8)) < 1e-6                        # 730 日窗口:第一笔仍在


def test_new_listing_first_valid_f():
    f_raw = np.full((10, 2), np.nan)
    f_raw[4:, 0] = 1.5
    f_raw[:, 1] = 1.0
    fv = SG.first_valid_f(f_raw)
    assert fv[0] == 1.5 and fv[1] == 1.0
    fff = pd.DataFrame(f_raw).ffill().to_numpy()
    v = SG.f_at(fff, np.array([2, 2, -1]), np.array([0, 1, 0]), fv)        # e 早于上市 -> 取首个有效 f
    assert v[0] == 1.5 and v[1] == 1.0 and v[2] == 1.5


def test_report_pit_visibility_roe_sigma():
    qi0 = 2015 * 4
    rows = []
    for k in range(10):
        e = pd.Timestamp(2015 + k // 4, 3 * (k % 4) + 3, 1) + pd.offsets.MonthEnd(0)
        rows.append(dict(stock_code="X.SZ", qi=qi0 + k, end_date=e, n_income_attr_p_q=10.0 + (k % 3),
                         total_hldr_eqy_exc_min_int=100.0, avail_q=e + pd.Timedelta(days=40 + k)))
    d = pd.DataFrame(rows)
    out = SG.roe_sigma_rows(d, 8, 6)
    assert out["end_date"].min() == pd.Timestamp("2016-06-30")              # 第 6 期起才有 sigma
    first = out.iloc[0]
    assert first["avail_ttm"] == d.loc[5, "avail_q"]                         # 窗口最晚可见日 = 第 6 期的 avail_q
    assert (out["avail_ttm"] >= out["end_date"]).all()


# ------------------------------------------------------------------ 缓冲带篮子
def test_style_buffer_holds_better_half():
    rng = np.random.default_rng(1)
    T, N = 30, 400
    F = rng.normal(size=(T, N)).astype(np.float32)
    M = np.ones((T, N), bool)
    W = BT.weights_exclude_buffer(F, M, 1, 0.45, 0.60)
    n_hold = (W > 0).sum(axis=1)
    assert 0.40 * N <= n_hold[0] <= 0.60 * N
    assert np.allclose(W.sum(axis=1), 1.0)
    assert (W[0][F[0] < np.quantile(F[0], 0.40)] == 0).all()               # 最差 40% 不持有


# ------------------------------------------------------------------ 合成与等风险
def test_combine_weights_normalizes_empty_baskets():
    T, N = 4, 6
    W1 = np.zeros((T, N), np.float32); W1[:, :3] = 1 / 3
    W2 = np.zeros((T, N), np.float32); W2[1:, 3:] = 1 / 3
    Wc = S5.combine_weights([W1, W2], [0.5, 0.5])
    assert np.allclose(Wc[0], W1[0])                       # 第 0 日 W2 空 -> 全给 W1
    assert np.allclose(Wc[1:].sum(axis=1), 1.0) and abs(Wc[1, 0] - 1 / 6) < 1e-7
    Wc2 = S5.combine_weights([W1, W2], np.array([[1.0, 0.0]] * T))
    assert np.allclose(Wc2, W1)
    Wz = S5.combine_weights([np.zeros((T, N), np.float32)] * 2, [0.5, 0.5])
    assert (Wz == 0).all()


def test_core_weights():
    a = S5.core_weights(5, 0, 0.5)
    assert abs(a.sum() - 1) < 1e-12 and a[0] == 0.5 and np.allclose(a[1:], 0.125)


def test_erc_weights_inverse_te_clip_and_lag():
    T, K = 600, 3
    rng = np.random.default_rng(2)
    sd = np.array([0.01, 0.02, 0.04])
    ex = pd.DataFrame(rng.normal(0, 1, (T, K)) * sd)
    a = S5.erc_weights(ex, 250, 2)
    assert np.allclose(a[:250 + 1], 1 / K)                 # 不足 250 日(含滞后)等权
    assert np.allclose(a.sum(axis=1), 1.0)
    assert a[400, 0] > a[400, 1] > a[400, 2]               # 低 TE 篮子权重更大
    # 预注册:截到 [0.5/K, 2/K] 后归一 -> 归一后可略微越界,但不会越出很多
    assert a[400].min() >= 0.15 and a[400].max() <= 2 / K
    # 只用 s <= t-2:改动第 t-1、t 日的超额不改变 t 日权重
    ex2 = ex.copy(); ex2.iloc[399:401] += 10.0
    a2 = S5.erc_weights(ex2, 250, 2)
    assert np.allclose(a[400], a2[400]) and not np.allclose(a[402], a2[402])


def test_erc_ignores_empty_basket_days():
    ex = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, (400, 2)))
    ex.iloc[:200, 1] = np.nan                               # 篮子 2 前 200 日空仓
    a = S5.erc_weights(ex, 250, 2)
    assert np.allclose(a[:452 - 52], 0.5)
    assert np.allclose(a[399], 0.5)                         # 窗口内仍有 NaN -> 等权


# ------------------------------------------------------------------ 生成检验 / 统计
def test_hac_regress_recovers_alpha_beta():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.01, 3000)
    y = 0.0003 + 0.4 * x + rng.normal(0, 0.003, 3000)
    r = S5.hac_regress(y, x, 5)
    assert abs(r["b"] - 0.4) < 0.05 and abs(r["a"] - 0.0003) < 0.0002 and r["t_a"] > 2 and 0.3 < r["r2"] < 0.9


def test_block_bootstrap_dir_sign():
    rng = np.random.default_rng(5)
    base = rng.normal(0.0002, 0.006, 2000)
    better = base * 0.7 + rng.normal(0.0002, 0.0005, 2000)   # 更低波动、相近超额
    bb = S5.block_bootstrap_dir(better, base, 20, 400, 0)
    assert bb["dir"] > 0 and bb["lo90"] < bb["dir"] < bb["hi90"] and bb["p_pos"] > 0.9


def test_rel_underwater_basic():
    idx = pd.bdate_range("2020-01-01", periods=10)
    rb = pd.Series(0.0, index=idx)
    rs = pd.Series([0.1, -0.05, -0.05, -0.05, 0.2, 0.0, -0.01, 0.0, 0.0, 0.0], index=idx)
    r = S5.rel_underwater(rs, rb)
    assert r["rel_uw_longest_days"] == 4 and r["rel_mdd"] < -0.13


def test_nonoverlap_sum():
    s = pd.Series(np.arange(45, dtype=float), index=pd.bdate_range("2020-01-01", periods=45))
    o = S5.nonoverlap_sum(s, 20)
    assert len(o) == 2 and o.iloc[0] == sum(range(20))


def test_verdict_rules():
    p = dict(P1=True, P2=True, P3=True, P4=True, P5=True, P6=True, P7=True, dIR_full=0.2, a_full=0.01)
    assert S5.verdict(p, True)["verdict"] == "有增量"
    assert S5.verdict(p, False)["verdict"] == "证据不足"
    assert S5.verdict(p, True, c_downgrade=True)["verdict"] == "证据不足"
    q = dict(p, P2=False)
    assert S5.verdict(q, True)["verdict"] == "证据不足"
    assert S5.verdict(dict(p, dIR_full=-0.01), True)["verdict"] == "无增量"
    assert S5.verdict(dict(p, a_full=-0.001), True)["verdict"] == "无增量"
    assert S5.verdict(dict(p, P5=False), True)["verdict"] == "无增量"


# ------------------------------------------------------------------ 配置登记与 C 轨可见日(不跑回测)
def test_config_registries_cover_prereg_grid():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import inc_t5_runs as R
    lh = R.lh_configs()
    # 篮子单独 x 深度 = 12;信号窗口变体 10;组合 6 x 2 + S4/S5 x 2;安慰剂 10;成本口径 4
    assert all(f"S|{b}|{d}" in lh for b in ("LIQ", "LOWVOL", "VALUE", "DIV", "QUAL", "REV") for d in ("45_60", "20_35"))
    assert all(f"S|{v}|45_60" in lh for v in R.VARIANT_SCORE)
    assert all(f"C|{s}|{d}" in lh for s in ("EW5", "CORE5", "ERC5", "EW6", "CORE6", "ERC6", "S4", "S5") for d in ("45_60", "20_35"))
    assert all(f"P|{s}" in lh for s in range(10))
    assert {"S|LIQ|20_35|old", "S|LIQ|20_35|pess", "C|EW5|45_60|old", "C|EW5|45_60|pess"} <= set(lh)
    a = R.a_configs()
    assert all(f"C|{s}_{w}" in a for s in ("A2", "A3", "COREA2") for w in (60, 120, 250))
    assert all(f"P|{s}" in a for s in range(10))
    # 风格篮子至少一半对应的 ERC 成员都有对应的单独篮子运行(ERC 的前置)
    for n, c in lh.items():
        if c["kind"] == "combo" and c["scheme"] == "ERC":
            for m in c["members"]:
                dd = "20_35" if m == "LIQ" else c["depth"]
                assert f"S|{m}|{dd}" in lh


def test_c_track_statutory_deadline():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import inc_t5_c as C
    assert C.deadline(pd.Timestamp("2008-03-31")) == pd.Timestamp("2008-04-30")
    assert C.deadline(pd.Timestamp("2008-06-30")) == pd.Timestamp("2008-08-31")
    assert C.deadline(pd.Timestamp("2008-09-30")) == pd.Timestamp("2008-10-31")
    assert C.deadline(pd.Timestamp("2008-12-31")) == pd.Timestamp("2009-04-30")
