"""前向检验对抗审查之后的补充检验(前缀 fwd;审查记录见 research/FORWARD_TEST_2026Q3.md §14)。

**全部是事后诊断 / 敏感性描述:不替换主读数,不改任何冻结参数,不进判定。** 只计算预注册评估窗(信号日 09-01 ~ 09-30,
收益到 10-09)的读数;2026-01 ~ 08 的数据只作特征输入,不算任何收益。结果写 research/ledger/fwd_review_fixes.csv。

各节:
  compare  复权基准修正(审查 P0)前后的主读数对照:修正前读数取 cache/_tmp/fwd_prefix_v1/ledger/fwd_results.csv(修正前备份),
           修正后取 research/ledger/fwd_results.csv
  kmeans   统计行业标签 KMeans 的输入敏感性(审查 P2):收盘价乘 exp(N(0, 1e-6)) 的中性微扰只喂给标签计算
           (fwd_adapters.statind_labels_for_quarters,与冻结算法、季度种子相同),其余输入不变,重算打分 -> T1 / T2 / T3 / T5。
           seed=-1 为不加噪声的重算(必须逐位复现账本,检验管线本身确定)
  cond     读数对窗口市场状态的条件位置(审查 P2):历史 17 日窗口上把读数回归到 (同频池窗口收益, 同频池 − 中证 1000 代理),
           前向读数的残差百分位与"相似状态"子集里的百分位(诊断,不改判定)
  st       ST 主板涨跌幅(审查 P1):数据显示 2026-07 起主板 ST 日涨跌幅可达 ±10%,面板(冻结的 panel._limit_frame)仍按 5% 判定。
           计数误判,并用只换涨跌幅规则的面板(fwd 适配,不改冻结代码)重跑 T1 / T2 / T3 与基准,看读数是否变化

    python3 scripts/fwd_review_checks.py compare kmeans cond st
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor.config import TMP, LEDGER, CACHE
from afactor.fwd_panel import ForwardPanel, forward_context, SIG_START
from afactor import minute5 as M5
import fwd_common as FC

log = FC.log
OUT = LEDGER / "fwd_review_fixes.csv"
PREFIX = TMP / "fwd_prefix_v1"                          # 复权基准修正之前的产物备份
SKIP = M5.FWD_SKIP_HEAD
DIST = pd.read_parquet(LEDGER / "freeze_ref_window_dist.parquet")
_rows: list[dict] = []


def rec(section, item, value, note=""):
    _rows.append(dict(section=section, item=item, value=value, note=note))
    v = f"{value:.6g}" if isinstance(value, (float, np.floating)) else value
    log(f"[{section}] {item} = {v}" + (f"  ({note})" if note else ""))


def flush():
    secs = {r["section"] for r in _rows}
    FC.ledger_append(OUT, _rows, drop_where=lambda o: o["section"].isin(secs))
    _rows.clear()


def dist(key):
    return DIST[DIST["key"] == key]["value"]


def wex(rs: pd.Series, rb: pd.Series) -> float:
    v, n = M5.forward_window_excess(pd.DataFrame({"strategy": rs, "benchmark": rb}), skip_head=SKIP)
    assert n == 17, n
    return v


# ============================================================================== compare
READ_KEYS = [("P1", "T1_excl", "old", "pool5"), ("P2", "T1_excl", "Y1", "pool5"), ("P3", "T2_top20", "old", "pool5"),
             ("T2 Y=1(描述)", "T2_top20", "Y1", "pool5"), ("P4", "T3_V1", "Y1", "csi1000_actual_exec"),
             ("P6", "T5_V1", "Y1", "csi1000_actual_exec"), ("P5(D1)", "T4_PEAD", "base", "B"),
             ("T1 对中证1000(次要)", "T1_excl", "old", "csi1000_actual_exec"),
             ("T2 对中证1000(次要)", "T2_top20", "old", "csi1000_actual_exec"),
             ("T3 对同频池(描述)", "T3_V1", "Y1", "pool5"),
             ("PEAD 账户 100 万(描述)", "T4_PEAD_account", "base_1m", "B"),
             ("PEAD 账户 1000 万(描述)", "T4_PEAD_account", "base_10m", "B"), ("CHK", "CHK_poolew", "Y1", "pool5")]


def stage_compare():
    old = pd.read_csv(PREFIX / "ledger" / "fwd_results.csv")
    new = pd.read_csv(LEDGER / "fwd_results.csv")

    def pick(R, o, c, b, d1=False):
        x = R[(R["section"].isin(["reading", "engine_check"])) & (R["object"] == o) & (R["cost"] == c) & (R["bench"] == b)]
        if o == "T4_PEAD":
            x = x[x["reading"].astype(str).str.contains("D1")] if d1 else x[~x["reading"].astype(str).str.contains("D1")]
        return x.iloc[0]
    for lab, o, c, b in READ_KEYS:
        a = pick(old, o, c, b, d1=("D1" in lab)); z = pick(new, o, c, b, d1=("D1" in lab))
        f = (lambda v: f"{v:.4f}") if o == "T5_V1" else (lambda v: f"{v * 100:+.3f}%")
        rec("compare", lab, f"{f(a['window_excess'])}({a['hist_pct']:.1f}) -> {f(z['window_excess'])}({z['hist_pct']:.1f})",
            f"修正前 -> 修正后;判定 {a['verdict']} -> {z['verdict']}")
        rec("compare", f"{lab}|delta", float(z["window_excess"] - a["window_excess"]),
            f"百分位 {a['hist_pct']:.2f} -> {z['hist_pct']:.2f}")
    t5 = new[(new["object"] == "T5_V1") & (new["section"] == "reading")].iloc[0]
    q5 = pd.read_csv(LEDGER / "freeze_ref_quantiles.csv")
    q5 = q5[(q5["target"] == "T5_V1") & (q5["n"] == 17)].iloc[0]
    rec("compare", "P6_chi2_band_position", f"b_d {t5['window_excess']:.4f} vs 区间 [{q5['chi2_band_lo']:.4f}, {q5['chi2_band_hi']:.4f}]",
        "修正前 1.2433 在区间内;修正后高出上界 0.0007 —— 落在边界上,见 kmeans 节的微扰区间")
    flush()


# ============================================================================== kmeans
def _t3_t5(p, pf, s, rm, cm, W, Wx, U, B):
    from afactor import strategy as ST, optimizer as OPT
    import fwd_run_test as RT
    m = p.mask_from(SIG_START)
    icd = pd.Series(RT.IC_FROZEN, index=p.dates)
    R3, Hs = [], []
    for ph in range(5):
        enh = OPT.IndexEnhancer(p, rm, W, s, icd, U, OPT.OptConfig(), tag="fwd_review")
        res = ST.simulate(enh, p, "open10", RT.CAPITAL, rebalance=5, date_mask=m, phase=ph, cost_model=cm, record=True)
        R3.append(res["rets"])
        Hs.append(pd.DataFrame(res["record"]["holdings"], index=res["record"]["daily"].index, columns=p.codes))
    r3 = pd.concat(R3, axis=1).mean(axis=1)
    H = sum(h.astype(np.float64).fillna(0.0) for h in Hs) / len(Hs)
    wd = p.dates[p.dates >= SIG_START]
    Wv = pd.DataFrame(Wx, index=wd, columns=p.codes).reindex(H.index).to_numpy(np.float64)
    tp = RT.te_pred_series(H, Wv, rm, p.dates, p.codes)
    z = (r3 - B["csi1000_exec"]).reindex(tp.index) / (tp / np.sqrt(242))
    return r3, float(z.iloc[SKIP:].std(ddof=1))


def stage_kmeans(seeds=(-1, 0, 1, 2, 3, 4, 5, 6, 7), sd=1e-6):
    from afactor import strategy as ST, fwd_adapters as FA_, fwd_weights as FW
    import fwd_run_test as RT
    pf, p = RT.panels()
    cm = RT.cost_model(pf)
    rm = RT.load_rm()
    B = pd.read_parquet(TMP / "fwd_bench.parquet")
    W = FW.csi1000_close_weights(p)
    Wx = np.load(TMP / "fwd_csi1000_Wx.npy").astype(np.float64)
    U = ST.universe(p, "V1000")
    m = p.mask_from(SIG_START)
    close = pf.raw("close_px")
    L0 = pf._derived["statind_L"]
    R = pd.read_csv(LEDGER / "fwd_results.csv")
    led = {(r.object, r.cost, r.bench): r.window_excess for r in R[R["section"] == "reading"].itertuples()}
    from sklearn.metrics import adjusted_rand_score
    out = []
    for sd_seed in seeds:
        t0 = time.time()
        if sd_seed < 0:
            Ln = L0
        else:
            rng = np.random.default_rng(20261010 + sd_seed)
            noise = np.exp(rng.normal(0.0, sd, close.shape)).astype(np.float32)
            Ln = FA_.statind_labels_for_quarters(close * noise, ["2025Q4", "2026Q1", "2026Q2", "2026Q3"])
            r25 = Ln.index.year == 2025
            Ln.loc[r25] = L0.loc[r25].to_numpy()
            Ln = Ln.astype(np.int16)
        a_, b_ = L0.loc["2026-09-01"], Ln.loc["2026-09-01"]
        ok = (a_ >= 0) & (b_ >= 0)
        ari = float(adjusted_rand_score(a_[ok], b_[ok]))

        class _P(ForwardPanel):
            def __init__(self):
                super().__init__(returns_view=False)
                self._derived["statind_L"] = Ln
        with forward_context("前向检验审查:统计行业标签微扰敏感性(只算预注册评估窗读数,不进判定)"):
            s = FC.comp_score(_P(), cache_file=None)
        row = dict(seed=sd_seed, ari_sep=ari)
        for plan, kw in RT.PLANS.items():
            for cost, cmx in (("old", None), ("Y1", cm)):
                r = ST.run_strategy(s, p, "V1000", "open10", date_mask=m, cost_model=cmx, **kw)["rets"]
                v = wex(r, B["pool5"])
                row[f"{plan}|{cost}"] = v
                row[f"{plan}|{cost}|pct"] = M5.percentile_in(v, dist(f"{plan}|{cost}|pool5|n17"))
        r3, bd = _t3_t5(p, pf, s, rm, cm, W, Wx, U, B)
        v3 = wex(r3, B["csi1000_exec"])
        row.update({"T3_V1": v3, "T3_V1|pct": M5.percentile_in(v3, dist("T3_V1|Y1|proxy_exec|n17")),
                    "T5_V1_bd": bd, "T5_V1_bd|pct": M5.percentile_in(bd, dist("T5_V1|Y1|proxy_exec|n17|b_d"))})
        out.append(row)
        log(f"[kmeans] seed {sd_seed}: ARI {ari:.3f} P1 {row['T1_excl|old']*100:+.3f}% P3 {row['T2_top20|old']*100:+.3f}%"
            f"({row['T2_top20|old|pct']:.1f}) P4 {v3*100:+.3f}%({row['T3_V1|pct']:.1f}) b_d {bd:.4f} [{time.time()-t0:.0f}s]")
        if sd_seed < 0:     # 不加噪声的重算必须逐位复现账本
            for k_, key in (("T1_excl|old", ("T1_excl", "old", "pool5")), ("T2_top20|old", ("T2_top20", "old", "pool5")),
                            ("T3_V1", ("T3_V1", "Y1", "csi1000_actual_exec")), ("T5_V1_bd", ("T5_V1", "Y1", "csi1000_actual_exec"))):
                rec("kmeans", f"reproduce_ledger[{k_}]", float(abs(row[k_] - led[key])), "不加噪声重算 vs 账本的绝对差(应 ≤ 1e-12)")
    D = pd.DataFrame(out)
    D.to_csv(TMP / "fwd_review_kmeans.csv", index=False)
    pert = D[D["seed"] >= 0]
    rec("kmeans", "n_perturbations", len(pert), f"收盘价乘 exp(N(0,{sd:g})) 只喂给统计行业标签;种子 20261010+k")
    rec("kmeans", "ARI_Sep_labels_range", f"{pert['ari_sep'].min():.3f} ~ {pert['ari_sep'].max():.3f}",
        "9 月所用标签(2026Q2 季末聚类)与账本标签的调整兰德指数")
    for lab, k_, pk, fmt in (("P1 T1 旧成本", "T1_excl|old", "T1_excl|old|pct", "pct"),
                             ("P2 T1 Y=1", "T1_excl|Y1", "T1_excl|Y1|pct", "pct"),
                             ("P3 T2 旧成本", "T2_top20|old", "T2_top20|old|pct", "pct"),
                             ("T2 Y=1(描述)", "T2_top20|Y1", "T2_top20|Y1|pct", "pct"),
                             ("P4 T3 对中证1000", "T3_V1", "T3_V1|pct", "pct"),
                             ("P6 b_d", "T5_V1_bd", "T5_V1_bd|pct", "raw")):
        x, q = pert[k_], pert[pk]
        if fmt == "raw":
            rng_ = f"{x.min():.4f} ~ {x.max():.4f}(中位 {x.median():.4f})"
        else:
            rng_ = f"{x.min()*100:+.3f}% ~ {x.max()*100:+.3f}%(中位 {x.median()*100:+.3f}%)"
        rec("kmeans", f"{lab}|range", rng_, f"百分位 {q.min():.1f} ~ {q.max():.1f};账本值 "
            + (f"{float(D.loc[D.seed < 0, k_].iloc[0]):.4f}" if fmt == "raw" else f"{float(D.loc[D.seed < 0, k_].iloc[0])*100:+.3f}%"))
    q5 = pd.read_csv(LEDGER / "freeze_ref_quantiles.csv")
    q5 = q5[(q5["target"] == "T5_V1") & (q5["n"] == 17)].iloc[0]
    inb = ((pert["T5_V1_bd"] >= q5["chi2_band_lo"]) & (pert["T5_V1_bd"] <= q5["chi2_band_hi"])).sum()
    rec("kmeans", "P6_in_chi2_band_count", f"{int(inb)}/{len(pert)}",
        f"微扰下 b_d 落在 χ² 区间 [{q5['chi2_band_lo']:.3f}, {q5['chi2_band_hi']:.3f}] 内的次数")
    flush()


# ============================================================================== cond
def _win(a: pd.Series, b: pd.Series | None = None, n=17) -> pd.Series:
    """连续序列去开头 SKIP 天后的全部 n 日窗口复利差(与 window_excess_distribution 同法;b=None 时为 Π(1+a)−1)。"""
    df = pd.DataFrame({"strategy": a, "benchmark": (b if b is not None else a * 0.0)})
    return M5.window_excess_distribution(df, n, skip_head=SKIP)


def stage_cond():
    import statsmodels.api as sm
    d = pd.read_parquet(LEDGER / "freeze_ref_daily.parquet")
    R = pd.read_csv(LEDGER / "fwd_results.csv")
    Bf = pd.read_parquet(TMP / "fwd_bench.parquet").iloc[SKIP:]
    f_pool = float(np.prod(1 + Bf["pool5"]) - 1)
    f_pmc = float(np.prod(1 + Bf["pool5"]) - np.prod(1 + Bf["csi1000_exec"]))
    g = lambda o, c, b: float(R[(R["section"] == "reading") & (R["object"] == o) & (R["cost"] == c) & (R["bench"] == b)]["window_excess"].iloc[0])
    specs = [("P1 T1 旧成本", "T1_excl|old|pool5|n17", g("T1_excl", "old", "pool5"), "C|bench_pool5", "C|bench_proxy_exec"),
             ("P2 T1 Y=1", "T1_excl|Y1|pool5|n17", g("T1_excl", "Y1", "pool5"), "C|bench_pool5", "C|bench_proxy_exec"),
             ("P3 T2 旧成本", "T2_top20|old|pool5|n17", g("T2_top20", "old", "pool5"), "C|bench_pool5", "C|bench_proxy_exec"),
             ("P4 T3 对中证1000", "T3_V1|Y1|proxy_exec|n17", g("T3_V1", "Y1", "csi1000_actual_exec"),
              "T3_V1|A|bench_pool5", "T3_V1|A|bench_proxy_exec")]
    for lab, key, fv, cpool, cprox in specs:
        x = DIST[DIST["key"] == key].set_index("window_end")["value"]
        pool_w = _win(d[cpool]); pmc = _win(d[cpool], d[cprox])
        df = pd.DataFrame({"x": x, "pool": pool_w.reindex(x.index), "pmc": pmc.reindex(x.index)}).dropna().astype(float)
        mdl = sm.OLS(df["x"], sm.add_constant(df[["pool", "pmc"]])).fit()
        pred = float(mdl.params @ np.array([1.0, f_pool, f_pmc]))
        resid_pct = 100.0 * float((mdl.resid < fv - pred).mean())
        sub = df[(df["pool"] <= -0.02) & (df["pmc"] >= 0.02)]
        rec("cond", lab, f"预测 {pred*100:+.2f}%,实际 {fv*100:+.2f}%,残差百分位 {resid_pct:.1f};"
                         f"相似状态子集 n={len(sub)} 中位 {sub['x'].median()*100:+.2f}%、前向值在子集内百分位 {M5.percentile_in(fv, sub['x']):.1f}",
            f"OLS x = a + b·池窗口收益 + c·(池−中证1000 代理);a={mdl.params['const']:+.4f} b={mdl.params['pool']:+.3f} "
            f"c={mdl.params['pmc']:+.3f} R²={mdl.rsquared:.2f} n={len(df)}(重叠窗口,R² 与系数只作描述);子集 = 池 ≤ −2% 且 池−中证 ≥ +2%,"
            f"{sub.index.to_period('M').nunique()} 个不同月份")
    pool_hist = _win(d["C|bench_pool5"]); pmc_hist = _win(d["C|bench_pool5"], d["C|bench_proxy_exec"])
    rec("cond", "window_state", f"同频池 {f_pool*100:+.2f}%(历史百分位 {M5.percentile_in(f_pool, pool_hist):.1f});"
                                f"池 − 中证1000 {f_pmc*100:+.2f}%(历史百分位 {M5.percentile_in(f_pmc, pmc_hist):.1f})",
        "前向评估窗的市场状态(C 口径历史 17 日窗口)")
    flush()


# ============================================================================== st
RULE_DATE = pd.Timestamp("2026-07-06")


def _st_fixed_panel(returns_view: bool):
    """只换涨跌幅规则的前向面板(fwd 适配,冻结的 panel._limit_frame 不改):RULE_DATE 起主板 ST 由 5% 改回 10%。
    limit_up / limit_down / one_word_* 用冻结函数同式((ret_c2c ≥ lim × TOL)、(rng < 阈值))在新涨跌幅上重算。"""
    from afactor import panel as PN
    p = ForwardPanel(returns_view=returns_view)
    lim = PN._limit_frame(p)
    base = PN._limit_arr(p).reindex(p.codes)
    main = pd.Series(np.isclose(base.to_numpy(np.float64), 0.10), index=p.codes)
    late = pd.Series(p.dates >= RULE_DATE, index=p.dates)
    fix = (lim == 0.05) & p["is_st"].reindex_like(lim).fillna(False)
    fix = fix & np.outer(late.to_numpy(), main.to_numpy())
    lim2 = lim.mask(fix, 0.10)
    rc = PN._ret_c2c(p)
    lu = rc >= lim2 * PN._LIMIT_TOL
    ld = rc <= -lim2 * PN._LIMIT_TOL
    rng = p.raw("rng") < PN._ONE_WORD_RNG
    p._derived.update({"limit_up": lu, "limit_down": ld, "one_word_up": lu & rng, "one_word_down": ld & rng})
    for k in ("twap_open10_f",):
        p._derived.pop(k, None)
    return p, int(fix.to_numpy().sum())


def stage_st():
    from afactor import panel as PN, strategy as ST, optimizer as OPT, fwd_weights as FW, index_proxy as IP
    import fwd_run_test as RT
    p0 = ForwardPanel()
    st = p0["is_st"]
    rc = PN._ret_c2c(p0)
    lim = PN._limit_frame(p0)
    m5 = st & (lim == 0.05)
    big = ((rc.abs() > 0.052) & m5)
    byM = big.sum(axis=1).groupby(p0.dates.to_period("M")).sum()
    nM = m5.sum(axis=1).groupby(p0.dates.to_period("M")).sum()
    rec("st", "ST_mainboard_abs_c2c_gt_5.2pct_by_month",
        "; ".join(f"{k}:{int(byM[k])}/{int(nM[k])}" for k in byM.index if str(k) >= "2025-06"),
        "主板 ST 股票日中 |收盘对收盘| > 5.2% 的个数 / 主板 ST 股票日数(5% 规则下只应是 tick 舍入)")
    dmax = rc.where(m5).abs().max(axis=1)
    cnt = ((rc.abs() > 0.055) & m5).sum(axis=1)
    jul = dmax.loc["2026-06-24":"2026-07-14"]
    rec("st", "ST_mainboard_daily_max_abs_c2c_0624_0714",
        "; ".join(f"{d.strftime('%m-%d')}:{v:.3f}({int(cnt[d])})" for d, v in jul.items()),
        "主板 ST 当日最大 |收盘对收盘|(括号内为 >5.5% 的只数);07-06 起稳定在 0.10 附近 -> 规则日取 07-06")
    first = cnt[(cnt.index >= "2026-06-01") & (cnt >= 3)].index
    rec("st", "rule_change_first_day_ge3_over_5.5pct", str(first[0].date()) if len(first) else "无", "")
    # 前向成交 / 实现窗口内的误判(te ∈ [09-02, 10-09])
    win = (p0.dates >= "2026-09-02") & (p0.dates <= "2026-10-09")
    dW = p0.dates[win]
    lu, ld = p0["limit_up"], p0["limit_down"]
    ow = p0["one_word_up"] | p0["one_word_down"]
    sts = m5.loc[dW]
    touch = (lu.loc[dW] | ld.loc[dW]) & sts
    false_touch = touch & (rc.loc[dW].abs() < 0.10 * 0.985)
    false_dead = ow.loc[dW] & sts & (rc.loc[dW].abs() < 0.10 * 0.985)
    fd_list = [(d, c) for d in dW for c in false_dead.columns[false_dead.loc[d].to_numpy()]]
    rec("st", "te_window_ST_touch_flags", int(touch.to_numpy().sum()),
        f"其中未到 10% 的 98.5%(按 10% 规则不算触板)的 {int(false_touch.to_numpy().sum())} 个")
    rec("st", "te_window_false_one_word", len(fd_list), "; ".join(f"{c}@{d.date()}" for d, c in fd_list))
    held = []
    for tag in ("T1_excl_old", "T1_excl_Y1", "T2_top20_old", "T2_top20_Y1"):
        H = pd.read_parquet(TMP / f"fwd_{tag}_holdings.parquet")
        n = 0
        for d, c in fd_list:
            i = p0.dates.get_loc(d)
            for t in (p0.dates[i - 1], p0.dates[i - 2]):
                if t in H.index and c in H.columns and H.at[t, c] > 1e-8:
                    n += 1
                    break
        held.append(f"{tag}:{n}")
    t3h = 0
    for ph in range(5):
        H = pd.read_parquet(TMP / f"fwd_T3_V1_ph{ph}_holdings.parquet")
        for d, c in fd_list:
            i = p0.dates.get_loc(d)
            for t in (p0.dates[i - 1], p0.dates[i - 2]):
                if t in H.index and c in H.columns and H.at[t, c] > 1e-8:
                    t3h += 1
                    break
    rec("st", "false_one_word_held", "; ".join(held) + f"; T3 相位合计:{t3h}", "误判一字板的股票日前后被持有的次数")
    U = ST.universe(p0, "V1000")
    sig = slice("2026-09-01", "2026-09-30")
    lu_pool = lu.loc[sig] & U.loc[sig]
    lu_false = lu_pool & m5.loc[sig] & (rc.loc[sig] < 0.10 * 0.985)
    rec("st", "Sep_signal_days_pool_limit_up", int(lu_pool.to_numpy().sum()),
        f"其中主板 ST 5%~10% 的 {int(lu_false.to_numpy().sum())} 个(按 10% 规则不是涨停)")
    # 敏感性:只换涨跌幅规则重跑(引擎 / 基准 / T3 只用一字板标记)
    pf2, nfix = _st_fixed_panel(False)
    p2, _ = _st_fixed_panel(True)
    rec("st", "fixed_panel_cells_relimited", nfix, f"{RULE_DATE.date()} 起主板 ST 股票日改按 10%")
    s = pd.read_parquet(TMP / "fwd_score5.parquet")
    m = p2.mask_from(SIG_START)
    B0 = pd.read_parquet(TMP / "fwd_bench.parquet")
    S0 = pd.read_parquet(TMP / "fwd_strategy_rets.parquet")
    b5 = ST.benchmark(p2, "V1000", "open10", date_mask=m, rebalance=5)["rets"]
    W = FW.csi1000_close_weights(p2)
    rb = IP.exec_time_bench_returns(p2, "open10", W).reindex(b5.index)
    rec("st", "bench_pool5_maxabsdiff", float((b5 - B0["pool5"]).abs().max()), "同频池基准:10% 规则 vs 账本")
    rec("st", "bench_csi1000_exec_maxabsdiff", float((rb - B0["csi1000_exec"]).abs().max()), "")
    cm = RT.cost_model(ForwardPanel())
    for plan, kw in RT.PLANS.items():
        for cost, cmx in (("old", None), ("Y1", cm)):
            r = ST.run_strategy(s, p2, "V1000", "open10", date_mask=m, cost_model=cmx, **kw)["rets"]
            v0 = wex(S0[f"{plan}|{cost}"], B0["pool5"]); v1 = wex(r, b5)
            rec("st", f"{plan}|{cost}|window_excess_10pct_rule", float(v1),
                f"账本(5% 规则){v0*100:+.4f}% -> 10% 规则 {v1*100:+.4f}%;逐日收益最大差 {float((r - S0[f'{plan}|{cost}']).abs().max()):.2e}")
    rm = RT.load_rm()
    U2 = ST.universe(p2, "V1000")
    icd = pd.Series(RT.IC_FROZEN, index=p2.dates)
    R3 = []
    for ph in range(5):
        enh = OPT.IndexEnhancer(p2, rm, W, s, icd, U2, OPT.OptConfig(), tag="fwd_review_st")
        R3.append(ST.simulate(enh, p2, "open10", RT.CAPITAL, rebalance=5, date_mask=m, phase=ph, cost_model=cm)["rets"])
    r3 = pd.concat(R3, axis=1).mean(axis=1)
    led = pd.read_csv(LEDGER / "fwd_results.csv")
    v0 = float(led[(led["section"] == "reading") & (led["object"] == "T3_V1") & (led["bench"] == "csi1000_actual_exec")]["window_excess"].iloc[0])
    rec("st", "T3_V1|window_excess_10pct_rule", float(wex(r3, rb)), f"账本 {v0*100:+.4f}%")
    flush()


# ============================================================================== p6gap
def stage_p6gap():
    """审查者反事实(P4 2.60 / P6 1.179)与本修复(2.654 / 1.283)差异的来源。按审查者探针的写法重建其修正倍数
    (只用 tushare adj_factor、|m−1| ≤ 1e-4 置 1、close 缺失的格 m = 1),只换风险模型重跑 T3 / T5(打分用本修复的)。"""
    from afactor import strategy as ST, optimizer as OPT, fwd_adapters as FA_, fwd_weights as FW
    from afactor.barra import RiskModel
    import fwd_run_test as RT
    import fwd_build_panel as FB
    BK = PREFIX / "panel"
    pf, p = RT.panels()
    c0 = pd.read_parquet(BK / "close_px.parquet").astype(np.float64)
    rc = pf.raw("raw_close").astype(np.float64)
    AF = FB.load_ts_adj_factor(pf.dates, pf.codes).reindex(pf.dates)
    rows = np.asarray(pf.dates <= FB.M1_END)
    mult = (AF / AF.loc[FB.M1_END]) / (c0 / rc)
    mult = mult.where(np.isfinite(mult) & ((mult - 1).abs() > 1e-4), 1.0)
    mult.loc[~rows] = 1.0
    mult = mult.where(rc.notna() | ~c0.notna(), np.nan).ffill().fillna(1.0)
    mult.loc[~rows] = 1.0
    M = mult.astype(np.float32)
    tw0 = pd.read_parquet(BK / "twap_open10.parquet")
    gap = (tw0.notna() & c0.isna()).loc[:FB.M1_END]
    byd = gap.sum(axis=1)
    rec("p6gap", "cells_close_nan_twap_finite_le_0817", int(gap.to_numpy().sum()),
        "收盘缺失(1 分钟午后缺 bar)而开盘 TWAP 有值的股票日;最多的日期 "
        + "、".join(f"{d.date()}:{int(v)}" for d, v in byd.sort_values(ascending=False).head(3).items()))

    class _RV(ForwardPanel):
        def raw(self, name):
            if name not in self._cache:
                if name in FB.PRICE_LEVEL_FIELDS:
                    df = pd.read_parquet(BK / f"{name}.parquet").astype(np.float32)
                    self._cache[name] = (df * M.reindex(index=df.index, columns=df.columns)).astype(np.float32)
                else:
                    super().raw(name)
            return self._cache[name]
    prv = _RV()
    for nm, pp in (("本修复", pf), ("审查者写法", prv)):
        tw = pp.raw("twap_open10").astype(np.float64).loc[:FB.M1_END]
        lr = np.log(tw / tw.shift(1)).abs().to_numpy()
        rec("p6gap", f"twap_abs_logret_gt_0.2[{nm}]", int((lr[np.isfinite(lr)] > 0.2).sum()),
            "<= 08-17 开盘 TWAP 日对数收益 |·| > 0.2 的股票日(风险模型 open10 口径的收益输入)")
    with forward_context("前向检验审查:审查者反事实写法下的风险模型(只作差异定位)"):
        with FA_.lifted_guards():
            rm_rv = RiskModel(cache_dir=TMP / "fwd_review_rvrisk").build(prv, "open10", force=True, verbose=False)
    rm = RT.load_rm()
    i = p.dates.get_loc(pd.Timestamp("2026-09-15"))
    s0, s1 = np.asarray(rm.spec[i]), np.asarray(rm_rv.spec[i])
    ok = np.isfinite(s0) & np.isfinite(s1) & (s0 > 0)
    rec("p6gap", "spec_var_ratio_reviewer_over_fix_0915", f"中位 {np.median(s1[ok] / s0[ok]):.4f}(p05 {np.quantile(s1[ok] / s0[ok], .05):.4f},"
        f"p95 {np.quantile(s1[ok] / s0[ok], .95):.4f})", "09-15 特质方差之比;暴露度两者逐列相关 1.0000")
    s = pd.read_parquet(TMP / "fwd_score5.parquet")
    B = pd.read_parquet(TMP / "fwd_bench.parquet")
    W = FW.csi1000_close_weights(p)
    Wx = np.load(TMP / "fwd_csi1000_Wx.npy").astype(np.float64)
    U = ST.universe(p, "V1000")
    cm = RT.cost_model(pf)
    for nm, rm_ in (("本修复风险模型", rm), ("审查者写法风险模型", rm_rv)):
        r3, bd = _t3_t5(p, pf, s, rm_, cm, W, Wx, U, B)
        v3 = wex(r3, B["csi1000_exec"])
        rec("p6gap", f"T3_T5[{nm}]", f"P4 {v3*100:+.3f}%({M5.percentile_in(v3, dist('T3_V1|Y1|proxy_exec|n17')):.1f});"
            f"b_d {bd:.4f}({M5.percentile_in(bd, dist('T5_V1|Y1|proxy_exec|n17|b_d')):.1f})", "打分相同(本修复),只换风险模型")
    flush()


# ============================================================================== may1314
def stage_may1314():
    """修复中新发现(非审查条目):2026-05-13 / 05-14 两天 1 分钟数据在 14:19 之后几乎全市场缺失(最后 41 根 bar),
    面板 close_px / ret_intraday 两天全市场为 NaN,其余分钟原料是 200 根 bar 的半截日。只影响长窗口(120 日统计行业标签、
    250 日成员、252 日 Barra)。敏感性:两天的收盘用 daily_basic 未复权收盘 × 同一目标复权系数补上、日内收益 = 收盘 / 开盘 − 1,
    重算标签 / 打分 / 风险模型 / T1–T5。只作描述,不替换主读数。"""
    from afactor import strategy as ST, fwd_adapters as FA_, fwd_weights as FW
    from afactor.barra import RiskModel
    import fwd_run_test as RT
    import fwd_build_panel as FB
    pf0, p0 = RT.panels()
    days = [pd.Timestamp("2026-05-13"), pd.Timestamp("2026-05-14")]
    c = pf0.raw("close_px")
    op = pf0.raw("open_auction_px")
    tw = pf0.raw("twap_open10")
    dbc = FB._aux("db_raw_close").reindex(index=days, columns=pf0.codes).astype(np.float64)
    tgt = FB.rebase_target(pf0.dates, pf0.codes).reindex(index=days)
    fill = (c.loc[days].isna() & tw.loc[days].notna()).to_numpy() & np.isfinite((dbc * tgt).to_numpy())
    rec("may1314", "cells_filled", int(fill.sum()), "两天收盘缺失、开盘 TWAP 有值、daily_basic 有收盘的股票日")
    c2 = c.copy()
    c2.loc[days] = np.where(fill, (dbc * tgt).to_numpy(), c.loc[days].to_numpy()).astype(np.float32)
    ri2 = pf0.raw("ret_intraday").copy()
    ri2.loc[days] = np.where(fill, (c2.loc[days] / op.loc[days] - 1.0).to_numpy(), ri2.loc[days].to_numpy()).astype(np.float32)
    L = FA_.statind_labels_for_quarters(c2, ["2025Q4", "2026Q1", "2026Q2", "2026Q3"])
    L.loc[L.index.year == 2025] = pf0._derived["statind_L"].loc[L.index.year == 2025].to_numpy()
    L = L.astype(np.int16)
    from sklearn.metrics import adjusted_rand_score
    a_, b_ = pf0._derived["statind_L"].loc["2026-09-01"], L.loc["2026-09-01"]
    ok = (a_ >= 0) & (b_ >= 0)
    rec("may1314", "ARI_Sep_labels", float(adjusted_rand_score(a_[ok], b_[ok])), "9 月所用标签:补齐后 vs 账本")
    raw_c = pf0.raw("raw_close").copy()
    raw_c.loc[days] = np.where(fill, dbc.to_numpy(), raw_c.loc[days].to_numpy()).astype(np.float32)

    def mk(rv):
        class _P(ForwardPanel):
            def __init__(self):
                super().__init__(returns_view=rv)
                self._derived["statind_L"] = L
                self._cache.update({"close_px": c2, "ret_intraday": ri2, "raw_close": raw_c})
        return _P()
    pf, p = mk(False), mk(True)
    with forward_context("前向检验审查:05-13/14 收盘补齐敏感性(只算预注册评估窗读数,不进判定)"):
        s = FC.comp_score(pf, cache_file=None)
        with FA_.lifted_guards():
            rm = RiskModel(cache_dir=TMP / "fwd_review_may1314_risk").build(pf, "open10", force=True, verbose=False)
    from afactor.impact import CostModel
    with FA_.lifted_guards(raw_close=pf.raw("raw_close")):
        cm = CostModel(pf, impact_y=1.0)
    B = pd.read_parquet(TMP / "fwd_bench.parquet")
    m = p.mask_from(SIG_START)
    b5 = ST.benchmark(p, "V1000", "open10", date_mask=m, rebalance=5)["rets"]
    rec("may1314", "bench_pool5_maxabsdiff", float((b5 - B["pool5"]).abs().max()), "同频池基准只用 9 月起的执行价,应为 0")
    R = pd.read_csv(LEDGER / "fwd_results.csv")
    led = lambda o, c_, b: float(R[(R["section"] == "reading") & (R["object"] == o) & (R["cost"] == c_) & (R["bench"] == b)]["window_excess"].iloc[0])
    for plan, kw in RT.PLANS.items():
        for cost, cmx in (("old", None), ("Y1", cm)):
            r = ST.run_strategy(s, p, "V1000", "open10", date_mask=m, cost_model=cmx, **kw)["rets"]
            v = wex(r, B["pool5"])
            rec("may1314", f"{plan}|{cost}", f"{v*100:+.3f}%({M5.percentile_in(v, dist(f'{plan}|{cost}|pool5|n17')):.1f})",
                f"账本 {led(plan, cost, 'pool5')*100:+.3f}%")
    W = FW.csi1000_close_weights(p)
    Wx = np.load(TMP / "fwd_csi1000_Wx.npy").astype(np.float64)
    r3, bd = _t3_t5(p, pf, s, rm, cm, W, Wx, ST.universe(p, "V1000"), B)
    v3 = wex(r3, B["csi1000_exec"])
    rec("may1314", "T3_V1", f"{v3*100:+.3f}%({M5.percentile_in(v3, dist('T3_V1|Y1|proxy_exec|n17')):.1f})",
        f"账本 {led('T3_V1', 'Y1', 'csi1000_actual_exec')*100:+.3f}%")
    rec("may1314", "T5_V1_bd", f"{bd:.4f}({M5.percentile_in(bd, dist('T5_V1|Y1|proxy_exec|n17|b_d')):.1f})",
        f"账本 {led('T5_V1', 'Y1', 'csi1000_actual_exec'):.4f}")
    flush()


# ============================================================================== stmt
def stage_stmt():
    """财报回退(审查 P2):原规则下财报沿用最后已知报告期(tushare 利润表到 08-20、资产负债表到 06-13)。
    (a) 陈旧程度:9 月首末信号日 V1000 池内,半年报已在面板财报里(tushare,PIT 可见)vs 东财快照显示已披露(公告日 <= 当日)的比例;
    (b) 上包络式敏感性:COMP_v5 去掉 3 个基本面成员(fx_fu_*)重算打分 -> T1 / T2 / T3。陈旧财报与及时财报的差别
        小于"整个成员缺席",所以这只给出影响的量级上限的参考,不是"用及时财报"的读数。只作描述,不进判定。"""
    from afactor import strategy as ST, fundamental as FU, fwd_weights as FW
    import fwd_run_test as RT
    import m5_bridge as MB
    pf, p = RT.panels()
    U = ST.universe(p, "V1000")
    with forward_context("前向检验审查:财报陈旧程度(只读公告,不算收益)"):
        rep = FU.build_report_level()
    rep = rep[rep["pit_vintage"] == FU.PIT_FINAL] if "pit_vintage" in rep else rep
    h1 = rep[rep["end_date"] == pd.Timestamp("2026-06-30")]
    avail_col = "avail" if "avail" in h1.columns else "ann_date"
    import glob
    em = pd.read_csv(sorted(glob.glob(str(CACHE / "forward" / "raw" / "ak_em" / "raw_lrb_20260630_fetched*.csv")))[-1],
                     usecols=["SECUCODE", "NOTICE_DATE"])
    em["NOTICE_DATE"] = pd.to_datetime(em["NOTICE_DATE"]).dt.normalize()
    for d in (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-30")):
        pool = set(U.columns[U.loc[d].to_numpy(bool)])
        ts_ok = set(h1.loc[pd.to_datetime(h1[avail_col]) <= d, "stock_code"]) & pool
        em_ok = set(em.loc[em["NOTICE_DATE"] <= d, "SECUCODE"]) & pool
        rec("stmt", f"H1_2026_visible_share[{d.date()}]", f"面板财报(tushare){len(ts_ok)/len(pool):.1%} / 东财已披露 {len(em_ok)/len(pool):.1%}",
            f"V1000 池 {len(pool)} 只;前者是 3 个基本面成员与 Barra 价值 / 盈利 / 成长 / 杠杆在当日实际用到半年报的比例")
    members, signs, _ = MB.load_members()
    fu = [m for m in members if m.startswith("fx_fu_")]
    keep = [m for m in members if m not in fu]
    with forward_context("前向检验审查:去掉基本面成员的打分(上包络敏感性,不进判定)"):
        s2 = MB._comp_score(pf, keep, {k: signs[k] for k in keep}).reindex(index=p.dates, columns=p.codes).astype(np.float32)
    s0 = pd.read_parquet(TMP / "fwd_score5.parquet")
    sig = p.signal_days()
    rk = np.nanmean([s0.loc[d].rank().corr(s2.loc[d].rank()) for d in sig])
    rec("stmt", "score_rankcorr_without_fu_members", float(rk), f"去掉 {fu} 后的打分 vs 账本打分,9 月信号日截面秩相关均值")
    B = pd.read_parquet(TMP / "fwd_bench.parquet")
    m = p.mask_from(SIG_START)
    cm = RT.cost_model(pf)
    R = pd.read_csv(LEDGER / "fwd_results.csv")
    led = lambda o, c_, b: float(R[(R["section"] == "reading") & (R["object"] == o) & (R["cost"] == c_) & (R["bench"] == b)]["window_excess"].iloc[0])
    for plan, kw in RT.PLANS.items():
        for cost, cmx in (("old", None), ("Y1", cm)):
            r = ST.run_strategy(s2, p, "V1000", "open10", date_mask=m, cost_model=cmx, **kw)["rets"]
            v = wex(r, B["pool5"])
            rec("stmt", f"no_fu|{plan}|{cost}", f"{v*100:+.3f}%({M5.percentile_in(v, dist(f'{plan}|{cost}|pool5|n17')):.1f})",
                f"账本 {led(plan, cost, 'pool5')*100:+.3f}%")
    W = FW.csi1000_close_weights(p)
    Wx = np.load(TMP / "fwd_csi1000_Wx.npy").astype(np.float64)
    r3, bd = _t3_t5(p, pf, s2, RT.load_rm(), cm, W, Wx, U, B)
    v3 = wex(r3, B["csi1000_exec"])
    rec("stmt", "no_fu|T3_V1", f"{v3*100:+.3f}%({M5.percentile_in(v3, dist('T3_V1|Y1|proxy_exec|n17')):.1f});b_d {bd:.4f}",
        f"账本 {led('T3_V1', 'Y1', 'csi1000_actual_exec')*100:+.3f}%")
    flush()


# ============================================================================== stmt_em
def _em_h1_income(cols: list[str], items: list[str]) -> pd.DataFrame:
    """东财 2026 半年报利润表快照 -> tushare 利润表同列格式(fundamental._load 的返回列)。
    映射:n_income_attr_p <- PARENT_NETPROFIT、operate_profit <- OPERATE_PROFIT、total_profit <- TOTAL_PROFIT、
    total_revenue <- TOTAL_OPERATE_INCOME、revenue <- OPERATE_INCOME,缺失时 TOTAL_OPERATE_INCOME(与 fwd_decisions 同一映射;
    一般企业两者相同,金融类不同 —— 只作敏感性)、oper_cost <- OPERATE_COST、sell_exp <- SALE_EXPENSE、admin_exp <- MANAGE_EXPENSE;
    n_income / rd_exp 东财快照没有,置 NaN。ann_date = f_ann_date = NOTICE_DATE。"""
    import glob
    f = sorted(glob.glob(str(CACHE / "forward" / "raw" / "ak_em" / "raw_lrb_20260630_fetched*.csv")))[-1]
    d = pd.read_csv(f, dtype={"SECURITY_CODE": str})
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    out = pd.DataFrame({"stock_code": d["SECUCODE"].astype(str),
                        "ann_date": pd.to_datetime(d["NOTICE_DATE"]).dt.normalize(),
                        "end_date": pd.to_datetime(d["REPORT_DATE"]).dt.normalize()})
    out["f_ann_date"] = out["ann_date"]
    m = {"n_income_attr_p": num("PARENT_NETPROFIT"), "operate_profit": num("OPERATE_PROFIT"),
         "total_profit": num("TOTAL_PROFIT"), "total_revenue": num("TOTAL_OPERATE_INCOME"),
         "revenue": num("OPERATE_INCOME").where(num("OPERATE_INCOME").notna(), num("TOTAL_OPERATE_INCOME")),
         "oper_cost": num("OPERATE_COST"), "sell_exp": num("SALE_EXPENSE"), "admin_exp": num("MANAGE_EXPENSE")}
    for it in items:
        out[it] = m.get(it, np.nan)
    out = out[out["stock_code"].str.endswith((".SH", ".SZ"))].drop_duplicates(["stock_code", "end_date"], keep="last")
    out["avail"] = out["ann_date"]
    return out


def stage_stmt_em():
    """财报敏感性(审查 P2 的可选项):只补利润表 —— 东财 2026 半年报补进 tushare 没有的 (股票, 报告期),
    其余(资产负债表、现金流量表、tushare 已有的行)不变;重算 COMP_v5 打分与风险模型(Barra 盈利 / 成长随之更新),
    重跑 T1 / T2 / T3 / T5。偏离预注册 §4.3 原规则,只作描述,不进判定、不替换主读数。"""
    from afactor import strategy as ST, fundamental as FU, fwd_adapters as FA_, fwd_weights as FW
    from afactor.barra import RiskModel
    import fwd_run_test as RT
    saved = FU._load
    n_add = {}

    def _load_em(name, items, cutoff=None):
        d = saved(name, items, cutoff)
        if name != FU.REPORT_TABLES[0]:
            return d
        add = _em_h1_income(list(d.columns), items)
        have = set(zip(d["stock_code"], d["end_date"]))
        add = add[[(a, b) not in have for a, b in zip(add["stock_code"], add["end_date"])]]
        cut = FU.data_cutoff() if cutoff is None else pd.Timestamp(cutoff)
        add = add[add["avail"] <= cut].copy()
        ct = d.sort_values("end_date").drop_duplicates("stock_code", keep="last").set_index("stock_code")["comp_type"]
        add["comp_type"] = add["stock_code"].map(ct)
        n_add["rows"] = len(add)
        return pd.concat([d, add[d.columns]], ignore_index=True).sort_values(["stock_code", "end_date", "avail"])
    FU._REP_CACHE.clear()
    FU._load = _load_em
    try:
        pf, p = ForwardPanel(), ForwardPanel(returns_view=True)
        with forward_context("前向检验审查:东财半年报补利润表的财报敏感性(偏离原规则,只作描述)"):
            s = FC.comp_score(pf, cache_file=None)
            with FA_.lifted_guards():
                rm = RiskModel(cache_dir=TMP / "fwd_review_stmt_em_risk").build(pf, "open10", force=True, verbose=False)
    finally:
        FU._load = saved
        FU._REP_CACHE.clear()
    rec("stmt_em", "em_h1_rows_spliced", int(n_add.get("rows", 0)), "补入的(股票, 2026-06-30)利润表行(tushare 已有的不动)")
    s0 = pd.read_parquet(TMP / "fwd_score5.parquet")
    sig = p.signal_days()
    rec("stmt_em", "score_rankcorr_vs_ledger", float(np.nanmean([s0.loc[d].rank().corr(s.loc[d].rank()) for d in sig])),
        "9 月信号日截面秩相关均值")
    B = pd.read_parquet(TMP / "fwd_bench.parquet")
    m = p.mask_from(SIG_START)
    cm = RT.cost_model(ForwardPanel())
    R = pd.read_csv(LEDGER / "fwd_results.csv")
    led = lambda o, c_, b: float(R[(R["section"] == "reading") & (R["object"] == o) & (R["cost"] == c_) & (R["bench"] == b)]["window_excess"].iloc[0])
    for plan, kw in RT.PLANS.items():
        for cost, cmx in (("old", None), ("Y1", cm)):
            r = ST.run_strategy(s, p, "V1000", "open10", date_mask=m, cost_model=cmx, **kw)["rets"]
            v = wex(r, B["pool5"])
            rec("stmt_em", f"{plan}|{cost}", f"{v*100:+.3f}%({M5.percentile_in(v, dist(f'{plan}|{cost}|pool5|n17')):.1f})",
                f"账本 {led(plan, cost, 'pool5')*100:+.3f}%")
    W = FW.csi1000_close_weights(p)
    Wx = np.load(TMP / "fwd_csi1000_Wx.npy").astype(np.float64)
    r3, bd = _t3_t5(p, pf, s, rm, cm, W, Wx, ST.universe(p, "V1000"), B)
    v3 = wex(r3, B["csi1000_exec"])
    rec("stmt_em", "T3_V1", f"{v3*100:+.3f}%({M5.percentile_in(v3, dist('T3_V1|Y1|proxy_exec|n17')):.1f})",
        f"账本 {led('T3_V1', 'Y1', 'csi1000_actual_exec')*100:+.3f}%")
    rec("stmt_em", "T5_V1_bd", f"{bd:.4f}({M5.percentile_in(bd, dist('T5_V1|Y1|proxy_exec|n17|b_d')):.1f})",
        f"账本 {led('T5_V1', 'Y1', 'csi1000_actual_exec'):.4f}")
    flush()


if __name__ == "__main__":
    for st_ in sys.argv[1:]:
        t0 = time.time()
        log(f"===== {st_} =====")
        globals()[f"stage_{st_}"]()
        log(f"===== {st_} 完成 {time.time() - t0:.0f}s =====")
