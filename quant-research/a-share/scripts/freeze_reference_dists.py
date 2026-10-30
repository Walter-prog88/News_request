"""前向检验的历史参考分布:在任何人看到 2026-08-18 以后的收益之前一次算好并冻结(前缀 freeze,2026-10-10)。

## 为什么要在冻结时就算好
前向检验(research/FORWARD_TEST_PREREG.md)要把前向窗口的超额放进"2021-2025 同长度滚动窗口历史分布"里查百分位。
桥接报告(MINUTE5_BRIDGE.md §5.5)只预先算了**旧成本**口径的排雷 / Top20 分布(m5_bridge_daily.parquet)。
预注册里另有 Y=1 冲击成本口径(T1/T2)、指增 V1(T3)和风险模型偏差统计量(T5)。如果等前向代理看过前向结果
再去建这些分布,"怎么建分布"本身就成了一个可以事后挑的自由度。所以这里只用研究期(<= 2025-12-31)数据,
按预注册写定的口径把它们算好、落盘,并把产物写进冻结清单(research/ledger/forward_freeze_manifest.json)。
前向检验只能查这些文件,不得重算或改口径。

## 口径(全部与上游已审查的代码路径相同,只换一个参数)
- T1/T2 Y=1(阶段 y1):与 scripts/m5_bridge.py backtest 阶段的 C 口径(5 分钟信号 + 5 分钟执行价)同一调用
  `ST.run_strategy(s5, p5, "V1000", "open10", date_mask=2021-2025, **PLANS[plan])`,只把 cost_model 换成
  `CostModel(p5, impact_y=1.0)`(一跳价差 + 平方根冲击,ENGINE_IMPACT_MODEL.md)。基准 = 同频池等权
  `ST.benchmark(p5, "V1000", "open10", date_mask, rebalance=5)`。**先复现**旧成本 C 口径与 m5_bridge_daily.parquet
  逐位一致(证明管线相同),不一致直接报错退出。
  另给次要参考:C 口径策略对中证 1000 PIT 代理(成交时点权重 × 5 分钟成交价,`IP.exec_time_bench_returns`)。
- T3(阶段 ie):V1 指增 5 相位均值的逐日收益(cache/_tmp/ie_V1_ph{0..4}_daily.parquet,1 分钟 A 口径,
  2021-04-20 冷启动;合并方式 = scripts/run_index_enhance.py::load_variant),对成交时点代理(主)/ 同频池等权 /
  真实 000852(收盘对收盘对齐到信号日,只作参考)。**这是 1 分钟口径**:指增在 5 分钟口径下的近似误差没有量化过,
  前向读数要带着这一条读(预注册 §6)。
- T5(阶段 te):V1 与全池等权判据组合(CHK_pool_ew:常数打分全池等权、5 日调仓、Y=1)的逐日事前 TE,
  算法逐行照搬 run_index_enhance.report() 的 te_pred(实际持仓 − 成交时点代理权重,风险模型 5 日协方差 + 特质方差);
  z_t = 实现日主动收益 / (te_pred_t / √242)。前向用"n 日内 z 的标准差"b_d 作偏差统计量。
- 分布(阶段 dist):`minute5.window_excess_distribution(df, n, how="compound", skip_head=4)`,
  n ∈ {16, 17, 18}(预计前向评估天数 17,日历另有两种可能,见预注册 §3)。b_d 用同样的 skip_head 与滚动窗口。
- T4(阶段 pead):只读 PEAD 研究已落盘的研究期产物(事件层槽位 e5、账户逐日收益),不重算;
  主读数 = 窗口内"大批次"(>= 50 个有效 E 槽位)D_T 的等权均值 D̄,按窗口内大批次个数 k 存分布(见 stage_pead)。

## 只读研究期
Panel() / m5_panel() 只含 2020-2025;不 import afactor.fwd_data,不读 cache/forward、cache/sealed,不放宽数据截止日。

用法:
    python3 scripts/freeze_reference_dists.py y1 ie te dist pead chk   # 全部;缓存齐全时约 1 分钟
产出:research/ledger/freeze_ref_daily.parquet、freeze_ref_te_daily.parquet、freeze_ref_window_dist.parquet、
      freeze_ref_quantiles.csv、freeze_ref_checks.csv;日志 logs/freeze_ref.log。
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

from afactor.config import LEDGER, TMP, RESEARCH_END

TD = 242
SKIP = 4                                   # = minute5.FWD_SKIP_HEAD,冷启动前 4 个信号日只建仓、不计
NS = (16, 17, 18)                          # 前向评估天数的三种日历可能(预注册 §3)
QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
OUT_DAILY = LEDGER / "freeze_ref_daily.parquet"
OUT_TE = LEDGER / "freeze_ref_te_daily.parquet"
OUT_DIST = LEDGER / "freeze_ref_window_dist.parquet"
OUT_Q = LEDGER / "freeze_ref_quantiles.csv"
OUT_CHK = LEDGER / "freeze_ref_checks.csv"
C_CFG = "C 全近似(5m信号+5m执行)"           # m5_bridge 的列名前缀(逐字)


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _checks_append(rows: list[dict]) -> None:
    old = pd.read_csv(OUT_CHK) if OUT_CHK.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old["check"].isin(new["check"])]
    pd.concat([old, new], ignore_index=True).to_csv(OUT_CHK, index=False)


def _daily_merge(cols: dict) -> None:
    D = pd.read_parquet(OUT_DAILY) if OUT_DAILY.exists() else pd.DataFrame()
    for k, s in cols.items():
        s = pd.Series(s).astype(np.float64)
        if len(D):
            D = D.reindex(D.index.union(s.index))
        D[k] = s
    D.index.name = "signal_date"
    D.sort_index().to_parquet(OUT_DAILY)


def _assert_research(idx) -> None:
    idx = pd.DatetimeIndex(idx)
    assert idx.max() <= pd.Timestamp(RESEARCH_END), f"出现研究期之后的日期 {idx.max()}"


# ================================================================ y1:T1/T2 的 Y=1 冲击口径(C 全近似)
def stage_y1():
    import m5_bridge as MB
    from afactor import strategy as ST, minute5 as M5, index_proxy as IP
    from afactor.panel import Panel
    from afactor.impact import CostModel
    p5 = M5.m5_panel(signal=True, exec_=True)
    s5 = pd.read_parquet(MB.SCORE5)
    _assert_research(p5.dates)
    m = MB.year_mask(p5.dates, MB.EVAL_Y0, MB.EVAL_Y1)
    Dm = pd.read_parquet(MB.DAILY_OUT)
    rows, cols = [], {}
    b = ST.benchmark(p5, "V1000", "open10", date_mask=m, rebalance=5)["rets"]
    for plan in MB.PLANS:
        ref_b = Dm[f"{plan}|{C_CFG}|benchmark"].dropna()
        d = float((b.reindex(ref_b.index) - ref_b).abs().max())
        rows.append(dict(check=f"bench_pool5_C_equals_m5_bridge[{plan}]", value=d, ok=bool(d == 0.0),
                         note="同频池基准(5m 执行价)与 m5_bridge_daily 逐位比较,最大绝对差"))
        if d != 0.0:
            raise AssertionError(f"同频基准与 m5_bridge_daily 不一致: {plan} {d}")
    cols["C|bench_pool5"] = b
    cm = CostModel(p5, impact_y=1.0)
    for plan, kw in MB.PLANS.items():
        tag = "T1_excl" if kw["scheme"] == "exclude" else "T2_top20"
        t0 = time.time()
        r0 = ST.run_strategy(s5, p5, "V1000", "open10", date_mask=m, **kw)
        ref = Dm[f"{plan}|{C_CFG}|strategy"].dropna()
        d = float((r0["rets"].reindex(ref.index) - ref).abs().max())
        same_idx = bool(r0["rets"].index.equals(ref.index))
        rows.append(dict(check=f"oldcost_C_reproduces_m5_bridge[{plan}]", value=d, ok=bool(d == 0.0 and same_idx),
                         note=f"旧成本 C 口径重跑 vs m5_bridge_daily 逐位最大绝对差;索引相同={same_idx};"
                              f"{time.time() - t0:.0f}s"))
        if d != 0.0 or not same_idx:
            raise AssertionError(f"旧成本 C 口径没有复现 m5_bridge_daily: {plan} diff={d} idx={same_idx}")
        _log(f"[y1] {plan} 旧成本 C 口径复现 m5_bridge_daily:最大差 {d:.1e}")
        t0 = time.time()
        r1 = ST.run_strategy(s5, p5, "V1000", "open10", date_mask=m, cost_model=cm, **kw)
        assert r1["rets"].index.equals(b.index)
        kpk, cs = MB._corr_peak(r1["rets"], b)
        if kpk != 0:
            raise AssertionError(f"{plan} Y=1 与同频基准错位 k={kpk}: {cs}")
        cols[f"{tag}|Y1|strategy"] = r1["rets"]
        cols[f"{tag}|old|strategy"] = r0["rets"]
        ex = r1["ann_return"] - float((1 + b).prod() ** (TD / len(b)) - 1)
        ex0 = r0["ann_return"] - float((1 + b).prod() ** (TD / len(b)) - 1)
        rows.append(dict(check=f"Y1_C_excess_vs_pool5_2021_2025[{plan}]", value=ex * 100, ok=True,
                         note=f"年化几何差 pp(冷启动 2021-01-04,含建仓爬坡);旧成本 {ex0 * 100:+.2f}pp;"
                              f"Y=1 cost_bps={r1['cost_bps']:.3f} 旧 {r0['cost_bps']:.3f};换手 {r1['turnover']:.4f};"
                              f"仓位 {r1['exposure']:.4f};峰值 k={kpk};{time.time() - t0:.0f}s"))
        _log(f"[y1] {plan} Y=1:年化超额 {ex * 100:+.2f}pp(旧成本 {ex0 * 100:+.2f}pp) cost_bps {r1['cost_bps']:.2f}")
    # 次要参考:C 口径策略对中证 1000 PIT 代理(成交时点权重 × 5m 成交价)
    W = IP.proxy_weights(Panel(), "000852")
    rp = IP.exec_time_bench_returns(p5, "open10", W)
    cols["C|bench_proxy_exec"] = rp.reindex(b.index)
    rows.append(dict(check="C_bench_proxy_exec_nan_days_2021_2025", value=float(rp.reindex(b.index).isna().sum()),
                     ok=bool(rp.reindex(b.index).notna().all()), note="代理基准在 2021-2025 信号日上的缺失天数"))
    _daily_merge(cols)
    _checks_append(rows)
    _log("[y1] 完成", OUT_DAILY.name)


# ================================================================ ie:T3 V1 指增(1 分钟 A 口径)
def stage_ie():
    import run_index_enhance as RIE
    from afactor import strategy as ST, index_proxy as IP
    p = RIE.P()
    _assert_research(p.dates)
    V = RIE.load_variant("V1")
    s = V["rets"].dropna()
    W = IP.proxy_weights(p, "000852")
    rb = IP.exec_time_bench_returns(p, "open10", W)
    mask = RIE.date_mask(RIE.START)
    pool5 = ST.benchmark(p, "V1000", "open10", date_mask=mask, rebalance=5)["rets"]
    real = IP.index_returns_on_signal_days(p, "open10", "000852")
    b = rb.reindex(s.index)
    assert b.notna().all(), "代理基准在 V1 区间有缺失"
    geo = float(((1 + s).prod() / (1 + b).prod()) ** (TD / len(s)) - 1)
    rows = [dict(check="T3_V1_reproduces_ie_variants_geo_excess", value=geo * 100,
                 ok=bool(abs(geo * 100 - 4.50) < 0.006),
                 note="V1 5 相位均值 vs 成交时点代理,全期几何超额 pp;ie_variants.csv V1 = +4.50pp"),
            dict(check="T3_V1_n_days", value=float(len(s)), ok=True,
                 note=f"{s.index.min().date()} ~ {s.index.max().date()}")]
    _log(f"[ie] V1 全期几何超额 {geo * 100:+.3f}pp(应为 +4.50),{len(s)} 天")
    if not rows[0]["ok"]:
        raise AssertionError(f"V1 没有复现 ie_variants: {geo * 100:.4f}")
    k1, cs = _corr_peak(s, b)
    rows.append(dict(check="T3_V1_corr_peak_vs_proxy", value=float(k1), ok=bool(k1 == 0), note=str(cs)))
    _daily_merge({"T3_V1|A|strategy": s, "T3_V1|A|bench_proxy_exec": b,
                  "T3_V1|A|bench_pool5": pool5.reindex(s.index), "T3_V1|A|bench_real000852": real.reindex(s.index)})
    _checks_append(rows)
    _log("[ie] 完成")


def _corr_peak(rs: pd.Series, rb: pd.Series, ks=(-2, -1, 0, 1, 2)):
    a, b = rs.align(rb, join="inner")
    cs = {k: float(a.corr(b.shift(k))) for k in ks}
    return max(cs, key=lambda k: cs[k]), {k: round(v, 4) for k, v in cs.items()}


# ================================================================ te:T5 事前 TE 与 z(ie 报告同式)
def stage_te():
    import run_index_enhance as RIE
    from afactor import index_proxy as IP
    from afactor.barra import RiskModel, psd_sqrt
    p = RIE.P()
    _assert_research(p.dates)
    dates, codes = p.dates, p.codes
    rm = RiskModel.load("open10")
    W = IP.proxy_weights(p, "000852")
    rb, Wv = IP.exec_time_bench_returns(p, "open10", W, return_weights=True)
    Ls = {}

    def L_at(i):
        if i not in Ls:
            Ls[i] = psd_sqrt(rm.cov[i])[1]
        return Ls[i]

    out, rows = {}, []
    for nm, tag in (("V1", "T5_V1"), ("CHK_pool_ew", "T5_poolew")):
        v = RIE.load_variant(nm)
        H = v["H"].reindex(columns=codes).fillna(0.0)
        idx = H.index[(H.index >= RIE.START) & (H.index <= RIE.END)]
        Hv = H.loc[idx].to_numpy(np.float64)
        pos = dates.get_indexer(idx)
        te_pred = np.full(len(idx), np.nan)
        for j, i in enumerate(pos):                     # 逐行照搬 run_index_enhance.report() 的 te_pred
            if not np.isfinite(rm.cov[i, 0, 0]):
                continue
            a = Hv[j] - Wv[i]
            m = np.asarray(rm.mask[i])
            am = a[m]
            X = np.nan_to_num(np.asarray(rm.X[i])[m].astype(np.float64))
            Hm = rm._onehot(rm.ind[m])
            d = np.asarray(rm.spec[i])[m].astype(np.float64)
            with np.errstate(all="ignore"):
                e = np.concatenate([[am.sum()], np.dot(X.T, am), np.dot(Hm.T, am)])
                y = np.dot(L_at(i).T, e)
            te_pred[j] = np.sqrt(np.sum(y * y) + np.sum(np.nan_to_num(d) * am * am)) * np.sqrt(TD / 5)
        tp = pd.Series(te_pred, index=idx)
        act = (v["rets"] - rb).reindex(idx)
        z = act / (tp / np.sqrt(TD))
        out[f"{tag}|te_pred_ann"] = tp
        out[f"{tag}|active"] = act
        out[f"{tag}|z"] = z
        zz = z.dropna().iloc[SKIP:]
        # 与 ie_exposures.csv 的 5 日 b 对照(同一实现,剔除建仓周 5 天,全期)
        act5 = act.rolling(5).sum().shift(-4)
        z5 = (act5 / (tp / np.sqrt(TD / 5))).iloc[5:].dropna()
        b5 = float(np.mean([np.std(z5.iloc[k::5], ddof=1) for k in range(5)]))
        rows.append(dict(check=f"{tag}_b_daily_fullperiod", value=float(zz.std(ddof=1)), ok=True,
                         note=f"全期逐日 z 的 std(去开头 {SKIP} 天);同期 5 日不重叠 b = {b5:.3f}"
                              f"(ie_exposures.csv 全期 V1 1.84、池等权 1.64);事前 TE 均值 {tp.iloc[5:].mean():.4f};"
                              f"事后 TE {act.iloc[5:].std() * np.sqrt(TD):.4f}"))
        _log(f"[te] {nm}: 逐日 b_d {zz.std(ddof=1):.3f};5 日 b {b5:.3f};事前 TE {tp.iloc[5:].mean():.2%};"
             f"事后 TE {act.iloc[5:].std() * np.sqrt(TD):.2%}")
    T = pd.DataFrame(out)
    T.index.name = "signal_date"
    T.to_parquet(OUT_TE)
    _checks_append(rows)
    _log("[te] 完成", OUT_TE.name)


# ================================================================ dist:滚动窗口分布与分位数
def stage_dist():
    from afactor import minute5 as M5
    from scipy.stats import chi2
    assert M5.FWD_SKIP_HEAD == SKIP
    D = pd.read_parquet(OUT_DAILY)
    Dm = pd.read_parquet(LEDGER / "m5_bridge_daily.parquet")
    spec = []          # (target, cost, bench_kind, strategy_series, bench_series, 主/次)
    for plan, tag in (("排雷缓冲20/35 r5", "T1_excl"), ("Top20 周频", "T2_top20")):
        so = Dm[f"{plan}|{C_CFG}|strategy"]
        bo = Dm[f"{plan}|{C_CFG}|benchmark"]
        spec.append((tag, "old", "pool5", so, bo, "主"))
        spec.append((tag, "Y1", "pool5", D[f"{tag}|Y1|strategy"], D["C|bench_pool5"], "主"))
        spec.append((tag, "old", "proxy_exec", so, D["C|bench_proxy_exec"], "次"))
        spec.append((tag, "Y1", "proxy_exec", D[f"{tag}|Y1|strategy"], D["C|bench_proxy_exec"], "次"))
    s3 = D["T3_V1|A|strategy"]
    spec.append(("T3_V1", "Y1", "proxy_exec", s3, D["T3_V1|A|bench_proxy_exec"], "主(1分钟A口径)"))
    spec.append(("T3_V1", "Y1", "pool5", s3, D["T3_V1|A|bench_pool5"], "次(1分钟A口径)"))
    spec.append(("T3_V1", "Y1", "real000852", s3, D["T3_V1|A|bench_real000852"], "参考(收盘对收盘,时点错位)"))
    dist, qrows = {}, []
    for tag, cost, bk, s, b, role in spec:
        df = pd.DataFrame({"strategy": s, "benchmark": b}).dropna()
        for n in NS:
            w = M5.window_excess_distribution(df, n, how="compound", skip_head=SKIP)
            key = f"{tag}|{cost}|{bk}|n{n}"
            dist[key] = w
            qrows.append(dict(target=tag, cost=cost, bench=bk, role=role, stat="window_excess_compound",
                              n=n, n_windows=len(w), first_end=w.index.min().date(), last_end=w.index.max().date(),
                              mean=w.mean(), std=w.std(), frac_pos=float((w > 0).mean()),
                              **{f"q{int(q * 100):02d}": float(w.quantile(q)) for q in QS}))
    # 与 minute5.reference_distribution 逐位一致(旧成本、同频基准,桥接预注册的读法)
    for plan, tag in (("排雷缓冲20/35 r5", "T1_excl"), ("Top20 周频", "T2_top20")):
        for n in NS:
            ref = M5.reference_distribution(plan, n)
            mine = dist[f"{tag}|old|pool5|n{n}"]
            assert ref.index.equals(mine.index) and float((ref - mine).abs().max()) == 0.0, (plan, n)
    # T5:n 日窗口内 z 的样本标准差 b_d(去开头 SKIP 天),以及"风险模型正确"时的 χ² 区间
    T = pd.read_parquet(OUT_TE)
    for tag in ("T5_V1", "T5_poolew"):
        z = T[f"{tag}|z"].dropna().iloc[SKIP:]
        for n in NS:
            w = z.rolling(n, min_periods=n).std(ddof=1).dropna()
            w.name = f"b_d_{n}d"
            dist[f"{tag}|Y1|proxy_exec|n{n}|b_d"] = w
            lo = float(np.sqrt(chi2.ppf(0.05, n - 1) / (n - 1)))
            hi = float(np.sqrt(chi2.ppf(0.95, n - 1) / (n - 1)))
            qrows.append(dict(target=tag, cost="Y1", bench="proxy_exec", role="主(1分钟A口径)", stat="b_d",
                              n=n, n_windows=len(w), first_end=w.index.min().date(), last_end=w.index.max().date(),
                              mean=w.mean(), std=w.std(), frac_pos=np.nan,
                              chi2_band_lo=lo, chi2_band_hi=hi,
                              frac_hist_above_chi2_hi=float((w > hi).mean()),
                              **{f"q{int(q * 100):02d}": float(w.quantile(q)) for q in QS}))
    Q = pd.DataFrame(qrows)
    Q.to_csv(OUT_Q, index=False)
    L = pd.concat({k: v.rename("value") for k, v in dist.items()}, names=["key", "window_end"]).reset_index()
    L.to_parquet(OUT_DIST)
    _checks_append([dict(check="dist_oldcost_equals_minute5_reference_distribution", value=0.0, ok=True,
                         note="T1/T2 旧成本同频分布与 minute5.reference_distribution(plan, n) 逐位一致,n=16/17/18")])
    show = Q[Q.n == 17][["target", "cost", "bench", "stat", "n_windows", "mean", "std", "q05", "q50", "q95"]]
    print(show.to_string(index=False), flush=True)
    _log("[dist] 完成", OUT_Q.name, OUT_DIST.name)


# ================================================================ pead:T4 事件层与账户层参考(只读已落盘的研究期产物)
PEAD_NS = (20, 21, 22)       # T4 不跳过建仓日(E 与 B 同时点、同目标现金),窗口 = 信号日个数 N
PEAD_BIG_SLOTS = 50          # 大批次门槛:该批有限 e5 的 E 槽位数


def stage_pead():
    """T4 的历史参考。输入全是 PEAD 研究已落盘的研究期产物,不重算任何东西:
      - cache/_tmp/pead_event_level_slots.parquet:E 入选槽位的 h5 事件层配对超额 e5_B / e5_C
        (scripts/pead_event_level.py,TWAP 理论收益,买不进记 0);
      - cache/_tmp/pead_daily_base_1m.parquet / pead_daily_base_10m.parquet:E / B 账户逐日收益(100 万 / 1000 万,基准费率)。
    事件层(主读数,预注册 §5 T4):D_T = 同一 t_signal 上 E 槽位 e5 的均值;只用"大批次"(该批有限 e5 的 E 槽位
            >= PEAD_BIG_SLOTS 个)—— 小批次(财报季边缘几只股票)的 D_T 噪声大一倍以上,前向 9 月初的批次都是几百只。
            窗口统计 D̄ = 窗口(N 个连续信号日)内大批次 D_T 的等权均值;同时记下窗口内大批次个数 k。
            前向查分布时只取 k ∈ [k_fwd−1, k_fwd+1] 的历史窗口(不足 50 个则两边各再放宽 1,直到 >= 50)。
            k 由公告日历决定、与收益无关,所以按 k 筛选不是事后挑分布。这里另给 k ∈ [3, 5] 的分位(预计 k_fwd = 4)作示例。
            另给"同季"参考:每年 9 月首个信号日起 N 个信号日的同一统计(2021-2025 各 1 个)。
    账户层:E − B 的复利差 Π(1+r_E) − Π(1+r_B),E = 5 相位均值,B = 3 个复制组合 × 5 相位均值;只作描述性参考
            (历史是连续运行切片,大量窗口两臂都空仓、差恰为 0)。"""
    from afactor.panel import Panel
    dates = Panel().dates
    _assert_research(dates)
    E = pd.read_parquet(TMP / "pead_event_level_slots.parquet")
    E = E[E["arm"] == "E"]
    rows, dist = [], {}
    sig = pd.Series(dates[(dates >= "2021-01-01") & (dates <= RESEARCH_END)])
    for ctrl in ("B", "C"):
        col = f"e5_{ctrl}"
        g = E[np.isfinite(E[col])].groupby("t_signal")[col]
        DT = g.mean().reindex(sig.values)                      # 无批次的信号日为 NaN
        nT = g.size().reindex(sig.values).fillna(0)
        big = DT.where(nT >= PEAD_BIG_SLOTS)
        for N in PEAD_NS:
            s = big.rolling(N, min_periods=1).mean()           # 窗口内大批次 D_T 的等权均值
            k = big.notna().astype(float).rolling(N, min_periods=N).sum()
            ok = k.notna() & (k >= 1)
            w, kk = s[ok], k[ok]
            w.name, kk.name = f"Dbar_big_{N}d", f"k_big_{N}d"
            dist[f"T4_E|base|{ctrl}|n{N}|Dbar_big"] = w
            dist[f"T4_E|base|{ctrl}|n{N}|k_big"] = kk
            same = []
            for y in range(2021, 2026):
                sep = sig[sig >= pd.Timestamp(f"{y}-09-01")].iloc[:N]
                v = big.reindex(sep.values)
                same.append(dict(year=y, value=float(v.mean()), n_batches=int(v.notna().sum()),
                                 n_slots=int(nT.reindex(sep.values).where(v.notna()).sum())))
            sm = pd.DataFrame(same)
            for lo, hi, lab in ((1, N, "全部 k>=1"), (3, 5, "k∈[3,5](示例:k_fwd=4)")):
                ww = w[(kk >= lo) & (kk <= hi)]
                rows.append(dict(target="T4_E", cost="base_1m", bench=f"{ctrl}(匹配对照)",
                                 role=("主" if ctrl == "B" else "次") + f"|{lab}",
                                 stat="Dbar_big_event_h5", n=N, n_windows=len(ww),
                                 first_end=ww.index.min().date(), last_end=ww.index.max().date(),
                                 mean=ww.mean(), std=ww.std(), frac_pos=float((ww > 0).mean()),
                                 **{f"q{int(q * 100):02d}": float(ww.quantile(q)) for q in QS},
                                 same_season="; ".join(f"{r.year}:{r.value:+.4f}({r.n_batches}批/{r.n_slots}槽)"
                                                       for r in sm.itertuples())))
    # 账户层 E − B(100 万 / 1000 万,基准费率)
    for sc, tag in (("base_1m", "100万"), ("base_10m", "1000万")):
        Dd = pd.read_parquet(TMP / f"pead_daily_{sc}.parquet")
        Dd = Dd[Dd["date"] <= RESEARCH_END]
        rE = Dd[Dd["arm"] == "E"].groupby("date")["ret"].mean()
        rB = Dd[Dd["arm"] == "B"].groupby("date")["ret"].mean()
        df = pd.DataFrame({"strategy": rE, "benchmark": rB}).dropna()
        df = df[df.index >= "2021-01-01"]
        for N in PEAD_NS:
            from afactor import minute5 as M5
            w = M5.window_excess_distribution(df, N, how="compound", skip_head=0)
            dist[f"T4_E|{sc}|B|n{N}|account"] = w
            rows.append(dict(target="T4_E", cost=sc, bench="B(匹配对照)", role="次(账户层,历史为连续运行切片)",
                             stat="account_E_minus_B_compound", n=N, n_windows=len(w),
                             first_end=w.index.min().date(), last_end=w.index.max().date(),
                             mean=w.mean(), std=w.std(), frac_pos=float((w > 0).mean()),
                             **{f"q{int(q * 100):02d}": float(w.quantile(q)) for q in QS}))
    Q = pd.read_csv(OUT_Q)
    Q = Q[Q["target"] != "T4_E"]
    Q = pd.concat([Q, pd.DataFrame(rows)], ignore_index=True)
    Q.to_csv(OUT_Q, index=False)
    L0 = pd.read_parquet(OUT_DIST)
    L0 = L0[~L0["key"].str.startswith("T4_E|")]
    L1 = pd.concat({k: v.rename("value") for k, v in dist.items()}, names=["key", "window_end"]).reset_index()
    pd.concat([L0, L1], ignore_index=True).to_parquet(OUT_DIST)
    print(pd.DataFrame(rows)[["cost", "bench", "stat", "n", "n_windows", "mean", "std", "q05", "q50", "q95"]]
          .to_string(index=False), flush=True)
    for r in rows:
        if r.get("same_season"):
            print(f"  同季参考 {r['bench']} n={r['n']}: {r['same_season']}", flush=True)
    _log("[pead] 完成")


# ================================================================ chk:前向引擎断言的历史容差
def stage_chk():
    """前向面板上的引擎判据"全池等权 vs 同频池等权:超额 + 成本 ≈ 0"只有 n 天,需要一个事先定好的容差。
    用研究期同一判据组合(ie_CHK_pool_ew:常数打分全池等权、5 日调仓、Y=1、1000 万、2021-04-20 冷启动)
    对同频池基准的逐日 (r − r_b + cost),去开头 4 天,取 n 日滚动和的分布;前向值落在 [q0.5, q99.5] 之外即判据不通过。"""
    import run_index_enhance as RIE
    D = pd.read_parquet(OUT_DAILY)
    v = RIE.load_variant("CHK_pool_ew")
    r = v["daily"]["ret"]
    c = v["daily"]["cost_total"]
    b = D["T3_V1|A|bench_pool5"]
    d = (r - b.reindex(r.index) + c).dropna().iloc[SKIP:]
    rows, dist = [], {}
    for n in NS:
        w = d.rolling(n, min_periods=n).sum().dropna()
        w.name = f"chk_{n}d"
        dist[f"CHK_poolew|Y1|pool5|n{n}|excess_plus_cost_sum"] = w
        rows.append(dict(target="CHK_poolew", cost="Y1", bench="pool5", role="引擎判据容差(1分钟A口径)",
                         stat="excess_plus_cost_sum", n=n, n_windows=len(w),
                         first_end=w.index.min().date(), last_end=w.index.max().date(),
                         mean=w.mean(), std=w.std(), frac_pos=float((w > 0).mean()),
                         q005=float(w.quantile(0.005)), q995=float(w.quantile(0.995)),
                         **{f"q{int(q * 100):02d}": float(w.quantile(q)) for q in QS}))
    Q = pd.read_csv(OUT_Q)
    Q = Q[Q["target"] != "CHK_poolew"]
    pd.concat([Q, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT_Q, index=False)
    L0 = pd.read_parquet(OUT_DIST)
    L0 = L0[~L0["key"].str.startswith("CHK_poolew|")]
    L1 = pd.concat({k: v_.rename("value") for k, v_ in dist.items()}, names=["key", "window_end"]).reset_index()
    pd.concat([L0, L1], ignore_index=True).to_parquet(OUT_DIST)
    print(pd.DataFrame(rows)[["n", "n_windows", "mean", "std", "q005", "q05", "q50", "q95", "q995"]].to_string(index=False))
    _log("[chk] 完成")


STAGES = {"y1": stage_y1, "ie": stage_ie, "te": stage_te, "dist": stage_dist, "pead": stage_pead, "chk": stage_chk}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STAGES)
    for st in todo:
        t0 = time.time()
        _log(f"===== 阶段 {st} =====")
        STAGES[st]()
        _log(f"===== 阶段 {st} 完成 {time.time() - t0:.0f}s =====")
