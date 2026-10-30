"""前向检验读数汇总(前缀 fwd;预注册 FORWARD_TEST_PREREG §5、§6):指标、百分位、判定、引擎断言、账本与图。

输入(scripts/fwd_run_test.py、fwd_pead_run.py 的落盘结果,全部只含信号日 >= 2026-09-01):
  cache/_tmp/fwd_strategy_rets.parquet、fwd_bench.parquet、fwd_<obj>_daily.parquet(引擎 record)、
  fwd_T3_V1_ph*_*.parquet、fwd_te_daily.parquet、fwd_pead_<mode>_*.parquet
参考分布(冻结时已算好,只查不重算):research/ledger/freeze_ref_window_dist.parquet、freeze_ref_quantiles.csv。
产出:research/ledger/fwd_results.csv(每个读数一行)、fwd_daily.csv(逐日序列)、reports/fwd_excess_path.png。

    python3 scripts/fwd_report.py
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

from afactor.config import TMP, LEDGER, REPORTS
from afactor import minute5 as M5, pead_engine as PE
from afactor.fwd_panel import assert_forward_index, SIG_START
import fwd_common as FC

log = FC.log
TD = 242
SKIP = M5.FWD_SKIP_HEAD
DIST = pd.read_parquet(LEDGER / "freeze_ref_window_dist.parquet")
QREF = pd.read_csv(LEDGER / "freeze_ref_quantiles.csv")
OUT = LEDGER / "fwd_results.csv"
OUT_D = LEDGER / "fwd_daily.csv"
rows: list[dict] = []


def dist(key: str) -> pd.Series:
    d = DIST[DIST["key"] == key]
    if d.empty:
        raise KeyError(key)
    return d.set_index("window_end")["value"]


def verdict(pct: float, top20: bool = False, readable: bool = True) -> str:
    """预注册 §6.1:[5, 95] 与历史一致;< 5 异常差,需调查;> 95 异常好,需检查数据问题。Top20 ±8 点内记边界。"""
    if not readable or not np.isfinite(pct):
        return "不可读"
    if pct < 5:
        v = "异常差,需调查"
    elif pct > 95:
        v = "异常好,需检查是否有数据问题"
    else:
        v = "与历史一致"
    if top20 and (abs(pct - 5) <= 8 or abs(pct - 95) <= 8):
        v += "(边界,读数不确定:Top20 近似误差约 ±8 点)"
    return v


def window_stats(rs: pd.Series, rb: pd.Series) -> dict:
    """评估窗(去掉开头 SKIP 个建仓日)的全部 §5.1 指标。"""
    df = pd.DataFrame({"strategy": rs, "benchmark": rb})
    v, n = M5.forward_window_excess(df, skip_head=SKIP, how="compound")
    e = df.dropna().iloc[SKIP:]
    d = (e["strategy"] - e["benchmark"]).to_numpy()
    se = PE.nw_se(d, 5)
    return dict(window_excess=v, n_eval=n, eval_first=str(e.index[0].date()), eval_last=str(e.index[-1].date()),
                ratio_geo_excess=float(np.prod(1 + e["strategy"]) / np.prod(1 + e["benchmark"]) - 1),
                strat_cum=float(np.prod(1 + e["strategy"]) - 1), bench_cum=float(np.prod(1 + e["benchmark"]) - 1),
                daily_mean=float(d.mean()), nw_t_L5=float(d.mean() / se) if se and se > 0 else np.nan,
                te_realized=float(d.std(ddof=1) * np.sqrt(TD)))


def corr_peak(rs: pd.Series, rb: pd.Series) -> tuple[int, dict]:
    a, b = rs.align(rb, join="inner")
    cs = {k: float(a.corr(b.shift(k))) for k in (-2, -1, 0, 1, 2)}
    return max(cs, key=lambda k: cs[k] if np.isfinite(cs[k]) else -9), cs


def cost_block(daily: pd.DataFrame) -> dict:
    d = daily.sort_index()
    bld, ev = d.iloc[:SKIP], d.iloc[SKIP:]
    out = {}
    for lab, x in (("build", bld), ("eval", ev)):
        out[f"cost_total_{lab}"] = float(x["cost_total"].sum())
        for c in ("cost_commission", "cost_stamp", "cost_spread", "cost_impact"):
            out[f"{c}_{lab}"] = float(x[c].sum())
        out[f"turnover_mean_{lab}"] = float(x["turnover"].mean())
        out[f"exposure_mean_{lab}"] = float(x["exposure"].mean())
        out[f"n_hold_mean_{lab}"] = float(x["n_hold"].mean())
    out["exposure_max"] = float(d["exposure"].max())
    return out


def add(**kw):
    rows.append(kw)


def main():
    t0 = time.time()
    S = pd.read_parquet(TMP / "fwd_strategy_rets.parquet")
    B = pd.read_parquet(TMP / "fwd_bench.parquet")
    for x, w in ((S, "策略"), (B, "基准")):
        assert_forward_index(x, w)
    sig = B.index
    assert len(sig) == 21 and sig[0] == SIG_START, (len(sig), sig[0])
    daily_out = {"bench_pool5": B["pool5"], "bench_csi1000_exec": B["csi1000_exec"], "bench_real000852": B["real000852"]}
    checks_ok = True

    # ------------------------------------------------ T1 / T2 / CHK
    objs = [("T1_excl", "old", "P1"), ("T1_excl", "Y1", "P2"), ("T2_top20", "old", "P3"), ("T2_top20", "Y1", None)]
    for tag, cost, pid in objs:
        rs = S[f"{tag}|{cost}"]
        rec = pd.read_parquet(TMP / f"fwd_{tag}_{cost}_daily.parquet")
        byp = pd.read_parquet(TMP / f"fwd_{tag}_{cost}_daily_by_phase.parquet")
        cb = cost_block(rec)
        kk, cs = corr_peak(rs, B["pool5"])
        ok_align = kk == 0
        ok_exp = cb["exposure_max"] <= 1 + 1e-9
        checks_ok &= ok_align and ok_exp
        daily_out[f"{tag}|{cost}"] = rs
        daily_out[f"{tag}|{cost}|cost"] = rec["cost_total"]
        daily_out[f"{tag}|{cost}|exposure"] = rec["exposure"]
        for bk, bcol, dkey, role in (("pool5", "pool5", f"{tag}|{cost}|pool5|n17", "主"),
                                     ("csi1000_actual_exec", "csi1000_exec", f"{tag}|{cost}|proxy_exec|n17",
                                      "次(历史为 PIT 代理)"),
                                     ("real000852", "real000852", None, "参考(收盘对收盘)")):
            st = window_stats(rs, B[bcol])
            pct = M5.percentile_in(st["window_excess"], dist(dkey)) if dkey else np.nan
            main_ = role == "主"
            add(section="reading", object=tag, cost=cost, bench=bk, role=role,
                reading=(pid if (main_ and pid) else ("描述" if main_ else "次要/参考")), **st,
                hist_key=dkey, hist_pct=pct, hist_n_windows=(len(dist(dkey)) if dkey else np.nan),
                hist_p05=(float(dist(dkey).quantile(0.05)) if dkey else np.nan),
                hist_p50=(float(dist(dkey).quantile(0.50)) if dkey else np.nan),
                hist_p95=(float(dist(dkey).quantile(0.95)) if dkey else np.nan),
                verdict=(verdict(pct, top20=(tag == "T2_top20"), readable=ok_align and ok_exp) if dkey else "参考,不判定"),
                corr_peak_k=(kk if bk == "pool5" else np.nan), corr_profile=(str({k: round(v, 4) for k, v in cs.items()})
                                                                         if bk == "pool5" else ""),
                **(cb if bk == "pool5" else {}))
        # 逐相位(描述;相位之间不独立)
        for ph, g in byp.groupby("phase"):
            r_ph = g["ret"].reindex(sig)
            st = window_stats(r_ph, B["pool5"])
            add(section="phase", object=tag, cost=cost, bench="pool5", role="相位(描述)", reading=f"phase{ph}", **st,
                hist_key=f"{tag}|{cost}|pool5|n17", hist_pct=M5.percentile_in(st["window_excess"], dist(f"{tag}|{cost}|pool5|n17")))
    # 全池等权判据(引擎断言 ③)
    rs = S["CHK_poolew|Y1"]
    rec = pd.read_parquet(TMP / "fwd_CHK_poolew_Y1_daily.parquet")
    e = (rs - B["pool5"] + rec["cost_total"].reindex(rs.index)).iloc[SKIP:]
    q = QREF[(QREF["target"] == "CHK_poolew") & (QREF["n"] == 17)].iloc[0]
    v = float(e.sum())
    ok3 = q["q005"] <= v <= q["q995"]
    kk, cs = corr_peak(rs, B["pool5"])
    cb = cost_block(rec)
    checks_ok &= bool(ok3) and kk == 0 and cb["exposure_max"] <= 1 + 1e-9
    daily_out["CHK_poolew|Y1"] = rs
    daily_out["CHK_poolew|Y1|cost"] = rec["cost_total"]
    add(section="engine_check", object="CHK_poolew", cost="Y1", bench="pool5", role="引擎断言③",
        reading="Σ(超额+成本) 评估窗", window_excess=v, n_eval=len(e), hist_key="CHK_poolew|Y1|pool5|n17|excess_plus_cost_sum",
        hist_p05=float(q["q005"]), hist_p95=float(q["q995"]),
        hist_pct=M5.percentile_in(v, dist("CHK_poolew|Y1|pool5|n17|excess_plus_cost_sum")),
        verdict=("通过" if ok3 else "不通过"), corr_peak_k=kk, corr_profile=str({k: round(x, 4) for k, x in cs.items()}),
        note=f"容差 [{q['q005']:.5f}, {q['q995']:.5f}](研究期同一判据 17 日滚动和的 0.5% / 99.5% 分位;hist_p05/p95 栏放的是这两个分位)",
        **cb)

    # ------------------------------------------------ T3
    t3_ok = (TMP / "fwd_T3_V1_ph4_daily.parquet").exists()
    if t3_ok:
        import fwd_run_test as RT
        V = RT.load_t3()
        r3 = V["rets"].reindex(sig)
        cb = cost_block(V["daily"])
        kk, cs = corr_peak(r3, B["csi1000_exec"])
        okc = kk == 0 and cb["exposure_max"] <= 1 + 1e-9
        checks_ok &= okc
        daily_out["T3_V1"] = r3
        daily_out["T3_V1|cost"] = V["daily"]["cost_total"]
        daily_out["T3_V1|exposure"] = V["daily"]["exposure"]
        for bk, bcol, dkey, role, pid in (("csi1000_actual_exec", "csi1000_exec", "T3_V1|Y1|proxy_exec|n17", "主", "P4"),
                                          ("pool5", "pool5", "T3_V1|Y1|pool5|n17", "次", None),
                                          ("real000852", "real000852", "T3_V1|Y1|real000852|n17", "参考(收盘对收盘)", None)):
            st = window_stats(r3, B[bcol])
            pct = M5.percentile_in(st["window_excess"], dist(dkey))
            add(section="reading", object="T3_V1", cost="Y1", bench=bk, role=role, reading=pid or "次要/参考", **st,
                hist_key=dkey, hist_pct=pct, hist_n_windows=len(dist(dkey)),
                hist_p05=float(dist(dkey).quantile(0.05)), hist_p50=float(dist(dkey).quantile(0.5)),
                hist_p95=float(dist(dkey).quantile(0.95)),
                verdict=(verdict(pct, readable=okc) if pid else "描述(不判定)"),
                corr_peak_k=(kk if pid else np.nan), corr_profile=(str({k: round(x, 4) for k, x in cs.items()}) if pid else ""),
                **(cb if pid else {}))
        sl = V["solver"]
        add(section="t3_solver", object="T3_V1", cost="Y1", bench="-", role="求解记录", reading="状态计数",
            note="; ".join(f"{k}:{v}" for k, v in sl["status"].value_counts().items()),
            window_excess=float(sl["te_ex_ante"].mean()) if "te_ex_ante" in sl else np.nan,
            n_eval=len(sl))
        for ph, g in V["by_phase"].groupby("phase"):
            st = window_stats(g["ret"].reindex(sig), B["csi1000_exec"])
            add(section="phase", object="T3_V1", cost="Y1", bench="csi1000_actual_exec", role="相位(描述)",
                reading=f"phase{ph}", **st, hist_key="T3_V1|Y1|proxy_exec|n17",
                hist_pct=M5.percentile_in(st["window_excess"], dist("T3_V1|Y1|proxy_exec|n17")))

    # ------------------------------------------------ T5
    if (TMP / "fwd_te_daily.parquet").exists():
        T5 = pd.read_parquet(TMP / "fwd_te_daily.parquet").reindex(sig)
        q5 = QREF[(QREF["target"] == "T5_V1") & (QREF["n"] == 17)].iloc[0]
        for tag, pid in (("T5_V1", "P6"), ("T5_poolew", None)):
            z = T5[f"{tag}|z"].iloc[SKIP:]
            bd = float(z.std(ddof=1))
            tp = T5[f"{tag}|te_pred_ann"].iloc[SKIP:]
            act = T5[f"{tag}|active"].iloc[SKIP:]
            key = f"{tag}|Y1|proxy_exec|n17|b_d"
            pct = M5.percentile_in(bd, dist(key))
            inband = bool(q5["chi2_band_lo"] <= bd <= q5["chi2_band_hi"])
            daily_out[f"{tag}|te_pred_ann"] = T5[f"{tag}|te_pred_ann"]
            daily_out[f"{tag}|z"] = T5[f"{tag}|z"]
            add(section="reading", object=tag, cost="Y1", bench="csi1000_actual_exec", role="主" if pid else "描述",
                reading=pid or "描述", window_excess=bd, n_eval=int(z.notna().sum()),
                eval_first=str(z.index[0].date()), eval_last=str(z.index[-1].date()),
                hist_key=key, hist_pct=pct, hist_n_windows=len(dist(key)),
                hist_p05=float(dist(key).quantile(0.05)), hist_p50=float(dist(key).quantile(0.5)),
                hist_p95=float(dist(key).quantile(0.95)),
                verdict=(verdict(pct) if pid else "描述(不判定)"),
                te_pred_mean=float(tp.mean()), te_realized=float(act.std(ddof=1) * np.sqrt(TD)),
                te_ratio=float(act.std(ddof=1) * np.sqrt(TD) / tp.mean()),
                note=f"b_d = 评估窗内 z 的样本标准差(ddof=1);模型正确时的 χ² 区间 [{q5['chi2_band_lo']:.2f}, {q5['chi2_band_hi']:.2f}],"
                     f"落在区间内={inband};历史上 {q5['frac_hist_above_chi2_hi']:.0%} 的 17 日窗口高于上界")

    # ------------------------------------------------ T4
    ev = pd.read_csv(LEDGER / "fwd_pead_events.csv")
    n_orig = int(ev[(ev["mode"] == "orig") & (ev["item"] == "batches_with_selection")]["value"].iloc[0])
    add(section="reading", object="T4_PEAD", cost="base", bench="B", role="主", reading="P5",
        verdict="事件不足,不作读数", n_eval=0,
        note=f"预注册原规则(财报沿用最后已知):窗口内批次 {n_orig} 个、大批次 0 个(fwd_pead_events.csv,mode=orig)")
    fslot = TMP / "fwd_pead_splice_event_slots.parquet"
    if fslot.exists():
        E = pd.read_parquet(fslot)
        E = E[E["h5_available"]]
        g = E[np.isfinite(E["e5_B"])].groupby("t_signal")["e5_B"]
        DT, nT = g.mean(), g.size()
        big = DT[nT >= 50]
        k = len(big)
        Dbar = float(big.mean()) if k else np.nan
        w = dist("T4_E|base|B|n21|Dbar_big")
        kb = dist("T4_E|base|B|n21|k_big").reindex(w.index)
        lo, hi = k - 1, k + 1
        while ((kb >= lo) & (kb <= hi)).sum() < 50 and lo > 0:
            lo, hi = lo - 1, hi + 1
        wf = w[(kb >= lo) & (kb <= hi)]
        pct = M5.percentile_in(Dbar, wf)
        q4 = QREF[(QREF["target"] == "T4_E") & (QREF["stat"] == "Dbar_big_event_h5") & (QREF["bench"].str.startswith("B"))
                  & (QREF["n"] == 21)].iloc[0]
        add(section="reading", object="T4_PEAD", cost="base", bench="B", role="偏离预注册 D1(补充,不判定)",
            reading="P5(D1 口径)", window_excess=Dbar, n_eval=k, hist_key=f"T4_E|base|B|n21|Dbar_big & k∈[{lo},{hi}]",
            hist_pct=pct, hist_n_windows=len(wf), hist_p05=float(wf.quantile(0.05)), hist_p50=float(wf.quantile(0.5)),
            hist_p95=float(wf.quantile(0.95)), verdict="D1 补充读数:" + verdict(pct),
            note="大批次 D_T:" + "; ".join(f"{d.date()}:{v:+.4f}({int(nT[d])}槽)" for d, v in big.items())
                 + f";小批次 {int((nT < 50).sum())} 个;同季参考 {q4['same_season']}")
        for d_, v_ in DT.items():
            add(section="t4_batch", object="T4_PEAD", cost="base", bench="B", role="D1 批次", reading=str(d_.date()),
                window_excess=float(v_), n_eval=int(nT[d_]), note="大批次" if nT[d_] >= 50 else "小批次(不进 D̄)")
        fd = TMP / "fwd_pead_splice_daily.parquet"
        if fd.exists():
            D = pd.read_parquet(fd)
            for sc in ("base_1m", "base_10m"):
                x = D[D["scenario"] == sc]
                rE = x[x["arm"] == "E"].groupby("date")["ret"].mean().reindex(sig)
                rB = x[x["arm"] == "B"].groupby("date")["ret"].mean().reindex(sig)
                df = pd.DataFrame({"strategy": rE, "benchmark": rB})
                v, n = M5.forward_window_excess(df, skip_head=0)
                key = f"T4_E|{sc}|B|n21|account"
                add(section="reading", object="T4_PEAD_account", cost=sc, bench="B", role="账户层(D1,描述)",
                    reading="E−B 复利差", window_excess=v, n_eval=n, hist_key=key,
                    hist_pct=M5.percentile_in(v, dist(key)), hist_p05=float(dist(key).quantile(0.05)),
                    hist_p50=float(dist(key).quantile(0.5)), hist_p95=float(dist(key).quantile(0.95)),
                    verdict="描述(不判定)", strat_cum=float(np.prod(1 + rE) - 1), bench_cum=float(np.prod(1 + rB) - 1),
                    exposure_mean_eval=float(x[x["arm"] == "E"].groupby("date")["exposure"].mean().mean()),
                    cost_total_eval=float(x[x["arm"] == "E"].groupby("date")["cost_total"].mean().sum()),
                    note="E = 5 相位均值;B = 3 个复制组合 × 5 相位均值;不跳过建仓日(预注册 §3.4)")
                daily_out[f"T4_E|{sc}"] = rE
                daily_out[f"T4_B|{sc}"] = rB

    # ------------------------------------------------ 引擎断言汇总
    add(section="engine_check", object="ALL", cost="-", bench="-", role="§5.5", reading="全部断言",
        verdict="通过" if checks_ok else "不通过(全部读数暂停)",
        note="① exposure<=1(引擎循环内硬断言 + 记录核对);② 各对象对主基准 corr 峰值 k=0(全部 N=21 天);"
             "③ 全池等权 Σ(超额+成本) 在容差内;④ 落盘序列首个索引 >= 2026-09-01")
    R = pd.DataFrame(rows)
    R.insert(0, "run_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    R.to_csv(OUT, index=False)
    Dd = pd.DataFrame(daily_out).reindex(sig)
    Dd.insert(0, "is_eval", np.arange(len(sig)) >= SKIP)
    Dd.index.name = "signal_date"
    assert_forward_index(Dd, "fwd_daily")
    Dd.to_csv(OUT_D)
    show = R[R["section"].isin(["reading", "engine_check"])][
        ["object", "cost", "bench", "reading", "window_excess", "n_eval", "hist_pct", "verdict"]]
    pd.set_option("display.width", 250)
    print(show.to_string(index=False))
    plot(Dd, R)
    log(f"写 {OUT.name}({len(R)} 行)、{OUT_D.name}、reports/fwd_excess_path.png [{time.time()-t0:.0f}s]")


def plot(Dd: pd.DataFrame, R: pd.DataFrame):
    """逐日累计超额路径(评估窗从 0 起算;建仓日灰底);右侧竖条 = 同长度历史窗口的 p05–p95 与中位数。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    SURF, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    SER = ["#2a78d6", "#eb6834", "#1baf7a"]
    sig = Dd.index
    ev = sig[SKIP:]
    panels = [("相对同频池等权基准(T1 / T2)", "bench_pool5",
               [("T1_excl|old", "T1 排雷 旧成本"), ("T1_excl|Y1", "T1 排雷 Y=1"), ("T2_top20|old", "T2 Top20 旧成本")]),
              ("相对中证 1000 实际成分(成交时点计价)", "bench_csi1000_exec", [("T3_V1", "T3 指增 V1")])]
    keymap = {"T1_excl|old": ("T1_excl", "old", "pool5"), "T1_excl|Y1": ("T1_excl", "Y1", "pool5"),
              "T2_top20|old": ("T2_top20", "old", "pool5"), "T3_V1": ("T3_V1", "Y1", "csi1000_actual_exec")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURF, sharey=False)
    for ax, (title, bcol, series) in zip(axes, panels):
        ax.set_facecolor(SURF)
        x = np.arange(len(sig))
        ax.axvspan(-0.5, SKIP - 0.5, color=GRID, alpha=0.6, lw=0)
        ax.text(SKIP / 2 - 0.5, 0.97, "建仓日\n(不计入)", transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=TXT2, fontsize=8)
        ax.axhline(0, color=TXT2, lw=0.8)
        xe = len(sig) - 1
        end_labels = []
        for k, (col, lab) in enumerate(series):
            if col not in Dd:
                continue
            rs, rb = Dd[col].loc[ev], Dd[bcol].loc[ev]
            path = (np.cumprod(1 + rs) - np.cumprod(1 + rb)) * 100
            xs = np.arange(SKIP - 1, len(sig))                  # 锚点:建仓结束(第 4 个信号日)记 0
            yv = np.concatenate([[0.0], path.to_numpy()])
            ax.plot(xs, yv, color=SER[k], lw=2, marker="o", ms=3.5, label=lab)
            end_labels.append((path.iloc[-1], f"{path.iloc[-1]:+.2f}%"))
            o, c, b = keymap[col]
            r = R[(R["section"] == "reading") & (R["object"] == o) & (R["cost"] == c) & (R["bench"] == b)]
            if len(r):
                r = r.iloc[0]
                xb = xe + 2.2 + 1.1 * k
                ax.plot([xb, xb], [r["hist_p05"] * 100, r["hist_p95"] * 100], color=SER[k], lw=5, alpha=0.35,
                        solid_capstyle="round")
                ax.plot([xb], [r["hist_p50"] * 100], marker="_", ms=10, color=SER[k])
                ax.plot([xb], [r["window_excess"] * 100], marker="o", ms=6, color=SER[k], mec=SURF, mew=1.5)
        # 末端标签:数值相近时上下错开,避免重叠
        lim = ax.get_ylim(); gap = 0.045 * (lim[1] - lim[0]); placed = []
        for yv0, txt in sorted(end_labels):
            yy = yv0
            for p_ in placed:
                if abs(yy - p_) < gap:
                    yy = p_ + gap
            placed.append(yy)
            ax.text(xe + 0.3, yy, txt, color=TXT, fontsize=8, va="center")
        ax.set_xlim(-0.5, len(sig) + 4.5)
        ticks = list(range(0, len(sig), 4)) + [len(sig) - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([sig[i].strftime("%m-%d") for i in ticks], fontsize=8)
        ax.set_title(title, color=TXT, fontsize=11, loc="left")
        ax.set_ylabel("评估窗累计复利超额(%)", color=TXT2, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=TXT2, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=TXT)
        ax.text(len(sig) + 2.6, 1.0, "历史 17 日窗口\np05–p95 / 中位", color=TXT2, fontsize=7,
                ha="center", va="bottom", transform=ax.get_xaxis_transform())
    fig.suptitle("前向检验 2026-09:信号日 09-07 ~ 09-30(n=17)的逐日累计超额,收益到 10-09 实现", color=TXT, fontsize=12,
                 x=0.01, ha="left")
    fig.text(0.01, 0.005, "5 分钟口径(baostock)信号与执行价;冷启动,前 4 个信号日只建仓。右侧竖条:2021-2025 同口径 17 日滚动窗口"
                          "(T1/T2 为 C 口径、T3 为 1 分钟口径对 PIT 代理)的 p05–p95,横线为中位,圆点为前向值。",
             color=TXT2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(REPORTS / "fwd_excess_path.png", dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    main()
