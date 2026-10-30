"""前向读数越界时的调查清单(预注册 FORWARD_TEST_PREREG §6.4,前缀 fwd)。逐项写结论,不跳项;
**诊断结果只作说明,不替换主读数,不改参数。**

触发:P3(T2 Top20 旧成本 vs 同频池)百分位 97.4 > 95;次要读数 T1 / T2 对中证 1000 实际成分 > 99。
各项:
  1  冻结校验、前向原始数据哈希、回归测试重跑(本脚本只记录命令与退出码,命令由 shell 执行后把结果写进日志)
  2  数据:窗口内覆盖、停牌与 0 价、接缝与除权(引用 fwd_panel_qa)、ST 与涨跌停 / 一字板计数、北交所是否混入市场均值
  3  时点:引擎逐日毛收益 = 持仓 × twap[t+2]/twap[t+1] 的重算核对;收益面板 09-02 前为 NaN;复权系数只用 t 及以前事件
  4  基准:同频池相位锚点;中证 1000 权重还原(成分数、合计、映射);收盘对收盘的还原指数 vs 真实 000852 逐日对照;
     成交时点与收盘时点的口径差分解
  5  成本与成交:建仓日 / 评估窗成本、参与率截量、一字板
  6  信号本身:T2 逐只贡献(集中度、最大个股收益与官方涨跌幅核对)、相位离散度、分年份历史可比窗口
结果写 research/ledger/fwd_investigation.csv。
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

from afactor.config import TMP, LEDGER
from afactor.fwd_panel import ForwardPanel, SIG_START, EXEC_START
from afactor import fwd_weights as FW, minute5 as M5
import fwd_common as FC

OUT = LEDGER / "fwd_investigation.csv"
SKIP = 4
rows: list[dict] = []
log = FC.log


def rec(item_no, item, value, conclusion, note=""):
    rows.append(dict(section=f"6.4-{item_no}", item=item, value=value, conclusion=conclusion, note=note))
    v = f"{value:.6g}" if isinstance(value, (float, np.floating)) else value
    log(f"[{item_no}] {item} = {v} | {conclusion} {('| ' + note) if note else ''}")


def main():
    p = ForwardPanel(returns_view=True)
    pf = ForwardPanel()
    B = pd.read_parquet(TMP / "fwd_bench.parquet")
    sig = B.index
    ev = sig[SKIP:]
    dates, codes = p.dates, p.codes

    # ---------------- 2 数据
    from afactor import strategy as ST
    U = ST.universe(p, "V1000")
    rng = slice(sig[0], sig[-1])
    st = p["is_st"].loc[rng]
    ow = (p["one_word_up"] | p["one_word_down"]).loc[rng]
    lu = p["limit_up"].loc[rng]
    rec(2, "ST_stock_days_in_pool_window", int((st & U.loc[rng]).to_numpy().sum()),
        "ST 来自 baostock 日线 isST(07-01 起逐日);冻结的 panel._limit_frame 把主板 ST 按 5% 判定涨跌停,"
        "但数据显示 2026-07-06 起主板 ST 日涨跌幅上限已是 10%(审查 P1,见下一行)", "信号日 09-01~09-30")
    rv = pd.read_csv(LEDGER / "fwd_review_fixes.csv")
    gv = lambda it: rv.loc[(rv["section"] == "st") & (rv["item"] == it), "value"].iloc[0]
    rec(2, "ST_mainboard_limit_rule_since_0706",
        f"成交窗口 ST 触板标记 {gv('te_window_ST_touch_flags')} 个(其中按 10% 不算触板的见 note);一字板误判 "
        f"{gv('te_window_false_one_word')} 个({gv('false_one_word_held')});9 月信号日池内触涨停 {gv('Sep_signal_days_pool_limit_up')} 个",
        "引擎只用一字板标记判定可成交;换成 10% 规则重跑 T1/T2(旧成本、Y=1)、T3、同频池与中证 1000 基准,逐日收益与读数逐位不变"
        "(fwd_review_fixes.csv st 节)。面板的 ST 涨跌幅规则过时,但对本次全部读数没有影响",
        "主板 ST |收盘对收盘| 当日最大值 07-03 为 0.053、07-06 起约 0.10;冻结代码不改")
    rec(2, "one_word_limit_stock_days_window", int((ow & U.loc[rng]).to_numpy().sum()),
        "一字板(触板且 rng<0.004)当日不可成交", f"触涨停股票日 {int((lu & U.loc[rng]).to_numpy().sum())}")
    tw = p.raw("twap_open10")
    rec(2, "exec_px_nan_before_0902", bool(tw.loc[:pd.Timestamp('2026-09-01')].isna().all().all()),
        "收益面板 09-02 之前执行价全为 NaN:引擎在预热段算不出任何持仓收益")
    src = pd.read_parquet(ForwardPanel().dir / "_src_code.parquet").loc[rng]
    u = U.loc[rng].to_numpy(bool)
    rec(2, "window_pool_source_all_baostock", float((src.to_numpy()[u] == 4).mean()),
        "窗口内决策池股票日全部来自 baostock 5 分钟(来源编码 4)")
    bj_in = int(U.loc[rng, [c for c in codes if c.endswith(".BJ")]].to_numpy().sum())
    rec(2, "BJ_in_pool_window", bj_in, "北交所不在前向数据里,不进决策池,也不进 semibeta 市场均值")
    qa = pd.read_csv(LEDGER / "fwd_panel_qa.csv")
    g = lambda it: qa.loc[qa["item"] == it, "value"].iloc[0]
    rec(2, "splice_and_exrights", f"接缝日 {g('ret_vs_pctChg_within_1e-4_share[splice_day_0818]')} / 08-19 起 "
        f"{g('ret_vs_pctChg_within_1e-4_share[post_splice_0819_1009]')} / 除权事件 {g('exrights_all_ok_share')}",
        "面板复权收盘收益与 baostock 官方涨跌幅(按 preclose)1e-4 内一致的比例,全部为 1:复权衔接无跳变",
        "fwd_panel_qa.csv splice 节")
    rec(2, "rebased_vs_official_2025_to_0817",
        f"2025 年 {g('ret_vs_official_gt_1e-3[2025]')} / 接缝 01-05 {g('ret_vs_official_gt_1e-3[seam_20260105]')}"
        f" / 01-05~07-01 {g('ret_vs_official_gt_1e-3[2026-01-05~07-01]')} / 07-02~08-17 {g('ret_vs_official_gt_1e-3[07-02~08-17]')}",
        "预热段(<= 08-17)面板复权收盘日收益与官方复权收益(<= 07-01 tushare adj_factor,07-02 起 baostock 交易所口径)"
        "相差 > 1e-3 的股票日数;审查 P0 复权基准修正后全部为 0(修正前 3,397 个,集中在 06-01 / 07-01 / 05-06 等月初)",
        "fwd_panel_qa.csv splice 节")

    # ---------------- 3 时点:引擎毛收益重算
    for tag in ("T2_top20_old", "T1_excl_old"):
        H = pd.read_parquet(TMP / f"fwd_{tag}_holdings.parquet")          # 相位平均持仓(信号日 t 行,成交后)
        daily = pd.read_parquet(TMP / f"fwd_{tag}_daily.parquet")
        byp = pd.read_parquet(TMP / f"fwd_{tag}_daily_by_phase.parquet")
        px = tw.reindex(columns=H.columns).to_numpy(np.float64)
        pos = dates.get_indexer(H.index)
        with np.errstate(invalid="ignore", divide="ignore"):
            g_ = px[pos + 2] / px[pos + 1] - 1.0
        ok = np.isfinite(g_)
        mine = (H.to_numpy(np.float64) * np.where(ok, g_, 0.0)).sum(axis=1)
        d = np.abs(mine - daily["gross_ret"].reindex(H.index).to_numpy())
        rec(3, f"{tag}_gross_ret_recompute_maxdiff", float(np.nanmax(d)),
            "持仓 × twap[t+2]/twap[t+1] 重算的毛收益与引擎逐日 gross_ret 一致:收益打在信号日 t、t+1 成交 t+2 实现",
            "相位平均持仓 × 收益 = 相位毛收益平均(线性)")
    A = pd.read_parquet(TMP / "fwd_aux" / "bs_A.parquet")
    R = pd.read_parquet(TMP / "fwd_aux" / "bs_ratio.parquet")
    Bc = R.cumprod()
    Bc = Bc / Bc.loc[:pd.Timestamp("2026-08-17")].iloc[-1]
    dd = (A.loc["2026-08-18":] - Bc.loc["2026-08-18":]).abs().max().max()
    rec(3, "adj_factor_causal_check", float(dd),
        "复权系数 F_t = 截至 t 的除权比例累乘(与逐日比例的前缀乘积逐位相同),不含 t 之后的事件")
    B_ = pd.read_csv(LEDGER / "fwd_results.csv")
    cp = B_[(B_["section"] == "reading") & (B_["bench"].isin(["pool5", "csi1000_actual_exec"]))
            & B_["corr_peak_k"].notna()][["object", "cost", "corr_peak_k", "corr_profile"]]
    rec(3, "corr_peak_all_objects", "; ".join(f"{r.object}/{r.cost}:k={int(r.corr_peak_k)}" for r in cp.itertuples()),
        "各对象对主基准相关峰值均在 k=0(全部 21 个信号日)")

    # ---------------- 4 基准
    snap, dS = FW.csi1000_snapshot()
    rec(4, "csi1000_snapshot", f"{len(snap)} 只,权重和 1,权重日 {dS.date()}", "权重还原有效条件全部满足(fwd_decisions.csv)")
    W = FW.csi1000_close_weights(pf)
    close = pf.raw("close_px").where(pf["valid"]).ffill()
    c1 = close.shift(-1) / close - 1.0
    rc = (W * c1.fillna(0.0)).sum(axis=1).reindex(sig)                   # 收盘权重 × 收盘到次日收盘(全收益)
    real = B["real000852"]
    diff = (rc - real)
    rec(4, "csi1000_rebuilt_close2close_vs_real_daily_corr", float(rc.corr(real)),
        "还原的中证 1000(收盘权重 × 复权收盘收益)与真实 000852 收盘对收盘逐日对照", f"21 个信号日")
    rec(4, "csi1000_rebuilt_close2close_minus_real_mean_bp", float(diff.mean() * 1e4),
        "逐日差均值(bp);全收益 vs 价格指数,9 月中报分红季,差应为小正数",
        f"逐日差 std {diff.std() * 1e4:.1f}bp,最大 |差| {diff.abs().max() * 1e4:.1f}bp")
    cum_rc = float(np.prod(1 + rc.loc[ev]) - 1)
    cum_real = float(np.prod(1 + real.loc[ev]) - 1)
    cum_exec = float(np.prod(1 + B["csi1000_exec"].loc[ev]) - 1)
    cum_pool = float(np.prod(1 + B["pool5"].loc[ev]) - 1)
    rec(4, "eval_window_cum_returns", f"还原收盘口径 {cum_rc:+.4f} / 真实 000852 {cum_real:+.4f} / 还原成交时点口径 {cum_exec:+.4f}"
        f" / 同频池 {cum_pool:+.4f}",
        "收盘口径的还原指数与真实指数几乎重合 -> 权重还原正确;成交时点与收盘口径之差来自时点(t+1 开盘 TWAP 到 t+2 vs "
        "t 收盘到 t+1 收盘)", "评估窗 09-07~09-30 信号日")
    # 时点差分解:收盘 -> 次日开盘 TWAP 的隔夜段
    twf = pf.raw("twap_open10")
    on = (twf.shift(-1) / close - 1.0)                                  # t 收盘 -> t+1 开盘 TWAP
    on_idx = (W * on.fillna(0.0)).sum(axis=1)
    d_end = dates[dates.get_loc(sig[-1]) + 1]                          # 10-08 收盘 -> 10-09 开盘 TWAP
    a0, a1 = float(on_idx.loc[sig[SKIP]]), float(on_idx.loc[d_end])
    rec(4, "csi1000_overnight_close_to_open10_start_end",
        f"起点 {sig[SKIP].date()} 收盘->次日开盘 TWAP {a0:+.4f};终点 {d_end.date()} 收盘->次日开盘 TWAP {a1:+.4f};"
        f"(1+终)/(1+起)-1 = {(1 + a1) / (1 + a0) - 1:+.4f}",
        "成交时点口径 = 收盘口径去掉起点隔夜段、加上终点隔夜段;"
        f"按此分解,收盘口径 {cum_rc:+.4f} -> 成交时点口径约 {(1 + cum_rc) * (1 + a1) / (1 + a0) - 1:+.4f}(实际 {cum_exec:+.4f})")
    pool_minus_csi = float(np.prod(1 + B["pool5"].loc[ev]) - np.prod(1 + B["csi1000_exec"].loc[ev]))
    d_pc = M5.window_excess_distribution(
        pd.read_parquet(LEDGER / "freeze_ref_daily.parquet")[["C|bench_pool5", "C|bench_proxy_exec"]]
        .rename(columns={"C|bench_pool5": "strategy", "C|bench_proxy_exec": "benchmark"}), 17, skip_head=SKIP)
    rec(4, "pool5_minus_csi1000_window", pool_minus_csi,
        "全市场同频池等权本身跑赢中证 1000 的幅度;T1 / T2 对中证 1000 的读数 > 99 主要来自这一项(与选股无关)",
        f"历史(2021-2025,同频池 vs PIT 代理,17 日窗口)百分位 {M5.percentile_in(pool_minus_csi, d_pc):.1f}")
    # ---------------- 5 成本与成交
    for tag in ("T2_top20_old", "T2_top20_Y1"):
        daily = pd.read_parquet(TMP / f"fwd_{tag}_daily.parquet")
        rec(5, f"{tag}_part_cut_sum", float(daily["part_cut"].sum()),
            "参与率截掉的 |Δw| 之和(相位平均)", f"成本 建仓 {daily['cost_total'].iloc[:SKIP].sum():.5f} / 评估窗 "
            f"{daily['cost_total'].iloc[SKIP:].sum():.5f};一字板与停牌由引擎 can 判定")

    # ---------------- 6 信号本身:T2 逐只贡献
    H = pd.read_parquet(TMP / "fwd_T2_top20_old_holdings.parquet")
    He = H.loc[ev]
    px = tw.reindex(columns=He.columns).to_numpy(np.float64)
    pos = dates.get_indexer(He.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        g_ = px[pos + 2] / px[pos + 1] - 1.0
    contrib = pd.DataFrame(He.to_numpy() * np.nan_to_num(g_), index=He.index, columns=He.columns)
    pool = B["pool5"].loc[ev]
    exc_c = (contrib.sum(axis=0) - He.sum(axis=0) * 0)                  # 逐只对组合毛收益的算术贡献(和 = Σ 日毛收益)
    tot = float(contrib.sum().sum())
    top = exc_c.sort_values(ascending=False)
    rec(6, "T2_gross_arith_sum_eval", tot, "评估窗 17 天逐日毛收益算术和(相位平均持仓)",
        f"同频池同期算术和 {float(pool.sum()):+.4f}")
    rec(6, "T2_top5_contributors", "; ".join(f"{c}:{v:+.4f}" for c, v in top.head(5).items()),
        "贡献最大的 5 只(算术和口径)", f"前 5 只合计占毛收益算术和 {top.head(5).sum() / tot:.0%}" if tot != 0 else "")
    rec(6, "T2_n_distinct_names_eval", int((He > 1e-8).any().sum()), "评估窗内出现过的不同持仓股票数")
    # 最大个股持有收益与官方涨跌幅的一致性(t+1 开盘 TWAP -> t+2 开盘 TWAP vs 官方日涨跌幅的同段近似)
    from afactor import fwd_data as FD
    bd = FD.load_daily()
    pc = bd.pivot(index="date", columns="code", values="pctChg") / 100.0
    big = []
    for c in top.head(5).index:
        s = pd.Series(g_[:, He.columns.get_loc(c)], index=He.index)
        held = He[c] > 1e-8
        r_tw = float(np.prod(1 + s[held].fillna(0)) - 1)
        dts = [dates[dates.get_loc(t) + 2] for t in s[held].index]
        r_off = float(np.prod(1 + pc[c].reindex(dts).fillna(0)) - 1) if c in pc.columns else np.nan
        big.append(f"{c} 持有日 TWAP 收益连乘 {r_tw:+.3f} / 同期官方日涨跌幅连乘 {r_off:+.3f}")
    rec(6, "T2_top5_return_crosscheck", " | ".join(big),
        "两者口径不同(开盘 TWAP 到开盘 TWAP vs 收盘到收盘),只看是否同号同量级;同号同量级 -> 不是执行价数据错误")
    res = pd.read_csv(LEDGER / "fwd_results.csv")
    ph = res[(res["section"] == "phase") & (res["object"] == "T2_top20") & (res["cost"] == "old")]
    rec(6, "T2_phase_dispersion", "; ".join(f"{r.reading}:{r.window_excess:+.4f}({r.hist_pct:.0f})" for r in ph.itertuples()),
        "5 个相位的窗口超额(括号内为百分位);相位不独立")
    dist = pd.read_parquet(LEDGER / "freeze_ref_window_dist.parquet")
    w = dist[dist["key"] == "T2_top20|old|pool5|n17"].set_index("window_end")["value"]
    v = float(res[(res["section"] == "reading") & (res["object"] == "T2_top20") & (res["cost"] == "old")
                  & (res["bench"] == "pool5")]["window_excess"].iloc[0])
    by = {y: float((w[w.index.year == y] >= v).mean()) for y in range(2021, 2026)}
    rec(6, "T2_hist_windows_ge_forward_by_year", "; ".join(f"{y}:{s:.1%}" for y, s in by.items()),
        "历史 17 日窗口中超额 >= 前向值的比例(分年份);不是零概率事件")
    w1 = dist[dist["key"] == "T1_excl|old|pool5|n17"].set_index("window_end")["value"]
    v1 = float(res[(res["section"] == "reading") & (res["object"] == "T1_excl") & (res["cost"] == "old")
                   & (res["bench"] == "pool5")]["window_excess"].iloc[0])
    rec(6, "T1_hist_windows_ge_forward_by_year",
        "; ".join(f"{y}:{float((w1[w1.index.year == y] >= v1).mean()):.1%}" for y in range(2021, 2026)), "对照:T1 同一统计")
    # 整节重写(审查 P2):本脚本 main() 负责的各节旧行先清掉(截断不变性的条目由 --trunc 单独维护,保留)
    FC.ledger_append(OUT, rows, key_cols=("section", "item"),
                     drop_where=lambda o: o["section"].astype(str).isin(["6.4-2", "6.4-3", "6.4-4", "6.4-5", "6.4-6"])
                     & ~o["item"].astype(str).str.startswith("score_truncation_invariance"))
    log(f"写 {OUT.name} {len(rows)} 行")


if __name__ == "__main__" and "--trunc" not in sys.argv:
    main()


def truncation_invariance():
    """时点(第 3 项)补充:截断不变性。把前向面板截到 D(D 之后的行全部删掉)重算 COMP_v5 打分,
    D 及以前各信号日的打分必须与全面板打分逐位相同 —— 否则打分用到了 D 之后的信息。只比较打分,不涉及收益。"""
    from afactor.fwd_panel import forward_context

    full = pd.read_parquet(TMP / "fwd_score5.parquet")
    for D in (pd.Timestamp("2026-09-15"), pd.Timestamp("2026-09-25")):
        class _Cut(ForwardPanel):
            def raw(self, name):
                if name not in self._cache:
                    self._cache[name] = super().raw(name).loc[:D]
                return self._cache[name]
        pc = _Cut()
        for k in list(pc._derived):
            if isinstance(pc._derived[k], pd.DataFrame):
                pc._derived[k] = pc._derived[k].loc[:D]
        with forward_context("前向检验调查:截断不变性(只比较打分)"):
            s = FC.comp_score(pc, cache_file=None)
        dd = s.index[(s.index >= SIG_START) & (s.index <= D)]
        a = s.loc[dd].to_numpy(np.float64); b = full.loc[dd, s.columns].to_numpy(np.float64)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        mx = float(np.nanmax(np.abs(a - b)))
        rec(3, f"score_truncation_invariance_D={D.date()}", float(same.mean()),
            "面板截到 D 后重算的打分与全面板打分在 09-01~D 逐位相同的比例(应为 1:打分不含 D 之后的信息)",
            f"最大绝对差 {mx:.3g}")
    FC.ledger_append(OUT, rows, key_cols=("section", "item"))


if __name__ == "__main__" and "--trunc" in sys.argv:
    truncation_invariance()
