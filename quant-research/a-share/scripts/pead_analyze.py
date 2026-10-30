"""PEAD 第 5 步:账户层汇总、必要比较、判定、成本与盈亏平衡、Barra 辅助归因。

输入:cache/_tmp/pead_daily_<scenario>.parquet(scripts/pead_run_arms.py 产出)。
口径(预注册第 3、7 节):
  - 臂收益:B / C / B_F = 3 个复制组合日收益平均;E_rand = 10 次抽样平均;每个相位是独立账户;
    主汇总 = 5 个相位日收益等权平均(只是汇总统计,不代表 500 万资金),另逐相位报告;
  - 比较序列 d_t = r_X,t − r_Y,t(相位平均后作差,两臂都已扣各自成本);年化 = 日均 × 242;
    NW(Bartlett)带宽 10 的 95% 区间;稳健性:日历季度为独立单位的 t;逐年、逐相位另报;
  - 窗口按收益索引日(信号日):开发 2021-01-01~2023-12-31,检验 2024-01-01~2025-12-31,2020 只作预热;
  - 判定(主情景 base_1m):支持 = 检验窗下界 > 0 且开发窗点估计 > 0;否定 = 检验窗上界 <= 0 或两窗点估计都 <= 0;
    其余 = 证据不足。门控:E−B 不获支持则 E−C、F−E_rand 只报数、不判"支持";
  - 盈亏平衡:c*_EB = mean(d) / mean(TV_E),c*_EB,both = mean(d) / mean(TV_E − TV_B)(TV = 买+卖成交额/净值,单位:每边 bp)。
产出:research/ledger/pead_arms.csv、pead_phases.csv、pead_compare.csv、pead_costs.csv、pead_barra.csv、pead_verdict.csv、
     pead_exec_stats.csv(执行计数器汇总)、pead_event_account_recon.csv(事件层 vs 账户层对账;后两者为审查 P2 补充)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP, LEDGER
from afactor.panel import Panel
from afactor import strategy as ST, earnings_events as EE, pead_engine as PE
from pead_engine_gate import pool_benchmark

TD = ST.TRADING_DAYS
SCEN = ["base_1m", "stress_1m", "base_10m", "stress_10m", "nomin_1m", "nolot_1m", "imp_1m", "imp_10m", "imp_100m"]
SCEN_DESC = {"base_1m": "100万 基准费率(主情景)", "stress_1m": "100万 +5bp 费用压力", "base_10m": "1000万 基准费率",
             "stress_10m": "1000万 +5bp", "nomin_1m": "100万 无最低佣金(诊断)",
             "nolot_1m": "100万 无整手无最低佣金(诊断)", "imp_1m": "100万 冲击模型 Y=1",
             "imp_10m": "1000万 冲击模型 Y=1", "imp_100m": "1亿 冲击模型 Y=1"}
ARMS = ["E", "F", "B", "C", "B_F", "E_rand"]
PAIRS = [("E", "B"), ("E", "C"), ("F", "E_rand"), ("F", "E"), ("F", "B_F"), ("E", "pool"), ("B", "pool"),
         ("C", "pool"), ("F", "pool")]
WIN = {"dev": ("2021-01-01", "2023-12-31"), "test": ("2024-01-01", "2025-12-31"),
       "all2125": ("2021-01-01", "2025-12-31"), "warmup2020": ("2020-01-01", "2020-12-31")}
NW_L = 10
COST_COLS = ["cost_total", "cost_commission", "cost_stamp", "cost_min", "cost_stress", "cost_spread_cm",
             "cost_impact_cm"]


def wslice(s, w):
    a, b = WIN[w]
    return s.loc[(s.index >= a) & (s.index <= b)]


def load(sc):
    D = pd.read_parquet(TMP / f"pead_daily_{sc}.parquet")
    D["date"] = pd.to_datetime(D["date"])
    # 目标仓位(S 时点,整手前/后)沿持有期前向填充:每个账户在自己的调仓日更新
    D = D.sort_values(["arm", "rep", "phase", "date"])
    for c in ("tgt_pre", "tgt_post"):
        D[c + "_ff"] = D.groupby(["arm", "rep", "phase"])[c].ffill().fillna(0.0)
    return D


def arm_frames(D, col):
    """返回 (phase 层 DataFrame[date x (arm, phase)], 相位平均 DataFrame[date x arm])。"""
    R = D.pivot_table(index="date", columns=["arm", "phase", "rep"], values=col, aggfunc="first")
    ap = R.T.groupby(level=["arm", "phase"]).mean().T
    aa = ap.T.groupby(level="arm").mean().T
    return ap, aa


def compare_rows(sc, X, Y, rx, ry, label_extra=None):
    d = (rx - ry).dropna()
    out = []
    for w in ("dev", "test", "all2125", "warmup2020"):
        x = wslice(d, w)
        ci = PE.mean_ci(x.to_numpy(), NW_L, ann=TD)
        qt = PE.quarterly_t(x)
        out.append(dict(scenario=sc, comparison=f"{X}-{Y}", slice=w, n_days=ci["n"], est_ann=ci["est"],
                        se_ann=ci["se"], t_nw=ci["t"], lo95=ci["lo"], hi95=ci["hi"], **qt, **(label_extra or {})))
    for y, g in d.groupby(d.index.year):
        ci = PE.mean_ci(g.to_numpy(), NW_L, ann=TD)
        out.append(dict(scenario=sc, comparison=f"{X}-{Y}", slice=f"y{y}", n_days=ci["n"], est_ann=ci["est"],
                        se_ann=ci["se"], t_nw=ci["t"], lo95=ci["lo"], hi95=ci["hi"], **(label_extra or {})))
    return out


def verdict(rows, comp):
    dv = [r for r in rows if r["comparison"] == comp and r["slice"] == "dev"][0]
    ts = [r for r in rows if r["comparison"] == comp and r["slice"] == "test"][0]
    if ts["lo95"] > 0 and dv["est_ann"] > 0:
        return "支持", dv, ts
    if ts["hi95"] <= 0 or (dv["est_ann"] <= 0 and ts["est_ann"] <= 0):
        return "否定", dv, ts
    return "证据不足", dv, ts


def main():
    t0 = time.time()
    p = Panel()
    Q = EE.qualified_pool(p)
    bench = pool_benchmark(p, Q)
    arms_rows, phase_rows, comp_rows, cost_rows, verdict_rows = [], [], [], [], []
    main_series = {}
    for sc in SCEN:
        D = load(sc)
        rp, ra = arm_frames(D, "ret")
        ra["pool"] = bench.reindex(ra.index)
        frames = {c: arm_frames(D, c) for c in ["exposure", "tv", "tgt_pre_ff", "tgt_post_ff", "stuck_w",
                                                "n_hold", "part_cut"] + COST_COLS}
        st = pd.read_parquet(TMP / f"pead_stats_{sc}.parquet")
        reb = D[D["is_reb"] & (D["n_sel"] > 0)]
        if sc == "base_1m":
            main_series = dict(ra=ra.copy(), rp=rp.copy(), tv=frames["tv"][1].copy())
        # ---------- 臂(相位平均)----------
        for arm in ARMS:
            r = ra[arm]
            cs = {k: float(r.corr(ra["pool"].shift(k))) for k in (-2, -1, 0, 1, 2)}
            peak = max(cs, key=cs.get)
            for w in ("dev", "test", "all2125", "warmup2020"):
                x = wslice(r, w)
                rec = dict(scenario=sc, scenario_desc=SCEN_DESC[sc], arm=arm, window=w, n_days=len(x),
                           ann_return=PE.ann_return(x), ann_arith=float(x.mean() * TD),
                           vol=float(x.std() * np.sqrt(TD)), max_dd=PE.max_dd(x),
                           pool_ann_return=PE.ann_return(wslice(ra["pool"], w)),
                           excess_vs_pool_arith=float((x - wslice(ra["pool"], w)).mean() * TD),
                           corr_peak_k=peak, corr_k0=cs[0])
                for c, (fp, fa) in frames.items():
                    v = wslice(fa[arm], w)
                    rec[c + "_mean"] = float(v.mean())
                rec["cash_actual"] = 1 - rec.pop("exposure_mean")
                rec["cash_target_S"] = 1 - rec.pop("tgt_pre_ff_mean")
                rec["cash_target_after_lots"] = 1 - rec.pop("tgt_post_ff_mean")
                rec["turnover_ann"] = rec["tv_mean"] * TD
                for c in COST_COLS:
                    rec[c + "_ann_pp"] = rec.pop(c + "_mean") * TD * 100
                rr = reb[(reb["arm"] == arm) & (reb["date"] >= WIN[w][0]) & (reb["date"] <= WIN[w][1])]
                rec["reb_active_n"] = len(rr)
                rec["reb_tgt_pre_mean"] = float(rr["tgt_pre"].mean())
                rec["reb_tgt_post_mean"] = float(rr["tgt_post"].mean())
                rec["reb_lot_drag_mean"] = float((rr["tgt_pre"] - rr["tgt_post"]).mean())
                rec["reb_untradable_T_mean"] = float(rr["tgt_untradable"].mean())
                rec["reb_exposure_after_mean"] = float(rr["exposure"].mean())
                rec["reb_zero_lot_share"] = float(rr["n_zero_lot"].sum() / max(rr["n_sel"].sum(), 1))
                sa = st[st["arm"] == arm]
                rec["pending_stock_days_per_acct_fullperiod"] = float(sa["pending_stock_days"].mean())
                rec["pending_days_per_acct_fullperiod"] = float(sa["pending_days"].mean())
                arms_rows.append(rec)
            # ---------- 逐相位 ----------
            for ph in range(5):
                rph = rp[(arm, ph)]
                for w in ("dev", "test", "all2125"):
                    x = wslice(rph, w)
                    phase_rows.append(dict(scenario=sc, arm=arm, phase=ph, window=w, ann_return=PE.ann_return(x),
                                           ann_arith=float(x.mean() * TD), max_dd=PE.max_dd(x),
                                           cash_actual=float(1 - wslice(frames["exposure"][0][(arm, ph)], w).mean()),
                                           turnover_ann=float(wslice(frames["tv"][0][(arm, ph)], w).mean() * TD),
                                           cost_ann_pp=float(wslice(frames["cost_total"][0][(arm, ph)], w).mean() * TD * 100)))
        # ---------- 比较 ----------
        rows_sc = []
        for X, Y in PAIRS:
            rows_sc += compare_rows(sc, X, Y, ra[X], ra[Y], dict(phase="avg"))
            if Y != "pool":
                for ph in range(5):
                    d = (rp[(X, ph)] - rp[(Y, ph)]).dropna()
                    for w in ("dev", "test"):
                        x = wslice(d, w)
                        ci = PE.mean_ci(x.to_numpy(), NW_L, ann=TD)
                        phase_rows.append(dict(scenario=sc, arm=f"{X}-{Y}", phase=ph, window=w,
                                               ann_arith=ci["est"], t_nw=ci["t"], lo95=ci["lo"], hi95=ci["hi"]))
        comp_rows += rows_sc
        # ---------- 判定 ----------
        vEB, dv, ts = verdict(rows_sc, "E-B")
        for comp in ("E-B", "E-C", "F-E_rand"):
            v, dv, ts = verdict(rows_sc, comp)
            gated = comp != "E-B" and vEB != "支持"
            verdict_rows.append(dict(scenario=sc, comparison=comp, raw_verdict=v,
                                     final=(f"未进入(门控:E−B 为{vEB});原始判据 {v}" if gated else v),
                                     dev_est=dv["est_ann"], dev_lo=dv["lo95"], dev_hi=dv["hi95"], dev_t=dv["t_nw"],
                                     test_est=ts["est_ann"], test_lo=ts["lo95"], test_hi=ts["hi95"], test_t=ts["t_nw"],
                                     dev_q_t=dv.get("q_t"), test_q_t=ts.get("q_t")))
        # ---------- 成本与盈亏平衡 ----------
        tv = frames["tv"][1]
        for X, Y in (("E", "B"), ("E", "C"), ("F", "E_rand")):
            d = ra[X] - ra[Y]
            for w in ("dev", "test", "all2125"):
                md = float(wslice(d, w).mean())
                tx, ty = float(wslice(tv[X], w).mean()), float(wslice(tv[Y], w).mean())
                c1 = md / tx * 1e4 if md > 0 and tx > 0 else np.nan
                c2 = md / (tx - ty) * 1e4 if md > 0 and (tx - ty) > 0 else np.nan
                cost_rows.append(dict(scenario=sc, kind="breakeven", comparison=f"{X}-{Y}", window=w,
                                      mean_d_ann_pp=md * TD * 100, tv_X_daily=tx, tv_Y_daily=ty,
                                      c_star_X_only_bp=c1, c_star_both_bp=c2,
                                      note=("无盈亏平衡空间(mean(d) <= 0)" if md <= 0 else
                                            ("" if (tx - ty) > 0 else "both 版本分母 <= 0:不适用"))))
        for arm in ARMS:
            for w in ("dev", "test", "all2125"):
                rec = dict(scenario=sc, kind="cost", comparison=arm, window=w,
                           tv_X_daily=float(wslice(tv[arm], w).mean()))
                for c in COST_COLS:
                    rec[c + "_ann_pp"] = float(wslice(frames[c][1][arm], w).mean() * TD * 100)
                rec["cost_per_turnover_bp"] = (rec["cost_total_ann_pp"] / 100 / TD) / rec["tv_X_daily"] * 1e4 \
                    if rec["tv_X_daily"] > 0 else np.nan
                cost_rows.append(rec)
        print(f"[{sc}] done [{time.time()-t0:.0f}s]", flush=True)
        del D, frames

    A = pd.DataFrame(arms_rows); A.to_csv(LEDGER / "pead_arms.csv", index=False)
    P = pd.DataFrame(phase_rows); P.to_csv(LEDGER / "pead_phases.csv", index=False)
    C = pd.DataFrame(comp_rows); C.to_csv(LEDGER / "pead_compare.csv", index=False)
    K = pd.DataFrame(cost_rows); K.to_csv(LEDGER / "pead_costs.csv", index=False)
    V = pd.DataFrame(verdict_rows)
    # 费用敏感标注(预注册 7.3 补充规则)
    st_test = C[(C.scenario == "stress_1m") & (C.comparison == "E-B") & (C.slice == "test")]["est_ann"].iloc[0]
    V["fee_sensitive_flag"] = np.where((V.scenario == "base_1m") & (V.comparison == "E-B") & (V.raw_verdict == "支持")
                                       & (st_test <= 0), "支持(费用敏感)", "")
    V.to_csv(LEDGER / "pead_verdict.csv", index=False)
    print(V[["scenario", "comparison", "final", "dev_est", "test_est", "test_lo", "test_hi"]].to_string())

    # ---------- Barra 辅助归因(只描述,不改判定)----------
    barra(main_series)
    # ---------- 审查 P2 补充:执行统计落 ledger、事件层 vs 账户层对账 ----------
    exec_stats()
    event_account_recon(p, C)
    print(f"完成 [{time.time()-t0:.0f}s]")


def exec_stats():
    """各情景 × 臂的执行统计(pead_engine.make_batch_target 的计数器),从 cache/_tmp/pead_stats_*.parquet 汇总:
      px_fallback        未复权收盘缺失、改用前复权收盘换算手数的次数(股票-次;含继续持有股的"一手价值"计算)
      pending_stock_days 应退出但没卖掉、之后逐日重试的股票日;pending_days 有此类持仓的交易日
      cont_hold / cont_trade 新一批仍入选的继续持有股:目标差不足一手而不交易 / 照常调整 的次数
    sum = 该臂全部账户(复制组合 × 相位)合计;per_acct = 每账户平均(全期)。"""
    cols = ["px_fallback", "pending_stock_days", "pending_days", "cont_hold", "cont_trade"]
    rows = []
    for sc in SCEN:
        st = pd.read_parquet(TMP / f"pead_stats_{sc}.parquet")
        for arm, g in [(a, st[st["arm"] == a]) for a in ARMS] + [("ALL", st)]:
            rec = dict(scenario=sc, arm=arm, n_accounts=len(g))
            for c in cols:
                rec[c + "_sum"] = int(g[c].sum())
                rec[c + "_per_acct"] = float(g[c].mean())
            rows.append(rec)
    X = pd.DataFrame(rows)
    X.to_csv(LEDGER / "pead_exec_stats.csv", index=False)
    print(X[X.arm == "ALL"].to_string())


def event_account_recon(p, C: pd.DataFrame):
    """事件层(理论 h5 收益)与账户层 E−B / E−C 的对账(审查 P2,只描述,不改判定)。

    为什么需要:pead_event_level.csv 的 h5 配对超额是"每批等权"(每个成交日 T 一个观测),账户层是资金加权
    (财报季的大批次一次几百只、满仓;平时的小批次只有几只、每只 5% 上限、大部分是现金)。两种平均口径
    可以符号相反。这里用同一批理论收益,换成账户层的权重口径重算:
      weight_ann_pp = Σ_T Σ_i w_i·e_i / 5 / 窗口交易日数 × 242 × 100(w = E 槽位 S 时点目标权重,整手前;
                      除以 5 = 5 个相位账户平均;每批 5 日持有收益记在其信号日上,非调仓日记 0,NW 带宽 10)。
      per_T_equal_mean = 每批等权平均(= pead_event_level.csv 的 paired_excess_per_T),单位:每批收益(小数)。
    批次按 n_sel >= 20 / < 20 拆分:w = min(1/n_sel, 5%),n_sel >= 20 时 5% 上限不再约束、S 时点目标满仓,
    这是口径本身的分界,不是看收益后挑的阈值。理论收益不含成本、整手、参与率与逐日重试,不能替代账户层判定。"""
    E = pd.read_parquet(TMP / "pead_event_level_slots.parquet",
                        columns=["T_pos", "slot", "t_signal", "r5", "e5_B", "e5_C"])
    W = pd.read_parquet(TMP / "pead_batches.parquet", columns=["arm", "T_pos", "slot", "w"])
    W = W[W["arm"] == "E"][["T_pos", "slot", "w"]]
    E = E.merge(W, on=["T_pos", "slot"], how="left", validate="one_to_one")
    assert E["w"].notna().all()
    E["n_sel"] = E.groupby("T_pos")["slot"].transform("size")
    E["big"] = E["n_sel"] >= 20
    dates = p.dates
    rows = []
    for ctrl in ("B", "C"):
        e = f"e5_{ctrl}"
        for w in ("dev", "test", "all2125"):
            a, b = WIN[w]
            x = E[(E["t_signal"] >= a) & (E["t_signal"] <= b)]
            days = dates[(dates >= a) & (dates <= b)]
            wsum = float(x["w"].sum())
            for sub, m in (("all", np.ones(len(x), bool)), ("n_sel>=20", x["big"].to_numpy()),
                           ("n_sel<20", ~x["big"].to_numpy())):
                xs = x[m]
                base = dict(control=ctrl, window=w, subset=sub, n_T=int(xs["T_pos"].nunique()),
                            n_slots=len(xs), weight_share=float(xs["w"].sum()) / wsum if wsum > 0 else np.nan)
                perT = xs.groupby("t_signal")[e].mean()
                ci = PE.mean_ci(perT.to_numpy(), NW_L)
                rows.append(dict(base, kind="per_T_equal_mean", est=ci["est"], lo95=ci["lo"], hi95=ci["hi"],
                                 t_nw=ci["t"]))
                ci = PE.mean_ci(xs[e].to_numpy(), NW_L)
                rows.append(dict(base, kind="slot_equal_mean", est=ci["est"]))
                D = ((xs[e] * xs["w"]).groupby(xs["t_signal"]).sum()
                     .reindex(days, fill_value=0.0) / 5.0)
                ci = PE.mean_ci(D.to_numpy(), NW_L, ann=TD)
                rows.append(dict(base, kind="weight_ann_pp", est=ci["est"] * 100, lo95=ci["lo"] * 100,
                                 hi95=ci["hi"] * 100, t_nw=ci["t"]))
            for sc in ("base_1m", "nolot_1m"):
                r = C[(C["scenario"] == sc) & (C["comparison"] == f"E-{ctrl}") & (C["slice"] == w)
                      & (C["phase"] == "avg")]
                if len(r):
                    r = r.iloc[0]
                    rows.append(dict(control=ctrl, window=w, subset="all", kind=f"account_ann_pp:{sc}",
                                     est=r["est_ann"] * 100, lo95=r["lo95"] * 100, hi95=r["hi95"] * 100,
                                     t_nw=r["t_nw"]))
    R = pd.DataFrame(rows)
    R.to_csv(LEDGER / "pead_event_account_recon.csv", index=False)
    print(R.to_string())


def barra(ms):
    F = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0)
    F.index = pd.to_datetime(F.index)
    F = F["open10"].astype(float)
    styles = [c for c in F.columns if not c.startswith("行业:") and c != "Country"]
    indus = [c for c in F.columns if c.startswith("行业:")]
    ra = ms["ra"]
    rows = []
    series = {"E-B": ra["E"] - ra["B"], "E-C": ra["E"] - ra["C"], "F-E_rand": ra["F"] - ra["E_rand"],
              "E-pool": ra["E"] - ra["pool"]}
    for name, y in series.items():
        idx = y.dropna().index.intersection(F.dropna(how="any").index)
        y = y.reindex(idx); Fi = F.reindex(idx)
        for spec, wins in (("fixed", ["all2125", "dev", "test"]), ("yearly", ["all2125"])):
            for w in wins:
                yw = wslice(y, w)
                if spec == "fixed":
                    parts = [(w, yw)]
                else:
                    parts = [(f"y{yy}", g) for yy, g in yw.groupby(yw.index.year)]
                tot_alpha, tot_sty, tot_ind, tot_n = [], [], [], 0
                resid_all = []
                for lab, yy in parts:
                    Fw = Fi.reindex(yy.index)
                    X = np.column_stack([np.ones(len(yy)), Fw[styles + indus].to_numpy()])
                    b, *_ = np.linalg.lstsq(X, yy.to_numpy(), rcond=None)
                    res = yy.to_numpy() - X @ b
                    sty = sum(b[1 + k] * Fw[s].mean() for k, s in enumerate(styles))
                    ind = sum(b[1 + len(styles) + k] * Fw[s].mean() for k, s in enumerate(indus))
                    tot_alpha.append(b[0] * len(yy)); tot_sty.append(sty * len(yy)); tot_ind.append(ind * len(yy))
                    tot_n += len(yy); resid_all.append(res + b[0])
                    if spec == "fixed":
                        for k, s in enumerate(styles):
                            rows.append(dict(series=name, spec=spec, window=w, term=s, beta=b[1 + k],
                                             contrib_ann_pp=b[1 + k] * Fw[s].mean() * TD * 100))
                ar = np.concatenate(resid_all)
                se = PE.nw_se(ar, NW_L)
                rows.append(dict(series=name, spec=spec, window=w, term="__alpha",
                                 contrib_ann_pp=sum(tot_alpha) / tot_n * TD * 100,
                                 t_nw=float(ar.mean() / se) if se > 0 else np.nan))
                rows.append(dict(series=name, spec=spec, window=w, term="__风格合计",
                                 contrib_ann_pp=sum(tot_sty) / tot_n * TD * 100))
                rows.append(dict(series=name, spec=spec, window=w, term="__行业合计",
                                 contrib_ann_pp=sum(tot_ind) / tot_n * TD * 100))
                rows.append(dict(series=name, spec=spec, window=w, term="__总差",
                                 contrib_ann_pp=float(yw.mean() * TD * 100), n_days=int(tot_n)))
    B = pd.DataFrame(rows)
    B.to_csv(LEDGER / "pead_barra.csv", index=False)
    print(B[B.term.str.startswith("__")].to_string())


if __name__ == "__main__":
    main()
