"""INC_T2 审查修复 F1:官方中证1000 名单版"上限"外推的证据核对(事后诊断,只报不判,不进任何判定)。

背景:原报告把 "PA1000 的 OV-A 0.17pp / 精确率 0.6 ≈ +0.3pp" 写成官方名单版的"上限"。审查指出这个外推
(1) 假定误报贡献为 0、(2) 借用中证500 的精确率,且与报告自己的 H7 三分位数据矛盾。本脚本用现有数据直接检验这两个假定,
**不下载任何附件、不读 2026 数据**:

  S1  H7 三分位在叠加层口径(A+1 open10 -> E open10,L1b,Y=1,1000 万)下的数字,与审查的独立重算对账。
  S2  事后情景(含选择偏差,**不可交易**):把 PA1000 篮子换成 E−1 竞价量的高 / 中 / 低三分位,在同一个叠加层引擎里跑
      OV-A 10%(1000 万,Y=1),得到"若官方名单 ≈ 高分位"时的年化增量。分组用到 E−1 收盘竞价成交额,而 L1 的出场价就是
      E−1 收盘价,分组变量与结果同期,会把高分位的收益高估;因此它不是上限也不是下限,只是一个情景。
  S3  误报的构成(A 日已知的信息):PA1000 的调入预测里,有多少官方 300/500 调整后名单里的股票;这部分其实是
      "进了更大指数"的真事件,不是纯噪声。按此分区报告 L1 超额与 OV-A。分区只用 A 盘后公开的官方 300/500 名单,
      可交易(A+1 买入),但它不是官方中证1000 名单的替代。

输出:research/ledger/inc_t2_fix_h7_legs.csv、inc_t2_fix_h7_overlay.csv、inc_t2_fix_pa1000_composition.csv
"""
from __future__ import annotations
import os, sys, time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*encountered in matmul")     # macOS Accelerate 的伪警告,数值已用 statsmodels 复核

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP, LEDGER
from afactor.panel import Panel
from afactor import strategy as ST, impact as IM
from inc_t2_common import summarize, WINDOWS
import inc_t2_overlay as OV


def tag_events():
    E = pd.read_parquet(TMP / "inc_t2_events.parquet")
    M = pd.read_parquet(TMP / "inc_t2_mechanism_events.parquet")[["ev_id", "lnauc_adv_Em1_diff"]]
    e = E[(E["source"] == "PA") & (E["index"] == "000852") & (E["direction"] == "in") & (E["etype"] == "ADD_pure")
          & E["q_A"] & E["in_panel"]].merge(M, on="ev_id", how="left")
    # H7 同款三分位(逐期 rank,仅 dev / test 窗内有有限值的事件)
    e["terc"] = np.nan
    ok = e["window"].isin(["dev", "test"]) & np.isfinite(e["lnauc_adv_Em1_diff"])
    e.loc[ok, "terc"] = e[ok].groupby("review")["lnauc_adv_Em1_diff"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) + 1 if len(s) >= 3 else np.nan)
    # 官方 300/500 调整后名单(A 盘后公开)
    OM = pd.read_parquet(TMP / "inc_t2_official_members.parquet")
    after = {rv: set(g["code"]) for (rv, w), g in OM[OM["when"] == "after"].groupby(["review", "when"])}
    before = {rv: set(g["code"]) for (rv, w), g in OM[OM["when"] == "before"].groupby(["review", "when"])}
    e["off_after"] = [c in after.get(rv, set()) for rv, c in zip(e["review"], e["code"])]
    e["off_before"] = [c in before.get(rv, set()) for rv, c in zip(e["review"], e["code"])]
    e["grp"] = np.where(e["off_after"] & ~e["off_before"], "官方新调入300/500",
                        np.where(e["off_after"] & e["off_before"], "原本就在官方300/500", "不在官方300/500"))
    return e


def legs_table(e):
    L = pd.read_parquet(TMP / "inc_t2_event_legs.parquet")
    rows = []
    sub = {"L1": L[(L["leg"] == "L1")], "L1b": L[(L["leg"] == "L1b")]}
    for leg, Lg in sub.items():
        x = Lg.merge(e[["ev_id", "terc", "grp"]], on="ev_id", how="inner")
        for part, col in (("tercile", "terc"), ("composition", "grp")):
            for k, g in x[x["window"].isin(["dev", "test"])].groupby(col):
                for wn, ws in (("dev", ["dev"]), ("test", ["test"]), ("all", ["dev", "test"])):
                    gw = g[g["window"].isin(ws)]
                    if gw.empty:
                        continue
                    s1 = summarize(gw["ar_matched"].to_numpy(), gw["review"].to_numpy())
                    s2 = summarize(gw["net_idx_Y1_10000000"].to_numpy(), gw["review"].to_numpy())
                    rows.append(dict(leg=leg, partition=part, group=k, window=wn, n=len(gw),
                                     gross=s1.get("mean"), t_gross=s1.get("t_cl"),
                                     net_Y1_1e7=s2.get("mean"), t_net=s2.get("t_cl"),
                                     pos_periods=s2.get("n_pos_periods"), n_periods=s2.get("n_periods")))
    T = pd.DataFrame(rows)
    T.to_csv(LEDGER / "inc_t2_fix_h7_legs.csv", index=False)
    return T


def composition_table(e):
    rows = []
    for wn, ws in (("dev", ["dev"]), ("test", ["test"]), ("all", ["dev", "test"])):
        g = e[e["window"].isin(ws)]
        n = len(g)
        for k, gg in g.groupby("grp"):
            rows.append(dict(window=wn, group=k, n=len(gg), share=len(gg) / n))
    C = pd.DataFrame(rows)
    C.to_csv(LEDGER / "inc_t2_fix_pa1000_composition.csv", index=False)
    return C


def overlay_scenarios(p, e):
    m = OV.sim_mask(p)
    W = np.where(m[:, None], OV.baseline_W(p, m), 0.0)
    cm = OV.ConstAUMCost(IM.CostModel(p, impact_y=1.0))
    B0 = pd.read_parquet(OV.RUN_DIR / "B0_10000000.parquet")["ret"]
    exb = None
    sets = {
        "全部 PA1000 调入(对账:应等于账本 OV-A 10% PA1000)": e,
        "E−1 竞价量 高分位(事后,不可交易)": e[e["terc"] == 3],
        "E−1 竞价量 中分位(事后,不可交易)": e[e["terc"] == 2],
        "E−1 竞价量 低分位(事后,不可交易)": e[e["terc"] == 1],
        "E−1 竞价量 中+高分位(事后,不可交易)": e[e["terc"].isin([2, 3])],
        "剔除官方新调入300/500(A 日可知,可交易)": e[e["grp"] != "官方新调入300/500"],
        "只含官方新调入300/500(A 日可知,可交易)": e[e["grp"] == "官方新调入300/500"],
        "只含不在官方300/500(A 日可知,可交易)": e[e["grp"] == "不在官方300/500"],
    }
    out, daily = [], {}
    for name, g in sets.items():
        t1 = time.time()
        wins = []
        for rv, gg in g.groupby("review"):
            js = np.unique(gg["j"].to_numpy()).astype(np.int64)
            wins.append(dict(review=rv, A=int(gg["A_i"].iloc[0]), Em1=int(gg["Em1_i"].iloc[0]), add=js,
                             dele=np.array([], dtype=np.int64)))
        R, D = OV.run_cell(p, W, m, cm, 1e7, wins, "A", 0.10)
        d = R - B0.reindex(R.index)
        daily[name] = d
        for wn, (a, b) in WINDOWS.items():
            sel = (d.index >= a) & (d.index <= b)
            s = OV.nw_stats(d[sel])
            out.append(dict(basket=name, window=wn, n_events=len(g[g["window"].isin(
                {"dev": ["dev"], "test": ["test"], "all": ["dev", "test"]}[wn])]),
                incr_ann_pp=s["ann"] * 100, incr_t_nw=s["t"], incr_lo_pp=s["lo"] * 100, incr_hi_pp=s["hi"] * 100))
        print(f"  {name:48s} all {out[-1]['incr_ann_pp']:+.3f}pp t {out[-1]['incr_t_nw']:+.2f}  {time.time()-t1:.0f}s", flush=True)
    O = pd.DataFrame(out)
    O.to_csv(LEDGER / "inc_t2_fix_h7_overlay.csv", index=False)
    pd.DataFrame(daily).to_parquet(TMP / "inc_t2_fix_h7_daily.parquet")
    return O


def main():
    t00 = time.time()
    e = tag_events()
    print(f"PA1000 调入事件(合格池内) {len(e)};分组:\n{e['grp'].value_counts().to_string()}", flush=True)
    T = legs_table(e)
    C = composition_table(e)
    pd.set_option("display.width", 220)
    print(C.round(3).to_string())
    print(T[(T["window"] == "all")].round(4).to_string())
    p = Panel()
    O = overlay_scenarios(p, e)
    print(O.round(3).to_string())
    print(f"done {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
