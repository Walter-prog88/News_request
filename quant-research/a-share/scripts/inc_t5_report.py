"""T5 统计、判定与账本(前缀 inc_t5;预注册 §9、§10、§13)。

阶段:
  stats    读取 cache/_tmp/inc_t5_runs 的全部回测 -> research/ledger/inc_t5_*.csv(汇总、网格、相关、生成检验、ΔIR 自助、
           P1-P7 判定、安慰剂、变体、稀释、净额、等风险、预期 TE、引擎断言)
  figs     reports/inc_t5_*.png
用法: python3 scripts/inc_t5_report.py stats figs
"""
from __future__ import annotations

import json
import sys
import time

from inc_t5_common import *      # noqa: F401,F403
from inc_t5_common import ROOT, SC, RUNS, TMP, LEDGER, WIN, LE, LHCtx, log, merge_csv, TD

import numpy as np
import pandas as pd

from afactor import inc_t5_stats as S5
from afactor.strategy import lr_variance_ratio
from inc_t5_runs import load_run, STYLE5, STYLE6

OUTD = LEDGER
WINS_LH = ("full", "dev", "test", "dev2", "test2")
BEARS = ("BEAR18", "BEAR22", "BEAR24")


# ================================================================ 载入
class TrackData:
    def __init__(self, track: str):
        self.track = track
        self.dir = RUNS / track
        self.b1 = self._b1()
        self.dates = self.b1.index
        self._cache: dict = {}

    def _b1(self) -> pd.Series:
        f = RUNS / f"{self.track}_b1.pkl"
        if f.exists():
            return pd.read_pickle(f)
        if self.track in ("lh", "a"):
            b = LHCtx(WIN[self.track]["rows_from"]).b1()
        else:
            from inc_t5_runs import MinCtx
            b = MinCtx().b1
        b.to_pickle(f)
        return b

    def run(self, name: str) -> dict:
        if name not in self._cache:
            self._cache[name] = load_run(self.track, name)
        return self._cache[name]

    def r(self, name: str) -> pd.Series:
        return self.run(name)["rets"]

    def ex(self, name: str) -> pd.Series:
        r = self.r(name)
        return (r - self.b1.reindex(r.index)).dropna()

    def names(self) -> list[str]:
        out = []
        for f in sorted(self.dir.glob("*.pkl")):
            out.append(f.stem.replace("__", "|"))
        return out


def win(track: str, w: str) -> tuple[str, str]:
    return WIN[track][w]


# ================================================================ 单配置统计行
def stat_row(td: TrackData, name: str, w: str, rec_cost: pd.DataFrame | None = None) -> dict:
    track = td.track
    a, b = win(track, w)
    rs = LE.slice_eval(td.r(name), a, b)
    rb = LE.slice_eval(td.b1, a, b)
    rs, rb = rs.align(rb, join="inner")
    ps, pb = LE.perf(rs), LE.perf(rb)
    rel = LE.rel_block(rs, rb)
    e = (rs - rb).dropna()
    vr = lr_variance_ratio(e.to_numpy(np.float64), 20)
    uw = S5.rel_underwater(rs, rb)
    y = LE.yearly(rs, rb)
    R = td.run(name)
    row = dict(track=track, config=name, window=w, start=str(rs.index[0].date()), end=str(rs.index[-1].date()),
               n_days=len(rs), s_ann=ps["ann"], s_vol=ps["vol"], s_vol_adj=ps["vol_adj"], s_mdd=ps["mdd"],
               s_calmar=ps["calmar"], s_uw_longest_days=ps["uw_longest_days"], b_ann=pb["ann"], b_mdd=pb["mdd"])
    row.update(rel)
    row["ir_adj"] = rel["ir"] / np.sqrt(vr) if np.isfinite(vr) and vr > 0 and np.isfinite(rel["ir"]) else np.nan
    row["rel_mdd_over_te"] = abs(rel["rel_mdd"]) / rel["te"] if rel["te"] and rel["te"] > 0 else np.nan
    row.update(uw)
    row["year_t"] = LE.year_t(y["excess"]) if len(y) >= 3 else np.nan
    row["n_years"] = len(y)
    row["turnover_ann_run"] = R.get("turnover", np.nan) * TD
    row["cost_ann_run"] = R.get("cost_bps", np.nan) * 1e-4 * TD
    row["holdings_run"] = R.get("holdings", np.nan)
    row["exposure_run"] = R.get("exposure", np.nan)
    return row


def all_rows(td: TrackData, wins: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for n in td.names():
        for w in wins:
            if w not in WIN[td.track]:
                continue
            try:
                rows.append(stat_row(td, n, w))
            except Exception as e:      # noqa: BLE001
                log(f"[stats] {td.track} {n} {w} 失败 {type(e).__name__} {e}")
    return pd.DataFrame(rows)


# ================================================================ 相关矩阵
def corr_tables(td: TrackData, names: dict[str, str]) -> pd.DataFrame:
    """names: 显示名 -> 配置名。日超额 Pearson / Spearman、20 日不重叠、熊市段条件相关。"""
    ex = pd.DataFrame({k: td.ex(v) for k, v in names.items()}).dropna()
    rows = []
    wins = {w: win(td.track, w) for w in ("full", "dev", "test")}
    for w, (a, b) in wins.items():
        x = ex.loc[a:b]
        for kind, c in (("pearson_daily", x.corr()), ("spearman_daily", x.corr(method="spearman")),
                        ("pearson_20d", x.apply(lambda s: S5.nonoverlap_sum(s, 20)).corr())):
            for i in c.index:
                for j in c.columns:
                    rows.append(dict(track=td.track, kind=kind, window=w, a=i, b=j, rho=float(c.loc[i, j])))
    bw = LE.WINDOWS_A if td.track in ("lh", "a") else {"BEAR22": LE.WINDOWS_A["BEAR22"], "BEAR24": LE.WINDOWS_A["BEAR24"]}
    for bn, (a, b) in bw.items():
        if bn not in BEARS:
            continue
        x = ex.loc[a:b]
        if len(x) < 40:
            continue
        c = x.corr()
        for i in c.index:
            for j in c.columns:
                rows.append(dict(track=td.track, kind="pearson_daily_bear", window=bn, a=i, b=j, rho=float(c.loc[i, j])))
    return pd.DataFrame(rows)


# ================================================================ 生成检验
def gen_tests(td: TrackData, y_names: dict[str, str], x_name: str) -> pd.DataFrame:
    rows = []
    xe = td.ex(x_name)
    for yn, cfg in y_names.items():
        ye = td.ex(cfg)
        for w in [w_ for w_ in ("full", "dev", "test") if w_ in WIN[td.track]]:
            a, b = win(td.track, w)
            df = pd.concat([ye, xe], axis=1, keys=["y", "x"]).loc[a:b].dropna()
            t = {q: S5.hac_regress(df["y"].to_numpy(), df["x"].to_numpy(), q) for q in (5, 10, 20)}
            r5 = t[5]
            rows.append(dict(track=td.track, y=yn, x=x_name, window=w, freq="daily", n=r5["n"], a_ann=r5["a"] * TD,
                             t_q5=t[5]["t_a"], t_q10=t[10]["t_a"], t_q20=t[20]["t_a"], b=r5["b"], t_b=r5["t_b"],
                             r2=r5["r2"], direction="y_on_x"))
            # 20 日不重叠
            y20, x20 = S5.nonoverlap_sum(df["y"], 20), S5.nonoverlap_sum(df["x"], 20)
            r20 = S5.hac_regress(y20.to_numpy(), x20.to_numpy(), 1)
            rows.append(dict(track=td.track, y=yn, x=x_name, window=w, freq="20d", n=r20["n"], a_ann=r20["a"] * TD / 20,
                             t_q5=r20["t_a"], t_q10=np.nan, t_q20=np.nan, b=r20["b"], t_b=r20["t_b"], r2=r20["r2"],
                             direction="y_on_x"))
            # 反向:LIQ 对 y
            rv = {q: S5.hac_regress(df["x"].to_numpy(), df["y"].to_numpy(), q) for q in (5,)}[5]
            rows.append(dict(track=td.track, y=yn, x=x_name, window=w, freq="daily", n=rv["n"], a_ann=rv["a"] * TD,
                             t_q5=rv["t_a"], t_q10=np.nan, t_q20=np.nan, b=rv["b"], t_b=rv["t_b"], r2=rv["r2"],
                             direction="x_on_y(reverse)"))
    return pd.DataFrame(rows)


# ================================================================ IR 比较
def ir_of(td: TrackData, name: str, w: str) -> float:
    a, b = win(td.track, w)
    rs, rb = LE.slice_eval(td.r(name), a, b), LE.slice_eval(td.b1, a, b)
    return LE.rel_block(rs, rb)["ir"]


def dir_table(td: TrackData, cmp_names: list[str], base: str, boot_names: list[str]) -> pd.DataFrame:
    rows = []
    for n in cmp_names:
        for w in ("full", "dev", "test"):
            a, b = win(td.track, w)
            ir_n, ir_b = ir_of(td, n, w), ir_of(td, base, w)
            row = dict(track=td.track, config=n, window=w, ir=ir_n, ir_base=ir_b, d_ir=ir_n - ir_b)
            if n in boot_names:
                e1, e0 = td.ex(n).loc[a:b], td.ex(base).loc[a:b]
                m = e1.index.intersection(e0.index)
                bb = S5.block_bootstrap_dir(e1.loc[m].to_numpy(), e0.loc[m].to_numpy(), 20, 2000, 0)
                row.update(boot_lo90=bb["lo90"], boot_hi90=bb["hi90"], boot_p_pos=bb["p_pos"])
            rows.append(row)
    return pd.DataFrame(rows)


# ================================================================ B-LH 主判定
def pass_lh(td: TrackData, out: dict) -> dict:
    tr = "lh"
    liq = "S|LIQ|20_35"
    cmb = "C|EW5|45_60"
    irs = {w: (ir_of(td, cmb, w), ir_of(td, liq, w)) for w in WIN_NAMES}
    P = {}
    P["P1"] = all(irs[w][0] > irs[w][1] for w in WIN_NAMES) and (irs["full"][0] - irs["full"][1]) >= 0.10
    P["dIR_full"] = irs["full"][0] - irs["full"][1]
    gt = out["gen"]
    g = gt[(gt.track == tr) & (gt.y == "S4") & (gt.freq == "daily") & (gt.direction == "y_on_x")].set_index("window")
    P["a_full"] = float(g.loc["full", "a_ann"])
    P["t_full"] = float(g.loc["full", "t_q5"])
    P["P2"] = bool(g.loc["full", "a_ann"] > 0 and g.loc["full", "t_q5"] >= 2.0 and g.loc["dev", "a_ann"] > 0 and g.loc["test", "a_ann"] > 0)
    sf = {w: stat_row(td, n, "full") for w, n in (("cmb", cmb), ("liq", liq))}
    P["P3"] = bool(sf["cmb"]["rel_uw_longest_days"] <= sf["liq"]["rel_uw_longest_days"] * 1.0
                   and sf["cmb"]["rel_mdd_over_te"] <= sf["liq"]["rel_mdd_over_te"])
    ex = {w: (LE.rel_block(LE.slice_eval(td.r(cmb), *win(tr, w)), LE.slice_eval(td.b1, *win(tr, w)))["excess_arith"],
              LE.rel_block(LE.slice_eval(td.r(liq), *win(tr, w)), LE.slice_eval(td.b1, *win(tr, w)))["excess_arith"])
          for w in WIN_NAMES}
    P["P4"] = bool(ex["full"][0] >= 0.5 * ex["full"][1] and ex["dev"][0] > 0 and ex["test"][0] > 0)
    plc = [ir_of(td, f"P|{s}", "full") - irs["full"][1] for s in range(10)]
    P["P5"] = bool(P["dIR_full"] > max(plc))
    P["placebo_max_dIR"] = float(max(plc))
    var_ok = []
    for sch in ("EW5", "CORE5", "ERC5"):
        for d in ("45_60", "20_35"):
            ok = all(ir_of(td, f"C|{sch}|{d}", w) > ir_of(td, liq, w) for w in WIN_NAMES)
            var_ok.append(ok)
    P["P6"] = bool(sum(var_ok) >= 4)
    P["P6_count"] = int(sum(var_ok))
    P["detail"] = dict(irs=irs, uw_cmb=sf["cmb"]["rel_uw_longest_days"], uw_liq=sf["liq"]["rel_uw_longest_days"],
                       rdd_te_cmb=sf["cmb"]["rel_mdd_over_te"], rdd_te_liq=sf["liq"]["rel_mdd_over_te"], ex=ex)
    return P


def pass_min(tdm: TrackData, out: dict) -> dict:
    liq, cmb = "S|LIQ|20_35", "C|EW5|45_60"
    irs = {w: (ir_of(tdm, cmb, w), ir_of(tdm, liq, w)) for w in WIN_NAMES}
    gt = out["gen"]
    g = gt[(gt.track == "min") & (gt.y == "S4") & (gt.freq == "daily") & (gt.direction == "y_on_x")].set_index("window")
    P = dict(P7_ir=bool(irs["dev"][0] > irs["dev"][1] and irs["test"][0] > irs["test"][1]),
             P7_a=bool(g.loc["dev", "a_ann"] > 0 and g.loc["test", "a_ann"] > 0))
    P["P7"] = P["P7_ir"] and P["P7_a"]
    P["irs"] = irs
    P["a_dev"], P["a_test"], P["a_full"] = float(g.loc["dev", "a_ann"]), float(g.loc["test", "a_ann"]), float(g.loc["full", "a_ann"])
    return P


def pass_a(tda: TrackData, out: dict) -> dict:
    """A 轨子判定:CMB-A2(EW,LOWVOL 窗口 120)对 LIQ,窗口 2006-2019 / 2020-2025。"""
    liq, cmb = "S|LIQ|20_35", "C|A2_120"
    irs = {w: (ir_of(tda, cmb, w), ir_of(tda, liq, w)) for w in WIN_NAMES}
    P = {"P1": all(irs[w][0] > irs[w][1] for w in WIN_NAMES) and (irs["full"][0] - irs["full"][1]) >= 0.10,
         "dIR_full": irs["full"][0] - irs["full"][1]}
    gt = out["gen"]
    g = gt[(gt.track == "a") & (gt.y == "LOWVOL120") & (gt.freq == "daily") & (gt.direction == "y_on_x")].set_index("window")
    P["a_full"], P["t_full"] = float(g.loc["full", "a_ann"]), float(g.loc["full", "t_q5"])
    P["P2"] = bool(g.loc["full", "a_ann"] > 0 and g.loc["full", "t_q5"] >= 2.0 and g.loc["dev", "a_ann"] > 0 and g.loc["test", "a_ann"] > 0)
    sf = {k: stat_row(tda, n, "full") for k, n in (("cmb", cmb), ("liq", liq))}
    P["P3"] = bool(sf["cmb"]["rel_uw_longest_days"] <= sf["liq"]["rel_uw_longest_days"]
                   and sf["cmb"]["rel_mdd_over_te"] <= sf["liq"]["rel_mdd_over_te"])
    ex = {w: (LE.rel_block(LE.slice_eval(tda.r(cmb), *win("a", w)), LE.slice_eval(tda.b1, *win("a", w)))["excess_arith"],
              LE.rel_block(LE.slice_eval(tda.r(liq), *win("a", w)), LE.slice_eval(tda.b1, *win("a", w)))["excess_arith"])
          for w in WIN_NAMES}
    P["P4"] = bool(ex["full"][0] >= 0.5 * ex["full"][1] and ex["dev"][0] > 0 and ex["test"][0] > 0)
    plc = [ir_of(tda, f"P|{s}", "full") - irs["full"][1] for s in range(10)]
    P["P5"] = bool(P["dIR_full"] > max(plc))
    P["placebo_max_dIR"] = float(max(plc))
    ok6 = []
    for sch, tag in (("A2", "EW"), ("COREA2", "CORE")):
        for w in (60, 120, 250):
            ok6.append(all(ir_of(tda, f"C|{sch}_{w}", ww) > ir_of(tda, liq, ww) for ww in WIN_NAMES))
    P["P6"] = bool(sum(ok6) >= 4)
    P["P6_count"] = int(sum(ok6))
    P["detail"] = dict(irs=irs, uw_cmb=sf["cmb"]["rel_uw_longest_days"], uw_liq=sf["liq"]["rel_uw_longest_days"],
                       rdd_te_cmb=sf["cmb"]["rel_mdd_over_te"], rdd_te_liq=sf["liq"]["rel_mdd_over_te"], ex=ex)
    return P


# ================================================================ 引擎断言(D5)
def corr_peak(rs: pd.Series, rb: pd.Series, ks=range(-3, 4)):
    a, b = rs.align(rb, join="inner")
    cs = {int(k): float(a.corr(b.shift(k))) for k in ks}
    return max(cs, key=cs.get), cs


def engine_checks(td: TrackData) -> list[dict]:
    rows = []
    tr = td.track
    a, b = win(tr, "full")
    rec_names = [n for n in td.names() if "rec_daily" in td.run(n)]
    for n in rec_names:
        mx = td.run(n).get("max_exposure_phase", np.nan)
        rows.append(dict(gate="D5", track=tr, item=f"E1_exposure_le_1[{n}]", value=mx, threshold="<= 1+1e-9", ok=bool(mx <= 1 + 1e-9),
                         note="逐相位逐日 exposure 最大值"))
    for n in td.names():
        ex_ = td.run(n).get("exposure", np.nan)
        if np.isfinite(ex_) and ex_ > 1 + 1e-9:
            rows.append(dict(gate="D5", track=tr, item=f"E1_avg_exposure[{n}]", value=ex_, threshold="<= 1", ok=False, note=""))
    for n in ("FP", "S|LIQ|20_35", "C|EW5|45_60", "C|S4|45_60") if tr != "a" else ("FP", "S|LIQ|20_35", "C|A2_120"):
        if n not in td.names():
            continue
        k, cs = corr_peak(LE.slice_eval(td.r(n), a, b), LE.slice_eval(td.b1, a, b))
        rows.append(dict(gate="D5", track=tr, item=f"E2_align_peak_k0[{n}]", value=k, threshold="== 0", ok=bool(k == 0),
                         note="corr(策略, B1.shift(k)): " + ", ".join(f"{kk}:{v:.3f}" for kk, v in cs.items())))
    if "FP" in td.names():
        fp = td.run("FP")
        s, bb = LE.slice_eval(fp["rets"], a, b), LE.slice_eval(td.b1, a, b)
        ex = LE.ann_geo(s) - LE.ann_geo(bb)
        cost = float(LE.slice_eval(fp["rec_daily"]["cost_total"], a, b).mean() * TD)
        rows.append(dict(gate="D5", track=tr, item="E3_fullpool_ew_excess_plus_cost_full", value=ex + cost, threshold="|.| <= 0.003",
                         ok=bool(abs(ex + cost) <= 0.003), note=f"评价期几何超额 {ex * 100:+.2f}pp + 年成本 {cost * 100:.2f}pp"))
        y = LE.yearly(s, bb)
        cc = LE.slice_eval(fp["rec_daily"]["cost_total"], a, b)
        yc = cc.groupby(cc.index.year).sum()
        resid = (y["excess"] + yc.reindex(y.index))
        rows.append(dict(gate="D5", track=tr, item="E3_fullpool_ew_yearly_max_abs", value=float(resid.abs().max()),
                         threshold="report", ok=True, note="逐年 |超额+成本| 最大 " + f"{resid.abs().max() * 100:.2f}pp({resid.abs().idxmax()});"
                         + ";".join(f"{int(i)}:{v * 100:+.2f}" for i, v in resid.items())))
    return rows


# ================================================================ 主流程
def stage_stats():
    t0 = time.time()
    tdl, tda, tdm = TrackData("lh"), TrackData("a"), TrackData("min")
    log(f"[stats] 载入 {time.time() - t0:.0f}s")
    out = {}
    # ---- 汇总与网格 ----
    gl = all_rows(tdl, WINS_LH)
    ga = all_rows(tda, WINS_LH)
    gm = all_rows(tdm, WIN_NAMES)
    grid = pd.concat([gl, ga, gm], ignore_index=True)
    grid.to_csv(OUTD / "inc_t5_grid.csv", index=False)
    main_cfg = {"lh": ["S|LIQ|20_35", "S|LOWVOL|45_60", "S|VALUE|45_60", "S|DIV|45_60", "S|QUAL|45_60", "S|REV|45_60",
                       "C|EW5|45_60", "C|CORE5|45_60", "C|ERC5|45_60", "C|EW6|45_60", "C|S4|45_60", "C|S5|45_60"],
                "a": ["S|LIQ|20_35", "S|LOWVOL|45_60", "S|REV|45_60", "C|A2_120", "C|A3_120", "C|COREA2_120"],
                "min": ["S|LIQ|20_35", "S|LOWVOL|45_60", "S|VALUE|45_60", "S|DIV|45_60", "S|QUAL|45_60", "S|REV|45_60",
                        "C|EW5|45_60", "C|CORE5|45_60", "C|ERC5|45_60", "C|EW6|45_60", "C|S4|45_60"]}
    summ = grid[grid.apply(lambda r: r["config"] in main_cfg[r["track"]], axis=1)]
    summ.to_csv(OUTD / "inc_t5_summary.csv", index=False)
    log(f"[stats] 网格 {len(grid)} 行 {time.time() - t0:.0f}s")

    # ---- 逐年 ----
    yr = []
    for td in (tdl, tda, tdm):
        a, b = win(td.track, "full")
        for n in main_cfg[td.track]:
            y = LE.yearly(LE.slice_eval(td.r(n), a, b), LE.slice_eval(td.b1, a, b))
            for yy, r in y.iterrows():
                yr.append(dict(track=td.track, config=n, year=int(yy), strategy=r["strategy"], bench=r["bench"], excess=r["excess"]))
    pd.DataFrame(yr).to_csv(OUTD / "inc_t5_yearly.csv", index=False)

    # ---- 相关矩阵 ----
    nm_lh = {"LIQ": "S|LIQ|20_35", "LOWVOL": "S|LOWVOL|45_60", "VALUE": "S|VALUE|45_60", "DIV": "S|DIV|45_60",
             "QUAL": "S|QUAL|45_60", "REV": "S|REV|45_60"}
    corr = pd.concat([corr_tables(tdl, nm_lh), corr_tables(tdm, nm_lh), corr_tables(tda, {"LIQ": "S|LIQ|20_35", "LOWVOL": "S|LOWVOL|45_60", "REV": "S|REV|45_60"})],
                     ignore_index=True)
    corr.to_csv(OUTD / "inc_t5_corr.csv", index=False)

    # ---- 生成检验 ----
    ylh = {"S4": "C|S4|45_60", "S5": "C|S5|45_60", "S4_20_35": "C|S4|20_35", "S5_20_35": "C|S5|20_35", "LOWVOL": "S|LOWVOL|45_60",
           "VALUE": "S|VALUE|45_60", "DIV": "S|DIV|45_60", "QUAL": "S|QUAL|45_60", "REV": "S|REV|45_60",
           "LOWVOL_20_35": "S|LOWVOL|20_35", "VALUE_20_35": "S|VALUE|20_35", "DIV_20_35": "S|DIV|20_35",
           "QUAL_20_35": "S|QUAL|20_35", "REV_20_35": "S|REV|20_35"}
    gen_l = gen_tests(tdl, ylh, "S|LIQ|20_35")
    ym = {"S4": "C|S4|45_60", "LOWVOL": "S|LOWVOL|45_60", "VALUE": "S|VALUE|45_60", "DIV": "S|DIV|45_60",
          "QUAL": "S|QUAL|45_60", "REV": "S|REV|45_60"}
    gen_m = gen_tests(tdm, ym, "S|LIQ|20_35")
    ya = {"LOWVOL120": "S|LOWVOL|45_60", "LOWVOL60": "S|LOWVOL60|45_60", "LOWVOL250": "S|LOWVOL250|45_60", "REV20": "S|REV|45_60"}
    gen_a = gen_tests(tda, ya, "S|LIQ|20_35")
    gen = pd.concat([gen_l, gen_m, gen_a], ignore_index=True)
    gen.to_csv(OUTD / "inc_t5_gen_test.csv", index=False)
    out["gen"] = gen
    log(f"[stats] 相关与生成检验 {time.time() - t0:.0f}s")

    # ---- ΔIR(含自助)----
    combos = [f"C|{s}|{d}" for s in ("EW5", "CORE5", "ERC5", "EW6", "CORE6", "ERC6", "S4", "S5") for d in ("45_60", "20_35")]
    stand = [f"S|{b}|{d}" for b in ("LOWVOL", "VALUE", "DIV", "QUAL", "REV") for d in ("45_60", "20_35")]
    plc = [f"P|{s}" for s in range(10)]
    dl = dir_table(tdl, combos + stand + plc, "S|LIQ|20_35", combos)
    cm = [f"C|{s}|45_60" for s in ("EW5", "CORE5", "ERC5", "EW6", "CORE6", "ERC6", "S4")] + [f"S|{b}|45_60" for b in ("LOWVOL", "VALUE", "DIV", "QUAL", "REV")] + [f"P|{s}" for s in range(5)]
    dm = dir_table(tdm, cm, "S|LIQ|20_35", cm[:7])
    ca = [f"C|{s}_{w}" for s in ("A2", "A3", "COREA2") for w in (60, 120, 250)]
    da = dir_table(tda, ca + [f"P|{s}" for s in range(10)] + ["S|LOWVOL|45_60", "S|LOWVOL60|45_60", "S|LOWVOL250|45_60", "S|REV|45_60"],
                   "S|LIQ|20_35", ca)
    dall = pd.concat([dl, dm, da], ignore_index=True)
    dall.to_csv(OUTD / "inc_t5_dir.csv", index=False)
    log(f"[stats] ΔIR 自助 {time.time() - t0:.0f}s")

    # ---- 判定 ----
    Pl = pass_lh(tdl, out)
    Pm = pass_min(tdm, out)
    Pa = pass_a(tda, out)
    qa = pd.read_csv(OUTD / "inc_t5_data_qa.csv")
    chk = []
    for td in (tdl, tda, tdm):
        chk += engine_checks(td)
    chk = pd.DataFrame(chk)
    chk.to_csv(OUTD / "inc_t5_checks.csv", index=False)
    gates_ok = bool(qa["ok"].astype(bool).all() and chk["ok"].astype(bool).all())
    gate_fail = list(pd.concat([qa, chk])[~pd.concat([qa, chk])["ok"].astype(bool)]["item"])
    log(f"[stats] 门槛未过: {gate_fail}")
    pmain = dict(P1=Pl["P1"], P2=Pl["P2"], P3=Pl["P3"], P4=Pl["P4"], P5=Pl["P5"], P6=Pl["P6"], P7=Pm["P7"],
                 dIR_full=Pl["dIR_full"], a_full=Pl["a_full"])
    v_no_c = S5.verdict(pmain, gates_ok)
    pa_ = dict(P1=Pa["P1"], P2=Pa["P2"], P3=Pa["P3"], P4=Pa["P4"], P5=Pa["P5"], P6=Pa["P6"], P7=True,
               dIR_full=Pa["dIR_full"], a_full=Pa["a_full"])
    v_a = S5.verdict(pa_, True)
    rows = [
        dict(item="P1_IR(CMB)>IR(LIQ) 三窗且全期ΔIR>=0.10", value=Pl["dIR_full"], ok=Pl["P1"], detail=json.dumps({k: [round(x, 3) for x in v] for k, v in Pl["detail"]["irs"].items()})),
        dict(item="P2_S4截距a>0 全期t(q5)>=2 且两窗a>0", value=Pl["a_full"], ok=Pl["P2"], detail=f"t_q5={Pl['t_full']:.2f}"),
        dict(item="P3_最长相对水下期与相对回撤/TE不劣于LIQ", value=Pl["detail"]["uw_cmb"], ok=Pl["P3"],
             detail=f"水下 CMB {Pl['detail']['uw_cmb']} vs LIQ {Pl['detail']['uw_liq']};回撤/TE CMB {Pl['detail']['rdd_te_cmb']:.2f} vs LIQ {Pl['detail']['rdd_te_liq']:.2f}"),
        dict(item="P4_全期超额>=0.5xLIQ 且两窗>0", value=Pl["detail"]["ex"]["full"][0], ok=Pl["P4"],
             detail="; ".join(f"{w}: CMB {v[0] * 100:+.2f} / LIQ {v[1] * 100:+.2f}pp" for w, v in Pl["detail"]["ex"].items())),
        dict(item="P5_ΔIR>10个安慰剂最大值", value=Pl["dIR_full"], ok=Pl["P5"], detail=f"安慰剂最大 ΔIR {Pl['placebo_max_dIR']:+.3f}"),
        dict(item="P6_6变体中>=4个三窗IR同向高于LIQ", value=Pl["P6_count"], ok=Pl["P6"], detail=""),
        dict(item="P7_B-MIN IR提高且S4截距a>0(两窗)", value=Pm["a_full"], ok=Pm["P7"],
             detail=f"IR dev {Pm['irs']['dev'][0]:.3f} vs {Pm['irs']['dev'][1]:.3f}; test {Pm['irs']['test'][0]:.3f} vs {Pm['irs']['test'][1]:.3f}; a_dev {Pm['a_dev'] * 100:+.2f}pp a_test {Pm['a_test'] * 100:+.2f}pp"),
        dict(item="GATES_D1-D6", value=int(len(gate_fail)), ok=gates_ok, detail=";".join(gate_fail)),
        dict(item="VERDICT_B-LH_before_C_downgrade", value=v_no_c["verdict"], ok=v_no_c["verdict"] == "有增量", detail=""),
        dict(item="A_P1", value=Pa["dIR_full"], ok=Pa["P1"], detail=json.dumps({k: [round(x, 3) for x in v] for k, v in Pa["detail"]["irs"].items()})),
        dict(item="A_P2", value=Pa["a_full"], ok=Pa["P2"], detail=f"t_q5={Pa['t_full']:.2f}"),
        dict(item="A_P3", value=Pa["detail"]["uw_cmb"], ok=Pa["P3"], detail=f"水下 {Pa['detail']['uw_cmb']} vs {Pa['detail']['uw_liq']};回撤/TE {Pa['detail']['rdd_te_cmb']:.2f} vs {Pa['detail']['rdd_te_liq']:.2f}"),
        dict(item="A_P4", value=Pa["detail"]["ex"]["full"][0], ok=Pa["P4"], detail="; ".join(f"{w}: {v[0] * 100:+.2f}/{v[1] * 100:+.2f}pp" for w, v in Pa["detail"]["ex"].items())),
        dict(item="A_P5", value=Pa["dIR_full"], ok=Pa["P5"], detail=f"安慰剂最大 ΔIR {Pa['placebo_max_dIR']:+.3f}"),
        dict(item="A_P6", value=Pa["P6_count"], ok=Pa["P6"], detail="{EW,CORE}x{60,120,250}"),
        dict(item="A_VERDICT", value=v_a["verdict"], ok=v_a["verdict"] == "有增量", detail=""),
    ]
    pd.DataFrame(rows).to_csv(OUTD / "inc_t5_pass.csv", index=False)
    json.dump(dict(main=pmain, Pl={k: v for k, v in Pl.items() if k != "detail"}, Pm={k: (v if not isinstance(v, dict) else {a: list(b) for a, b in v.items()}) for k, v in Pm.items()},
                   Pa={k: v for k, v in Pa.items() if k != "detail"}, verdict_lh=v_no_c, verdict_a=v_a, gate_fail=gate_fail),
              open(OUTD / "inc_t5_pass.json", "w"), ensure_ascii=False, indent=1, default=float)
    log(f"[stats] 判定(未含 C 轨降级): B-LH {v_no_c}  A 轨 {v_a}")

    # ---- 安慰剂 / 变体 / 稀释 / 净额 / ERC / 预期 TE ----
    ex_rows = []
    for track, td, liq in (("lh", tdl, "S|LIQ|20_35"), ("a", tda, "S|LIQ|20_35"), ("min", tdm, "S|LIQ|20_35")):
        for s in range(10 if track != "min" else 5):
            n = f"P|{s}"
            for w in WIN_NAMES:
                a, b = win(track, w)
                rb = LE.rel_block(LE.slice_eval(td.r(n), a, b), LE.slice_eval(td.b1, a, b))
                rl = LE.rel_block(LE.slice_eval(td.r(liq), a, b), LE.slice_eval(td.b1, a, b))
                ex_rows.append(dict(track=track, placebo=s, window=w, excess_arith=rb["excess_arith"], te=rb["te"], ir=rb["ir"],
                                    d_ir_vs_liq=rb["ir"] - rl["ir"]))
    pd.DataFrame(ex_rows).to_csv(OUTD / "inc_t5_placebo.csv", index=False)

    vrows = []
    for sch in ("EW5", "CORE5", "ERC5", "EW6", "CORE6", "ERC6"):
        for d in ("45_60", "20_35"):
            r = dict(config=f"C|{sch}|{d}")
            for w in WIN_NAMES:
                r[f"ir_{w}"] = ir_of(tdl, r["config"], w)
                r[f"ir_liq_{w}"] = ir_of(tdl, "S|LIQ|20_35", w)
                r[f"better_{w}"] = r[f"ir_{w}"] > r[f"ir_liq_{w}"]
            r["three_window_better"] = all(r[f"better_{w}"] for w in WIN_NAMES)
            vrows.append(r)
    pd.DataFrame(vrows).to_csv(OUTD / "inc_t5_variants.csv", index=False)

    drow = []
    for track, td in (("lh", tdl), ("min", tdm), ("a", tda)):
        liq = td.r("S|LIQ|20_35")
        for alpha in (0.2, 0.5):
            mix = alpha * liq + (1 - alpha) * td.b1.reindex(liq.index)
            for w in WIN_NAMES:
                a, b = win(track, w)
                rel = LE.rel_block(LE.slice_eval(mix, a, b), LE.slice_eval(td.b1, a, b))
                drow.append(dict(track=track, alpha=alpha, window=w, excess_arith=rel["excess_arith"], te=rel["te"], ir=rel["ir"]))
    pd.DataFrame(drow).to_csv(OUTD / "inc_t5_dilution.csv", index=False)

    nrow, trow, erow = [], [], []
    for track, td in (("lh", tdl), ("min", tdm)):
        for sch in ("EW5", "CORE5", "ERC5", "S4"):
            d = "45_60"
            mem = STYLE5 if sch != "S4" else STYLE5[1:]
            cfg = f"C|{sch}|{d}"
            R = td.run(cfg)
            if sch == "EW5":
                a_ = np.full(len(mem), 1 / len(mem))
            elif sch == "CORE5":
                a_ = S5.core_weights(len(mem), 0, 0.5)
            elif sch == "S4":
                a_ = np.full(len(mem), 1 / len(mem))
            else:
                al = R["erc_alpha"]
                a_ = al.loc[win(track, "full")[0]:win(track, "full")[1]].mean().to_numpy()
                erow.append(dict(track=track, scheme=sch, window="full", **{f"mean_{m}": float(v) for m, v in zip(mem, a_)},
                                 **{f"min_{m}": float(al.loc[win(track, 'full')[0]:win(track, 'full')[1]][m].min()) for m in mem},
                                 **{f"max_{m}": float(al.loc[win(track, 'full')[0]:win(track, 'full')[1]][m].max()) for m in mem}))
            members = {m: td.r(f"S|{m}|{'20_35' if m == 'LIQ' else d}") for m in mem}
            naive = sum(w_ * members[m] for w_, m in zip(a_, mem))
            for w in WIN_NAMES:
                aa, bb = win(track, w)
                rn = LE.rel_block(LE.slice_eval(naive, aa, bb), LE.slice_eval(td.b1, aa, bb))
                rc = LE.rel_block(LE.slice_eval(td.r(cfg), aa, bb), LE.slice_eval(td.b1, aa, bb))
                nrow.append(dict(track=track, scheme=sch, window=w, excess_combined=rc["excess_arith"], excess_naive=rn["excess_arith"],
                                 netting_gain=rc["excess_arith"] - rn["excess_arith"], te_combined=rc["te"], te_naive=rn["te"],
                                 ir_combined=rc["ir"], ir_naive=rn["ir"]))
            # 预期 TE:用篮子单独超额协方差
            mex = pd.DataFrame({m: td.ex(f"S|{m}|{'20_35' if m == 'LIQ' else d}") for m in mem}).dropna()
            for w in WIN_NAMES:
                aa, bb = win(track, w)
                C = mex.loc[aa:bb].cov().to_numpy() * TD
                exp_te = float(np.sqrt(a_ @ C @ a_))
                rc = LE.rel_block(LE.slice_eval(td.r(cfg), aa, bb), LE.slice_eval(td.b1, aa, bb))
                trow.append(dict(track=track, scheme=sch, window=w, te_expected=exp_te, te_realized=rc["te"],
                                 te_avg_member=float(np.mean(np.sqrt(np.diag(C)))), te_max_member=float(np.max(np.sqrt(np.diag(C))))))
    pd.DataFrame(nrow).to_csv(OUTD / "inc_t5_netting.csv", index=False)
    pd.DataFrame(trow).to_csv(OUTD / "inc_t5_te_expected.csv", index=False)
    pd.DataFrame(erow).to_csv(OUTD / "inc_t5_erc_alpha.csv", index=False)

    # ---- 成本口径对照 ----
    crow = []
    for track, td, items in (("lh", tdl, [("S|LIQ|20_35", "Y1"), ("S|LIQ|20_35|old", "old"), ("S|LIQ|20_35|pess", "pess"),
                                           ("C|EW5|45_60", "Y1"), ("C|EW5|45_60|old", "old"), ("C|EW5|45_60|pess", "pess")]),
                             ("min", tdm, [("S|LIQ|20_35", "Y1"), ("S|LIQ|20_35|old", "old"), ("C|EW5|45_60", "Y1"), ("C|EW5|45_60|old", "old")])):
        for n, cst in items:
            for w in WIN_NAMES:
                r = stat_row(td, n, w)
                crow.append(dict(track=track, config=n, cost=cst, window=w, excess_arith=r["excess_arith"], excess_geo=r["excess_geo"],
                                 te=r["te"], ir=r["ir"], nw_t5=r["nw_t5"], cost_ann_run=r["cost_ann_run"], turnover_ann_run=r["turnover_ann_run"]))
    pd.DataFrame(crow).to_csv(OUTD / "inc_t5_cost_variants.csv", index=False)
    # ---- 事后:样本内最优长仓权重的 IR 上界(描述性,不进判定)----
    from scipy.optimize import minimize
    brow = []
    for track, td, items in (("lh", tdl, {"LIQ": "S|LIQ|20_35", "LOWVOL": "S|LOWVOL|45_60", "VALUE": "S|VALUE|45_60", "DIV": "S|DIV|45_60",
                                             "QUAL": "S|QUAL|45_60", "REV": "S|REV|45_60"}),
                             ("lh20", tdl, {"LIQ": "S|LIQ|20_35", "LOWVOL": "S|LOWVOL|20_35", "VALUE": "S|VALUE|20_35", "DIV": "S|DIV|20_35",
                                              "QUAL": "S|QUAL|20_35", "REV": "S|REV|20_35"}),
                             ("min", tdm, {"LIQ": "S|LIQ|20_35", "LOWVOL": "S|LOWVOL|45_60", "VALUE": "S|VALUE|45_60", "DIV": "S|DIV|45_60",
                                             "QUAL": "S|QUAL|45_60", "REV": "S|REV|45_60"})):
        tr = "lh" if track == "lh20" else track
        ex = pd.DataFrame({k: td.ex(v) for k, v in items.items()}).dropna()
        for w in WIN_NAMES:
            a, b = win(tr, w)
            x = ex.loc[a:b]
            mu, C = x.mean().to_numpy() * TD, x.cov().to_numpy() * TD
            K = len(mu)
            f = lambda w_: -(w_ @ mu) / np.sqrt(w_ @ C @ w_)
            res = minimize(f, np.full(K, 1.0 / K), method="SLSQP", bounds=[(0, 1)] * K,
                           constraints=[{"type": "eq", "fun": lambda w_: w_.sum() - 1}])
            w_ = res.x
            brow.append(dict(track=track, window=w, ir_max_insample=float(-res.fun), ir_liq=float(mu[0] / np.sqrt(C[0, 0])),
                             ir_equal=float((np.full(K, 1 / K) @ mu) / np.sqrt(np.full(K, 1 / K) @ C @ np.full(K, 1 / K))),
                             **{f"w_{k}": float(v) for k, v in zip(items, w_)}))
    pd.DataFrame(brow).to_csv(OUTD / "inc_t5_insample_bound.csv", index=False)

    # ---- 假设 H1-H8 ----
    corr_ = pd.read_csv(OUTD / "inc_t5_corr.csv")
    rho = lambda tr, b, w="full": float(corr_[(corr_.track == tr) & (corr_.kind == "pearson_daily") & (corr_.window == w) & (corr_.a == "LIQ") & (corr_.b == b)]["rho"].iloc[0])
    hrows = [
        dict(h="H1", text="低波动与排雷超额相关 > 0.5,加入后 IR 不提高", result=f"B-LH ρ={rho('lh', 'LOWVOL'):.2f}、B-MIN ρ={rho('min', 'LOWVOL'):.2f};A 轨 CMB-A2 IR {Pa['detail']['irs']['full'][0]:.3f} vs 排雷 {Pa['detail']['irs']['full'][1]:.3f}",
             held=bool(rho('lh', 'LOWVOL') > 0.5 and Pa['detail']['irs']['full'][0] <= Pa['detail']['irs']['full'][1])),
        dict(h="H2", text="价值 / 红利 / 质量与排雷的日超额相关 |ρ| <= 0.3", result=f"B-LH ρ: 价值 {rho('lh', 'VALUE'):.2f} 红利 {rho('lh', 'DIV'):.2f} 质量 {rho('lh', 'QUAL'):.2f};B-MIN: {rho('min', 'VALUE'):.2f} {rho('min', 'DIV'):.2f} {rho('min', 'QUAL'):.2f}",
             held=bool(max(abs(rho('lh', 'VALUE')), abs(rho('lh', 'DIV')), abs(rho('lh', 'QUAL'))) <= 0.3)),
    ]
    sm = pd.read_csv(OUTD / "inc_t5_summary.csv")
    gx = lambda cfg, w, tr="lh": float(sm[(sm.track == tr) & (sm.config == cfg) & (sm.window == w)]["excess_arith"].iloc[0])
    h3 = {b: (gx(f"S|{b}|45_60", "dev"), gx(f"S|{b}|45_60", "test")) for b in ("VALUE", "DIV", "QUAL")}
    hrows.append(dict(h="H3", text="价值 / 红利 / 质量至少一个在开发窗与检验窗都有成本后正超额", result="; ".join(f"{k} {v[0] * 100:+.2f}/{v[1] * 100:+.2f}pp" for k, v in h3.items()),
                      held=bool(any(v[0] > 0 and v[1] > 0 for v in h3.values()))))
    hrows.append(dict(h="H4", text="CMB-EW5 相对排雷:IR 提高且生成检验截距显著 > 0", result=f"ΔIR {Pl['dIR_full']:+.3f},S4 截距 {Pl['a_full'] * 100:+.2f}pp (t {Pl['t_full']:.2f})", held=bool(Pl["P1"] and Pl["P2"])))
    hrows.append(dict(h="H5", text="合成后最长相对水下期与相对回撤/TE 不比排雷差", result=f"水下 {Pl['detail']['uw_cmb']} vs {Pl['detail']['uw_liq']} 日;回撤/TE {Pl['detail']['rdd_te_cmb']:.2f} vs {Pl['detail']['rdd_te_liq']:.2f}", held=bool(Pl["P3"])))
    ev6 = dict(ew5=ir_of(tdl, "C|EW5|45_60", "full"), ew6=ir_of(tdl, "C|EW6|45_60", "full"))
    hrows.append(dict(h="H6", text="反转篮子成本后超额 <= 0,加入后 IR 下降",
                      result=f"REV(45/60) 全期算术超额 {gx('S|REV|45_60', 'full') * 100:+.2f}pp;EW6 IR {ev6['ew6']:.3f} vs EW5 {ev6['ew5']:.3f}",
                      held=bool(gx('S|REV|45_60', 'full') <= 0 and ev6["ew6"] < ev6["ew5"])))
    erc = [(ir_of(tdl, f"C|ERC5|{d}", "full") - ir_of(tdl, f"C|EW5|{d}", "full")) for d in ("45_60", "20_35")] + \
          [ir_of(tdm, "C|ERC5|45_60", "full") - ir_of(tdm, "C|EW5|45_60", "full")]
    hrows.append(dict(h="H7", text="逆 TE 等风险权重相对等权没有稳定优势", result="ERC5 − EW5 的 IR 差:B-LH 45/60 {:+.3f}, 20/35 {:+.3f}, B-MIN {:+.3f}".format(*erc),
                      held=bool(not all(e > 0 for e in erc))))
    plr = pd.read_csv(OUTD / "inc_t5_placebo.csv")
    mx = plr[plr.window == "full"].groupby("track")["d_ir_vs_liq"].max()
    hrows.append(dict(h="H8", text="随机篮子安慰剂替换风格篮子后 ΔIR < 0", result="安慰剂最大 ΔIR: " + ", ".join(f"{k} {v:+.3f}" for k, v in mx.items()),
                      held=bool((mx < 0).all())))
    pd.DataFrame(hrows).to_csv(OUTD / "inc_t5_hypotheses.csv", index=False)
    log(f"[stats] 完成 {time.time() - t0:.0f}s")


def stage_bears():
    """熊市段(预注册 §8:BEAR18 / BEAR22 / BEAR24;窗口按 lh_engine.WINDOWS_A 的信号日行切片):窗口收益与对 B1 的差。"""
    rows = []
    for track, names in (("lh", ["S|LIQ|20_35", "S|LOWVOL|45_60", "S|VALUE|45_60", "S|DIV|45_60", "S|QUAL|45_60", "S|REV|45_60",
                                  "C|EW5|45_60", "C|CORE5|45_60", "C|S4|45_60"]),
                         ("min", ["S|LIQ|20_35", "S|LOWVOL|45_60", "S|VALUE|45_60", "S|DIV|45_60", "S|QUAL|45_60", "S|REV|45_60",
                                  "C|EW5|45_60", "C|CORE5|45_60", "C|S4|45_60"])):
        td = TrackData(track)
        for wn in BEARS:
            a, b = LE.WINDOWS_A[wn]
            ix = LE.window_rows(td.dates, a, b)
            if len(ix) < 20 or (track == "min" and wn == "BEAR18"):
                continue
            bb = td.b1.iloc[ix] if False else td.b1.reindex(td.dates[ix])
            if bb.dropna().empty:
                continue
            for n in names:
                r = td.r(n).reindex(td.dates[ix])
                ret_s, ret_b = float((1 + r.fillna(0)).prod() - 1), float((1 + bb.fillna(0)).prod() - 1)
                rows.append(dict(track=track, window=wn, start=a, end=b, config=n, n_days=len(ix), ret_strategy=ret_s,
                                 ret_bench=ret_b, rel_diff=ret_s - ret_b))
    pd.DataFrame(rows).to_csv(OUTD / "inc_t5_bear_windows.csv", index=False)
    log("[bears] 完成")


def stage_mktbeta():
    """相对收益对市场(B1 日收益)的敏感度:e_k = a + beta * r_B1;负 beta = 熊市里相对更好、牛市里相对更差。另报 BULL19 窗口。"""
    rows = []
    for track, names in (("lh", ["S|LIQ|20_35", "S|LOWVOL|45_60", "S|VALUE|45_60", "S|DIV|45_60", "S|QUAL|45_60", "S|REV|45_60", "C|EW5|45_60", "C|CORE5|45_60", "C|S4|45_60"]),
                         ("min", ["S|LIQ|20_35", "S|LOWVOL|45_60", "S|VALUE|45_60", "S|DIV|45_60", "S|QUAL|45_60", "S|REV|45_60", "C|EW5|45_60", "C|CORE5|45_60", "C|S4|45_60"])):
        td = TrackData(track)
        for w in WIN_NAMES:
            a, b = win(track, w)
            rb = LE.slice_eval(td.b1, a, b)
            for n in names:
                e = td.ex(n).loc[a:b]
                m = e.index.intersection(rb.index)
                r = S5.hac_regress(e.loc[m].to_numpy(), rb.loc[m].to_numpy(), 5)
                up = rb.loc[m] > 0
                rows.append(dict(track=track, window=w, config=n, beta_to_b1=r["b"], t_beta=r["t_b"], r2=r["r2"],
                                 alpha_ann=r["a"] * TD, excess_up_days_ann=float(e.loc[m][up].mean() * TD),
                                 excess_down_days_ann=float(e.loc[m][~up].mean() * TD)))
    td = TrackData("lh")
    a, b = LE.WINDOWS_A["BULL19"]
    ix = LE.window_rows(td.dates, a, b)
    bb = td.b1.reindex(td.dates[ix])
    for n in ("S|LIQ|20_35", "S|LOWVOL|45_60", "S|VALUE|45_60", "S|DIV|45_60", "S|QUAL|45_60", "C|EW5|45_60", "C|CORE5|45_60"):
        r = td.r(n).reindex(td.dates[ix])
        rows.append(dict(track="lh", window="BULL19", config=n, beta_to_b1=np.nan, t_beta=np.nan, r2=np.nan, alpha_ann=np.nan,
                         excess_up_days_ann=float((1 + r.fillna(0)).prod() - (1 + bb.fillna(0)).prod())))
    pd.DataFrame(rows).to_csv(OUTD / "inc_t5_mkt_beta.csv", index=False)
    log("[mktbeta] 完成")


def stage_posthoc():
    """事后探索(看到结果后加,不进判定):REV 20/35 深度与 LIQ+REV 等权;A 轨 2006-2016 对该假说是样本外。"""
    rows = []
    for track in ("lh", "a"):
        td = TrackData(track)
        wins = {"full": win(track, "full"), "dev": win(track, "dev"), "test": win(track, "test")}
        if track == "a":
            wins["2006-2016(样本外)"] = ("2006-01-04", "2016-12-31")
            wins["2017-2025(选出该假说的样本)"] = ("2017-01-03", "2025-12-31")
        for n in ("S|LIQ|20_35", "S|REV|20_35", "C|LR_20_35"):
            for wn, (a, b) in wins.items():
                rs, rb = LE.slice_eval(td.r(n), a, b), LE.slice_eval(td.b1, a, b)
                rel = LE.rel_block(rs, rb)
                e_l = td.ex("S|LIQ|20_35").loc[a:b]
                e_n = td.ex(n).loc[a:b]
                m = e_l.index.intersection(e_n.index)
                gt = S5.hac_regress(e_n.loc[m].to_numpy(), e_l.loc[m].to_numpy(), 5) if n != "S|LIQ|20_35" else {}
                uw = S5.rel_underwater(rs, rb)
                rows.append(dict(track=track, config=n, window=wn, excess_arith=rel["excess_arith"], excess_geo=rel["excess_geo"],
                                 te=rel["te"], ir=rel["ir"], nw_t5=rel["nw_t5"], rho_vs_liq=float(e_n.loc[m].corr(e_l.loc[m])),
                                 a_vs_liq_ann=gt.get("a", np.nan) * TD if gt else np.nan, t_a_vs_liq=gt.get("t_a", np.nan) if gt else np.nan,
                                 rel_mdd=rel["rel_mdd"], rel_uw_longest_days=uw["rel_uw_longest_days"],
                                 turnover_ann_run=td.run(n)["turnover"] * TD, cost_ann_run=td.run(n)["cost_bps"] * 1e-4 * TD))
    out = pd.DataFrame(rows)
    out.to_csv(OUTD / "inc_t5_posthoc_rev.csv", index=False)
    log("[posthoc] 完成")


if __name__ == "__main__":
    for c in sys.argv[1:]:
        globals()[f"stage_{c}"]()
