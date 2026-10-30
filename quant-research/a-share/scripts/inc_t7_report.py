"""INC-T7 账本 / 判定 / 图(前缀 inc_t7;预注册 research/INC_T7_PREREG.md §6-§8)。

输入:scripts/inc_t7_grid.py 落盘的 cache/_tmp/inc_t7_*(只读,不再模拟)。
产出(research/ledger/):
  inc_t7_grid_minute.csv / inc_t7_grid_lh.csv   每格 × 成本 × 窗口:净超额(对同频基准与 k=5 基准)、毛超额、年成本四项、
                                                 换手、持仓、仓位、回撤、TE/IR
  inc_t7_event.csv                              事件式变体(5 起点平均)同上 + 逐起点极差
  inc_t7_paired.csv                             对现行 (5, 20/35) 的配对差 Δ:几何 / 算术、NW t(q=20 / 5)、±1 NW se、
                                                 年 t、三段分解(再平衡频率 / 信号时效 / 成本)与恒等式残差
  inc_t7_criteria.csv                           §7 判定条件 C1-C6 逐格,及最终判定(inc_t7_verdict.json)
  inc_t7_yearly.csv                             逐年 Δ(Y1)与逐年净超额
  inc_t7_checks.csv                             §4.5 / §8 断言与复现
  inc_t7_decay.csv                              §6.3 机制诊断汇总
  inc_t7_placebo.csv                            §6.4 安慰剂
  inc_t7_sensitivity.csv                        §5.3 敏感性
  inc_t7_hypotheses.csv                         H1-H5 逐条
图:reports/inc_t7_frontier_{minute,lh}.png、reports/inc_t7_delta_{minute,lh}.png、reports/inc_t7_decay.png

用法:PYTHONPATH=src python3 scripts/inc_t7_report.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor import strategy as ST
from afactor import lh_engine as LE
from afactor.config import LEDGER, TMP, REPORTS
import inc_t7_grid as G

TD = 242
PANELS = ("minute", "lh")
WIN = {
    "minute": {"full": ("2021-01-01", "2025-12-31"), "dev": ("2021-01-01", "2023-12-31"),
               "test": ("2024-01-01", "2025-12-31")},
    "lh": {"full": ("2006-01-04", "2025-12-31"), "dev": ("2006-01-04", "2019-12-31"),
           "test": ("2020-01-01", "2025-12-31"), "lhpre_dev": ("2006-01-04", "2014-12-31"),
           "lhpre_test": ("2015-01-05", "2025-12-31")},
}
CRASH = ("2024-01-02", "2024-02-07")
EXC = {"minute": ("2024-01-02", "2024-02-07"), "lh": ("2018-01-01", "2018-12-31")}   # 剔除窗口:分钟=微盘踩踏(§9.8);长历史=2018(4b 已知残差年,D10)
THR = {"minute": 0.005, "lh": 0.003}
CUR = "k5|20/35"
PRIMARY = {"k10|20/35": "P1", "k5|30/45": "P2", "E1_6": "P3"}
COLS = ["ret", "gross_ret", "cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact",
        "turnover", "exposure", "n_hold"]


# ====================================================================== 统计工具
def nw(x, q: int):
    """日序列均值的 Newey-West t 与标准误(Bartlett,带宽 q;与 LE.nw_t 同一公式)。返回 (t, se_daily)。"""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30 or x.std() <= 0:
        return np.nan, np.nan
    vr = ST.lr_variance_ratio(x, q)
    se = x.std(ddof=1) * np.sqrt(vr) / np.sqrt(n)
    return float(x.mean() / se), float(se)


def ann_geo(r: pd.Series) -> float:
    return LE.ann_geo(r)


def wslice(df, panel: str, win: str):
    if win.endswith("_exc"):
        base = win[:-4]
        a, b = WIN[panel][base]
        out = df.loc[pd.Timestamp(a):pd.Timestamp(b)]
        drop = (out.index >= pd.Timestamp(EXC[panel][0])) & (out.index <= pd.Timestamp(EXC[panel][1]))
        return out[~drop]
    a, b = WIN[panel][win]
    return df.loc[pd.Timestamp(a):pd.Timestamp(b)]


def windows(panel: str) -> list[str]:
    w = list(WIN[panel])
    w += ["full_exc", "test_exc"]
    return w


def mdd(r: pd.Series) -> float:
    nav = (1.0 + r.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


# ====================================================================== 载入与序列
class PanelData:
    def __init__(self, panel: str):
        self.panel = panel
        self.B = pd.read_parquet(TMP / f"inc_t7_bench_{panel}.parquet")
        self.M = pd.read_csv(TMP / f"inc_t7_grid_meta_{panel}.csv")
        D = pd.read_parquet(TMP / f"inc_t7_grid_daily_{panel}.parquet")
        self.S: dict = {}                      # (base, cost) -> DataFrame[COLS + bench + b5]
        self.kind: dict = {}
        for cell, d in D.groupby("cell", sort=False):
            k, band, cost = cell.split("|")
            base = f"{k}|{band}"
            d = d.set_index("date").sort_index()
            kk = int(k[1:])
            f = d[COLS].copy()
            f["bench"] = self.B[f"b{kk}"].reindex(f.index)
            f["b5"] = self.B["b5"].reindex(f.index)
            self.S[(base, cost)] = f
            self.kind[base] = dict(type="fp" if band == "fp" else "grid", k=kk, band=band)
        del D
        self.EM = pd.DataFrame()
        self.offset_rets: dict = {}
        fe = TMP / f"inc_t7_event_daily_{panel}.parquet"
        if fe.exists():
            E = pd.read_parquet(fe)
            self.EM = pd.read_csv(TMP / f"inc_t7_event_meta_{panel}.csv")
            MB = pd.read_parquet(TMP / f"inc_t7_event_matchbench_{panel}.parquet") \
                if (TMP / f"inc_t7_event_matchbench_{panel}.parquet").exists() else pd.DataFrame()
            for (v, cost), grp in self.EM.groupby(["variant", "cost"], sort=False):
                frames, benches = [], []
                for cell in grp["cell"]:
                    d = E[E.cell == cell].set_index("date").sort_index()
                    frames.append(d[COLS])
                    if v.startswith("E1"):
                        benches.append(MB[cell].reindex(d.index))
                common = frames[0].index
                for fr in frames[1:]:
                    common = common.intersection(fr.index)
                avg = sum(fr.loc[common] for fr in frames) / len(frames)
                if v.startswith("E1"):
                    avg["bench"] = sum(b.loc[common] for b in benches) / len(benches)
                else:
                    avg["bench"] = self.B[f"b{G.EVENTS[v][1]}"].reindex(common)
                avg["b5"] = self.B["b5"].reindex(common)
                self.S[(v, cost)] = avg
                self.offset_rets[(v, cost)] = [fr["ret"] for fr in frames]
                self.kind[v] = dict(type="event", k=np.nan, band="20/35")
            del E

    def bases(self, types=("grid", "event")):
        return [b for b, kd in self.kind.items() if kd["type"] in types]


# ====================================================================== 每格统计
def cell_stats(f: pd.DataFrame) -> dict:
    r, b, b5 = f["ret"], f["bench"], f["b5"]
    out = dict(n_days=len(r), ann=ann_geo(r), bench_ann=ann_geo(b), bench5_ann=ann_geo(b5))
    out["excess_geo"] = out["ann"] - out["bench_ann"]
    out["excess_geo_b5"] = out["ann"] - out["bench5_ann"]
    out["excess_arith"] = float((r - b).mean() * TD)
    for c in ("cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact"):
        out[f"{c}_ann"] = float(f[c].mean() * TD)
    out["gross_excess"] = out["excess_geo"] + out["cost_total_ann"]         # §6.2 定义:净超额 + 年成本
    out["gross_excess_arith"] = float((f["gross_ret"] - b).mean() * TD)
    out["turnover_ann"] = float(f["turnover"].mean() * TD)
    out["holdings"] = float(f["n_hold"].mean())
    out["exposure"] = float(f["exposure"].mean())
    out["max_dd"] = mdd(r)
    es = ST.excess_stats(r, b, bandwidth=20, t_bandwidth=20)
    out.update(te_adj=es.get("te_adj", np.nan), ir_adj=es.get("ir_adj", np.nan), t_nw20=es.get("t_nw", np.nan))
    return out


def paired_stats(fc: pd.DataFrame, fu: pd.DataFrame) -> dict:
    a, u = fc.align(fu, join="inner", axis=0)
    d = a["ret"] - u["ret"]
    t20, se20 = nw(d, 20)
    t5, _ = nw(d, 5)
    yrs = d.index.year
    yc = (1 + a["ret"]).groupby(yrs).prod() - 1
    yu = (1 + u["ret"]).groupby(yrs).prod() - 1
    yd = yc - yu
    out = dict(n_days=len(d), d_geo=ann_geo(a["ret"]) - ann_geo(u["ret"]), d_arith=float(d.mean() * TD),
               t_nw20=t20, se_nw20=se20 * TD if np.isfinite(se20) else np.nan, t_nw5=t5,
               year_t=LE.year_t(yd) if len(yd) >= 3 else np.nan, n_years=len(yd),
               frac_years_pos=float((yd > 0).mean()) if len(yd) else np.nan)
    # 三段分解(算术年化):Δ = [b_C − b_5] + [(g_C − b_C) − (g_cur − b_5)] − [cost_C − cost_cur]
    rebal = float((a["bench"] - u["b5"]).mean() * TD)
    signal = float(((a["gross_ret"] - a["bench"]) - (u["gross_ret"] - u["b5"])).mean() * TD)
    costd = float((a["cost_total"] - u["cost_total"]).mean() * TD)
    out.update(dec_rebal=rebal, dec_signal=signal, dec_cost=costd,
               dec_resid=out["d_arith"] - (rebal + signal - costd))
    return out, yd


# ====================================================================== 账本
def build_grid_and_paired(PD: dict):
    grid_rows, ev_rows, pair_rows, year_rows = [], [], [], []
    for panel, pdat in PD.items():
        for base in pdat.bases(("grid", "fp", "event")):
            kd = pdat.kind[base]
            for cost in G.COSTS:
                f = pdat.S.get((base, cost))
                if f is None:
                    continue
                fu = pdat.S[(CUR, cost)]
                for w in windows(panel):
                    fw, uw = wslice(f, panel, w), wslice(fu, panel, w)
                    st = cell_stats(fw)
                    row = dict(panel=panel, cell=base, type=kd["type"], k=kd["k"], band=kd["band"], cost=cost,
                               window=w, primary=PRIMARY.get(base, ""), **st)
                    if kd["type"] == "event":
                        offs = pdat.offset_rets[(base, cost)]
                        uu = wslice(fu, panel, w)["ret"]
                        dd = [ann_geo(wslice(o.to_frame(), panel, w)["ret"]) - ann_geo(uu) for o in offs]
                        row.update(d_geo_offset_min=float(np.min(dd)), d_geo_offset_max=float(np.max(dd)),
                                   n_offsets=len(offs))
                        ev_rows.append(row)
                    else:
                        grid_rows.append(row)
                    if base == CUR or kd["type"] == "fp":
                        continue
                    ps, yd = paired_stats(fw, uw)
                    pair_rows.append(dict(panel=panel, cell=base, type=kd["type"], k=kd["k"], band=kd["band"],
                                          cost=cost, window=w, primary=PRIMARY.get(base, ""), **ps))
                    if w == "full" and cost == "Y1":
                        for y, v in yd.items():
                            year_rows.append(dict(panel=panel, cell=base, cost=cost, year=int(y), d_year=float(v)))
    grid = pd.DataFrame(grid_rows)
    ev = pd.DataFrame(ev_rows)
    pair = pd.DataFrame(pair_rows)
    yearly = pd.DataFrame(year_rows)
    # 逐年净超额(对同频基准,Y1,全部格子含现行)
    yex = []
    for panel, pdat in PD.items():
        for base in pdat.bases(("grid", "fp", "event")):
            f = pdat.S.get((base, "Y1"))
            if f is None:
                continue
            fw = wslice(f, panel, "full")
            yrs = fw.index.year
            ys = (1 + fw["ret"]).groupby(yrs).prod() - 1
            yb = (1 + fw["bench"]).groupby(yrs).prod() - 1
            for y in ys.index:
                yex.append(dict(panel=panel, cell=base, cost="Y1", year=int(y), excess_year=float(ys[y] - yb[y]),
                                cost_year=float(fw["cost_total"][yrs == y].sum())))
    yex = pd.DataFrame(yex)
    yearly = yex.merge(yearly, on=["panel", "cell", "cost", "year"], how="left")
    return grid, ev, pair, yearly


def neighbors(base: str) -> list[str]:
    """D02 邻接表。"""
    if base.startswith("E1_"):
        th = [3, 6, 10, 15]
        i = th.index(int(base.split("_")[1]))
        return [f"E1_{th[j]}" for j in (i - 1, i + 1) if 0 <= j < len(th)]
    if base.startswith("E2b_"):
        return ["E2b_20" if base == "E2b_10" else "E2b_10"]
    k, band = base.split("|")
    k = int(k[1:])
    e, x = [int(v) for v in band.split("/")]
    out = []
    ki = G.KS.index(k)
    for j in (ki - 1, ki + 1):
        if 0 <= j < len(G.KS):
            out.append(f"k{G.KS[j]}|{band}")
    exits = {20: [20, 30, 35, 45], 30: [30, 40, 45, 55]}
    xi = exits[e].index(x)
    for j in (xi - 1, xi + 1):
        if 0 <= j < 4:
            out.append(f"k{k}|{e}/{exits[e][j]}")
    other = 30 if e == 20 else 20
    out.append(f"k{k}|{other}/{exits[other][xi]}")
    return out


def _decide(C: pd.DataFrame) -> dict:
    """§7.2 / §7.3 机械判定(由 criteria() 抽出,审查修订后复用;逻辑与抽出前逐字相同)。"""
    mx = {p: float(C[f"{p}_min_dev_test"].max()) for p in PANELS}
    arg = {p: str(C.loc[C[f"{p}_min_dev_test"].idxmax(), "cell"]) for p in PANELS}
    if C["pass_all"].any():
        verdict, why = "有增量", "存在满足 C1–C5(探索格子加 C6)的格子:" + ",".join(C.loc[C.pass_all, "cell"])
    elif C["rule73_trigger"].any():
        verdict = "证据不足"
        why = ("§7.3:长历史 2006-2019 有格子明显优于现行(Δ ≥ 0.3pp 且 t ≥ 2)而分钟面板没有:"
               + ",".join(C.loc[C.rule73_trigger, "cell"]) + ";列为 2026 前向检验假说,不改现行参数")
    elif mx["minute"] < THR["minute"] and mx["lh"] < THR["lh"]:
        verdict, why = "无增量", "两面板所有格子 max(min(Δ_dev, Δ_test)) 均低于 C3 门槛"
    else:
        verdict, why = "证据不足", "其余情况(有格子两窗口同时超过门槛但未满足 C1–C5 全部)"
    return dict(verdict=verdict, why=why, max_min_dev_test=mx, argmax_min_dev_test=arg,
                n_pass_minute=int(C["minute_pass"].sum()), n_pass_lh=int(C["lh_pass"].sum()),
                pass_minute=C.loc[C.minute_pass, "cell"].tolist(), pass_lh=C.loc[C.lh_pass, "cell"].tolist(),
                rule73_cells=C.loc[C.rule73_trigger, "cell"].tolist(),
                lh0619_clear_better=C.loc[C.lh0619_clear_better, "cell"].tolist())


FIX_PRICE = LEDGER / "inc_t7_fix_price.csv"


def lh_price_artifact_test():
    """D18(2026-10-11 审查 P0):长历史 k=1 的优势在可成交价格下还在吗?

    把同一信号 S1=-turn20 映射到分钟面板(同池 V1000、同成本 Y1、1000 万恒定 AUM),在开盘 10 分钟 TWAP 下做 k1 对 k5 的配对差,
    套用长历史侧的 C1(开发 / 检验都为正)与 C3(全期 >= 0.3pp)。读 inc_t7_fix_price.csv(scripts/inc_t7_fix.py price 产出)。"""
    if not FIX_PRICE.exists():
        return None
    pr = pd.read_csv(FIX_PRICE)

    def pick(price):
        x = pr[(pr.panel == "minute") & (pr.signal == "S1_turn20") & (pr.price == price) & (pr.cost == "Y1")].set_index("window")
        return x
    tw, au = pick("twap"), pick("auction")
    if not len(tw):
        return None
    out = dict(twap_full=float(tw.loc["full", "d_geo"]), twap_dev=float(tw.loc["dev", "d_geo"]),
               twap_test=float(tw.loc["test", "d_geo"]), twap_t=float(tw.loc["full", "t_nw20"]),
               auction_full=float(au.loc["full", "d_geo"]), auction_dev=float(au.loc["dev", "d_geo"]),
               auction_test=float(au.loc["test", "d_geo"]), auction_t=float(au.loc["full", "t_nw20"]))
    out["C1"] = bool(out["twap_dev"] > 0 and out["twap_test"] > 0)
    out["C3"] = bool(out["twap_full"] >= THR["lh"])
    out["still_valid"] = bool(out["C1"] and out["C3"])
    return out


def revise_verdict(C: pd.DataFrame, verdict: dict):
    """审查修订(D18 / D19):返回 (修订后的 criteria 表, 带 verdict_mechanical_prereg 的判定 dict)。

    R1:长历史侧 k=1 通过若在 TWAP 下不再成立(lh_price_artifact_test),则所有 k=1 格子的长历史侧不计通过(C5 不成立)。
        该检验直接覆盖 k1|20/35;其余 k=1 格子与它同机制(每日在同一价格上重置),全池等权零信号对照在 TWAP / 收盘价下都是负、
        在集合竞价价下是正,与信号无关(inc_t7_fix_price.csv pool_EW 行),按机制外推,在 revision 里如实标注。
    R2:判定顺序沿用 _decide():无格子满足 C1–C5 时,§7.3 的 rule73 格子(D04 曾把它排在"有增量"之后)按字面生效。"""
    t = lh_price_artifact_test()
    if t is None:
        return C, verdict
    C2 = C.copy()
    inval = []
    if not t["still_valid"]:
        inval = [c for c, ok in zip(C2.cell, C2.lh_pass) if ok and str(c).startswith("k1|")]
        C2["lh_price_artifact"] = C2.cell.isin(inval)
        C2.loc[C2.cell.isin(inval), ["lh_pass", "C5", "pass_all"]] = False
    else:
        C2["lh_price_artifact"] = False
    rev = _decide(C2)
    out = dict(rev)
    out["verdict_mechanical_prereg"] = verdict["verdict"]
    out["why_mechanical_prereg"] = verdict["why"]
    out["why"] = ("机械口径(预注册 §7.2 逐字执行)为“" + verdict["verdict"] + "”:" + verdict["why"] + "。2026-10-11 审查后修订为“"
                  + rev["verdict"] + "”:(R1)长历史侧 k=1 的优势不是可成交价格上的增量——同一信号 S1 映射到分钟面板,开盘 10 分钟 TWAP 下 k1−k5 全期 "
                  + f"{t['twap_full'] * 100:+.2f}pp(开发 {t['twap_dev'] * 100:+.2f} / 检验 {t['twap_test'] * 100:+.2f}),"
                  + ("不过 C1/C3" if not t["still_valid"] else "仍过 C1/C3")
                  + f";换成开盘集合竞价单点价则 {t['auction_full'] * 100:+.2f}pp,故 C5 不成立,k=1 各格仅分钟面板通过;"
                  + "(R2)按 §7.3 字面,rule73 格子 " + ",".join(rev["rule73_cells"]) + " 已触发。修订后无格子满足 C1–C5。")
    out["revision"] = dict(price_artifact_test=t, invalidated_lh_cells=inval, source="inc_t7_fix_price.csv",
                           note="R1 的检验直接覆盖 k1|20/35,其余 k=1 格子按同机制外推;E2b / E1_3 本来就只在分钟面板通过(长历史 C3/C4 或 C6 不过),不受 R1 影响")
    out["pass_lh_prereg_mechanical"] = verdict["pass_lh"]
    out["pass_minute_prereg_mechanical"] = verdict["pass_minute"]
    return C2, out


def criteria(pair: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    P = pair[pair.window.isin(["full", "dev", "test"])]
    piv = P.pivot_table(index=["cell", "panel"], columns=["cost", "window"], values=["d_geo", "t_nw20", "year_t"])
    rows = []
    cells = sorted(set(P.cell))
    c1 = {}
    for cell in cells:
        for panel in PANELS:
            if (cell, panel) not in piv.index:
                continue
            r = piv.loc[(cell, panel)]
            dv, dt, df_ = r[("d_geo", "Y1", "dev")], r[("d_geo", "Y1", "test")], r[("d_geo", "Y1", "full")]
            c1[(cell, panel)] = bool(dv > 0 and dt > 0)
    for cell in cells:
        row = dict(cell=cell, primary=PRIMARY.get(cell, ""), exploratory=cell not in PRIMARY)
        ok_all = True
        for panel in PANELS:
            if (cell, panel) not in piv.index:
                ok_all = False
                continue
            r = piv.loc[(cell, panel)]
            dv, dt, dfu = r[("d_geo", "Y1", "dev")], r[("d_geo", "Y1", "test")], r[("d_geo", "Y1", "full")]
            t20, yt = r[("t_nw20", "Y1", "full")], r[("year_t", "Y1", "full")]
            C1 = c1[(cell, panel)]
            C2 = bool(t20 >= 2.0) if panel == "minute" else bool(t20 >= 2.0 or yt >= 2.1)
            C3 = bool(dfu >= THR[panel])
            C4 = bool(r[("d_geo", "old", "full")] > 0 and r[("d_geo", "pess", "full")] > 0)
            nb = neighbors(cell)
            C6 = any(c1.get((n, panel), False) for n in nb)
            pp = f"{panel}_"
            row.update({pp + "d_dev": dv, pp + "d_test": dt, pp + "d_full": dfu, pp + "t_nw20": t20,
                        pp + "year_t": yt, pp + "d_full_old": r[("d_geo", "old", "full")],
                        pp + "d_full_pess": r[("d_geo", "pess", "full")], pp + "min_dev_test": min(dv, dt),
                        pp + "C1": C1, pp + "C2": C2, pp + "C3": C3, pp + "C4": C4, pp + "C6": C6,
                        pp + "neighbors": ";".join(nb)})
            ok = C1 and C2 and C3 and C4 and (C6 if cell not in PRIMARY else True)
            row[pp + "pass"] = ok
            ok_all &= ok
        row["C5"] = bool(row.get("minute_C1", False) and row.get("lh_C1", False))
        row["pass_all"] = bool(ok_all and row["C5"])
        # D04:§7.3 特别规则
        lhd = P[(P.cell == cell) & (P.panel == "lh") & (P.cost == "Y1") & (P.window == "dev")]
        lh_dev_clear = bool(len(lhd) and lhd["d_geo"].iloc[0] >= THR["lh"] and lhd["t_nw20"].iloc[0] >= 2.0)
        minute_not = (not row.get("minute_C1", False)) or (row.get("minute_d_full", np.nan) < THR["minute"])
        row["lh0619_clear_better"] = lh_dev_clear
        row["rule73_trigger"] = bool(lh_dev_clear and minute_not)
        rows.append(row)
    C = pd.DataFrame(rows)
    for col in [c for c in C.columns if c.split("_")[-1] in ("pass", "C1", "C2", "C3", "C4", "C6") or
                c in ("C5", "pass_all", "rule73_trigger", "lh0619_clear_better")]:
        C[col] = C[col].fillna(False).astype(bool)          # 某面板缺该格(例如事件式未跑完)按未通过处理
    meta = _decide(C)
    return C, meta


# ====================================================================== 断言
def build_checks(PD: dict, pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rep = json.loads((TMP / "inc_t7_repro.json").read_text())
    for k, v in rep.items():
        if isinstance(v, dict):
            rows.append(dict(check=f"repro_{k}", value=v.get("value"), ref=v.get("ref"), diff=v.get("diff"),
                             ok=v.get("ok"), note=v.get("ref_src", "") + ("" if v.get("required", True) else
                                                                        "(附加,非预注册断言)")))
    for panel in PANELS:
        for tag in ("grid", "event", "placebo", "sens"):
            f = TMP / f"inc_t7_{tag}_meta_{panel}.csv"
            if not f.exists():
                continue
            M = pd.read_csv(f)
            bad = M[M["assert_fail"].fillna("").astype(str) != ""]
            rows.append(dict(check=f"{panel}_{tag}_exposure_align", value=float(M["exp_max"].max()),
                             ok=bool(bad.empty and (M["exp_max"] <= 1 + 1e-9).all()),
                             note=f"{len(M)} 格;exposure 最大 {M['exp_max'].max():.6f};对齐峰值非 0 的格数 "
                                  f"{int((M['align_peak_k'].fillna(0) != 0).sum())};断言不过 {len(bad)} 格"
                                  + ("" if bad.empty else ":" + ";".join(bad.cell.astype(str)))))
    fi = TMP / "inc_t7_indep.json"
    if fi.exists():
        for k, v in json.loads(fi.read_text()).items():
            rows.append(dict(check=f"indep_{k}", value=v, ref=0.0, diff=v, ok=bool(v == 0.0),
                             note="独立复核:官方入口 ST.run_weights / LE.run 重跑,与网格落盘的日收益逐日最大绝对差(沿用的网格结果)"))
    ff = LEDGER / "inc_t7_fix_verify.csv"
    if ff.exists():                                     # 审查修复阶段:自写模拟器 MinSim 对官方引擎已落盘序列的逐位核对(inc_t7_fix.py verify)
        for _, r in pd.read_csv(ff).iterrows():
            rows.append(dict(check=f"fix_verify_{r['item']}", value=float(r["max_abs_diff_daily_ret"]), ref=0.0,
                             diff=float(r["max_abs_diff_daily_ret"]), ok=bool(r["ok"]),
                             note="审查修复:MinSim(scripts/inc_t7_fix.py)重跑,与引擎落盘日收益逐日最大绝对差(容差 1e-12)"))
    # ④b 全池等权
    for panel, pdat in PD.items():
        for k in G.KS:
            for cost in ("Y1", "old"):
                f = pdat.S.get((f"k{k}|fp", cost))
                if f is None:
                    continue
                fw = wslice(f, panel, "full")
                ex = ann_geo(fw["ret"]) - ann_geo(fw["bench"])
                ca = float(fw["cost_total"].mean() * TD)
                val = ex + ca
                if panel == "minute":
                    ok = -0.005 <= val <= 0.015
                    rows.append(dict(check=f"4b_minute_fp_k{k}_{cost}", value=val, ok=bool(ok) if cost == "Y1" else None,
                                     note=f"2021-2025 全池等权 对 benchmark(rebalance={k}) 几何超额 {ex * 100:+.3f}pp + 年成本 "
                                          f"{ca * 100:.3f}pp;门槛 [−0.5, +1.5]pp" + ("" if cost == "Y1" else "(old 只记录)")))
                else:
                    yrs = fw.index.year
                    lr = (np.log1p(fw["ret"]) - np.log1p(fw["bench"])).groupby(yrs).sum() + \
                        fw["cost_total"].groupby(yrs).sum()
                    late = lr[lr.index >= 2016]
                    worst_y = int(late.abs().idxmax())
                    ok = bool((late.abs() <= 0.01).all())
                    fails = ";".join(f"{y}:{v * 100:+.2f}pp" for y, v in late.items() if abs(v) > 0.01)
                    early = ";".join(f"{y}:{v * 100:+.2f}" for y, v in lr[lr.index < 2016].items())
                    rows.append(dict(check=f"4b_lh_fp_k{k}_{cost}", value=float(late.loc[worst_y]),
                                     ok=ok if cost == "Y1" else None,
                                     note=f"对数口径逐年 Σlog(1+fp)−Σlog(1+B1)+Σ成本;2016 年后最大 |残差| {worst_y} "
                                          f"{late.loc[worst_y] * 100:+.2f}pp;超 1pp 的年份:{fails or '无'};"
                                          f"全期几何超额 {ex * 100:+.3f}pp + 年成本 {ca * 100:.3f}pp = {val * 100:+.3f}pp;"
                                          f"2006-2015(只记录,pp):{early}"))
    # 分解恒等式
    rows.append(dict(check="decomposition_identity_max_resid", value=float(pair["dec_resid"].abs().max()),
                     ok=bool(pair["dec_resid"].abs().max() < 0.001),
                     note="三段分解(算术年化)残差最大值;容差 0.1pp(§6.2)"))
    # 附加:长历史 k=5 30/45 与无缓冲剔 30% 对既有账本
    lhg = pd.read_csv(LEDGER / "lh_step1_grid.csv")
    for base, cfg in (("k5|30/45", "grid_S1_b30_45"), ("k5|30/30", "grid_S1_nobuf30")):
        f = PD["lh"].S.get((base, "Y1"))
        if f is None:
            continue
        fw = wslice(f, "lh", "full")
        v = ann_geo(fw["ret"]) - ann_geo(fw["bench"])
        ref = float(lhg.loc[lhg.config == cfg, "full_excess"].iloc[0])
        rows.append(dict(check=f"extra_lh_{base}_vs_{cfg}", value=v, ref=ref, diff=v - ref, ok=bool(abs(v - ref) < 1e-6),
                         note="附加核对(非预注册断言):本轨道长历史格子 vs lh_step1_grid.csv" +
                              ("(30/30 用缓冲带函数 enter=exit,既有账本用 target_weights,两者剔除集同义)"
                               if cfg.endswith("nobuf30") else "")))
    return pd.DataFrame(rows)


# ====================================================================== 安慰剂
def build_placebo(PD: dict, crit: pd.DataFrame) -> pd.DataFrame:
    f = TMP / "inc_t7_placebo_daily_minute.parquet"
    if not f.exists():
        return pd.DataFrame()
    D = pd.read_parquet(f)
    B = PD["minute"].B
    S = {}
    for cell, d in D.groupby("cell", sort=False):
        k, band, cost, seed = cell.split("|")
        d = d.set_index("date").sort_index()
        S[(seed, f"{k}|{band}")] = d
    rows = []
    for seed in sorted(set(s for s, _ in S)):
        cur = S[(seed, CUR)]
        for (sd, base), d in S.items():
            if sd != seed or base == CUR:
                continue
            k = int(base.split("|")[0][1:])
            rec = dict(seed=seed, cell=base, k=k, band=base.split("|")[1])
            for w in ("full", "dev", "test"):
                a, u = wslice(d, "minute", w), wslice(cur, "minute", w)
                dd = a["ret"] - u["ret"]
                t20, se = nw(dd, 20)
                rec[f"d_{w}"] = ann_geo(a["ret"]) - ann_geo(u["ret"])
                rec[f"t20_{w}"] = t20
                if w == "full":
                    rec["excess_geo"] = ann_geo(a["ret"]) - ann_geo(B[f"b{k}"].reindex(a.index))
                    rec["cost_ann"] = float(a["cost_total"].mean() * TD)
                    rec["turnover_ann"] = float(a["turnover"].mean() * TD)
            rows.append(rec)
    P = pd.DataFrame(rows)
    P["sig_pos"] = (P.d_full > 0) & (P.t20_full >= 2)
    P["c1_c3"] = (P.d_dev > 0) & (P.d_test > 0) & (P.d_full >= THR["minute"])
    summ = P.groupby("seed").agg(n_cells=("cell", "size"), n_sig_pos=("sig_pos", "sum"), n_c1_c3=("c1_c3", "sum"),
                                 max_min_dev_test=("d_dev", lambda s: np.nan)).reset_index()
    summ["max_min_dev_test"] = [float(np.minimum(P[P.seed == s].d_dev, P[P.seed == s].d_test).max())
                                for s in summ.seed]
    real = crit[~crit.cell.str.startswith("E")]
    real_sig = int(((real.minute_d_full > 0) & (real.minute_t_nw20 >= 2)).sum())
    real_c1c3 = int((real.minute_C1 & real.minute_C3).sum())
    summ = pd.concat([summ, pd.DataFrame([dict(seed="real(COMP_v5)", n_cells=len(real), n_sig_pos=real_sig,
                                               n_c1_c3=real_c1c3,
                                               max_min_dev_test=float(real.minute_min_dev_test.max()))])])
    P["row"] = "cell"
    summ["row"] = "summary"
    return pd.concat([summ, P], ignore_index=True)


# ====================================================================== 敏感性
def build_sens(PD: dict, crit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    focus = ["k10|20/35", "k5|30/45", "k20|20/35", "E1_6"] + crit.loc[crit.pass_all | crit.minute_pass |
                                                                       crit.lh_pass, "cell"].tolist()
    focus += ["k1|20/30", "k1|20/35", "k1|20/45", "E1_3", "E2b_10", "E2b_20"]      # 补充:已知有候选的格子也报敏感性
    focus = list(dict.fromkeys(focus))
    for panel, pdat in PD.items():
        f = TMP / f"inc_t7_sens_daily_{panel}.parquet"
        if not f.exists():
            continue
        D = pd.read_parquet(f)
        M = pd.read_csv(TMP / f"inc_t7_sens_meta_{panel}.csv")
        fmb = TMP / f"inc_t7_sens_matchbench_{panel}.parquet"
        MB = pd.read_parquet(fmb) if fmb.exists() else None
        S = {}
        for cell, d in D.groupby("cell", sort=False):
            d = d.set_index("date").sort_index()
            S[cell] = d
        # sampled
        for cell in [c for c in S if c.endswith("|sampled")]:
            k, band, cost, _ = cell.split("|")
            base = f"{k}|{band}"
            f_ = S[cell][COLS].copy()
            kk = int(k[1:])
            f_["bench"] = pdat.B[f"b{kk}"].reindex(f_.index); f_["b5"] = pdat.B["b5"].reindex(f_.index)
            for w in ("full", "dev", "test"):
                st = cell_stats(wslice(f_, panel, w))
                ps, _ = paired_stats(wslice(f_, panel, w), wslice(pdat.S[(CUR, "Y1")], panel, w))
                pm, _ = paired_stats(wslice(f_, panel, w), wslice(pdat.S[(base, "Y1")], panel, w))
                rows.append(dict(panel=panel, test="sampled 语义(状态只在调仓日更新)", cell=base, capital=1e7,
                                 window=w, excess_geo=st["excess_geo"], cost_ann=st["cost_total_ann"],
                                 turnover_ann=st["turnover_ann"], d_vs_cur=ps["d_geo"], t20_vs_cur=ps["t_nw20"],
                                 d_vs_main_semantics=pm["d_geo"]))
        # 资金
        caps = sorted(set(M.loc[M.kind.isin(["sens_aum", "sens_aum_event"]), "capital"]))
        for cap in caps:
            Ms = M[(M.capital == cap) & M.kind.isin(["sens_aum", "sens_aum_event"])]
            for const in sorted(set(Ms.const_aum)):
                Mc = Ms[Ms.const_aum == const]
                ser = {}
                for cell in Mc[Mc.kind == "sens_aum"].cell:
                    k, band, cost, _ = cell.split("|")
                    f_ = S[cell][COLS].copy()
                    kk = int(k[1:])
                    f_["bench"] = pdat.B[f"b{kk}"].reindex(f_.index); f_["b5"] = pdat.B["b5"].reindex(f_.index)
                    ser[f"{k}|{band}"] = f_
                evall = Mc[Mc.kind == "sens_aum_event"]
                for vname, ev in evall.groupby("variant", sort=False):
                    frames = [S[c][COLS] for c in ev.cell]
                    common = frames[0].index
                    for fr in frames[1:]:
                        common = common.intersection(fr.index)
                    avg = sum(fr.loc[common] for fr in frames) / len(frames)
                    if vname.startswith("E1"):
                        # 匹配基准:sens 阶段按各路径实际全量调仓日重算(与 event 阶段同法),5 起点平均
                        avg["bench"] = MB[list(ev.cell)].reindex(common).mean(axis=1) if MB is not None else np.nan
                    else:
                        avg["bench"] = pdat.B[f"b{G.EVENTS[vname][1]}"].reindex(common)
                    avg["b5"] = pdat.B["b5"].reindex(common)
                    ser[vname] = avg
                if CUR not in ser:
                    continue
                for base, f_ in ser.items():
                    for w in ("full", "dev", "test"):
                        fw = wslice(f_, panel, w)
                        ps, _ = paired_stats(fw, wslice(ser[CUR], panel, w)) \
                            if base != CUR else ({"d_geo": 0.0, "t_nw20": np.nan}, None)
                        rows.append(dict(panel=panel, test=f"资金 {cap:.0e}" + ("(恒定 AUM)" if const else "(随净值滚动)"),
                                         cell=base, capital=cap, window=w,
                                         excess_geo=ann_geo(fw["ret"]) - ann_geo(fw["bench"]),
                                         excess_geo_b5=ann_geo(fw["ret"]) - ann_geo(fw["b5"]),
                                         cost_ann=float(fw["cost_total"].mean() * TD),
                                         turnover_ann=float(fw["turnover"].mean() * TD),
                                         d_vs_cur=ps["d_geo"], t20_vs_cur=ps["t_nw20"],
                                         supplementary=base not in (CUR,) and base not in
                                         crit.loc[crit.pass_all, "cell"].tolist()))
    # 分钟面板剔除 2024-01-02 ~ 02-07,与长历史 LH 预注册窗口(从 paired 取)
    return pd.DataFrame(rows), focus


# ====================================================================== 全池等权对照(事后描述,D12)
def build_fp_control(PD: dict) -> pd.DataFrame:
    """零信号对照:全池等权 fp 在同频 k 与 k=5 之间的净收益差(同引擎、同成本、同可成交约束)=纯"调仓频率"的可投资效应。
    策略 Δ 减去它 = 与选股(打分时效)相关的净部分。只描述,不进判定。"""
    rows = []
    for panel, pdat in PD.items():
        for cost in G.COSTS:
            ref = pdat.S.get(("k5|fp", cost))
            if ref is None:
                continue
            for k in (1, 10, 20):
                f = pdat.S.get((f"k{k}|fp", cost))
                if f is None:
                    continue
                for w in ("full", "dev", "test"):
                    a, u = wslice(f, panel, w), wslice(ref, panel, w)
                    ps, _ = paired_stats(a, u)
                    rows.append(dict(panel=panel, cost=cost, k=k, window=w, fp_d_geo=ps["d_geo"], fp_d_arith=ps["d_arith"],
                                     fp_t_nw20=ps["t_nw20"], fp_cost_diff=ps["dec_cost"], fp_bench_diff=ps["dec_rebal"],
                                     fp_gross_diff_vs_bench=ps["dec_signal"]))
    F = pd.DataFrame(rows)
    out = []
    for panel, pdat in PD.items():
        for cost in G.COSTS:
            for k in (1, 10, 20):
                for band in ("20/20", "20/30", "20/35", "20/45", "30/30", "30/40", "30/45", "30/55"):
                    cell = f"k{k}|{band}"
                    f, fu = pdat.S.get((cell, cost)), pdat.S.get((CUR, cost))
                    if f is None or fu is None:
                        continue
                    for w in ("full", "dev", "test"):
                        ps, _ = paired_stats(wslice(f, panel, w), wslice(fu, panel, w))
                        fr = F[(F.panel == panel) & (F.cost == cost) & (F.k == k) & (F.window == w)]
                        if fr.empty:
                            continue
                        fr = fr.iloc[0]
                        out.append(dict(panel=panel, cost=cost, cell=cell, k=k, band=band, window=w,
                                        d_arith=ps["d_arith"], fp_d_arith=fr.fp_d_arith,
                                        d_minus_fp=ps["d_arith"] - fr.fp_d_arith, fp_bench_diff=fr.fp_bench_diff,
                                        fp_cost_diff=fr.fp_cost_diff))
    return F, pd.DataFrame(out)


# ====================================================================== 逐年三段分解(事后描述,D12)
def build_decomp_yearly(PD: dict, cells=("k1|20/35", "k10|20/35", "k20|20/35", "k5|30/45", "E1_6", "E2b_10", "E2b_20")):
    """对现行格子的 Δ 按日历年(算术年化)拆成 再平衡频率效应 / 信号时效 / 成本,Y1。只描述,不进判定。"""
    rows = []
    for panel, pdat in PD.items():
        fu = pdat.S[(CUR, "Y1")]
        for cell in cells:
            f = pdat.S.get((cell, "Y1"))
            if f is None:
                continue
            fw, uw = wslice(f, panel, "full"), wslice(fu, panel, "full")
            a, u = fw.align(uw, join="inner", axis=0)
            comp = pd.DataFrame({
                "delta": a["ret"] - u["ret"],
                "rebal": a["bench"] - u["b5"],
                "signal": (a["gross_ret"] - a["bench"]) - (u["gross_ret"] - u["b5"]),
                "cost": -(a["cost_total"] - u["cost_total"]),
                "bench_cell": a["bench"], "bench_cur": u["bench"]})
            g = comp.groupby(comp.index.year)
            for y, d in g:
                rows.append(dict(panel=panel, cell=cell, year=int(y), n_days=len(d),
                                 **{c + "_ann": float(d[c].mean() * TD) for c in ["delta", "rebal", "signal", "cost"]}))
            rows.append(dict(panel=panel, cell=cell, year=0, n_days=len(comp),
                             **{c + "_ann": float(comp[c].mean() * TD) for c in ["delta", "rebal", "signal", "cost"]}))
    return pd.DataFrame(rows)


# ====================================================================== 机制诊断
def build_decay() -> pd.DataFrame:
    rows = []
    for panel in PANELS:
        f = TMP / f"inc_t7_decay_daily_{panel}.parquet"
        if not f.exists():
            continue
        D = pd.read_parquet(f).set_index("date").sort_index()
        for w in ("full", "dev", "test"):
            Dw = wslice(D, panel, w)
            ps = Dw[Dw.stat == "persist"]
            for L, g in ps.groupby("lag"):
                rows.append(dict(panel=panel, window=w, stat="persist", lag=int(L), jaccard=float(g.jaccard.mean()),
                                 rank_ac=float(g.rank_ac.mean()), n=len(g)))
            ic = Dw[Dw.stat == "ic"]
            for h, g in ic.groupby("lag"):
                h = int(h)
                q = max(5, 2 * h)
                t_ic, _ = nw(g.ic, q)
                t_sp, _ = nw(g.ex_spread, q)
                rows.append(dict(panel=panel, window=w, stat="ic", lag=h, ic_mean=float(g.ic.mean()), ic_t_nw=t_ic,
                                 ex_spread_h=float(g.ex_spread.mean()),
                                 ex_spread_per_day_ann=float(g.ex_spread.mean() / h * TD), ex_spread_t_nw=t_sp,
                                 nw_bw=q, n=len(g)))
    return pd.DataFrame(rows)


# ====================================================================== 假设
def build_hypotheses(grid: pd.DataFrame, ev: pd.DataFrame, pair: pd.DataFrame, crit: pd.DataFrame,
                     decay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    g = grid[(grid.cost == "Y1") & (grid.window == "full")]
    for panel in PANELS:
        gp = g[(g.panel == panel) & (g.type == "grid")]
        # H1
        mono_k = []
        for band, gb in gp.groupby("band"):
            c = gb.sort_values("k")["cost_total_ann"].to_numpy()
            mono_k.append(bool(np.all(np.diff(c) < 0)))
        mono_w = []
        for k, gk in gp.groupby("k"):
            for e in (20, 30):
                ge = gk[gk.band.str.startswith(f"{e}/")].copy()
                ge["x"] = ge.band.str.split("/").str[1].astype(int)
                c = ge.sort_values("x")["cost_total_ann"].to_numpy()
                mono_w.append(bool(np.all(np.diff(c) < 0)))
        cb = gp[gp.band == "20/35"].set_index("k")["cost_total_ann"]
        rows.append(dict(panel=panel, hypothesis="H1 成本单调",
                         result=f"按 k 单调下降 {sum(mono_k)}/{len(mono_k)} 个带;同 enter 按 exit 加宽单调下降 "
                                f"{sum(mono_w)}/{len(mono_w)} 组;20/35 带年成本 k1 {cb[1] * 100:.2f} / k5 {cb[5] * 100:.2f} / "
                                f"k10 {cb[10] * 100:.2f} / k20 {cb[20] * 100:.2f}pp;k5→10 省 {(cb[5] - cb[10]) * 100:.2f}pp,"
                                f"k5→20 省 {(cb[5] - cb[20]) * 100:.2f}pp",
                         supported=bool(all(mono_k) and all(mono_w))))
        # H2
        gx = gp[gp.band == "20/35"].set_index("k")["gross_excess"]
        flat = abs(gx[20] - gx[5]) <= 0.003
        dec = [gp[gp.band == b].set_index("k")["gross_excess"] for b in sorted(set(gp.band))]
        n_dec = sum(bool(s[20] < s[5]) for s in dec)
        rows.append(dict(panel=panel, hypothesis="H2 毛超额随 k 衰减",
                         result=f"20/35 带毛超额(净超额 + 年成本,对同频基准)k1 {gx[1] * 100:+.2f} / k5 {gx[5] * 100:+.2f} / "
                                f"k10 {gx[10] * 100:+.2f} / k20 {gx[20] * 100:+.2f}pp;k5→20 变化 {(gx[20] - gx[5]) * 100:+.2f}pp"
                                f"({'平坦 ±0.3pp 以内' if flat else '超出 ±0.3pp'});8 个带中 k20 毛超额低于 k5 的 {n_dec} 个",
                         supported=bool(gx[20] < gx[5] and not flat), value=float(gx[20] - gx[5])))
        # H3
        pp = pair[(pair.panel == panel) & (pair.cost == "Y1") & (pair.window == "full")]
        best = pp.loc[pp.d_geo.idxmax()]
        cp = crit[[f"{panel}_min_dev_test", "cell"]].sort_values(f"{panel}_min_dev_test", ascending=False)
        res = (f"全期 Δ 最大格 {best.cell} {best.d_geo * 100:+.2f}pp(t {best.t_nw20:.2f});"
               f"min(Δ_dev, Δ_test) 最大格 {cp.iloc[0].cell} {cp.iloc[0][f'{panel}_min_dev_test'] * 100:+.2f}pp")
        if panel == "lh":
            pd_ = pair[(pair.panel == "lh") & (pair.cost == "Y1") & (pair.window == "dev") & (pair.type == "grid")]
            vals = pd.concat([pd_.set_index("cell").d_geo, pd.Series({CUR: 0.0})]).sort_values(ascending=False)
            rank = int(list(vals.index).index(CUR)) + 1
            res += (f";2006-2019(唯一真样本外)上现行在 32 个固定格子中排第 {rank},到最优格 {vals.index[0]} 的距离 "
                    f"{vals.iloc[0] * 100:.2f}pp")
        rows.append(dict(panel=panel, hypothesis="H3 现行在前沿附近", result=res,
                         supported=bool(cp.iloc[0][f"{panel}_min_dev_test"] < THR["minute"]) if panel == "minute"
                         else None, value=float(cp.iloc[0][f"{panel}_min_dev_test"])))
        # H4
        e = ev[(ev.panel == panel) & (ev.cost == "Y1") & (ev.window == "full")].set_index("cell")
        g5 = gp[(gp.band == "20/35")].set_index("k")
        if len(e):
            ee = e.loc["E1_6"]
            dE = pp.set_index("cell").loc["E1_6"]
            rows.append(dict(panel=panel, hypothesis="H4 事件式调仓",
                             result=f"E1 θ=6%:毛超额 {ee.gross_excess * 100:+.2f}pp(k5 {g5.loc[5, 'gross_excess'] * 100:+.2f})、"
                                    f"年成本 {ee.cost_total_ann * 100:.2f}pp(k5 {g5.loc[5, 'cost_total_ann'] * 100:.2f} / "
                                    f"k10 {g5.loc[10, 'cost_total_ann'] * 100:.2f});对现行 Δ {dE.d_geo * 100:+.2f}pp(t {dE.t_nw20:.2f});"
                                    + ";".join(f"{c} Δ {pp.set_index('cell').loc[c, 'd_geo'] * 100:+.2f}"
                                               for c in G.EVENTS if c in pp.cell.values),
                             supported=bool(0 <= dE.d_geo <= 0.004), value=float(dE.d_geo)))
    # H5
    pf = pair[(pair.cost == "Y1") & (pair.window == "full")].set_index(["panel", "cell"])
    s1 = np.sign(pf.loc[("minute", "k10|20/35"), "d_geo"]), np.sign(pf.loc[("lh", "k10|20/35"), "d_geo"])
    s2 = np.sign(pf.loc[("minute", "k5|30/45"), "d_geo"]), np.sign(pf.loc[("lh", "k5|30/45"), "d_geo"])
    rows.append(dict(panel="both", hypothesis="H5 两面板一致",
                     result=f"频率轴 P1(k10 vs k5)Δ 符号 分钟 {s1[0]:+.0f} / 长历史 {s1[1]:+.0f}(Δ {pf.loc[('minute', 'k10|20/35'), 'd_geo'] * 100:+.2f} / "
                            f"{pf.loc[('lh', 'k10|20/35'), 'd_geo'] * 100:+.2f}pp);带宽轴 P2(30/45 vs 20/35)符号 分钟 {s2[0]:+.0f} / 长历史 {s2[1]:+.0f}"
                            f"(Δ {pf.loc[('minute', 'k5|30/45'), 'd_geo'] * 100:+.2f} / {pf.loc[('lh', 'k5|30/45'), 'd_geo'] * 100:+.2f}pp)",
                     supported=bool(s1[0] == s1[1]), value=np.nan))
    return pd.DataFrame(rows)


# ====================================================================== 图
INK, INK2, MUTED, GRID_C, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
KMARK = {1: "o", 5: "s", 10: "^", 20: "D"}


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, color=GRID_C, lw=0.7)
    ax.set_axisbelow(True)


def _envelope(x, y):
    o = np.argsort(x)
    xs, ys = np.asarray(x)[o], np.asarray(y)[o]
    ex, ey, best = [], [], -np.inf
    for a, b in zip(xs, ys):
        if b > best:
            ex.append(a); ey.append(b); best = b
    return ex, ey


def plot_panel(panel: str, grid: pd.DataFrame, ev: pd.DataFrame, pair: pd.DataFrame, what: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cap_capacity_curve as CAP
    CAP._setup_fonts()
    g = grid[(grid.panel == panel) & (grid.cost == "Y1") & (grid.window == "full") & (grid.type == "grid")].copy()
    e = ev[(ev.panel == panel) & (ev.cost == "Y1") & (ev.window == "full")].copy()
    p = pair[(pair.panel == panel) & (pair.cost == "Y1") & (pair.window == "full")].set_index("cell")
    g["x"] = g.cost_total_ann * 100
    e["x"] = e.cost_total_ann * 100
    if what == "frontier":
        g["y"] = g.excess_geo * 100; e["y"] = e.excess_geo * 100
    else:
        g["y"] = [0.0 if c == CUR else p.loc[c, "d_arith"] * 100 for c in g.cell]
        g["se"] = [0.0 if c == CUR else p.loc[c, "se_nw20"] * 100 for c in g.cell]
        e["y"] = [p.loc[c, "d_arith"] * 100 for c in e.cell]
        e["se"] = [p.loc[c, "se_nw20"] * 100 for c in e.cell]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, facecolor=SURF)
    ex_all, ey_all = _envelope(pd.concat([g.x, e.x]), pd.concat([g.y, e.y]))
    for ai, enter in enumerate((20, 30)):
        ax = axes[ai]
        ax.set_facecolor(SURF)
        _style(ax)
        bands = [b for b in ["20/20", "20/30", "20/35", "20/45"] if enter == 20] or ["30/30", "30/40", "30/45", "30/55"]
        for ci, band in enumerate(bands):
            gb = g[g.band == band].sort_values("k")
            col = SERIES[ci]
            ax.plot(gb.x, gb.y, color=col, lw=2, zorder=2, label=f"带 {band}")
            for _, r in gb.iterrows():
                if what == "delta" and r.cell != CUR:
                    ax.errorbar(r.x, r.y, yerr=r.se, color=col, lw=1, capsize=2, zorder=2, alpha=0.8)
                ax.scatter(r.x, r.y, s=46, marker=KMARK[int(r.k)], color=col, edgecolor=SURF, lw=1.5, zorder=3)
        if enter == 20:
            cur = g[g.cell == CUR].iloc[0]
            ax.scatter(cur.x, cur.y, s=170, facecolor="none", edgecolor=INK, lw=1.6, zorder=4)
            ax.annotate("现行 k5 20/35\n(样本内选出)" if panel == "minute" else "现行 k5 20/35",
                        (cur.x, cur.y), xytext=(-70, 22), textcoords="offset points", fontsize=8.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
            for _, r in e.iterrows():
                if what == "delta":
                    ax.errorbar(r.x, r.y, yerr=r.se, color=MUTED, lw=1, capsize=2, zorder=2)
                ax.scatter(r.x, r.y, s=40, marker="X", color=INK2, edgecolor=SURF, lw=1, zorder=3)
                ax.annotate(r.cell.replace("_", " θ=" if r.cell.startswith("E1") else " k=").replace("E1 θ=", "θ").replace("E2b k=", "E2b")
                            + ("%" if r.cell.startswith("E1") else ""),
                            (r.x, r.y), xytext={"E2b_10": (6, -14), "E2b_20": (6, 5)}.get(r.cell, (5, -9)),
                            textcoords="offset points", fontsize=7, color=INK2)
        if what == "frontier":
            ax.plot(ex_all, ey_all, color=MUTED, lw=1.2, ls="--", zorder=1, label="上包络(全部格子)")
        else:
            ax.axhline(0, color="#c3c2b7", lw=1)
            ax.axhline(THR[panel] * 100, color=MUTED, lw=1, ls=":", label=f"C3 门槛 +{THR[panel] * 100:.1f}pp")
        ax.set_xlabel("年成本(Y=1 一跳,pp/年;全期)", color=INK2, fontsize=9)
        ax.set_title(f"enter = {enter}%" + ("(含事件式 ×)" if enter == 20 else ""), color=INK, fontsize=10.5, loc="left")
        h, l = ax.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        h += [Line2D([], [], color=MUTED, marker=KMARK[k], ls="", markersize=6) for k in G.KS]
        l += [f"k={k}" for k in G.KS]
        if enter == 20:
            h += [Line2D([], [], color=INK2, marker="X", ls="", markersize=6)]
            l += ["事件式(E1 θ / E2b k_eq)"]
        ax.legend(h, l, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5,
                  labelcolor=INK2)
    yl = ("净超额(对同频池等权基准,几何年化 pp)" if what == "frontier"
          else "Δ 对现行 (5, 20/35)(算术年化 pp,±1 NW 标准误 q=20)")
    axes[0].set_ylabel(yl, color=INK2, fontsize=9)
    pname = {"minute": "分钟面板 COMP_v5 2021-2025(V1000,1000 万恒定)",
             "lh": "长历史 S1 换手率版 2006-2025(含退市股,1000 万随净值)"}[panel]
    title = ("成本-净超额前沿" if what == "frontier" else "对现行格子的配对净差") + " · " + pname
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.text(0.01, -0.02, "注:点的横轴为 Y=1 年成本;" + ("纵轴对**各自同频**基准,k=1 与 k=5 的基准不同,不能跨 k 直接读收益差(跨 k 比较看配对 Δ 图)。"
                                         if what == "frontier" else "纵轴为净日收益之差(基准无关);误差线 ±1 NW 标准误(q=20);虚线为 C3 幅度门槛。"),
             color=MUTED, fontsize=8)
    out = REPORTS / f"inc_t7_{what}_{panel}.png"
    fig.savefig(out, dpi=140, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_decay(decay: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cap_capacity_curve as CAP
    CAP._setup_fonts()
    if decay.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURF)
    lab = {"minute": "分钟 COMP_v5(2021-2025)", "lh": "长历史 S1(2006-2025)"}
    for i, panel in enumerate(PANELS):
        d = decay[(decay.panel == panel) & (decay.window == "full")]
        ps = d[d.stat == "persist"].sort_values("lag")
        ic = d[d.stat == "ic"].sort_values("lag")
        axes[0].plot(ps.lag, ps.jaccard, color=SERIES[i], lw=2, marker="o", ms=5, label=lab[panel])
        axes[1].plot(ic.lag, ic.ic_mean, color=SERIES[i], lw=2, marker="o", ms=5, label=lab[panel])
        axes[2].plot(ic.lag, ic.ex_spread_per_day_ann * 100, color=SERIES[i], lw=2, marker="o", ms=5, label=lab[panel])
    titles = ["剔除集(最差 30%)Jaccard 随滞后", "Rank IC 随持有期 h(open 口径)", "剔除集 减 其余 的日均收益差(年化 pp)"]
    xl = ["滞后(交易日)", "持有期 h(交易日)", "持有期 h(交易日)"]
    for ax, t, x in zip(axes, titles, xl):
        ax.set_facecolor(SURF); _style(ax)
        ax.set_title(t, color=INK, fontsize=10, loc="left"); ax.set_xlabel(x, color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    axes[2].axhline(0, color="#c3c2b7", lw=1)
    axes[1].set_xticks([1, 5, 10, 20]); axes[2].set_xticks([1, 5, 10, 20]); axes[0].set_xticks([1, 5, 10, 20, 40])
    fig.suptitle("打分持续性与信号衰减(机制诊断,只解释,不进判定)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = REPORTS / "inc_t7_decay.png"
    fig.savefig(out, dpi=140, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fix_price():
    """审查修复图(D18):同一池、同一成本下,k=1 减 k=5 在三种成交价上的配对差(分钟面板 2021-2025,Y1)。
    读 inc_t7_fix_price.csv;信号 = 长历史 S1 / COMP_v5 / 全池等权(零信号)。"""
    f = LEDGER / "inc_t7_fix_price.csv"
    if not f.exists():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cap_capacity_curve as CAP
    CAP._setup_fonts()
    P = pd.read_csv(f)
    P = P[(P.panel == "minute") & (P.cost == "Y1") & (P.window == "full")]
    sigs = [("S1_turn20", "S1=-turn20\n(长历史信号)"), ("COMP_v5", "COMP_v5"), ("pool_EW", "全池等权\n(零信号对照)")]
    prices = [("twap", "开盘 10 分钟 TWAP", "#2a78d6"), ("close", "收盘价", "#1baf7a"), ("auction", "开盘集合竞价单点价", "#eb6834")]
    INK, INK2 = "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
    w = 0.26
    for j, (pk, pl, col) in enumerate(prices):
        xs, ys = [], []
        for i, (sk, _) in enumerate(sigs):
            v = float(P[(P.signal == sk) & (P.price == pk)].iloc[0].d_geo) * 100
            xs.append(i + (j - 1) * (w + 0.02)); ys.append(v)
        ax.bar(xs, ys, width=w, color=col, label=pl, linewidth=0)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:+.2f}", (x, y), xytext=(0, 3 if y >= 0 else -10), textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.15, hi + 0.1)
    ax.set_xticks(range(len(sigs)))
    ax.set_xticklabels([l for _, l in sigs], color=INK, fontsize=9.5)
    ax.set_ylabel("k=1 减 k=5 的年化几何差(pp,Y1,分钟面板 2021-2025)", color=INK2, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("每日调仓的增量取决于成交价:S1 在 TWAP / 收盘价上为零,集合竞价价上 +0.95pp", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = REPORTS / "inc_t7_fix_price.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ====================================================================== 主程序
def main():
    PD = {p: PanelData(p) for p in PANELS}
    grid, ev, pair, yearly = build_grid_and_paired(PD)
    for panel in PANELS:
        grid[grid.panel == panel].to_csv(LEDGER / f"inc_t7_grid_{panel}.csv", index=False)
    ev.to_csv(LEDGER / "inc_t7_event.csv", index=False)
    pair.to_csv(LEDGER / "inc_t7_paired.csv", index=False)
    yearly.to_csv(LEDGER / "inc_t7_yearly.csv", index=False)
    crit, verdict = criteria(pair)
    crit.to_csv(LEDGER / "inc_t7_criteria.csv", index=False)           # 预注册 §7 逐字执行的机械结果,审查后不改
    crit_rev, verdict = revise_verdict(crit, verdict)                   # 审查修订(D18 / D19)
    if "revision" in verdict:
        crit_rev.to_csv(LEDGER / "inc_t7_criteria_revised.csv", index=False)
    checks = build_checks(PD, pair)
    checks.to_csv(LEDGER / "inc_t7_checks.csv", index=False)
    pl = build_placebo(PD, crit)
    if len(pl):
        pl.to_csv(LEDGER / "inc_t7_placebo.csv", index=False)
    sens, focus = build_sens(PD, crit)
    # 追加:分钟面板剔除 2024 年初踩踏、长历史 LH 预注册窗口(取自 paired)
    extra = pair[(pair.cost == "Y1") & pair.window.isin(["full_exc", "test_exc", "lhpre_dev", "lhpre_test"])
                 & pair.cell.isin(focus)]
    extra = extra.assign(test=np.where(extra.window.str.endswith("_exc"),
                                       np.where(extra.panel == "minute", "剔除 2024-01-02~02-07", "剔除 2018 年"),
                                       "LH 预注册窗口 2006-2014 / 2015-2025"),
                         d_vs_cur=extra.d_geo, t20_vs_cur=extra.t_nw20)[
        ["panel", "test", "cell", "window", "d_vs_cur", "t20_vs_cur"]]
    sens = pd.concat([sens, extra], ignore_index=True)
    sens.to_csv(LEDGER / "inc_t7_sensitivity.csv", index=False)
    fpc, fpc_cells = build_fp_control(PD)
    fpc.to_csv(LEDGER / "inc_t7_fp_control.csv", index=False)
    fpc_cells.to_csv(LEDGER / "inc_t7_fp_control_cells.csv", index=False)
    dy = build_decomp_yearly(PD)
    dy.to_csv(LEDGER / "inc_t7_decomp_yearly.csv", index=False)
    decay = build_decay()
    if len(decay):
        decay.to_csv(LEDGER / "inc_t7_decay.csv", index=False)
    hyp = build_hypotheses(grid, ev, pair, crit, decay)
    if "revision" in verdict:                       # 审查修订(D18):H5 的"符号一致"在可成交价格下不成立
        t = verdict["revision"]["price_artifact_test"]
        hyp = pd.concat([hyp, pd.DataFrame([dict(
            panel="both", hypothesis="H5 两面板一致(2026-10-11 审查修订)",
            result=(f"长历史侧 k=1 的正号来自开盘集合竞价单点价:同一信号 S1 映射到分钟面板,TWAP 下 k1−k5 {t['twap_full'] * 100:+.2f}pp"
                    f"(开发 {t['twap_dev'] * 100:+.2f} / 检验 {t['twap_test'] * 100:+.2f}),集合竞价价下 {t['auction_full'] * 100:+.2f}pp;"
                    "频率轴两面板符号一致不能当作独立证据"),
            supported=bool(t["still_valid"]), value=t["twap_full"])])], ignore_index=True)
    hyp.to_csv(LEDGER / "inc_t7_hypotheses.csv", index=False)
    (LEDGER / "inc_t7_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1))
    figs = []
    for panel in PANELS:
        for what in ("frontier", "delta"):
            figs.append(plot_panel(panel, grid, ev, pair, what))
    figs.append(plot_decay(decay))
    figs.append(plot_fix_price())
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    print("figs:", [str(f) for f in figs if f])


if __name__ == "__main__":
    main()
