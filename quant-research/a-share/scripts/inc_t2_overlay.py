"""INC_T2 步骤 5:排雷 + 指数调样叠加层(H8、H9),以及 Gate 1 引擎断言。

预注册 §7.3。基线 B0 = 排雷 COMP_v5(缓冲 20/35、5 日调仓、等权、V1000、open10、5 相位平均),
Y=1 CostModel(ConstAUMCost,与 capacity_grid.csv 同口径),资金 1000 万 / 1 亿。

实现(不改引擎):strategy.simulate 的 callable 目标、rebalance=1、每个相位单独一条路径 ——
  · 基线调仓日 = {t : (t − t_first − φ) % 5 == 0}(与 run_weights(rebalance=5, phase=φ) 的调仓日历逐日相同),
    当天返回基线目标 W[t];非调仓日返回 None(不交易)。B0 用这一包装必须与 ST.run_weights(W, rebalance=5)
    逐位一致(Gate 1 断言之一)。
  · 叠加层额外在事件日交易:信号日 A(→ A+1 open10 成交)按"叠加目标"调仓;信号日 E−1(→ E open10)
    回到基线目标;[A, E−2] 内遇到基线调仓日也用叠加目标。"基线目标"= 本相位最近一次基线调仓的 W。
      OV-A  (1−x)·基线 + x·调入篮子等权
      OV-B  基线中调出股置 0、其余按比例归一(与排雷剔除同法)
      OV-C  A + B
      PL-A  同 OV-A,篮子换成调入股在 A 日匹配的对照(rep 1/2/3 各跑一遍,取均值)
      PL-B  同 OV-B,剔除的是调出股的对照
篮子(偏离预注册,见报告):中证1000 官方名单缺 ->
      O500   官方中证500 ADD_800 / DEL_800(次级判定格)
      Oall   官方 300+500 的 ADD_800 / DEL_800
      PA1000 代理预测(P^A)中证1000 ADD_pure / DEL_pure(主格的替身,只报不判)
所有篮子只含公告日 A 在合格池内的事件股(预注册 §3.4)。

子命令:
  checks            Gate 1 四条断言 -> research/ledger/inc_t2_overlay_checks.csv
  run [--part i/n]  全部叠加层格(可分进程),每格落 cache/_tmp/inc_t2_overlay_runs/<id>.parquet
  summarize         -> research/ledger/inc_t2_overlay.csv、inc_t2_overlay_daily.parquet、inc_t2_overlay_byperiod.csv
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP, LEDGER
from afactor.panel import Panel
from afactor import strategy as ST, impact as IM
from afactor.strategy import TRADING_DAYS
from afactor.pead_engine import nw_se
from inc_t2_common import calendar, WINDOWS

MODE = "open10"
VERSION = "V1000"
K = 5
SIM = ("2020-07-01", "2025-12-31")
RUN_DIR = TMP / "inc_t2_overlay_runs"
NW_BW = 10
CAP_REF = {1e7: 0.159740, 1e8: 0.155909}       # capacity_grid.csv COMP_v5_exclude open10 Y1.0 pov const 2021-2025
UNIVERSES = {"O500": dict(src="O", idx=["000905"], add="ADD_800", dele="DEL_800"),
             "Oall": dict(src="O", idx=["000300", "000905"], add="ADD_800", dele="DEL_800"),
             "PA1000": dict(src="PA", idx=["000852"], add="ADD_pure", dele="DEL_pure")}
KEEP = ["ret", "cost_total", "cost_spread", "cost_impact", "turnover", "exposure"]


class ConstAUMCost:
    """与 scripts/cap_capacity_curve.py::ConstAUMCost 相同:冲击 Q = |Δw| × capital(nav 固定为 1)。"""

    def __init__(self, base):
        self.base = base

    def extra_cost(self, te, abs_delta, capital, nav):
        return self.base.extra_cost(te, abs_delta, capital, 1.0)

    def describe(self):
        d = self.base.describe()
        d["aum_mode"] = "const"
        return d


def sim_mask(p) -> np.ndarray:
    return np.asarray((p.dates >= pd.Timestamp(SIM[0])) & (p.dates <= pd.Timestamp(SIM[1])))


def baseline_W(p, m):
    comp = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    return ST.build_weights(comp, p, VERSION, scheme="exclude", n=20, buffer=(0.20, 0.35), date_mask=m)


def baskets(p, uni: str, rep: int | None):
    """每期 (A_i, Em1_i, add_js, del_js)。rep=None -> 事件股本身;rep=1..3 -> 其 A 日匹配对照。"""
    E = pd.read_parquet(TMP / "inc_t2_events.parquet")
    cfg = UNIVERSES[uni]
    e = E[(E["source"] == cfg["src"]) & (E["index"].isin(cfg["idx"])) & E["q_A"]]
    if rep is not None:
        C = pd.read_parquet(TMP / "inc_t2_controls.parquet")
        C = C[(C["t0_kind"] == "A") & (C["tol"] == "tol10") & (C["rep"] == rep)]
        cj = dict(zip(C["ev_id"], C["ctrl_j"]))
    out = []
    for rv, g in e.groupby("review"):
        def js(sel):
            x = g[sel]
            if rep is None:
                arr = x["j"].to_numpy()
            else:
                arr = np.array([cj.get(i, -1) for i in x["ev_id"]])
            return np.unique(arr[arr >= 0]).astype(np.int64)
        add = js((g["direction"] == "in") & (g["etype"] == cfg["add"]))
        dele = js((g["direction"] == "out") & (g["etype"] == cfg["dele"]))
        out.append(dict(review=rv, A=int(g["A_i"].iloc[0]), Em1=int(g["Em1_i"].iloc[0]), add=add, dele=dele))
    return out


def make_target(W: np.ndarray, t_first: int, phase: int, wins: list, variant: str, x: float):
    """variant ∈ none | A | B | C(PL-A 用 A + 对照篮子,PL-B 用 B + 对照篮子)。"""
    T = W.shape[0]
    act = np.full(T, -1, np.int64)
    entry, exit_ = {}, {}
    for wi, w in enumerate(wins):
        act[w["A"]:w["Em1"]] = wi                 # 信号日 A .. E−2
        entry[w["A"]] = wi
        exit_[w["Em1"]] = wi
    st = {"b": None}

    def fn(t, h, ctx):
        reb = ((t - t_first - phase) % K) == 0
        if reb:
            st["b"] = W[t]
        b = st["b"]
        if b is None:
            return None
        if variant == "none":
            return W[t] if reb else None
        if not (reb or t in entry or t in exit_):
            return None
        wi = act[t]
        if wi < 0:
            return np.array(b, dtype=np.float64)
        w = np.array(b, dtype=np.float64)
        win = wins[wi]
        if variant in ("B", "C") and len(win["dele"]):
            w[win["dele"]] = 0.0
            s = w.sum()
            if s > 0:
                w = w / s
        if variant in ("A", "C") and len(win["add"]):
            s = w.sum()
            w = (1.0 - x) * (w / s if s > 0 else w)
            w[win["add"]] += x / len(win["add"])
        return w
    return fn


def run_cell(p, W, m, cm, capital, wins, variant, x, record=True):
    t_first = int(np.flatnonzero(m)[0])
    rets, dailies = [], []
    for ph in range(K):
        fn = make_target(W, t_first, ph, wins, variant, x)
        r = ST.simulate(fn, p, MODE, capital, rebalance=1, date_mask=m, phase=0, cost_model=cm, record=record)
        rets.append(r["rets"])
        if record:
            dailies.append(r["record"]["daily"][KEEP])
            del r["record"]
    R = pd.concat(rets, axis=1).mean(axis=1)
    D = None
    if record:
        D = sum(d for d in dailies) / len(dailies)
        D["ret"] = R
        D["exp_max_phase"] = pd.concat([d["exposure"] for d in dailies], axis=1).max(axis=1)
    return R, D


def geo_ann(r: pd.Series) -> float:
    r = r.dropna()
    return float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0) if len(r) else np.nan


def align_profile(rs, rb, ks=(-2, -1, 0, 1, 2)):
    a, b = rs.align(rb, join="inner")
    return {k: float(a.corr(b.shift(k))) for k in ks}


# ---------------------------------------------------------------------------- Gate 1
def cmd_checks():
    t00 = time.time()
    p = Panel()
    m = sim_mask(p)
    W = baseline_W(p, m)
    W = np.where(m[:, None], W, 0.0)
    base = IM.CostModel(p, impact_y=1.0)
    cm = ConstAUMCost(base)
    rows = []
    bench = ST.benchmark(p, VERSION, MODE, rebalance=K)["rets"]
    for cap in (1e7, 1e8):
        t1 = time.time()
        ref = ST.run_weights(W, p, MODE, cap, rebalance=K, date_mask=m, cost_model=cm)
        R, D = run_cell(p, W, m, cm, cap, [], "none", 0.0)
        dmax = float(np.nanmax(np.abs(R.to_numpy() - ref["rets"].reindex(R.index).to_numpy())))
        rows.append(dict(check="B0_callable_equals_run_weights", item=f"{cap:.0e}", value=dmax, reference=0.0,
                         passed=dmax < 1e-12, note="rebalance=1 + 相位日历包装 vs run_weights(rebalance=5) 相位平均日收益最大绝对差"))
        rs = R[(R.index >= "2021-01-01")]
        a = geo_ann(rs)
        rows.append(dict(check="B0_vs_capacity_grid", item=f"{cap:.0e} 2021-2025 ann", value=a, reference=CAP_REF[cap],
                         passed=abs(a - CAP_REF[cap]) * 100 <= 0.05, note="差 ≤ 0.05pp"))
        rows.append(dict(check="exposure_le_1", item=f"B0 {cap:.0e}", value=float(D["exp_max_phase"].max()),
                         reference=1.0, passed=float(D["exp_max_phase"].max()) <= 1 + 1e-9, note="逐相位最大"))
        prof = align_profile(R, bench)
        pk = max(prof, key=prof.get)
        rows.append(dict(check="align_peak_k0", item=f"B0 {cap:.0e} vs benchmark(rebalance=5)", value=float(pk),
                         reference=0.0, passed=pk == 0, note=" ".join(f"k={k:+d}:{v:.3f}" for k, v in prof.items())))
        R.rename("ret").to_frame().to_parquet(TMP / f"inc_t2_B0_{int(cap)}.parquet")
        D.to_parquet(RUN_DIR / f"B0_{int(cap)}.parquet") if RUN_DIR.exists() else None
        print(rows[-4:], f"{time.time()-t1:.0f}s", flush=True)
    # 全池等权 vs 同频基准 ≈ −成本
    ones = pd.DataFrame(1.0, index=p.dates, columns=p.codes)
    W5 = ST.build_weights(ones, p, VERSION, scheme="exclude", exclude_frac=0.0, date_mask=m)
    r = ST.run_weights(W5, p, MODE, 1e7, rebalance=K, date_mask=m, cost_model=cm, record=True)
    rs = r["rets"][r["rets"].index.year >= 2021]
    cost = r["record"]["daily"]["cost_total"].reindex(rs.index)
    b2 = bench.reindex(rs.index)
    ex_g = (geo_ann(rs) - geo_ann(b2)) * 100
    c_pp = float(cost.mean() * TRADING_DAYS * 100)
    prof = align_profile(r["rets"], bench)
    pk = max(prof, key=prof.get)
    rows.append(dict(check="fullpool_vs_samefreq_eq_minus_cost", item="Y1.0 1e7 2021-2025", value=ex_g + c_pp,
                     reference=0.0, passed=(-0.5 <= ex_g + c_pp <= 1.5) and pk == 0 and r["exposure"] <= 1,
                     note=f"excess_geo={ex_g:.3f}pp cost={c_pp:.3f}pp peak_k={pk};判据同 cap_capacity_curve(−0.5~+1.5pp)"))
    print(rows[-1], flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(LEDGER / "inc_t2_overlay_checks.csv", index=False)
    print(C.to_string())
    print(f"done {time.time()-t00:.0f}s")
    assert C["passed"].all(), "Gate 1 未通过:修引擎前不得读收益"


# ---------------------------------------------------------------------------- 叠加层
def cell_list():
    cells = []
    for cap in (1e7, 1e8):
        for uni in UNIVERSES:
            for x in (0.05, 0.10):
                cells.append(("OVA", uni, x, None, cap))
                cells.append(("OVC", uni, x, None, cap))
                for rep in (1, 2, 3):
                    cells.append(("PLA", uni, x, rep, cap))
            cells.append(("OVB", uni, 0.0, None, cap))
            for rep in (1, 2, 3):
                cells.append(("PLB", uni, 0.0, rep, cap))
    # 预注册 §7.3"另报 10 亿资金下的增量":只跑 B0 与 OV-A 10% / OV-B(不跑安慰剂)
    cells.append(("B0", "none", 0.0, None, 1e9))
    for uni in ("O500", "PA1000"):
        cells.append(("OVA", uni, 0.10, None, 1e9))
        cells.append(("OVB", uni, 0.0, None, 1e9))
    return cells


def cell_id(c):
    v, uni, x, rep, cap = c
    if v == "B0":
        return f"B0_{int(cap)}"
    return f"{v}_{uni}_x{int(round(x*100)):02d}_r{rep or 0}_{int(cap):d}"


def cmd_run(part: str | None):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    cells = cell_list()
    if part:
        i, n = map(int, part.split("/"))
        cells = cells[i::n]
    todo = [c for c in cells if not (RUN_DIR / f"{cell_id(c)}.parquet").exists()]
    print(f"{len(todo)} / {len(cells)} 格待算", flush=True)
    if not todo:
        return
    p = Panel()
    m = sim_mask(p)
    W = np.where(m[:, None], baseline_W(p, m), 0.0)
    cm = ConstAUMCost(IM.CostModel(p, impact_y=1.0))
    bcache = {}
    for c in todo:
        v, uni, x, rep, cap = c
        t1 = time.time()
        key = (uni, rep)
        if key not in bcache:
            bcache[key] = [] if v == "B0" else baskets(p, uni, rep)
        variant = {"OVA": "A", "OVB": "B", "OVC": "C", "PLA": "A", "PLB": "B", "B0": "none"}[v]
        R, D = run_cell(p, W, m, cm, cap, bcache[key], variant, x)
        assert float(D["exp_max_phase"].max()) <= 1 + 1e-9
        D.to_parquet(RUN_DIR / f"{cell_id(c)}.parquet")
        print(f"  {cell_id(c):32s} ann(2021-25) {geo_ann(R[R.index >= '2021-01-01'])*100:7.3f}%  {time.time()-t1:.0f}s", flush=True)
    print(f"done {time.time()-t00:.0f}s")


def nw_stats(d: pd.Series) -> dict:
    d = d.dropna()
    if len(d) < 20:
        return dict(n=len(d), ann=np.nan, t=np.nan, lo=np.nan, hi=np.nan)
    m = float(d.mean()); se = nw_se(d.to_numpy(), NW_BW)
    q = d.groupby(d.index.to_period("Q")).mean()
    qt = float(q.mean() / (q.std(ddof=1) / np.sqrt(len(q)))) if len(q) > 2 and q.std(ddof=1) > 0 else np.nan
    return dict(n=len(d), ann=m * TRADING_DAYS, t=m / se if se > 0 else np.nan,
                lo=(m - 1.96 * se) * TRADING_DAYS, hi=(m + 1.96 * se) * TRADING_DAYS, t_quarter=qt)


def cmd_summarize():
    p = Panel()
    cal = calendar(p.dates)
    bench = ST.benchmark(p, VERSION, MODE, rebalance=K)["rets"]
    out, daily_all, byp = [], [], []
    for cap in (1e7, 1e8, 1e9):
        if not (RUN_DIR / f"B0_{int(cap)}.parquet").exists():
            continue
        B0 = pd.read_parquet(RUN_DIR / f"B0_{int(cap)}.parquet")
        b0 = B0["ret"]
        exb = (b0 - bench.reindex(b0.index))
        runs = {}
        for c in cell_list():
            if c[4] != cap or c[0] == "B0":
                continue
            f = RUN_DIR / f"{cell_id(c)}.parquet"
            if f.exists():
                runs[c] = pd.read_parquet(f)
        # PL 三个复制取均值
        groups = {}
        for c, D in runs.items():
            v, uni, x, rep, _ = c
            groups.setdefault((v, uni, x), []).append(D)
        for (v, uni, x), Ds in groups.items():
            r = sum(D["ret"] for D in Ds) / len(Ds)
            cost = sum(D["cost_total"] for D in Ds) / len(Ds)
            turn = sum(D["turnover"] for D in Ds) / len(Ds)
            d = (r - b0.reindex(r.index))
            pl = {"OVA": ("PLA", x), "OVC": None, "OVB": ("PLB", 0.0)}.get(v)
            dpl = None
            if pl and (pl[0], uni, pl[1]) in groups:
                Dp = groups[(pl[0], uni, pl[1])]
                rp = sum(D["ret"] for D in Dp) / len(Dp)
                dpl = d - (rp - b0.reindex(rp.index))
            daily_all.append(pd.DataFrame({"d": d, "d_net_pl": dpl if dpl is not None else np.nan,
                                           "cost_ov": cost, "cost_b0": B0["cost_total"].reindex(d.index),
                                           "turn_ov": turn, "turn_b0": B0["turnover"].reindex(d.index)})
                             .assign(variant=v, universe=uni, x=x, capital=cap))
            for wn, (a, b) in WINDOWS.items():
                sel = (d.index >= a) & (d.index <= b)
                s = nw_stats(d[sel])
                rec = dict(variant=v, universe=uni, x=x, capital=cap, window=wn, n_reps=len(Ds),
                           incr_ann_pp=s["ann"] * 100, incr_t_nw=s["t"], incr_lo_pp=s["lo"] * 100, incr_hi_pp=s["hi"] * 100,
                           incr_t_quarter=s.get("t_quarter"),
                           ann_ov=geo_ann(r[sel]), ann_b0=geo_ann(b0.reindex(r.index)[sel]),
                           incr_geo_pp=(geo_ann(r[sel]) - geo_ann(b0.reindex(r.index)[sel])) * 100,
                           cost_ov_pp=float(cost[sel].mean() * TRADING_DAYS * 100),
                           cost_b0_pp=float(B0["cost_total"].reindex(d.index)[sel].mean() * TRADING_DAYS * 100),
                           turn_ov_ann=float(turn[sel].mean() * TRADING_DAYS),
                           turn_b0_ann=float(B0["turnover"].reindex(d.index)[sel].mean() * TRADING_DAYS),
                           corr_d_b0excess=float(d[sel].corr(exb.reindex(d.index)[sel])))
                if dpl is not None:
                    s2 = nw_stats(dpl[sel])
                    rec.update(net_pl_ann_pp=s2["ann"] * 100, net_pl_t_nw=s2["t"], net_pl_lo_pp=s2["lo"] * 100,
                               net_pl_hi_pp=s2["hi"] * 100, net_pl_t_quarter=s2.get("t_quarter"))
                out.append(rec)
            # 逐期:A1 .. E(含退出日)窗口内的累计增量
            for rr in cal.itertuples():
                lo, hi = p.dates[rr.A_i], p.dates[rr.Em1_i]
                sel = (d.index >= lo) & (d.index <= hi)
                if sel.sum() == 0:
                    continue
                byp.append(dict(variant=v, universe=uni, x=x, capital=cap, review=rr.review, window=rr.window,
                                cum_incr_pp=float(((1 + r[sel]).prod() - (1 + b0.reindex(r.index)[sel]).prod()) * 100),
                                cum_incr_net_pl_pp=float(dpl[sel].sum() * 100) if dpl is not None else np.nan))
    O = pd.DataFrame(out)
    O.to_csv(LEDGER / "inc_t2_overlay.csv", index=False)
    pd.concat(daily_all).to_parquet(LEDGER / "inc_t2_overlay_daily.parquet")
    pd.DataFrame(byp).to_csv(LEDGER / "inc_t2_overlay_byperiod.csv", index=False)
    print(O[(O["window"] != "x")][["variant", "universe", "x", "capital", "window", "incr_ann_pp", "incr_t_nw",
                                   "net_pl_ann_pp", "net_pl_t_nw", "cost_ov_pp", "cost_b0_pp",
                                   "corr_d_b0excess"]].round(3).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "checks":
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        cmd_checks()
    elif cmd == "run":
        part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else None
        cmd_run(part)
    elif cmd == "summarize":
        cmd_summarize()
    else:
        print(__doc__)
