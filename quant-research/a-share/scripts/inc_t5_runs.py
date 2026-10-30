"""T5 篮子权重与引擎回测(前缀 inc_t5;预注册 §5、§11、§12 步骤 3-4)。

阶段(日志 logs/inc_t5_run.log;结果 cache/_tmp/inc_t5_runs/<轨道>/<配置>.pkl;权重 cache/_tmp/inc_t5_w/):
  weights_lh   B-LH 各篮子目标权重(2016-01-04 起连续运行,20/35 与 45/60 两档)+ 月末持仓重合 Jaccard
  weights_a    A 轨权重(全 5101 行)
  runs_lh      B-LH 全部配置(并行 --workers)
  runs_a       A 轨
  runs_min     B-MIN(Panel,V1000,open10,1000 万):不并行
  d6           D6(a):C 面板复现冻结序列
用法: python3 scripts/inc_t5_runs.py weights_lh weights_a runs_lh runs_a runs_min d6 [--workers 5] [--only a,b]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time

from inc_t5_common import *      # noqa: F401,F403
from inc_t5_common import (ROOT, SC, RUNS, TMP, LEDGER, WIN, LE, LHCtx, log, log_dev, merge_csv, slim, TD)

import numpy as np
import pandas as pd

from afactor import inc_t5_stats as S5

WDIR = TMP / "inc_t5_w"
DEPTHS = {"45_60": (0.45, 0.60), "20_35": (0.20, 0.35)}
# 篮子 -> 打分文件名(LongPanel,lh_<名>.npy)
BASKET_SCORE = {"LOWVOL": "LOWVOL120", "VALUE": "VALUE", "DIV": "DY365", "QUAL": "QUAL", "REV": "REV20"}
VARIANT_SCORE = {"LOWVOL60": "LOWVOL60", "LOWVOL250": "LOWVOL250", "REV60": "REV60", "DIV730": "DY730",
                 "DIVI": "DY365I", "VALUE_BP": "BP", "VALUE_EP": "EP", "QUAL_ROE": "Q_ROE", "QUAL_STAB": "Q_STAB",
                 "QUAL_LEV": "Q_LEVR"}
STYLE5 = ["LIQ", "LOWVOL", "VALUE", "DIV", "QUAL"]
STYLE6 = STYLE5 + ["REV"]


def _score_name(b: str) -> str:
    return BASKET_SCORE.get(b) or VARIANT_SCORE[b]


def wpath(track: str, basket: str, depth: str):
    return WDIR / f"{track}_{basket}_{depth}.npy"


def lh_basket_list() -> list[tuple[str, str]]:
    """(篮子, 深度):主篮子两档 + LIQ 两档 + 变体(45/60)。"""
    out = [(b, d) for b in ["LIQ"] + list(BASKET_SCORE) for d in DEPTHS]
    out += [(v, "45_60") for v in VARIANT_SCORE]
    return out


# ================================================================ 权重
def build_W(ctx: LHCtx, basket: str, depth: str) -> np.ndarray:
    buf = DEPTHS[depth]
    if basket == "LIQ":
        return ctx.liq_weights(buf)
    return ctx.weights(ctx.score(_score_name(basket)), buf)


def stage_weights(track: str):
    t0 = time.time()
    WDIR.mkdir(parents=True, exist_ok=True)
    ctx = LHCtx(WIN[track]["rows_from"])
    if track == "lh":
        lst = lh_basket_list()
    else:   # A 轨:LIQ(20/35)与价格类篮子(45/60)
        lst = [("LIQ", "20_35"), ("LOWVOL60", "45_60"), ("LOWVOL", "45_60"), ("LOWVOL250", "45_60"),
               ("REV", "45_60"), ("REV60", "45_60"), ("REV", "20_35")]
    Ws = {}
    for b, d in lst:
        f = wpath(track, b, d)
        W = build_W(ctx, b, d)
        np.save(f, W)
        Ws[(b, d)] = W
        n = (W > 0).sum(axis=1)
        log(f"[weights_{track}] {b}@{d} 平均持仓 {n[n > 0].mean():.0f} 空仓日 {int((n == 0).sum())} {time.time() - t0:.0f}s")
    if track == "lh":
        pool = ctx.U.to_numpy(bool)
        dates = ctx.dates
        sel = pd.Series(np.arange(len(dates)), index=dates).loc[WIN["lh"]["full"][0]:WIN["lh"]["full"][1]]
        me = sel.groupby([sel.index.year, sel.index.month]).last().to_numpy()
        names = ["LIQ|20_35"] + [f"{b}|45_60" for b in list(BASKET_SCORE)]
        rows = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                Wa = Ws[(a.split("|")[0], a.split("|")[1])]; Wb = Ws[(b.split("|")[0], b.split("|")[1])]
                rows.append(dict(a=a, b=b, jaccard=S5.jaccard_holdings(Wa, Wb, me), n_months=len(me)))
        # 篮子占池比例
        frac = []
        for nm in names:
            W = Ws[(nm.split("|")[0], nm.split("|")[1])]
            f_ = ((W[me] > 0).sum(axis=1) / pool[me].sum(axis=1))
            frac.append(dict(basket=nm, mean_frac_of_pool=float(f_.mean()), min=float(f_.min()), max=float(f_.max())))
        pd.DataFrame(rows).to_csv(LEDGER / "inc_t5_jaccard.csv", index=False)
        pd.DataFrame(frac).to_csv(LEDGER / "inc_t5_basket_size.csv", index=False)
    log(f"[weights_{track}] 完成 {time.time() - t0:.0f}s")


# ================================================================ 配置
def lh_configs() -> dict:
    cf = {}
    for b, d in lh_basket_list():
        cf[f"S|{b}|{d}"] = dict(kind="basket", basket=b, depth=d, cost="Y1")
    # 成本口径对照(主配置 CMB-EW5 与 LIQ)
    for c in ("old", "pess"):
        cf[f"S|LIQ|20_35|{c}"] = dict(kind="basket", basket="LIQ", depth="20_35", cost=c)
    combos = {"EW5": (STYLE5, "EW"), "CORE5": (STYLE5, "CORE"), "ERC5": (STYLE5, "ERC"),
              "EW6": (STYLE6, "EW"), "CORE6": (STYLE6, "CORE"), "ERC6": (STYLE6, "ERC"),
              "S4": (STYLE5[1:], "EW"), "S5": (STYLE6[1:], "EW")}
    for nm, (mem, sch) in combos.items():
        for d in DEPTHS:
            cf[f"C|{nm}|{d}"] = dict(kind="combo", members=mem, scheme=sch, depth=d, cost="Y1",
                                     record=(nm == "EW5" and d == "45_60"))
    for c in ("old", "pess"):
        cf[f"C|EW5|45_60|{c}"] = dict(kind="combo", members=STYLE5, scheme="EW", depth="45_60", cost=c)
    for s in range(10):
        cf[f"P|{s}"] = dict(kind="placebo", s=s, cost="Y1")
    cf["FP"] = dict(kind="fullpool", record=True)
    cf["S|LIQ|20_35|rec"] = dict(kind="basket", basket="LIQ", depth="20_35", cost="Y1", record=True)
    # 事后探索(看到结果后加:REV 20/35 深度的 IR 高于排雷,见 inc_t5_deviations.csv posthoc_rev_2035;不进判定)
    cf["C|LR_20_35"] = dict(kind="combo", members=["LIQ", "REV"], scheme="EW", depth="20_35", cost="Y1")
    return cf


def a_configs() -> dict:
    cf = {"S|LIQ|20_35": dict(kind="basket", basket="LIQ", depth="20_35", cost="Y1")}
    for b in ("LOWVOL60", "LOWVOL", "LOWVOL250", "REV", "REV60"):
        cf[f"S|{b}|45_60"] = dict(kind="basket", basket=b, depth="45_60", cost="Y1")
    wmap = {"LOWVOL60": "60", "LOWVOL": "120", "LOWVOL250": "250"}
    for lv, w in wmap.items():
        cf[f"C|A2_{w}"] = dict(kind="combo", members=["LIQ", lv], scheme="EW", depth="45_60", cost="Y1", track="a")
        cf[f"C|A3_{w}"] = dict(kind="combo", members=["LIQ", lv, "REV"], scheme="EW", depth="45_60", cost="Y1", track="a")
        cf[f"C|COREA2_{w}"] = dict(kind="combo", members=["LIQ", lv], scheme="CORE23", depth="45_60", cost="Y1", track="a")
    for s in range(10):
        cf[f"P|{s}"] = dict(kind="placebo_a", s=s, cost="Y1")
    # 事后探索(同上;2006-2016 对该假说是样本外):REV 20/35 单独与 LIQ+REV 等权
    cf["S|REV|20_35"] = dict(kind="basket", basket="REV", depth="20_35", cost="Y1")
    cf["C|LR_20_35"] = dict(kind="combo", members=["LIQ", "REV"], scheme="EW", depth="20_35", cost="Y1", track="a")
    cf["FP"] = dict(kind="fullpool", record=True)
    cf["S|LIQ|20_35|rec"] = dict(kind="basket", basket="LIQ", depth="20_35", cost="Y1", record=True)
    return cf


def _basket_depth(m: str, d: str) -> tuple[str, str]:
    return (m, "20_35") if m == "LIQ" else (m, d)


def _load_W(track: str, b: str, d: str) -> np.ndarray:
    return np.load(wpath(track, b, d), mmap_mode="r")


def _alpha(ctx: LHCtx, track: str, spec: dict, Ws: list[np.ndarray], mem: list[str]) -> np.ndarray | list:
    K = len(mem)
    sch = spec["scheme"]
    if sch == "EW":
        return np.full(K, 1.0 / K)
    if sch == "CORE":
        return S5.core_weights(K, mem.index("LIQ"), 0.5)
    if sch == "CORE23":
        a = np.full(K, 1.0 / 3); a[mem.index("LIQ")] = 2.0 / 3
        return a
    if sch == "ERC":
        b1 = ctx.b1().reindex(ctx.dates)
        ex = {}
        for m, W in zip(mem, Ws):
            d_ = "20_35" if m == "LIQ" else spec["depth"]
            r = load_run(track, f"S|{m}|{d_}")["rets"].reindex(ctx.dates)
            e = r - b1
            has = np.asarray(W.sum(axis=1) > 1e-12)
            ex[m] = e.where(has)
        return S5.erc_weights(pd.DataFrame(ex), 250, 2)
    raise ValueError(sch)


def run_job(track: str, name: str) -> dict:
    t0 = time.time()
    cf = lh_configs() if track == "lh" else a_configs()
    spec = cf[name]
    ctx = _CTX.get(track) or _CTX.setdefault(track, LHCtx(WIN[track]["rows_from"]))
    kind = spec["kind"]
    extra = {}
    if kind == "basket":
        W = np.asarray(_load_W(track, spec["basket"], spec["depth"]))
    elif kind == "combo":
        mem = spec["members"]
        Ws = [np.asarray(_load_W(track, *_basket_depth(m, spec["depth"]))) for m in mem]
        a = _alpha(ctx, track, spec, Ws, mem)
        W = S5.combine_weights(Ws, a)
        if spec["scheme"] == "ERC":
            extra["erc_alpha"] = pd.DataFrame(a, index=ctx.dates, columns=mem)
    elif kind == "placebo":
        s = spec["s"]
        Ws = [np.asarray(_load_W(track, "LIQ", "20_35"))]
        for j in range(4):
            Ws.append(ctx.weights(ctx.rand_score(1000 * s + j), DEPTHS["45_60"]))
        W = S5.combine_weights(Ws, np.full(5, 0.2))
    elif kind == "placebo_a":
        s = spec["s"]
        Ws = [np.asarray(_load_W(track, "LIQ", "20_35")), ctx.weights(ctx.rand_score(1000 * s + 9), DEPTHS["45_60"])]
        W = S5.combine_weights(Ws, np.full(2, 0.5))
    elif kind == "fullpool":
        W = ctx.fullpool_W()
    else:
        raise ValueError(kind)
    res = ctx.run(W, cost=spec.get("cost", "Y1"), record=spec.get("record", False))
    out = slim(res)
    out.update(extra)
    out["spec"] = {k: v for k, v in spec.items() if k != "members"} | ({"members": spec["members"]} if "members" in spec else {})
    out["elapsed"] = time.time() - t0
    d = RUNS / track
    d.mkdir(parents=True, exist_ok=True)
    with open(d / (name.replace("|", "__") + ".pkl"), "wb") as f:
        pickle.dump(out, f)
    return dict(name=name, elapsed=out["elapsed"])


_CTX: dict = {}


def load_run(track: str, name: str) -> dict:
    with open(RUNS / track / (name.replace("|", "__") + ".pkl"), "rb") as f:
        return pickle.load(f)


def stage_runs(track: str, workers: int, only: list | None):
    import concurrent.futures as cf
    import multiprocessing as mp
    t0 = time.time()
    names = list((lh_configs() if track == "lh" else a_configs()))
    if only:
        names = [n for n in names if n in only]
    # 先单独篮子(ERC 依赖),再其余
    first = [n for n in names if n.startswith("S|") or n == "FP"]
    rest = [n for n in names if n not in first]
    log(f"[runs_{track}] {len(names)} 个配置,{workers} 进程")
    for batch in (first, rest):
        if not batch:
            continue
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(run_job, track, n): n for n in batch}
            for i, f in enumerate(cf.as_completed(futs), 1):
                try:
                    r = f.result()
                    log(f"[runs_{track}] {r['name']} {r['elapsed']:.0f}s ({i}/{len(batch)})")
                except Exception as e:      # noqa: BLE001
                    log(f"[runs_{track}] FAILED {futs[f]}: {type(e).__name__} {e}")
                    raise
    log(f"[runs_{track}] 完成 {time.time() - t0:.0f}s")


# ================================================================ B-MIN(Panel)
class MinCtx:
    def __init__(self):
        from afactor import strategy as ST
        from afactor.panel import Panel
        from afactor.impact import CostModel
        self.ST = ST
        self.p = Panel()
        self.dates, self.codes = self.p.dates, self.p.codes
        self.mask = np.asarray((self.dates >= "2021-01-01") & (self.dates <= "2025-12-31"))
        self.U = ST.universe(self.p, "V1000")
        self.cm = CostModel(self.p, impact_y=1.0)
        self.b1 = ST.benchmark(self.p, "V1000", "open10", date_mask=self.mask, rebalance=5)["rets"]
        self.liq = pd.read_parquet(TMP / "comp_v5_score_open10.parquet").reindex(index=self.dates, columns=self.codes)

    def score(self, name: str) -> pd.DataFrame:
        if name == "LIQ":
            return self.liq
        return pd.read_parquet(SC / f"min_{name}.parquet").reindex(index=self.dates, columns=self.codes)

    def W(self, name: str, depth: str) -> np.ndarray:
        return self.ST.build_weights(self.score(name), self.p, "V1000", scheme="exclude", buffer=DEPTHS[depth],
                                     date_mask=self.mask)

    def run(self, W: np.ndarray, cost: str = "Y1", record: bool = False) -> dict:
        cm = self.cm if cost == "Y1" else None
        return self.ST.run_weights(W, self.p, "open10", 1e7, rebalance=5, date_mask=self.mask, cost_model=cm,
                                   record=record)


def stage_runs_min():
    t0 = time.time()
    c = MinCtx()
    log(f"[runs_min] Panel 载入 {time.time() - t0:.0f}s")
    d = RUNS / "min"
    d.mkdir(parents=True, exist_ok=True)

    def save(name, res, **extra):
        out = {k: res[k] for k in ("rets", "turnover", "cost_bps", "exposure", "holdings") if k in res}
        out.update(extra)
        with open(d / (name.replace("|", "__") + ".pkl"), "wb") as f:
            pickle.dump(out, f)

    Wb = {}
    for b in STYLE6:
        dep = "20_35" if b == "LIQ" else "45_60"
        Wb[b] = c.W(b, dep)
        r = c.run(Wb[b])
        save(f"S|{b}|{dep}", r)
        log(f"[runs_min] S|{b}|{dep} {time.time() - t0:.0f}s")
    # 风格篮子 20/35 变体不跑(预注册 B-MIN 只要 45/60)
    ex = {}
    combos = {"EW5": (STYLE5, "EW"), "CORE5": (STYLE5, "CORE"), "ERC5": (STYLE5, "ERC"),
              "EW6": (STYLE6, "EW"), "CORE6": (STYLE6, "CORE"), "ERC6": (STYLE6, "ERC"), "S4": (STYLE5[1:], "EW")}
    b1 = c.b1.reindex(c.dates)
    for nm, (mem, sch) in combos.items():
        Ws = [Wb[m] for m in mem]
        if sch == "EW":
            a = np.full(len(mem), 1.0 / len(mem))
        elif sch == "CORE":
            a = S5.core_weights(len(mem), mem.index("LIQ"), 0.5)
        else:
            exs = {}
            for m, W in zip(mem, Ws):
                dep = "20_35" if m == "LIQ" else "45_60"
                with open(d / f"S__{m}__{dep}.pkl", "rb") as f:
                    r = pickle.load(f)["rets"].reindex(c.dates)
                exs[m] = (r - b1).where(np.asarray(W.sum(axis=1) > 1e-12))
            a = S5.erc_weights(pd.DataFrame(exs), 250, 2)
        W = S5.combine_weights(Ws, a)
        r = c.run(W, record=(nm == "EW5"))
        extra = {"erc_alpha": pd.DataFrame(a, index=c.dates, columns=mem)} if sch == "ERC" else {}
        if nm == "EW5":
            extra["rec_daily"] = r["record"]["daily"][["ret", "cost_total", "cost_commission", "cost_stamp",
                                                       "cost_spread", "cost_impact", "turnover", "exposure"]]
            extra["max_exposure_phase"] = float(r["record"]["daily_by_phase"]["exposure"].max())
        save(f"C|{nm}|45_60", r, **extra)
        log(f"[runs_min] C|{nm} {time.time() - t0:.0f}s")
    # 安慰剂 5 个(LIQ + 4 随机)
    for s in range(5):
        Ws = [Wb["LIQ"]]
        for j in range(4):
            sc = np.random.default_rng(1000 * s + j).random(len(c.codes)).astype(np.float32)
            S = pd.DataFrame(np.tile(sc, (len(c.dates), 1)), index=c.dates, columns=c.codes)
            Ws.append(c.ST.build_weights(S, c.p, "V1000", scheme="exclude", buffer=DEPTHS["45_60"], date_mask=c.mask))
        save(f"P|{s}", c.run(S5.combine_weights(Ws, np.full(5, 0.2))))
        log(f"[runs_min] P|{s} {time.time() - t0:.0f}s")
    # 成本对照:LIQ 与 EW5 旧口径
    save("S|LIQ|20_35|old", c.run(Wb["LIQ"], cost="old"))
    save("C|EW5|45_60|old", c.run(S5.combine_weights([Wb[m] for m in STYLE5], np.full(5, 0.2)), cost="old"))
    # 全池等权(D5)
    Ua = c.U.to_numpy(bool) & c.mask[:, None]
    n = Ua.sum(1, keepdims=True)
    Wf = np.divide(Ua.astype(np.float32), n, out=np.zeros(Ua.shape, np.float32), where=n > 0)
    r = c.run(Wf, record=True)
    save("FP", r, rec_daily=r["record"]["daily"][["ret", "cost_total", "exposure"]],
         max_exposure_phase=float(r["record"]["daily_by_phase"]["exposure"].max()))
    # 月末持仓 Jaccard
    sel = pd.Series(np.arange(len(c.dates)), index=c.dates).loc["2021-01-04":"2025-12-31"]
    me = sel.groupby([sel.index.year, sel.index.month]).last().to_numpy()
    names = ["LIQ"] + [b for b in STYLE6[1:]]
    rows = []
    for i, a_ in enumerate(names):
        for b_ in names[i + 1:]:
            rows.append(dict(a=a_, b=b_, jaccard=S5.jaccard_holdings(Wb[a_], Wb[b_], me), n_months=len(me)))
    pd.DataFrame(rows).to_csv(LEDGER / "inc_t5_jaccard_min.csv", index=False)
    log(f"[runs_min] 完成 {time.time() - t0:.0f}s")


# ================================================================ D6(a)
def stage_d6():
    """C 面板(5 分钟信号 + 5 分钟执行)上用同一流水线复现冻结序列(冻结口径 = lh_step1 bridge 的 E5 检验)。"""
    import m5_bridge as MB
    from afactor import strategy as ST, minute5 as M5, factors as FA
    from afactor.impact import CostModel
    FA.build_all()
    p5 = M5.m5_panel(signal=True, exec_=True)
    s5 = pd.read_parquet(MB.SCORE5).reindex(index=p5.dates, columns=p5.codes)
    m5 = MB.year_mask(p5.dates, 2021, 2025)
    cm5 = CostModel(p5, impact_y=1.0)
    W = ST.build_weights(s5, p5, "V1000", date_mask=m5, scheme="exclude", buffer=(0.20, 0.35))
    D = pd.read_parquet(LEDGER / "freeze_ref_daily.parquet")
    r = ST.run_weights(W, p5, "open10", 1e7, rebalance=5, date_mask=m5, cost_model=cm5)["rets"]
    b = ST.benchmark(p5, "V1000", "open10", date_mask=m5, rebalance=5)["rets"]

    def mad(a, b_):
        a, b_ = a.align(b_, join="outer")
        return float((a - b_).abs().max())
    d1 = mad(r, D["T1_excl|Y1|strategy"].dropna())
    d2 = mad(b, D["C|bench_pool5"].dropna())
    # A 口径(Panel + comp_v5_score_open10)与冻结序列的差(只报告)
    c = MinCtx()
    WA = c.W("LIQ", "20_35")
    rA = c.run(WA)["rets"]
    d3 = mad(rA, D["T1_excl|Y1|strategy"].dropna())
    d4 = mad(c.b1, D["C|bench_pool5"].dropna())
    rows = [dict(gate="D6a", item="C面板_COMP_v5排雷_vs_冻结_maxabs", value=d1, threshold="<= 1e-10", ok=d1 <= 1e-10, note="m5_panel 流水线"),
            dict(gate="D6a", item="C面板_B1_vs_冻结_maxabs", value=d2, threshold="<= 1e-10", ok=d2 <= 1e-10, note=""),
            dict(gate="D6b", item="A面板_COMP_v5排雷_vs_冻结_maxabs", value=d3, threshold="report", ok=True,
                 note="Panel()+comp_v5_score_open10 与 C 口径冻结序列的差,只报告"),
            dict(gate="D6b", item="A面板_B1_vs_冻结_maxabs", value=d4, threshold="report", ok=True, note="")]
    merge_csv(LEDGER / "inc_t5_data_qa.csv", pd.DataFrame(rows), ["gate", "item"])
    log(f"[d6] C 面板 排雷 {d1:.2e} B1 {d2:.2e};A 面板 排雷 {d3:.2e} B1 {d4:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None
    for s in a.stages:
        if s == "weights_lh":
            stage_weights("lh")
        elif s == "weights_a":
            stage_weights("a")
        elif s == "runs_lh":
            stage_runs("lh", a.workers, only)
        elif s == "runs_a":
            stage_runs("a", a.workers, only)
        elif s == "runs_min":
            stage_runs_min()
        elif s == "d6":
            stage_d6()
        else:
            raise SystemExit(f"unknown stage {s}")
