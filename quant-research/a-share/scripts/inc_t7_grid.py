"""INC-T7 计算脚本:排雷的调仓频率 k × 缓冲带 (enter, exit) 成本-超额前沿(前缀 inc_t7)。

预注册 research/INC_T7_PREREG.md(2026-10-11 04:02 JST 冻结,哈希 research/ledger/inc_t7_prereg_manifest.json)。
本脚本只做模拟与落盘中间结果(cache/_tmp/inc_t7_*);账本 / 判定 / 图在 scripts/inc_t7_report.py。
实现细节与预注册的差异逐条登记在 research/ledger/inc_t7_deviations.csv(D05 等)。

两块面板(§3):
  minute  研究期分钟聚合面板 Panel(),COMP_v5 冻结打分,V1000 池,open10 成交,1000 万恒定 AUM(ConstAUMCost),
          2020-07-01 起连续模拟(2020H2 预热),报告 2021-2025。
  lh      长历史 LongPanel 输入(cache/_tmp/lh_step1_inputs,含 293 只退市股),S1 = -turn20,q92 池,
          日开盘价成交,1000 万初始随净值滚动,史实印花税(LE.StampFn),2005-01-04 起连续运行,报告 2006-2025。

阶段(按顺序;重任务后台跑,日志 logs/inc_t7_*.log):
  verify            冻结哈希核对(scripts/inc_t7_freeze.py verify)
  repro             §4.5 复现断言 ①②(先于网格,任一不过即停,写 cache/_tmp/inc_t7_repro.json)
  grid  <panel>     8 带 × k∈{1,5,10,20}(跑满相位)× {Y1, old, pess} + 全池等权 fp_k + 同频基准
  event <panel>     E1 θ∈{3,6,10,15}% / E2b k_eq∈{10,20} × 5 个起点偏移 × 3 成本 + E1 匹配基准
  placebo           分钟面板 5 个随机打分(20 日均值噪声)× 32 格 × Y1
  sens  <panel>     sampled 语义(k 10/20 × 20/35)、长历史恒定 AUM、资金 1e8 / 1e9
  decay <panel>     机制诊断:剔除集 Jaccard、打分秩自相关、Rank IC 随持有期、剔除集收益差

用法:PYTHONPATH=src python3 scripts/inc_t7_grid.py <stage> [panel] [--workers 2]
只读 <= 2025-12-31 的数据(config.DATA_CUTOFF 默认截断;本脚本不调用 set_data_cutoff,不读 sealed / forward)。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
from afactor.config import LEDGER, TMP, RESEARCH_END

TD = 242
BANDS = [(0.20, 0.20), (0.20, 0.30), (0.20, 0.35), (0.20, 0.45),
         (0.30, 0.30), (0.30, 0.40), (0.30, 0.45), (0.30, 0.55)]
KS = [1, 5, 10, 20]
COSTS = ["Y1", "old", "pess"]
CUR_BAND, CUR_K = (0.20, 0.35), 5
MIN_START = "2020-07-01"
CAPITAL = 1e7
OFFSETS = [0, 1, 2, 3, 4]
EVENTS = {"E1_3": ("E1", 0.03), "E1_6": ("E1", 0.06), "E1_10": ("E1", 0.10), "E1_15": ("E1", 0.15),
          "E2b_10": ("E2b", 10), "E2b_20": ("E2b", 20)}
KEEP = ["ret", "gross_ret", "cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact",
        "turnover", "exposure", "cash", "n_hold", "stuck_w", "part_cut"]
LOGDIR = ROOT / "logs"
OUT = TMP


def btag(b) -> str:
    return "fp" if b is None else f"{int(round(b[0] * 100))}/{int(round(b[1] * 100))}"


def cell_name(k, band, cost, extra: str = "") -> str:
    return f"k{k}|{btag(band)}|{cost}" + (f"|{extra}" if extra else "")


_LOGF = None


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    if _LOGF is not None:
        with open(_LOGF, "a") as f:
            f.write(s + "\n")


# ====================================================================== 面板上下文(每个进程一份)
class Ctx:
    """两块面板的统一接口:weights / fp_weights / cost / bench / stamp_fn / mask。"""

    def __init__(self, panel: str):
        self.panel = panel
        self._models: dict = {}
        self._bench: dict = {}
        if panel == "minute":
            from afactor.panel import Panel
            self.p = Panel()
            d = self.p.dates
            self.mask = np.asarray((d >= pd.Timestamp(MIN_START)) & (d <= pd.Timestamp(RESEARCH_END)))
            self.t_start = int(np.flatnonzero(self.mask)[0])
            self.score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
            self.stamp_fn = None
            self.U = ST.universe(self.p, "V1000").reindex(index=d, columns=self.p.codes).fillna(False)
        elif panel == "lh":
            self.X = LE.load_inputs()
            self.p = LE.adapter(self.X)
            self.mask = None
            self.t_start = 0
            self.score = None                      # LE.weights 默认读 score_S1
            self.stamp_fn = LE.StampFn(self.p.dates, extra_bps=0.0)
            self.U = LE.pool(self.X, "q92")
        else:
            raise ValueError(panel)
        self.T, self.N = len(self.p.dates), len(self.p.codes)

    # ---- 目标权重 ----
    def weights(self, band, mask="default", score=None) -> np.ndarray:
        m = self.mask if (isinstance(mask, str) and mask == "default") else mask
        if self.panel == "minute":
            s = self.score if score is None else score
            return ST.build_weights(s, self.p, "V1000", scheme="exclude", buffer=band, date_mask=m)
        return LE.weights(self.X, "S1", buffer=band, score=score, date_mask=m)

    def fp_weights(self, mask="default") -> np.ndarray:
        m = self.mask if (isinstance(mask, str) and mask == "default") else mask
        Ua = self.U.to_numpy(bool)
        if m is not None:
            Ua = Ua & m[:, None]
        n = Ua.sum(1, keepdims=True)
        return np.divide(Ua.astype(np.float32), n, out=np.zeros(Ua.shape, np.float32), where=n > 0)

    def offset_mask(self, o: int) -> np.ndarray:
        return np.arange(self.T) >= (self.t_start + int(o))

    def raw_score(self) -> pd.DataFrame:
        if self.panel == "minute":
            return self.score.reindex(index=self.p.dates, columns=self.p.codes)
        return self.X.df("score_S1")

    # ---- 成本 ----
    def cost(self, kind: str, const: bool | None = None):
        """minute 默认恒定 AUM(容量报告口径);lh 默认随净值滚动(lh_step1 口径)。"""
        if kind == "old":
            return None
        if kind not in self._models:
            if self.panel == "minute":
                from afactor import impact as IM
                y, sm = {"Y1": (1.0, 1.0), "pess": (1.5, 2.0)}[kind]
                self._models[kind] = IM.CostModel(self.p, impact_y=y, spread_mult=sm)
            else:
                self._models[kind] = LE.cost_model(self.p, kind)
        base = self._models[kind]
        use_const = (self.panel == "minute") if const is None else bool(const)
        if use_const:
            import cap_capacity_curve as CAP
            return CAP.ConstAUMCost(base)
        return base

    # ---- 基准 ----
    def bench(self, k: int) -> pd.Series:
        if k not in self._bench:
            if self.panel == "minute":
                self._bench[k] = ST.benchmark(self.p, "V1000", "open10", rebalance=k)["rets"]
            else:
                self._bench[k] = LE.bench(self.p, self.U, k)
        return self._bench[k]

    def exec_px_pool(self):
        px, _ = ST.exec_prices(self.p, "open10")
        P = px.reindex(index=self.p.dates, columns=self.p.codes).to_numpy(np.float64)
        U = self.U.reindex(index=self.p.dates, columns=self.p.codes).fillna(False).to_numpy(bool)
        return P, U


_CTX: Ctx | None = None


def _init(panel: str):
    global _CTX
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _CTX = Ctx(panel)


def ctx() -> Ctx:
    assert _CTX is not None
    return _CTX


# ====================================================================== 相位循环(D05)
def run_cell(c: Ctx, target, k: int, cm, capital: float = CAPITAL, mask="default", normalize: bool = True,
             target_by_phase=None):
    """逐相位 ST.simulate(record=True) -> 丢 holdings -> ST._aggregate_phases。参数与 run_weights / LE.run 相同。
    target_by_phase(ph) 给定时每个相位用自己的目标(sampled 语义敏感性)。"""
    m = c.mask if (isinstance(mask, str) and mask == "default") else mask
    k = max(int(k), 1)
    phs = list(range(k)) if k > 1 else [0]
    subs, dailies = [], []
    for ph in phs:
        tg = target_by_phase(ph) if target_by_phase is not None else target
        r = ST.simulate(tg, c.p, "open10", capital, rebalance=k, date_mask=m, phase=ph, cost_model=cm,
                        record=True, normalize=normalize, stamp_fn=c.stamp_fn)
        rec = r.pop("record")
        d = rec["daily"][KEEP + ["rebalance"]].copy()
        d["phase"] = ph
        del rec
        if r.get("rets") is None or len(r["rets"]) <= 5:
            continue
        subs.append(r); dailies.append(d)
        del tg
    out = ST._aggregate_phases(subs) if k > 1 else subs[0]
    by = pd.concat(dailies)
    g = by.groupby(level=0)
    daily = g[KEEP].mean()
    daily["n_reb"] = g["rebalance"].sum().astype(float)
    daily["exp_max_phase"] = g["exposure"].max()
    dret = float(np.nanmax(np.abs(daily["ret"].to_numpy() - out["rets"].reindex(daily.index).to_numpy())))
    daily["ret"] = out["rets"].reindex(daily.index)
    out["_dret"] = dret
    out["_phase_ann"] = [float(x["ann_return"]) for x in subs]
    return out, daily


def align_peak(rs: pd.Series, rb: pd.Series, ks=(-2, -1, 0, 1, 2)) -> tuple[int, dict]:
    a, b = rs.align(rb, join="inner")
    cs = {int(j): float(a.corr(b.shift(j))) for j in ks}
    return max(cs, key=cs.get), cs


def cell_meta(c: Ctx, name: str, kind: str, band, k, cost, capital, res, daily, bench_k: pd.Series | None,
              t_el: float, **extra) -> dict:
    pk, cs = align_peak(res["rets"], bench_k) if bench_k is not None else (np.nan, {})
    exp_max = float(daily["exp_max_phase"].max())
    meta = dict(panel=c.panel, cell=name, kind=kind, band=btag(band) if kind != "event" else "20/35",
                enter=(band[0] if band else np.nan), exit=(band[1] if band else np.nan),
                k=k, cost=cost, capital=capital, n_phases=res.get("n_phases", 1),
                turnover_daily=res["turnover"], cost_bps_daily=res["cost_bps"], holdings=res["holdings"],
                exposure=res["exposure"], exp_max=exp_max, align_peak_k=pk,
                align_corr0=cs.get(0, np.nan), rec_ret_maxdiff=res["_dret"],
                phase_ann_min=float(np.min(res["_phase_ann"])), phase_ann_max=float(np.max(res["_phase_ann"])),
                elapsed_s=round(t_el, 2))
    meta.update(extra)
    if exp_max > 1.0 + 1e-9 or res["exposure"] > 1.0 + 1e-9:
        meta["assert_fail"] = f"exposure>1 ({exp_max})"
    elif bench_k is not None and pk != 0:
        meta["assert_fail"] = f"align peak {pk}"
    elif res["_dret"] > 1e-12:
        meta["assert_fail"] = f"record.ret != rets ({res['_dret']})"
    else:
        meta["assert_fail"] = ""
    return meta


# ====================================================================== 作业
def job_grid(spec: dict):
    c = ctx()
    band = spec["band"]
    W = c.fp_weights() if band is None else c.weights(band)
    out = []
    for k in spec["ks"]:
        bk = c.bench(k)
        for cost in spec["costs"]:
            t1 = time.time()
            res, daily = run_cell(c, W, k, c.cost(cost))
            name = cell_name(k, band, cost)
            meta = cell_meta(c, name, "fp" if band is None else "grid", band, k, cost, CAPITAL, res, daily, bk,
                             time.time() - t1)
            out.append((meta, daily))
            print(f"  [{c.panel}] {name} ann={res['ann_return']:.4f} to={res['turnover']:.4f} "
                  f"{time.time() - t1:.1f}s {meta['assert_fail']}", flush=True)
    return out


def job_bench(spec: dict):
    c = ctx()
    return pd.DataFrame({f"b{k}": c.bench(k) for k in spec["ks"]})


class E1Target:
    """E1 换手阈值触发:τ_t = ½Σ|W_t − h_dec| ≥ θ 时全量调到 W_t,否则不交易(只用决策时点信息)。"""

    def __init__(self, W: np.ndarray, theta: float):
        self.W, self.theta = W, float(theta)

    def __call__(self, t, h, ctx_):
        w = self.W[t].astype(np.float64)
        tau = 0.5 * float(np.abs(w - ctx_["h_dec"]).sum())
        return w if tau >= self.theta else None


class E2bTarget:
    """E2b 只交易状态变化 + 每 k_eq 个信号日全量再平衡(normalize=False 路径,D06)。
    增量日:应剔除仍持有 -> 0;应持有未持有 -> 1/n_t;其余 NaN(不动)。全量日:W_t 限于可成交、缩放到 budget_dec。"""

    def __init__(self, W: np.ndarray, keq: int, t_start: int, held_tol: float = 1e-10):
        self.W, self.keq, self.t_start, self.tol = W, int(keq), int(t_start), held_tol

    def __call__(self, t, h, ctx_):
        w = self.W[t].astype(np.float64)
        if (t - self.t_start) % self.keq == 0:
            tgt = np.where(ctx_["can"], w, 0.0)
            s = tgt.sum()
            return tgt / s * ctx_["budget_dec"] if s > 0 else tgt
        hd = ctx_["h_dec"]
        held = hd > self.tol
        want = w > 0
        sell = held & ~want
        buy = ~held & want
        if not (sell.any() or buy.any()):
            return None
        tgt = np.full(w.shape, np.nan)
        tgt[sell] = 0.0
        tgt[buy] = 1.0 / max(int(want.sum()), 1)
        return tgt


def pool_ew_custom(P: np.ndarray, U: np.ndarray, reset: np.ndarray, lag: int = 1) -> np.ndarray:
    """单路径池等权基准:reset[t]=True 的信号日按信号日池在有持有收益的股票上重置等权,其余日漂移
    (与 strategy._pool_ew_drift_returns 单相位同一逻辑,只把相位日历换成给定的重置日;D07)。"""
    T, N = P.shape
    n_sig = T - 1 - lag
    with np.errstate(invalid="ignore", divide="ignore"):
        G = P[1 + lag:1 + lag + n_sig] / P[lag:lag + n_sig] - 1.0
    w = np.zeros(N)
    R = np.zeros(n_sig)
    for t in range(n_sig):
        g = G[t]
        fin = np.isfinite(g)
        gz = np.where(fin, g, 0.0)
        if reset[t] or w.sum() <= 0:
            m = U[t] & fin
            n = int(m.sum())
            w = np.where(m, 1.0 / n, 0.0) if n > 0 else np.zeros(N)
        r = float((w * gz).sum())
        R[t] = r
        if 1.0 + r > 1e-9:
            w = w * (1.0 + gz) / (1.0 + r)
    return R


def job_event(spec: dict):
    c = ctx()
    vname, off = spec["variant"], spec["offset"]
    kind, par = EVENTS[vname]
    m = c.offset_mask(off)
    t_start = int(np.flatnonzero(m)[0])
    W = c.weights(CUR_BAND, mask=m)
    fn = E1Target(W, par) if kind == "E1" else E2bTarget(W, par, t_start)
    P = U = None
    out = []
    for cost in spec["costs"]:
        t1 = time.time()
        res, daily = run_cell(c, fn, 1, c.cost(cost), mask=m, normalize=(kind == "E1"))
        mb = None
        if kind == "E1":
            if P is None:
                P, U = c.exec_px_pool()
            reset = np.zeros(c.T - 2, bool)
            pos = c.p.dates.get_indexer(daily.index[daily["n_reb"] > 0])
            reset[pos] = True
            mb = pd.Series(pool_ew_custom(P, U, reset), index=c.p.dates[:c.T - 2])
            bk = mb
        else:
            bk = c.bench(int(par))
        name = f"{vname}|o{off}|{cost}"
        n_full = (int((daily["n_reb"] > 0).sum()) if kind == "E1"
                  else int(sum(1 for t in c.p.dates.get_indexer(daily.index) if (t - t_start) % int(par) == 0)))
        meta = cell_meta(c, name, "event", CUR_BAND, 1, cost, CAPITAL, res, daily, bk, time.time() - t1,
                         variant=vname, offset=off, n_full_rebal=n_full,
                         n_trade_days=int((daily["n_reb"] > 0).sum()))
        out.append((meta, daily, mb))
        print(f"  [{c.panel}] {name} ann={res['ann_return']:.4f} to={res['turnover']:.4f} "
              f"{time.time() - t1:.1f}s {meta['assert_fail']}", flush=True)
    return out


def placebo_score(c: Ctx, seed: int) -> pd.DataFrame:
    """D01:iid N(0,1) 噪声的 20 日滚动均值(min_periods 10)。"""
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal((c.T, c.N)).astype(np.float32)
    s = pd.DataFrame(z, index=c.p.dates, columns=c.p.codes).rolling(20, min_periods=10).mean().astype(np.float32)
    return s


def job_placebo(spec: dict):
    c = ctx()
    seed = spec["seed"]
    S = placebo_score(c, seed)
    out = []
    for band in BANDS:
        W = c.weights(band, score=S)
        for k in KS:
            t1 = time.time()
            res, daily = run_cell(c, W, k, c.cost("Y1"))
            name = cell_name(k, band, "Y1", f"pl{seed}")
            meta = cell_meta(c, name, "placebo", band, k, "Y1", CAPITAL, res, daily, c.bench(k), time.time() - t1,
                             seed=seed)
            out.append((meta, daily[["ret", "gross_ret", "cost_total", "turnover", "exposure", "exp_max_phase"]]))
        del W
        print(f"  [placebo s{seed}] {btag(band)} done", flush=True)
    return out


def job_sens(spec: dict):
    c = ctx()
    out = []
    kind = spec["kind"]
    if kind == "sampled":
        k, band = spec["k"], spec["band"]
        S0 = c.raw_score()
        S0v = S0.to_numpy(np.float32)
        t0 = c.t_start

        def tbp(ph):
            rows = (np.arange(c.T) - t0 - ph) % k != 0
            Sv = S0v.copy()
            Sv[rows] = np.nan
            S = pd.DataFrame(Sv, index=S0.index, columns=S0.columns)
            return c.weights(band, score=S)
        t1 = time.time()
        res, daily = run_cell(c, None, k, c.cost("Y1"), target_by_phase=tbp)
        name = cell_name(k, band, "Y1", "sampled")
        out.append((cell_meta(c, name, "sens_sampled", band, k, "Y1", CAPITAL, res, daily, c.bench(k),
                              time.time() - t1), daily))
    elif kind == "aum":
        cap, const = spec["capital"], spec.get("const", True)
        for (k, band) in spec["cells"]:
            W = c.weights(band)
            for cost in spec.get("costs", ["Y1"]):
                t1 = time.time()
                res, daily = run_cell(c, W, k, c.cost(cost, const=const), capital=cap)
                tag = f"aum{cap:.0e}" + ("c" if const else "r")
                name = cell_name(k, band, cost, tag)
                out.append((cell_meta(c, name, "sens_aum", band, k, cost, cap, res, daily, c.bench(k),
                                      time.time() - t1, const_aum=const), daily))
                print(f"  [{c.panel}] {name} ann={res['ann_return']:.4f} {time.time() - t1:.1f}s", flush=True)
            del W
    elif kind == "aum_event":
        cap, const = spec["capital"], spec.get("const", True)
        vname = spec["variant"]
        k_, par = EVENTS[vname]
        P, U = c.exec_px_pool()
        for off in OFFSETS:
            m = c.offset_mask(off)
            t_start = int(np.flatnonzero(m)[0])
            W = c.weights(CUR_BAND, mask=m)
            fn = E1Target(W, par) if k_ == "E1" else E2bTarget(W, par, t_start)
            t1 = time.time()
            res, daily = run_cell(c, fn, 1, c.cost("Y1", const=const), capital=cap, mask=m,
                                  normalize=(k_ == "E1"))
            if k_ == "E1":
                reset = np.zeros(c.T - 2, bool)
                reset[c.p.dates.get_indexer(daily.index[daily["n_reb"] > 0])] = True
                mb = pd.Series(pool_ew_custom(P, U, reset), index=c.p.dates[:c.T - 2])
            else:
                mb = None                      # E2b 的匹配基准 = k_eq 同频基准(报告阶段取 bench(k_eq),与 event 阶段同法)
            tag = f"aum{cap:.0e}" + ("c" if const else "r")
            name = f"{vname}|o{off}|Y1|{tag}"
            out.append((cell_meta(c, name, "sens_aum_event", CUR_BAND, 1, "Y1", cap, res, daily, mb,
                                  time.time() - t1, variant=vname, offset=off, const_aum=const), daily, mb))
            print(f"  [{c.panel}] {name} {time.time() - t1:.1f}s", flush=True)
    return out


# ====================================================================== 机制诊断
def job_decay(spec: dict):
    """§6.3:剔除集(分位 ≤ 0.30)Jaccard、打分截面秩自相关(滞后 1/5/10/20/40)、Rank IC(h = 1/5/10/20,
    open 口径:te 窗口价 -> te+h 窗口价)、剔除集相对池内其余股票的 h 日收益差。只解释,不进判定。"""
    from scipy.stats import rankdata
    c = ctx()
    S = c.raw_score().to_numpy(np.float64)
    U = c.U.reindex(index=c.p.dates, columns=c.p.codes).fillna(False).to_numpy(bool)
    P, _ = c.exec_px_pool()
    T, N = S.shape
    ok = U & np.isfinite(S)
    Q = np.full((T, N), np.nan)
    for t in range(T):
        idx = np.flatnonzero(ok[t])
        if len(idx) >= 50:
            Q[t, idx] = rankdata(S[t, idx]) / len(idx)
    E = Q <= 0.30                                   # NaN 比较为 False
    rows = []
    dates = c.p.dates
    for L in (1, 5, 10, 20, 40):
        for t in range(T - L):
            a, b = E[t], E[t + L]
            na, nb = int(a.sum()), int(b.sum())
            if na < 20 or nb < 20:
                continue
            inter = int((a & b).sum())
            j = inter / (na + nb - inter)
            both = ok[t] & ok[t + L]
            x, y = S[t, both], S[t + L, both]
            rho = np.corrcoef(rankdata(x), rankdata(y))[0, 1] if both.sum() >= 50 else np.nan
            rows.append(dict(date=dates[t], stat="persist", lag=L, jaccard=j, rank_ac=rho))
    lag = 1
    for h in (1, 5, 10, 20):
        with np.errstate(invalid="ignore", divide="ignore"):
            F = np.full((T, N), np.nan)
            F[:T - lag - h] = P[lag + h:T] / P[lag:T - h] - 1.0
        for t in range(T - lag - h):
            m = ok[t] & np.isfinite(F[t])
            if m.sum() < 50:
                continue
            x, y = S[t, m], F[t, m]
            ic = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
            e = E[t, m]
            spread = float(y[e].mean() - y[~e].mean()) if e.sum() >= 10 and (~e).sum() >= 10 else np.nan
            rows.append(dict(date=dates[t], stat="ic", lag=h, ic=ic, ex_spread=spread, n=int(m.sum())))
        print(f"  [{c.panel}] decay h={h} done", flush=True)
    return pd.DataFrame(rows)


# ====================================================================== 驱动
def _pool(panel: str, workers: int):
    import concurrent.futures as cf
    import multiprocessing as mp
    return cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                  initializer=_init, initargs=(panel,))


def _collect(results, panel: str, tag: str):
    metas, dailies, mbs = [], [], {}
    for item in results:
        meta, daily = item[0], item[1]
        metas.append(meta)
        d = daily.copy()
        d.index.name = "date"
        d = d.reset_index()
        d.insert(0, "cell", meta["cell"])
        dailies.append(d)
        if len(item) > 2 and item[2] is not None:
            mbs[meta["cell"]] = item[2]
    M = pd.DataFrame(metas)
    D = pd.concat(dailies, ignore_index=True)
    M.to_csv(OUT / f"inc_t7_{tag}_meta_{panel}.csv", index=False)
    D.to_parquet(OUT / f"inc_t7_{tag}_daily_{panel}.parquet")
    if mbs:
        pd.DataFrame(mbs).to_parquet(OUT / f"inc_t7_{tag}_matchbench_{panel}.parquet")
    bad = M[M["assert_fail"].astype(str) != ""]
    log(f"[{tag}] {panel}: {len(M)} 格写入 cache/_tmp/inc_t7_{tag}_*_{panel};断言不过 {len(bad)} 格"
        + ("" if bad.empty else ":" + "; ".join(f"{r.cell}:{r.assert_fail}" for r in bad.itertuples())))
    return M


def stage_verify():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "inc_t7_freeze.py"), "verify"],
                       capture_output=True, text=True)
    log(r.stdout.strip())
    if r.returncode != 0:
        log("[verify] 冻结核对不过,停止"); sys.exit(2)


def stage_repro():
    """§4.5 ①②:先于网格,任一不过即停。"""
    out = {}
    cap = pd.read_csv(LEDGER / "capacity_grid.csv")
    lhg = pd.read_csv(LEDGER / "lh_step1_grid.csv")
    sel = (cap.strategy == "COMP_v5_exclude") & (cap["mode"] == "open10") & (cap.aum == 1e7) & \
          (cap.sizing == "pov") & (cap.window == "2021-2025")
    ref_min = {"Y1": float(cap.loc[sel & (cap.variant == "Y1.0"), "excess_pool"].iloc[0]),
               "old": float(cap.loc[sel & (cap.variant == "legacy"), "excess_pool"].iloc[0])}
    ref_lh = {(5, "Y1"): float(lhg.loc[lhg.config == "main", "full_excess"].iloc[0]),
              (20, "Y1"): float(lhg.loc[lhg.config == "main_k20", "full_excess"].iloc[0]),
              (5, "old"): float(lhg.loc[lhg.config == "main_old", "full_excess"].iloc[0]),
              (5, "pess"): float(lhg.loc[lhg.config == "main_pess", "full_excess"].iloc[0])}
    ok_all = True
    _init("minute")
    c = ctx()
    W = c.weights(CUR_BAND)
    b5 = c.bench(5)
    for cost in ("Y1", "old"):
        res, daily = run_cell(c, W, 5, c.cost(cost))
        rs = daily["ret"][(daily.index.year >= 2021) & (daily.index.year <= 2025)]
        v = LE.ann_geo(rs) - LE.ann_geo(b5.reindex(rs.index))
        ok = abs(v - ref_min[cost]) <= 1e-10
        ok_all &= ok
        out[f"minute_k5_20/35_{cost}"] = dict(value=v, ref=ref_min[cost], diff=v - ref_min[cost], ok=bool(ok),
                                             ref_src="capacity_grid.csv COMP_v5_exclude open10 1e7 pov 2021-2025 "
                                                     + ("Y1.0" if cost == "Y1" else "legacy"))
        log(f"[repro] 分钟 k5 20/35 {cost}: {v:.10f} vs {ref_min[cost]:.10f} diff {v - ref_min[cost]:.2e} "
            f"{'OK' if ok else 'FAIL'}")
    del W, c
    globals()["_CTX"] = None
    _init("lh")
    c = ctx()
    W = c.weights(CUR_BAND)
    for (k, cost), ref in ref_lh.items():
        res, daily = run_cell(c, W, k, c.cost(cost))
        b = c.bench(k)
        s = LE.slice_eval(daily["ret"], LE.EVAL_START, "2025-12-31")
        bb = LE.slice_eval(b, LE.EVAL_START, "2025-12-31")
        v = LE.ann_geo(s) - LE.ann_geo(bb)
        tol = 1e-10
        ok = abs(v - ref) <= tol
        required = (k, cost) in ((5, "Y1"), (20, "Y1"))
        if required:
            ok_all &= ok
        out[f"lh_k{k}_20/35_{cost}"] = dict(value=v, ref=ref, diff=v - ref, ok=bool(ok), required=required,
                                           ref_src="lh_step1_grid.csv " + {(5, "Y1"): "main", (20, "Y1"): "main_k20",
                                                                          (5, "old"): "main_old",
                                                                          (5, "pess"): "main_pess"}[(k, cost)])
        log(f"[repro] 长历史 k{k} 20/35 {cost}: {v:.10f} vs {ref:.10f} diff {v - ref:.2e} {'OK' if ok else 'FAIL'}"
            + ("" if required else "(附加,非预注册断言)"))
    # 匹配基准实现核对:pool_ew_custom 按 k=5 相位日历重置、5 相位平均 == benchmark(rebalance=5)
    P, U = c.exec_px_pool()
    acc = np.zeros(c.T - 2)
    for ph in range(5):
        reset = ((np.arange(c.T - 2) - 0 - ph) % 5) == 0
        acc += pool_ew_custom(P, U, reset)
    mb5 = pd.Series(acc / 5, index=c.p.dates[:c.T - 2])
    dmax = float((mb5 - c.bench(5).reindex(mb5.index)).abs().max())
    out["lh_pool_ew_custom_vs_bench5"] = dict(value=dmax, ok=bool(dmax < 1e-12))
    ok_all &= dmax < 1e-12
    log(f"[repro] 匹配基准实现 vs benchmark(rebalance=5) 最大日差 {dmax:.2e}")
    out["all_ok"] = bool(ok_all)
    out["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "inc_t7_repro.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    if not ok_all:
        log("[repro] 复现断言不过,停止(§4.5:任一不过即停)"); sys.exit(3)
    log("[repro] 全部通过")


def _require_repro():
    f = OUT / "inc_t7_repro.json"
    if not f.exists() or not json.loads(f.read_text()).get("all_ok"):
        log("复现断言未通过或未运行,停止"); sys.exit(3)


def stage_grid(panel: str, workers: int):
    _require_repro()
    t0 = time.time()
    specs = [dict(band=None, ks=KS, costs=COSTS)] + [dict(band=b, ks=KS, costs=COSTS) for b in BANDS]
    # 慢的先跑:k=20 的相位最多,但每个 spec 都含全部 k,顺序按带宽即可;把现行带放最前
    specs.sort(key=lambda s: 0 if s["band"] == CUR_BAND else 1)
    with _pool(panel, workers) as ex:
        fb = ex.submit(job_bench, dict(ks=KS))
        futs = [ex.submit(job_grid, s) for s in specs]
        res = []
        for f in futs:
            res += f.result()
        B = fb.result()
    B.to_parquet(OUT / f"inc_t7_bench_{panel}.parquet")
    _collect(res, panel, "grid")
    log(f"[grid] {panel} 完成 {time.time() - t0:.0f}s")


def stage_event(panel: str, workers: int):
    _require_repro()
    t0 = time.time()
    specs = [dict(variant=v, offset=o, costs=COSTS) for v in EVENTS for o in OFFSETS]
    with _pool(panel, workers) as ex:
        futs = [ex.submit(job_event, s) for s in specs]
        res = []
        for f in futs:
            res += f.result()
    _collect(res, panel, "event")
    log(f"[event] {panel} 完成 {time.time() - t0:.0f}s")


def stage_placebo(workers: int):
    _require_repro()
    t0 = time.time()
    with _pool("minute", workers) as ex:
        futs = [ex.submit(job_placebo, dict(seed=s)) for s in range(5)]
        res = []
        for f in futs:
            res += f.result()
    _collect(res, "minute", "placebo")
    log(f"[placebo] 完成 {time.time() - t0:.0f}s")


def stage_sens(panel: str, workers: int, extra_cells: list[tuple[int, tuple]] | None = None,
               extra_events: list[str] | None = None):
    _require_repro()
    t0 = time.time()
    base_cells = [(5, (0.20, 0.35)), (10, (0.20, 0.35)), (5, (0.30, 0.45)), (20, (0.20, 0.35))]
    cells = base_cells + [c for c in (extra_cells or []) if c not in base_cells]
    specs = [dict(kind="sampled", k=10, band=CUR_BAND), dict(kind="sampled", k=20, band=CUR_BAND)]
    evs = ["E1_6"] + [v for v in (extra_events or []) if v != "E1_6"]
    for cap in (1e8, 1e9):
        specs.append(dict(kind="aum", capital=cap, const=True, cells=cells))
        for v in evs:
            specs.append(dict(kind="aum_event", capital=cap, const=True, variant=v))
    if panel == "lh":
        specs.append(dict(kind="aum", capital=1e7, const=True, cells=cells))
        for v in evs:
            specs.append(dict(kind="aum_event", capital=1e7, const=True, variant=v))
    with _pool(panel, workers) as ex:
        futs = [ex.submit(job_sens, s) for s in specs]
        res = []
        for f in futs:
            res += f.result()
    _collect(res, panel, "sens")
    log(f"[sens] {panel} 完成 {time.time() - t0:.0f}s")


def stage_decay(panel: str):
    t0 = time.time()
    _init(panel)
    D = job_decay({})
    D.to_parquet(OUT / f"inc_t7_decay_daily_{panel}.parquet")
    log(f"[decay] {panel} 完成 {time.time() - t0:.0f}s,{len(D)} 行")


def main():
    global _LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("panel", nargs="?", default="minute")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--extra", default="", help="sens 的附加格子,如 '10:0.3:0.45,20:0.2:0.3'")
    ap.add_argument("--extra-events", default="", help="sens 的附加事件式变体,如 'E2b_10,E2b_20'")
    a = ap.parse_args()
    LOGDIR.mkdir(exist_ok=True)
    _LOGF = LOGDIR / f"inc_t7_{a.stage}{'' if a.stage in ('verify', 'repro', 'placebo') else '_' + a.panel}.log"
    log(f"=== inc_t7 {a.stage} {a.panel} workers={a.workers}")
    if a.stage == "verify":
        stage_verify()
    elif a.stage == "repro":
        stage_verify(); stage_repro()
    elif a.stage == "grid":
        stage_grid(a.panel, a.workers)
    elif a.stage == "event":
        stage_event(a.panel, a.workers)
    elif a.stage == "placebo":
        stage_placebo(a.workers)
    elif a.stage == "sens":
        extra = []
        for tok in [x for x in a.extra.split(",") if x]:
            k, e, x = tok.split(":")
            extra.append((int(k), (float(e), float(x))))
        stage_sens(a.panel, a.workers, extra, [x for x in a.extra_events.split(",") if x])
    elif a.stage == "decay":
        stage_decay(a.panel)
    else:
        raise SystemExit(f"未知阶段 {a.stage}")


if __name__ == "__main__":
    main()
