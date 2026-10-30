"""INC-T3 S3:排雷叠加剔除(未来 N_pre 日内有 >= X% 解禁 -> 剔出决策池)+ 安慰剂 + C 层事后上界。

预注册 research/INC_T3_PREREG.md §4.1、§6、§7.3-7.5、§9、§10(G5)。

剔除面板 E_{X,N,M}(t, j) = 1 iff 股票 j 有事件 (D, ratio_plan >= X),且信号日 t ∈ [max(D - N_pre, K), D + M - 1]
(K = 已知日:A 层 D-60、B 层 D-3;M = 0 时信号日 D 起重新纳入,成交在 D+1 开盘)。
叠加臂 U_over = universe(V1000) & ~E,以 universe_override 传入 build_weights(与 run_strategy 同法);
基准不变(同频 5 日池等权,cap_capacity_curve 的缓存 cap_bench_open10.parquet,run 前核对与 ST.benchmark 逐位一致)。
成本 / 资金 / 模拟区间 / 引擎调用全部复用 scripts/cap_capacity_curve.py(run_cell、cost_setup、ConstAUMCost、
sim_mask):open10、5 日调仓 5 相位平均、恒定 AUM 1000 万、pov 截量、2020-07-01 起预热。

安慰剂(§7.4,实现口径见报告偏离 #4,读收益前定):每个真实剔除窗口(事件 × 窗口)换成一只"伪事件股",
在窗口首个在池日 t0 从池内按(上市年龄桶 × 真实市值池内五分位)同层随机抽取,排除窗口内本身被真实剔除的股票
与正被其它伪窗口占用的股票,伪股票剔除同样的信号日窗口 —— 每日剔除数量与层分布与真实剔除一致,且保留
真实剔除的持续性(逐日独立重抽会人为制造换手、使安慰剂偏差)。种子 default_rng([20261011, 32, cell_idx, r])。

用法
  python3 scripts/inc_t3_overlay.py base       # G5:基线逐位复现 capacity_grid、全池等权 ≈ -成本、基准缓存核对
  python3 scripts/inc_t3_overlay.py grid       # 24 格 × {Y1, legacy} + 主格敏感性 + 全 B 层 + C 层上界
  python3 scripts/inc_t3_overlay.py placebo [lo hi]   # 24 格 × R=10 安慰剂(Y1、1000 万);可按格号分片
产出
  cache/_tmp/inc_t3_overlay_daily.parquet、inc_t3_placebo_daily_<lo>_<hi>.parquet、inc_t3_placebo_meta_*.csv
  research/ledger/inc_t3_checks.csv(G5 与逐格断言)
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t3_common as C                                   # noqa: E402
import cap_capacity_curve as CAP                            # noqa: E402
from afactor.panel import Panel                             # noqa: E402
from afactor import strategy as ST, newdata as ND           # noqa: E402

MODE = "open10"
AUM = 1e7
REB = 5
XS = [0.05, 0.10, 0.20]
NPRES = [3, 5, 10, 20]
MS = [0, 10]
MAIN = (0.10, 10, 0)
CELLS = [(x, n, mm) for x in XS for n in NPRES for mm in MS]          # 24 格,固定顺序 = cell_idx
R_PLACEBO = 10
CHECKS = C.LEDGER / "inc_t3_checks.csv"
DAILY = C.TMP / "inc_t3_overlay_daily.parquet"
KEEP = ["ret", "gross_ret", "cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact",
        "exposure", "cash", "turnover", "n_hold", "exp_max_phase"]


def cell_name(x, n, mm, tier="main"):
    return f"X{int(round(x*100))}_N{n}_M{mm}" + ("" if tier == "main" else f"_{tier}")


def upsert_checks(rows: list[dict]):
    new = pd.DataFrame(rows)
    new["written_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    if CHECKS.exists():
        old = pd.read_csv(CHECKS)
        key = new["check"] + "|" + new["cell"].astype(str)
        old = old[~(old["check"] + "|" + old["cell"].astype(str)).isin(key)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(CHECKS, index=False)


# ====================================================================== 剔除面板
def exclusion_windows(ev: pd.DataFrame, x: float, n_pre: int, m_post: int, T: int, tier_rule: str = "main",
                      size_col: str = "ratio_plan") -> pd.DataFrame:
    """真实剔除窗口表:每个 ratio >= x 的事件一行,[start, end] 为被剔除的信号日(含端点,已截到 [0, T-1])。
    tier_rule='main':A 层 K = D-60、B 层 K = D-3;'allB':全部 K = D-3;'none':不设已知日(C 层事后上界)。"""
    e = ev[ev[size_col] >= x]
    d = e["d_pos"].to_numpy(np.int64)
    if tier_rule == "main":
        K = np.where(e["tier"].to_numpy() == "A", d - C.K_A, d - C.K_B)
    elif tier_rule == "allB":
        K = d - C.K_B
    elif tier_rule == "none":
        K = np.full(len(d), -10 ** 6)
    else:
        raise ValueError(tier_rule)
    start = np.maximum(d - n_pre, K)
    end = d + m_post - 1
    out = pd.DataFrame(dict(j=e["j"].to_numpy(np.int64), d_pos=d, start=np.clip(start, 0, T - 1),
                            end=np.minimum(end, T - 1), raw_start=start, raw_end=end))
    return out[(out["raw_end"] >= 0) & (out["raw_start"] <= T - 1) & (out["end"] >= out["start"])].reset_index(drop=True)


def panel_from_windows(win: pd.DataFrame, T: int, N: int) -> np.ndarray:
    E = np.zeros((T, N), bool)
    for j, a, b in zip(win["j"].to_numpy(), win["start"].to_numpy(), win["end"].to_numpy()):
        E[a:b + 1, j] = True
    return E


def exclusion_panel(ev, x, n_pre, m_post, T, N, tier_rule="main", size_col="ratio_plan"):
    return panel_from_windows(exclusion_windows(ev, x, n_pre, m_post, T, tier_rule, size_col), T, N)


# ====================================================================== 安慰剂
def strata_panel(p: Panel, U: np.ndarray) -> np.ndarray:
    """(上市年龄桶 0/1/2) × (真实市值池内五分位 0..4) -> 0..14;池外或缺失 -> -1。"""
    dates, codes = p.dates, p.codes
    age = C.listing_age_panel(dates, codes)
    ab = C.age_bucket(age)
    del age
    mv = ND.total_mv(dates, codes)
    circ = ND.circ_mv(dates, codes)
    mvv = mv.where(mv > 0).fillna(circ.where(circ > 0)).to_numpy(np.float64)
    del mv, circ
    r = pd.DataFrame(np.where(U, mvv, np.nan)).rank(axis=1, pct=True).to_numpy()
    q = np.clip(np.ceil(r * 5) - 1, 0, 4)
    S = np.where(U & np.isfinite(q) & (ab >= 0), ab * 5 + np.nan_to_num(q, nan=0).astype(np.int64), -1)
    return S.astype(np.int16)


def placebo_windows(win: pd.DataFrame, E: np.ndarray, U: np.ndarray, S: np.ndarray, seed) -> tuple[pd.DataFrame, dict]:
    """每个真实窗口 -> 一只同层伪事件股、同样的信号日窗口。返回伪窗口表与统计。"""
    rng = np.random.default_rng(seed)
    T, N = U.shape
    busy = np.full(N, -1, np.int64)                    # 伪窗口占用到哪一天
    w = win.sort_values(["start", "j"]).reset_index(drop=True)
    rows = []
    n_skip, n_fb1, n_fb2, n_fail = 0, 0, 0, 0
    for j, a, b in zip(w["j"].to_numpy(), w["start"].to_numpy(), w["end"].to_numpy()):
        inU = np.flatnonzero(U[a:b + 1, j])
        if inU.size == 0:
            n_skip += 1                                 # 窗口内从未在池:真实剔除无作用,伪事件也不需要
            continue
        t0 = a + int(inU[0])
        s = int(S[t0, j])
        base = U[t0] & ~E[a:b + 1].any(axis=0) & (busy < a)
        base[j] = False
        cand = np.flatnonzero(base & (S[t0] == s)) if s >= 0 else np.array([], np.int64)
        if cand.size == 0 and s >= 0:
            n_fb1 += 1
            cand = np.flatnonzero(base & (S[t0] >= 0) & (S[t0] // 5 == s // 5))
        if cand.size == 0:
            n_fb2 += 1
            cand = np.flatnonzero(base)
        if cand.size == 0:
            n_fail += 1
            continue
        c = int(cand[rng.integers(cand.size)])
        busy[c] = b
        rows.append((c, a, b))
    pw = pd.DataFrame(rows, columns=["j", "start", "end"])
    return pw, dict(n_windows=len(w), n_skip_never_in_pool=n_skip, n_fallback_age_only=n_fb1,
                    n_fallback_any=n_fb2, n_fail=n_fail, n_placebo=len(pw))


# ====================================================================== 引擎
class Ctx:
    def __init__(self):
        t0 = time.time()
        self.p = Panel()
        self.T, self.N = len(self.p.dates), len(self.p.codes)
        self.comp = pd.read_parquet(C.COMP_PATH)
        self.m = CAP.sim_mask(self.p, "COMP_v5_exclude")
        self.Udf = ST.universe(self.p, C.VERSION)
        self.U = self.Udf.to_numpy(bool)
        self.B = pd.read_parquet(CAP.BENCH_CACHE.format(MODE))
        self.models: dict = {}
        self.ev = C.load_events()
        print(f"[{time.time()-t0:.0f}s] 上下文就绪 T={self.T} N={self.N}", flush=True)

    def weights(self, E: np.ndarray | None):
        uo = None if E is None else pd.DataFrame(self.U & ~E, index=self.p.dates, columns=self.p.codes)
        return ST.build_weights(self.comp, self.p, C.VERSION, scheme="exclude", buffer=(0.20, 0.35),
                                date_mask=self.m, universe_override=uo)

    def run(self, W, variant="Y1.0", aum=AUM, phases=True):
        res, daily, dret = CAP.run_cell(self.p, W, MODE, aum, variant, self.m, self.models, "pov")
        prof = CAP.align_profile(res["rets"], self.B["pool"])
        peak = max(prof, key=prof.get)
        exp_max = float(daily["exp_max_phase"].max())
        if peak != 0:
            raise AssertionError(f"对齐峰值 k={peak} {prof}")
        if exp_max > 1.0 + 1e-9 or res["exposure"] > 1.0 + 1e-9:
            raise AssertionError(f"exposure>1 ({exp_max})")
        if dret > 1e-12:
            raise AssertionError(f"record.ret 与 rets 不一致 {dret}")
        out = daily[KEEP].copy()
        if phases:
            by = res["record"]["daily_by_phase"]
            ph = by.reset_index().pivot_table(index=by.index.name or "index", columns="phase", values="ret")
            ph.columns = [f"ret_p{int(c)}" for c in ph.columns]
            out = out.join(ph)
        return res, out, dict(align_peak_k=peak, exp_max=exp_max, align_prof=json.dumps(prof))


def excess_arith(rs: pd.Series, B: pd.DataFrame, y0: int, y1: int) -> tuple[float, float]:
    r = rs[(rs.index.year >= y0) & (rs.index.year <= y1)]
    es = ST.excess_stats(r, B["pool"].reindex(r.index), bandwidth=20, t_bandwidth=CAP.NW_T_BW)
    return es.get("excess_arith", np.nan), es.get("t_nw", np.nan)


# ====================================================================== G5:基线
def cmd_base():
    t0 = time.time()
    X = Ctx()
    rows = []
    # 基准缓存核对:cap_bench_open10.parquet 的 pool 列 == ST.benchmark(..., rebalance=5)
    b5 = ST.benchmark(X.p, C.VERSION, MODE, rebalance=REB)["rets"]
    dB = float(np.nanmax(np.abs(b5.reindex(X.B.index).to_numpy() - X.B["pool"].to_numpy())))
    rows.append(dict(check="G5_bench_cache_eq_samefreq", cell="base", value=dB, threshold="<= 1e-12",
                     passed=bool(dB <= 1e-12)))
    cg = pd.read_csv(C.CAPGRID)
    cg = cg[(cg.strategy == "COMP_v5_exclude") & (cg["mode"] == MODE) & (cg.sizing == "pov")]
    W0 = X.weights(None)
    W_cap = CAP.strategy_weights(X.p, dict(comp=X.comp), "COMP_v5_exclude")
    dW = float(np.abs(W0 - W_cap).max())
    rows.append(dict(check="G5_weights_eq_cap", cell="base", value=dW, threshold="== 0", passed=bool(dW == 0)))
    daily_all = []
    for variant, aum in (("Y1.0", 1e7), ("legacy", 1e7), ("Y0.5", 1e7), ("Y1.5", 1e7), ("Y1.0", 1e6), ("Y1.0", 1e8)):
        res, d, chk = X.run(W0, variant, aum)
        tag = f"base|{variant}|{aum:.0e}"
        for wname, (y0, y1) in (("2021-2025", (2021, 2025)), ("2021-2023", (2021, 2023)), ("2024-2025", (2024, 2025))):
            ex, t = excess_arith(d["ret"], X.B, y0, y1)
            ref = cg[(cg.variant == variant) & (cg.aum == aum) & (cg.window == wname)]["excess_pool_arith"]
            ref = float(ref.iloc[0]) if len(ref) else np.nan
            gate = (variant == "Y1.0" and aum == 1e7)
            rows.append(dict(check=("G5_baseline_reproduce" if gate else "baseline_reproduce_extra") + f"_{wname}",
                             cell=tag, value=f"{ex:.9f} vs capacity_grid {ref:.9f}(差 {abs(ex-ref):.2e})",
                             threshold="|差| <= 1e-6", passed=bool(abs(ex - ref) <= 1e-6)))
        rows.append(dict(check="G5_exposure_le1_align_k0", cell=tag, value=f"exp_max {chk['exp_max']:.6f} peak {chk['align_peak_k']}",
                         threshold="<= 1 且 k = 0", passed=True))
        daily_all.append(d.assign(cell="BASE", variant=variant, aum=aum, arm="base").reset_index())
        print(rows[-2], f"[{time.time()-t0:.0f}s]", flush=True)
    # 全池等权常数打分(5 日调仓)vs 同频基准:超额 + 成本 ∈ [-0.5, +1.5]pp(Y1、1000 万,2021-2025)
    const = pd.DataFrame(1.0, index=X.p.dates, columns=X.p.codes, dtype=np.float32)
    Wc = ST.build_weights(const, X.p, C.VERSION, scheme="exclude", buffer=(0.20, 0.35), date_mask=X.m)
    res, d, chk = X.run(Wc, "Y1.0", 1e7, phases=False)
    ex, t = excess_arith(d["ret"], X.B, 2021, 2025)
    w = (d.index.year >= 2021)
    cost = float(d.loc[w, "cost_total"].mean() * 242)
    rows.append(dict(check="engine_fullpool_ew_vs_samefreq", cell="const_score", value=f"超额 {ex*100:+.3f}pp,成本 {cost*100:.3f}pp,"
                     f"超额+成本 {(ex+cost)*100:+.3f}pp", threshold="超额+成本 ∈ [-0.5, +1.5]pp",
                     passed=bool(-0.005 <= ex + cost <= 0.015)))
    upsert_checks(rows)
    pd.concat(daily_all, ignore_index=True).to_parquet(C.TMP / "inc_t3_base_daily.parquet")
    print(pd.DataFrame(rows)[["check", "cell", "value", "passed"]].to_string())
    bad = [r for r in rows if not r["passed"]]
    if any(r["check"].startswith("G5") for r in bad):
        raise SystemExit("G5 不通过:停")
    print(f"done base {time.time()-t0:.0f}s")


# ====================================================================== 网格
def cmd_grid():
    t0 = time.time()
    X = Ctx()
    base = pd.read_parquet(C.TMP / "inc_t3_base_daily.parquet")
    nh_base = base[(base.variant == "Y1.0") & (base.aum == 1e7)].set_index("date")["n_hold"]
    rows, daily_all = [], []
    ev = X.ev
    runs = []
    for ci, (x, n, mm) in enumerate(CELLS):
        runs.append((cell_name(x, n, mm), x, n, mm, "main", "Y1.0", 1e7, "plan"))
        runs.append((cell_name(x, n, mm), x, n, mm, "main", "legacy", 1e7, "plan"))
    x, n, mm = MAIN
    for v, a in (("Y0.5", 1e7), ("Y1.5", 1e7), ("Y1.0", 1e6), ("Y1.0", 1e8)):
        runs.append((cell_name(x, n, mm), x, n, mm, "main", v, a, "plan"))
    runs.append((cell_name(x, n, mm, "allB"), x, n, mm, "allB", "Y1.0", 1e7, "plan"))
    runs.append((cell_name(x, n, mm, "allB"), x, n, mm, "allB", "legacy", 1e7, "plan"))
    runs.append((cell_name(x, n, mm, "Ctier"), x, n, mm, "none", "Y1.0", 1e7, "jump"))
    runs.append((cell_name(x, n, mm, "Ctier"), x, n, mm, "none", "legacy", 1e7, "jump"))
    cev = pd.read_csv(C.LEDGER / "inc_t3_c_tier_events.csv")
    cev["tier"] = "C"
    Ecache: dict = {}
    for k, (name, x, n, mm, tier, v, a, src) in enumerate(runs):
        t1 = time.time()
        key = (name, tier, src)
        if key not in Ecache:
            Ecache.clear()
            if src == "plan":
                win = exclusion_windows(ev, x, n, mm, X.T, tier)
            else:
                win = exclusion_windows(cev, x, n, mm, X.T, tier, size_col="ratio_jump")
            E = panel_from_windows(win, X.T, X.N)
            W = X.weights(E)
            Ecache[key] = (E, W, len(win))
        E, W, nwin = Ecache[key]
        res, d, chk = X.run(W, v, a)
        ex_pool = (X.U & E)[X.m].sum() / max(X.U[X.m].sum(), 1)
        tag = f"{name}|{v}|{a:.0e}"
        nh = d["n_hold"]
        dh = (nh - nh_base.reindex(nh.index)).dropna()
        rows.append(dict(check="G5_cell_exposure_align", cell=tag, value=f"exp_max {chk['exp_max']:.6f} peak {chk['align_peak_k']}",
                         threshold="<= 1 且 k = 0", passed=True))
        if v == "Y1.0" and a == 1e7:
            rows.append(dict(check="G5_holdings_le_base", cell=tag,
                             value=f"均值差 {dh.mean():+.2f} 只;叠加 > 基线的日占比 {(dh > 0.5).mean():.4f};最大 {dh.max():+.1f}",
                             threshold="均值差 <= 0(剔除只减不增;逐日因池内重排名可有个别 +1)",
                             passed=bool(dh.mean() <= 0)))
            rows.append(dict(check="info_pool_share_excluded", cell=tag, value=f"{ex_pool:.5f}(窗口 {nwin})",
                             threshold="描述", passed=True))
        daily_all.append(d.assign(cell=name, variant=v, aum=a, arm="overlay", tier=tier, X=x, N_pre=n, M=mm,
                                  src=src).reset_index())
        print(f"  [{k+1}/{len(runs)}] {tag} pool_excl {ex_pool:.4f} [{time.time()-t1:.0f}s / {time.time()-t0:.0f}s]", flush=True)
        if (k + 1) % 8 == 0:
            pd.concat(daily_all, ignore_index=True).to_parquet(DAILY)
    pd.concat(daily_all, ignore_index=True).to_parquet(DAILY)
    upsert_checks(rows)
    print(f"done grid {time.time()-t0:.0f}s")


def cmd_placebo(lo: int = 0, hi: int = len(CELLS)):
    t0 = time.time()
    X = Ctx()
    S = strata_panel(X.p, X.U)
    print(f"[{time.time()-t0:.0f}s] 层面板就绪", flush=True)
    out, meta = [], []
    fn = C.TMP / f"inc_t3_placebo_daily_{lo}_{hi}.parquet"
    for ci in range(lo, hi):
        x, n, mm = CELLS[ci]
        name = cell_name(x, n, mm)
        win = exclusion_windows(X.ev, x, n, mm, X.T, "main")
        E = panel_from_windows(win, X.T, X.N)
        cnt_true = (X.U & E).sum(axis=1)
        for r in range(R_PLACEBO):
            t1 = time.time()
            pw, st = placebo_windows(win, E, X.U, S, [C.SEED, 32, ci, r])
            Ep = panel_from_windows(pw, X.T, X.N)
            cnt_pl = (X.U & Ep).sum(axis=1)
            mm_ = X.m & (cnt_true > 0)
            st.update(cell=name, cell_idx=ci, rep=r,
                      count_ratio=float(cnt_pl[X.m].sum() / max(cnt_true[X.m].sum(), 1)),
                      count_mad=float(np.abs(cnt_pl - cnt_true)[mm_].mean()) if mm_.any() else 0.0,
                      overlap_true=int((Ep & E).sum()))
            W = X.weights(Ep)
            res, d, chk = X.run(W, "Y1.0", AUM, phases=False)
            out.append(d[["ret", "turnover", "cost_total", "n_hold"]].assign(cell=name, cell_idx=ci, rep=r).reset_index())
            st.update(exp_max=chk["exp_max"], align_peak_k=chk["align_peak_k"])
            meta.append(st)
            print(f"  {name} r{r} ratio {st['count_ratio']:.3f} fb {st['n_fallback_age_only']}/{st['n_fallback_any']} "
                  f"[{time.time()-t1:.0f}s / {time.time()-t0:.0f}s]", flush=True)
        pd.concat(out, ignore_index=True).to_parquet(fn)
        pd.DataFrame(meta).to_csv(C.TMP / f"inc_t3_placebo_meta_{lo}_{hi}.csv", index=False)
    print(f"done placebo {lo}-{hi} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "base"
    if cmd == "base":
        cmd_base()
    elif cmd == "grid":
        cmd_grid()
    elif cmd == "placebo":
        lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(CELLS)
        cmd_placebo(lo, hi)
    else:
        raise SystemExit(cmd)
