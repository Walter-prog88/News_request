"""前向检验 T4:PEAD E 臂 vs B 臂(前缀 fwd;预注册 FORWARD_TEST_PREREG §2 T4、§3.4、§5.3,方案 v0.2 + 补充登记原样沿用)。

两种口径(决定见 research/ledger/fwd_decisions.csv,都在计算任何前向收益之前定下):
  orig    预注册原规则:财报一致率未全部达 99%(东财营业收入 98.5%)-> 沿用最后已知报告期(tushare 利润表到 08-20)。
          主事件只到 V <= 08-21,窗口(t_signal 09-01 ~ 09-30)内预计没有批次 -> P5 记"事件不足,不作读数"。
  splice  偏离预注册 D1(补充读数,不进判定):PEAD 主信号只依赖营业利润,东财 OPERATE_PROFIT 与 tushare 一致率 99.94%
          (2026Q1)/ 100%(半年报重叠);用它补 tushare 没有的 2026 半年报 / 三季报,其余口径完全不变。
阶段:
  events     事件表、合格池、窗口批次(E / B / B@tol05)、平衡性表 —— 只读公告与 t_signal 收盘前的量,**不读收益**
  accounts   E / B 独立账户(100 万主、1000 万压力;基准费率、整手、最低 5 元),只跑窗口批次,收益面板
  eventlvl   事件层 h5 配对超额 D_T、大批次 D̄
产出:cache/_tmp/fwd_pead_<mode>_*;research/ledger/fwd_pead_events.csv、fwd_pead_balance.csv。
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
from afactor.fwd_panel import ForwardPanel, forward_context, assert_forward_index, SIG_START
from afactor import fwd_pead as FP, earnings_events as EE
import fwd_common as FC

log = FC.log
WIN_T = (pd.Timestamp("2026-09-02"), pd.Timestamp("2026-10-08"))    # 成交日 T:t_signal ∈ [09-01, 09-30]
SIG_END = pd.Timestamp("2026-09-30")
MODES = ("orig", "splice")
BIG = 50


def _prep(mode: str):
    p = ForwardPanel()
    cal = FP.ext_calendar(p.dates)
    offset = int(cal.get_loc(p.dates[0]))
    with FP.lifted_cutoff(cal):
        inc = FP.load_income_fwd(splice_em=(mode == "splice"))
        ev = FP.main_events(inc, cal)
        pool = EE.event_pool(ev, cal)
    with forward_context("前向检验 T4:合格池(ST 用前向面板 is_st)"):
        with FP.panel_st(p):
            Q = EE.qualified_pool(p)
    Qn = Q.to_numpy(bool)
    pool["i"] = p.dates.get_indexer(pool["t_signal"])
    pool["j"] = p.codes.get_indexer(pool["stock_code"])
    pool["T_pos"] = p.dates.get_indexer(pool["T"])
    ok = (pool["i"] >= 0) & (pool["j"] >= 0) & (pool["T_pos"] >= 0)
    pool = pool[ok].copy()
    pool = pool[Qn[pool["i"].to_numpy(), pool["j"].to_numpy()]].copy()
    assert (pool["T_pos"] == pool["i"] + 1).all()
    assert ((pool["T_pos"] + offset).to_numpy() == cal.get_indexer(pool["T"])).all()
    return p, cal, offset, inc, ev, pool, Q, Qn


def stage_events():
    import pead_build_batches as PBB
    rows = []
    for mode in MODES:
        t0 = time.time()
        p, cal, offset, inc, ev, pool, Q, Qn = _prep(mode)
        inwin = pool[(pool["T"] >= WIN_T[0]) & (pool["T"] <= WIN_T[1])]
        ev26 = ev[ev["P"] >= pd.Timestamp("2026-07-01")]
        rows += [dict(mode=mode, item="main_events_P_since_0701", value=len(ev26),
                      note=f"P 最晚 {ev['P'].max().date()},V 最晚 {ev['V'].max().date()}"),
                 dict(mode=mode, item="em_rows_spliced", value=inc.attrs.get("n_em_rows", 0), note="东财补入的(股票, 报告期)行"),
                 dict(mode=mode, item="pool_pairs_in_window", value=len(inwin), note="合格事件池 (T, 股票) 对,T ∈ [09-02, 10-08]"),
                 dict(mode=mode, item="pool_T_in_window", value=int(inwin["T"].nunique()), note="")]
        qv, raw_vals, ind_lab = FP.match_inputs(p, Q, p.raw("mv"), p.raw("circ_mv"))
        B, M = FP.build_batches(pool, Qn, qv, ind_lab, p.dates, p.codes, offset, T_range=WIN_T)
        if len(B):
            B.to_parquet(TMP / f"fwd_pead_{mode}_batches.parquet", index=False)
        M.to_parquet(TMP / f"fwd_pead_{mode}_batch_meta.parquet", index=False)
        nE = int((B["arm"] == "E").sum()) if len(B) else 0
        act = M[M["n_sel"] > 0] if len(M) else M
        rows += [dict(mode=mode, item="batches_with_selection", value=len(act),
                      note="; ".join(f"T={r.date_T.date()} φ={r.phase} 池{r.n_pool} 入选{r.n_sel}" for r in act.itertuples())),
                 dict(mode=mode, item="E_slots_total", value=nE, note=""),
                 dict(mode=mode, item="phases_with_batch", value=int(act["phase"].nunique()) if len(act) else 0,
                      note=",".join(str(x) for x in sorted(act["phase"].unique())) if len(act) else "")]
        if len(B):
            jj = B["j"].clip(lower=0).to_numpy(); ii = B["t_pos"].to_numpy()
            ok = (B["j"] >= 0).to_numpy()
            V = pd.DataFrame({"arm": B["arm"], "rep": B["rep"], "t_signal": B["t_signal"], "slot": B["slot"],
                              "T_pos": B["T_pos"], "ok": ok, "ind": np.where(ok, ind_lab[jj], -99),
                              "logmv": np.where(ok, raw_vals[ii, jj, 0], np.nan),
                              "logadv": np.where(ok, raw_vals[ii, jj, 1], np.nan),
                              "vol": np.where(ok, raw_vals[ii, jj, 2], np.nan), "level": B["level"]})
            V.to_parquet(TMP / f"fwd_pead_{mode}_match_vars.parquet", index=False)
            saved = PBB.WINDOWS
            PBB.WINDOWS = {"fwd": ("2026-09-01", "2026-09-30")}
            try:
                bal = PBB.balance_table(V[V["arm"].isin(["E", "B", "B@tol05"])])
            finally:
                PBB.WINDOWS = saved
            bal = bal[bal["control"].isin(["B", "B@tol05"])].assign(mode=mode)
            FC.ledger_append(LEDGER / "fwd_pead_balance.csv", bal.to_dict("records"), key_cols=("mode", "control", "window"))
            for r in bal.to_dict("records"):
                log(f"[events] {mode} 平衡性 {r['control']}: SMD mv {r['smd_logmv']:+.3f} adv {r['smd_logadv']:+.3f} "
                    f"vol {r['smd_vol']:+.3f} 行业最大差 {r['ind_maxdiff_pp']:.2f}pp 失败 {r['n_failed']} "
                    f"-> {'达标' if r['pass'] else '不达标'}")
        log(f"[events] {mode}: 窗口批次 {len(act)} 个,E 槽位 {nE} [{time.time()-t0:.0f}s]")
    FC.ledger_append(LEDGER / "fwd_pead_events.csv", [dict(section="T4", **r) for r in rows], key_cols=("mode", "item"))
    for r in rows:
        log(f"[events] {r['mode']} {r['item']} = {r['value']}  {r['note'][:200]}")


def _ctrl_arm(mode: str) -> str:
    """平衡性门槛(补充登记 §5.5):B 不达标时只补救一次(L1/L3 容差 0.05,B@tol05);仍不达标则降级。"""
    bal = pd.read_csv(LEDGER / "fwd_pead_balance.csv")
    b = bal[(bal["mode"] == mode)].set_index("control")
    if bool(b.at["B", "pass"]):
        return "B"
    return "B@tol05"


def stage_accounts():
    from afactor import pead_engine as PE
    p = ForwardPanel(returns_view=True)
    raw_px = p.raw("raw_close").to_numpy(np.float64)
    adj_px = p.raw("close_px").to_numpy(np.float64)
    star = PE.star_mask(p.codes)
    import pead_run_arms as PRA
    for mode in MODES:
        f = TMP / f"fwd_pead_{mode}_batches.parquet"
        if not f.exists():
            log(f"[accounts] {mode}: 窗口内没有批次,跳过")
            continue
        cal = FP.ext_calendar(p.dates)
        offset = int(cal.get_loc(p.dates[0]))
        Ball = pd.read_parquet(f)
        carm = _ctrl_arm(mode)
        out = []
        for sc, cap in (("base_1m", 1e6), ("base_10m", 1e7)):
            for arm in ("E", carm):
                reps = sorted(Ball.loc[Ball["arm"] == arm, "rep"].unique())
                for rep in reps:
                    for ph in range(5):
                        g = Ball[(Ball["arm"] == arm) & (Ball["rep"] == rep) & (Ball["phase"] == ph)]
                        batches = PRA.account_batches(g) if len(g) else {}
                        res = PE.run_account(p, batches, (ph - offset) % 5, cap, raw_px, adj_px, star, lots=True,
                                             min_commission=5.0, stress_bps=0.0)
                        d = res["daily"]
                        d = d[d.index >= SIG_START]
                        d = d[d.index <= SIG_END]
                        d.insert(0, "phase", ph); d.insert(0, "rep", int(rep))
                        d.insert(0, "arm", "B" if arm.startswith("B") else arm); d.insert(0, "scenario", sc)
                        out.append(d.reset_index())
            log(f"[accounts] {mode} {sc} 完成(对照臂 {carm})")
        D = pd.concat(out, ignore_index=True)
        assert D["date"].min() >= SIG_START
        D.to_parquet(TMP / f"fwd_pead_{mode}_daily.parquet", index=False)


def stage_eventlvl():
    from afactor import strategy as ST
    p = ForwardPanel(returns_view=True)
    T, N = len(p.dates), len(p.codes)
    twap = p.raw("twap_open10").to_numpy(np.float64)
    amt10 = p.raw("amt_open10").to_numpy(np.float64)
    dead = (p["one_word_up"] | p["one_word_down"]).to_numpy(bool)
    buyable = np.isfinite(twap) & (twap > 0) & ~dead & (amt10 > 0)
    twap_ff = pd.DataFrame(twap).ffill().to_numpy()
    for mode in MODES:
        f = TMP / f"fwd_pead_{mode}_batches.parquet"
        if not f.exists():
            continue
        carm = _ctrl_arm(mode)
        B = pd.read_parquet(f)

        def r_h(Tp, j, h):                                   # pead_event_level.r_h 同式
            Tp = np.asarray(Tp); j = np.asarray(j)
            out = np.full(len(Tp), np.nan)
            ok = (j >= 0) & (Tp + h < T)
            Te, je = Tp[ok], j[ok]
            r = twap_ff[Te + h, je] / twap[Te, je] - 1.0
            r = np.where(buyable[Te, je], r, 0.0)
            out[ok] = r
            return out
        E = B[B["arm"] == "E"].copy()
        E["r5"] = r_h(E["T_pos"], E["j"], 5)
        c = B[B["arm"] == carm].copy()
        c["r5"] = r_h(c["T_pos"], c["j"], 5)
        cm = c.groupby(["T_pos", "slot"])["r5"].mean()
        E["e5_B"] = E["r5"].to_numpy() - cm.reindex(pd.MultiIndex.from_arrays([E["T_pos"], E["slot"]])).to_numpy()
        E["h5_available"] = (E["T_pos"] + 5) <= (T - 1)
        assert_forward_index(E.set_index("t_signal"), "T4 事件层")
        E.to_parquet(TMP / f"fwd_pead_{mode}_event_slots.parquet", index=False)
        log(f"[eventlvl] {mode}:E 槽位 {len(E)},h5 可得 {int(E['h5_available'].sum())},对照臂 {carm}")


if __name__ == "__main__":
    for st in sys.argv[1:]:
        t0 = time.time()
        log(f"===== {st} =====")
        globals()[f"stage_{st}"]()
        log(f"===== {st} 完成 {time.time() - t0:.0f}s =====")
