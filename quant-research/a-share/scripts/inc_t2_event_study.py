"""INC_T2 步骤 4:事件研究(价)—— 各腿超额、成本、网格、CAR 图。

预注册 §6、§7.1、§9。**只在 Gate 1(inc_t2_overlay_checks.csv 全部通过)之后运行。**

腿(价格取研究面板前复权字段):
  L0     C+1 open10 -> A 收盘        (t0 = C,可交易,信息腿)
  gap    A 收盘 -> A+1 open10        (不可交易,单报)
  L1     A+1 open10 -> E−1 收盘      (t0 = A,主腿;卖出在 E−1 收盘集合竞价)
  L1b    A+1 open10 -> E open10      (t0 = A,引擎口径)
  L2     E−1 收盘 -> E 收盘;L2o = E−1 收盘 -> E 开盘集合竞价价(诊断)
  L3_h   E+1 open10 -> E+h 收盘,h ∈ {5, 20, 60}(t0 = E,反转腿)
  preC   C−20 收盘 -> C 收盘           (t0 = C,诊断)
入场不可成交(无价 / 一字板 / 开盘窗口无量)-> 该事件该腿记 0(不持仓)并计数;出场日无价用此前最后收盘价。
对照:matched(k=3,本腿 t0 匹配,容差 0.10,均值;对照入场不可成交则剔除该对照)、
      pool(t0 日 V1000 决策池等权买入持有)、proxy(事件所属指数的 PIT 代理,入场前一日收盘权重买入持有)。
成本(只对可交易腿;对照不计成本):
  none / old(佣金 1bp ×2 + 卖出印花税)/ Y0.5 / Y1 / Y1.5(CostModel:+ 半价差 + Y σ √(Q/ADV20))。
  Q = 篮子资金 / 本期本格事件数,篮子 ∈ {1e6, 1e7(主), 1e8, 1e9};买入参与率上限 10% × amt_open10(超出部分不成交,记 0)。
  L1 的 E−1 收盘卖出:冲击分母改为 E−1 实现的 amt_close_auction,参与率上限 10% × amt_close_auction;
  超出部分改在 E 日 open10 卖出(L1b 价格与标准成本)—— 事后量,读作"若竞价成交额如实现值"。
统计:按期聚类(主)、期均值 t、i.i.d.(乐观下界)、正期数;窗口 dev(2021-2023 生效)/ test(2024-2025)/ all / preheat。
输出:research/ledger/inc_t2_legs_grid.csv、inc_t2_car.csv、inc_t2_car_summary.csv、inc_t2_h7_tercile.csv、
      cache/_tmp/inc_t2_event_legs.parquet、reports/inc_t2_car.png
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP, LEDGER, REPORTS, COST_BPS_ONEWAY
from afactor.panel import Panel
from afactor import strategy as ST, impact as IM, index_proxy as IP
from inc_t2_common import summarize, K_CTRL

LEGS = {  # name: (t0_kind, entry_ref, entry_off, entry_field, exit_ref, exit_off, exit_field, tradable)
    "preC": ("C", "C", -20, "close", "C", 0, "close", False),
    "L0": ("C", "C", 1, "open10", "A", 0, "close", True),
    "gap": ("A", "A", 0, "close", "A1", 0, "open10", False),
    "L1": ("A", "A1", 0, "open10", "Em1", 0, "close", True),
    "L1b": ("A", "A1", 0, "open10", "E", 0, "open10", True),
    "L2": ("A", "Em1", 0, "close", "E", 0, "close", False),
    "L2o": ("A", "Em1", 0, "close", "E", 0, "oauction", False),
    "L3_5": ("E", "E", 1, "open10", "E", 5, "close", True),
    "L3_20": ("E", "E", 1, "open10", "E", 20, "close", True),
    "L3_60": ("E", "E", 1, "open10", "E", 60, "close", True),
}
CAPS = [1e6, 1e7, 1e8, 1e9]
COSTS = ["none", "old", "Y0.5", "Y1", "Y1.5"]
MAXP = 0.10
TAUS = np.arange(-30, 61)
WINSETS = {"dev": ["dev"], "test": ["test"], "all": ["dev", "test"], "preheat": ["preheat"]}


def main():
    t00 = time.time()
    chk = pd.read_csv(LEDGER / "inc_t2_overlay_checks.csv")
    assert chk["passed"].all(), "Gate 1 未通过,不得读收益"
    p = Panel()
    dates, codes = p.dates, p.codes
    T, N = len(dates), len(codes)
    E = pd.read_parquet(TMP / "inc_t2_events.parquet")
    Cn = pd.read_parquet(TMP / "inc_t2_controls.parquet")
    Cn = Cn[Cn["tol"] == "tol10"]
    valid = p["valid"].to_numpy(bool)
    O10 = p.raw("twap_open10").to_numpy(np.float64)
    CL = p.raw("close_px").to_numpy(np.float64)
    CLf = p.raw("close_px").where(p["valid"]).ffill().to_numpy(np.float64)
    OA = p.raw("open_auction_px").to_numpy(np.float64)
    AO10 = p.raw("amt_open10").to_numpy(np.float64)
    AUC = p.raw("amt_close_auction").to_numpy(np.float64)
    dead = (p["one_word_up"] | p["one_word_down"]).to_numpy(bool)
    cm = IM.CostModel(p, impact_y=1.0)
    HS, SIG, ADV = cm._hs.astype(np.float64), cm._sigma.astype(np.float64), cm._adv.astype(np.float64)
    stamp_switch = cm._stamp_switch
    U = ST.universe(p, "V1000").reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
    PW = {k: IP.proxy_weights(p, k).to_numpy(np.float32) for k in ("000300", "000905", "000852")}
    print(f"[{time.time()-t00:.0f}s] 面板就绪", flush=True)

    def stamp(te):
        return (10.0 if te < stamp_switch else 5.0) * 1e-4

    def entry_ok(d, j, field):
        jj = np.where(j >= 0, j, 0)
        if field == "open10":
            ok = valid[d, jj] & np.isfinite(O10[d, jj]) & (AO10[d, jj] > 0) & ~dead[d, jj]
        else:
            ok = valid[d, jj] & np.isfinite(CL[d, jj])
        return ok & (j >= 0)

    def price(d, j, field, is_exit):
        jj = np.where(j >= 0, j, 0)
        if field == "open10":
            v = O10[d, jj]
        elif field == "oauction":
            v = OA[d, jj]
        else:
            v = CL[d, jj]
        if is_exit:
            v = np.where(np.isfinite(v) & valid[d, jj], v, CLf[d, jj])
        return v

    def leg_ret(d0, d1, j, f0, f1):
        ok = entry_ok(d0, j, f0)
        p0 = price(d0, j, f0, False)
        p1 = price(d1, j, f1, True)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = p1 / p0 - 1.0
        return np.where(ok & np.isfinite(r), r, np.nan), ok

    ref_i = {"C": "C_i", "A": "A_i", "A1": "A1_i", "Em1": "Em1_i", "E": "E_i"}
    ev = E[E["in_panel"]].copy()
    rows_ev = []
    for leg, (k0, er, eo, ef, xr, xo, xf, trad) in LEGS.items():
        q = ev[ev[f"q_{k0}"]].copy()
        d0 = q[ref_i[er]].to_numpy() + eo
        d1 = q[ref_i[xr]].to_numpy() + xo
        okd = (d0 >= 0) & (d1 < T) & (d1 > d0)
        q, d0, d1 = q[okd], d0[okd], d1[okd]
        j = q["j"].to_numpy()
        r_ev, ok_ev = leg_ret(d0, d1, j, ef, xf)
        cc = Cn[Cn["t0_kind"] == k0].pivot_table(index="ev_id", columns="rep", values="ctrl_j", aggfunc="first")
        CJ = cc.reindex(q["ev_id"]).fillna(-1).to_numpy(np.int64)
        r_c = np.full(CJ.shape, np.nan)
        for rr in range(CJ.shape[1]):
            r_c[:, rr], _ = leg_ret(d0, d1, CJ[:, rr], ef, xf)
        rc_mean = np.nanmean(r_c, axis=1) if CJ.shape[1] else np.full(len(q), np.nan)
        # pool / proxy 对照
        r_pool = np.full(len(q), np.nan)
        r_prx = np.full(len(q), np.nan)
        for key, g in pd.DataFrame({"d0": d0, "d1": d1, "t0": q[ref_i[k0]].to_numpy(), "idx": q["index"].to_numpy()}
                                   ).groupby(["d0", "d1", "t0", "idx"]).groups.items():
            a, b, t0, idx = key
            allj = np.arange(N)
            rr_all, ok_all = leg_ret(np.full(N, a), np.full(N, b), allj, ef, xf)
            sel = U[t0] & ok_all
            r_pool[np.asarray(g)] = np.nanmean(rr_all[sel]) if sel.any() else np.nan
            w = PW[idx][max(a - 1, 0)].astype(np.float64)
            m = (w > 0) & np.isfinite(rr_all)
            r_prx[np.asarray(g)] = float((w[m] * rr_all[m]).sum() / w[m].sum()) if m.any() else np.nan
        df = q[["ev_id", "source", "review", "window", "index", "code", "j", "direction", "etype", "subtype"]].copy()
        df["leg"] = leg
        df["tradable_leg"] = trad
        df["entry_ok"] = ok_ev
        df["d0"], df["d1"] = d0, d1
        df["r_ev"] = r_ev
        df["r_ctrl"] = rc_mean
        df["n_ctrl_ok"] = np.isfinite(r_c).sum(axis=1)
        df["r_pool"] = r_pool
        df["r_proxy"] = r_prx
        # 不可入场 -> 0(不持仓)
        for c, rcol in (("ar_matched", "r_ctrl"), ("ar_pool", "r_pool"), ("ar_proxy", "r_proxy")):
            df[c] = np.where(ok_ev, df["r_ev"] - df[rcol], 0.0)
            df.loc[ok_ev & ~np.isfinite(df[rcol]), c] = np.nan
        # L1 专用:E 日 open10 出场价(竞价参与率超限部分)
        if leg == "L1":
            r_b, _ = leg_ret(d0, q["E_i"].to_numpy(), j, "open10", "open10")
            df["r_ev_L1b"] = r_b
            df["auc_em1"] = AUC[d1, np.where(j >= 0, j, 0)]
        rows_ev.append(df)
        print(f"[{time.time()-t00:.0f}s] 腿 {leg}: {len(df)} 事件-腿", flush=True)
    L = pd.concat(rows_ev, ignore_index=True)

    # ---- 成本 ----
    # 每期每格(来源×指数×方向×类型×腿)事件数 -> 单笔 Q
    L["n_cell"] = L.groupby(["source", "index", "direction", "etype", "leg", "review"])["ev_id"].transform("size")
    L["n_cell_all"] = L.groupby(["source", "direction", "etype", "leg", "review"])["ev_id"].transform("size")
    jj = L["j"].to_numpy()
    d0 = L["d0"].to_numpy(); d1 = L["d1"].to_numpy()
    hs0, sg0, adv0 = HS[d0, jj], SIG[d0, jj], ADV[d0, jj]
    hs1, sg1, adv1 = HS[d1, jj], SIG[d1, jj], ADV[d1, jj]
    st1 = np.array([stamp(x) for x in d1])
    ao0 = AO10[d0, jj]
    is_l1 = (L["leg"] == "L1").to_numpy()
    auc1 = np.where(is_l1, AUC[d1, jj], np.nan)
    # L1 超限部分在 E open10 卖出:E 日的成本参数
    dE = np.where(is_l1, d1 + 1, d1)
    hsE, sgE, advE = HS[dE, jj], SIG[dE, jj], ADV[dE, jj]
    stE = np.array([stamp(x) for x in dE])
    rL1b = L["r_ev_L1b"].to_numpy() if "r_ev_L1b" in L else np.full(len(L), np.nan)
    trad = L["tradable_leg"].to_numpy() & L["entry_ok"].to_numpy()
    for scope, ncol in (("idx", "n_cell"), ("all", "n_cell_all")):
        n = L[ncol].to_numpy(np.float64)
        for cap in CAPS:
            Q = cap / n
            fill = np.clip(MAXP * ao0 / Q, 0, 1)
            Qf = Q * fill
            for cv in COSTS:
                if cv == "none":
                    cb = cs = np.zeros(len(L)); y = 0.0
                elif cv == "old":
                    cb = np.full(len(L), COST_BPS_ONEWAY * 1e-4); cs = COST_BPS_ONEWAY * 1e-4 + st1; y = 0.0
                else:
                    y = float(cv[1:])
                    cb = COST_BPS_ONEWAY * 1e-4 + hs0 + y * sg0 * np.sqrt(Qf / adv0)
                    cs = COST_BPS_ONEWAY * 1e-4 + st1 + hs1 + y * sg1 * np.sqrt(Qf / adv1)
                # L1:竞价卖出(冲击分母 = 实现的竞价成交额),超出 10% 竞价的部分改在 E open10 卖
                ar = L["ar_matched"].to_numpy()
                net = ar - cb - cs
                if cv not in ("none",):
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sell_auc_frac = np.where(is_l1, np.clip(MAXP * auc1 / Qf, 0, 1), 1.0)
                    if cv == "old":
                        cs_auc = COST_BPS_ONEWAY * 1e-4 + st1
                        cs_E = COST_BPS_ONEWAY * 1e-4 + stE
                    else:
                        Qa = Qf * sell_auc_frac
                        with np.errstate(invalid="ignore", divide="ignore"):
                            cs_auc = COST_BPS_ONEWAY * 1e-4 + st1 + hs1 + y * sg1 * np.sqrt(Qa / np.maximum(auc1, 1e3))
                        cs_E = COST_BPS_ONEWAY * 1e-4 + stE + hsE + y * sgE * np.sqrt(Qf * (1 - sell_auc_frac) / advE)
                    r_ev = L["r_ev"].to_numpy(); r_c = L["r_ctrl"].to_numpy()
                    net_l1 = (sell_auc_frac * (r_ev - cs_auc) + (1 - sell_auc_frac) * (rL1b - cs_E)) - r_c - cb
                    net = np.where(is_l1, net_l1, net)
                    if scope == "idx" and cap == 1e7 and cv == "Y1":
                        L["sell_auc_frac_1e7"] = np.where(is_l1, sell_auc_frac, np.nan)
                val = np.where(trad, fill * net, np.where(L["tradable_leg"], 0.0, np.nan))
                val = np.where(L["tradable_leg"] & L["entry_ok"] & ~np.isfinite(L["r_ctrl"].to_numpy()), np.nan, val)
                L[f"net_{scope}_{cv}_{int(cap):d}"] = val
                if cv == "Y1" and scope == "idx":
                    L[f"fill_idx_{int(cap):d}"] = np.where(L["tradable_leg"], fill, np.nan)
                if cv == "Y1" and cap == 1e7 and scope == "idx":
                    L["fill_1e7"] = np.where(L["tradable_leg"], fill, np.nan)
                    # 【审查修复 F2】往返成本 = 净值里**实际扣除**的成本:L1 的卖出按竞价分母(cs_auc)与
                    # 超限部分在 E open10 卖出(cs_E)加权;其余腿 = cb + cs。原列记录的是 cb + cs(冲击分母
                    # 取 ADV20),对 L1 低估约 0.15pp;原值保留在 cost_rt_adv20ref_Y1_1e7。
                    L["cost_rt_adv20ref_Y1_1e7"] = np.where(trad, cb + cs, np.nan)
                    L["cost_rt_Y1_1e7"] = np.where(trad, np.where(
                        is_l1, cb + sell_auc_frac * cs_auc + (1 - sell_auc_frac) * cs_E, cb + cs), np.nan)
    L.to_parquet(TMP / "inc_t2_event_legs.parquet", index=False)
    print(f"[{time.time()-t00:.0f}s] 事件-腿表 {len(L)} 行", flush=True)

    # ---- 网格 ----
    grid = []
    L["index_all"] = "ALL"
    for idx_col in ("index", "index_all"):
        for key, g in L.groupby(["source", idx_col, "direction", "etype", "leg"]):
            src, idx, d, et, leg = key
            scope = "idx" if idx_col == "index" else "all"
            for wn, ws in WINSETS.items():
                gw = g[g["window"].isin(ws)]
                if gw.empty:
                    continue
                base = dict(source=src, index=idx, direction=d, etype=et, leg=leg, window=wn,
                            n_events=len(gw), entry_ok_share=float(gw["entry_ok"].mean()))
                for ctl in ("matched", "pool", "proxy"):
                    s = summarize(gw[f"ar_{ctl}"].to_numpy(), gw["review"].to_numpy())
                    grid.append(dict(base, control=ctl, cost="gross", capital=np.nan, **s))
                if g["tradable_leg"].iloc[0]:
                    for cv in COSTS:
                        for cap in CAPS:
                            col = f"net_{scope}_{cv}_{int(cap):d}"
                            if cv in ("none", "old") and cap != 1e7:
                                continue
                            s = summarize(gw[col].to_numpy(), gw["review"].to_numpy())
                            grid.append(dict(base, control="matched", cost=cv, capital=cap, **s))
    G = pd.DataFrame(grid)
    G.to_csv(LEDGER / "inc_t2_legs_grid.csv", index=False)
    print(f"[{time.time()-t00:.0f}s] 网格 {len(G)} 行", flush=True)

    # ---- 成交比例随资金(审查修复 F6):10 亿篮子的买入受 10% × amt_open10 限制,未成交部分记 0 ----
    fl = []
    Lt = L[L["tradable_leg"] & L["entry_ok"] & L["window"].isin(["dev", "test"])]
    for key, g in Lt.groupby(["source", "index", "direction", "etype", "leg"]):
        for cap in CAPS:
            f_ = g[f"fill_idx_{int(cap):d}"].to_numpy(np.float64)
            v_ = g[f"net_idx_Y1_{int(cap):d}"].to_numpy(np.float64)
            ok = np.isfinite(f_) & np.isfinite(v_)
            if not ok.any() or f_[ok].sum() <= 0:
                continue
            fl.append(dict(zip(["source", "index", "direction", "etype", "leg"], key), capital=cap, n=int(ok.sum()),
                           fill_mean=float(f_[ok].mean()), fill_median=float(np.median(f_[ok])),
                           fill_share_full=float((f_[ok] >= 0.999).mean()),
                           net_Y1_per_event=float(v_[ok].mean()),
                           net_Y1_per_filled=float(v_[ok].sum() / f_[ok].sum())))
    pd.DataFrame(fl).to_csv(LEDGER / "inc_t2_fill_by_capital.csv", index=False)

    # ---- 逐期表(关键格)----
    keyc = [("O", "000905", "in", "ADD_800"), ("O", "000905", "out", "DEL_800"), ("O", "000300", "in", "ADD_800"),
            ("O", "000300", "out", "DEL_800"), ("PA", "000852", "in", "ADD_pure"), ("PA", "000852", "out", "DEL_pure"),
            ("PC", "000852", "in", "ADD_pure"), ("O", "000300", "in", "MIG_up"), ("O", "000905", "in", "MIG_down")]
    per = []
    for src, idx, d, et in keyc:
        g = L[(L["source"] == src) & (L["index"] == idx) & (L["direction"] == d) & (L["etype"] == et)]
        for (leg, rv), gg in g.groupby(["leg", "review"]):
            rec = dict(source=src, index=idx, direction=d, etype=et, leg=leg, review=rv, window=gg["window"].iloc[0],
                       n=len(gg), entry_ok_share=float(gg["entry_ok"].mean()), ar_matched=float(gg["ar_matched"].mean()),
                       ar_pool=float(gg["ar_pool"].mean()), ar_proxy=float(gg["ar_proxy"].mean()))
            if gg["tradable_leg"].iloc[0]:
                rec.update(net_Y1_1e7=float(gg["net_idx_Y1_10000000"].mean()),
                           net_Y1_1e8=float(gg["net_idx_Y1_100000000"].mean()),
                           fill_1e7=float(gg["fill_1e7"].mean()), cost_rt_Y1_1e7=float(gg["cost_rt_Y1_1e7"].mean()),
                           cost_rt_adv20ref_Y1_1e7=float(gg["cost_rt_adv20ref_Y1_1e7"].mean()))
            per.append(rec)
    pd.DataFrame(per).to_csv(LEDGER / "inc_t2_legs_by_period.csv", index=False)

    # ---- H7:按 E−1 异常竞价成交额三分位(事后分组,只诊断)----
    M = pd.read_parquet(TMP / "inc_t2_mechanism_events.parquet")[["ev_id", "lnauc_adv_Em1_diff"]]
    l1 = L[L["leg"] == "L1"].merge(M, on="ev_id", how="left")
    h7 = []
    for (src, idx, d, et), g in l1.groupby(["source", "index", "direction", "etype"]):
        g = g[g["window"].isin(["dev", "test"]) & np.isfinite(g["lnauc_adv_Em1_diff"])]
        if len(g) < 30:
            continue
        g = g.assign(terc=g.groupby("review")["lnauc_adv_Em1_diff"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if len(s) >= 3 else np.nan))
        for tq, gg in g.groupby("terc"):
            s1 = summarize(gg["ar_matched"].to_numpy(), gg["review"].to_numpy())
            s2 = summarize(gg["net_idx_Y1_10000000"].to_numpy(), gg["review"].to_numpy())
            h7.append(dict(source=src, index=idx, direction=d, etype=et, tercile=int(tq) + 1, n=len(gg),
                           lnauc_mean=float(gg["lnauc_adv_Em1_diff"].mean()), car_gross=s1.get("mean"),
                           t_gross=s1.get("t_cl"), car_net_Y1_1e7=s2.get("mean"), t_net=s2.get("t_cl")))
    pd.DataFrame(h7).to_csv(LEDGER / "inc_t2_h7_tercile.csv", index=False)

    # ---- CAR 路径(τ = −30..+60,τ=0 = E,对照 = A 日匹配)----
    ret = (p.raw("close_px").where(p["valid"]).ffill().pct_change()).to_numpy(np.float64)
    ret = np.where(np.isfinite(ret), ret, 0.0)
    q = ev[ev["q_A"]].copy()
    cc = Cn[Cn["t0_kind"] == "A"].pivot_table(index="ev_id", columns="rep", values="ctrl_j", aggfunc="first")
    CJ = cc.reindex(q["ev_id"]).fillna(-1).to_numpy(np.int64)
    Ei = q["E_i"].to_numpy()
    car_rows = []
    AR = np.full((len(q), len(TAUS)), np.nan)
    jq = q["j"].to_numpy()
    for k, tau in enumerate(TAUS):
        d = Ei + tau
        okd = (d >= 1) & (d < T)
        dd = np.clip(d, 0, T - 1)
        re = ret[dd, jq]
        rc = np.where(CJ >= 0, ret[dd[:, None], np.where(CJ >= 0, CJ, 0)], np.nan)
        AR[:, k] = np.where(okd, re - np.nanmean(rc, axis=1), np.nan)
    q = q.reset_index(drop=True)
    for (src, idx, d, et), g in q.groupby(["source", "index", "direction", "etype"]):
        for wn, ws in WINSETS.items():
            gi = g.index[g["window"].isin(ws)].to_numpy()
            if len(gi) == 0:
                continue
            C = np.nancumsum(AR[gi], axis=1)
            for k, tau in enumerate(TAUS):
                s = summarize(C[:, k], g.loc[gi, "review"].to_numpy())
                car_rows.append(dict(source=src, index=idx, direction=d, etype=et, window=wn, tau=int(tau),
                                     car=s.get("mean"), se_cl=s.get("se_cl"), lo95=s.get("lo95"), hi95=s.get("hi95"),
                                     n=s.get("n")))
    CAR = pd.DataFrame(car_rows)
    CAR.to_csv(LEDGER / "inc_t2_car.csv", index=False)
    # 汇总:各关键时点
    cal = pd.read_csv(LEDGER / "inc_t2_review_calendar.csv")
    G[(G["leg"].isin(["L0", "gap", "L1", "L1b", "L2", "L3_20"])) & (G["cost"].isin(["gross", "Y1"]))
      & ((G["capital"].isna()) | (G["capital"] == 1e7)) & (G["control"] == "matched")].to_csv(
        LEDGER / "inc_t2_car_summary.csv", index=False)
    plot(CAR)
    print(f"done {time.time()-t00:.0f}s")


def plot(CAR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    panels = [("O", "000905", "in", "ADD_800", "官方 中证500 调入(ADD_800)"),
              ("O", "000905", "out", "DEL_800", "官方 中证500 调出(DEL_800)"),
              ("O", "000300", "in", "ADD_800", "官方 沪深300 调入(800 外)"),
              ("O", "000300", "in", "MIG_up", "官方 500→300 迁移"),
              ("PA", "000852", "in", "ADD_pure", "代理预测 中证1000 调入(P^A)"),
              ("PA", "000852", "out", "DEL_pure", "代理预测 中证1000 调出(P^A)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (src, idx, d, et, title) in zip(axes.ravel(), panels):
        for wn, col in (("dev", "#3b6ea8"), ("test", "#c8553d")):
            s = CAR[(CAR["source"] == src) & (CAR["index"] == idx) & (CAR["direction"] == d) & (CAR["etype"] == et)
                    & (CAR["window"] == wn)].sort_values("tau")
            if s.empty:
                continue
            ax.plot(s["tau"], s["car"] * 100, color=col, lw=1.6, label=f"{'开发 2021-23' if wn == 'dev' else '检验 2024-25'}")
            ax.fill_between(s["tau"], s["lo95"] * 100, s["hi95"] * 100, color=col, alpha=0.12, lw=0)
        ax.axvline(0, color="#444", lw=0.8)
        ax.axvline(-1, color="#888", lw=0.8, ls="--")
        ax.axvline(-10, color="#888", lw=0.8, ls=":")
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("τ(交易日,0 = 生效日 E;虚线 E−1,点线 ≈ 公告日 A)", fontsize=8)
        ax.set_ylabel("CAR(%,对 k=3 匹配对照)", fontsize=8)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("INC_T2 指数调样事件 CAR(按期聚类 95% 区间;收盘到收盘,未扣成本)", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS / "inc_t2_car.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
