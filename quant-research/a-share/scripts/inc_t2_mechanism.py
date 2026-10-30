"""INC_T2 步骤 3:机制(量)—— E−1 收盘集合竞价异常成交额(H1、Gate 3)。**不含任何收益。**

预注册 §7.2:
  x_d = ln(amt_close_auction[d] / ADV20_A),d ∈ {E−2(安慰剂), E−1(被动成交日), E, E+1};
  另报 amt_closeauc_share[d]、amt_last10_share[d]。
  事件 − 对照(k=3,公告日 A 匹配,容差 0.10;对照取均值,缺失的对照不计),按期聚类 t;开发 / 检验分报。
  ADV20_A = 公告日 A 的 20 日日均成交额(A 收盘已知);竞价成交额为 0 的股票日按 1000 元下限取对数(全面板占 0.13%)。
  逐期均值(被动资金规模的代理时间序列)单独落盘。
输出:research/ledger/inc_t2_mechanism.csv、inc_t2_mechanism_by_period.csv、inc_t2_gate3.csv
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP, LEDGER
from afactor.panel import Panel
from inc_t2_common import summarize, K_CTRL

DAYS = {"Em2": -2, "Em1": -1, "E0": 0, "Ep1": 1}
FLOOR = 1e3


def main():
    t0 = time.time()
    p = Panel()
    E = pd.read_parquet(TMP / "inc_t2_events.parquet")
    Cn = pd.read_parquet(TMP / "inc_t2_controls.parquet")
    Cn = Cn[(Cn["t0_kind"] == "A") & (Cn["tol"] == "tol10")]
    auc = p.raw("amt_close_auction").to_numpy(np.float64)
    share = p.raw("amt_closeauc_share").to_numpy(np.float64)
    l10 = p.raw("amt_last10_share").to_numpy(np.float64)
    valid = p["valid"].to_numpy(bool)
    adv = p["adv20"].to_numpy(np.float64)
    T = auc.shape[0]

    ev = E[E["q_A"]].copy()
    ctrl = Cn.pivot_table(index="ev_id", columns="rep", values="ctrl_j", aggfunc="first").reindex(ev["ev_id"])
    CJ = ctrl.to_numpy(np.int64) if len(ctrl) else np.zeros((0, K_CTRL), np.int64)
    rows = []
    a_i = ev["A_i"].to_numpy()
    jj = ev["j"].to_numpy()

    def metric(name, dd, j_arr):
        """j_arr: (n,) 或 (n,k);返回同形状数组(无效为 NaN)。"""
        jn = np.where(j_arr >= 0, j_arr, 0)
        ai = a_i if j_arr.ndim == 1 else a_i[:, None]
        di = (ev["E_i"].to_numpy() + dd)
        di = di if j_arr.ndim == 1 else di[:, None]
        ok = (j_arr >= 0) & valid[di, jn]
        if name == "lnauc_adv":
            v = np.log(np.maximum(auc[di, jn], FLOOR) / adv[ai, jn])
        elif name == "auc_share":
            v = share[di, jn]
        else:
            v = l10[di, jn]
        return np.where(ok & np.isfinite(v), v, np.nan)

    out = ev[["ev_id", "source", "review", "window", "index", "direction", "etype", "subtype", "code"]].copy()
    for dn, dd in DAYS.items():
        for m in ("lnauc_adv", "auc_share", "last10_share"):
            xe = metric(m, dd, jj)
            xc = np.nanmean(metric(m, dd, CJ), axis=1) if len(CJ) else np.array([])
            out[f"{m}_{dn}_ev"] = xe
            out[f"{m}_{dn}_ctrl"] = xc
            out[f"{m}_{dn}_diff"] = xe - xc
    out.to_parquet(TMP / "inc_t2_mechanism_events.parquet", index=False)

    for (src, idx, d, et), g in out.groupby(["source", "index", "direction", "etype"]):
        for wn, wsel in (("dev", ["dev"]), ("test", ["test"]), ("all", ["dev", "test"]), ("preheat", ["preheat"])):
            gw = g[g["window"].isin(wsel)]
            if gw.empty:
                continue
            for dn in DAYS:
                for m in ("lnauc_adv", "auc_share", "last10_share"):
                    s = summarize(gw[f"{m}_{dn}_diff"].to_numpy(), gw["review"].to_numpy())
                    rows.append(dict(source=src, index=idx, direction=d, etype=et, window=wn, day=dn, metric=m,
                                     ev_level=float(np.nanmean(gw[f"{m}_{dn}_ev"])),
                                     ctrl_level=float(np.nanmean(gw[f"{m}_{dn}_ctrl"])), **s))
    M = pd.DataFrame(rows)
    M.to_csv(LEDGER / "inc_t2_mechanism.csv", index=False)
    byp = (out.groupby(["source", "index", "direction", "etype", "review", "window"])
           [[f"lnauc_adv_{d}_diff" for d in DAYS] + ["auc_share_Em1_ev", "auc_share_Em1_ctrl"]]
           .mean().reset_index())
    byp.to_csv(LEDGER / "inc_t2_mechanism_by_period.csv", index=False)

    # ---- Gate 3:各 (来源, 指数, 方向, 类型) 的 H1 判定(主格为中证1000 官方 ADD_pure,缺)----
    g3 = []
    for (src, idx, d, et), g in M[M["metric"] == "lnauc_adv"].groupby(["source", "index", "direction", "etype"]):
        def get(w, day, col):
            r = g[(g["window"] == w) & (g["day"] == day)]
            return float(r[col].iloc[0]) if len(r) else np.nan
        t_all = get("all", "Em1", "t_cl")
        dev, test = get("dev", "Em1", "mean"), get("test", "Em1", "mean")
        t_pl = get("all", "Em2", "t_cl")
        g3.append(dict(source=src, index=idx, direction=d, etype=et, Em1_mean_all=get("all", "Em1", "mean"),
                       Em1_t_cl=t_all, Em1_dev=dev, Em1_test=test, Em2_mean_all=get("all", "Em2", "mean"),
                       Em2_t_cl=t_pl, E0_mean_all=get("all", "E0", "mean"), Ep1_mean_all=get("all", "Ep1", "mean"),
                       gate3_pass=bool(t_all >= 2 and dev > 0 and test > 0 and abs(t_pl) < 2)))
    G3 = pd.DataFrame(g3)
    G3.to_csv(LEDGER / "inc_t2_gate3.csv", index=False)
    print(G3.round(3).to_string())
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
