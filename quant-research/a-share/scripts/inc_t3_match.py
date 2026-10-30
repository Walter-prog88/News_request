"""INC-T3 S1:事件的匹配对照 B1 / B2 + 平衡性 G6(读任何收益之前)。

预注册 §7.1:每个在池事件配 k=3 只对照;候选 = 信号行 s = D-21 的 V1000 池(非 ST),剔除事件股与
在 [s, D+60] 内自身有 ratio_plan >= 5% 解禁的股票;匹配变量 = 申万一级 + 三个池内分位
(log 真实市值、log ADV20、20 日 ret_c2c 波动),层级 L1-L4(同行业且三分位差 <= tol1 / <= 0.20 /
不限行业 <= tol1 / 最近),tol1 = 0.10;同一解禁日 D 的事件为一组,组内每个复制号 r 不放回;
同一事件的 3 个对照互不相同。B2 = B1 + 同上市年龄桶(< 250 / 250~750 / > 750 交易日,硬约束)。
种子 np.random.default_rng([20261011, 31, event_idx, r])(只用于距离平局)。

平衡性 G6(§7.1):B1、B2 在开发窗 / 检验窗上 |SMD| <= 0.10、行业占比最大差 <= 5pp;
主样本 = 在池且 ratio_plan >= 5% 的事件。不达标 -> 读收益前把 tol1 收紧到 0.05 重配一次(--tol 0.05)。

用法  python3 scripts/inc_t3_match.py [--tol 0.10]
产出  cache/_tmp/inc_t3_matches.parquet、research/ledger/inc_t3_balance.csv
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t3_common as C                                   # noqa: E402
from afactor.panel import Panel                             # noqa: E402
from afactor import strategy as ST, newdata as ND           # noqa: E402

K_CTRL = 3
TOL2 = 0.20
ARM_SEED = 31


def pool_quantile(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """同日池内分位(0~1];池内缺失 -> 0.5;池外 NaN。同 pead_build_batches.pool_quantile。"""
    df = pd.DataFrame(np.where(Q, X, np.nan))
    r = df.rank(axis=1, pct=True).to_numpy()
    r = np.where(Q & ~np.isfinite(r), 0.5, r)
    return np.where(Q, r, np.nan)


def match_group(ev_idx, treat_j, cand_j, qv, ind, k=K_CTRL, tol1=0.10, tol2=TOL2, age_t=None, age_c=None):
    """一组(同一 D)事件的匹配。返回 ctrl (k, nT) 股票列号(-1 失败)、lvl、dist。
    逐级放宽同 scripts/pead_build_batches.match_arm;age_t / age_c 给定时同年龄桶为硬约束(B2)。"""
    nT, nC = len(treat_j), len(cand_j)
    ctrl = np.full((k, nT), -1, np.int64)
    pos = np.full((k, nT), -1, np.int64)
    lvl = np.zeros((k, nT), np.int8)
    dist = np.full((k, nT), np.nan)
    if nC == 0:
        return ctrl, lvl, dist
    qT, qC = qv[treat_j], qv[cand_j]
    iT, iC = ind[treat_j], ind[cand_j]
    for r in range(1, k + 1):
        used = np.zeros(nC, bool)
        for i in range(nT):                               # 组内按 event_idx 升序(确定性)
            rng = np.random.default_rng([C.SEED, ARM_SEED, int(ev_idx[i]), r])
            avail = ~used
            for rr in range(r - 1):
                if pos[rr, i] >= 0:
                    avail[pos[rr, i]] = False
            if age_t is not None:
                avail = avail & (age_c == age_t[i])
            if not avail.any():
                continue
            dq = np.abs(qC - qT[i])
            mx = dq.max(axis=1)
            d = np.sqrt((dq * dq).sum(axis=1))
            same = iC == iT[i]
            chosen_level, mm = 4, avail
            for L, m in ((1, same & (mx <= tol1)), (2, same & (mx <= tol2)), (3, mx <= tol1)):
                cand = m & avail
                if cand.any():
                    chosen_level, mm = L, cand
                    break
            ci = np.flatnonzero(mm)
            dd = d[ci]
            best = dd.min()
            ties = ci[dd <= best + 1e-12]
            c = int(ties[rng.integers(len(ties))]) if len(ties) > 1 else int(ties[0])
            used[c] = True
            pos[r - 1, i] = c
            ctrl[r - 1, i] = cand_j[c]
            lvl[r - 1, i] = chosen_level
            dist[r - 1, i] = best
    return ctrl, lvl, dist


def balance_table(V: pd.DataFrame, tag: str) -> pd.DataFrame:
    """处理组 vs 池化对照:按窗口(事件日 D)算 SMD、行业占比最大差、层级占比、失败数。"""
    out = []
    wins = {"2020": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")), **C.WINDOWS}
    for arm in ("B1", "B2"):
        for wn, (a, b) in wins.items():
            x = V[(V["role"] == "T") & (V["D"] >= a) & (V["D"] <= b)]
            y = V[(V["role"] == arm) & (V["D"] >= a) & (V["D"] <= b)]
            yo = y[y["ok"]]
            rec = dict(sample=tag, control=arm, window=wn, n_treated=len(x), n_control_slots=len(y),
                       n_failed=int((~y["ok"]).sum()), fail_share=float((~y["ok"]).mean()) if len(y) else np.nan)
            for v in ("logmv", "logadv", "vol", "age_bucket"):
                xt, xc = x[v].dropna(), yo[v].dropna()
                s = np.sqrt((xt.var() + xc.var()) / 2.0)
                rec[f"smd_{v}"] = float((xt.mean() - xc.mean()) / s) if s > 0 else np.nan
                rec[f"mean_t_{v}"] = float(xt.mean()); rec[f"mean_c_{v}"] = float(xc.mean())
            st = x["ind"].value_counts(normalize=True)
            sc = yo["ind"].value_counts(normalize=True)
            u = st.index.union(sc.index)
            dif = st.reindex(u, fill_value=0) - sc.reindex(u, fill_value=0)
            rec["ind_maxdiff_pp"] = float(dif.abs().max() * 100) if len(dif) else np.nan
            for L in (1, 2, 3, 4):
                rec[f"level{L}_share"] = float((yo["level"] == L).mean()) if len(yo) else np.nan
            rec["pass"] = bool(all(abs(rec[f"smd_{v}"]) <= 0.10 for v in ("logmv", "logadv", "vol"))
                               and rec["ind_maxdiff_pp"] <= 5.0)
            out.append(rec)
    return pd.DataFrame(out)


def main(tol1: float = 0.10):
    t0 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    T, N = len(dates), len(codes)
    ev = C.load_events()
    allev = ev.copy()
    ev = ev[ev["in_pool"]].copy().sort_values(["d_pos", "event_idx"]).reset_index(drop=True)
    print(f"在池事件 {len(ev)}", flush=True)

    U = ST.universe(p, C.VERSION).to_numpy(bool)
    st = p["is_st"].reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
    Q = U & ~st
    mv = ND.total_mv(dates, codes)
    circ = ND.circ_mv(dates, codes)
    logmv = np.log(mv.where(mv > 0)).fillna(np.log(circ.where(circ > 0))).to_numpy(np.float64)
    del mv, circ
    logadv = np.log(p["adv20"].where(p["adv20"] > 0)).to_numpy(np.float64)
    vol = p["ret_c2c"].rolling(20, min_periods=15).std().to_numpy(np.float64)
    ind = ND.sw_industry_labels(dates[:1], codes).iloc[0].to_numpy(np.int64)
    age = C.listing_age_panel(dates, codes)
    ageb = C.age_bucket(age)
    print(f"[{time.time()-t0:.0f}s] 匹配变量就绪", flush=True)

    # 防污染:每只股票 ratio_plan >= 5% 的事件日(全部事件,不限在池)
    big = allev[allev["ratio_plan"] >= 0.05]
    big_pos = {j: g["d_pos"].to_numpy() for j, g in big.groupby("j")}
    big_j = np.array(sorted(big_pos.keys()), np.int64)

    rows, vrows = [], []
    for d_pos, g in ev.groupby("d_pos", sort=True):
        s = int(d_pos) - 21
        if s < 0:
            continue
        qv = np.stack([pool_quantile(logmv[s][None, :], Q[s][None, :])[0],
                       pool_quantile(logadv[s][None, :], Q[s][None, :])[0],
                       pool_quantile(vol[s][None, :], Q[s][None, :])[0]], axis=1)
        cand = np.flatnonzero(Q[s])
        # 剔除事件股 + 在 [s, D+60] 内自身有 >= 5% 解禁的股票
        hi = int(d_pos) + 60
        bad = [j for j in big_j if ((big_pos[j] >= s) & (big_pos[j] <= hi)).any()]
        cand = np.setdiff1d(cand, np.union1d(g["j"].to_numpy(np.int64), np.asarray(bad, np.int64)))
        tj = g["j"].to_numpy(np.int64)
        eidx = g["event_idx"].to_numpy(np.int64)
        for arm in ("B1", "B2"):
            kw = {} if arm == "B1" else dict(age_t=ageb[s, tj], age_c=ageb[s, cand])
            ctrl, lvl, dist = match_group(eidx, tj, cand, qv, ind, tol1=tol1, **kw)
            for r in range(K_CTRL):
                for i in range(len(tj)):
                    rows.append((arm, r + 1, int(eidx[i]), int(d_pos), s, int(ctrl[r, i]), int(lvl[r, i]), dist[r, i]))
        if d_pos % 100 == 0:
            print(f"  d_pos {d_pos} n_ev {len(g)} n_cand {len(cand)} [{time.time()-t0:.0f}s]", flush=True)

    M = pd.DataFrame(rows, columns=["arm", "rep", "event_idx", "d_pos", "s_pos", "j_ctrl", "level", "dist"])
    M["tol1"] = tol1
    M.to_parquet(C.TMP / "inc_t3_matches.parquet", index=False)
    print(f"[{time.time()-t0:.0f}s] 匹配 {len(M)} 行;失败 {(M['j_ctrl'] < 0).sum()}", flush=True)

    # ---- 平衡性 ----
    def vars_at(s_pos, j):
        ok = j >= 0
        jj = np.where(ok, j, 0)
        return dict(ok=ok, ind=np.where(ok, ind[jj], -99), logmv=np.where(ok, logmv[s_pos, jj], np.nan),
                    logadv=np.where(ok, logadv[s_pos, jj], np.nan), vol=np.where(ok, vol[s_pos, jj], np.nan),
                    age_bucket=np.where(ok, ageb[s_pos, jj], np.nan).astype(np.float64))
    evm = ev[ev["d_pos"] >= 21].copy()
    tv = pd.DataFrame(dict(role="T", event_idx=evm["event_idx"].to_numpy(), D=evm["D"].to_numpy(), level=0,
                           **vars_at(evm["d_pos"].to_numpy() - 21, evm["j"].to_numpy(np.int64))))
    Dmap = evm.set_index("event_idx")["D"]
    cv = pd.DataFrame(dict(role=M["arm"].to_numpy(), event_idx=M["event_idx"].to_numpy(),
                           D=Dmap.reindex(M["event_idx"]).to_numpy(), level=M["level"].to_numpy(),
                           **vars_at(M["s_pos"].to_numpy(), M["j_ctrl"].to_numpy(np.int64))))
    V = pd.concat([tv, cv], ignore_index=True)
    rp = ev.set_index("event_idx")["ratio_plan"]
    V["ratio_plan"] = rp.reindex(V["event_idx"]).to_numpy()
    bal = pd.concat([balance_table(V[V["ratio_plan"] >= 0.05], "in_pool_ge5(主样本,G6)"),
                     balance_table(V[V["ratio_plan"] >= 0.10], "in_pool_ge10"),
                     balance_table(V, "in_pool_all")], ignore_index=True)
    bal["tol1"] = tol1
    bal["gate"] = np.where(bal["sample"].str.contains("G6") & bal["window"].isin(["2021-2023", "2024-2025"]), "G6", "info")
    bal.to_csv(C.LEDGER / "inc_t3_balance.csv", index=False)
    show = ["sample", "control", "window", "n_treated", "fail_share", "smd_logmv", "smd_logadv", "smd_vol",
            "smd_age_bucket", "ind_maxdiff_pp", "level1_share", "level4_share", "pass", "gate"]
    print(bal[show].round(3).to_string())
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    tol = 0.10
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])
    main(tol)
