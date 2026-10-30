"""INC_T2 步骤 2:事件表、匹配对照(k=3)、平衡性(**不读任何收益**)。

预注册 §3.4、§4 步骤 2、§7.1、Gate 2。
事件来源:
  O   官方名单(沪深300 / 中证500;中证1000 缺,见 Gate 0)
  PA  代理预测(ST@E−10,现有缓存)       PC  代理预测(ST@C)
事件类型:
  预测链(PA/PC)三指数成分都在链内 -> ADD_pure / DEL_pure / MIG_up / MIG_down(预注册定义)。
  官方(O)只有 300/500 的官方成分,中证1000 的官方成分未知 -> 偏离:
      调入 k 且上期不在 300∪500  -> ADD_800(= 预注册 ADD_pure ∪ 1000→k 迁移,无法用官方数据区分)
      调出 k 且本期不在 300∪500  -> DEL_800(= DEL_pure ∪ k→1000 迁移)
      300↔500 之间 -> MIG_up / MIG_down(官方可确认)
    另用 PA 链的中证1000 成分给 ADD_800 / DEL_800 贴"代理子类"(ADD_pure* / MIG_up* 等),只作描述。
匹配:每个(来源, 期, 指数, 方向, 腿信号日 t0 ∈ {C, A, E})一组,处理股 = 本组在 t0 合格池内的事件股;
  候选(t0 = A、E):t0 合格池 − 本期任何来源、任何指数有变动的股票(官方名单在 A 盘后已公开,A、E 时点已知)。
  候选(t0 = C):t0 合格池 − 截至 C 已知的预测变动股(PC 来源,三指数、双方向)− 本组处理股本身。
      【审查修复 F3】原实现对 C 也剔除全部来源(含 A 日才公开的官方名单)的变动股,等于用 A 日的信息挑 C 日的对照
      (约 1-2% 候选的轻微前视,影响 L0 / preC 两腿);官方名单要到 A 盘后才公开,故 C 日不得用。
  pead_build_batches.match_arm(行业 + log 市值 /
  log ADV20 / 20 日波动的池内分位,L1-L4 逐级放宽,k=3,组内不放回);容差 0.10 与补救 0.05 都算好。
输出:
  cache/_tmp/inc_t2_events.parquet     事件(每行一个 来源×期×指数×股票×方向)
  cache/_tmp/inc_t2_controls.parquet   ev_id, t0_kind, tol, rep, ctrl_j, level, dist
  cache/_tmp/inc_t2_match_vars.parquet 平衡性复核用匹配变量
  research/ledger/inc_t2_balance.csv
  research/ledger/inc_t2_event_counts.csv
"""
from __future__ import annotations
import os, sys, time, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP, LEDGER
from afactor.panel import Panel
from afactor import earnings_events as EE, newdata as ND, index_proxy as IP
from pead_build_batches import match_arm, pool_quantile
from inc_t2_common import (IDX, IDX_RANK, calendar, window_of, K_CTRL, SEED, T0_KINDS)


def classify_chain(before: dict, after: dict, code: str, idx: str, direction: str) -> str:
    """预测链内分类。before/after: {index: set}。"""
    if direction == "in":
        src = [k for k in IDX if k != idx and code in before[k]]
        if not src:
            return "ADD_pure"
        return "MIG_up" if IDX_RANK[idx] < IDX_RANK[src[0]] else "MIG_down"
    dst = [k for k in IDX if k != idx and code in after[k]]
    if not dst:
        return "DEL_pure"
    return "MIG_up" if IDX_RANK[dst[0]] < IDX_RANK[idx] else "MIG_down"


def main():
    t00 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    cal = calendar(dates)
    Q = EE.qualified_pool(p)
    Qn = Q.to_numpy(bool)
    print(f"[{time.time()-t00:.0f}s] 合格池 {Qn.sum(1).mean():.0f} 只/日", flush=True)

    # ---- 成分 ----
    PA = pd.read_parquet(TMP / "inc_t2_pa_members.parquet")        # ST 取 A 日(审查修复 F4)
    PA0 = pd.read_parquet(IP.MEMBER_CACHE)                         # 项目缓存(ST 取 E−10)
    assert (PA.groupby(["review", "index"])["code"].apply(frozenset)
            .equals(PA0.groupby(["review", "index"])["code"].apply(frozenset))), "P^A(ST@A)与缓存成分不同,须重算下游"
    PC = pd.read_parquet(TMP / "inc_t2_pc_members.parquet")
    chains = {"PA": {(r, k): set(g["code"]) for (r, k), g in PA.groupby(["review", "index"])},
              "PC": {(r, k): set(g["code"]) for (r, k), g in PC.groupby(["review", "index"])}}
    OM = pd.read_parquet(TMP / "inc_t2_official_members.parquet")
    om = {(r, k, w): set(g["code"]) for (r, k, w), g in OM.groupby(["review", "index", "when"])}
    O = pd.read_csv(LEDGER / "inc_t2_official_changes.csv", dtype={"index": str})
    O["index"] = O["index"].str.zfill(6)
    P = pd.read_csv(LEDGER / "inc_t2_predicted_changes.csv", dtype={"index": str})
    P["index"] = P["index"].str.zfill(6)
    reviews = list(cal["review"])

    rows = []
    for _, r in O.iterrows():
        rv, idx, c, d = r["review"], r["index"], r["code"], r["direction"]
        other = "000905" if idx == "000300" else "000300"
        k = reviews.index(rv)
        if d == "in":
            if c in om[(rv, other, "before")]:
                et = "MIG_up" if idx == "000300" else "MIG_down"
                sub = et
            else:
                et = "ADD_800"
                if k == 0:
                    sub = "NA"
                else:
                    sub = "MIG_up*" if c in chains["PA"][(reviews[k - 1], "000852")] else "ADD_pure*"
        else:
            if c in om[(rv, other, "after")]:
                et = "MIG_down" if idx == "000300" else "MIG_up"
                sub = et
            else:
                et = "DEL_800"
                sub = "MIG_down*" if c in chains["PA"][(rv, "000852")] else "DEL_pure*"
        rows.append(dict(source="O", review=rv, index=idx, code=c, direction=d, etype=et, subtype=sub))
    for _, r in P.iterrows():
        src, rv, idx, c, d = r["source"], r["review"], r["index"], r["code"], r["direction"]
        k = reviews.index(rv)
        ch = chains[src]
        before = {x: ch[(reviews[k - 1], x)] for x in IDX}
        after = {x: ch[(rv, x)] for x in IDX}
        et = classify_chain(before, after, c, idx, d)
        rows.append(dict(source=src, review=rv, index=idx, code=c, direction=d, etype=et, subtype=et,
                         pred_rank=r["rank"]))
    E = pd.DataFrame(rows)
    E["window"] = E["review"].map(window_of)
    E["j"] = codes.get_indexer(E["code"])
    E["in_panel"] = E["j"] >= 0
    E = E.merge(cal[["review", "C_i", "A_i", "A1_i", "Em1_i", "E_i"]], on="review", how="left")
    for k in T0_KINDS:
        ii = E[f"{k}_i"].to_numpy()
        jj = E["j"].clip(lower=0).to_numpy()
        E[f"q_{k}"] = np.where(E["in_panel"], Qn[ii, jj], False)
    E = E.sort_values(["source", "review", "index", "direction", "code"]).reset_index(drop=True)
    E["ev_id"] = np.arange(len(E))
    print(f"[{time.time()-t00:.0f}s] 事件 {len(E)} 行", flush=True)

    # ---- 本期任何来源、任何指数有变动的股票(对照候选剔除)----
    changed = E.groupby("review")["j"].apply(lambda s: set(s[s >= 0].tolist())).to_dict()
    # 截至 C 已知的预测变动股(审查修复 F3):只含 PC(P^C 在 C 收盘即可形成)
    changed_pc = (E[E["source"] == "PC"].groupby("review")["j"]
                  .apply(lambda s: set(s[s >= 0].tolist())).to_dict())

    # ---- 匹配变量 ----
    inp = IP.load_inputs(p)
    mv, circ = inp["mv"], inp["circ"]
    logmv = np.log(mv.where(mv > 0)).fillna(np.log(circ.where(circ > 0)))
    logadv = np.log(p["adv20"].where(p["adv20"] > 0))
    vol = p["ret_c2c"].rolling(20, min_periods=15).std()
    raw_vals = np.stack([logmv.to_numpy(np.float64), logadv.to_numpy(np.float64), vol.to_numpy(np.float64)], axis=2)
    ind_lab = ND.sw_industry_labels(dates[:1], codes).iloc[0].to_numpy(np.int64)
    t_need = sorted(set(E[[f"{k}_i" for k in T0_KINDS]].to_numpy().ravel().tolist()))
    qv_rows = {}
    for t in t_need:
        qt = Q.iloc[[t]]
        qv_rows[t] = np.stack([pool_quantile(logmv.iloc[[t]], qt)[0], pool_quantile(logadv.iloc[[t]], qt)[0],
                               pool_quantile(vol.iloc[[t]], qt)[0]], axis=1)
    print(f"[{time.time()-t00:.0f}s] 匹配变量就绪({len(t_need)} 个 t0)", flush=True)

    # ---- 匹配 ----
    crow, vrow = [], []
    for (src, rv, idx, d), g in E.groupby(["source", "review", "index", "direction"], sort=True):
        for k in T0_KINDS:
            gk = g[g[f"q_{k}"]]
            if gk.empty:
                continue
            t = int(gk[f"{k}_i"].iloc[0])
            ej = gk["j"].to_numpy(np.int64)
            if k == "C":
                drop = changed_pc.get(rv, set()) | set(g.loc[g["j"] >= 0, "j"].tolist())
            else:
                drop = changed[rv]
            cand = np.setdiff1d(np.flatnonzero(Qn[t]), np.array(sorted(drop), dtype=np.int64))
            arm = zlib.crc32(f"{src}|{idx}|{d}|{k}".encode()) % 100000
            for tol_tag, tol1 in (("tol10", 0.10), ("tol05", 0.05)):
                ctrl, lvl, dist = match_arm(ej, cand, qv_rows[t], ind_lab, arm, 0, t, tol1)
                for rr in range(K_CTRL):
                    for s, ev in enumerate(gk["ev_id"].to_numpy()):
                        crow.append((ev, k, tol_tag, rr + 1, int(ctrl[rr, s]), int(lvl[rr, s]), dist[rr, s]))
            for s, ev in enumerate(gk["ev_id"].to_numpy()):
                vrow.append((ev, k, "treat", 0, int(ind_lab[ej[s]]), *raw_vals[t, ej[s]]))
    Cn = pd.DataFrame(crow, columns=["ev_id", "t0_kind", "tol", "rep", "ctrl_j", "level", "dist"])
    Cn.to_parquet(TMP / "inc_t2_controls.parquet", index=False)
    # 对照的匹配变量
    t0_of = E.set_index("ev_id")[[f"{k}_i" for k in T0_KINDS]]
    for (k, tol), g in Cn.groupby(["t0_kind", "tol"]):
        tt = t0_of.loc[g["ev_id"], f"{k}_i"].to_numpy()
        jj = g["ctrl_j"].to_numpy()
        ok = jj >= 0
        v = np.full((len(g), 3), np.nan)
        v[ok] = raw_vals[tt[ok], jj[ok]]
        for i, (ev, rep, lv, cj) in enumerate(zip(g["ev_id"], g["rep"], g["level"], jj)):
            vrow.append((ev, k, f"ctrl_{tol}", rep, int(ind_lab[cj]) if cj >= 0 else -99, *v[i]))
    V = pd.DataFrame(vrow, columns=["ev_id", "t0_kind", "grp", "rep", "ind", "logmv", "logadv", "vol"])
    V = V.merge(E[["ev_id", "source", "review", "window", "index", "direction", "etype"]], on="ev_id")
    V.to_parquet(TMP / "inc_t2_match_vars.parquet", index=False)
    lv = Cn.merge(E[["ev_id", "source", "review", "window", "index", "direction", "etype"]], on="ev_id")

    # ---- 平衡性 ----
    bal = []
    groups = V[V["grp"] == "treat"].groupby(["source", "index", "etype", "t0_kind"])
    for (src, idx, et, k), gt in groups:
        for tol in ("tol10", "tol05"):
            gc = V[(V["grp"] == f"ctrl_{tol}") & (V["source"] == src) & (V["index"] == idx)
                   & (V["etype"] == et) & (V["t0_kind"] == k)]
            gl = lv[(lv["tol"] == tol) & (lv["source"] == src) & (lv["index"] == idx) & (lv["etype"] == et)
                    & (lv["t0_kind"] == k)]
            for wn, wsel in (("dev", ["dev"]), ("test", ["test"]), ("all", ["dev", "test"])):
                x = gt[gt["window"].isin(wsel)]
                y = gc[gc["window"].isin(wsel)]
                yl = gl[gl["window"].isin(wsel)]
                if len(x) == 0:
                    continue
                yo = y[y["ind"] != -99]
                rec = dict(source=src, index=idx, etype=et, t0_kind=k, tol=tol, window=wn, n_treated=len(x),
                           n_ctrl_slots=len(y), n_failed=int((yl["ctrl_j"] < 0).sum()))
                for vname in ("logmv", "logadv", "vol"):
                    xt, xc = x[vname].dropna(), yo[vname].dropna()
                    s = np.sqrt((xt.var() + xc.var()) / 2.0) if len(xt) > 1 and len(xc) > 1 else np.nan
                    rec[f"smd_{vname}"] = float((xt.mean() - xc.mean()) / s) if s and s > 0 else np.nan
                st_ = x["ind"].value_counts(normalize=True)
                sc_ = yo["ind"].value_counts(normalize=True)
                u = st_.index.union(sc_.index)
                rec["ind_maxdiff_pp"] = float((st_.reindex(u, fill_value=0) - sc_.reindex(u, fill_value=0)).abs().max() * 100)
                for L in (1, 2, 3, 4):
                    rec[f"level{L}_share"] = float((yl.loc[yl["ctrl_j"] >= 0, "level"] == L).mean()) if len(yl) else np.nan
                rec["pass"] = bool(all(abs(rec[f"smd_{v}"]) <= 0.10 for v in ("logmv", "logadv", "vol")
                                       if np.isfinite(rec[f"smd_{v}"])) and rec["ind_maxdiff_pp"] <= 5.0)
                bal.append(rec)
    B = pd.DataFrame(bal)
    B.to_csv(LEDGER / "inc_t2_balance.csv", index=False)

    # ---- 事件计数 ----
    cnt = (E.groupby(["source", "index", "direction", "etype", "review"])
           .agg(n=("ev_id", "size"), n_in_panel=("in_panel", "sum"), n_q_C=("q_C", "sum"),
                n_q_A=("q_A", "sum"), n_q_E=("q_E", "sum")).reset_index())
    cnt.to_csv(LEDGER / "inc_t2_event_counts.csv", index=False)
    E.to_parquet(TMP / "inc_t2_events.parquet", index=False)
    piv = cnt.pivot_table(index=["source", "index", "direction", "etype"], columns="review", values="n_q_A",
                          aggfunc="sum", fill_value=0)
    print(piv.to_string())
    show = B[(B["window"] == "all") & (B["tol"] == "tol10")]
    print(show[["source", "index", "etype", "t0_kind", "n_treated", "n_failed", "smd_logmv", "smd_logadv",
                "smd_vol", "ind_maxdiff_pp", "pass"]].round(3).to_string())
    print(f"done {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
