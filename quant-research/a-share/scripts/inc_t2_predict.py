"""INC_T2 步骤 1:预测名单(P^A、P^C)与预测准确率(不读任何收益)。

预注册 §3.3:
  P^A  现有缓存 cache/_tmp/proxy_membership.parquet(index_proxy.select_members,ST 取 E−10 公告日代理)。
       先用同一输入重跑 select_members 核对缓存逐只一致。
       【审查修复 F4】缓存的 ST 状态取 E−10,而官方公告日 A 在 10 期里有 7 期早于 E−10 一天(E−10 = A+1),
       以 A 为信号日的 P^A 因此用到 A+1 的 ST 状态(1 天前视)。修复:本脚本另算 ann_idx = A_i(ST 与"已退市"
       判断都取 A 日)的 P^A,存 cache/_tmp/inc_t2_pa_members.parquet,**本轨道后续一律使用这一版**;
       与缓存版的逐期差异写入 research/ledger/inc_t2_pa_st_diff.csv。项目共用缓存不动。
  P^C  新增变体:review_schedule 的 ann_idx = cut_idx、announce = cutoff(ST 状态与"已退市"判断都取截止日 C),
       其余规则不变 —— 截止日收盘即可形成。
  预测变动 = 本链本期成分 − 本链上期成分(调入),上期 − 本期(调出);2020H1 为初始状态,无变动。
准确率:每期每指数,召回 |P∩O|/|O|、精确率 |P∩O|/|P|,分调入/调出、分 P^A/P^C;中证1000 无官方名单(Gate 0),只报预测数。
另记官方名单中不在研究面板的数量(幸存者缺口)。

输出:
  research/ledger/inc_t2_predicted_changes.csv   source(PA/PC), review, index, code, direction, rank
  research/ledger/inc_t2_prediction_accuracy.csv
  research/ledger/inc_t2_official_coverage.csv   每期 × 指数 × 方向:官方数、面板内数(合格池数在步骤 2 补)
  cache/_tmp/inc_t2_pc_members.parquet           P^C 成分长表
  cache/_tmp/inc_t2_pa_members.parquet           P^A(ST 取 A 日)成分长表(审查修复 F4)
  research/ledger/inc_t2_pa_st_diff.csv          P^A(ST@A)与缓存版(ST@E−10)的逐期逐指数差异
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP, LEDGER
from afactor.panel import Panel
from afactor import index_proxy as IP
from inc_t2_common import calendar

IDX = ["000300", "000905", "000852"]


def chain_changes(mem: dict, reviews: list, src: str) -> pd.DataFrame:
    rows = []
    for k in range(1, len(reviews)):
        rv, prv = reviews[k], reviews[k - 1]
        for idx in IDX:
            cur, prev = mem[rv][idx], set(mem[prv][idx])
            rank = {c: i + 1 for i, c in enumerate(cur)}
            for c in cur:
                if c not in prev:
                    rows.append((src, rv, idx, c, "in", rank[c]))
            for c in prev - set(cur):
                rows.append((src, rv, idx, c, "out", np.nan))
    return pd.DataFrame(rows, columns=["source", "review", "index", "code", "direction", "rank"])


def main():
    t0 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    assert dates.max() <= pd.Timestamp("2025-12-31")
    inp = IP.load_inputs(p)
    valid = p["valid"]
    amount = p.raw("amount")
    sched = IP.review_schedule(dates)
    print(f"[{time.time()-t0:.0f}s] 输入就绪;{len(sched)} 期", flush=True)

    # ---- P^A:重跑核对缓存 ----
    memA, diagA = IP.select_members(inp, valid, amount, sched)
    cache = pd.read_parquet(IP.MEMBER_CACHE)
    mism = 0
    for rv, g in cache.groupby("review"):
        for idx, gg in g.groupby("index"):
            a = list(gg.sort_values("rank")["code"])
            if a != memA[rv][idx]:
                mism += 1
    print(f"[{time.time()-t0:.0f}s] P^A 重跑 vs 缓存 不一致的 (期, 指数) 数 = {mism}", flush=True)
    assert mism == 0, "P^A 重跑与缓存不一致"

    # ---- P^A(审查修复 F4):ST / 退市判断取官方公告日 A,而不是 E−10 ----
    cal = calendar(dates)
    a_idx = dict(zip(cal["review"], cal["A_i"]))
    sa = sched.copy()
    sa["ann_idx"] = sa["review"].map(a_idx).astype(int)
    sa["announce"] = [dates[i] for i in sa["ann_idx"]]
    memA_st, diagA_st = IP.select_members(inp, valid, amount, sa)
    rows = []
    for rv in memA_st:
        for idx in IDX:
            for j, c in enumerate(memA_st[rv][idx]):
                rows.append((rv, idx, c, j + 1))
    pd.DataFrame(rows, columns=["review", "index", "code", "rank"]).to_parquet(TMP / "inc_t2_pa_members.parquet", index=False)
    diagA_st.assign(source="PA").to_csv(LEDGER / "inc_t2_pa_diag.csv", index=False)
    dif = []
    for rv in sched["review"]:
        for idx in IDX:
            a, b = set(memA[rv][idx]), set(memA_st[rv][idx])
            dif.append(dict(review=rv, index=idx, n_cache=len(a), n_stA=len(b), only_cache=len(a - b), only_stA=len(b - a),
                            only_cache_codes=";".join(sorted(a - b)), only_stA_codes=";".join(sorted(b - a)),
                            ann_idx_cache=int(sched.loc[sched["review"] == rv, "ann_idx"].iloc[0]),
                            ann_idx_stA=int(sa.loc[sa["review"] == rv, "ann_idx"].iloc[0])))
    DF = pd.DataFrame(dif)
    DF.to_csv(LEDGER / "inc_t2_pa_st_diff.csv", index=False)
    print("P^A(ST@A) vs 缓存(ST@E−10) 逐期差异:", flush=True)
    print(DF.drop(columns=["only_cache_codes", "only_stA_codes"]).to_string(), flush=True)

    # ---- P^C:ST / 退市判断取截止日 ----
    sc = sched.copy()
    sc["ann_idx"] = sc["cut_idx"]
    sc["announce"] = sc["cutoff"]
    memC, diagC = IP.select_members(inp, valid, amount, sc)
    rows = []
    for rv in memC:
        for idx in IDX:
            for j, c in enumerate(memC[rv][idx]):
                rows.append((rv, idx, c, j + 1))
    pd.DataFrame(rows, columns=["review", "index", "code", "rank"]).to_parquet(TMP / "inc_t2_pc_members.parquet", index=False)
    diagC.assign(source="PC").to_csv(LEDGER / "inc_t2_pc_diag.csv", index=False)

    reviews = list(sched["review"])
    # 后续一律用 ST@A 的 P^A(审查修复 F4)
    P = pd.concat([chain_changes(memA_st, reviews, "PA"), chain_changes(memC, reviews, "PC")], ignore_index=True)
    P.to_csv(LEDGER / "inc_t2_predicted_changes.csv", index=False)

    # ---- 准确率 ----
    O = pd.read_csv(LEDGER / "inc_t2_official_changes.csv", dtype={"index": str})
    O["index"] = O["index"].str.zfill(6)
    cal = pd.read_csv(LEDGER / "inc_t2_review_calendar.csv")
    win = dict(zip(cal["review"], cal["window"]))
    acc = []
    for src in ("PA", "PC"):
        for rv in reviews[1:]:
            for idx in IDX:
                for d in ("in", "out"):
                    pset = set(P.query("source == @src and review == @rv and index == @idx and direction == @d")["code"])
                    o = O.query("review == @rv and index == @idx and direction == @d")
                    has_o = len(o) > 0
                    oset = set(o["code"])
                    hit = len(pset & oset)
                    acc.append(dict(source=src, review=rv, window=win.get(rv), index=idx, direction=d,
                                    n_pred=len(pset), n_official=len(oset) if has_o else np.nan,
                                    n_hit=hit if has_o else np.nan,
                                    recall=hit / len(oset) if has_o and len(oset) else np.nan,
                                    precision=hit / len(pset) if has_o and len(pset) else np.nan,
                                    note="" if has_o else "官方名单缺(Gate 0)"))
    A = pd.DataFrame(acc)
    A.to_csv(LEDGER / "inc_t2_prediction_accuracy.csv", index=False)
    print(A.pivot_table(index=["review"], columns=["source", "index", "direction"], values="recall").round(2).to_string())
    print(A.pivot_table(index=["review"], columns=["source", "index", "direction"], values="precision").round(2).to_string())

    # ---- 覆盖(面板内)----
    cov = []
    cs = set(codes)
    for (rv, idx, d), g in O.groupby(["review", "index", "direction"]):
        miss = sorted(set(g["code"]) - cs)
        cov.append(dict(review=rv, window=win.get(rv), index=idx, direction=d, n_official=len(g),
                        n_in_panel=len(g) - len(miss), not_in_panel=";".join(miss)))
    Cv = pd.DataFrame(cov)
    Cv.to_csv(LEDGER / "inc_t2_official_coverage.csv", index=False)
    print(Cv.drop(columns=["not_in_panel"]).pivot_table(index="review", columns=["index", "direction"],
                                                        values=["n_official", "n_in_panel"]).to_string())
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
