"""T5 C 轨(2007-2016,指示性;预注册 §3.2、§10.3)。

东方财富基本面(利润表 / 资产负债表 / 业绩报表)只有幸存者、数值为最新重述版、公告日期不可用:
  可见日 = 法定截止日 + 1 个交易日(一季报 / 年报 4-30,中报 8-31,三季报 10-31);
  池 = 长历史 q92 池 ∩ EM 覆盖股票(静态:任一报告期出现在 EM 业绩报表中的幸存股)——这本身带幸存者偏差(已声明)。
VALUE = 0.5 rank(BP) + 0.5 rank(EP):BP = 每股净资产 x f_e / close_px,EP = TTM 每股收益 x f_e / close_px(EM 业绩报表的每股值);
QUAL = mean[rank(ROE_ttm), rank(-sigma(ROE_q, 8 期, >=6)), rank(-Lev)](ROE = 归母净利润 / 股东权益合计;金融股 Lev 分量 0.5);
DIV = 沿用 lh_DY365(EM 分红对幸存股本就完整)。

阶段: scores_c  runs_c  report_c
用法: python3 scripts/inc_t5_c.py scores_c runs_c report_c
"""
from __future__ import annotations

import pickle
import sys
import time

from inc_t5_common import *      # noqa: F401,F403
from inc_t5_common import (ROOT, SC, RUNS, TMP, LEDGER, FETCH, WIN, LE, LHCtx, log, log_dev, qa_row, merge_csv, slim, TD)

import numpy as np
import pandas as pd

from afactor import inc_t5_signals as SG
from afactor import inc_t5_stats as S5
from afactor import fundamental as FU

FIN_KW = ("银行", "保险", "证券", "多元金融", "金融")


def deadline(end: pd.Timestamp) -> pd.Timestamp:
    m = end.month
    if m == 3:
        return pd.Timestamp(end.year, 4, 30)
    if m == 6:
        return pd.Timestamp(end.year, 8, 31)
    if m == 9:
        return pd.Timestamp(end.year, 10, 31)
    return pd.Timestamp(end.year + 1, 4, 30)


def load_em_fin() -> tuple[pd.DataFrame, dict]:
    d = FETCH / "em_fin"
    parts = []
    st = {"files": 0}
    for f in sorted(d.glob("yjbb_*.parquet")):
        per = f.stem.split("_")[1]
        y = pd.read_parquet(f)
        z_f = d / f"zcfz_{per}.parquet"
        if not len(y) or not z_f.exists():
            continue
        z = pd.read_parquet(z_f)
        y2 = pd.DataFrame({"code6": y["股票代码"].astype(str).str.zfill(6),
                           "eps_ytd": pd.to_numeric(y["每股收益"], errors="coerce"),
                           "ni_ytd": pd.to_numeric(y["净利润-净利润"], errors="coerce"),
                           "bps": pd.to_numeric(y["每股净资产"], errors="coerce"),
                           "industry": y["所处行业"].astype(str)})
        z2 = pd.DataFrame({"code6": z["股票代码"].astype(str).str.zfill(6),
                           "equity": pd.to_numeric(z["股东权益合计"], errors="coerce"),
                           "assets": pd.to_numeric(z["资产-总资产"], errors="coerce"),
                           "liab": pd.to_numeric(z["负债-总负债"], errors="coerce")})
        m = y2.drop_duplicates("code6").merge(z2.drop_duplicates("code6"), on="code6", how="outer")
        m["end_date"] = pd.Timestamp(per)
        parts.append(m)
        st["files"] += 1
    df = pd.concat(parts, ignore_index=True)
    return df, st


def stage_scores_c():
    t0 = time.time()
    X = LE.load_inputs()
    dates, codes = X.dates, X.codes
    pool = np.asarray(X.arr("pool_q92"), bool)
    from afactor.lh_panel import LongPanel
    p = LongPanel()
    close = p.raw("close_px").to_numpy(np.float64)
    raw = p.raw("close_raw").to_numpy(np.float64)
    amt = p.raw("amount").to_numpy(np.float64)
    valid = p.raw("trade").to_numpy(bool) & (amt > 0) & np.isfinite(close)
    f_raw = np.where(np.isfinite(close) & np.isfinite(raw) & (raw > 0), close / raw, np.nan)
    em, st = load_em_fin()
    c6 = pd.Series(codes, index=[c[:6] for c in codes])
    c6 = c6[~c6.index.duplicated(keep=False)]
    em["stock_code"] = em["code6"].map(c6.to_dict())
    n0 = len(em)
    em = em[em["stock_code"].notna()].copy()
    # 覆盖股票(静态):任一报告期有每股净资产
    covered = set(em.loc[em["bps"].notna(), "stock_code"])
    cov_mask = codes.isin(covered)
    log(f"[scores_c] EM 行 {n0} 映射后 {len(em)};覆盖股票 {int(cov_mask.sum())}/{len(codes)}")
    np.save(SC / "c_cov_mask.npy", cov_mask)
    # 季度网格上的单季值 / TTM
    em["qi"] = em["end_date"].dt.year * 4 + (em["end_date"].dt.quarter - 1)
    em["quarter"] = em["end_date"].dt.quarter
    em = em.sort_values(["stock_code", "qi"])
    out = []
    for code, g in em.groupby("stock_code", sort=False):
        g = g.drop_duplicates("qi").set_index("qi")
        full = np.arange(int(g.index.min()), int(g.index.max()) + 1)
        g = g.reindex(full)
        q = (full % 4) + 1
        for c in ("eps", "ni"):
            ytd = g[f"{c}_ytd"]
            sq = np.where(q == 1, ytd, ytd - ytd.shift(1))
            g[f"{c}_sq"] = sq
            g[f"{c}_ttm"] = pd.Series(sq, index=full).rolling(4, min_periods=4).sum()
        E = g["equity"]
        g["roe_ttm"] = g["ni_ttm"] / E.where(E > 0)
        g["roe_q"] = g["ni_sq"] / E.where(E > 0)
        g["roe_sig_ttm"] = g["roe_q"].rolling(8, min_periods=6).std()
        g["lev"] = g["liab"] / g["assets"].where(g["assets"] > 0)
        g["isfin"] = g["industry"].fillna("").astype(str).apply(lambda s: any(k in s for k in FIN_KW))
        g["stock_code"] = code
        g["qi"] = full
        out.append(g.reset_index(drop=True))
    fe = pd.concat(out, ignore_index=True)
    fe["end_date"] = pd.to_datetime((fe["qi"] // 4).astype(int).astype(str) + "Q" + ((fe["qi"] % 4) + 1).astype(int).astype(str)) \
        + pd.offsets.QuarterEnd(0)
    fe = fe[fe["bps"].notna() | fe["eps_ytd"].notna() | fe["equity"].notna()]
    fe["avail"] = fe["end_date"].map(deadline) + pd.Timedelta(days=1)       # 法定截止日 + 1 天 -> 其后第一个交易日可见
    fe["avail_ttm"] = fe["avail"]
    fe["cj"] = codes.get_indexer(fe["stock_code"])
    rows = SG.asof_row(dates, fe["end_date"], "right")
    fff = pd.DataFrame(f_raw).ffill().to_numpy()
    first = SG.first_valid_f(f_raw)
    fe["f_e"] = SG.f_at(fff, rows, fe["cj"].to_numpy(), first)
    fe["bps_h"] = fe["bps"].where(fe["bps"] > 0) * fe["f_e"]
    fe["epsh_ttm"] = fe["eps_ttm"] * fe["f_e"]
    rr = fe[["stock_code", "end_date", "avail", "avail_ttm", "bps_h", "epsh_ttm", "roe_ttm", "roe_sig_ttm", "lev"]]
    d = lambda c: FU._pit_daily(rr, c, dates, codes).to_numpy(np.float64)
    bps, epsh, roe, sgm, lev = d("bps_h"), d("epsh_ttm"), d("roe_ttm"), d("roe_sig_ttm"), d("lev")
    isfin_s = fe.dropna(subset=["isfin"]).drop_duplicates("stock_code", keep="last").set_index("stock_code")["isfin"]
    isfin = isfin_s.reindex(codes).fillna(False).to_numpy(bool)
    with np.errstate(all="ignore"):
        BP = np.where(valid, bps / close, np.nan)
        # v2(预注册公式 EP = ROE_ttm x BP;v1 用 EM 每股收益 TTM,但 2005 年 Q1/Q3 的每股收益缺失 98%,2007 年 EP 覆盖仅 63%)
        EP = np.where(valid, roe * BP, np.nan)
        EP_v1 = np.where(valid, epsh / close, np.nan)
    poolC = pool & cov_mask[None, :]
    save = lambda nm, a: np.save(SC / f"c_{nm}.npy", np.asarray(a, np.float32))
    save("BP", BP); save("EP", EP); save("EP_v1", EP_v1)
    save("VALUE", 0.5 * SG.cs_rank_pool(BP, poolC) + 0.5 * SG.cs_rank_pool(EP, poolC))
    r_roe = SG.cs_rank_pool(np.where(valid, roe, np.nan), poolC)
    r_stab = SG.cs_rank_pool(np.where(valid, -sgm, np.nan), poolC)
    r_lev = SG.cs_rank_pool(np.where(valid & ~isfin[None, :], -lev, np.nan), poolC)
    r_lev = np.where(isfin[None, :] & poolC & valid, np.float32(0.5), r_lev)
    save("QUAL", (r_roe + r_stab + r_lev) / 3.0)
    # D8 与覆盖报告
    ff = sorted((FETCH / "em_fin").glob("*.parquet"))
    nexp = 3 * 4 * 11 + 3 * 4
    ratio = poolC.sum() / max(pool.sum(), 1)
    cover_rows = []
    for y in range(2007, 2017):
        ix = np.flatnonzero(dates.year == y)
        pm = poolC[ix]
        den = pm.sum()
        for nm in ("BP", "EP", "QUAL"):
            a = np.load(SC / f"c_{nm}.npy", mmap_mode="r")[ix]
            cover_rows.append(qa_row("D8", f"C_{nm}@{y}", float((np.isfinite(np.asarray(a)) & pm).sum() / den), "report", True,
                                     "C 池内非空比例"))
    cover_rows.append(qa_row("D8", "em_fin_files", len(ff), f"expected {nexp}", len(ff) >= 0.9 * nexp, "抓取失败率 <= 10%"))
    cover_rows.append(qa_row("D8", "C_pool_vs_q92_stockdays", float(ratio), "report", True, "C 池股票-日 / q92 池股票-日"))
    pd.DataFrame(cover_rows).to_csv(LEDGER / "inc_t5_c_qa.csv", index=False)
    log(f"[scores_c] 完成 {time.time() - t0:.0f}s;C 池占 q92 池 {ratio:.3f}")


def _run(ctx: LHCtx, W, cost="Y1", record=False):
    return slim(ctx.run(W, cost=cost, record=record))


def stage_runs_c():
    t0 = time.time()
    ctx = LHCtx("2006-01-04")
    cov = np.load(SC / "c_cov_mask.npy")
    U = ctx.U & cov[None, :]
    d = RUNS / "c"
    d.mkdir(parents=True, exist_ok=True)
    save = lambda name, r: pickle.dump(r, open(d / (name.replace("|", "__") + ".pkl"), "wb"))
    sc = lambda nm: pd.DataFrame(np.load(SC / f"{nm}.npy", mmap_mode="r")[ctx.i0:], index=ctx.dates, columns=ctx.codes)
    B = {"LOWVOL120": sc("lh_LOWVOL120"), "DY365": sc("lh_DY365"), "VALUE": sc("c_VALUE"), "QUAL": sc("c_QUAL")}
    S1 = pd.DataFrame(np.asarray(ctx.X.arr("score_S1"))[ctx.i0:], index=ctx.dates, columns=ctx.codes)
    W = {}
    W["LIQ"] = ctx.weights_U(S1, (0.20, 0.35), U)
    W["LOWVOL"] = ctx.weights_U(B["LOWVOL120"], (0.45, 0.60), U)
    W["VALUE"] = ctx.weights_U(B["VALUE"], (0.45, 0.60), U)
    W["DIV"] = ctx.weights_U(B["DY365"], (0.45, 0.60), U)
    W["QUAL"] = ctx.weights_U(B["QUAL"], (0.45, 0.60), U)
    for k, w in W.items():
        n = (w > 0).sum(axis=1)
        log(f"[runs_c] W {k} 平均持仓 {n[n > 0].mean():.0f} 空仓日 {int((n == 0).sum())}")
    b1 = LE.bench(ctx.ad, U, 5)
    b1.to_pickle(RUNS / "c_b1.pkl")
    for k, dep in (("LIQ", "20_35"), ("LOWVOL", "45_60"), ("VALUE", "45_60"), ("DIV", "45_60"), ("QUAL", "45_60")):
        save(f"S|{k}|{dep}", _run(ctx, W[k]))
        log(f"[runs_c] S|{k}|{dep} {time.time() - t0:.0f}s")
    mem5 = ["LIQ", "LOWVOL", "VALUE", "DIV", "QUAL"]
    Ws = [W[m] for m in mem5]
    save("C|EW5|45_60", _run(ctx, S5.combine_weights(Ws, np.full(5, 0.2)), record=True))
    save("C|CORE5|45_60", _run(ctx, S5.combine_weights(Ws, S5.core_weights(5, 0, 0.5))))
    save("C|S4|45_60", _run(ctx, S5.combine_weights(Ws[1:], np.full(4, 0.25))))
    Ua = U.to_numpy(bool)
    n = Ua.sum(1, keepdims=True)
    Wf = np.divide(Ua.astype(np.float32), n, out=np.zeros(Ua.shape, np.float32), where=n > 0)
    save("FP", _run(ctx, Wf, record=True))
    log(f"[runs_c] 完成 {time.time() - t0:.0f}s")


def stage_report_c():
    from inc_t5_report import TrackData, stat_row, ir_of, gen_tests
    td = TrackData("c")
    a, b = WIN["c"]["full"]
    rows = []
    for n in ("S|LIQ|20_35", "S|LOWVOL|45_60", "S|VALUE|45_60", "S|DIV|45_60", "S|QUAL|45_60", "C|EW5|45_60", "C|CORE5|45_60", "C|S4|45_60"):
        rows.append(stat_row(td, n, "full"))
    # 不分窗,另按 2007-2011 / 2012-2016 报告 IR(描述性)
    sub = []
    for n in ("S|LIQ|20_35", "C|EW5|45_60", "C|CORE5|45_60", "C|S4|45_60"):
        for lab, (x, y) in (("2007-2011", ("2007-01-04", "2011-12-31")), ("2012-2016", ("2012-01-01", "2016-12-30"))):
            rel = LE.rel_block(LE.slice_eval(td.r(n), x, y), LE.slice_eval(td.b1, x, y))
            sub.append(dict(config=n, period=lab, excess_arith=rel["excess_arith"], te=rel["te"], ir=rel["ir"], nw_t5=rel["nw_t5"]))
    pd.DataFrame(sub).to_csv(LEDGER / "inc_t5_c_halves.csv", index=False)
    gen = gen_tests(td, {"S4": "C|S4|45_60", "LOWVOL": "S|LOWVOL|45_60", "VALUE": "S|VALUE|45_60", "DIV": "S|DIV|45_60",
                         "QUAL": "S|QUAL|45_60"}, "S|LIQ|20_35")
    gen = gen[gen.window == "full"]
    gen.to_csv(LEDGER / "inc_t5_c_gen.csv", index=False)
    fp = td.run("FP")
    s, bb = LE.slice_eval(fp["rets"], a, b), LE.slice_eval(td.b1, a, b)
    ex = LE.ann_geo(s) - LE.ann_geo(bb)
    cost = float(LE.slice_eval(fp["rec_daily"]["cost_total"], a, b).mean() * TD)
    dIR = ir_of(td, "C|EW5|45_60", "full") - ir_of(td, "S|LIQ|20_35", "full")
    g4 = gen[(gen.y == "S4") & (gen.freq == "daily") & (gen.direction == "y_on_x")].iloc[0]
    res = pd.DataFrame(rows)
    res["note"] = ""
    res.to_csv(LEDGER / "inc_t5_c_summary.csv", index=False)
    down = bool(dIR <= 0 or g4["a_ann"] <= 0)
    pd.DataFrame([dict(item="C_dIR_EW5_vs_LIQ", value=dIR), dict(item="C_S4_intercept_ann", value=g4["a_ann"]),
                  dict(item="C_S4_intercept_t_q5", value=g4["t_q5"]), dict(item="C_fullpool_excess_plus_cost", value=ex + cost),
                  dict(item="C_downgrade_flag", value=int(down))]).to_csv(LEDGER / "inc_t5_c_verdict.csv", index=False)
    log(f"[report_c] ΔIR {dIR:+.3f} S4 截距 {g4['a_ann'] * 100:+.2f}pp t {g4['t_q5']:.2f} 降级={down}")


if __name__ == "__main__":
    for c in sys.argv[1:]:
        globals()[f"stage_{c}"]()
