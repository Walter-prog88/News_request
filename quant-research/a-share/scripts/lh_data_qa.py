"""lh_data 第 4 步:长历史日线面板验收。只做数据一致性统计,**不计算任何策略收益、IC 或回测**。

输出
  research/ledger/lh_data_qa.csv            主表(section, metric, year, value, note)
  research/ledger/lh_data_qa_gates.csv      预注册 §2.3 Q1-Q9 门槛逐条(值、门槛、是否通过、口径说明)
  research/ledger/lh_data_qa_twosource.csv  新浪 vs baostock 逐股一致性(重叠股票)
  research/ledger/lh_data_qa_delisted.csv   逐只退市股(最后交易日、退市整理期、最后价格等)
  research/ledger/lh_data_qa_exec.csv       日开盘价 vs Panel 开盘 10 分钟 TWAP 的误差分位数
  research/ledger/lh_data_qa_q5.csv         越带(超出当日涨跌幅上限)且无法归入已知特殊日的股票日
  research/ledger/lh_data_qa_missing.csv    Q1/Q2 缺失清单
报告 research/LH_DATA_QA.md 引用这些数字。

data_8:21 的外部表按项目纪律"先截断再读"(<= 2025-12-31)。
用法: python3 scripts/lh_data_qa.py
"""
from __future__ import annotations

import gzip
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from afactor.config import LEDGER, MIN_ADV20, RESEARCH_END          # noqa: E402
from afactor.lh_panel import LongPanel                             # noqa: E402

OUT = ROOT / "cache" / "long_daily"
PARTS, PARTS_SINA, PARTS_ST = OUT / "parts", OUT / "parts_sina", OUT / "parts_st"
ROWS: list[tuple] = []
GATES: list[dict] = []
END8 = RESEARCH_END.replace("-", "")
A_PREFIX = ("sh.60", "sh.68", "sz.00", "sz.30")


def add(section, metric, value, year=None, note=""):
    v = float(value) if value is not None and not isinstance(value, str) else value
    ROWS.append((section, metric, "" if year is None else int(year), v, note))
    print(f"[{section}] {metric}{'' if year is None else ' ' + str(year)}: {value} {note}", flush=True)


def gate(q, check, value, threshold, passed, note=""):
    GATES.append(dict(q=q, check=check, value=value, threshold=threshold, passed=bool(passed), note=note))
    print(f"[GATE {q}] {check}: {value} 门槛 {threshold} -> {'通过' if passed else '不通过'} {note}", flush=True)


def q(s: pd.Series, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    s = s[np.isfinite(s)]
    return {f"p{int(round(x*100)):02d}": float(s.quantile(x)) for x in qs}


def stack_valid(df: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.Series:
    a = df.to_numpy(np.float64)
    m = np.isfinite(a) if mask is None else (np.isfinite(a) & mask.to_numpy(bool))
    return pd.Series(a[m])


def read_gz(f: Path) -> pd.DataFrame:
    with gzip.open(f, "rt") as fh:
        return pd.read_csv(fh)


def to_bs(ts: str) -> str:
    n, ex = ts.split(".")
    return f"{ex.lower()}.{n}"


# =============================================================== 两源一致性
def two_source(uni: pd.DataFrame) -> pd.DataFrame:
    """两源都有的股票(现存股中 baostock 已抓的 + 全部退市股):baostock 全字段 vs 新浪,同股同日比较。

    逐股记录:行集合差异、后复权日收益差(按"复牌/除权跨停牌"与"普通日"分开)、不复权收盘差、量额差、换手率差。"""
    stt = dict(zip(uni["code"], uni["status"]))
    ov = sorted(c for c in uni["code"] if (PARTS / f"{c}.csv.gz").exists() and (PARTS_SINA / f"{c}.csv.gz").exists())
    recs, yr_rows, ex_rows, big_rows, drift_rows = [], [], [], [], []
    for c in ov:
        b = read_gz(PARTS / f"{c}.csv.gz")
        s = read_gz(PARTS_SINA / f"{c}.csv.gz")
        k6 = ["open", "high", "low", "close", "volume", "amount"]
        s = s[~(s[k6].to_numpy() == s[k6].shift(1).to_numpy()).all(axis=1)]          # 与构建相同:去掉残行
        bdates = b["date"].to_numpy()
        b = b[(b["tradestatus"] == 1) & (b["volume"] > 0)].copy()
        m = b.merge(s, on="date", how="outer", suffixes=("_b", "_s"), indicator=True)
        only_b = int((m["_merge"] == "left_only").sum())
        only_s_rows = m[m["_merge"] == "right_only"]
        only_s_in_bs_range = int(only_s_rows["date"].isin(bdates).sum())
        m = m[m["_merge"] == "both"].sort_values("date").reset_index(drop=True)
        if len(m) < 2:
            continue
        cb = m["close_b"].to_numpy(); chs = (m["close_s"] * m["hfq_factor"]).to_numpy()
        rb = cb[1:] / cb[:-1] - 1; rs = chs[1:] / chs[:-1] - 1
        dret = np.abs(rb - rs) * 1e4
        # 跨停牌(两行之间 baostock 有停牌行)的日子:股改对价等资本事件多发生在复牌日
        bpos = {d: i for i, d in enumerate(bdates)}
        ib = np.array([bpos[d] for d in m["date"]])
        gap = (ib[1:] - ib[:-1]) > 1
        fchg = np.abs(m["hfq_factor"].to_numpy()[1:] / m["hfq_factor"].to_numpy()[:-1] - 1) > 1e-9
        draw = np.abs(m["close_raw"] - m["close_s"]).to_numpy()
        rawd = (draw[1:] > 0.005) | (draw[:-1] > 0.005)
        rv = (m["volume_s"] / m["volume_b"]).to_numpy(); ra = (m["amount_s"] / m["amount_b"]).to_numpy()
        tb = m["turn"].to_numpy(np.float64); ts_ = (m["turnover"] * 100).to_numpy(np.float64)
        okt = (tb > 0) & (ts_ > 0)
        rt = ts_[okt] / tb[okt]
        big = dret > 10
        dlog = np.log1p(rs) - np.log1p(rb)                       # 新浪 - baostock 的对数收益差
        sr = gap & fchg                                          # 跨停牌且因子变化(股改对价等)
        exo = fchg & ~gap                                        # 普通除权日
        yy1 = m["date"].str[:4].to_numpy()[1:]
        for y in np.unique(yy1):
            kk = yy1 == y
            drift_rows.append((int(y), c, stt.get(c), int(kk.sum()), float(dlog[kk & sr].sum()),
                               float(dlog[kk & exo].sum()), float(dlog[kk & ~sr & ~exo].sum())))
        recs.append(dict(code=c, status=stt.get(c), n_bs_trade=len(b), n_sina=len(s), n_common=len(m),
                         only_bs=only_b, only_sina=len(only_s_rows), only_sina_in_bs_rows=only_s_in_bs_range,
                         ret_diff_med_bp=float(np.median(dret)), ret_diff_max_bp=float(np.max(dret)),
                         ret_diff_gt1bp=int((dret > 1).sum()), ret_diff_gt10bp=int(big.sum()),
                         gt10_resume_factor=int((big & gap & fchg).sum()),
                         gt10_rawclose=int((big & rawd & ~(gap & fchg)).sum()),
                         gt10_other=int((big & ~rawd & ~(gap & fchg)).sum()),
                         raw_close_diff_gt_half_tick=int((draw > 0.005).sum()),
                         vol_mismatch=int((np.abs(rv - 1) > 1e-3).sum()), amt_mismatch=int((np.abs(ra - 1) > 1e-3).sum()),
                         turn_ratio_med=float(np.median(rt)) if len(rt) else np.nan,
                         turn_off5=int((np.abs(rt - 1) > 0.05).sum()), turn_n=int(okt.sum())))
        for k in np.flatnonzero(big):
            big_rows.append(dict(code=c, date=m["date"].iat[k + 1], prev_date=m["date"].iat[k], ret_bs=rb[k], ret_sina=rs[k],
                                 diff_bp=dret[k], across_suspension=bool(gap[k]), factor_change=bool(fchg[k]),
                                 raw_close_diff=bool(rawd[k]),
                                 raw_bs=m["close_raw"].iat[k + 1], raw_sina=m["close_s"].iat[k + 1]))
        yy = m["date"].str[:4].to_numpy()
        for y in np.unique(yy):
            kk = (yy == y) & okt
            if kk.any():
                r = ts_[kk] / tb[kk]
                yr_rows.append((int(y), c, len(r), float(np.median(np.abs(r - 1))), int((np.abs(r - 1) > 0.05).sum())))
        ex = np.flatnonzero(fchg) + 1
        for e in ex:
            for k in (e - 1, e, e + 1):
                if 1 <= k < len(m):
                    ex_rows.append((c, m["date"].iat[e], k - e, float(dret[k - 1]), bool(gap[e - 1])))
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df.to_csv(LEDGER / "lh_data_qa_twosource.csv", index=False)
    pd.DataFrame(big_rows).to_csv(LEDGER / "lh_data_qa_twosource_bigdiff.csv", index=False)
    sec = "3b 两源一致"
    for lab, g in (("现存股", df[df["status"] == "1"]), ("退市股", df[df["status"] == "0"])):
        if g.empty:
            continue
        add(sec, f"[{lab}] 重叠股票数", len(g))
        add(sec, f"[{lab}] 共同交易日(股票-日)", int(g["n_common"].sum()))
        add(sec, f"[{lab}] baostock 有成交而新浪无行 股票-日", int(g["only_bs"].sum()))
        add(sec, f"[{lab}] 新浪有行而 baostock 无成交 股票-日", int(g["only_sina"].sum()),
            note=f"其中 baostock 当日有停牌行 {int(g['only_sina_in_bs_rows'].sum())}")
        add(sec, f"[{lab}] 后复权日收益 |差| 中位(bp,逐股中位的中位)", float(g["ret_diff_med_bp"].median()))
        n_ret = int(g["n_common"].sum() - len(g))
        add(sec, f"[{lab}] 后复权日收益 |差|>1bp 股票-日占比", float(g["ret_diff_gt1bp"].sum() / max(n_ret, 1)))
        add(sec, f"[{lab}] 后复权日收益 |差|>10bp 股票-日", int(g["ret_diff_gt10bp"].sum()))
        add(sec, f"[{lab}]   其中 跨停牌且新浪因子变化(股改对价等,baostock 漏复权)", int(g["gt10_resume_factor"].sum()),
            note=f"涉及 {int((g['gt10_resume_factor'] > 0).sum())} 只")
        add(sec, f"[{lab}]   其中 两源不复权收盘不同", int(g["gt10_rawclose"].sum()), note=f"涉及 {int((g['gt10_rawclose'] > 0).sum())} 只")
        add(sec, f"[{lab}]   其中 其他", int(g["gt10_other"].sum()), note=f"涉及 {int((g['gt10_other'] > 0).sum())} 只")
        add(sec, f"[{lab}] 不复权收盘 |差|>半个价位 股票-日", int(g["raw_close_diff_gt_half_tick"].sum()))
        add(sec, f"[{lab}] 成交量 |比值-1|>0.1% 股票-日", int(g["vol_mismatch"].sum()))
        add(sec, f"[{lab}] 成交额 |比值-1|>0.1% 股票-日", int(g["amt_mismatch"].sum()))
    cur = df[df["status"] == "1"]
    if len(cur):
        add(sec, "[现存股] 换手率 新浪x100/baostock turn 比值 中位(逐股中位的中位)", float(cur["turn_ratio_med"].median()))
        add(sec, "[现存股] 换手率 |比值-1|>5% 占比(股票-日)", float(cur["turn_off5"].sum() / max(cur["turn_n"].sum(), 1)))
        add(sec, "[现存股] 换手率 |比值-1|>5% 的日子超过 10% 的股票数", int((cur["turn_off5"] / cur["turn_n"].clip(lower=1) > 0.10).sum()))
        yr = pd.DataFrame(yr_rows, columns=["year", "code", "n", "med_absdev", "off5"])
        for y, g in yr.groupby("year"):
            add(sec, "[现存股] 换手率 |比值-1|>5% 占比", float(g["off5"].sum() / g["n"].sum()), y)
    exd = pd.DataFrame(ex_rows, columns=["code", "exdate", "k", "diff_bp", "across_susp"])
    if len(exd):
        e0 = exd[exd["k"] == 0]
        add(sec, "除权日数(新浪因子变化,重叠股票)", len(e0))
        add(sec, "除权日 收益|差| 中位(bp)", float(e0["diff_bp"].median()))
        add(sec, "除权日 收益|差|>1bp 占比", float((e0["diff_bp"] > 1).mean()))
        add(sec, "除权日 收益|差|>1bp 占比(不跨停牌的普通除权日)", float((e0.loc[~e0["across_susp"], "diff_bp"] > 1).mean()))
        add(sec, "除权日 收益|差|>1bp 占比(跨停牌复牌即除权,股改多属此类)", float((e0.loc[e0["across_susp"], "diff_bp"] > 1).mean()))
        # 预注册 Q6:随机抽 20 只有除权的股票,比较除权日前后(-1,0,+1)的日收益
        rng = np.random.default_rng(20260310)
        pick = rng.choice(sorted(exd["code"].unique()), size=min(20, exd["code"].nunique()), replace=False)
        e20 = exd[exd["code"].isin(pick)]
        e20n = e20[~e20["across_susp"]]
        gate("Q6", "后复权连续性:随机 20 只有除权的股票,除权日 -1/0/+1 日收益两源差(bp)",
             f"全部:中位 {e20['diff_bp'].median():.2e},最大 {e20['diff_bp'].max():.0f}(n={len(e20)});"
             f"不跨停牌的除权:中位 {e20n['diff_bp'].median():.2e},最大 {e20n['diff_bp'].max():.2e}(n={len(e20n)})",
             "中位 < 1bp,最大 < 10bp", (e20["diff_bp"].median() < 1) and (e20["diff_bp"].max() < 10),
             "第二来源 = baostock(东财 push2his 在本机被代理拒绝)。大差来自两类:跨停牌复牌即除权(股改送股对价,"
             "baostock 漏复权,新浪对)与少数普通除权日因子不一致(2020+ 用 tushare 仲裁见 §6),见报告 §2")
    # 累计漂移:新浪 - baostock 的对数收益差,按"股改对价类 / 普通除权日 / 其他日"分解,换算成每股票-年(bp)
    dr = pd.DataFrame(drift_rows, columns=["year", "code", "status", "n", "d_reform", "d_exdiv", "d_other"])
    dr.to_csv(LEDGER / "lh_data_qa_twosource_drift.csv", index=False)
    for lab, cols in (("股改对价类(跨停牌因子变化)", "d_reform"), ("普通除权日", "d_exdiv"), ("其他日", "d_other")):
        tot = dr[cols].sum(); n_sy = dr["n"].sum() / 242.0
        add(sec, f"累计对数收益差 新浪-baostock [{lab}] 每股票-年(bp)", float(tot / max(n_sy, 1e-9) * 1e4))
    for y, g in dr.groupby("year"):
        n_sy = g["n"].sum() / 242.0
        add(sec, "新浪-baostock 每股票-年对数收益差(bp)[股改对价类 / 普通除权 / 其他]",
            f"{g['d_reform'].sum() / n_sy * 1e4:.1f} / {g['d_exdiv'].sum() / n_sy * 1e4:.2f} / {g['d_other'].sum() / n_sy * 1e4:.2f}", y)
    flag = df[(df["gt10_resume_factor"] + df["gt10_rawclose"] + df["gt10_other"] > 0) | (df["only_bs"] > 0)
              | (df["amt_mismatch"] > 0) | (df["turn_off5"] / df["turn_n"].clip(lower=1) > 0.10)]
    add(sec, "超阈值股票数(收益差>10bp / baostock 有成交新浪无行 / 成交额不一致 / 换手率>5%偏差的日子>10%)", len(flag))
    add(sec, "  其中只因股改对价类(跨停牌因子变化)", int(((flag["gt10_resume_factor"] > 0) & (flag["gt10_rawclose"] + flag["gt10_other"] == 0)
                                                 & (flag["only_bs"] == 0) & (flag["amt_mismatch"] == 0)).sum()))
    return df


def bs_raw_checks(cal: pd.DatetimeIndex, st_info: pd.DataFrame | None = None) -> dict:
    """baostock 原始逐股文件(parts/,含退市股与现存股抽样)的行数完整性(Q3)与停牌行成交量(Q8);
    给 st_info 时顺带比较 baostock isST 与项目 st_info_latest(2016-08-09 起,交易日)。"""
    pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(cal)}
    n_rows = n_missing = n_ts0 = n_ts0_vol = n_off = 0
    st_cmp = dict(n=0, agree=0, bs_only=0, info_only=0, n_st_bs=0)
    if st_info is not None and len(st_info):
        i0 = st_info.index.min()
        st_r = st_info.reindex(cal[cal >= i0], method="ffill").fillna(False).astype(bool)
        st_r.index = st_r.index.strftime("%Y-%m-%d")
    miss_codes = []
    pc_diffs = []
    for f in sorted(PARTS.glob("*.csv.gz")):
        b = read_gz(f)
        if st_info is not None and len(st_info):
            code = f.name[:-7]
            bb = b[(b["tradestatus"] == 1) & b["date"].isin(st_r.index)]
            if len(bb):
                a_bs = bb["isST"].to_numpy() == 1
                a_in = (st_r.loc[bb["date"], code].to_numpy(bool) if code in st_r.columns
                        else np.zeros(len(bb), bool))
                st_cmp["n"] += len(bb); st_cmp["agree"] += int((a_bs == a_in).sum())
                st_cmp["bs_only"] += int((a_bs & ~a_in).sum()); st_cmp["info_only"] += int((~a_bs & a_in).sum())
                st_cmp["n_st_bs"] += int(a_bs.sum())
        pp = np.array([pos.get(d, -1) for d in b["date"]])
        n_off += int((pp < 0).sum())
        pp = pp[pp >= 0]
        if len(pp) == 0:
            continue
        span = int(pp.max() - pp.min() + 1)
        miss = span - len(np.unique(pp))
        n_rows += len(b); n_missing += miss
        if miss > 0:
            miss_codes.append(f.name[:-7])
        ts0 = b["tradestatus"] != 1
        n_ts0 += int(ts0.sum()); n_ts0_vol += int((ts0 & (b["volume"].fillna(0) > 0)).sum())
        t1 = b[(b["tradestatus"] == 1) & (b["preclose"] > 0)]
        dpc = np.abs((t1["close"] / t1["preclose"] - 1) * 100 - t1["pctChg"]).to_numpy()
        pc_diffs.append(dpc[np.isfinite(dpc)])
    return dict(n_files=len(list(PARTS.glob("*.csv.gz"))), n_rows=n_rows, n_missing=n_missing, n_ts0=n_ts0,
                n_ts0_vol=n_ts0_vol, n_off=n_off, miss_codes=miss_codes, st_cmp=st_cmp,
                pc_diff=np.concatenate(pc_diffs) if pc_diffs else np.array([]))


# =============================================================== main
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", default=None)
    args = ap.parse_args()
    t0 = time.time()
    lp = LongPanel(panel_dir=args.panel_dir)
    meta = lp.meta
    info = lp.code_info.copy()
    dates, codes = lp.dates, lp.codes
    yr = pd.Series(dates.year, index=dates)
    years = sorted(set(dates.year))
    uni = pd.read_csv(OUT / "universe.csv", dtype=str, keep_default_na=False)
    add("0 构建", "交易日数", len(dates), note=f"{dates[0].date()}..{dates[-1].date()}")
    add("0 构建", "股票数(有数据)", len(codes))
    add("0 构建", "清单股票数", len(uni))
    add("0 构建", "清单 现存(status=1)", int((uni["status"] == "1").sum()))
    add("0 构建", "清单 已退市(status=0,含 2026 年退市)", int((uni["status"] == "0").sum()))
    for k, v in meta.get("sources", {}).items():
        add("0 构建", f"主源={k} 股票数", v)
    add("0 构建", "未下载完成股票数", meta.get("n_missing", 0), note=",".join(meta.get("missing", [])[:20]))
    add("0 构建", "无数据股票数(.empty)", meta.get("n_empty", 0))
    add("0 构建", "新浪查无此股(.nosina)", meta.get("n_nosina", 0))
    for k, v in info["st_src"].value_counts().items():
        add("0 构建", f"ST 来源={k} 股票数", int(v))
    add("0 构建", "新浪无后复权因子而整只退回 baostock 的股票", meta.get("n_sina_nohfq", 0))
    if "sina_dup_rows" in info.columns:
        add("0 构建", "新浪残行(与上一行 OHLC+量额完全相同)去掉的行数", int(info["sina_dup_rows"].fillna(0).sum()),
            note=str(info.loc[info["sina_dup_rows"].fillna(0) > 0].index[:10].tolist()))

    # ---------------- 1 清单与覆盖 ----------------
    for s, n in uni["source"].value_counts().items():
        add("1 清单", f"清单来源={s}", n)
    for pre in A_PREFIX:
        add("1 清单", f"板块前缀 {pre}", int(uni["code"].str.startswith(pre).sum()))
    sb = pd.read_csv(ROOT / "data_8:21" / "stock_basic.csv", dtype=str, keep_default_na=False)
    sb = sb[sb["list_date"] <= END8].copy()                                      # 先截断
    sb.loc[sb["delist_date"] > END8, "delist_date"] = ""
    sb["bs"] = sb["ts_code"].map(to_bs)
    sb = sb[sb["bs"].str.startswith(A_PREFIX)]
    in_mkt = sb[(sb["delist_date"] == "") | (sb["delist_date"] >= "20050104")]
    listed = lp.raw("listed").astype(bool)
    have_any = listed.any()
    have_bs = {to_bs(c) for c in have_any[have_any].index}
    cov_all = float(in_mkt["bs"].isin(have_bs).mean())
    add("1 清单", "stock_basic 2005-2025 在市过的沪深A股数", len(in_mkt))
    add("1 清单", "其中面板有数据比例", cov_all)
    miss_rows = [dict(kind="Q1", code=r.bs, name=r.name, list_date=r.list_date, delist_date=r.delist_date)
                 for r in in_mkt[~in_mkt["bs"].isin(have_bs)].itertuples()]
    ly = listed.groupby(yr).any()
    cov_y = {}
    for y in years:
        ys, ye = f"{y}0101", f"{y}1231"
        sub = in_mkt[(in_mkt["list_date"] <= ye) & ((in_mkt["delist_date"] == "") | (in_mkt["delist_date"] >= ys))]
        hv = {to_bs(c) for c in ly.columns[ly.loc[y].to_numpy()]}
        cov_y[y] = float(sub["bs"].isin(hv).mean()) if len(sub) else np.nan
        add("1 清单", "stock_basic 当年在市股票 面板当年有行比例", cov_y[y], y, note=f"n={len(sub)}")
    gate("Q1", "覆盖率:stock_basic 2005-2025 在市过的沪深A股在面板中有数据的比例",
         f"全期 {cov_all:.4f};逐年最低 {min(cov_y.values()):.4f}({min(cov_y, key=cov_y.get)})",
         "全期 >= 99%,每年 >= 98%", cov_all >= 0.99 and min(cov_y.values()) >= 0.98,
         "逐年口径:当年在市(上市日 <= 年末且退市日空或 >= 年初)的股票当年在面板有行(含停牌行)")
    dl = sb[(sb["delist_date"] >= "20050104") & (sb["delist_date"] <= END8)]
    q2 = float(dl["bs"].isin(have_bs).mean()) if len(dl) else np.nan
    add("1 清单", "stock_basic 2005-2025 退市沪深A股数", len(dl))
    add("1 清单", "其中在本清单内", int(dl["bs"].isin(uni["code"]).sum()))
    add("1 清单", "其中面板有数据", int(dl["bs"].isin(have_bs).sum()))
    miss_rows += [dict(kind="Q2", code=r.bs, name=r.name, list_date=r.list_date, delist_date=r.delist_date)
                  for r in dl[~dl["bs"].isin(have_bs)].itertuples()]
    gate("Q2", "退市股覆盖:stock_basic delist_date ∈ [2005-01-04, 2025-12-31] 的股票有数据比例",
         f"{q2:.4f}({int(dl['bs'].isin(have_bs).sum())}/{len(dl)})", ">= 95%", q2 >= 0.95)
    pd.DataFrame(miss_rows, columns=["kind", "code", "name", "list_date", "delist_date"]).to_csv(
        LEDGER / "lh_data_qa_missing.csv", index=False)
    m = dl.merge(uni[["code", "outDate"]], left_on="bs", right_on="code", how="left")
    add("1 清单", "baostock outDate 与 stock_basic delist_date 一致比例",
        float((m["outDate"].str.replace("-", "") == m["delist_date"]).mean()))
    bs_del = uni[(uni["outDate"] >= "2005-01-01") & (uni["outDate"] <= RESEARCH_END)]
    add("1 清单", "baostock 2005-2025 退市数", len(bs_del))
    add("1 清单", "baostock 退市但不在 stock_basic 退市名单", int((~bs_del["code"].isin(dl["bs"])).sum()))
    add("1 清单", "清单中 2026 年才退市的股票(2005-2025 在市,baostock 源)",
        int(((uni["outDate"] > RESEARCH_END) & (uni["status"] == "0")).sum()))

    # ---------------- 2 年度统计 ----------------
    trade = lp.raw("trade").astype(bool)
    susp = lp.raw("suspended").astype(bool)
    amount = lp.raw("amount")
    adv20 = lp["adv20"]
    univ = lp["universe"]
    out_y = pd.to_datetime(info["outDate"].fillna("").replace("", np.nan), errors="coerce").dt.year
    ft = pd.to_datetime(info["first_trade"])
    ipo = pd.to_datetime(info["ipoDate"].fillna("").replace("", np.nan), errors="coerce")
    tr_y = trade.groupby(yr).any()
    is_sina = info["src"].isin(["sina", "sina+bs"]).to_numpy()
    tv = trade.to_numpy()
    lv = listed.to_numpy()
    sv = susp.to_numpy()
    stv = lp.raw("is_st").to_numpy(bool)
    for y in years:
        my = (yr == y).to_numpy()
        add("2 年度", "有成交股票数", int(tr_y.loc[y].sum()), y)
        add("2 年度", "其中新浪源", int(tr_y.loc[y].to_numpy()[is_sina].sum()), y)
        add("2 年度", "退市股数(outDate 在该年)", int((out_y == y).sum()), y)
        add("2 年度", "新上市股数(首个交易日在该年且 ipoDate>=2005)", int(((ft.dt.year == y) & (ipo >= "2005-01-04")).sum()), y)
        add("2 年度", "日均上市股数", float(lv[my].sum(1).mean()), y)
        add("2 年度", "日均交易股数", float(tv[my].sum(1).mean()), y)
        add("2 年度", "停牌股票-日占上市股票-日比例", float(sv[my].sum() / max(lv[my].sum(), 1)), y)
        add("2 年度", f"日均入池股数(adv20>={MIN_ADV20/1e4:.0f}万, 上市>=6日)", float(univ.to_numpy()[my].sum(1).mean()), y)
        a = adv20.to_numpy()[my]
        add("2 年度", "adv20 截面中位数(元,年内日均)", float(np.nanmean(np.nanmedian(np.where(tv[my], a, np.nan), axis=1))), y)
        add("2 年度", "全市场日均成交额(亿元)", float(np.nanmean(np.nansum(amount.to_numpy()[my], axis=1)) / 1e8), y)
        add("2 年度", "ST 股票-日占交易股票-日比例", float((stv[my] & tv[my]).sum() / max(tv[my].sum(), 1)), y)
    del adv20, univ

    # ---------------- 3 缺失与异常 ----------------
    sec = "3 缺失异常"
    pos = {d: i for i, d in enumerate(dates)}
    bsm = ~info["src"].isin(["sina", "sina+bs"])
    fr = pd.to_datetime(info["first_row"]).map(pos)
    lr = pd.to_datetime(info["last_row"]).map(pos)
    gap = (lr - fr + 1 - info["n_rows"] + info["off_cal"]).clip(lower=0)
    n_rows_bs = int(info.loc[bsm, "n_rows"].sum())
    miss_bs = int(gap[bsm].sum())
    add(sec, "整只 baostock 源股票:上市区间内缺行数(首行..末行之间的交易日没有行)", miss_bs)
    if "sina_off_cal" in info.columns:
        add(sec, "不在交易日历上的行数(新浪源)", int(info.loc[~bsm, "sina_off_cal"].fillna(0).sum()))
        add(sec, "新浪源:补出的停牌行(上市区间内新浪无行的交易日)", int(info.loc[~bsm, "sina_fill_rows"].fillna(0).sum()))
        lead = info.loc[~bsm, "sina_lead_fill"].fillna(0)
        add(sec, "新浪源:窗口首个交易前补出的停牌行(2005 年前上市、2005-01-04 起停牌)", int(lead.sum()),
            note=f"{int((lead > 0).sum())} 只")
    add(sec, "停牌行收盘 != 最近交易日收盘 的行数", int(info["sus_close_mismatch"].sum()))
    o, h, l, c = (lp.raw(k).to_numpy(np.float64) for k in ("open_px", "high_px", "low_px", "close_px"))
    add(sec, "交易日价格非正或缺失", int((tv & ~((o > 0) & (h > 0) & (l > 0) & (c > 0))).sum()))
    add(sec, "交易日 最高<最低", int((tv & (h < l * (1 - 1e-9))).sum()))
    add(sec, "交易日 收盘越出[最低,最高](相对容差 1e-6)", int((tv & ((c > h * (1 + 1e-6)) | (c < l * (1 - 1e-6)))).sum()))
    add(sec, "交易日 开盘越出[最低,最高](相对容差 1e-6)", int((tv & ((o > h * (1 + 1e-6)) | (o < l * (1 - 1e-6)))).sum()))
    del o, h, l, c
    turn = lp.raw("turn").to_numpy(np.float64)
    tmiss = tv & ~(turn > 0)
    add(sec, "交易日 换手率缺失或为 0", int(tmiss.sum()))
    add(sec, "交易日数(全部股票-日)", int(tv.sum()))
    q4 = {}
    for y in years:
        my = (yr == y).to_numpy()
        q4[y] = float(tmiss[my].sum() / max(tv[my].sum(), 1))
        add(sec, "交易日 换手率缺失或为 0 占比", q4[y], y,
            note=f"新浪源 {int(tmiss[my][:, is_sina].sum())},baostock 源 {int(tmiss[my][:, ~is_sina].sum())}")
    gate("Q4", "turn 缺失:有成交的股票日中 turn 缺失或为 0 的比例(逐年)",
         f"逐年最高 {max(q4.values()):.4%}({max(q4, key=q4.get)})", "每年 < 1%", max(q4.values()) < 0.01)
    del turn, tmiss
    rhold = lp.raw("ret").to_numpy(np.float64)
    nl = lp.raw("no_limit_day").to_numpy(bool)
    add(sec, "|持有收益|>50% 的交易日", int((tv & (np.abs(rhold) > 0.5)).sum()))
    add(sec, "其中为无涨跌幅限制日(新股/复牌等)", int((tv & (np.abs(rhold) > 0.5) & nl).sum()))
    prev_susp = np.vstack([np.zeros((1, tv.shape[1]), bool), sv[:-1]])
    add(sec, "|持有收益|>50% 且前一行停牌(复牌跳空)", int((tv & (np.abs(rhold) > 0.5) & prev_susp).sum()))
    pc = lp.raw("pct_chg").to_numpy(np.float64)
    d_pc = np.abs(rhold * 100 - pc)
    ok = tv & np.isfinite(d_pc)
    add(sec, "后复权收益 vs pctChg(仅 baostock 源):股票-日", int(ok.sum()))
    add(sec, "后复权收益 vs pctChg:|差|中位(百分点)", float(np.median(d_pc[ok])))
    add(sec, "后复权收益 vs pctChg:|差|p99(百分点)", float(np.quantile(d_pc[ok], 0.99)))
    add(sec, "后复权收益 vs pctChg:|差|>0.01 百分点占比", float((d_pc[ok] > 0.01).mean()))
    add(sec, "后复权收益 vs pctChg:|差|>0.1 百分点 股票-日数", int((d_pc[ok] > 0.1).sum()))
    del d_pc, pc
    # Q3 / Q8:baostock 原始文件(parts/,退市股 + 现存股抽样)
    import importlib.util
    _spec = importlib.util.spec_from_file_location("lhb", ROOT / "scripts" / "lh_build_panel.py")
    lhb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lhb)
    st_info_m = lhb.StLookup._read_st_info()
    bc = bs_raw_checks(dates, st_info_m)
    q3_bs = bc["n_missing"] / max(bc["n_rows"] + bc["n_missing"], 1)
    add(sec, "baostock 原始文件数", bc["n_files"])
    add(sec, "baostock 原始:首行..末行之间缺行数", bc["n_missing"], note=",".join(bc["miss_codes"][:10]))
    add(sec, "baostock 原始:缺行率", q3_bs)
    add(sec, "baostock 原始:不在交易日历上的行", bc["n_off"])
    add(sec, "baostock 原始:tradestatus=0 行数", bc["n_ts0"])
    add(sec, "baostock 原始:tradestatus=0 且 volume>0 行数", bc["n_ts0_vol"])
    n_ts0, n_volpos = bc["n_ts0"], bc["n_ts0_vol"]
    dpc = bc["pc_diff"]
    add(sec, "baostock 原始:后复权 收盘/前收-1 vs pctChg 股票-日", len(dpc))
    add(sec, "baostock 原始:后复权收益 vs pctChg |差| 中位(百分点)", float(np.median(dpc)))
    add(sec, "baostock 原始:后复权收益 vs pctChg |差| p99(百分点)", float(np.quantile(dpc, 0.99)))
    add(sec, "baostock 原始:后复权收益 vs pctChg |差|>0.01 百分点占比", float((dpc > 0.01).mean()))
    add(sec, "baostock 原始:后复权收益 vs pctChg |差|>0.1 百分点 股票-日", int((dpc > 0.1).sum()))
    gate("Q8", "停牌一致:tradestatus=0 的行 volume 应为 0(baostock 原始文件)",
         f"{n_volpos / max(n_ts0, 1):.4%}({n_volpos}/{n_ts0})", "违例 < 0.01%", n_volpos / max(n_ts0, 1) < 1e-4,
         "新浪停牌日无行(构建补出的停牌行 volume 恒为 0),不适用")

    # ---------------- 3b 两源一致 ----------------
    tw = two_source(uni)
    q3_note = ""
    if len(tw):
        rate_s = float((tw["only_bs"].sum()) / max(tw["n_bs_trade"].sum(), 1))
        q3_note = f";新浪:两源重叠 {len(tw)} 只上 baostock 有成交而新浪无行 {int(tw['only_bs'].sum())}/{int(tw['n_bs_trade'].sum())} = {rate_s:.4%}"
    else:
        rate_s = 0.0
    gate("Q3", "行数完整:上市期间缺行率(防 rs.next() 空转或截断)",
         f"baostock 原始文件 {q3_bs:.4%}{q3_note}", "缺行率 < 0.5%", (q3_bs < 0.005) and (rate_s < 0.005),
         "新浪停牌日本来无行,'缺行'无法与停牌区分 -> 用重叠股票上 baostock 的成交日作对照;另见 §6 与 Panel 的成交状态一致率")

    # ---------------- 4 涨跌停 / 一字板 / ST ----------------
    sec = "4 涨跌停"
    lu, ld = lp.raw("limit_up").to_numpy(bool), lp.raw("limit_down").to_numpy(bool)
    owu, owd = lp.raw("one_word_up").to_numpy(bool), lp.raw("one_word_down").to_numpy(bool)
    olu, old = lp.raw("ow_lim_up").to_numpy(bool), lp.raw("ow_lim_down").to_numpy(bool)
    oau, oad = lp.raw("open_at_up").to_numpy(bool), lp.raw("open_at_dn").to_numpy(bool)
    ob = lp.raw("out_band").to_numpy(bool)
    rate = lp.raw("lim_up_rate").to_numpy(np.float64)
    main_cols = np.array([not c.startswith(("300", "301", "302", "688", "689")) for c in codes])
    for y in years:
        my = (yr == y).to_numpy()[:, None]
        add(sec, "涨停(收盘=涨停价)股票-日", int((lu & my).sum()), y)
        add(sec, "一字涨停(最高==最低 且 >+4.5%)", int((owu & my).sum()), y)
        add(sec, "一字涨停(精确口径:收盘=涨停价 且 振幅<0.4%)", int((olu & my).sum()), y)
        add(sec, "开盘即涨停", int((oau & my).sum()), y)
        add(sec, "跌停(收盘=跌停价)股票-日", int((ld & my).sum()), y)
        add(sec, "一字跌停(最高==最低 且 <-4.5%)", int((owd & my).sum()), y)
        add(sec, "开盘即跌停", int((oad & my).sum()), y)
        add(sec, "无涨跌幅限制日(含新股首日/注册制前5日/越带兜底)", int((nl & tv & my).sum()), y)
    a1, a2 = owu | owd, olu | old
    add(sec, "一字板 两种口径 Jaccard(全期)", float((a1 & a2).sum() / max((a1 | a2).sum(), 1)))
    add(sec, "一字板 简单口径有而精确口径无", int((a1 & ~a2).sum()))
    add(sec, "一字板 精确口径有而简单口径无", int((a2 & ~a1).sum()))
    for lab, msk in (("新浪源", is_sina), ("baostock 源", ~is_sina)):
        b1, b2 = a1[:, msk], a2[:, msk]
        add(sec, f"一字板 两种口径 Jaccard [{lab}]", float((b1 & b2).sum() / max((b1 | b2).sum(), 1)))
    del a1, a2
    # 主板 ST 10% 规则(2025-07-07)实测
    d07 = (dates >= pd.Timestamp("2025-07-07"))
    for lab, msk in (("2025-01-01..2025-07-04", (dates >= pd.Timestamp("2025-01-01")) & ~d07),
                     ("2025-07-07..2025-12-31", d07)):
        sel = tv & stv & main_cols[None, :] & msk[:, None]
        add(sec, f"主板 ST 交易日数 [{lab}]", int(sel.sum()))
        add(sec, f"主板 ST 交易日中 10% 带占比 [{lab}]", float((sel & np.isclose(rate, 0.10)).sum() / max(sel.sum(), 1)))
    add(sec, "5% 带越带后按 10% 带判定的股票-日(退市整理期等)", int(info["n_esc10"].sum()))
    # Q5:越带日(超出当日涨跌幅上限、且不是规则内的无涨跌幅日),逐条归类
    obt = ob & tv
    ii, jj = np.nonzero(obt)
    cat = np.full(len(ii), "未解释", dtype=object)
    reform = (dates.year[ii] <= 2006) & prev_susp[ii, jj]
    resume = prev_susp[ii, jj] & ~reform
    lt_pos = pd.to_datetime(info["last_trade"]).map(pos).to_numpy()
    is_del = ((info["outDate"].fillna("") != "") & (info["outDate"].fillna("") <= RESEARCH_END)).to_numpy()
    near_end = is_del[jj] & (ii >= lt_pos[jj] - 45)
    cat[near_end] = "退市整理期(最后交易日前 45 日内)"
    ft_pos = pd.to_datetime(info["first_trade"]).map(pos).to_numpy()
    cat[ii == ft_pos[jj]] = "新股上市首日(2014-01 等首日规则与 +44% 带不同)"
    cat[resume] = "停牌后复牌首日(含恢复上市)"
    cat[reform] = "2005-2006 停牌后复牌首日(股改)"
    q5 = pd.DataFrame(dict(date=dates[ii], code=codes[jj], ret=rhold[ii, jj], is_st=stv[ii, jj],
                           st_known=lp.raw("st_known").to_numpy(bool)[ii, jj], src=info["src"].to_numpy()[jj],
                           category=cat))
    q5.to_csv(LEDGER / "lh_data_qa_q5.csv", index=False)
    q5_y = {}
    for y in years:
        my = (yr == y).to_numpy()
        n_tr = int(tv[my].sum())
        g = q5[q5["date"].dt.year == y]
        un = int((g["category"] == "未解释").sum())
        q5_y[y] = un / max(n_tr, 1)
        add(sec, "越带日(超出规则带)股票-日", len(g), y, note=g["category"].value_counts().to_dict().__repr__())
        add(sec, "越带日 未解释 占交易股票-日比例", q5_y[y], y)
    worst = max(q5_y, key=q5_y.get)
    gate("Q5", "收益异常:超出当日涨跌幅上限的股票日(扣除股改复牌/复牌首日/退市整理期/新股首日),逐年占比",
         f"逐年最高 {q5_y[worst]:.4%}({worst});未解释共 {int((q5['category'] == '未解释').sum())} 条",
         "每年 < 0.05%;超出部分逐条列出(lh_data_qa_q5.csv)", q5_y[worst] < 5e-4,
         "判定用不复权收盘 vs 涨跌停价 ± 半个价位(比 +0.5pp 更严),ST 5% 带越带时先试 10% 带;新股首日与注册制前 5 日是规则内无涨跌幅日,不计")
    # ST 身份与来源
    sk = lp.raw("st_known").to_numpy(bool)
    for y in years:
        my = (yr == y).to_numpy()
        tvs = tv[my][:, is_sina]
        add(sec, "新浪源 交易股票-日中 ST 身份有来源的比例", float((sk[my][:, is_sina] & tvs).sum() / max(tvs.sum(), 1)), y)
    # st_info(项目名单)作 ST 来源的近似质量:baostock 原始文件(退市股 + 现存股抽样)2016-08-09 起交易日上比较
    sc = bc["st_cmp"]
    add(sec, "st_info vs baostock isST 一致比例(baostock 原始文件,2016-08-09 起交易股票-日)", sc["agree"] / max(sc["n"], 1),
        note=f"n={sc['n']},baostock ST 股票-日 {sc['n_st_bs']}")
    add(sec, "baostock ST 而 st_info 非 ST 股票-日", sc["bs_only"])
    add(sec, "st_info ST 而 baostock 非 ST 股票-日", sc["info_only"])
    # 新股首日规则
    new = info[(ipo >= "2005-01-04")].copy()
    add(sec, "2005 后上市股票数", len(new))
    add(sec, "其中首个交易日 == ipoDate 比例", float((pd.to_datetime(new["first_trade"]) == pd.to_datetime(new["ipoDate"])).mean()))
    ft_pos = pd.to_datetime(new["first_trade"]).map(pos).to_numpy()
    okn = np.isfinite(ft_pos)
    jn = np.array([codes.get_loc(c) for c in new.index])[okn]
    ftp = ft_pos[okn].astype(int)
    first_lu = lu[ftp, jn]; first_nl = nl[ftp, jn]
    newi = new[okn]
    is_reg = np.array([c.startswith(("688", "689")) or (c.startswith(("300", "301", "302")) and d >= pd.Timestamp("2020-08-24"))
                       or (not c.startswith(("300", "301", "302", "688", "689")) and d >= pd.Timestamp("2023-04-10"))
                       for c, d in zip(newi.index, pd.to_datetime(newi["ipoDate"]))])
    era44 = (pd.to_datetime(newi["first_trade"]) >= pd.Timestamp("2014-01-01")).to_numpy() & ~is_reg
    add(sec, "2014+ 非注册制新股数", int(era44.sum()))
    add(sec, "其中首日收盘 = +44% 涨停价 比例", float(first_lu[era44].mean()) if era44.any() else np.nan)
    add(sec, "其中首日被兜底为无涨跌幅日比例", float(first_nl[era44].mean()) if era44.any() else np.nan)
    pre14 = (pd.to_datetime(newi["first_trade"]) < pd.Timestamp("2014-01-01")).to_numpy()
    add(sec, "2005-2013 新股首日 无涨跌幅日 比例(规则即无限制)", float(first_nl[pre14].mean()) if pre14.any() else np.nan)
    add(sec, "注册制新股数", int(is_reg.sum()))
    del lu, ld, owu, owd, olu, old, oau, oad, ob, rate, sk

    # ---------------- 5 退市 ----------------
    sec = "5 退市"
    del_info = info[(info["outDate"].fillna("") >= "2005-01-01") & (info["outDate"].fillna("") <= RESEARCH_END)].copy()
    rows = []
    close_raw = lp.raw("close_raw").to_numpy(np.float64)
    owd_all = lp.raw("one_word_down").to_numpy(bool)
    for code, r in del_info.iterrows():
        j = codes.get_loc(code)
        lt = pd.Timestamp(r["last_trade"]) if pd.notna(r["last_trade"]) else pd.NaT
        od = pd.Timestamp(r["outDate"])
        p_lt = int(pos.get(lt, -1)) if pd.notna(lt) else -1
        p_od = int(np.searchsorted(dates.values, od.to_datetime64()))
        tcol = tv[:, j]
        run = 0
        if p_lt >= 0:
            k = p_lt
            while k >= 0 and tcol[k]:
                run += 1; k -= 1
        w0 = max(p_lt - 59, 0) if p_lt >= 0 else 0
        rows.append(dict(code=code, src=r["src"], outDate=r["outDate"], last_trade=lt,
                         last_row=r["last_row"], gap_tdays_last_trade_to_out=(p_od - p_lt) if p_lt >= 0 else np.nan,
                         run_trade_before_last=run, n_trade_last60=int(tcol[w0:p_lt + 1].sum()) if p_lt >= 0 else 0,
                         n_one_word_down_last30=int(owd_all[max(p_lt - 29, 0):p_lt + 1, j].sum()) if p_lt >= 0 else 0,
                         last_close_raw=float(close_raw[p_lt, j]) if p_lt >= 0 else np.nan,
                         ever_st=bool(stv[:, j].any()), st_at_last=bool(stv[p_lt, j]) if p_lt >= 0 else False))
    dd = pd.DataFrame(rows)
    dd.to_csv(LEDGER / "lh_data_qa_delisted.csv", index=False)
    add(sec, "2005-2025 退市股(有数据)", len(dd))
    add(sec, "最后交易日距 outDate 交易日数 中位", float(dd["gap_tdays_last_trade_to_out"].median()))
    add(sec, "最后交易日距 outDate <= 5 个交易日 占比", float((dd["gap_tdays_last_trade_to_out"] <= 5).mean()))
    add(sec, "最后交易日距 outDate > 60 个交易日 占比", float((dd["gap_tdays_last_trade_to_out"] > 60).mean()))
    add(sec, "最后交易日前连续交易 >= 10 日 占比(退市整理期在数据里)", float((dd["run_trade_before_last"] >= 10).mean()))
    add(sec, "退市时 ST 占比(最后交易日 isST)", float(dd["st_at_last"].mean()))
    add(sec, "最后不复权收盘 < 1 元 占比", float((dd["last_close_raw"] < 1).mean()))
    dd["oy"] = pd.to_datetime(dd["outDate"]).dt.year
    for y, g in dd.groupby("oy"):
        add(sec, "退市数", len(g), y)
        add(sec, "最后交易日前连续交易>=10日占比", float((g["run_trade_before_last"] >= 10).mean()), y)
        add(sec, "最后30交易日一字跌停数 中位", float(g["n_one_word_down_last30"].median()), y)
        add(sec, "最后不复权收盘<1元占比", float((g["last_close_raw"] < 1).mean()), y)
        add(sec, "最后交易日距 outDate 交易日数 中位", float(g["gap_tdays_last_trade_to_out"].median()), y)
    del close_raw, owd_all
    # 与 data_8:21/delisted_history.csv(tushare,2020+ 退市股)交叉
    dh = pd.read_csv(ROOT / "data_8:21" / "delisted_history.csv")
    dh = dh[dh["trade_date"] <= int(END8)].copy()      # 先截断
    dh["date"] = pd.to_datetime(dh["trade_date"].astype(str))
    dh = dh[dh["ts_code"].isin(codes)]
    sub_c = sorted(set(dh["ts_code"]))
    if sub_c:
        cr = lp.raw("close_raw")[sub_c].stack().rename("c_lh")
        am = lp.raw("amount")[sub_c].stack().rename("a_lh")
        lh_x = pd.concat([cr, am], axis=1).reset_index()
        lh_x.columns = ["date", "ts_code", "c_lh", "a_lh"]
        x = dh.merge(lh_x, on=["date", "ts_code"], how="inner")
        x = x[np.isfinite(x["c_lh"])]
        add(sec, "delisted_history.csv(项目旧抓 BaoStock 未复权,仅 10 只)与面板重叠股票-日", len(x),
            note=f"{dh['ts_code'].nunique()} 只")
        add(sec, "  不复权收盘 |相对差|>0.1% 占比", float((np.abs(x["close"] / x["c_lh"] - 1) > 1e-3).mean()))
        add(sec, "  成交额(元)比值中位", float((x["amount"] / x["a_lh"]).median()))

    # ---------------- 6 与项目 Panel 重叠(2020-2025)----------------
    sec = "6 Panel 重叠"
    from afactor.panel import Panel
    from afactor import newdata as ND
    P = Panel()
    pd_, pc_ = P.dates, P.codes
    add(sec, "Panel 交易日数", len(pd_))
    ov_d = pd_.intersection(dates)
    add(sec, "共同交易日数", len(ov_d))
    pc_a = pd.Index([c for c in pc_ if c.endswith((".SH", ".SZ"))])
    add(sec, "Panel 沪深代码数", len(pc_a))
    ov_c = pc_a.intersection(codes)
    add(sec, "共同代码数", len(ov_c))
    add(sec, "Panel 有而长历史无的沪深代码", len(pc_a.difference(codes)), note=",".join(pc_a.difference(codes)[:10]))
    lh_in_ov = trade.loc[ov_d].any()
    lh_only = lh_in_ov[lh_in_ov].index.difference(pc_a)
    add(sec, "2020-2025 有成交但 Panel 没有的股票数(幸存者偏差缺口)", len(lh_only))
    add(sec, "其中 2020-2025 内退市", int(info.loc[lh_only, "outDate"].fillna("").between("2020-01-01", RESEARCH_END).sum()))
    add(sec, "缺口股票的交易股票-日占长历史 2020-2025 交易股票-日比例",
        float(trade.loc[ov_d, lh_only].to_numpy().sum() / trade.loc[ov_d].to_numpy().sum()))

    def sub(df):
        return df.reindex(index=ov_d, columns=ov_c)

    src_ov = info["src"].reindex(ov_c).to_numpy()
    c_lh = sub(lp.raw("close_px")); c_p = sub(P.raw("close_px"))
    r_lh = c_lh / c_lh.shift(1) - 1; r_p = c_p / c_p.shift(1) - 1
    both = r_lh.notna() & r_p.notna()
    dmat = ((r_lh - r_p) * 1e4).to_numpy(np.float64)
    bm = both.to_numpy()
    s_sina = np.isin(src_ov, ["sina", "sina+bs"])
    for lab, cm in (("全部", np.ones(len(ov_c), bool)), ("新浪价格", s_sina), ("baostock 价格", ~s_sina)):
        dr = pd.Series(dmat[:, cm][bm[:, cm]])
        if not len(dr):
            continue
        add(sec, f"日收益 共同股票-日 [{lab}]", len(dr))
        add(sec, f"日收益 |差| 中位(bp) [{lab}]", float(dr.abs().median()))
        add(sec, f"日收益 |差| p99(bp) [{lab}]", float(dr.abs().quantile(0.99)))
        add(sec, f"日收益 |差|>1bp 占比 [{lab}]", float((dr.abs() > 1).mean()))
        add(sec, f"日收益 |差|>10bp 占比 [{lab}]", float((dr.abs() > 10).mean()))
        add(sec, f"日收益 |差|>100bp 股票-日 [{lab}]", int((dr.abs() > 100).sum()))
    add(sec, "日收益 相关系数", float(np.corrcoef(stack_valid(r_lh, both), stack_valid(r_p, both))[0, 1]))
    dl_ = (np.log1p(r_lh) - np.log1p(r_p)).where(both)
    add(sec, "对数收益差 长历史-Panel 等权漂移(bp/年,全部共同股票-日)", float(dl_.mean(1).mean() * 242 * 1e4))
    tot_ = dl_.sum()
    add(sec, "对数收益差 长历史-Panel 逐股累计 中位 / p05 / p95(bp)",
        f"{tot_.median()*1e4:.2f} / {tot_.quantile(.05)*1e4:.2f} / {tot_.quantile(.95)*1e4:.2f}")
    del dl_, tot_
    del dmat
    v_lh = sub(lp.raw("close_px").notna()); v_p = sub(P.raw("close_px").notna())
    sl = sub(listed).to_numpy()
    n_lh = v_lh.to_numpy().sum(0); n_p = v_p.to_numpy().sum(0)
    p_has = n_p >= 0.5 * np.maximum(n_lh, 1)            # Panel 价格天数 >= 长历史交易天数的一半
    add(sec, "共同代码中 Panel 价格天数不到长历史交易天数一半的股票数(Panel 分钟数据缺口)", int((~p_has).sum()),
        note=",".join(ov_c[~p_has][:15]))
    add(sec, "  这些股票在长历史中的交易股票-日 / 在 Panel 中有价格的股票-日",
        f"{int(n_lh[~p_has].sum())} / {int(n_p[~p_has].sum())}")
    vl2, vp2, sl2 = v_lh.to_numpy()[:, p_has], v_p.to_numpy()[:, p_has], sl[:, p_has]
    add(sec, "有价格(交易)状态一致比例(共同上市期,剔除上述 Panel 缺口股票)", float((vl2 == vp2)[sl2].mean()))
    add(sec, "长历史交易而 Panel 无价格 股票-日(同上)", int((vl2 & ~vp2).sum()))
    add(sec, "Panel 有价格而长历史停牌/无行 股票-日", int((~vl2 & vp2).sum()))
    xs = pd.Series((~vl2 & vp2).sum(0), index=ov_c[p_has])
    add(sec, "  其中最多的股票", str(xs[xs > 0].sort_values(ascending=False).head(8).to_dict()))
    del vl2, vp2, sl2
    a_lh = sub(lp.raw("amount")); a_p = sub(P.raw("amount"))
    ra = stack_valid(a_lh / a_p, (a_lh > 0) & (a_p > 0))
    add(sec, "成交额 长历史/Panel 比值 中位", float(ra.median()))
    for k, v in q(ra).items():
        add(sec, f"成交额 比值 {k}", v)
    add(sec, "成交额 |比值-1|<=1% 占比", float((np.abs(ra - 1) <= 0.01).mean()))
    add(sec, "成交额 |比值-1|<=5% 占比", float((np.abs(ra - 1) <= 0.05).mean()))
    del ra
    # 开盘价 vs Panel 开盘集合竞价价 / 开盘 10 分钟 TWAP(各自同源收盘做分母,消掉复权口径差)
    o_lh = sub(lp.raw("open_px"))
    oa_p = sub(P.raw("open_auction_px")); tw_p = sub(P.raw("twap_open10"))
    d_auc = stack_valid(np.log((o_lh / c_lh) / (oa_p / c_p)) * 1e4, (o_lh > 0) & (oa_p > 0) & (c_p > 0))
    add(sec, "日开盘价 vs Panel 09:30 集合竞价价 |差|中位(bp)", float(d_auc.abs().median()))
    add(sec, "日开盘价 vs Panel 09:30 集合竞价价 |差|>1bp 占比", float((d_auc.abs() > 1).mean()))
    del d_auc
    d_tw = stack_valid(np.log((o_lh / c_lh) / (tw_p / c_p)) * 1e4, (o_lh > 0) & (tw_p > 0) & (c_p > 0))
    ex_rows = [dict(item="log(日开盘/收盘) - log(twap_open10/收盘), bp", n=len(d_tw), mean=float(d_tw.mean()),
                    std=float(d_tw.std()), mad=float(d_tw.abs().mean()), **q(d_tw))]
    add(sec, "执行价近似:日开盘 vs open10 TWAP 差 均值(bp,正=开盘高于 TWAP)", float(d_tw.mean()))
    add(sec, "执行价近似:|差| 中位(bp)", float(d_tw.abs().median()))
    add(sec, "执行价近似:差 标准差(bp)", float(d_tw.std()))
    for k, v in q(d_tw).items():
        add(sec, f"执行价近似:差 {k}(bp)", v)
    del d_tw
    oo = o_lh / o_lh.shift(1) - 1; tt = tw_p / tw_p.shift(1) - 1
    bb = oo.notna() & tt.notna()
    d_ret = stack_valid((oo - tt) * 1e4, bb)
    ex_rows.append(dict(item="开-开日收益 - TWAP-TWAP 日收益, bp", n=len(d_ret), mean=float(d_ret.mean()),
                        std=float(d_ret.std()), mad=float(d_ret.abs().mean()), **q(d_ret)))
    add(sec, "开-开 vs TWAP-TWAP 日收益 相关", float(np.corrcoef(stack_valid(oo, bb), stack_valid(tt, bb))[0, 1]))
    add(sec, "开-开 vs TWAP-TWAP 日收益差 标准差(bp)", float(d_ret.std()))
    del d_ret, oo, tt, bb
    for y in sorted(set(ov_d.year)):
        my = (ov_d.year == y)
        dy = stack_valid(((o_lh / c_lh) / (tw_p / c_p)).loc[my].pipe(np.log) * 1e4,
                         ((o_lh > 0) & (tw_p > 0) & (c_p > 0)).loc[my])
        add(sec, "执行价近似:差 均值(bp)", float(dy.mean()), y)
        add(sec, "执行价近似:|差| 中位(bp)", float(dy.abs().median()), y)
    pd.DataFrame(ex_rows).to_csv(LEDGER / "lh_data_qa_exec.csv", index=False)
    del o_lh, oa_p, tw_p
    # 开盘 10 分钟成交额占比(Panel 2020-2025)
    sec7 = "7 开盘成交额占比"
    add(sec7, "amt_open10/amount 股票-日中位(LongPanel 采用值)", meta["open10_amt_ratio"])
    add(sec7, "均值", meta["open10_amt_ratio_mean"])
    add(sec7, "p10", meta["open10_amt_ratio_p10"])
    add(sec7, "p90", meta["open10_amt_ratio_p90"])
    for y, v in meta["open10_amt_ratio_by_year"].items():
        add(sec7, "中位", v, int(y))
    ao = P.raw("amt_open10").loc[: RESEARCH_END]; aa = P.raw("amount").loc[: RESEARCH_END]
    adv_p = P["adv20"].loc[: RESEARCH_END]
    rr = (ao / aa.where(aa > 0))
    up = P["universe"].loc[: RESEARCH_END].reindex_like(rr).fillna(False).astype(bool)
    add(sec7, "中位(只看 Panel 决策池 universe 股票-日)", float(stack_valid(rr, up).median()))
    qb = adv_p.rank(axis=1, pct=True)
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)):
        mm = (qb > lo) & (qb <= hi)
        add(sec7, f"中位 adv20 截面分位({lo:.1f},{hi:.1f}]", float(stack_valid(rr, mm).median()))
    del ao, aa, adv_p, rr, qb, up
    # 换手率 vs daily_basic(tushare turnover_rate)
    tb = ND.turnover_rate(ov_d, ov_c)
    t_lh = sub(lp.raw("turn"))
    rt = stack_valid(t_lh / tb, (t_lh > 0) & (tb > 0))
    add(sec, "换手率 长历史/daily_basic 比值 中位", float(rt.median()))
    add(sec, "换手率 |比值-1|<=1% 占比", float((np.abs(rt - 1) <= 0.01).mean()))
    add(sec, "换手率 |比值-1|<=5% 占比", float((np.abs(rt - 1) <= 0.05).mean()))
    for k, v in q(rt).items():
        add(sec, f"换手率 比值 {k}", v)
    for lab, cm in (("新浪换手率(现存股)", src_ov == "sina"), ("baostock 换手率(退市股等)", src_ov != "sina")):
        r2 = stack_valid((t_lh / tb).loc[:, cm], ((t_lh > 0) & (tb > 0)).loc[:, cm])
        if len(r2):
            add(sec, f"换手率 |比值-1|<=5% 占比 [{lab}]", float((np.abs(r2 - 1) <= 0.05).mean()), note=f"n={len(r2)}")
    # Q7:2021-2025 同一批幸存股票(Panel 决策池 & 两边都有),turn 日截面秩相关;等权 c2c 日收益相关
    d21 = ov_d[ov_d >= pd.Timestamp("2021-01-01")]
    upool = P["universe"].reindex(index=d21, columns=ov_c).fillna(False).astype(bool)
    tl, tbb = t_lh.loc[d21].where(upool), tb.loc[d21].where(upool)
    m2 = (tl > 0) & (tbb > 0)
    rc = tl.where(m2).rank(axis=1).corrwith(tbb.where(m2).rank(axis=1), axis=1)
    add(sec, "Q7 换手率 日截面秩相关 中位(2021-2025,Panel 决策池)", float(rc.median()), note=f"日数 {int(rc.notna().sum())}")
    add(sec, "Q7 换手率 日截面秩相关 最低", float(rc.min()))
    pool_prev = upool.shift(1, fill_value=False)
    rl, rp = r_lh.loc[d21], r_p.loc[d21]
    mm = pool_prev & rl.notna() & rp.notna()
    ew_l, ew_p = rl.where(mm).mean(1), rp.where(mm).mean(1)
    ewc = float(ew_l.corr(ew_p))
    add(sec, "Q7 等权 c2c 日收益相关(同一池子同一股票-日,2021-2025)", ewc)
    # 再看"各自全池"(长历史含退市股、Panel 不含):池子不同,相关只作参考
    ew_l2 = r_lh.loc[d21].where(pool_prev & r_lh.loc[d21].notna()).mean(1)
    add(sec, "Q7 参考:等权 c2c 日收益相关(长历史含该池全部共同代码 vs Panel)", float(ew_l2.corr(ew_p)))
    gate("Q7", "与研究面板一致(2021-2025):turn 与 tushare turnover_rate 日截面秩相关中位;同一池子等权日收益相关",
         f"秩相关中位 {rc.median():.4f};等权日收益相关 {ewc:.6f}", "秩相关中位 >= 0.95;等权相关 >= 0.99",
         (rc.median() >= 0.95) and (ewc >= 0.99), "池子 = Panel 决策池(前一日)∩ 两边都有收益")
    del tl, tbb, rc, rl, rp, mm, t_lh, tb
    # 两源大差(>10bp)在 2020+ 的日子用项目 Panel(tushare 复权)仲裁:哪一源与 Panel 一致
    try:
        bd = pd.read_csv(LEDGER / "lh_data_qa_twosource_bigdiff.csv")
        bd = bd[bd["date"] >= "2020-01-03"]
        cpx = P.raw("close_px")
        arb = []
        for r in bd.itertuples():
            k = f"{r.code.split('.')[1]}.{r.code.split('.')[0].upper()}"
            d0, d1 = pd.Timestamp(r.prev_date), pd.Timestamp(r.date)
            if k not in cpx.columns or d0 not in cpx.index or d1 not in cpx.index:
                continue
            rp = cpx.at[d1, k] / cpx.at[d0, k] - 1
            if not np.isfinite(rp):
                continue
            who = ("sina" if abs(rp - r.ret_sina) < 5e-4 else "baostock" if abs(rp - r.ret_bs) < 5e-4 else "neither")
            arb.append((who, bool(r.factor_change), bool(r.across_suspension)))
        arb = pd.DataFrame(arb, columns=["who", "fchg", "susp"])
        add(sec, "两源收益差>10bp 的 2020+ 股票-日数(可用 Panel 仲裁)", len(arb))
        for w, n in arb["who"].value_counts().items():
            add(sec, f"  与 Panel(tushare 复权)一致的一方 = {w}", int(n))
        ex = arb[arb["fchg"] & ~arb["susp"]]
        add(sec, "  其中普通除权日:与 Panel 一致的是 baostock / 新浪 / 都不是",
            f"{int((ex['who'] == 'baostock').sum())} / {int((ex['who'] == 'sina').sum())} / {int((ex['who'] == 'neither').sum())}")
    except Exception as e:                                       # noqa: BLE001
        add(sec, "两源大差 Panel 仲裁失败", str(e)[:200])
    # ST 与一字板一致性(与 Panel)
    st_p = sub(P["is_st"]).fillna(False).astype(bool); st_l = sub(lp.raw("is_st"))
    vv = sub(trade).to_numpy() & v_p.to_numpy()
    add(sec, "ST 标记一致比例(共同交易股票-日)", float((st_p.to_numpy() == st_l.to_numpy())[vv].mean()))
    add(sec, "长历史 ST 而项目名单非 ST 股票-日", int((st_l.to_numpy() & ~st_p.to_numpy() & vv).sum()))
    add(sec, "项目名单 ST 而长历史非 ST 股票-日", int((~st_l.to_numpy() & st_p.to_numpy() & vv).sum()))
    for nm in ("one_word_up", "one_word_down", "limit_up", "limit_down"):
        a1 = sub(lp[nm]).to_numpy(bool) & vv; a2 = sub(P[nm]).fillna(False).to_numpy(bool) & vv
        add(sec, f"{nm} 长历史股票-日", int(a1.sum()))
        add(sec, f"{nm} Panel 股票-日", int(a2.sum()))
        add(sec, f"{nm} 两者都标", int((a1 & a2).sum()))
        add(sec, f"{nm} Jaccard", float((a1 & a2).sum() / max((a1 | a2).sum(), 1)))
    del P

    # ---------------- 8 Q9 指数对照 ----------------
    sec = "8 指数对照"
    ix = pd.read_csv(ROOT / "cache" / "_tmp" / "lh_prereg_index_prices.csv")
    gz = ix[ix["code"] == "sz.399317"].copy()
    gz["date"] = pd.to_datetime(gz["date"])
    gz = gz.set_index("date")["close"].astype(float).reindex(dates)
    r_ix = gz / gz.shift(1) - 1
    c_all = lp.raw("close_px")
    r_all = (c_all / c_all.shift(1) - 1)
    ew = r_all.where(np.isfinite(r_all) & (r_all.abs() < 1.0)).mean(1)          # 自建全 A 等权收盘指数日收益(防错位粗检)
    corr_y = {}
    for y in years:
        my = (dates.year == y)
        corr_y[y] = float(ew[my].corr(r_ix[my]))
        add(sec, "自建全A等权 vs 国证A股 日收益相关", corr_y[y], y)
    lag = {k: float(ew.corr(r_ix.shift(k))) for k in (-1, 0, 1)}
    add(sec, "相关 vs 错位 k(-1/0/+1)", str({k: round(v, 4) for k, v in lag.items()}))
    # 参照:项目 Panel(研究面板,无退市股)的全池等权 c2c 日收益与国证A股的相关(2020-2025),看低相关是否来自
    # "等权 vs 市值加权"本身(与数据无关)
    try:
        from afactor.panel import Panel as _P
        P2 = _P(); cp = P2.raw("close_px"); rp2 = (cp / cp.shift(1) - 1)
        ewp = rp2.where(np.isfinite(rp2) & (rp2.abs() < 1.0)).mean(1)
        for y in sorted(set(ewp.index.year)):
            mk = ewp.index.year == y
            add(sec, "参照:项目 Panel 全池等权 vs 国证A股 日收益相关", float(ewp[mk].corr(r_ix.reindex(ewp.index)[mk])), y)
        del P2, cp, rp2
    except Exception as e:                       # noqa: BLE001
        add(sec, "参照计算失败", str(e)[:200])
    wy = min(corr_y, key=corr_y.get)
    gate("Q9", "指数对照:自建全A等权收盘指数与国证A股的日收益相关(按年)",
         f"逐年最低 {corr_y[wy]:.4f}({wy});峰值错位 k={max(lag, key=lag.get)}", "每年 >= 0.85", corr_y[wy] >= 0.85,
         "全 A 等权 = 当日有收盘的全部股票(含退市股)c2c 收益均值;只作防错位粗检,不是任何策略")
    del c_all, r_all

    q_df = pd.DataFrame(ROWS, columns=["section", "metric", "year", "value", "note"])
    q_df.to_csv(LEDGER / "lh_data_qa.csv", index=False)
    pd.DataFrame(GATES).to_csv(LEDGER / "lh_data_qa_gates.csv", index=False)
    print(f"写出 {LEDGER / 'lh_data_qa.csv'}({len(q_df)} 行),门槛 {sum(g['passed'] for g in GATES)}/{len(GATES)} 通过,"
          f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
