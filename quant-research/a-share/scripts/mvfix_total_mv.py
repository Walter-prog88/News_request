"""真实市值日频外推修复的量化(前缀 mvfix,2026-10-10;报告 research/TOTAL_MV_FIX.md)。

改了什么:newdata.total_mv 的总股本外推。旧口径(newdata.TOTAL_SHARE_METHOD="legacy")把流通股本相对
"PIT 总股本最近一次数值变化时"的全部增长乘到总股本上,限售解禁让倍数一直累积;新口径每份财报按报告期末重新锚定,
只外推识别出的送转(newdata.total_share 的说明)。本脚本在同一进程里切换开关,其余代码完全相同。

    python3 scripts/mvfix_total_mv.py --stage eval        # 新旧真实市值、送转识别验证、事后真值对照、案例(约 3 分钟)
    python3 scripts/mvfix_total_mv.py --stage barra       # Barra 风格暴露新旧对照(约 6 分钟)
    python3 scripts/mvfix_total_mv.py --stage shadow      # 建两个影子根目录(legacy / new),下游报告在里面重跑
    python3 scripts/mvfix_total_mv.py --stage downstream  # 汇总影子根目录里的重跑结果 -> mvfix_downstream_*.csv

下游重跑(风险模型 / 中证1000 代理 / 容量 / 指增)在影子根目录 cache/_tmp/mvfix_shadow/{legacy,new} 里进行:
两边代码只差 newdata.TOTAL_SHARE_METHOD 的默认值,缓存与账本各自独立,**不覆盖**本项目 research/ledger 下的原账本。
具体命令见 --stage shadow 的输出与 research/TOTAL_MV_FIX.md §5。

产出(research/ledger/):
  mvfix_extrapolation.csv      审查指标(total_mv / (未复权价 × PIT 总股本))新旧分布、长期高估股票数、案例
  mvfix_songzhuan_events.csv   流通股本跳升事件逐条:比例、价比、识别结果、新浪复权因子标签、事后财报股本比
  mvfix_songzhuan_confusion.csv 送转识别 vs 复权因子标签,按比例分档
  mvfix_anchor_timing.csv      期后送转(报告期末之后、公告之前)是否已反映在该期股本里
  mvfix_truth_eval.csv         事后真值对照(只用于评估,用到了 t 之后的财报):新旧误差分布
  mvfix_cases.csv              个股案例
  mvfix_barra_delta.csv        Barra 暴露新旧对照(格子级差异、截面秩相关、按市值分档)
  mvfix_downstream_*.csv       下游四份报告的新旧对照
纪律:只用 <= 2025-12-31 的数据;复权因子(adj_factor.csv)只当验证标签,读入后立即截到截止日;不碰 cache/sealed。
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from afactor import config as C, newdata as ND, fundamental as FU   # noqa: E402

LEDGER = C.LEDGER
OUT = C.TMP / "mvfix"
SHADOW = C.TMP / "mvfix_shadow"
CUT = pd.Timestamp(C.DATA_CUTOFF)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def _mv(method: str, dates, codes) -> pd.DataFrame:
    old = ND.TOTAL_SHARE_METHOD
    ND.TOTAL_SHARE_METHOD = method
    try:
        return ND.total_mv(dates, codes)
    finally:
        ND.TOTAL_SHARE_METHOD = old


def panels(refresh: bool = False):
    """研究面板上的新旧真实市值与原料(落 cache/_tmp/mvfix/,原精度)。"""
    from afactor.panel import Panel
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["mv_legacy", "mv_new", "raw_close", "float_share", "pit_share", "sh_new"]
    if not refresh and all((OUT / f"{n}.parquet").exists() for n in names):
        return {n: pd.read_parquet(OUT / f"{n}.parquet") for n in names}
    p = Panel()
    d, c = p.dates, p.codes
    assert d.max() <= CUT
    out = {"mv_legacy": _mv("legacy", d, c), "mv_new": _mv("songzhuan", d, c),
           "raw_close": ND.raw_close(d, c), "float_share": ND.float_share(d, c),
           "pit_share": FU._pit_daily(FU.build_report_level(), "total_share", d, c).replace(0, np.nan),
           "sh_new": ND.total_share(d, c).astype(np.float64)}
    for n, v in out.items():
        v.to_parquet(OUT / f"{n}.parquet")
    return out


# ====================================================================== eval
def adj_factor_panel(dates, codes) -> pd.DataFrame:
    """新浪复权因子(独立来源)。只当验证标签:读入后立即截到截止日。"""
    af = pd.read_csv(ND.DATA / "adj_factor.csv")
    af = af[af["trade_date"] <= int(CUT.strftime("%Y%m%d"))]
    af["trade_date"] = ND._ymd(af["trade_date"])
    return af.pivot(index="trade_date", columns="ts_code", values="adj_factor").reindex(index=dates, columns=codes)


def float_events(cal, codes):
    """daily_basic 完整日历上的流通股本跳升事件(r >= 1.02,相对上一个有效观测)+ 识别结果 + 两种标签。"""
    fs = ND.float_share(cal, codes).to_numpy(np.float64)
    px = ND.raw_close(cal, codes).to_numpy(np.float64)
    steps = ND._songzhuan_steps(fs, px)
    adj = adj_factor_panel(cal, codes).to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = fs / ND._ffill_shift(fs)
        lp0 = np.log(px / ND._ffill_shift(px))
        la0 = np.log(adj / ND._ffill_shift(adj))
    lp1 = np.vstack([np.full((1, len(codes)), np.nan), lp0[:-1]])
    la1 = np.vstack([np.full((1, len(codes)), np.nan), la0[:-1]])
    ii, jj = np.nonzero(np.nan_to_num(r) >= 1.02)
    lr = np.log(r[ii, jj])
    # 标签:复权因子在 t 或 t-1 跳升 log r(允许现金分红再多 0~10%)= 送转除权
    lab = np.zeros(len(ii), bool)
    for la in (la0, la1):
        x = la[ii, jj]
        lab |= (x >= lr - 0.015) & (x <= lr + 0.10)
    ev = pd.DataFrame({"code": codes[jj], "date": cal[ii], "r": r[ii, jj], "lp_t": lp0[ii, jj], "lp_t_1": lp1[ii, jj],
                       "is_songzhuan_rule": steps[ii, jj] > 1.0, "adj_label": lab,
                       "adj_avail": np.isfinite(la0[ii, jj]) | np.isfinite(la1[ii, jj])})
    # 第二标签(事后):下一期财报股本 / 上一期财报股本(最终版),只用于评估
    rep = FU._final_rows(FU.build_report_level()).dropna(subset=["total_share"])
    q = rep.pivot_table(index="qi", columns="stock_code", values="total_share", aggfunc="last")
    qi_t = (ev["date"].dt.year * 4 + ev["date"].dt.quarter - 1).to_numpy()
    s_hi = _lookup(q, qi_t, ev["code"].to_numpy())
    s_lo = _lookup(q, qi_t - 1, ev["code"].to_numpy())
    ev["report_ratio_after_before"] = s_hi / s_lo
    ev["full_float_before"] = (fs[ii, jj] / r[ii, jj]) >= 0.98 * s_lo           # 跳升前流通股本 ≈ 总股本
    return ev, steps


def _lookup(q: pd.DataFrame, qi, codes):
    """q[qi, code] 逐元素取值(缺失 -> NaN)。"""
    ri = q.index.get_indexer(qi)
    ci = q.columns.get_indexer(codes)
    v = q.to_numpy(np.float64)
    ok = (ri >= 0) & (ci >= 0)
    out = np.full(len(qi), np.nan)
    out[ok] = v[ri[ok], ci[ok]]
    return out


def rbin(r):
    return pd.cut(r, [1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 100.0], right=False).astype(str)


def stage_eval():
    P = panels(refresh=ARGS.refresh)
    mv0, mv1, rc, fs, sh = P["mv_legacy"], P["mv_new"], P["raw_close"], P["float_share"], P["pit_share"]
    dates, codes = mv0.index, mv0.columns
    rows = []

    def put(section, check, value, note=""):
        rows.append(dict(section=section, check=check, value=value, note=note))
        print(f"[{section}] {check}: {value} {note}", flush=True)

    # ---- 0. 旧口径开关能否逐位复现 fixb 留下的当前代码面板
    ref = Path(os.environ.get("MVFIX_FIXB_REF", C.TMP / "fixb_impact" / "new")) / "nd_total_mv.parquet"
    if ref.exists():
        r0 = pd.read_parquet(ref)
        same = r0.index.equals(mv0.index) and r0.columns.equals(mv0.columns) and \
            np.array_equal(r0.to_numpy(), mv0.to_numpy(), equal_nan=True)
        put("复现", "legacy 开关 vs fixb 面板 cache/_tmp/fixb_impact/new/nd_total_mv.parquet 逐位相同", bool(same),
            f"{r0.shape};fixb_review2_checks.csv 的'附带发现'就是在这个面板上算的")
    # ---- 1. 审查指标(与 fixb_review2_checks.py 同一定义)
    base = (rc * sh)
    keep = mv0.notna() & sh.notna() & (fs <= sh * 1.01)
    for tag, mv in [("旧", mv0), ("新", mv1)]:
        ratio = (mv / base).where(keep)
        r = ratio.to_numpy()
        r = r[np.isfinite(r)]
        put("审查指标", f"{tag}:total_mv / (未复权价 x PIT 总股本) 分布(流通股本 <= 总股本的格子)",
            f">1.05: {(r > 1.05).mean():.1%};>1.5: {(r > 1.5).mean():.1%};>2: {(r > 2).mean():.1%};<0.95: {(r < 0.95).mean():.1%}",
            f"共 {r.size} 格")
        long = (ratio > 1.5).sum()
        put("审查指标", f"{tag}:比值 > 1.5 持续 >= 250 个交易日的股票数", int((long >= 250).sum()),
            f"有过 > 1.5 的股票 {int((long > 0).sum())} / {int(ratio.notna().any().sum())}")
        for c in ["600339.SH", "600390.SH", "600519.SH"]:
            x = ratio[c].dropna()
            put("审查指标", f"{tag}:{c} 比值中位 / 最大", f"{x.median():.2f} / {x.max():.2f}")
    q = (mv1 / mv0).to_numpy()
    q = q[np.isfinite(q)]
    put("新/旧", "新 / 旧 total_mv 分位(1% / 5% / 25% / 50% / 75% / 99% / 99.9%)",
        " / ".join(f"{v:.3f}" for v in np.quantile(q, [.01, .05, .25, .5, .75, .99, .999])), f"共 {q.size} 格")
    put("新/旧", "变化 > 0.1% 的格子占比;其中新 < 旧 / 新 > 旧",
        f"{(np.abs(q - 1) > 1e-3).mean():.1%};{(q < 0.999).mean():.1%} / {(q > 1.001).mean():.2%}")
    put("新/旧", "新 / 旧 < 1/1.5、< 1/2 的格子占比", f"{(q < 1 / 1.5).mean():.1%} / {(q < 0.5).mean():.1%}")
    nan0, nan1 = mv0.isna().to_numpy(), mv1.isna().to_numpy()
    put("新/旧", "NaN 位置是否完全相同", bool((nan0 == nan1).all()), f"NaN 格 {int(nan0.sum())}")
    chg_st = ((mv1 / mv0 - 1).abs() > 1e-3).any()
    put("新/旧", "有格子变化 > 0.1% 的股票数", int(chg_st.sum()), f"/ {int(mv0.notna().any().sum())} 只有市值的股票")
    # 截面市值排名的变化(中证1000 代理与 Size 都靠排名)
    rk0, rk1 = mv0.rank(axis=1, pct=True), mv1.rank(axis=1, pct=True)
    sp = rk0.corrwith(rk1, axis=1)
    put("新/旧", "逐日截面市值秩相关(中位 / 最小)", f"{sp.median():.4f} / {sp.min():.4f}")
    dr = (rk1 - rk0).abs().to_numpy()
    dr = dr[np.isfinite(dr)]
    put("新/旧", "截面分位变化 |Δpct| 中位 / 90% / 99% 分位", " / ".join(f"{v:.3f}" for v in np.quantile(dr, [.5, .9, .99])))
    dec0 = np.ceil(rk0 * 10).to_numpy()
    dec1 = np.ceil(rk1 * 10).to_numpy()
    ok = np.isfinite(dec0) & np.isfinite(dec1)
    put("新/旧", "市值十分位改变的格子占比", f"{(dec0[ok] != dec1[ok]).mean():.1%}")
    pd.DataFrame(rows).to_csv(LEDGER / "mvfix_extrapolation.csv", index=False)

    # ---- 2. 送转识别 vs 复权因子标签(在 daily_basic 完整日历上)
    db = ND._load_daily_basic()
    cal = pd.DatetimeIndex(db["trade_date"].unique()).sort_values()
    allc = pd.Index(sorted(set(db["ts_code"].unique()) | set(codes)))
    ev, _ = float_events(cal, allc)
    ev.to_csv(LEDGER / "mvfix_songzhuan_events.csv", index=False)
    ev["rbin"] = rbin(ev["r"].to_numpy())
    e5 = ev[(ev["r"] >= 1.05) & ev["adj_avail"]]
    cm = []
    for b, g in list(e5.groupby("rbin")) + [("合计", e5)]:
        tp = int((g.is_songzhuan_rule & g.adj_label).sum())
        fp = int((g.is_songzhuan_rule & ~g.adj_label).sum())
        fn = int((~g.is_songzhuan_rule & g.adj_label).sum())
        tn = int((~g.is_songzhuan_rule & ~g.adj_label).sum())
        rr = g["report_ratio_after_before"]
        flat = (np.abs(np.log(rr)) < 0.02)
        cm.append(dict(r_bin=b, n=len(g), TP=tp, FP=fp, FN=fn, TN=tn,
                       recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                       legacy_extrapolated=len(g),            # 旧口径:每一次跳升都乘进总股本
                       report_flat_share_among_TN=float(flat[~g.is_songzhuan_rule & ~g.adj_label].mean()),
                       report_flat_share_among_FP=float(flat[g.is_songzhuan_rule & ~g.adj_label].mean()) if fp else np.nan,
                       sum_logr_FP=float(np.log(g.r[g.is_songzhuan_rule & ~g.adj_label]).sum()),
                       sum_logr_FN=float(np.log(g.r[~g.is_songzhuan_rule & g.adj_label]).sum())))
    cm = pd.DataFrame(cm)
    cm.to_csv(LEDGER / "mvfix_songzhuan_confusion.csv", index=False)
    print(cm.to_string())

    # ---- 3. 期后送转是否已反映在该期股本里(决定锚点用报告期末还是生效日)
    rep = FU._final_rows(FU.build_report_level()).dropna(subset=["total_share"])
    g = {c: x.sort_values("end_date") for c, x in rep.groupby("stock_code")}
    conf = ev[ev.adj_label & (ev.r >= 1.1)]
    at = []
    for c, d, r in conf[["code", "date", "r"]].itertuples(index=False):
        x = g.get(c)
        if x is None:
            continue
        prev = x[(x.end_date < d) & (x.avail < d)]
        if len(prev) == 0:
            continue
        s0 = prev.total_share.iloc[-1]
        for _, row in x[(x.end_date < d) & (x.avail >= d)].iterrows():
            qq = row.total_share / s0
            kind = ("期末口径(未反映)" if abs(np.log(qq)) < 0.02 else
                    "已反映期后送转" if abs(np.log(qq / r)) < 0.02 else "其它(期内另有股本变化)")
            at.append(dict(code=c, exdate=d, r=r, end_date=row.end_date, avail=row.avail, s_prev=s0,
                           s_report=row.total_share, ratio=qq, kind=kind))
    at = pd.DataFrame(at)
    at.to_csv(LEDGER / "mvfix_anchor_timing.csv", index=False)
    print(at["kind"].value_counts())

    # ---- 3b. 新股:"报告期末早于上市日"的财报是不是发行前股本(决定新股加法与"已含发行股份"判据)
    li = ND.listing_info(codes)["list_date"]
    fin = rep.copy()
    qtab = fin.pivot_table(index="qi", columns="stock_code", values="total_share", aggfunc="last")
    ip = []
    for c in codes:
        L = li.get(c)
        f = fs[c].dropna() if c in fs.columns else pd.Series(dtype=float)
        if pd.isna(L) or L <= dates[0] or not len(f) or c not in qtab.columns:
            continue
        lqi = L.year * 4 + L.quarter - 1
        s_first = qtab[c].get(lqi, np.nan)
        if not np.isfinite(s_first):
            continue
        x = g.get(c)
        pre = x[x.end_date < L].sort_values("avail")
        at_listing = pre[pre.avail < L].tail(1)                       # 上市当天生效的那份
        after = pre[pre.avail >= L]                                   # 上市之后才公告的上市前报告期
        for kind, rows_ in [("上市当天生效", at_listing), ("上市后公告", after)]:
            for _, r_ in rows_.iterrows():
                ip.append(dict(code=c, list_date=L, kind=kind, end_date=r_.end_date, avail=r_.avail,
                               share=r_.total_share, s_first=s_first, first_float=float(f.iloc[0]),
                               includes_ipo=bool(abs(np.log(r_.total_share / s_first)) < 0.01)))
    ip = pd.DataFrame(ip)
    ip.to_csv(LEDGER / "mvfix_ipo_reports.csv", index=False)
    print(ip.groupby("kind")["includes_ipo"].agg(["size", "sum", "mean"]))
    # ---- 4. 事后真值对照(只用于评估:用到了 t 之后公告的财报)
    truth, kind, ipo_truth = expost_truth(dates, codes, ev, sh)
    tr = []
    for tag, shx in [("旧", mv0 / rc), ("新", mv1 / rc), ("PIT 总股本(不外推)", sh.where(sh.notna(), fs))]:
        for kd in ["全部", "平稳季度", "单次送转季度", "新股窗口"]:
            if kd == "新股窗口":
                e = np.log(shx / ipo_truth).to_numpy()
                m = np.isfinite(e)
            else:
                e = np.log(shx / truth).to_numpy()
                m = np.isfinite(e) & ((kind.to_numpy() != "") if kd == "全部" else (kind.to_numpy() == kd))
            x = e[m]
            if not x.size:
                continue
            tr.append(dict(method=tag, cells=kd, n=x.size, med_abs_log_err=float(np.median(np.abs(x))),
                           mean_abs_log_err=float(np.mean(np.abs(x))),
                           share_within_1pct=float((np.abs(x) < 0.01).mean()),
                           share_within_5pct=float((np.abs(x) < np.log(1.05)).mean()),
                           share_over_1p5x=float((x > np.log(1.5)).mean()),
                           share_under_1_over_1p5=float((x < -np.log(1.5)).mean()),
                           share_over_1p05x=float((x > np.log(1.05)).mean()),
                           share_under_0p95x=float((x < np.log(0.95)).mean())))
    tr = pd.DataFrame(tr)
    cov = pd.DataFrame([dict(method="覆盖", cells=k, n=int((kind == k).to_numpy().sum())) for k in ["平稳季度", "单次送转季度"]]
                       + [dict(method="覆盖", cells="新股窗口", n=int(ipo_truth.notna().to_numpy().sum()),
                               n_stocks=int(ipo_truth.notna().any().sum()))]
                       + [dict(method="覆盖", cells="有真实市值的格子", n=int(mv0.notna().to_numpy().sum()))])
    pd.concat([tr, cov]).to_csv(LEDGER / "mvfix_truth_eval.csv", index=False)
    print(tr.to_string())

    # ---- 5. 个股案例
    stage_cases(P, truth.fillna(ipo_truth))
    log("eval 完成")


def expost_truth(dates, codes, ev, sh):
    """事后真值(仅评估用):t 所在季度的期初、期末最终版股本 s_lo / s_hi。
      平稳季度    |log(s_hi / s_lo)| < 0.1%           -> 真值 = s_lo
      单次送转季度 季内恰有一次复权因子确认的送转(比例 r),且 s_hi = s_lo × r(0.5% 内)
                  -> 除权日之前 s_lo,之后 s_hi(除权日取复权因子跳变日,沪市常比流通股本早一天)
    其余格子无真值。
    新股窗口(单独一套真值):上市首日起、到 PIT 股本第一次等于"上市所在季度期末的最终版股本"s_first 之前,
    真值 = s_first(这段时间生效的财报是发行前股本;期内若另有股本变化会混进误差,量级小)。"""
    rep = FU._final_rows(FU.build_report_level()).dropna(subset=["total_share"])
    q = rep.pivot_table(index="qi", columns="stock_code", values="total_share", aggfunc="last").reindex(columns=codes)
    qi_t = (dates.year * 4 + dates.quarter - 1).to_numpy()
    lo = q.reindex(qi_t - 1).to_numpy(np.float64)
    hi = q.reindex(qi_t).to_numpy(np.float64)
    truth = np.full(lo.shape, np.nan)
    kind = np.full(lo.shape, "", dtype=object)
    flat = np.abs(np.log(hi / lo)) < 1e-3
    truth[flat] = lo[flat]
    kind[flat] = "平稳季度"
    # 单次送转季度
    sz = ev[ev.adj_label & ev.code.isin(codes)].copy()
    sz = sz[sz.date.between(dates[0], dates[-1])]
    # 除权日 = 复权因子跳变日(t 或 t-1)
    adj_day = np.where(np.isfinite(sz.lp_t_1) & (sz.lp_t_1 < sz.lp_t), 1, 0)       # 价格先跌的一天
    pos = dates.get_indexer(sz.date) - adj_day
    sz["pos"] = np.maximum(pos, 0)
    sz["qi"] = (sz.date.dt.year * 4 + sz.date.dt.quarter - 1).to_numpy()
    cnt = sz.groupby(["code", "qi"]).size()
    one = sz.set_index(["code", "qi"]).loc[cnt[cnt == 1].index].reset_index()
    cidx = codes.get_indexer(one.code)
    for (c, qq, r, p0), j in zip(one[["code", "qi", "r", "pos"]].itertuples(index=False), cidx):
        rows = np.nonzero(qi_t == qq)[0]
        if not len(rows):
            continue
        a, b = lo[rows[0], j], hi[rows[0], j]
        if not (np.isfinite(a) and np.isfinite(b)) or abs(np.log(b / (a * r))) > 0.005:
            continue
        pre = rows[rows < p0]
        post = rows[rows >= p0]
        truth[pre, j] = a
        truth[post, j] = b
        kind[rows, j] = "单次送转季度"
    li = ND.listing_info(codes)["list_date"]
    ipo = np.full(lo.shape, np.nan)
    shv = sh.to_numpy(np.float64)
    for j, ld in enumerate(li.to_numpy()):
        if pd.isna(ld) or ld <= np.datetime64(dates[0]) or codes[j] not in q.columns:
            continue
        lqi = int(pd.Timestamp(ld).year * 4 + pd.Timestamp(ld).quarter - 1)
        s_first = q[codes[j]].get(lqi, np.nan)
        if not np.isfinite(s_first):
            continue
        i0 = int(np.searchsorted(dates.values, ld))
        hit = np.nonzero(np.abs(np.log(shv[i0:, j] / s_first)) <= 1e-3)[0]
        i1 = i0 + (hit[0] if len(hit) else 0)                  # 从未等于 s_first 的不计(窗口不明确)
        ipo[i0:i1, j] = s_first
    return (pd.DataFrame(truth, index=dates, columns=codes), pd.DataFrame(kind, index=dates, columns=codes),
            pd.DataFrame(ipo, index=dates, columns=codes))


CASES = {
    "600339.SH": "审查例:总股本 = 流通股本 55.83 亿,旧口径全程 3.60 倍",
    "600390.SH": "审查例:旧口径 7.30 倍",
    "600519.SH": "审查例:无变化",
    "002594.SZ": "A+H;2025-07-29 送转 10 送 8 转 12(3 倍)",
    "000617.SZ": "期后送转已反映在该期股本里(锚点改为生效日)",
    "601777.SH": "重整转增 3.5 倍、价格不除权:由流通股本下界接住",
    "301165.SZ": "新股:解禁外推(fixb 第二轮恢复外推的 11 只之一)",
    "688802.SH": "新股(2025-12-17 上市):招股书股本 + 首日流通股本",
    "301275.SZ": "新股(2025 上市):fixb 文档的上市窗口例子",
}


def stage_cases(P, truth):
    mv0, mv1, rc, fs, sh = P["mv_legacy"], P["mv_new"], P["raw_close"], P["float_share"], P["pit_share"]
    out = []
    for c, note in CASES.items():
        if c not in mv0.columns:
            continue
        d = mv0[c].dropna().index
        if not len(d):
            continue
        picks = sorted(set([d[0], d[len(d) // 4], d[len(d) // 2], d[3 * len(d) // 4], d[-1]]))
        if c == "002594.SZ":
            picks = sorted(set(picks + list(d[(d >= "2025-07-25") & (d <= "2025-08-05")]) + list(d[(d >= "2025-08-28") & (d <= "2025-09-03")])))
        for t in picks:
            out.append(dict(code=c, date=t.date(), note=note, raw_close=float(rc.at[t, c]),
                            pit_total_share_yi=float(sh.at[t, c]) / 1e8, float_share_yi=float(fs.at[t, c]) / 1e8,
                            expost_truth_share_yi=float(truth.at[t, c]) / 1e8,
                            mv_legacy_yi=float(mv0.at[t, c]) / 1e8, mv_new_yi=float(mv1.at[t, c]) / 1e8,
                            px_x_truth_yi=float(rc.at[t, c] * truth.at[t, c]) / 1e8))
    out = pd.DataFrame(out)
    out.to_csv(LEDGER / "mvfix_cases.csv", index=False)
    print(out.to_string())


# ====================================================================== barra
def stage_barra():
    """Barra 风格暴露(风险模型默认 v3 参数,与 fixb_cutoff_impact.py 的口径相同):legacy vs new。"""
    from afactor.panel import Panel
    from afactor import barra as BR, factors as FA
    FA.build_all()
    res = {}
    for method in ["legacy", "songzhuan"]:
        ND.TOTAL_SHARE_METHOD = method
        p = Panel()                                      # 新面板:factors._real 的缓存挂在面板对象上
        dates, codes = p.dates, p.codes
        rm = BR.RiskModel()
        u = p["universe"].reindex(index=dates, columns=codes).fillna(False)
        cover = BR.coverage_mask(p).reindex(index=dates, columns=codes).fillna(False)
        X = BR.build_exposures(p, u, cover=cover, raw_clip=rm.params["raw_clip"], size_clip=rm.params["size_clip"])
        d = OUT / f"barra_{method}"
        d.mkdir(parents=True, exist_ok=True)
        for k, v in X.items():
            v.reindex(index=dates, columns=codes).astype(np.float32).to_parquet(d / f"{k.split()[0]}.parquet")
        cover.astype(np.float32).to_parquet(d / "cover.parquet")
        res[method] = ({k.split()[0]: v.reindex(index=dates, columns=codes) for k, v in X.items()}, cover, u)
        log(f"暴露 {method}")
    ND.TOTAL_SHARE_METHOD = "songzhuan"
    (X0, cov0, u), (X1, cov1, _) = res["legacy"], res["songzhuan"]
    mv1 = pd.read_parquet(OUT / "mv_new.parquet") if (OUT / "mv_new.parquet").exists() else None
    rows = []
    # 与 fixb 缓存的一致性(legacy 应逐位等于 cache/_tmp/fixb_impact/new/barra_*.parquet,若存在)
    ref = Path(os.environ.get("MVFIX_FIXB_REF", C.TMP / "fixb_impact" / "new"))    # fixb 留下的当前代码面板(若有)
    for k in X0:
        a, b = X0[k].where(cov0 > 0), X1[k].where(cov1 > 0)
        av, bv = a.to_numpy(np.float64), b.to_numpy(np.float64)
        both = np.isfinite(av) & np.isfinite(bv)
        dd = np.abs(av - bv)[both]
        rc = a.corrwith(b, axis=1)                      # 逐日截面相关(覆盖池)
        ra = a.rank(axis=1).corrwith(b.rank(axis=1), axis=1)
        ur = a.where(u).rank(axis=1).corrwith(b.where(u).rank(axis=1), axis=1)
        row = dict(factor=k, n_cells=int(both.sum()), share_changed_1e3=float((dd > 1e-3).mean()),
                   share_abs_gt_0p1=float((dd > 0.1).mean()), share_abs_gt_0p5=float((dd > 0.5).mean()),
                   q50_abs=float(np.quantile(dd, .5)), q99_abs=float(np.quantile(dd, .99)), max_abs=float(dd.max()),
                   xs_corr_median=float(rc.median()), xs_corr_min=float(rc.min()),
                   xs_rankcorr_median=float(ra.median()), xs_rankcorr_min=float(ra.min()),
                   estu_rankcorr_median=float(ur.median()), nan_flip=int((np.isfinite(av) != np.isfinite(bv)).sum()))
        f = ref / f"barra_{k}.parquet"
        if f.exists():
            r0 = pd.read_parquet(f).reindex(index=a.index, columns=a.columns).where(cov0 > 0).to_numpy(np.float64)
            row["legacy_vs_fixb_cache_maxabs"] = float(np.nanmax(np.abs(r0 - av))) if np.isfinite(r0).any() else np.nan
        rows.append(row)
    # Size 按新市值十分位的平均变化(谁的暴露动了)
    if mv1 is not None:
        dec = np.ceil(mv1.where(cov1 > 0).rank(axis=1, pct=True) * 10)
        for k in ["Size", "NonlinSize"]:
            dlt = (X1[k] - X0[k]).where(cov1 > 0)
            for q in range(1, 11):
                m = dec == q
                v = dlt.where(m).to_numpy()
                v = v[np.isfinite(v)]
                rows.append(dict(factor=f"{k} 按新市值十分位 D{q}", n_cells=v.size, q50_abs=float(np.median(v)),
                                 share_abs_gt_0p1=float((np.abs(v) > 0.1).mean()),
                                 share_abs_gt_0p5=float((np.abs(v) > 0.5).mean())))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / "mvfix_barra_delta.csv", index=False)
    print(out.to_string())
    log("barra 完成")


# ====================================================================== shadow
SHADOW_LINKS_RO = ["data_8:21", "new_data", "stock_sw_industry.csv"]       # 只读输入:软链
# 与真实市值无关、下游脚本只读不建的缓存(拷贝进影子;comp_v5 打分、容量的策略日收益、指增的 IC 序列)
SHADOW_SEED_TMP = ["comp_v5_score_open10.parquet", "ie_ic_comp.parquet", "ie_ic_gbt.parquet",
                   "cap_daily_open10.parquet", "cap_daily_open10_exante.parquet", "cap_daily_vwap1.parquet",
                   "cap_daily_vwap1_exante.parquet", "cap_cells_open10.parquet", "cap_cells_open10_exante.parquet",
                   "cap_cells_vwap1.parquet", "cap_cells_vwap1_exante.parquet"]
SHADOW_STEPS = {
    "proxy": [["scripts/run_index_proxy.py"]],
    "risk": [["scripts/run_risk_model.py", "--stage", "all", "--force"]],
    "cap": [["scripts/cap_capacity_curve.py", "bench"], ["scripts/cap_capacity_curve.py", "checks"],
            ["scripts/cap_capacity_curve.py", "summarize"]],
    "extra": [["scripts/mvfix_shadow_extra.py"]],
    "ie": [["scripts/run_index_enhance.py", "--stage", "all", "--procs", "2"]],   # 两个影子同时跑:各 2 个子进程
}


def _sub(method):
    return "legacy" if method == "legacy" else "new"


def stage_shadow():
    """两个影子根目录:代码 = 本项目 src/scripts 的拷贝,只差 TOTAL_SHARE_METHOD 默认值;只读数据软链,
    cache/_tmp、research/ledger、reports、logs 各自独立(ledger 是本项目的拷贝,下游脚本的读-改-写只落在影子里)。
    种子缓存从 MVFIX_SEED_TMP(默认本项目 cache/_tmp)拷贝。"""
    root = C.ROOT
    seed = Path(os.environ.get("MVFIX_SEED_TMP", C.TMP))
    for method in ["legacy", "songzhuan"]:
        sr = SHADOW / _sub(method)
        if sr.exists() and not ARGS.refresh:
            print(f"已存在 {sr}(--refresh 重建)")
            continue
        if sr.exists():
            shutil.rmtree(sr)
        sr.mkdir(parents=True)
        for d in ["src", "scripts"]:
            shutil.copytree(root / d, sr / d, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
        nd = sr / "src" / "afactor" / "newdata.py"
        s = nd.read_text()
        key = 'TOTAL_SHARE_METHOD = "songzhuan"'
        assert key in s
        nd.write_text(s.replace(key, f'TOTAL_SHARE_METHOD = "{method}"'))
        for n in SHADOW_LINKS_RO:
            os.symlink(os.path.realpath(root / n), sr / n)
        (sr / "cache" / "_tmp").mkdir(parents=True)
        os.symlink(os.path.realpath(root / "cache" / "daily"), sr / "cache" / "daily")
        shutil.copytree(root / "research" / "ledger", sr / "research" / "ledger", symlinks=True,
                        ignore=shutil.ignore_patterns("mvfix_*"))
        for n in SHADOW_SEED_TMP:
            if (seed / n).exists():
                shutil.copy2(seed / n, sr / "cache" / "_tmp" / n)
            else:
                print(f"  ⚠ 缺种子 {seed / n}")
        for d in ["reports", "logs"]:
            (sr / d).mkdir()
        print(f"建好 {sr}")


def stage_shadow_run():
    """在影子根目录里按顺序跑下游脚本(cwd = 影子根;脚本自己把 ../src 放进 sys.path,用的是影子里的代码)。"""
    import subprocess
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.setdefault("INDEX_DIR", str(ND.INDEX_DIR))
    sr = SHADOW / ARGS.which
    assert (sr / "src").exists(), f"先跑 --stage shadow({sr} 不存在)"
    for step in ARGS.steps.split(","):
        for cmd in SHADOW_STEPS[step]:
            lg = sr / "logs" / f"mvfix_{step}_{os.path.basename(cmd[0]).split('.')[0]}_{'_'.join(cmd[1:2])}.log"
            log(f"[{ARGS.which}] {' '.join(cmd)} -> {lg.name}")
            with open(lg, "w") as fh:
                rc = subprocess.call([sys.executable] + cmd, cwd=sr, env=env, stdout=fh, stderr=subprocess.STDOUT)
            if rc != 0:
                raise SystemExit(f"[{ARGS.which}] {' '.join(cmd)} 失败(退出码 {rc}),见 {lg}")
    log(f"[{ARGS.which}] {ARGS.steps} 完成")


# ====================================================================== downstream
DOWNSTREAM_FILES = {
    "RISK_MODEL": ["risk_bias_stats.csv", "risk_bias_diagnostics.csv", "risk_attribution_comp_v5.csv",
                   "risk_attribution_comp_v5_factors.csv", "risk_move_check.csv"],
    "CSI1000_PROXY": ["proxy_validation.csv", "proxy_membership_stats.csv", "proxy_gap_attribution.csv",
                      "proxy_survivor_check.csv", "proxy_engine_checks.csv", "mvfix_shadow_proxy_rows.csv"],
    "CAPACITY": ["capacity_grid.csv", "capacity_summary.csv", "capacity_checks.csv", "capacity_decomp.csv",
                 "capacity_diagnostics.csv", "capacity_scaling.csv"],
    "INDEX_ENHANCEMENT": ["ie_variants.csv", "ie_paired.csv", "ie_yearly.csv", "ie_exposures.csv",
                          "ie_attribution.csv", "ie_bench_compare.csv", "ie_engine_checks.csv", "ie_solver_log.csv"],
}


def _read(path):
    return pd.read_csv(path) if path.exists() else None


def _num_diff(a: pd.DataFrame, b: pd.DataFrame):
    """逐行对齐(两边代码相同、行序确定)后数值列的差;非数值列必须相同。"""
    if a is None or b is None:
        return dict(status="缺文件")
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return dict(status=f"形状不同 {a.shape} vs {b.shape}")
    obj = [c for c in a.columns if not pd.api.types.is_numeric_dtype(a[c])]
    key_same = all(a[c].fillna("").astype(str).equals(b[c].fillna("").astype(str)) for c in obj)
    num = [c for c in a.columns if c not in obj]
    x, y = a[num].to_numpy(np.float64), b[num].to_numpy(np.float64)
    both = np.isfinite(x) & np.isfinite(y)
    d = np.abs(x - y)[both]
    rel = (np.abs(x - y) / np.maximum(np.abs(x), 1e-12))[both]
    return dict(status="ok" if key_same else "文本列不同", n_numeric=int(both.sum()),
                n_changed=int((d > 1e-9 * np.maximum(1, np.abs(x[both]))).sum()),
                max_abs_diff=float(d.max()) if d.size else 0.0, q90_rel_diff=float(np.quantile(rel, .9)) if rel.size else 0.0,
                nan_flip=int((np.isfinite(x) != np.isfinite(y)).sum()))


CORE = ["2021-2025", "2021-2023", "2024-2025"]
IE_WIN = ["全期", "开发窗2021-23", "检验窗2024-25", "2023(样本外首年)"]
HEADLINES = [
    # (报告, 项目, 文件, 过滤条件, 列)
    *[("RISK_MODEL", f"偏差 b:{pf} {y}", "risk_bias_stats.csv", dict(model="v3", portfolio=pf, year=y), "bias")
      for pf in ["Country", "__风格平均", "__行业平均", "Size 市值", "NonlinSize 非线性市值", "median_of_200",
                 "V1000 池等权(总风险)", "COMP_v5 排雷20/35 目标 − 池等权", "B×GBT Top20 目标 − 池等权"]
      for y in CORE + ["2024"]],
    *[("RISK_MODEL", f"诊断:{t} 2021-2025", "risk_bias_diagnostics.csv", dict(model="v3", test=t, year="2021-2025"), "bias")
      for t in ["comp_v5_active|因子部分", "comp_v5_active|特质部分"]],
    ("RISK_MODEL", "诊断:20 日反转解释 COMP 主动特质方差占比 2021-2025", "risk_bias_diagnostics.csv",
     dict(model="v3", test="missing_factor|COMP_v5 主动特质收益", portfolio="rev20", year="2021-2025"), "value"),
    *[("RISK_MODEL", f"COMP_v5 持仓归因 {c} 2021-2025", "risk_attribution_comp_v5.csv", dict(model="v3", year="2021-2025"), c)
      for c in ["total_ann", "style_ann", "industry_ann", "specific_ann", "specific_t_nw5"]],
    *[("RISK_MODEL", f"COMP_v5 风格贡献 {f} 2021-2025", "risk_attribution_comp_v5_factors.csv", dict(factor=f), "2021-2025")
      for f in ["Size 市值", "NonlinSize 非线性市值", "Liquidity 流动性", "BookToPrice 价值"]],
    *[("CSI1000_PROXY", f"{ix} main 价格口径 {per} {c}", "proxy_validation.csv",
       dict(index=ix, variant="main", kind="pr", timing="close_to_close", period=per), c)
      for ix in [852, 905, 300] for per in ["2021-2025", "dev_2021-2023", "test_2024-2025"]
      for c in ["corr_daily", "te_ann", "gap_ann_geom", "t_nw_gap", "beta_proxy_on_index"]],
    *[("CSI1000_PROXY", f"852 main 价格口径 {y} {c}", "proxy_validation.csv",
       dict(index=852, variant="main", kind="pr", timing="close_to_close", period=y), c)
      for y in ["2021", "2022", "2023", "2024", "2025"] for c in ["gap_ann_geom", "te_ann"]],
    *[("CSI1000_PROXY", f"852 权重方案 {v} 2021-2025 {c}", "proxy_validation.csv",
       dict(index=852, variant=v, kind="pr", timing="close_to_close", period="2021-2025"), c)
      for v in ["w_total", "w_banded", "w_equal"] for c in ["te_ann", "gap_ann_geom"]],
    ("CSI1000_PROXY", "852 main 全收益 2021-2025 ann_proxy", "proxy_validation.csv",
     dict(index=852, variant="main", kind="tr", timing="close_to_close", period="2021-2025"), "ann_proxy"),
    *[("CSI1000_PROXY", f"852 收益差暴露回归 {c}", "proxy_gap_attribution.csv", dict(index=852, period="2021-2025"), c)
      for c in ["alpha_ann", "t_alpha", "b_size", "t_b_size"]],
    *[("CSI1000_PROXY", f"§7 推荐基准 {pr} 2021-2025 {c}", "mvfix_shadow_proxy_rows.csv", dict(pair=pr, period="2021-2025"), c)
      for pr in ["proxy_exec_vs_index", "pool_r5_vs_proxy_exec", "pool_d1_vs_proxy_exec"]
      for c in ["ann_a", "ann_b", "gap_ann_geom", "t_nw_gap", "te_ann"]],
    *[("CAPACITY", f"{st} {md} 1000万 {c}", "capacity_summary.csv",
       dict(strategy=st, mode=md, variant="Y1.0", sizing="pov", window=w), c)
      for st, w in [("COMP_v5_exclude", "2021-2025"), ("COMP_v5_top20", "2021-2025"), ("COMP_v5_top50", "2021-2025"),
                    ("B_gbt_top20", "2023-2025")] for md in ["open10", "vwap1"]
      for c in ["excess_pool_1e7", "cap_be_pool", "cap_be_pool_flag", "excess_csi1000p_1e7", "cap_be_csi1000p",
                "cap_be_csi1000p_flag", "cap_half_csi1000p", "cap_half_csi1000p_flag", "excess_cashcsi_1e7",
                "cap_be_cashcsi", "cap_be_cashcsi_flag"]],
    *[("CAPACITY", f"{st} {md} {aum:.0e} 网格 {c}", "capacity_grid.csv",
       dict(strategy=st, mode=md, variant="Y1.0", sizing="pov", window="2021-2025", window_role="full", aum=float(aum)), c)
      for st in ["COMP_v5_exclude"] for md in ["open10", "vwap1"] for aum in [1e7, 1e10]
      for c in ["excess_pool", "bench_csi1000p_ann", "excess_csi1000p", "t_nw_csi1000p"]],
    *[("CAPACITY", f"{st} {md} legacy 成本 现金按代理补仓 {c}", "capacity_summary.csv",
       dict(strategy=st, mode=md, variant="legacy", sizing="pov", window="2021-2025"), c)
      for st in ["COMP_v5_top20", "COMP_v5_top50"] for md in ["open10", "vwap1"] for c in ["cap_be_cashcsi", "cap_be_cashpool"]],
    *[("INDEX_ENHANCEMENT", f"{v} {w} {c}", "ie_variants.csv", dict(variant=v, window=w), c)
      for v in ["V1", "V1_ph0", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "CAP1e7", "CAP1e8", "CAP1e9", "CAP5e9",
                "CTRL_excl", "CTRL_top20"] for w in IE_WIN for c in ["excess_geo", "te", "ir", "t_nw"]],
    ("INDEX_ENHANCEMENT", "V1 事前 TE 均值", "ie_variants.csv", dict(variant="V1", window="全期"), "te_ex_ante_mean"),
    *[("INDEX_ENHANCEMENT", f"评价基准 {b} {md} {w} ann_geo", "ie_bench_compare.csv", dict(kind="bench", mode=md, bench=b, window=w), "ann_geo")
      for b in ["exec", "legacy"] for md in ["open10", "vwap1"] for w in ["全期", "开发窗2021-23", "检验窗2024-25"]],
    *[("INDEX_ENHANCEMENT", f"{v} {w} ann_ret", "ie_variants.csv", dict(variant=v, window=w), "ann_ret")
      for v in ["V1", "V1_ph0", "V7", "CTRL_excl", "CAP1e7"] for w in ["全期", "2023(样本外首年)"]],
    *[("INDEX_ENHANCEMENT", f"V1 {w} 偏差统计量 b {c}", "ie_exposures.csv",
       dict(variant="V1", window=w, source="TE", factor="偏差统计量 b(5日,不重叠,5起点均值)"), c)
      for w in ["全期", "开发窗2021-23", "检验窗2024-25"] for c in ["ci_lo", "ci_hi"]],
    *[("INDEX_ENHANCEMENT", f"V1 全期归因 {comp}", "ie_attribution.csv", dict(variant="V1", window="全期", component=comp), "ann_pp")
      for comp in ["total", "style", "industry", "specific", "rev20", "specific_ex_rev20", "cost", "engine_excess",
                   "style:Size 市值", "style:NonlinSize 非线性市值", "style:Liquidity 流动性"]],
    *[("INDEX_ENHANCEMENT", f"{v} 全期归因 {comp}", "ie_attribution.csv", dict(variant=v, window="全期", component=comp), "ann_pp")
      for v in ["V8", "CTRL_excl"] for comp in ["specific", "specific_ex_rev20", "style:Size 市值", "style:NonlinSize 非线性市值"]],
    *[("INDEX_ENHANCEMENT", f"{v} 全期实际主动暴露 {f} {c}", "ie_exposures.csv",
       dict(variant=v, window="全期", source="实际持仓(逐日,相位合并)", factor=f), c)
      for v in ["V1", "CHK_pool_ew", "CTRL_excl"] for f in ["Size 市值", "NonlinSize 非线性市值"] for c in ["mean", "frac_over_bound"]],
    *[("INDEX_ENHANCEMENT", f"V1 {w} TE {f}", "ie_exposures.csv", dict(variant="V1", window=w, source="TE", factor=f), "mean")
      for w in ["全期", "开发窗2021-23", "检验窗2024-25"] for f in ["事前TE(实际持仓,均值)", "偏差统计量 b(5日,不重叠,5起点均值)"]],
    *[("INDEX_ENHANCEMENT", f"配对 {v} − V1_ph0 {w}", "ie_paired.csv", dict(variant=v, vs="V1_ph0", window=w), c)
      for v in ["V6", "V8"] for w in ["全期", "开发窗2021-23", "检验窗2024-25"] for c in ["diff_arith_ann", "t_nw"]],
]


def _pick(df, filt, col):
    if df is None or col not in df.columns:
        return np.nan
    m = np.ones(len(df), bool)
    for k, v in filt.items():
        if k not in df.columns:
            return np.nan
        m &= (df[k].astype(str) == str(v)).to_numpy()
    x = df.loc[m, col]
    return float(x.iloc[0]) if len(x) == 1 and pd.api.types.is_number(x.iloc[0]) else (np.nan if len(x) != 1 else x.iloc[0])


def headline_table(L, N):
    cache = {}

    def rd(root, f):
        k = (str(root), f)
        if k not in cache:
            cache[k] = _read(root / f)
        return cache[k]
    rows = []
    for rpt, item, f, filt, col in HEADLINES:
        o = _pick(rd(LEDGER, f), filt, col)
        a = _pick(rd(L / "research" / "ledger", f), filt, col)
        b = _pick(rd(N / "research" / "ledger", f), filt, col)
        num = all(isinstance(x, (int, float, np.floating)) and not isinstance(x, bool) for x in (a, b))
        rows.append(dict(report=rpt, item=item, file=f, column=col, original=o, legacy=a, new=b,
                         delta=(b - a) if num else np.nan))
    return pd.DataFrame(rows)


def stage_downstream():
    """汇总影子根目录(legacy / new)的重跑结果:逐文件差异、关键数字、代理成分与权重、风险模型规模。
    另把 legacy 影子与本项目原账本对照(说明原账本能否由当前代码复现,以及差在哪里)。"""
    L, N = SHADOW / "legacy", SHADOW / "new"
    rows = []
    for rpt, files in DOWNSTREAM_FILES.items():
        for f in files:
            a, b = _read(L / "research" / "ledger" / f), _read(N / "research" / "ledger" / f)
            o = _read(LEDGER / f)
            rows.append(dict(report=rpt, file=f, compare="legacy 影子 -> new 影子", **_num_diff(a, b)))
            rows.append(dict(report=rpt, file=f, compare="原账本 -> legacy 影子", **_num_diff(o, a)))
    files = pd.DataFrame(rows)
    files.to_csv(LEDGER / "mvfix_downstream_files.csv", index=False)
    print(files.to_string())
    hl = headline_table(L, N)
    hl.to_csv(LEDGER / "mvfix_downstream_headlines.csv", index=False)
    print(hl.to_string())
    # 代理成分的市值分布(CSI1000:每期成分日均总市值的下限 / 中位 / 上限,亿元)
    ms = []
    for tag, root in [("原账本", LEDGER), ("legacy", L / "research" / "ledger"), ("new", N / "research" / "ledger")]:
        m = _read(root / "proxy_membership_stats.csv")
        m = m[m["index"].astype(str).str.zfill(6) == "000852"]
        for c in ["mv_avg_min_yi", "mv_avg_med_yi", "mv_avg_max_yi", "eff_n", "w_max", "rebalance_turnover_oneway", "n_mv_fallback"]:
            ms.append(dict(source=tag, column=c, min=float(m[c].min()), max=float(m[c].max()), mean=float(m[c].mean())))
    pd.DataFrame(ms).to_csv(LEDGER / "mvfix_downstream_proxy_members.csv", index=False)
    print(pd.DataFrame(ms).to_string())

    # 代理成分:每期 legacy vs new 的成员重合、换入换出股票的市值特征
    mem = []
    ml, mn = pd.read_parquet(L / "cache" / "_tmp" / "proxy_membership.parquet"), pd.read_parquet(N / "cache" / "_tmp" / "proxy_membership.parquet")
    for (rv, ix), g in ml.groupby(["review", "index"], sort=False):
        h = mn[(mn.review == rv) & (mn["index"] == ix)]
        sa, sb = set(g.code), set(h.code)
        mem.append(dict(review=rv, index=ix, n_legacy=len(sa), n_new=len(sb), overlap=len(sa & sb) / max(len(sa), 1),
                        n_out=len(sa - sb), n_in=len(sb - sa)))
    mem = pd.DataFrame(mem)
    mem.to_csv(LEDGER / "mvfix_downstream_membership.csv", index=False)
    print(mem.groupby("index")[["overlap", "n_out", "n_in"]].agg(["mean", "min", "max"]).to_string())
    # 代理权重:逐日 0.5·Σ|w_new − w_legacy|
    wr = []
    for ix, nm in [("000852", "csi1000"), ("000905", "csi500"), ("000300", "csi300")]:
        wl = pd.read_parquet(L / "cache" / "_tmp" / f"proxy_{nm}_weights.parquet")
        wn = pd.read_parquet(N / "cache" / "_tmp" / f"proxy_{nm}_weights.parquet").reindex_like(wl)
        dd = 0.5 * (wn.fillna(0) - wl.fillna(0)).abs().sum(axis=1)
        dd = dd[dd.index >= "2021-01-01"]
        wr.append(dict(index=ix, mean_half_l1=float(dd.mean()), max_half_l1=float(dd.max()), median_half_l1=float(dd.median())))
    pd.DataFrame(wr).to_csv(LEDGER / "mvfix_downstream_proxy_weights.csv", index=False)
    print(pd.DataFrame(wr).to_string())
    # 风险模型规模(risk_model_summary.json 的标量)
    jl = json.load(open(L / "research" / "ledger" / "risk_model_summary.json"))
    jn = json.load(open(N / "research" / "ledger" / "risk_model_summary.json"))
    jo = json.load(open(LEDGER / "risk_model_summary.json"))
    rs = []

    def walk(x, y, z, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in x:
                if k in y:
                    walk(x[k], y[k], z.get(k) if isinstance(z, dict) else None, path + [str(k)])
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            rs.append(dict(key="/".join(path), original=z if isinstance(z, (int, float)) else np.nan,
                           legacy=x, new=y, delta=y - x))
    walk(jl, jn, jo, [])
    pd.DataFrame(rs).to_csv(LEDGER / "mvfix_downstream_risk_summary.csv", index=False)
    log("downstream 完成")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["eval", "barra", "shadow", "shadow_run", "downstream"])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--which", choices=["legacy", "new"], help="shadow_run:哪个影子根")
    ap.add_argument("--steps", default="proxy,risk,cap,extra,ie", help="shadow_run:逗号分隔," + "/".join(SHADOW_STEPS))
    ARGS = ap.parse_args()
    {"eval": stage_eval, "barra": stage_barra, "shadow": stage_shadow, "shadow_run": stage_shadow_run,
     "downstream": lambda: stage_downstream()}[ARGS.stage]()
