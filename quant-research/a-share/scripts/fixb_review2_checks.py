"""第二轮对抗审查(fixb)的逐条复核,结果写 research/ledger/fixb_review2_checks.csv(前缀 fixb,2026-10-10)。

  P1  源覆盖守卫:默认截止日下把 ST 名单 / 龙虎榜模拟成"源只到某天"(读入后截断并登记源最晚日期),
      请求到 2025-12-31。修复前 build_st_panel 静默 ffill、0 条警告(审查实测错 524 格 / 20 只);修复后默认报错,
      显式接受后才返回旧状态(同时量化错格)。不读 2026 数据。
  P2  as-of 快照:合成用例上把实现换回上一版 S_k(以及"只对早期版本置 0 股本"),滚动原点真值检验应当失败;
      新实现通过(同一用例即 tests/test_data_cutoff.py::test_asof_snapshots_match_rolling_origin_truth)。
  回归 研究期面板:trunc(上一版代码)vs trunc2(本版代码,--no-early)逐位一致;new_r1(上一版)vs new(本版)的差异。

    python3 scripts/fixb_review2_checks.py      # 约 3 分钟;需要先跑完 fixb_cutoff_impact.py 的 panels(new / trunc2)
纪律:只用 <= 2025-12-31 的数据,不算任何收益 / IC。
"""
from __future__ import annotations
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from afactor import config as C, fundamental as FU, newdata as ND   # noqa: E402

OUT = C.TMP / "fixb_impact"
ROWS: list[dict] = []


def put(finding, check, value, note=""):
    ROWS.append(dict(finding=finding, check=check, value=value, note=note))
    print(f"[{finding}] {check}: {value} {note}", flush=True)


# ---------------------------------------------------------------- P1
def p1() -> None:
    from afactor.panel import Panel
    p = Panel()
    dates = p.dates[p.dates >= "2025-07-01"]
    codes = p.codes
    saved = dict(C._SOURCE_END)
    true = FU.build_st_panel(dates, codes)
    orig = FU._read_st

    def short():
        x = orig()
        C.record_source_end("st_info_latest.csv", pd.Timestamp("2025-09-30"))
        return x[x["trade_date"] <= "2025-09-30"]
    FU._read_st = short
    try:
        try:
            FU.build_st_panel(dates, codes)
            put("P1", "ST 模拟缺口:默认行为", "未报错(守卫失效)")
        except C.SourceCoverageError:
            put("P1", "ST 模拟缺口:默认行为", "SourceCoverageError", "源只到 2025-09-30,请求到 2025-12-31")
        with C.accept_source_gap("st_info_latest.csv", reason="fixb 复核:量化显式接受 ST 缺口后的错格"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                stale = FU.build_st_panel(dates, codes)
        win = dates > pd.Timestamp("2025-09-30")
        diff = (true.loc[win] != stale.loc[win]).to_numpy()
        put("P1", "ST 模拟缺口:显式接受后的 RuntimeWarning 条数", len([x for x in w if issubclass(x.category, RuntimeWarning)]))
        put("P1", "ST 模拟缺口:显式接受后的错格", int(diff.sum()),
            f"{int(diff.any(axis=0).sum())} 只股票;模拟前向 {int(win.sum())} 天;真实 ST 格 {int(true.loc[win].to_numpy().sum())}")
    finally:
        FU._read_st = orig
        C._SOURCE_END.clear(); C._SOURCE_END.update(saved)

    tl_true = ND.toplist_panels(dates, codes)
    orig_r = ND._read_dated_csv

    def short_r(name, col, **kw):
        x = orig_r(name, col, **kw)
        if name == "top_list.csv":
            C.record_source_end(name, pd.Timestamp("2025-11-28"))
            x = x[x[col] <= 20251128]
        return x
    ND._read_dated_csv = short_r
    try:
        try:
            ND.toplist_panels(dates, codes)
            put("P1", "龙虎榜模拟缺口:默认行为", "未报错(守卫失效)")
        except C.SourceCoverageError:
            put("P1", "龙虎榜模拟缺口:默认行为", "SourceCoverageError", "源只到 2025-11-28")
        with C.accept_source_gap("top_list.csv", reason="fixb 复核:龙虎榜显式接受缺口后的行为"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tl = ND.toplist_panels(dates, codes)
        k = dates > pd.Timestamp("2025-12-01")
        put("P1", "龙虎榜模拟缺口:接受后 12-01 之后 tl_on_list 全为 NaN", bool(tl["tl_on_list"].loc[k].isna().all().all()),
            f"修复前会填 0;真值在这段有 {int(tl_true['tl_on_list'].loc[k].to_numpy().sum())} 个上榜格")
        put("P1", "龙虎榜模拟缺口:12-01(11-28 上榜的生效日)与真值一致",
            bool(tl["tl_on_list"].loc["2025-12-01"].equals(tl_true["tl_on_list"].loc["2025-12-01"])))
    finally:
        ND._read_dated_csv = orig_r
        C._SOURCE_END.clear(); C._SOURCE_END.update(saved)


# ---------------------------------------------------------------- P2:旧 S_k 在合成用例上失败
def _old_sk(tabs, final):
    """上一版 _early_vintages(按"第 k 版"拼快照,只对早期版本置 0 股本),原样保留作反证。"""
    nmax = max([int(t.groupby(["stock_code", "end_date"], sort=False).size().max()) if len(t) else 1 for t in tabs])
    if nmax <= 1:
        return None
    fin = final.set_index(["stock_code", "qi"])
    out = []
    for k in range(nmax - 1):
        aff = set()
        for t in tabs:
            n = t.groupby(["stock_code", "end_date"], sort=False)["avail"].transform("size")
            aff |= set(t.loc[n.to_numpy() > k + 1, "stock_code"])
        if not aff:
            continue
        dk = FU._derive(FU._merge_tables(*[FU._pick_version(t[t["stock_code"].isin(aff)], k) for t in tabs]))
        dk.loc[dk["total_share"] <= 0, "total_share"] = np.nan
        a = dk.set_index(["stock_code", "qi"])
        b = fin.reindex(a.index)
        same = np.ones(len(a), dtype=bool)
        for c in a.columns:
            x, y = a[c], b[c]
            same &= (x == y).fillna(False).to_numpy(bool) | (x.isna().to_numpy() & y.isna().to_numpy())
        dk = dk[~same].copy()
        if len(dk):
            dk["pit_vintage"] = np.int16(k)
            out.append(dk)
    return pd.concat(out, ignore_index=True) if out else None


def p2_synthetic() -> None:
    import test_data_cutoff as T
    T.test_asof_snapshots_match_rolling_origin_truth()
    put("P2-版本依赖未来", "合成用例(新实现)", "通过", "tests/test_data_cutoff.py::test_asof_snapshots_match_rolling_origin_truth")
    keep = (FU._asof_snapshots, FU._zero_share_as_missing)
    for label, zero_all in [("旧 S_k + 只对早期版本置 0 股本", False), ("旧 S_k + 0 股本一视同仁", True)]:
        FU._asof_snapshots = _old_sk

        def _z(df, zero_all=zero_all):
            if zero_all:
                df.loc[df["total_share"] <= 0, "total_share"] = np.nan
            return df
        FU._zero_share_as_missing = _z
        try:
            T.test_asof_snapshots_match_rolling_origin_truth()
            put("P2-版本依赖未来", f"合成用例({label})", "通过(反证失败)")
        except AssertionError as e:
            first = str(e).split("), (")[0].lstrip("[(")
            put("P2-版本依赖未来", f"合成用例({label})", "失败(符合预期)", f"首个不一致: {first}")
        finally:
            FU._asof_snapshots, FU._zero_share_as_missing = keep


# ---------------------------------------------------------------- 回归与前后对照
def panels() -> None:
    names = sorted(f.stem for f in (OUT / "trunc").glob("*.parquet"))
    bad = [nm for nm in names if not np.array_equal(pd.read_parquet(OUT / "trunc" / f"{nm}.parquet").to_numpy(),
                                                     pd.read_parquet(OUT / "trunc2" / f"{nm}.parquet").to_numpy(),
                                                     equal_nan=True)]
    put("回归", "trunc(上一版)vs trunc2(本版 --no-early)逐位一致的面板数", f"{len(names) - len(bad)}/{len(names)}",
        "非版本 PIT 的改动(源覆盖守卫、缓存键)不改变研究期数值" if not bad else f"不一致: {bad}")
    if not (OUT / "new_r1").exists():
        return
    for nm in ["comp_v5_score_open10", "nd_total_mv", "fu_market_cap", "fa_fu_bp", "fa_fu_ep", "fa_fu_roe",
               "fu_is_st", "barra_Size", "barra_NonlinSize"]:
        a = pd.read_parquet(OUT / "new_r1" / f"{nm}.parquet").to_numpy(np.float64)
        b = pd.read_parquet(OUT / "new" / f"{nm}.parquet").to_numpy(np.float64)
        both = np.isfinite(a) & np.isfinite(b)
        chg = both & (a != b)
        flip = np.isfinite(a) != np.isfinite(b)
        st = np.unique(np.nonzero(chg | flip)[1])
        rel = np.abs(b[chg] / a[chg] - 1) if chg.any() else np.array([0.0])
        put("前后对照(new_r1->new)", nm, int(chg.sum()),
            f"NaN 翻转 {int(flip.sum())};涉及 {len(st)} 只;最大相对差 {rel.max():.3g}")


def total_mv_extrapolation() -> None:
    """附带发现(既有问题,本次不修):ND.total_mv 的日频外推 mult = 流通股本 / "PIT 总股本最近一次变化时的流通股本"。
    只有 PIT 总股本**数值变化**时才重新锚定(sh.diff() != 0,首日强制锚定);总股本多年不变而流通股本因限售解禁增长时,
    mult 一直累积,真实市值被持续高估。r1->r2 的 total_mv 差异按机制拆开:上一版用流通股本兜底的(股本 0 占位,
    本次纠正低估)vs 上一版外推倍数因 NaN 段失效、本次恢复外推的(随之带上这个高估)。"""
    mv = pd.read_parquet(OUT / "new" / "nd_total_mv.parquet")
    rc = pd.read_parquet(OUT / "new" / "nd_raw_close.parquet")
    fs = pd.read_parquet(OUT / "new" / "nd_float_share.parquet")
    sh = FU._pit_daily(FU.build_report_level(), "total_share", mv.index, mv.columns).replace(0, np.nan)
    ratio = (mv / (rc * sh)).where(mv.notna() & sh.notna() & (fs <= sh * 1.01))   # 排除"PIT 总股本过期"(流通 > 总)
    r = ratio.to_numpy()
    r = r[np.isfinite(r)]
    put("附带发现:total_mv 外推高估", "total_mv / (未复权价 x PIT 总股本) 的分布(流通股本 <= 总股本的格子)",
        f">1.05: {(r > 1.05).mean():.1%};>1.5: {(r > 1.5).mean():.1%};>2: {(r > 2).mean():.1%}",
        f"共 {r.size} 格;其中一部分是两次财报之间的送转 / 增发(合理外推),其余是限售解禁")
    long = (ratio > 1.5).sum()
    put("附带发现:total_mv 外推高估", "比值 > 1.5 持续 >= 250 个交易日的股票数", int((long >= 250).sum()),
        f"有过 > 1.5 的股票 {int((long > 0).sum())} / {int(ratio.notna().any().sum())}")
    for c in ["600339.SH", "600390.SH", "600519.SH"]:
        x = ratio[c].dropna()
        put("附带发现:total_mv 外推高估", f"例:{c} 比值中位 / 最大", f"{x.median():.2f} / {x.max():.2f}",
            f"PIT 总股本 {sh[c].dropna().iloc[-1] / 1e8:.2f} 亿,流通股本 {fs[c].dropna().iloc[-1] / 1e8:.2f} 亿(期末)")
    if (OUT / "new_r1").exists():
        o = pd.read_parquet(OUT / "new_r1" / "nd_total_mv.parquet")
        chg = (o != mv) & o.notna() & mv.notna()
        fb = (o - rc * fs).abs() <= 1e-4 * o.abs()
        q = mv / o
        for lab, w in [("上一版 = 价 x 流通股本(0 股本兜底),本次纠正低估", chg & fb),
                       ("上一版外推倍数因 NaN 段失效(=1),本次恢复外推", chg & ~fb)]:
            v = q.to_numpy()[w.to_numpy()]
            put("前后对照(new_r1->new)", f"nd_total_mv 拆分:{lab}", int(w.to_numpy().sum()),
                f"{int(w.any().sum())} 只;新/旧 中位 {np.median(v):.3f}、范围 {v.min():.3f}~{v.max():.3f}")


if __name__ == "__main__":
    C.CUTOFF_LOG = C.ROOT / "logs" / "fixb_review2_checks_cutoff.log"      # 复核的切换记录单独存放
    p1()
    p2_synthetic()
    panels()
    total_mv_extrapolation()
    pd.DataFrame(ROWS).to_csv(C.LEDGER / "fixb_review2_checks.csv", index=False)
    print(f"写出 {C.LEDGER / 'fixb_review2_checks.csv'}")
