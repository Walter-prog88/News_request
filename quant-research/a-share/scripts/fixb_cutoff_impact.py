"""数据截止日修复的影响量化(前缀 fixb,2026-10-10)。

改了什么(见 research/DATA_CUTOFF_DISCIPLINE.md):newdata / fundamental 的加载器读入后立即按
config.data_cutoff()(默认 2025-12-31)截断,财报版本选择改为 PIT。本脚本回答"改前改后研究期数值差多少"。

旧口径 = git HEAD 的 config / newdata / fundamental / factors 四个文件(其余模块与当前工作区相同)。
要在同一台机器上复现旧口径,用 --make-old-root 生成一个"旧代码 + 数据软链"的影子根目录,再用 --root 指向它:

    python3 scripts/fixb_cutoff_impact.py --make-old-root /tmp/fixb_old_root
    python3 scripts/fixb_cutoff_impact.py --stage panels --tag old --root /tmp/fixb_old_root   # 后台约 10 分钟
    python3 scripts/fixb_cutoff_impact.py --stage panels --tag trunc --no-early                 # 当前代码,只截断
    python3 scripts/fixb_cutoff_impact.py --stage panels --tag new                              # 当前代码
    python3 scripts/fixb_cutoff_impact.py --stage fundcols    # 报告表逐列:旧 / 仅截断 / 截断+版本PIT
    python3 scripts/fixb_cutoff_impact.py --stage rowcounts   # 各加载器截掉的行数、版本结构、依赖可见日缺口
    python3 scripts/fixb_cutoff_impact.py --stage compare     # 汇总 -> research/ledger/fixb_*.csv

产出(research/ledger/):
  fixb_impact_summary.csv     面板级对比(真实市值、近似市值、ST、COMP_v5、Barra 暴露、预告/快报/龙虎榜面板…),
                              分三步:截断(old->trunc)、版本 PIT(trunc->new)、合计(old->new)
  fixb_fundamental_columns.csv 报告表每列 _pit_daily 面板:旧 vs 仅截断 vs 截断+版本 PIT
  fixb_cutoff_rowcounts.csv   每个源文件 > 截止日被截掉的行数、财报多版本结构
  fixb_pit_dependency_gap.csv 衍生列"映射的可见日"早于"真实依赖可见日"的行数(既有问题,只报告不修)
  fixb_comp_v5_downstream.csv COMP_v5 打分变化对标准排雷 / Top20 / Top50 的影响(仅当打分有变化)
中间面板落 cache/_tmp/fixb_impact/<tag>/。

第二轮审查后(同日):"版本 PIT"一步改为 as-of 快照 + 股本 0 占位一视同仁(fundamental._asof_snapshots /
_zero_share_as_missing),--no-early 仍复现"仅截断"口径(两者都关);sue_bt / earn_streak 的真实依赖可见日改用
报告表里按季度网格算好的 avail_sue(14 季)/ avail_streak(17 季);rowcounts 另报 as-of 快照规模、
股本 0 占位行数、build_market_cap 送转还原缺口的历史频率估计。PIT 的滚动原点检验见 scripts/fixb_rolling_origin.py。

纪律:只读研究期面板(<= 2025-12-31),不触碰 cache/sealed;COMP_v5 成员在 2020-2025 选出,
本脚本的组合数字只用于"改前改后是否一致"的对照,属信号样本内,数据不含已退市股(幸存者偏差)。
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings

ROOT_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ap = argparse.ArgumentParser()
_ap.add_argument("--stage", choices=["panels", "fundcols", "rowcounts", "compare"])
_ap.add_argument("--tag", default="new")
_ap.add_argument("--root", default=ROOT_DEFAULT, help="代码根目录(其下 src/afactor);默认本仓库")
_ap.add_argument("--make-old-root", default=None, help="生成旧口径影子根目录到该路径后退出")
_ap.add_argument("--no-early", action="store_true", help="只截断、不追加早期版本快照(FU.PIT_EARLY_VERSIONS=False)")
ARGS = _ap.parse_args()

OLD_FILES = ["config", "newdata", "fundamental", "factors"]
LINKS = ["data_8:21", "cache", "new_data", "research", "reports", "logs", "stock_sw_industry.csv", "tests"]


def make_old_root(dst: str) -> None:
    """旧口径影子根:src/afactor 复制当前工作区,再用 git HEAD 覆盖四个被修改的文件;数据目录软链回本仓库。"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "src"))
    shutil.copytree(os.path.join(ROOT_DEFAULT, "src", "afactor"), os.path.join(dst, "src", "afactor"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    for f in OLD_FILES:
        txt = subprocess.check_output(["git", "-C", ROOT_DEFAULT, "show", f"HEAD:src/afactor/{f}.py"])
        with open(os.path.join(dst, "src", "afactor", f"{f}.py"), "wb") as fh:
            fh.write(txt)
    for d in LINKS:
        os.symlink(os.path.join(ROOT_DEFAULT, d), os.path.join(dst, d))
    print(f"旧口径影子根目录已生成: {dst}")


if ARGS.make_old_root:
    make_old_root(ARGS.make_old_root)
    sys.exit(0)

os.environ.setdefault("INDEX_DIR", os.path.join(
    os.path.dirname(ROOT_DEFAULT), "ETF-Research", "index project", "data", "raw",
    "akshare_index_history_frozen"))
sys.path.insert(0, os.path.join(ARGS.root, "src"))
warnings.filterwarnings("ignore")

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from afactor import config as C   # noqa: E402

REPO = C.ROOT                 # 影子根下 C.ROOT 是影子根本身,数据目录是软链
LEDGER = C.LEDGER
OUT = C.TMP / "fixb_impact"
CUTOFF = pd.Timestamp(C.RESEARCH_END)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _save(df: pd.DataFrame, tag: str, name: str) -> None:
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    df.astype(np.float32).to_parquet(d / f"{name}.parquet", compression="zstd")


# ====================================================================== 1. 面板
def stage_panels(tag: str) -> None:
    from afactor.panel import Panel
    from afactor import factors as FA, combine, newdata as ND, fundamental as FU, barra as BR
    if ARGS.no_early:
        FU.PIT_EARLY_VERSIONS = False
    p = Panel()
    dates, codes = p.dates, p.codes
    assert dates.max() <= CUTOFF
    log(f"[{tag}] 代码根 {ARGS.root};早期版本 {getattr(FU, 'PIT_EARLY_VERSIONS', '旧代码无此开关')};面板 {dates[0].date()}~{dates[-1].date()} x {len(codes)}")

    for nm, fn in [("nd_total_mv", ND.total_mv), ("nd_circ_mv", ND.circ_mv), ("nd_raw_close", ND.raw_close),
                   ("nd_turnover_rate", ND.turnover_rate), ("nd_float_share", ND.float_share)]:
        _save(fn(dates, codes), tag, nm)
    log("真实市值 / 换手 / 流通股本")
    _save(FU.build_market_cap(dates, codes, p.raw("close_px")), tag, "fu_market_cap")
    _save(FU.build_st_panel(dates, codes).astype(np.float32), tag, "fu_is_st")
    log("近似市值 / ST")
    for k, v in ND.surprise_panels(dates, codes, 1).items():
        _save(v, tag, f"nd_sp_{k}")
    for k, v in ND.toplist_panels(dates, codes, 1).items():
        _save(v, tag, f"nd_tl_{k}")
    log("预告 / 快报 / 龙虎榜面板")
    idx = ND.index_daily()
    _save(idx[idx.index <= CUTOFF], tag, "nd_index_daily")
    li = ND.listing_info(codes)
    _save(pd.DataFrame({c: li[c].astype("int64").where(li[c].notna()) / 86400e9 for c in li.columns}),
          tag, "nd_listing_info_days")
    _save(ND.industry_labels(dates[:1], codes), tag, "nd_industry_labels")

    FA.build_all()
    for n in ["fu_roe", "fu_rev_yoy", "fu_bp", "fu_ep", "fu_leverage", "fx_fu_rev_yoy__P__ivol_20",
              "fx_fu_roe__P__ivol_20", "fx_fu_roe__P__id_ret_intraday_m60"]:
        _save(FA.REGISTRY[n][1](p).reindex(index=dates, columns=codes), tag, f"fa_{n}")
    log("基本面因子")

    # COMP_v5(open10 口径;与 run_barra_attribution 相同的合成法)
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members, signs = [n for n in man["members"] if n in FA.REGISTRY], man["signs"]
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=dates, columns=codes).replace([np.inf, -np.inf], np.nan)
           for n in members}
    score = combine.combine_equal(combine.prepare(fac, signs, p["universe"]))
    del fac
    score.to_parquet(OUT / tag / "comp_v5_score_open10.parquet")       # 原精度,便于逐位比对
    log(f"COMP_v5 打分({len(members)} 个成员)")

    # Barra 暴露(风险模型默认 v3 参数:raw_clip=5, size_clip=False,覆盖池 cover)
    rm = BR.RiskModel()
    u = p["universe"].reindex(index=dates, columns=codes).fillna(False)
    cover = BR.coverage_mask(p).reindex(index=dates, columns=codes).fillna(False)
    X = BR.build_exposures(p, u, cover=cover, raw_clip=rm.params["raw_clip"], size_clip=rm.params["size_clip"])
    for k, v in X.items():
        _save(v.reindex(index=dates, columns=codes), tag, "barra_" + k.split()[0])
    _save(cover.astype(np.float32), tag, "barra_cover")
    log("Barra 暴露;完成")


# ====================================================================== 2. 报告表逐列
SKIP_COLS = {"stock_code", "end_date", "comp_type", "qi", "quarter", "pit_vintage"}


def _old_fundamental():
    """旧版 fundamental.py(git HEAD)作为独立模块加载;它只依赖 afactor.config 的 ROOT / DAILY。"""
    import importlib.util
    import tempfile
    txt = subprocess.check_output(["git", "-C", ROOT_DEFAULT, "show", "HEAD:src/afactor/fundamental.py"])
    f = os.path.join(tempfile.mkdtemp(prefix="fixb_"), "fundamental_head.py")
    with open(f, "wb") as fh:
        fh.write(txt)
    spec = importlib.util.spec_from_file_location("fundamental_head", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _diff_stats(a: np.ndarray, b: np.ndarray, dates, codes) -> dict:
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = np.zeros_like(a, dtype=np.float64)
    d[both] = np.abs(a[both].astype(np.float64) - b[both].astype(np.float64))
    chg = both & (d > 0)
    den = np.maximum(np.abs(a[both].astype(np.float64)), 1e-12)
    rel = d[both] / den
    any_chg = chg | (fa != fb)
    rows, cols = np.nonzero(any_chg)
    db = d[both]
    return dict(n_cells=int(fa.sum()), n_changed=int(chg.sum()), n_nan_to_val=int((~fa & fb).sum()),
                n_val_to_nan=int((fa & ~fb).sum()), max_abs_diff=float(d.max()) if both.any() else np.nan,
                q99_abs_diff=float(np.quantile(db, 0.99)) if db.size else np.nan,
                q9999_abs_diff=float(np.quantile(db, 0.9999)) if db.size else np.nan,
                share_abs_gt_1e3=float((db > 1e-3).mean()) if db.size else np.nan,
                share_abs_gt_0p1=float((db > 0.1).mean()) if db.size else np.nan,
                max_rel_diff=float(rel.max()) if rel.size else np.nan,
                n_stocks=int(len(np.unique(cols))), n_days=int(len(np.unique(rows))),
                first_date=_fmt(dates[rows.min()]) if rows.size else "",
                last_date=_fmt(dates[rows.max()]) if rows.size else "")


def _fmt(x) -> str:
    return str(x.date()) if isinstance(x, pd.Timestamp) else str(x)


def stage_fundcols() -> None:
    from afactor.panel import Panel
    from afactor import fundamental as FU
    p = Panel()
    dates, codes = p.dates, p.codes
    old_mod = _old_fundamental()
    rep_old = old_mod.build_report_level()
    rep_new = FU.build_report_level()
    # "仅截断" = PIT_EARLY_VERSIONS=False 重建(既不追加 as-of 快照,也不做股本 0 占位处理),与 panels --no-early 同口径
    FU.PIT_EARLY_VERSIONS = False
    try:
        rep_trunc = FU.build_report_level()
    finally:
        FU.PIT_EARLY_VERSIONS = True
    cols = [c for c in rep_old.columns if c not in SKIP_COLS and not c.startswith("avail")
            and pd.api.types.is_numeric_dtype(rep_old[c])]
    log(f"报告表 旧 {rep_old.shape} / 新 {rep_new.shape}(早期版本行 {int((rep_new.pit_vintage != FU.PIT_FINAL).sum())});{len(cols)} 列")
    rows = []
    for i, c in enumerate(cols):
        a = old_mod._pit_daily(rep_old, c, dates, codes).to_numpy(np.float32)
        t = FU._pit_daily(rep_trunc, c, dates, codes).to_numpy(np.float32)
        n = FU._pit_daily(rep_new, c, dates, codes).to_numpy(np.float32)
        for step, x, y in [("截断(旧->仅截断)", a, t), ("版本PIT(仅截断->新)", t, n), ("合计(旧->新)", a, n)]:
            rows.append(dict(column=c, step=step, **_diff_stats(x, y, dates, codes)))
        if i % 10 == 0:
            log(f"  {i + 1}/{len(cols)} {c}")
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / "fixb_fundamental_columns.csv", index=False)
    log(f"写出 {LEDGER / 'fixb_fundamental_columns.csv'}")


# ====================================================================== 3. 行数 / 版本 / 依赖缺口
def stage_rowcounts() -> None:
    from afactor import newdata as ND, fundamental as FU
    D = ND.DATA
    ci = int(CUTOFF.strftime("%Y%m%d"))
    rows = []

    def add(src, key, n_all, n_drop, note=""):
        rows.append(dict(source=src, date_key=key, rows_total=int(n_all), rows_after_cutoff=int(n_drop),
                         share_dropped=n_drop / max(n_all, 1), note=note))

    db = pd.read_csv(D / "daily_basic.csv", usecols=["trade_date"])
    add("daily_basic.csv", "trade_date", len(db), (db.trade_date > ci).sum())
    for f, k in [("forecast.csv", "ann_date"), ("express.csv", "ann_date"), ("top_list.csv", "trade_date")]:
        x = pd.read_csv(D / f, usecols=[k])
        add(f, k, len(x), (x[k] > ci).sum())
    fc = pd.read_csv(D / "forecast.csv", usecols=["ts_code", "end_date", "ann_date"])
    g = fc.groupby(["ts_code", "end_date"])["ann_date"].agg(["min", "max"])
    add("forecast.csv", "ann_date", len(g), int(((g["min"] <= ci) & (g["max"] > ci)).sum()),
        "(股票,报告期)对:截止日前有版本、截止日后又有修正 —— 旧口径会把截止日前版本挤掉")
    st = pd.read_csv(D / "st_info_latest.csv", usecols=["trade_date"])
    add("st_info_latest.csv", "trade_date", len(st), (pd.to_datetime(st.trade_date) > CUTOFF).sum())
    sb = pd.read_csv(D / "stock_basic.csv")
    add("stock_basic.csv", "list_date", len(sb), (sb.list_date > ci).sum(), "截止日后上市:整行丢弃")
    add("stock_basic.csv", "delist_date", len(sb), (sb.delist_date > ci).sum(), "截止日后退市:退市日置空")
    for f in ["stock_income_2015_now.pkl", "stock_cashflow_2015_now.pkl", "stock_balancesheet_2015_now.pkl"]:
        d = pd.read_pickle(D / f)[["stock_code", "ann_date", "f_ann_date", "end_date"]]
        for c in ("ann_date", "f_ann_date", "end_date"):
            d[c] = pd.to_datetime(d[c], errors="coerce")
        av = d[["ann_date", "f_ann_date"]].max(axis=1)
        add(f, "max(ann,f_ann)", len(d), (av > CUTOFF).sum())
        add(f, "max(ann,f_ann)", len(d), int(((av > CUTOFF) & (d.end_date <= "2025-09-30")).sum()),
            "其中报告期 <= 2025Q3(截止日前本应已有版本,多为 2026 年的更正/补报)")
        k = d[av <= CUTOFF]
        add(f, "f_ann_date>ann_date", len(k), int((k.f_ann_date > k.ann_date).sum()),
            "截止日前可见、实际公告日晚于首次公告日的行:数据源用更正版覆盖了原行,原版本的公告日已不可知(§4 第 2 条机制 a)")
        n = k.groupby(["stock_code", "end_date"]).size()
        add(f, "版本数", len(n), int((n > 1).sum()),
            f"截止日前多版本(股票,报告期)对;涉及 {n[n > 1].reset_index().stock_code.nunique()} 只股票;最多 {int(n.max())} 版")
    # build_market_cap 送转还原为什么必须用前复权基准的股本:若截到 2025-12-31,
    # 2026 年送转的股票整个研究期的近似市值都会少乘这一步送转倍数
    q_full = FU._adjbase_quarterly_shares().replace(0, np.nan)
    last_q = CUTOFF.year * 4 + (CUTOFF.quarter - 1)
    step = FU._songzhuan_factor(q_full)                     # 从各季度到样本末的累计送转倍数
    after = step.loc[last_q] if last_q in step.index else pd.Series(dtype=float)
    hit = after[after > 1.0 + 1e-9]
    add("build_market_cap 送转还原", "前复权基准", len(after.dropna()), len(hit),
        f"2025Q4 之后(2026 年)识别到送转的股票数;若截到研究期末,它们 2020-2025 的近似市值整体偏低,"
        f"倍数中位 {hit.median():.3f}、最大 {hit.max():.3f}" if len(hit) else "无")
    # as-of 快照规模与股本 0 占位(第二轮审查 P2)
    tabs = [FU._load(n, []) for n in FU.REPORT_TABLES]
    snaps = FU._snapshot_dates(tabs)
    rep_all = FU.build_report_level()
    early = rep_all[rep_all["pit_vintage"] != FU.PIT_FINAL]
    add("as-of 快照", "快照日", 0 if snaps is None else len(snaps), len(early),
        f"有被替代版本的股票 {0 if snaps is None else snaps.stock_code.nunique()} 只、快照日 {0 if snaps is None else len(snaps)} 个"
        f"(单股最多 {0 if snaps is None else int(snaps['rank'].max()) + 1} 个);rows_after_cutoff 列 = 追加的快照行数,"
        f"涉及 {early.stock_code.nunique()} 只股票")
    bs = FU._load(FU.REPORT_TABLES[2], ["total_share"])
    bs_fin = FU._pick_version(bs, None)
    z_fin, z_all = bs_fin[bs_fin.total_share <= 0], bs[bs.total_share <= 0]
    add("资产负债表 total_share<=0", "占位行", len(bs_fin), len(z_fin),
        f"最终版 {len(z_fin)} 行、{z_fin.stock_code.nunique()} 只;所有版本合计 {len(z_all)} 行"
        f"(其中被后续版本替代的 {len(z_all) - len(z_fin)} 行)。现一律视为未披露")
    # build_market_cap 送转还原缺口(第二轮审查 P2):资产负债表原始文件最晚报告期 2026Q1,
    # 2026-04~08 除权的送转不在股本序列里。用研究期 2021-2025 每年同期(Q1->Q2、Q2->Q3)的送转频率估计量级。
    fin = FU._final_rows(rep_all)
    q = fin.pivot_table(index="qi", columns="stock_code", values="total_share", aggfunc="last")
    r = q / q.shift(1)
    x = r * 10.0
    clean = (r >= 1.05) & ((x - x.round()).abs() < 0.02) & (r <= 11.0)
    for y in range(2021, 2026):
        q2, q3 = y * 4 + 1, y * 4 + 2
        n12 = int(clean.loc[q2].sum()) if q2 in clean.index else 0
        n23 = int(clean.loc[q3].sum()) if q3 in clean.index else 0
        med = float(r.loc[q2][clean.loc[q2]].median()) if n12 else np.nan
        add("build_market_cap 送转还原缺口估计", f"{y} 年 Q1->Q2 / Q2->Q3", int(q.loc[q2].notna().sum()), n12 + n23,
            f"{y} 年 Q1->Q2 送转跳变 {n12} 只(倍数中位 {med:.2f})、Q2->Q3 {n23} 只;"
            f"2026 年同期的送转已压低前复权价、但不在截到 2026Q1 的股本序列里,未被还原")
    idx_n = sum(len(v.dropna()) for _, v in ND.index_daily().items())
    rows.append(dict(source="宽基指数快照", date_key="date", rows_total=np.nan, rows_after_cutoff=np.nan,
                     share_dropped=np.nan, note=f"截断后 {idx_n} 个指数日;快照到 2026-07-14"))
    pd.DataFrame(rows).to_csv(LEDGER / "fixb_cutoff_rowcounts.csv", index=False)
    log(f"写出 {LEDGER / 'fixb_cutoff_rowcounts.csv'}")

    # ---- 既有问题(只报告不修):衍生列的映射可见日没有覆盖它的全部依赖 ----
    rep = FU.build_report_level()
    rep = rep[rep["pit_vintage"] == FU.PIT_FINAL].copy()
    rep = rep.sort_values(["stock_code", "qi"])
    g = rep.groupby("stock_code", sort=False)
    ns = lambda s: s.astype("int64").where(s.notna())            # noqa: E731
    av, avq, avy = ns(rep["avail"]), ns(rep["avail_q"]), ns(rep["avail_yoy"])
    # sue_bt / earn_streak 用报告表里按季度网格算好的 avail_sue(P..P-13,14 季)/ avail_streak(P..P-16,17 季)
    # (第二轮审查 P2:此前 avail_sue 是 rolling(12),漏了 P-12 / P-13,会略微低估缺口);
    # earn_vol_q 仍按行 rolling(rep 已去掉网格空行),只是近似"按季度",仅用于量级报告
    true_dep = {
        "sue_bt": ns(rep["avail_sue"]),
        "sue_ta": np.fmax(avy, ns(rep["avail_avg"])),
        "earn_vol_q": avq.groupby(rep["stock_code"], sort=False).transform(lambda s: s.rolling(8, min_periods=1).max()),
        "earn_streak": ns(rep["avail_streak"]),
        "d_ni_q": avy, "d_revenue_q": avy, "d_operate_profit_q": avy, "d_n_cashflow_act_q": avy,
    }
    mapped = {c: (avq if c.endswith("_q") else av) for c in true_dep}
    gap_rows = []
    for c, dep in true_dep.items():
        ok = rep[c].notna().to_numpy() & np.isfinite(dep.to_numpy(np.float64)) & mapped[c].notna().to_numpy()
        gap = (dep.to_numpy(np.float64) - mapped[c].to_numpy(np.float64))[ok] / 86400e9
        gap_rows.append(dict(column=c, mapped_avail=FU._avail_col(c), rows_with_value=int(ok.sum()),
                             rows_dep_later=int((gap > 0).sum()), share=float((gap > 0).mean()) if ok.any() else np.nan,
                             median_gap_days=float(np.median(gap[gap > 0])) if (gap > 0).any() else np.nan,
                             factors="pead_sue_bt/pead_jump_combo 等" if c.startswith("sue") else ""))
    pd.DataFrame(gap_rows).to_csv(LEDGER / "fixb_pit_dependency_gap.csv", index=False)
    log(f"写出 {LEDGER / 'fixb_pit_dependency_gap.csv'}")


# ====================================================================== 4. 汇总
GROUPS = {
    "nd_total_mv": "真实总市值 ND.total_mv", "nd_circ_mv": "流通市值", "nd_raw_close": "未复权收盘价",
    "nd_turnover_rate": "真实换手率", "nd_float_share": "流通股本", "fu_market_cap": "近似市值 build_market_cap",
    "fu_is_st": "ST 面板", "nd_index_daily": "宽基指数(<=2025)", "nd_listing_info_days": "上市/退市日",
    "nd_industry_labels": "证监会行业标签",
}


def _load_pair(name, ta="old", tb="new"):
    a = pd.read_parquet(OUT / ta / f"{name}.parquet")
    b = pd.read_parquet(OUT / tb / f"{name}.parquet")
    assert a.index.equals(b.index) and a.columns.equals(b.columns), name
    return a, b


STEPS = [("合计(旧->新)", "old", "new"), ("截断(旧->仅截断)", "old", "trunc"), ("版本PIT(仅截断->新)", "trunc", "new")]


def stage_compare() -> None:
    names = sorted(f.stem for f in (OUT / "new").glob("*.parquet"))
    rows = []
    for step, ta, tb in STEPS:
        for nm in names:
            if not ((OUT / ta / f"{nm}.parquet").exists() and (OUT / tb / f"{nm}.parquet").exists()):
                continue
            a, b = _load_pair(nm, ta, tb)
            st = _diff_stats(a.to_numpy(np.float64), b.to_numpy(np.float64), a.index, a.columns)
            rows.append(dict(object=nm, step=step, desc=GROUPS.get(nm, ""), **st))
    summ = pd.DataFrame(rows)

    # ---- COMP_v5:旧代码重算 vs 冻结缓存(配方核对)、新 vs 旧(本次修复的影响)----
    frozen = pd.read_parquet(C.TMP / "comp_v5_score_open10.parquet")
    old, new = _load_pair("comp_v5_score_open10")
    fz = frozen.reindex(index=old.index, columns=old.columns)
    st = _diff_stats(fz.to_numpy(np.float64), old.to_numpy(np.float64), old.index, old.columns)
    rows2 = [dict(object="comp_v5_score_open10", step="配方核对(冻结缓存->旧代码重算)", desc="", **st)]
    summ = pd.concat([summ, pd.DataFrame(rows2)], ignore_index=True)

    changed = not np.array_equal(old.to_numpy(), new.to_numpy(), equal_nan=True)
    if changed:
        dst = C.TMP / "comp_v5_score_open10_fixb.parquet"
        new.to_parquet(dst)
        log(f"COMP_v5 打分有变化 -> 新缓存 {dst}(旧文件保留)")
        _comp_downstream(old, new)
    summ.to_csv(LEDGER / "fixb_impact_summary.csv", index=False)
    log(f"写出 {LEDGER / 'fixb_impact_summary.csv'}")
    print(summ.to_string(max_colwidth=40))


def _comp_downstream(old: pd.DataFrame, new: pd.DataFrame) -> None:
    """COMP_v5 打分变化 -> 截面秩相关、排雷剔除集合、Top20/50、标准回测年化。"""
    from afactor.panel import Panel
    from afactor import strategy as ST
    p = Panel()
    u = p["universe"].reindex(index=old.index, columns=old.columns).fillna(False).to_numpy()
    A, B = old.to_numpy(np.float64), new.to_numpy(np.float64)
    rows = []
    rk = []
    chg_days = []
    for i in range(len(old)):
        m = u[i] & np.isfinite(A[i]) & np.isfinite(B[i])
        if m.sum() < 50:
            continue
        if not np.array_equal(A[i][m], B[i][m]):
            chg_days.append(old.index[i])
            ra = pd.Series(A[i][m]).rank().to_numpy(); rb = pd.Series(B[i][m]).rank().to_numpy()
            rk.append(np.corrcoef(ra, rb)[0, 1])
    rows.append(dict(item="打分有变化的交易日数", value=len(chg_days), note=f"共 {len(old)} 日"))
    rows.append(dict(item="变化日截面 Spearman 最小值", value=float(np.min(rk)) if rk else 1.0, note="决策池内"))
    rows.append(dict(item="变化日截面 Spearman 中位数", value=float(np.median(rk)) if rk else 1.0, note=""))
    for nm, kw in [("exclude_std", dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5)),
                   ("top20_r5", dict(scheme="top", n=20, rebalance=5)),
                   ("top50_r5", dict(scheme="top", n=50, rebalance=5))]:
        ra = ST.run_strategy(old, p, "V1000", "open10", **kw)["rets"]
        rb = ST.run_strategy(new, p, "V1000", "open10", **kw)["rets"]
        d = (rb - ra).dropna()
        for win, lo, hi in [("2021-2023", "2021-01-01", "2023-12-31"), ("2024-2025", "2024-01-01", "2025-12-31")]:
            x, y = ra.loc[lo:hi], rb.loc[lo:hi]
            rows.append(dict(item=f"{nm} 年化收益 {win} 旧", value=float((1 + x).prod() ** (242 / len(x)) - 1), note="信号样本内"))
            rows.append(dict(item=f"{nm} 年化收益 {win} 新", value=float((1 + y).prod() ** (242 / len(y)) - 1), note="信号样本内"))
        rows.append(dict(item=f"{nm} 日收益差 最大绝对值", value=float(d.abs().max()), note=""))
        rows.append(dict(item=f"{nm} 日收益差 非零天数", value=int((d.abs() > 0).sum()), note=f"共 {len(d)} 天"))
    pd.DataFrame(rows).to_csv(LEDGER / "fixb_comp_v5_downstream.csv", index=False)
    log(f"写出 {LEDGER / 'fixb_comp_v5_downstream.csv'}")


if __name__ == "__main__":
    t0 = time.time()
    {"panels": lambda: stage_panels(ARGS.tag), "fundcols": stage_fundcols,
     "rowcounts": stage_rowcounts, "compare": stage_compare}[ARGS.stage]()
    log(f"总耗时 {time.time() - t0:.0f}s")
