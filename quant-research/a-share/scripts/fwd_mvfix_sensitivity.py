"""前向检验的市值修复敏感性(前缀 fwd_mvfix;**事后敏感性,不改冻结判定、不替换任何 fwd_* 原结果**)。

背景:research/TOTAL_MV_FIX.md 把 newdata.total_share 改成"按报告期末重新锚定、只外推识别出的送转、新股加首日流通股本、
流通股本为下界"(TOTAL_SHARE_METHOD="songzhuan"),合并发生在前向检验(research/FORWARD_TEST_2026Q3.md)完成之后。
前向面板的市值**从来不是**研究期的 ND.total_mv:按预注册 §4.3 / §7.4,前向面板 mv = 未复权收盘 × PIT 总股本
(tushare 资产负债表 + baostock 季频 totalShare,按 pubDate 生效)× "报告期末以来的除权因子比"(含现金分红),
无 PIT 股本者为 NaN(scripts/fwd_build_panel.py stage_daily / assemble)。本脚本把 newdata 新口径的同一套规则
套在前向面板**同一组输入**上(PIT 股本 / 报告期序号 / 流通股本 / 未复权价 / 上市日),只换"两次财报之间怎么外推",
重建 mv,再重算依赖市值的读数:
  - P1 / P2(T1 排雷)、P3(T2 Top20):打分、决策池、引擎、同频池基准都不读市值 —— 重算以实测"逐位不变";
  - P4(T3 指增 V1)、P6(T5 b_d):经 Barra 风险模型(Size / NonlinSize、市值加权标准化、Beta / 残差波动的市值加权市场收益、
    WLS 权重、覆盖池 mv > 0、特质风险收缩的市值分档)与优化器的风格约束;
  - 中证 1000 实际成分权重 = 快照权重按复权收盘反推(fwd_weights),不读市值 —— 重算以实测"逐位不变"。
历史参考分布(freeze_ref_*)是研究期旧口径算的,本脚本**不重算、不重新判定**,只报告"新口径窗口值 − 旧口径窗口值";
百分位一律标注"旧口径历史分布,仅供参考"。

纪律:只算预注册评估窗(信号日 >= 2026-09-01)的收益;引擎 / 基准吃收益面板(twap_open10 在 09-02 前为 NaN),
2026-01 ~ 08 的数据只作特征与风险模型估计输入(因子收益只存 cache/_tmp/fwd_mvfix_risk/,不报告)。
截止日只经 fwd_panel.forward_context() / config.set_data_cutoff 放宽(本文件名前缀 fwd)。
不写任何 fwd_* 原产物:新面板字段 cache/_tmp/fwd_mvfix_panel/,其余 cache/_tmp/fwd_mvfix_*,账本 research/ledger/fwd_mvfix_sensitivity.csv。

阶段(按顺序;每段的日志 logs/fwd_mvfix_<阶段>.log):
    python3 scripts/fwd_mvfix_sensitivity.py freeze equiv      # 冻结校验现状(只读);新口径数组版 vs newdata.total_share 研究期逐位等价
    python3 scripts/fwd_mvfix_sensitivity.py build             # 前向新口径股本 / 市值 + 数据层对照(不算收益)
    python3 scripts/fwd_mvfix_sensitivity.py xcheck            # 2025 年行:前向旧 / 新 vs 研究期 ND 旧(legacy)/ 新口径
    python3 scripts/fwd_mvfix_sensitivity.py score t12 risk t3 t5 diag report

审查 P2 之后的补充阶段(2026-10-10 第二轮;都是事后描述,不进判定;只写新的 section,不覆盖上面各节):
    python3 scripts/fwd_mvfix_sensitivity.py freeze_timeline   # 冻结日志逐条 + 当前只读快照(section=freeze_timeline)
    python3 scripts/fwd_mvfix_sensitivity.py sz_label          # 研究期 TOTAL_MV_FIX §2.2 的送转标签原样用到前向窗口(section=sz_label)
    python3 scripts/fwd_mvfix_sensitivity.py pead              # P5′ D1 补充读数在新口径下重算(section=pead;先复现原批次与 D̄)
    python3 scripts/fwd_mvfix_sensitivity.py kmeans stmt_em    # 统计行业标签微扰、补利润表变体在新口径下的 P4 / P6(section=kmeans / stmt_em)
    python3 scripts/fwd_mvfix_sensitivity.py may1314 nofu      # 05-13/14 收盘补齐、去基本面成员两个变体在新口径下的 P4 / P6(section=may1314 / nofu)
    python3 scripts/fwd_mvfix_sensitivity.py t3extra cond      # §6 T3 描述量旧 → 新;§9 P4 条件位置(只读落盘结果)
  每个补算阶段先在原面板 + 原缓存上复现旧值;新风险模型缓存写 cache/_tmp/fwd_mvfix_{stmt_em,may1314}_risk/,不碰 fwd_review_*_risk。
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor.config import TMP, LEDGER, CACHE, ROOT
from afactor.fwd_panel import ForwardPanel, forward_context, assert_forward_index, SIG_START, PANEL_DIR, FWD_CUTOFF
from afactor import minute5 as M5
import fwd_common as FC

log = FC.log
OUT = LEDGER / "fwd_mvfix_sensitivity.csv"
MV_DIR = TMP / "fwd_mvfix_panel"                     # 新口径的 mv / total_share(其余字段仍读 cache/forward/panel)
RISK_DIR = TMP / "fwd_mvfix_risk"
SCORE_FILE = TMP / "fwd_mvfix_score5.parquet"
BENCH_FILE = TMP / "fwd_mvfix_bench.parquet"
RETS_FILE = TMP / "fwd_mvfix_strategy_rets.parquet"
TE_FILE = TMP / "fwd_mvfix_te_daily.parquet"
AUX = TMP / "fwd_aux"                                # fwd_build_panel 的中间件(只读)
PRE = "fwd_mvfix"                                    # 引擎 record 落盘前缀(cache/_tmp/fwd_mvfix_*)
M1_END = pd.Timestamp("2026-08-17")                  # 与 fwd_build_panel 相同:daily_basic / 1 分钟末日
SKIP = M5.FWD_SKIP_HEAD
TD = 242
DIST = pd.read_parquet(LEDGER / "freeze_ref_window_dist.parquet")
QREF = pd.read_csv(LEDGER / "freeze_ref_quantiles.csv")
PCT_NOTE = "百分位为旧口径历史分布(freeze_ref,研究期旧市值),仅供参考,不重新判定"
_rows: list[dict] = []


# ============================================================================================ 账本
def rec(section, item, old=None, new=None, value=None, note="", old_pct_ref=None, new_pct_ref=None, unit=""):
    d = dict(section=section, item=item, old=old, new=new, delta=None, value=value, unit=unit,
             old_pct_ref=old_pct_ref, new_pct_ref=new_pct_ref, note=note,
             run_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    if isinstance(old, (int, float, np.integer, np.floating)) and isinstance(new, (int, float, np.integer, np.floating)):
        d["delta"] = float(new) - float(old)
    _rows.append(d)
    show = lambda v: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
    if old is not None or new is not None:
        log(f"[{section}] {item}: {show(old)} -> {show(new)}" + (f"  ({note})" if note else ""))
    else:
        log(f"[{section}] {item} = {show(value)}" + (f"  ({note})" if note else ""))


def flush():
    if not _rows:
        return
    secs = {r["section"] for r in _rows}
    FC.ledger_append(OUT, _rows, drop_where=lambda o: o["section"].isin(secs))
    _rows.clear()


def dist(key: str) -> pd.Series:
    d = DIST[DIST["key"] == key]
    if d.empty:
        raise KeyError(key)
    return d.set_index("window_end")["value"]


def eq_nan(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a, np.float64), np.asarray(b, np.float64), equal_nan=True))


def max_abs_log(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(np.max(np.abs(np.log(a[ok] / b[ok])))) if ok.any() else 0.0


# ============================================================================================ 新口径数组版
def total_share_core(cal, sh, qi, fs, px, lst, return_pre: bool = False):
    """newdata.total_share 新口径(TOTAL_SHARE_METHOD="songzhuan")的数组版:**输入**换成调用者给的
    (日历 cal、PIT 总股本 sh、该股本的报告期序号 qi、流通股本 fs、未复权价 px、上市日 lst),**计算**逐行照抄 newdata.total_share
    ——送转识别 ND._songzhuan_steps、报告期末 ND._quarter_end、常量 ND.REFLECT_TOL / ND.IPO_MAX_LAG 直接调用冻结后的 newdata,
    逐股循环(锚点、期后送转已含判定、新股加法)与流通股本下界原样照抄。
    研究期等价性(stage_equiv):用 newdata 自己的输入调用本函数,与 newdata.total_share 逐位比较。
    return_pre=True 时另返回取流通股本下界之前的值(诊断用)。"""
    from afactor import newdata as ND
    cal = pd.DatetimeIndex(cal)
    sh = np.asarray(sh, np.float64)
    qi = np.asarray(qi, np.float64)
    fs = np.asarray(fs, np.float64)
    C = np.cumsum(np.log(ND._songzhuan_steps(fs, px)), axis=0)       # 第 0..t 日送转的对数累计
    calv = cal.values.astype("datetime64[D]")
    T, N = sh.shape
    rows = np.arange(T)
    lst = np.asarray(lst).astype("datetime64[D]")
    has_fs = np.isfinite(fs)
    first = np.where(has_fs.any(axis=0), has_fs.argmax(axis=0), -1)
    out = np.full((T, N), np.nan)
    for j in range(N):
        s, q, c = sh[:, j], qi[:, j], C[:, j]
        ok = np.isfinite(s) & np.isfinite(q)
        if not ok.any():
            continue
        L, f0 = lst[j], int(first[j])
        ipo = (not np.isnat(L) and f0 >= 0 and calv[0] < L <= calv[f0]
               and f0 - int(np.searchsorted(calv, L)) <= ND.IPO_MAX_LAG)
        base = s.copy()
        okq = np.where(ok, q, np.nan)
        starts = np.nonzero(ok & ~(np.r_[np.nan, okq[:-1]] == okq))[0]
        anchor = np.zeros(T, np.int64)
        prev = None
        for i0, i1 in zip(starts, np.r_[starts[1:], T]):
            qe = ND._quarter_end(q[i0])
            if ipo and qe < L and i1 > f0:
                A, sL = fs[f0, j], s[f0]
                seg = np.arange(max(i0, f0), i1)
                done = np.isfinite(sL) & (s[seg] - sL >= 0.5 * A)
                base[seg] = np.where(done, s[seg], s[seg] + A)
            a = min(int(np.searchsorted(calv, qe, side="right")) - 1, i0)
            ca = c[a] if a >= 0 else 0.0
            if prev is not None and c[i0] - ca > 1e-12:
                pi, pa = prev
                old_est = base[pi] * np.exp(c[i0] - (c[pa] if pa >= 0 else 0.0))
                if abs(np.log(base[i0] / old_est)) < ND.REFLECT_TOL:
                    a = i0
            anchor[i0:i1] = a
            prev = (i1 - 1, a)
        a = np.minimum(anchor, rows)
        ca = np.where(a >= 0, c[np.maximum(a, 0)], 0.0)
        out[:, j] = np.where(ok, base * np.exp(c - ca), np.nan)
    pre = out.copy() if return_pre else None
    out = np.fmax(out, fs)
    return (out, pre) if return_pre else out


# ============================================================================================ freeze
def stage_freeze():
    """冻结校验现状(只读:调用 fwd_verify_freeze.verify(),不写 logs/fwd_verify_freeze.log)。
    市值修复合并改了 src/afactor/newdata.py(清单 core 类),前向检验完成之后的预期结果,不去"修"。"""
    import fwd_verify_freeze as VF
    r = VF.verify(VF.DEFAULT_MANIFEST, VF.ROOT)
    rec("freeze", "exit_code", value=int(r["exit_code"]),
        note=f"{r['n_ok']}/{r['n_files']} 一致;校验时刻 {r['checked_at']}(只读调用 verify(),未写冻结日志)")
    for e in r.get("mismatched", []):
        why = ("市值修复合并(TOTAL_MV_FIX.md),前向检验完成之后;预期,不修" if e["path"] == "src/afactor/newdata.py"
               else "前向检验完成之后被市值修复及其下游重跑改动(旧版备份在 research/ledger/_pre_mvfix/ 或 cache/_tmp/pre_mvfix_backup/);预期,不修")
        rec("freeze", f"mismatch:{e['path']}", value=e["category"], note=f"{why};现 mtime {e.get('now_mtime')}")
    for e in r.get("missing", []):
        rec("freeze", f"missing:{e['path']}", value=e["category"])
    un = r.get("unregistered", [])
    rec("freeze", "unregistered_src_py", value=len(un),
        note=f"其中非 fwd 前缀 {sum(not e['fwd_prefixed'] for e in un)} 个(fwd 前缀的新模块按设计不算失败)")
    flush()


# ============================================================================================ equiv
def stage_equiv():
    """研究期等价性(只用 <= 2025-12-31 数据,默认截止日):total_share_core 喂 newdata.total_share 自己的输入,
    与 newdata.total_share 逐位比较。"""
    from afactor import newdata as ND, fundamental as FU
    assert ND.TOTAL_SHARE_METHOD == "songzhuan", ND.TOTAL_SHARE_METHOD
    t0 = time.time()
    codes = pd.read_parquet(CACHE / "daily" / "close_px.parquet").columns
    cal = pd.DatetimeIndex(ND._load_daily_basic()["trade_date"].unique()).sort_values()
    ref = ND.total_share(cal, codes).to_numpy(np.float64)
    log(f"[equiv] newdata.total_share {ref.shape} [{time.time()-t0:.0f}s]")
    rep = FU.build_report_level()
    sh = FU._pit_daily(rep, "total_share", cal, codes).replace(0, np.nan).to_numpy(np.float64)
    qrep = rep.assign(sh_qi=rep["qi"].where(rep["total_share"].notna()).astype(np.float64))
    qi = FU._pit_daily(qrep, "sh_qi", cal, codes).to_numpy(np.float64)
    fs = ND.float_share(cal, codes).to_numpy(np.float64)
    px = ND.raw_close(cal, codes).to_numpy()
    lst = ND.listing_info(codes)["list_date"].to_numpy("datetime64[D]")
    mine = total_share_core(cal, sh, qi, fs, px, lst)
    same = eq_nan(mine, ref)
    rec("equiv", "core_vs_newdata_total_share_bitwise", value=bool(same),
        note=f"研究期 {cal[0].date()}~{cal[-1].date()} × {len(codes)} 只;同输入逐位比较(NaN 位置一致且数值逐位相同)")
    rec("equiv", "core_vs_newdata_max_abs_log", value=max_abs_log(mine, ref), note="有限格最大 |log 比|")
    rec("equiv", "n_cells_finite", value=int(np.isfinite(ref).sum()))
    log(f"[equiv] 完成 [{time.time()-t0:.0f}s]")
    if not same:
        flush()
        raise AssertionError("数组版与 newdata.total_share 不逐位一致,不得用于前向")
    flush()


# ============================================================================================ build
def _listing_dates(codes) -> np.ndarray:
    """上市日:stock_basic(前向截止日下,源到 08-20);缺失者用 baostock query_stock_basic 的 ipoDate(预注册 §4.3)。"""
    from afactor import newdata as ND
    li = ND.listing_info(codes)["list_date"]
    bs = pd.read_csv(CACHE / "forward" / "raw" / "bs_meta" / "query_stock_basic_20261010.csv")
    bs = bs[bs["type"] == 1]
    ex = bs["code"].str[:2].map({"sh": ".SH", "sz": ".SZ", "bj": ".BJ"})
    ipo = pd.Series(pd.to_datetime(bs["ipoDate"], errors="coerce").to_numpy(), index=bs["code"].str[3:] + ex)
    ipo = ipo[~ipo.index.duplicated()]
    filled = li.isna() & ipo.reindex(codes).notna().to_numpy()
    li = li.where(~filled, ipo.reindex(codes))
    return li.to_numpy("datetime64[D]"), int(filled.sum())


def stage_build():
    """前向面板的新口径股本 / 市值。输入与 fwd_build_panel.stage_daily 相同(并逐位核对),只换外推规则:
      日历     研究期 daily_basic 交易日(< 2025-01-02,默认截止日)∪ 前向面板日期 —— 与 newdata 一样从 2020-01-02 起累计送转;
      PIT 股本 tushare 资产负债表 total_share(前向截止日 + accept_source_gap,与 stage_daily 同一调用)与 baostock 季频 totalShare
               的合并(报告期更新者优先,同一报告期以 tushare 为准),报告期序号同步;
      流通股本 研究期段 = daily_basic;面板段 = 面板 float_share(<= 08-17 daily_basic,之后 baostock volume / turn,停牌沿用前值);
      未复权价 研究期段 = daily_basic;面板段 = daily_basic(<= 08-17)接 baostock 未复权收盘(与 assemble 的 splice 同式,
               不按 close_px 掩码 —— newdata 的送转识别用的也是 daily_basic 收盘,05-13 / 05-14 等 1 分钟缺口日的收盘在这里仍有值);
      上市日   stock_basic,08-20 以后的新股用 baostock ipoDate(预注册 §4.3)。
    市值 = 面板未复权收盘(assemble 同一 float64 拼接,按 close_px 有无掩码)× 股本,存 float32(与面板同)。不算任何收益。"""
    from afactor import newdata as ND, fundamental as FU, fwd_data as FD, config as C
    t0 = time.time()
    mv_old = pd.read_parquet(PANEL_DIR / "mv.parquet")
    dates, codes = mv_old.index, mv_old.columns
    T, N = len(dates), len(codes)
    # ---- 研究期段(默认截止日 2025-12-31,只用 < 2025-01-02 的行)
    tds = pd.DatetimeIndex(ND._load_daily_basic()["trade_date"].unique()).sort_values()
    pre = tds[tds < dates[0]]
    cal = pre.append(dates)
    assert cal.is_monotonic_increasing and cal.is_unique
    fs_pre = ND.float_share(pre, codes).to_numpy(np.float64)
    px_pre = ND.raw_close(pre, codes).to_numpy(np.float64)
    fs25_research = ND.float_share(dates[dates.year == 2025], codes)
    log(f"[build] 扩展日历 {cal[0].date()}~{cal[-1].date()}({len(pre)} + {T} 行) [{time.time()-t0:.0f}s]")
    # ---- 面板段
    aux = lambda n: pd.read_parquet(AUX / f"{n}.parquet").reindex(index=dates, columns=codes)
    late = np.asarray(dates > M1_END)
    px_f = aux("db_raw_close").copy()
    px_f.loc[late] = aux("bs_raw_close").loc[late]                      # assemble 的 splice("db_raw_close", "bs_raw_close")
    close_px = pd.read_parquet(PANEL_DIR / "close_px.parquet").reindex(index=dates, columns=codes)
    raw64 = px_f.where(np.isfinite(close_px.to_numpy()))                  # assemble:raw_close = splice(...).where(fin)
    fs_f = pd.read_parquet(PANEL_DIR / "float_share.parquet").reindex(index=dates, columns=codes).to_numpy(np.float64)
    r25 = np.asarray(dates.year == 2025)
    rec("build", "check_fs2025_panel_vs_research_daily_basic", value=eq_nan(fs_f[r25], fs25_research.to_numpy()),
        note="面板 2025 年流通股本 = 研究期 daily_basic(逐位)")
    # ---- PIT 股本(前向截止日;与 stage_daily 同一调用)
    with C.set_data_cutoff(FWD_CUTOFF, reason="前向检验市值修复敏感性(fwd_mvfix):扩展日历上重建前向 PIT 股本输入;只作特征,不算收益"):
        with C.accept_source_gap(FU.REPORT_TABLES, reason="同 fwd_build_panel.stage_daily:tushare 股本只到 2026-06-13,之后接 baostock 季频 totalShare"):
            rep = FU.build_report_level()
            rep = rep.assign(ref_qi=np.where(rep["total_share"].notna(),
                                             rep["end_date"].dt.year * 4 + (rep["end_date"].dt.quarter - 1), np.nan))
            sh_ts = FU._pit_daily(rep, "total_share", cal, codes)
            qi_ts = FU._pit_daily(rep, "ref_qi", cal, codes)
        lst, n_bs_ipo = _listing_dates(codes)
    rec("build", "check_sh_ts_vs_aux", value=eq_nan(sh_ts.loc[dates], aux("sh_ts")), note="扩展日历上的 tushare PIT 股本,面板段 vs fwd_aux/sh_ts")
    rec("build", "check_qi_ts_vs_aux", value=eq_nan(qi_ts.loc[dates], aux("qi_ts")), note="同上,报告期序号")
    pr = FD.load_profit()
    pr = pr[(pr["totalShare"] > 0) & pr["pubDate"].notna() & pr["statDate"].notna()].copy()
    pr["qi"] = pr["statDate"].dt.year * 4 + (pr["statDate"].dt.quarter - 1)
    pr = pr.sort_values(["code", "pubDate", "statDate"])
    pr["front"] = pr.groupby("code")["qi"].cummax()
    pr = pr[pr["qi"] >= pr["front"]].drop_duplicates(["code", "pubDate"], keep="last")

    def _pit(col):
        pv = pr.pivot_table(index="pubDate", columns="code", values=col, aggfunc="last").reindex(columns=codes)
        return pv.reindex(pv.index.union(cal)).ffill().reindex(cal)
    sh_bs, qi_bs = _pit("totalShare"), _pit("qi")
    use_bs = qi_bs.notna() & (qi_ts.isna() | (qi_bs > qi_ts))
    S = sh_ts.where(~use_bs, sh_bs)
    QI = qi_ts.where(~use_bs, qi_bs)
    rec("build", "check_QI_vs_aux_total_share_qi", value=eq_nan(QI.loc[dates], aux("total_share_qi")),
        note="合并后的报告期序号,面板段 vs fwd_aux/total_share_qi(stage_daily 落盘)")
    S_old = aux("total_share_pit").to_numpy(np.float64)                  # 旧口径(预注册 §4.3 / §7.4)= S × 除权因子比
    Sp = S.loc[dates].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mult_old = S_old / Sp
    rec("build", "check_old_S_nan_pattern", value=eq_nan(np.isfinite(S_old), np.isfinite(Sp)),
        note="旧口径股本与重建的 S 有值位置相同(旧口径 = S × 报告期末以来的除权因子比)")
    rec("build", "old_mult_range", value=f"{np.nanmin(mult_old):.4f} ~ {np.nanmax(mult_old):.4f}",
        note="旧口径倍数 = total_share_pit / S;>1 来自报告期末之后的除权(含现金分红)")
    mv_old_repro = (raw64.to_numpy(np.float64) * S_old).astype(np.float32)
    rec("build", "check_old_mv_reproduced", value=eq_nan(mv_old_repro, mv_old.to_numpy()),
        note="面板 mv.parquet = (未复权收盘 float64 拼接 × total_share_pit).astype(float32),逐位复现")
    log(f"[build] 输入就绪(baostock 股本接管 {int(use_bs.loc[dates].to_numpy().sum())} 格) [{time.time()-t0:.0f}s]")
    # ---- 新口径
    sh_arr = S.replace(0, np.nan).to_numpy(np.float64)                   # newdata 同式:股本 0 当缺失
    qi_arr = QI.to_numpy(np.float64)
    fs_arr = np.vstack([fs_pre, fs_f])
    px_arr = np.vstack([px_pre, px_f.to_numpy(np.float64)])
    new_ext, pre_ext = total_share_core(cal, sh_arr, qi_arr, fs_arr, px_arr, lst, return_pre=True)
    noipo_ext = total_share_core(cal, sh_arr, qi_arr, fs_arr, px_arr, np.full(N, np.datetime64("NaT"), "datetime64[D]"))
    steps = ND._songzhuan_steps(fs_arr, px_arr)
    k = len(pre)
    S_new = new_ext[k:]
    S_pre = pre_ext[k:]
    S_noipo = noipo_ext[k:]
    mv_new = (raw64.to_numpy(np.float64) * S_new).astype(np.float32)
    MV_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mv_new, index=dates, columns=codes).to_parquet(MV_DIR / "mv.parquet")
    pd.DataFrame(S_new.astype(np.float32), index=dates, columns=codes).to_parquet(MV_DIR / "total_share.parquet")
    pd.DataFrame(steps[k:], index=dates, columns=codes).astype(np.float32).to_parquet(MV_DIR / "_songzhuan_steps.parquet")
    json.dump(dict(built_at=time.strftime("%Y-%m-%d %H:%M:%S %z"), rule="newdata.total_share 新口径(songzhuan),数组版 total_share_core",
                   inputs="fwd_build_panel 同一组 PIT 股本 / 报告期 / 流通股本 / 未复权价;上市日 stock_basic + baostock ipoDate",
                   base_panel=str(PANEL_DIR), n_dates=T, n_codes=N),
              open(MV_DIR / "_meta.json", "w"), ensure_ascii=False, indent=1)
    log(f"[build] 写 {MV_DIR}/mv.parquet、total_share.parquet [{time.time()-t0:.0f}s]")

    # ---- 数据层对照(不涉及收益)
    a, b = mv_old.to_numpy(np.float64), mv_new.astype(np.float64)
    fa, fb = np.isfinite(a) & (a > 0), np.isfinite(b) & (b > 0)
    both = fa & fb
    r = np.full(a.shape, np.nan)
    r[both] = b[both] / a[both]
    lr = np.abs(np.log(r[both]))
    rec("build", "cells_mv_finite", old=int(fa.sum()), new=int(fb.sum()), note="面板 426 日 × 5,581 只")
    rec("build", "cells_new_only", value=int((fb & ~fa).sum()),
        note=f"旧口径 NaN(无 PIT 股本)、新口径按 newdata 以流通股本兜底的格;涉及 {int((fb & ~fa).any(axis=0).sum())} 只")
    rec("build", "cells_old_only", value=int((fa & ~fb).sum()))
    for th in (1e-3, 1e-2, 5e-2, 0.2, np.log(1.5)):
        rec("build", f"share_abs_log_ratio_gt_{th:.4g}", value=float((lr > th).mean()), note="两口径都有值的格中 |log(新/旧)| 超过阈值的比例")
    q = np.quantile(r[both], [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    rec("build", "ratio_new_over_old_quantiles", value="; ".join(f"{p_}:{v:.4f}" for p_, v in
                                                                zip(("0.1%", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "99.9%"), q)))
    rec("build", "share_new_lt_old_0.1pct", value=float((r[both] < 1 - 1e-3).mean()))
    rec("build", "share_new_gt_old_0.1pct", value=float((r[both] > 1 + 1e-3).mean()))
    # 分解(格子可重叠):新股加法、流通股本下界、旧口径的期后除权放大(新口径未外推的部分)
    ipo_cells = np.isfinite(S_new) & np.isfinite(S_noipo) & (np.abs(S_new / np.where(S_noipo > 0, S_noipo, np.nan) - 1) > 1e-12)
    lb_cells = np.isfinite(fs_f) & (np.isnan(S_pre) | (S_pre < fs_f)) & np.isfinite(Sp)
    with np.errstate(invalid="ignore", divide="ignore"):
        new_mult = S_new / Sp
    old_ex = np.isfinite(mult_old) & (mult_old > 1 + 1e-9)
    div_only = old_ex & (np.abs(new_mult - 1) < 1e-9)
    rec("build", "decomp_ipo_addition_cells", value=int((ipo_cells & both).sum()),
        note=f"新口径的新股加法改变了股本的格(旧口径无此项,用发行前股本);涉及 {int((ipo_cells & both).any(axis=0).sum())} 只;"
             f"这些格 新/旧 中位 {np.nanmedian(r[ipo_cells & both]) if (ipo_cells & both).any() else np.nan:.3f}")
    lb_mat = lb_cells & both & (fs_f > 1.01 * np.where(np.isfinite(S_pre), S_pre, 0.0))
    rec("build", "decomp_float_lower_bound_cells", value=int((lb_cells & both).sum()),
        note=f"有 PIT 股本、新口径取到流通股本下界的格(多数是全流通股 fs 与总股本只差舍入);其中 fs 超出外推股本 1% 以上的 "
             f"{int(lb_mat.sum())} 格 / {int(lb_mat.any(axis=0).sum())} 只(定增上市、转债转股等使流通股本超过陈旧财报股本)")
    rec("build", "decomp_old_exrights_mult_gt1_cells", value=int((old_ex & both).sum()),
        note=f"旧口径按报告期末以来除权因子放大(>1)的格;倍数中位 {np.nanmedian(mult_old[old_ex]):.4f}")
    rec("build", "decomp_old_dividend_only_cells", value=int((div_only & both).sum()),
        note=f"旧口径放大、新口径不外推(期后只有现金分红,或新口径没识别出送转)的格;旧倍数中位 "
             f"{np.nanmedian(mult_old[div_only]) if div_only.any() else np.nan:.4f}、p90 "
             f"{np.nanquantile(mult_old[div_only], 0.9) if div_only.any() else np.nan:.4f}(≈ 旧口径的股息放大)")
    rec("build", "listing_from_baostock_ipoDate", value=n_bs_ipo, note="stock_basic 缺上市日、用 baostock ipoDate 补的股票数")
    # ---- 前向窗口内的送转识别(07-02 起有交易所口径除权比例 bs_ratio = 前收 / preclose)
    ratio = aux("bs_ratio").to_numpy(np.float64)
    st_ = steps[k:]
    w = np.asarray(dates >= pd.Timestamp("2026-07-02"))
    ev = np.argwhere((ratio >= 1.05) & w[:, None])
    hit, cash, miss = 0, [], []
    for i, j in ev:
        lo, hi = max(i - 1, 0), min(i + 3, T - 1)
        if (st_[lo:hi + 1, j] > 1).any():
            hit += 1
            continue
        seg = fs_f[max(i - 3, 0):min(i + 11, T), j]
        with np.errstate(invalid="ignore", divide="ignore"):
            fj = np.nanmax(seg[1:] / seg[:-1]) if np.isfinite(seg).sum() > 1 else np.nan
        (cash if not (fj >= 1.01) else miss).append(f"{codes[j]}@{dates[i].date()}(除权比 {ratio[i, j]:.3f})")
    det = np.argwhere((st_ > 1) & w[:, None])
    fps = [(i, j) for i, j in det if not (ratio[max(i - 3, 0):min(i + 1, T - 1) + 1, j] >= 1.02).any()]
    rec("build", "fwd_exrights_ge_1.05_events_0702_1009", value=int(len(ev)),
        note=f"交易所口径除权比例(baostock 前收 / preclose)≥ 1.05 的事件;新口径在 [t−1, t+3] 内识别出送转 {hit} 个;"
             f"其余 {len(cash)} 个前后 2 周流通股本无跳升 = 高股息现金分红,新口径按规则不外推(旧口径按除权比放大股本 5% ~ 9%);"
             f"疑似漏判的送转 {len(miss)} 个:{'; '.join(miss)}")
    last = T - 1
    rec("build", "fwd_songzhuan_steps_0702_1009", value=int(len(det)),
        note=f"新口径识别出的送转步;其中前 3 日至后 1 日没有 ≥ 1.02 除权的(疑似误判:解禁 + 同期大跌){len(fps)} 个:"
             + "; ".join(f"{codes[j]}@{dates[i].date()}(步长 {st_[i, j]:.3f},10-09 新/旧 {b[last, j] / a[last, j]:.3f})" for i, j in fps))
    # 9 月信号日:决策池与中证 1000 成分内 |log(新/旧)| > 1% 的格按来源归类(先到先得)
    from afactor import fwd_weights as FW
    snap, _ = FW.csi1000_snapshot()
    in_csi = snap.reindex(codes).fillna(0.0).to_numpy() > 0
    fp_codes = np.zeros(N, bool)
    fp_codes[[j for _, j in fps]] = True
    with np.errstate(invalid="ignore", divide="ignore"):
        lrr = np.log(b / a)
    big = np.abs(lrr) > 0.01
    cats = [("新股加法", ipo_cells),
            ("流通股本下界(超出 1%)", lb_mat),
            ("疑似误判送转", fp_codes[None, :] & (new_mult > 1 + 1e-9)),
            ("旧口径期后除权放大、新口径不外推(现金分红为主)", div_only),
            ("两口径都外推、倍数不同", old_ex & (np.abs(new_mult - 1) >= 1e-9))]
    _pf = ForwardPanel()
    _sig = _pf.signal_days()
    _isg = dates.get_indexer(_sig)
    _U = _pf["universe"].reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
    for setname, M in (("V1000 决策池", _U[_isg]), ("中证 1000 成分", np.broadcast_to(in_csi, (len(_isg), N)))):
        base_ = M & both[_isg]
        left = big[_isg] & base_
        parts = []
        for nm, cm_ in cats:
            c_ = left & cm_[_isg]
            parts.append(f"{nm} {c_.sum() / max(base_.sum(), 1):.2%}(新/旧中位 {np.exp(np.median(lrr[_isg][c_])) if c_.any() else np.nan:.3f})")
            left = left & ~c_
        parts.append(f"其它 {left.sum() / max(base_.sum(), 1):.2%}")
        rec("build", f"sep_change_by_channel|{setname}", value=float((big[_isg] & base_).sum() / max(base_.sum(), 1)),
            note="9 月信号日该集合内 |log(新/旧)| > 1% 的格比例;按来源:" + ";".join(parts))
    js = np.flatnonzero(np.isfinite(fs_f[dates.get_loc(M1_END)]) & np.isfinite(fs_f[dates.get_loc(M1_END) + 1]))
    i0_ = dates.get_loc(M1_END)
    jump = np.abs(np.log(fs_f[i0_ + 1, js] / fs_f[i0_, js]))
    rec("build", "fs_splice_0817_0818_abs_log_gt_0.05", value=int((jump > 0.05).sum()),
        note=f"流通股本在 daily_basic -> baostock 接缝处 |log 跳变| > 5% 的股票数(共 {len(js)} 只两日都有值)")
    # ---- 9 月信号日:决策池内的截面变化
    pf = ForwardPanel()
    sig = pf.signal_days()
    U = pf["universe"].reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
    isg = dates.get_indexer(sig)
    rc, ch1, ch5 = [], [], []
    for i in isg:
        m = U[i] & both[i]
        ra, rb = pd.Series(a[i, m]).rank(), pd.Series(b[i, m]).rank()
        rc.append(float(np.corrcoef(ra, rb)[0, 1]))
        ch1.append(float((np.abs(np.log(b[i, m] / a[i, m])) > 0.01).mean()))
        ch5.append(float((np.abs(np.log(b[i, m] / a[i, m])) > 0.05).mean()))
    rec("build", "sep_pool_mv_rankcorr_min_median", value=f"{min(rc):.6f} / {np.median(rc):.6f}",
        note="9 月 21 个信号日 V1000 决策池内 旧 / 新 市值截面秩相关 最低 / 中位")
    rec("build", "sep_pool_share_abs_log_gt_1pct", value=float(np.mean(ch1)), note="9 月信号日决策池内 |log(新/旧)| > 1% 的格比例(日均)")
    rec("build", "sep_pool_share_abs_log_gt_5pct", value=float(np.mean(ch5)), note="同上,> 5%")
    newonly_sep = (fb & ~fa)[isg] & U[isg]
    rec("build", "sep_pool_new_only_cells", value=int(newonly_sep.sum()),
        note=f"9 月信号日决策池内旧口径无市值、新口径有的格;涉及 {int(newonly_sep.any(axis=0).sum())} 只")
    flush()
    log(f"[build] 完成 [{time.time()-t0:.0f}s]")


# ============================================================================================ xcheck
def stage_xcheck():
    """2025 年行(研究期数据,默认截止日)的交叉对照:前向旧 / 新口径 vs 研究期 ND.total_mv 新口径与旧(legacy)口径。
    说明前向面板从来没有 legacy 的解禁外推高估,以及前向新口径在研究期上与 newdata 新口径一致到什么程度。"""
    from afactor import newdata as ND
    t0 = time.time()
    old = pd.read_parquet(PANEL_DIR / "mv.parquet")
    new = pd.read_parquet(MV_DIR / "mv.parquet")
    d25 = old.index[old.index.year == 2025]
    rcodes = pd.read_parquet(CACHE / "daily" / "close_px.parquet").columns
    o = old.loc[d25, rcodes].to_numpy(np.float64)
    n = new.loc[d25, rcodes].to_numpy(np.float64)
    nd_new = ND.total_mv(d25, rcodes).to_numpy(np.float64)
    saved = ND.TOTAL_SHARE_METHOD
    try:
        ND.TOTAL_SHARE_METHOD = "legacy"                                 # 诊断开关:只供新旧对照
        nd_leg = ND.total_mv(d25, rcodes).to_numpy(np.float64)
    finally:
        ND.TOTAL_SHARE_METHOD = saved
    log(f"[xcheck] 研究期 ND.total_mv 新 / legacy 2025 年 [{time.time()-t0:.0f}s]")

    def cmp(x, y, lab, note=""):
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        lr = np.log(x[ok] / y[ok])
        rec("xcheck", f"{lab}|within_1pct", value=float((np.abs(lr) <= 0.01).mean()), note=note)
        rec("xcheck", f"{lab}|gt_1.05x", value=float((lr > np.log(1.05)).mean()))
        rec("xcheck", f"{lab}|gt_1.5x", value=float((lr > np.log(1.5)).mean()))
        rec("xcheck", f"{lab}|lt_1/1.05x", value=float((lr < -np.log(1.05)).mean()))
        rec("xcheck", f"{lab}|n_cells", value=int(ok.sum()))
    cmp(nd_leg, o, "ND_legacy_vs_fwd_old", "研究期 legacy(解禁外推)/ 前向旧口径;2025 年 × 研究面板 5,565 只")
    cmp(nd_new, o, "ND_new_vs_fwd_old", "研究期新口径 / 前向旧口径")
    cmp(n, nd_new, "fwd_new_vs_ND_new", "前向新口径 / 研究期新口径(差异只来自 PIT 股本合并了 baostock 季频股本)")
    rec("xcheck", "fwd_new_vs_ND_new|bitwise_share", value=float(np.mean((n == nd_new) | (np.isnan(n) & np.isnan(nd_new)))),
        note="逐位相同(含 NaN 位置)的格比例")
    flush()


# ============================================================================================ 面板
class MVPanel(ForwardPanel):
    """前向面板,只把市值换成新口径(cache/_tmp/fwd_mvfix_panel/mv.parquet):派生量 market_cap、_real(mv) 与 raw("mv"/"total_share")。
    其余字段、派生量、两个视图与 ForwardPanel 完全相同。"""

    def __init__(self, returns_view: bool = False):
        super().__init__(returns_view=returns_view)
        mv = pd.read_parquet(MV_DIR / "mv.parquet").astype(np.float32)
        assert mv.index.equals(self.dates) and mv.columns.equals(self.codes)
        self._derived["market_cap"] = mv
        for tag in ("", "@" + pd.Timestamp(FWD_CUTOFF).strftime("%Y%m%d")):
            self._derived[f"__real__mv__0{tag}"] = mv

    def raw(self, name: str) -> pd.DataFrame:
        if name in ("mv", "total_share"):
            if name not in self._cache:
                self._cache[name] = pd.read_parquet(MV_DIR / f"{name}.parquet").astype(np.float32)
            return self._cache[name]
        return super().raw(name)


_P: dict = {}


def panels():
    if "ret" not in _P:
        _P["feat"] = MVPanel()
        _P["ret"] = MVPanel(returns_view=True)
    return _P["feat"], _P["ret"]


def cost_model(p):
    from afactor.impact import CostModel
    from afactor import fwd_adapters as FA_
    if "cm" not in _P:
        with FA_.lifted_guards(raw_close=p.raw("raw_close")):
            _P["cm"] = CostModel(p, impact_y=1.0)
    return _P["cm"]


def score5():
    pf, _ = panels()
    if "s" not in _P:
        with forward_context("前向检验市值修复敏感性(fwd_mvfix):COMP_v5 打分(新口径市值面板;不进判定)"):
            _P["s"] = FC.comp_score(pf, cache_file=SCORE_FILE)
    return _P["s"]


def _save(df, path, what):
    assert_forward_index(df, what)
    (df.to_frame() if isinstance(df, pd.Series) else df).to_parquet(path)


# ============================================================================================ score
def stage_score():
    s_old = pd.read_parquet(TMP / "fwd_score5.parquet")
    with forward_context("前向检验市值修复敏感性(fwd_mvfix):COMP_v5 打分(新口径市值面板;不进判定)"):
        pf, _ = panels()
        s = FC.comp_score(pf, cache_file=SCORE_FILE, force=True)
    _P["s"] = s
    sig = pf.signal_days()
    rec("score", "score_bitwise_full_panel", value=eq_nan(s.to_numpy(), s_old.reindex_like(s).to_numpy()),
        note="新口径面板上重算的 COMP_v5 打分 vs fwd_score5.parquet(全部 426 日;24 个成员没有一个读市值)")
    d = np.abs(s.loc[sig].to_numpy(np.float64) - s_old.loc[sig].to_numpy(np.float64))
    rec("score", "score_max_abs_diff_sep", value=float(np.nanmax(d)) if np.isfinite(d).any() else 0.0,
        note="9 月信号日打分最大绝对差")
    flush()


# ============================================================================================ t12
def stage_t12():
    """T1 / T2 / 全池等权判据 + 三个基准,调用与 fwd_run_test.stage_t12 逐字相同,只换面板(新口径市值)。"""
    from afactor import strategy as ST, index_proxy as IP, fwd_weights as FW
    import fwd_run_test as RT
    pf, p = panels()
    s = score5()
    m = p.mask_from(SIG_START)
    cm = cost_model(pf)
    t0 = time.time()
    b5 = ST.benchmark(p, "V1000", "open10", date_mask=m, rebalance=5)["rets"]
    W = FW.csi1000_close_weights(p)
    rb_csi, Wx = IP.exec_time_bench_returns(p, "open10", W, return_weights=True)
    rb_csi = rb_csi.reindex(b5.index)
    real = FW.real_index_on_signal_days(p.dates, "open10").reindex(b5.index)
    B = pd.DataFrame({"pool5": b5, "csi1000_exec": rb_csi, "real000852": real})
    _save(B, BENCH_FILE, "基准(mvfix)")
    np.save(TMP / "fwd_mvfix_csi1000_Wx.npy", Wx[np.asarray(p.dates >= SIG_START)].astype(np.float32))
    W_old = pd.read_parquet(TMP / "fwd_csi1000_W.parquet")
    rec("t12", "csi1000_W_bitwise", value=eq_nan(W.loc[W.index >= SIG_START].to_numpy(), W_old.to_numpy()),
        note="中证 1000 实际成分收盘权重(快照按复权收盘反推,不读市值)vs fwd_csi1000_W.parquet")
    B_old = pd.read_parquet(TMP / "fwd_bench.parquet")
    rec("t12", "bench_max_abs_diff", value=float(np.nanmax(np.abs(B.to_numpy() - B_old.reindex_like(B).to_numpy()))),
        note="三个基准逐日收益 vs fwd_bench.parquet")
    runs = {}
    for plan, kw in RT.PLANS.items():
        for cost, cmx in (("old", None), ("Y1", cm)):
            r = ST.run_strategy(s, p, "V1000", "open10", date_mask=m, cost_model=cmx, record=True, **kw)
            assert r["rets"].index.equals(b5.index)
            runs[f"{plan}|{cost}"] = r
            ST.save_record(r, f"{PRE}_{plan}_{cost}")
    const = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype=np.float32)
    r = ST.run_strategy(const, p, "V1000", "open10", scheme="top", n=100000, rebalance=5, date_mask=m,
                        cost_model=cm, record=True)
    ST.save_record(r, f"{PRE}_CHK_poolew_Y1")
    runs["CHK_poolew|Y1"] = r
    D = pd.DataFrame({k: v["rets"] for k, v in runs.items()})
    _save(D, RETS_FILE, "策略日收益(mvfix)")
    for k, v in runs.items():
        assert_forward_index(v["record"]["daily"], k)
    S_old = pd.read_parquet(TMP / "fwd_strategy_rets.parquet")
    for k in D.columns:
        rec("t12", f"rets_max_abs_diff[{k}]", value=float(np.nanmax(np.abs(D[k].to_numpy() - S_old[k].reindex(D.index).to_numpy()))),
            note="新口径面板 vs 原前向逐日收益")
    log(f"[t12] 完成 [{time.time()-t0:.0f}s]")
    flush()


# ============================================================================================ risk
def stage_risk():
    from afactor.barra import RiskModel
    from afactor import fwd_adapters as FA_
    pf, _ = panels()
    t0 = time.time()
    with forward_context("前向检验市值修复敏感性(fwd_mvfix):风险模型逐日估计(2026-01~08 的因子收益只作估计输入,不报告)"):
        with FA_.lifted_guards():
            rm = RiskModel(cache_dir=RISK_DIR).build(pf, "open10", force=True)
    sig = pf.signal_days()
    av = [rm.available(d) for d in sig]
    rec("risk", "first_forecast", value=str(rm.meta.get("first_forecast")),
        note=f"前向信号日可用 {sum(av)}/{len(sig)};缓存 {RISK_DIR}(因子收益只作估计输入,不报告)")
    log(f"[risk] 完成 [{time.time()-t0:.0f}s]")
    flush()


def load_rm(cache_dir=RISK_DIR):
    from afactor.barra import RiskModel
    from afactor import fwd_adapters as FA_
    with FA_.lifted_guards():
        return RiskModel.load("open10", cache_dir=cache_dir)


# ============================================================================================ t3
def _run_t3(p, pf, s, rm, cm, prefix=None):
    """fwd_run_test.stage_t3 同一调用(IndexEnhancer + simulate,5 相位);prefix 给定时落盘 record 与求解记录。"""
    from afactor import strategy as ST, optimizer as OPT, fwd_weights as FW
    import fwd_run_test as RT
    W = FW.csi1000_close_weights(p)
    U = ST.universe(p, "V1000")
    icd = pd.Series(RT.IC_FROZEN, index=p.dates)
    m = p.mask_from(SIG_START)
    out = []
    for ph in range(5):
        t0 = time.time()
        enh = OPT.IndexEnhancer(p, rm, W, s, icd, U, OPT.OptConfig(), tag="fwd_V1")
        res = ST.simulate(enh, p, "open10", RT.CAPITAL, rebalance=5, date_mask=m, phase=ph, cost_model=cm, record=True)
        assert_forward_index(res["rets"], f"T3 ph{ph}")
        lg = enh.log_frame()
        if prefix:
            ST.save_record(res, f"{prefix}_ph{ph}")
            lg.to_parquet(TMP / f"{prefix}_ph{ph}_solver.parquet")
        out.append((res, lg))
        log(f"[t3] ph{ph}:求解 {len(lg)} 次,状态 {lg['status'].value_counts().to_dict()} [{time.time()-t0:.0f}s]")
    return out


def load_t3(prefix: str) -> dict:
    """fwd_run_test.load_t3 同一合并口径(日收益 = 相位均值,持仓 = 相位均值),只换文件前缀。"""
    from afactor.strategy import _REC_COLS
    ds, Hs, sl = [], [], []
    for ph in range(5):
        pre = TMP / f"{prefix}_ph{ph}"
        ds.append(pd.read_parquet(f"{pre}_daily.parquet").assign(phase=ph))
        Hs.append(pd.read_parquet(f"{pre}_holdings.parquet"))
        sl.append(pd.read_parquet(f"{pre}_solver.parquet").assign(phase=ph))
    byp = pd.concat(ds)
    idx = ds[0].index
    num = [c for c in _REC_COLS if c != "rebalance"]
    daily = byp.groupby(level=0)[num].mean().reindex(idx)
    cols = Hs[0].columns
    for h in Hs[1:]:
        cols = cols.union(h.columns)
    H = sum(h.reindex(index=idx, columns=cols).fillna(0.0).astype(np.float64) for h in Hs) / len(Hs)
    return dict(daily=daily, rets=daily["ret"], by_phase=byp, H=H, solver=pd.concat(sl, ignore_index=True))


def stage_t3():
    import fwd_run_test as RT
    t0 = time.time()
    # 1) 管线复现:原面板 + 原风险模型(cache/_tmp/fwd_risk,只读)+ 原打分,必须逐位复现原 T3 逐日收益(不落盘)
    p0, pf0 = ForwardPanel(returns_view=True), ForwardPanel()
    s0 = pd.read_parquet(TMP / "fwd_score5.parquet")
    rm0 = load_rm(TMP / "fwd_risk")
    from afactor.impact import CostModel
    from afactor import fwd_adapters as FA_
    with FA_.lifted_guards(raw_close=pf0.raw("raw_close")):
        cm0 = CostModel(pf0, impact_y=1.0)
    rep = _run_t3(p0, pf0, s0, rm0, cm0, prefix=None)
    r_rep = pd.concat([x[0]["rets"] for x in rep], axis=1).mean(axis=1)
    V0 = RT.load_t3()
    rec("t3", "harness_reproduces_original_T3_max_abs_diff", value=float(np.nanmax(np.abs(r_rep.to_numpy() - V0["rets"].reindex(r_rep.index).to_numpy()))),
        note="原面板 + 原风险模型重跑 5 相位,相位均值日收益 vs 原 fwd_T3_V1(应为 0)")
    del rep, rm0
    # 2) 新口径:新面板 + 新风险模型
    pf, p = panels()
    s = score5()
    rm = load_rm(RISK_DIR)
    cm = cost_model(pf)
    _run_t3(p, pf, s, rm, cm, prefix=f"{PRE}_T3_V1")
    log(f"[t3] 完成 [{time.time()-t0:.0f}s]")
    flush()


# ============================================================================================ t5
def stage_t5():
    """fwd_run_test.stage_t5 同式(te_pred_series 原样调用),风险模型换成新口径。"""
    import fwd_run_test as RT
    pf, p = panels()
    rm = load_rm(RISK_DIR)
    B = pd.read_parquet(BENCH_FILE)
    Wx = np.load(TMP / "fwd_mvfix_csi1000_Wx.npy")
    wdates = p.dates[p.dates >= SIG_START]
    V = load_t3(f"{PRE}_T3_V1")
    out = {}
    for nm, H, r in (("T5_V1", V["H"], V["rets"]),
                     ("T5_poolew", pd.read_parquet(TMP / f"{PRE}_CHK_poolew_Y1_holdings.parquet"),
                      pd.read_parquet(TMP / f"{PRE}_CHK_poolew_Y1_daily.parquet")["ret"])):
        Wv = pd.DataFrame(Wx, index=wdates, columns=p.codes).reindex(H.index).to_numpy(np.float64)
        tp = RT.te_pred_series(H, Wv, rm, p.dates, p.codes)
        act = (r - B["csi1000_exec"]).reindex(tp.index)
        out[f"{nm}|te_pred_ann"] = tp
        out[f"{nm}|active"] = act
        out[f"{nm}|z"] = act / (tp / np.sqrt(TD))
    T = pd.DataFrame(out)
    _save(T, TE_FILE, "T5(mvfix)")
    log(f"[t5] V1 事前 TE 中位 {T['T5_V1|te_pred_ann'].median():.4f}")


# ============================================================================================ diag
def stage_diag():
    """风险模型与 T3 组合的变化(9 月信号日;只看暴露、风险预测与持仓,不看任何预热段收益)。"""
    import fwd_run_test as RT
    pf, p = panels()
    sig = pf.signal_days()
    rm0, rm1 = load_rm(TMP / "fwd_risk"), load_rm(RISK_DIR)
    assert rm0.style_names == rm1.style_names
    isg = p.dates.get_indexer(sig)
    for k, nm in enumerate(rm0.style_names):
        rcs, mad = [], []
        for i in isg:
            m = np.asarray(rm0.mask[i]) & np.asarray(rm1.mask[i])
            x0, x1 = np.asarray(rm0.X[i])[m, k], np.asarray(rm1.X[i])[m, k]
            ok = np.isfinite(x0) & np.isfinite(x1)
            rcs.append(float(pd.Series(x0[ok]).rank().corr(pd.Series(x1[ok]).rank())))
            mad.append(float(np.median(np.abs(x1[ok] - x0[ok]))))
        rec("diag", f"exposure|{nm}", value=f"秩相关 最低 {min(rcs):.4f} / 中位 {np.median(rcs):.4f};|Δ| 中位 {np.median(mad):.4f}",
            note="9 月信号日,两个风险模型都可用的覆盖池内;旧 = cache/_tmp/fwd_risk,新 = fwd_mvfix_risk")
    n0 = [int(np.asarray(rm0.mask[i]).sum()) for i in isg]
    n1 = [int(np.asarray(rm1.mask[i]).sum()) for i in isg]
    rec("diag", "risk_mask_count_sep_median", old=float(np.median(n0)), new=float(np.median(n1)), note="9 月信号日风险模型可用股票数中位")
    sp = []
    for i in isg:
        m = np.asarray(rm0.mask[i]) & np.asarray(rm1.mask[i])
        sp.append(float(np.median(np.sqrt(np.asarray(rm1.spec[i])[m] / np.asarray(rm0.spec[i])[m]))))
    rec("diag", "spec_vol_ratio_new_over_old_median", value=float(np.median(sp)), note="特质波动 新/旧 的逐日截面中位,再取 9 月中位")
    # T3 组合
    V0, V1 = RT.load_t3(), load_t3(f"{PRE}_T3_V1")
    H0, H1 = V0["H"], V1["H"]
    cols = H0.columns.union(H1.columns)
    a0 = H0.reindex(columns=cols).fillna(0.0)
    a1 = H1.reindex(index=H0.index, columns=cols).fillna(0.0)
    hd = 0.5 * (a1 - a0).abs().sum(axis=1)
    rec("diag", "T3_holdings_half_L1_diff_eval_mean", value=float(hd.iloc[SKIP:].mean()),
        note="评估窗内(去开头 4 天)相位平均持仓的 ½Σ|Δw| 日均;0 = 完全相同,1 = 完全不重合")
    s0, s1 = V0["solver"], V1["solver"]
    rec("diag", "T3_solver_status", old="; ".join(f"{k}:{v}" for k, v in s0["status"].value_counts().items()),
        new="; ".join(f"{k}:{v}" for k, v in s1["status"].value_counts().items()))
    for c in ("te_ex_ante", "alpha_ex_ante", "n_hold", "bench_w_in", "exp:Size 市值", "exp:NonlinSize 非线性市值",
              "exp:Beta 市场敏感", "exp:ResidVol 残差波动", "exp:Liquidity 流动性"):
        rec("diag", f"T3_solver_mean|{c}", old=float(s0[c].mean()), new=float(s1[c].mean()),
            note="21 次求解(5 相位)的均值;exp:* 为求解时的事前主动风格暴露(各自风险模型下)")
    rec("diag", "T3_te_cap_binding_count", old=int((s0["te_ex_ante"] >= 0.0499).sum()), new=int((s1["te_ex_ante"] >= 0.0499).sum()),
        note="事前 TE ≥ 4.99%(顶在 5% 上限)的求解次数")
    # 新口径下才有市值、进入了风险模型覆盖池的股票
    mv0 = pd.read_parquet(PANEL_DIR / "mv.parquet")
    mv1 = pd.read_parquet(MV_DIR / "mv.parquet")
    newonly = (~np.isfinite(mv0.to_numpy())) & np.isfinite(mv1.to_numpy())
    cnt = [int((newonly[i] & np.asarray(rm1.mask[i])).sum()) for i in isg]
    held = [c for c in H1.columns if newonly[:, p.codes.get_loc(c)][isg].any() and H1[c].abs().sum() > 0]
    rec("diag", "new_only_mv_in_risk_mask_sep", value=f"日均 {np.mean(cnt):.1f}、最多 {max(cnt)}",
        note=f"9 月信号日:旧口径无市值(无 PIT 股本,多为新股)、新口径以流通股本兜底后进入风险模型可用集的股票;新 T3 曾持有其中 {len(held)} 只 {held[:10]}")
    flush()


# ============================================================================================ report
def _window_stats(rs: pd.Series, rb: pd.Series) -> dict:
    """fwd_report.window_stats 同式(只取本节用到的量)。"""
    df = pd.DataFrame({"strategy": rs, "benchmark": rb})
    v, n = M5.forward_window_excess(df, skip_head=SKIP, how="compound")
    e = df.dropna().iloc[SKIP:]
    assert n == 17, n
    d = (e["strategy"] - e["benchmark"]).to_numpy()
    return dict(window_excess=v, strat_cum=float(np.prod(1 + e["strategy"]) - 1),
                bench_cum=float(np.prod(1 + e["benchmark"]) - 1), te_realized=float(d.std(ddof=1) * np.sqrt(TD)))


def _corr_peak(rs, rb):
    a, b = rs.align(rb, join="inner")
    cs = {k: float(a.corr(b.shift(k))) for k in (-2, -1, 0, 1, 2)}
    return max(cs, key=lambda k: cs[k] if np.isfinite(cs[k]) else -9)


def stage_report():
    import fwd_run_test as RT
    R = pd.read_csv(LEDGER / "fwd_results.csv")
    led = lambda o, c, b: R[(R["section"] == "reading") & (R["object"] == o) & (R["cost"] == c) & (R["bench"] == b)].iloc[0]
    S0, S1 = pd.read_parquet(TMP / "fwd_strategy_rets.parquet"), pd.read_parquet(RETS_FILE)
    B0, B1 = pd.read_parquet(TMP / "fwd_bench.parquet"), pd.read_parquet(BENCH_FILE)
    sig = B0.index
    assert len(sig) == 21 and sig[0] == SIG_START and B1.index.equals(sig)
    items = [("P1", "T1_excl", "old", "pool5", "pool5"), ("P2", "T1_excl", "Y1", "pool5", "pool5"),
             ("P3", "T2_top20", "old", "pool5", "pool5"), ("T2 Y=1(描述)", "T2_top20", "Y1", "pool5", "pool5"),
             ("T1 旧成本 对中证1000(次要)", "T1_excl", "old", "csi1000_actual_exec", "csi1000_exec"),
             ("T2 旧成本 对中证1000(次要)", "T2_top20", "old", "csi1000_actual_exec", "csi1000_exec")]
    for lab, o, c, bk, bcol in items:
        st0 = _window_stats(S0[f"{o}|{c}"], B0[bcol])
        st1 = _window_stats(S1[f"{o}|{c}"], B1[bcol])
        L = led(o, c, bk)
        key = L["hist_key"]
        repro = abs(st0["window_excess"] - L["window_excess"])
        rec("reading", f"{lab}|{o}|{c}|{bk}", old=st0["window_excess"], new=st1["window_excess"],
            old_pct_ref=M5.percentile_in(st0["window_excess"], dist(key)) if isinstance(key, str) else None,
            new_pct_ref=M5.percentile_in(st1["window_excess"], dist(key)) if isinstance(key, str) else None,
            note=f"窗口复利超额;旧 = 原前向序列重算(与 fwd_results.csv 差 {repro:.1e});{PCT_NOTE}")
    # T3
    V0, V1 = RT.load_t3(), load_t3(f"{PRE}_T3_V1")
    for lab, bk, bcol, key in (("P4", "csi1000_actual_exec", "csi1000_exec", "T3_V1|Y1|proxy_exec|n17"),
                               ("T3 对同频池(描述)", "pool5", "pool5", "T3_V1|Y1|pool5|n17"),
                               ("T3 对真实000852(描述)", "real000852", "real000852", "T3_V1|Y1|real000852|n17")):
        st0 = _window_stats(V0["rets"].reindex(sig), B0[bcol])
        st1 = _window_stats(V1["rets"].reindex(sig), B1[bcol])
        L = led("T3_V1", "Y1", bk)
        rec("reading", f"{lab}|T3_V1|Y1|{bk}", old=st0["window_excess"], new=st1["window_excess"],
            old_pct_ref=M5.percentile_in(st0["window_excess"], dist(key)),
            new_pct_ref=M5.percentile_in(st1["window_excess"], dist(key)),
            note=f"窗口复利超额;旧 = 原前向序列重算(与 fwd_results.csv 差 {abs(st0['window_excess'] - L['window_excess']):.1e});{PCT_NOTE}")
        if bk == "csi1000_actual_exec":
            rec("reading", "P4|T3 策略评估窗累计", old=st0["strat_cum"], new=st1["strat_cum"], note="基准累计不变(中证 1000 实际成分不读市值)")
            rec("reading", "P4|T3 实现 TE(年化)", old=st0["te_realized"], new=st1["te_realized"])
    # T5
    T0, T1 = pd.read_parquet(TMP / "fwd_te_daily.parquet").reindex(sig), pd.read_parquet(TE_FILE).reindex(sig)
    q5 = QREF[(QREF["target"] == "T5_V1") & (QREF["n"] == 17)].iloc[0]
    for lab, tag in (("P6", "T5_V1"), ("全池等权判据组合 b_d(描述)", "T5_poolew")):
        key = f"{tag}|Y1|proxy_exec|n17|b_d"
        bd0 = float(T0[f"{tag}|z"].iloc[SKIP:].std(ddof=1))
        bd1 = float(T1[f"{tag}|z"].iloc[SKIP:].std(ddof=1))
        L = led(tag, "Y1", "csi1000_actual_exec")
        inb = lambda v: bool(q5["chi2_band_lo"] <= v <= q5["chi2_band_hi"])
        rec("reading", f"{lab}|{tag}|b_d", old=bd0, new=bd1, old_pct_ref=M5.percentile_in(bd0, dist(key)),
            new_pct_ref=M5.percentile_in(bd1, dist(key)),
            note=f"z 的样本标准差;旧与 fwd_results.csv 差 {abs(bd0 - L['window_excess']):.1e};χ² 区间 [{q5['chi2_band_lo']:.4f}, "
                 f"{q5['chi2_band_hi']:.4f}] 内:旧 {inb(bd0)} / 新 {inb(bd1)};{PCT_NOTE}")
        tp0, tp1 = T0[f"{tag}|te_pred_ann"].iloc[SKIP:], T1[f"{tag}|te_pred_ann"].iloc[SKIP:]
        rec("reading", f"{lab}|{tag}|事前 TE 均值", old=float(tp0.mean()), new=float(tp1.mean()))
        a0, a1 = T0[f"{tag}|active"].iloc[SKIP:], T1[f"{tag}|active"].iloc[SKIP:]
        rec("reading", f"{lab}|{tag}|实现 TE / 事前 TE", old=float(a0.std(ddof=1) * np.sqrt(TD) / tp0.mean()),
            new=float(a1.std(ddof=1) * np.sqrt(TD) / tp1.mean()))
    # 引擎断言(新口径 T3;T1 / T2 / CHK 逐位不变时自然成立)
    cb = V1["daily"]
    rec("engine", "T3_new|exposure_max", value=float(cb["exposure"].max()), note="≤ 1")
    rec("engine", "T3_new|corr_peak_k_vs_csi1000", value=int(_corr_peak(V1["rets"].reindex(sig), B1["csi1000_exec"])), note="应为 0")
    rec("engine", "T3_new|index_first", value=str(V1["rets"].index.min().date()), note="首个索引 ≥ 2026-09-01")
    for lab, d in (("old", V0["daily"]), ("new", cb)):
        ev = d.sort_index().iloc[SKIP:]
        rec("engine", f"T3_{lab}|cost_eval_total", value=float(ev["cost_total"].sum()),
            note=f"评估窗成本合计;日均换手 {ev['turnover'].mean():.4f}")
    flush()


# ============================================================================================ 审查 P2 补充:冻结时间线
def stage_freeze_timeline():
    """冻结校验时间线(审查 P2):逐条读 logs/fwd_verify_freeze.log(只读),再只读调用一次 verify() 记当前快照。
    不覆盖 section=freeze(10:14 的只读快照原样保留)。并行的 mvfix 下游会话仍可能继续改动清单内的 aux 文件,
    所以这里的"当前"只代表本次调用的时刻。"""
    import fwd_verify_freeze as VF
    man = json.load(open(VF.DEFAULT_MANIFEST, encoding="utf-8"))
    cat = {e["path"]: e["category"] for e in man["files"]}
    for ln in open(ROOT / "logs" / "fwd_verify_freeze.log", encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        e = json.loads(ln)
        mm = [f"{x}({cat.get(x, '?')})" for x in e.get("mismatched", [])]
        rec("freeze_timeline", f"log|{e['time']}", value=int(e["exit_code"]),
            note=f"{e['n_ok']}/{e['n_files']} 一致;不一致 {len(mm)} 个" + (f":{', '.join(mm)}" if mm else "")
                 + f";缺失 {len(e.get('missing', []))}(logs/fwd_verify_freeze.log)")
    r = VF.verify(VF.DEFAULT_MANIFEST, VF.ROOT)
    mm = [f"{e['path']}({e['category']},mtime {e.get('now_mtime')})" for e in r.get("mismatched", [])]
    rec("freeze_timeline", f"readonly_verify|{r['checked_at']}", value=int(r["exit_code"]),
        note=f"{r['n_ok']}/{r['n_files']} 一致;不一致 {len(mm)} 个:{', '.join(mm)};缺失 {len(r.get('missing', []))}"
             f"(只读调用 verify(),未写冻结日志;并行会话之后的改动还会继续增加 aux 不一致)")
    flush()


# ============================================================================================ 审查 P2 补充:送转标签同口径
def stage_sz_label():
    """送转误判率的同口径对照(审查 P2)。研究期 TOTAL_MV_FIX §2.2(scripts/mvfix_total_mv.float_events)的标签原样用到前向窗口:
      事件 = 流通股本相对上一个有效观测跳升 r >= 1.05,且标签可得;
      标签 = 复权因子在 t 或 t−1 跳升 log ∈ [log r − 0.015, log r + 0.10](允许现金分红再多 0~10%)= 送转除权;
      FP = 新口径识别为送转(_songzhuan_steps > 1)但标签为否;精度 = TP / (TP + FP)。
    前向窗口没有新浪复权因子;复权因子在 t 的跳升 = 前收 / 交易所 preclose,即 fwd_aux/bs_ratio(07-02 起有),是同一个量。
    另取研究期逐条事件 mvfix_songzhuan_events.csv 中每年 07-02 ~ 10-09 同季段作参照。只读面板 / fwd_aux / 账本,不算收益。"""
    from afactor import newdata as ND
    mv_old = pd.read_parquet(PANEL_DIR / "mv.parquet")
    dates, codes = mv_old.index, mv_old.columns
    fs = pd.read_parquet(PANEL_DIR / "float_share.parquet").reindex(index=dates, columns=codes).to_numpy(np.float64)
    steps = pd.read_parquet(MV_DIR / "_songzhuan_steps.parquet").reindex(index=dates, columns=codes).to_numpy(np.float64)
    ratio = pd.read_parquet(AUX / "bs_ratio.parquet").reindex(index=dates, columns=codes).to_numpy(np.float64)
    mv_new = pd.read_parquet(MV_DIR / "mv.parquet").reindex(index=dates, columns=codes).to_numpy(np.float64)
    a = mv_old.to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = fs / ND._ffill_shift(fs)
        la0 = np.log(ratio)
    la1 = np.vstack([np.full((1, len(codes)), np.nan), la0[:-1]])
    w = np.asarray(dates >= pd.Timestamp("2026-07-02"))
    avail = np.isfinite(la0) | np.isfinite(la1)
    ii, jj = np.nonzero((np.nan_to_num(r) >= 1.05) & w[:, None] & avail)
    lr = np.log(r[ii, jj])
    lab = np.zeros(len(ii), bool)
    for la in (la0, la1):
        x = la[ii, jj]
        lab |= (x >= lr - 0.015) & (x <= lr + 0.10)
    sz = steps[ii, jj] > 1.0
    tp, fp, fn, tn = int((sz & lab).sum()), int((sz & ~lab).sum()), int((~sz & lab).sum()), int((~sz & ~lab).sum())
    last = len(dates) - 1
    fps = [f"{codes[j]}@{dates[i].date()}(r {r[i, j]:.3f},10-09 新/旧 {mv_new[last, j] / a[last, j]:.3f})"
           for i, j, s_, l_ in zip(ii, jj, sz, lab) if s_ and not l_]
    rec("sz_label", "fwd_0702_1009|confusion", value=f"事件 {len(ii)};TP {tp} FP {fp} FN {fn} TN {tn}",
        note=f"研究期标签原样(复权因子跳升以交易所 bs_ratio 代);识别为送转 {tp + fp} 个,FP 占 {fp / max(tp + fp, 1):.1%},"
             f"召回 {tp / max(tp + fn, 1):.3f};FP:{'; '.join(fps)}")
    rec("sz_label", "fwd_0702_1009|fp_share_of_detected", value=fp / max(tp + fp, 1), note="FP / (TP + FP)")
    rec("sz_label", "fwd_0702_1009|fp_r_ge_1.1", value=int(sum(1 for i, j, s_, l_ in zip(ii, jj, sz, lab) if s_ and not l_ and r[i, j] >= 1.1)),
        note="FP 中跳升比例 r >= 1.1 的个数(新口径股本按 r 外推,量级即 r − 1)")
    # 研究期参照(同一标签;TOTAL_MV_FIX 的逐条事件账本)
    ev = pd.read_csv(LEDGER / "mvfix_songzhuan_events.csv", parse_dates=["date"])
    e5 = ev[(ev["r"] >= 1.05) & ev["adj_avail"]]
    for lab_, g in [("研究期全期", e5)] + [(f"{y} 年 07-02~10-09", e5[(e5["date"] >= f"{y}-07-02") & (e5["date"] <= f"{y}-10-09")])
                                          for y in range(2020, 2026)]:
        s_, l_ = g["is_songzhuan_rule"].astype(bool), g["adj_label"].astype(bool)
        tp_, fp_ = int((s_ & l_).sum()), int((s_ & ~l_).sum())
        rec("sz_label", f"research|{lab_}", value=fp_ / max(tp_ + fp_, 1),
            note=f"FP / (TP + FP) = {fp_} / {tp_ + fp_};FP 中 r >= 1.1 的 {int((s_ & ~l_ & (g['r'] >= 1.1)).sum())} 个、"
                 f"r >= 1.2 的 {int((s_ & ~l_ & (g['r'] >= 1.2)).sum())} 个(research/ledger/mvfix_songzhuan_events.csv)")
    flush()


# ============================================================================================ 审查 P2 补充:PEAD D1
def stage_pead():
    """P5′ D1 补充读数在新口径市值下重算(审查 P2)。B 臂匹配变量之一是同日合格池内 log 市值分位
    (fwd_pead_run.stage_events:match_inputs(p, Q, p.raw("mv"), p.raw("circ_mv"))),换成新口径;合格池(valid / ADV / 上市天数 / ST)、
    事件、E 臂、收益面板都不读市值。同一段代码先用原面板跑一遍,必须逐位复现原批次表与原 D̄ / 账户层(管线自检),再用新口径面板重算。
    只用 t_signal ∈ [09-01, 09-30]、收益到 10-09 的槽位(与原 stage_events / accounts / eventlvl 相同)。
    产物只写 cache/_tmp/fwd_mvfix_pead_*;不改 fwd_pead_* 原产物、fwd_pead_balance.csv / fwd_pead_events.csv。"""
    import fwd_pead_run as PR
    import pead_build_batches as PBB
    import pead_run_arms as PRA
    from afactor import fwd_pead as FP, pead_engine as PE
    R = pd.read_csv(LEDGER / "fwd_results.csv")
    Ld = R[(R["section"] == "reading") & (R["object"] == "T4_PEAD") & (R["reading"] == "P5(D1 口径)")].iloc[0]
    La = {r_.cost: r_.window_excess for r_ in R[(R["section"] == "reading") & (R["object"] == "T4_PEAD_account")].itertuples()}
    B0_file = pd.read_parquet(TMP / "fwd_pead_splice_batches.parquet")
    bal0 = pd.read_csv(LEDGER / "fwd_pead_balance.csv")
    pr = ForwardPanel(returns_view=True)
    T = len(pr.dates)
    sig = pd.read_parquet(TMP / "fwd_bench.parquet").index
    twap = pr.raw("twap_open10").to_numpy(np.float64)
    amt10 = pr.raw("amt_open10").to_numpy(np.float64)
    dead = (pr["one_word_up"] | pr["one_word_down"]).to_numpy(bool)
    buyable = np.isfinite(twap) & (twap > 0) & ~dead & (amt10 > 0)
    twap_ff = pd.DataFrame(twap).ffill().to_numpy()
    raw_px = pr.raw("raw_close").to_numpy(np.float64)
    adj_px = pr.raw("close_px").to_numpy(np.float64)
    star = PE.star_mask(pr.codes)

    def r_h(Tp, j, h):                                   # fwd_pead_run.stage_eventlvl 同式
        Tp = np.asarray(Tp); j = np.asarray(j)
        out = np.full(len(Tp), np.nan)
        ok = (j >= 0) & (Tp + h < T)
        Te, je = Tp[ok], j[ok]
        r = twap_ff[Te + h, je] / twap[Te, je] - 1.0
        out[ok] = np.where(buyable[Te, je], r, 0.0)
        return out

    res = {}
    saved = PR.ForwardPanel
    for tag, cls in (("old", ForwardPanel), ("new", MVPanel)):
        t0 = time.time()
        PR.ForwardPanel = cls
        try:
            p, cal, offset, inc, ev, pool, Q, Qn = PR._prep("splice")
        finally:
            PR.ForwardPanel = saved
        qv, raw_vals, ind_lab = FP.match_inputs(p, Q, p.raw("mv"), p.raw("circ_mv"))
        B, M = FP.build_batches(pool, Qn, qv, ind_lab, p.dates, p.codes, offset, T_range=PR.WIN_T)
        jj = B["j"].clip(lower=0).to_numpy(); ii = B["t_pos"].to_numpy()
        ok = (B["j"] >= 0).to_numpy()
        V = pd.DataFrame({"arm": B["arm"], "rep": B["rep"], "t_signal": B["t_signal"], "slot": B["slot"],
                          "T_pos": B["T_pos"], "ok": ok, "ind": np.where(ok, ind_lab[jj], -99),
                          "logmv": np.where(ok, raw_vals[ii, jj, 0], np.nan),
                          "logadv": np.where(ok, raw_vals[ii, jj, 1], np.nan),
                          "vol": np.where(ok, raw_vals[ii, jj, 2], np.nan), "level": B["level"]})
        sw = PBB.WINDOWS
        PBB.WINDOWS = {"fwd": ("2026-09-01", "2026-09-30")}
        try:
            bal = PBB.balance_table(V[V["arm"].isin(["E", "B", "B@tol05"])])
        finally:
            PBB.WINDOWS = sw
        bal = bal[bal["control"].isin(["B", "B@tol05"])].set_index("control")
        carm = "B" if bool(bal.at["B", "pass"]) else "B@tol05"       # fwd_pead_run._ctrl_arm 同一规则
        # 事件层
        E = B[B["arm"] == "E"].copy()
        E["r5"] = r_h(E["T_pos"], E["j"], 5)
        c = B[B["arm"] == carm].copy()
        c["r5"] = r_h(c["T_pos"], c["j"], 5)
        cmn = c.groupby(["T_pos", "slot"])["r5"].mean()
        E["e5_B"] = E["r5"].to_numpy() - cmn.reindex(pd.MultiIndex.from_arrays([E["T_pos"], E["slot"]])).to_numpy()
        E["h5_available"] = (E["T_pos"] + 5) <= (T - 1)
        assert_forward_index(E.set_index("t_signal"), f"T4 事件层(mvfix {tag})")
        Eh = E[E["h5_available"]]
        g = Eh[np.isfinite(Eh["e5_B"])].groupby("t_signal")["e5_B"]
        DT, nT = g.mean(), g.size()
        big = DT[nT >= PR.BIG]
        k = len(big)
        Dbar = float(big.mean()) if k else np.nan
        # 账户层(fwd_pead_run.stage_accounts 同式)
        acc = []
        for sc, cap in (("base_1m", 1e6), ("base_10m", 1e7)):
            for arm in ("E", carm):
                for rep in sorted(B.loc[B["arm"] == arm, "rep"].unique()):
                    for ph in range(5):
                        gg = B[(B["arm"] == arm) & (B["rep"] == rep) & (B["phase"] == ph)]
                        batches = PRA.account_batches(gg) if len(gg) else {}
                        ra = PE.run_account(pr, batches, (ph - offset) % 5, cap, raw_px, adj_px, star, lots=True,
                                            min_commission=5.0, stress_bps=0.0)
                        d = ra["daily"]
                        d = d[(d.index >= SIG_START) & (d.index <= PR.SIG_END)]
                        d.insert(0, "phase", ph); d.insert(0, "rep", int(rep))
                        d.insert(0, "arm", "B" if arm.startswith("B") else arm); d.insert(0, "scenario", sc)
                        acc.append(d.reset_index())
        D = pd.concat(acc, ignore_index=True)
        assert D["date"].min() >= SIG_START
        accv = {}
        for sc in ("base_1m", "base_10m"):
            x = D[D["scenario"] == sc]
            rE = x[x["arm"] == "E"].groupby("date")["ret"].mean().reindex(sig)
            rB = x[x["arm"] == "B"].groupby("date")["ret"].mean().reindex(sig)
            v, n = M5.forward_window_excess(pd.DataFrame({"strategy": rE, "benchmark": rB}), skip_head=0)
            accv[sc] = (v, float(np.prod(1 + rE) - 1), float(np.prod(1 + rB) - 1))
        if tag == "new":
            B.to_parquet(TMP / "fwd_mvfix_pead_splice_batches.parquet", index=False)
            E.to_parquet(TMP / "fwd_mvfix_pead_splice_event_slots.parquet", index=False)
            D.to_parquet(TMP / "fwd_mvfix_pead_splice_daily.parquet", index=False)
        res[tag] = dict(B=B, bal=bal, carm=carm, DT=DT, nT=nT, big=big, k=k, Dbar=Dbar, acc=accv)
        log(f"[pead] {tag}: 批次 {B['T_pos'].nunique()},E 槽位 {int((B['arm'] == 'E').sum())},对照臂 {carm},"
            f"D̄ {Dbar * 100:+.4f}%(k={k}),账户 1m {accv['base_1m'][0] * 100:+.3f}% 10m {accv['base_10m'][0] * 100:+.3f}% [{time.time() - t0:.0f}s]")
    o, nw = res["old"], res["new"]
    cols = [c_ for c_ in B0_file.columns]
    same_batches = o["B"][cols].reset_index(drop=True).equals(B0_file.reset_index(drop=True))
    rec("pead", "harness|old_batches_equal_fwd_pead_splice_batches", value=bool(same_batches),
        note="原面板 + 本段代码重建的批次表 vs cache/_tmp/fwd_pead_splice_batches.parquet(应为 True)")
    rec("pead", "harness|old_Dbar_vs_ledger_abs_diff", value=float(abs(o["Dbar"] - Ld["window_excess"])),
        note="原面板重算 D̄ vs fwd_results.csv P5(D1 口径)(应为 0)")
    for sc in ("base_1m", "base_10m"):
        rec("pead", f"harness|old_account_{sc}_vs_ledger_abs_diff", value=float(abs(o["acc"][sc][0] - La[sc])),
            note="原面板重算账户层 E−B vs fwd_results.csv T4_PEAD_account(应为 0)")
    b0 = bal0[(bal0["mode"] == "splice") & (bal0["control"] == "B")].iloc[0]
    rec("pead", "harness|old_smd_logmv_vs_ledger_abs_diff", value=float(abs(o["bal"].at["B", "smd_logmv"] - b0["smd_logmv"])),
        note="原面板重算平衡性 SMD log 市值 vs fwd_pead_balance.csv(应为 0)")
    # 新旧对照
    Eo, En = o["B"][o["B"]["arm"] == "E"], nw["B"][nw["B"]["arm"] == "E"]
    rec("pead", "E_slots_identical", value=bool(Eo.reset_index(drop=True).equals(En.reset_index(drop=True))),
        note=f"E 臂槽位(合格池、事件、选股都不读市值);E 槽位 {len(En)}")
    for arm in ("B", "B@tol05"):
        bo = o["B"][o["B"]["arm"] == arm].reset_index(drop=True)
        bn = nw["B"][nw["B"]["arm"] == arm].reset_index(drop=True)
        assert len(bo) == len(bn)
        ch = float((bo["j"].to_numpy() != bn["j"].to_numpy()).mean())
        rec("pead", f"control_slots_changed|{arm}", value=ch, note=f"{arm} 臂对照股换了的槽位比例(共 {len(bn)} 个槽位 = E 槽位 × 3 个复制)")
    rec("pead", "control_arm", old=o["carm"], new=nw["carm"], note="平衡性门槛选出的对照臂(B 达标则用 B)")
    for c_ in ("smd_logmv", "smd_logadv", "smd_vol", "ind_maxdiff_pp", "n_failed"):
        rec("pead", f"balance_B|{c_}", old=float(o["bal"].at["B", c_]), new=float(nw["bal"].at["B", c_]),
            note=f"E vs B,09 月窗口;达标 旧 {bool(o['bal'].at['B', 'pass'])} / 新 {bool(nw['bal'].at['B', 'pass'])}")
    for d_ in nw["DT"].index:
        rec("pead", f"D_T|{d_.date()}", old=float(o["DT"].get(d_, np.nan)), new=float(nw["DT"][d_]),
            note=f"{int(nw['nT'][d_])} 槽;" + ("大批次" if nw["nT"][d_] >= PR.BIG else "小批次(不进 D̄)"))
    w = dist("T4_E|base|B|n21|Dbar_big")
    kb = dist("T4_E|base|B|n21|k_big").reindex(w.index)
    k = nw["k"]
    lo, hi = k - 1, k + 1
    while ((kb >= lo) & (kb <= hi)).sum() < 50 and lo > 0:
        lo, hi = lo - 1, hi + 1
    wf = w[(kb >= lo) & (kb <= hi)]
    rec("pead", "P5'|D1|Dbar", old=o["Dbar"], new=nw["Dbar"], old_pct_ref=M5.percentile_in(o["Dbar"], wf),
        new_pct_ref=M5.percentile_in(nw["Dbar"], wf),
        note=f"大批次 D_T 均值(k 旧 {o['k']} / 新 {k});历史窗口 k∈[{lo},{hi}] 共 {len(wf)} 个;{PCT_NOTE};补充读数,本来就不进判定")
    for sc in ("base_1m", "base_10m"):
        key = f"T4_E|{sc}|B|n21|account"
        vo, vn = o["acc"][sc][0], nw["acc"][sc][0]
        rec("pead", f"account|{sc}|E-B", old=vo, new=vn, old_pct_ref=M5.percentile_in(vo, dist(key)),
            new_pct_ref=M5.percentile_in(vn, dist(key)),
            note=f"账户层 E−B 复利差(不跳过建仓日);E 累计 旧 {o['acc'][sc][1] * 100:+.3f}% / 新 {nw['acc'][sc][1] * 100:+.3f}%,"
                 f"B 累计 旧 {o['acc'][sc][2] * 100:+.3f}% / 新 {nw['acc'][sc][2] * 100:+.3f}%;{PCT_NOTE}")
    flush()


# ============================================================================================ 审查 P2 补充:kmeans / stmt_em
def stage_kmeans(seeds=(-1, 0, 1, 2, 3, 4, 5, 6, 7), sd=1e-6):
    """统计行业标签微扰(fwd_review_checks.stage_kmeans 同一噪声、同一种子、同一 _t3_t5)在新口径下重跑(审查 P2:
    报告里作参照的 P6 微扰区间 1.2814 ~ 1.2831 是旧口径)。标签只用收盘价、打分不读市值,所以只有 T3 / T5 随新风险模型变;
    T1 / T2 也重跑一遍,与 cache/_tmp/fwd_review_kmeans.csv 的逐种子旧值对照(应逐位相同)。seed=-1 = 不加噪声,必须复现新口径主表的 P4 / P6。"""
    import fwd_review_checks as RC
    import fwd_run_test as RT
    from afactor import strategy as ST, fwd_adapters as FA_, fwd_weights as FW
    pf, p = panels()
    cm = cost_model(pf)
    rm = load_rm(RISK_DIR)
    B = pd.read_parquet(BENCH_FILE)
    W = FW.csi1000_close_weights(p)
    Wx = np.load(TMP / "fwd_mvfix_csi1000_Wx.npy").astype(np.float64)
    U = ST.universe(p, "V1000")
    m = p.mask_from(SIG_START)
    close = pf.raw("close_px")
    L0 = pf._derived["statind_L"]
    K0 = pd.read_csv(TMP / "fwd_review_kmeans.csv").set_index("seed")
    q5 = QREF[(QREF["target"] == "T5_V1") & (QREF["n"] == 17)].iloc[0]
    inb = lambda v: bool(q5["chi2_band_lo"] <= v <= q5["chi2_band_hi"])
    out = []
    for sd_seed in seeds:
        t0 = time.time()
        if sd_seed < 0:
            Ln = L0
        else:
            rng = np.random.default_rng(20261010 + sd_seed)
            noise = np.exp(rng.normal(0.0, sd, close.shape)).astype(np.float32)
            Ln = FA_.statind_labels_for_quarters(close * noise, ["2025Q4", "2026Q1", "2026Q2", "2026Q3"])
            r25 = Ln.index.year == 2025
            Ln.loc[r25] = L0.loc[r25].to_numpy()
            Ln = Ln.astype(np.int16)

        class _P(MVPanel):
            def __init__(self):
                super().__init__(returns_view=False)
                self._derived["statind_L"] = Ln
        with forward_context("前向检验市值修复敏感性(fwd_mvfix):统计行业标签微扰在新口径下重跑(只算评估窗读数,不进判定)"):
            s = FC.comp_score(_P(), cache_file=None)
        row = dict(seed=sd_seed)
        t12diff = 0.0
        for plan, kw in RT.PLANS.items():
            for cost, cmx in (("old", None), ("Y1", cm)):
                rr = ST.run_strategy(s, p, "V1000", "open10", date_mask=m, cost_model=cmx, **kw)["rets"]
                v = RC.wex(rr, B["pool5"])
                row[f"{plan}|{cost}"] = v
                t12diff = max(t12diff, abs(v - float(K0.at[sd_seed, f"{plan}|{cost}"])))
        r3, bd = RC._t3_t5(p, pf, s, rm, cm, W, Wx, U, B)
        v3 = RC.wex(r3, B["csi1000_exec"])
        row.update(T3_V1=v3, T5_V1_bd=bd, t12_maxdiff_vs_old=t12diff)
        out.append(row)
        o3, obd = float(K0.at[sd_seed, "T3_V1"]), float(K0.at[sd_seed, "T5_V1_bd"])
        rec("kmeans", f"seed{sd_seed}|P4", old=o3, new=v3, old_pct_ref=M5.percentile_in(o3, dist("T3_V1|Y1|proxy_exec|n17")),
            new_pct_ref=M5.percentile_in(v3, dist("T3_V1|Y1|proxy_exec|n17")),
            note=("不加噪声(须复现主表新口径 P4)" if sd_seed < 0 else f"种子 20261010+{sd_seed}")
                 + f";T1 / T2 四个读数与旧口径同种子最大差 {t12diff:.1e};{PCT_NOTE}")
        rec("kmeans", f"seed{sd_seed}|P6_b_d", old=obd, new=bd, old_pct_ref=M5.percentile_in(obd, dist("T5_V1|Y1|proxy_exec|n17|b_d")),
            new_pct_ref=M5.percentile_in(bd, dist("T5_V1|Y1|proxy_exec|n17|b_d")),
            note=f"χ² 区间 [{q5['chi2_band_lo']:.4f}, {q5['chi2_band_hi']:.4f}] 内:旧 {inb(obd)} / 新 {inb(bd)};{PCT_NOTE}")
        log(f"[kmeans] seed {sd_seed}: P4 {o3 * 100:+.3f}% -> {v3 * 100:+.3f}%  b_d {obd:.4f} -> {bd:.4f}  T1/T2 差 {t12diff:.1e} [{time.time() - t0:.0f}s]")
    Dn = pd.DataFrame(out)
    Dn.to_csv(TMP / "fwd_mvfix_kmeans.csv", index=False)
    pert = Dn[Dn["seed"] >= 0]
    K0p = K0.loc[pert["seed"]]
    rec("kmeans", "P4|range", old=f"{K0p['T3_V1'].min() * 100:+.3f}% ~ {K0p['T3_V1'].max() * 100:+.3f}%(中位 {K0p['T3_V1'].median() * 100:+.3f}%)",
        new=f"{pert['T3_V1'].min() * 100:+.3f}% ~ {pert['T3_V1'].max() * 100:+.3f}%(中位 {pert['T3_V1'].median() * 100:+.3f}%)",
        note=f"8 次微扰;旧 = fwd_review_fixes.csv kmeans(cache/_tmp/fwd_review_kmeans.csv)")
    rec("kmeans", "P6_b_d|range", old=f"{K0p['T5_V1_bd'].min():.4f} ~ {K0p['T5_V1_bd'].max():.4f}(中位 {K0p['T5_V1_bd'].median():.4f})",
        new=f"{pert['T5_V1_bd'].min():.4f} ~ {pert['T5_V1_bd'].max():.4f}(中位 {pert['T5_V1_bd'].median():.4f})",
        note="8 次微扰")
    rec("kmeans", "P6_in_chi2_band_count", old=f"{int(sum(inb(v) for v in K0p['T5_V1_bd']))}/{len(K0p)}",
        new=f"{int(sum(inb(v) for v in pert['T5_V1_bd']))}/{len(pert)}",
        note=f"微扰下 b_d 落在 χ² 区间 [{q5['chi2_band_lo']:.4f}, {q5['chi2_band_hi']:.4f}] 内的次数")
    rec("kmeans", "T1T2_maxdiff_vs_old_all_seeds", value=float(Dn["t12_maxdiff_vs_old"].max()),
        note="T1 / T2 四个读数新口径 vs 旧口径同种子的最大差(打分与 T1 / T2 不读市值,应为 0)")
    flush()


def stage_stmt_em():
    """补利润表变体(fwd_review_checks.stage_stmt_em,§11)在新口径下重跑(审查 P2)。同一个 _load 补丁、同一个 _t3_t5;
    风险模型在补丁 + 新口径面板上重建,缓存写 cache/_tmp/fwd_mvfix_stmt_em_risk/(不碰原 fwd_review_stmt_em_risk)。
    管线自检:原面板 + 原缓存 fwd_review_stmt_em_risk(只读)重跑 T3 / T5,应复现 fwd_review_fixes.csv 的 +2.665% / 1.2597。"""
    import fwd_review_checks as RC
    import fwd_run_test as RT
    from afactor import strategy as ST, fundamental as FU, fwd_adapters as FA_, fwd_weights as FW
    from afactor.barra import RiskModel
    saved = FU._load
    n_add = {}

    def _load_em(name, items, cutoff=None):                 # fwd_review_checks.stage_stmt_em 同式
        d = saved(name, items, cutoff)
        if name != FU.REPORT_TABLES[0]:
            return d
        add = RC._em_h1_income(list(d.columns), items)
        have = set(zip(d["stock_code"], d["end_date"]))
        add = add[[(a_, b_) not in have for a_, b_ in zip(add["stock_code"], add["end_date"])]]
        cut = FU.data_cutoff() if cutoff is None else pd.Timestamp(cutoff)
        add = add[add["avail"] <= cut].copy()
        ct = d.sort_values("end_date").drop_duplicates("stock_code", keep="last").set_index("stock_code")["comp_type"]
        add["comp_type"] = add["stock_code"].map(ct)
        n_add["rows"] = len(add)
        return pd.concat([d, add[d.columns]], ignore_index=True).sort_values(["stock_code", "end_date", "avail"])
    FU._REP_CACHE.clear()
    FU._load = _load_em
    try:
        pf0, pf = ForwardPanel(), MVPanel()                 # 与原 stage_stmt_em 一样在补丁内新建面板,不复用缓存的面板
        with forward_context("前向检验市值修复敏感性(fwd_mvfix):补利润表变体在新口径下重跑(偏离原规则,只作描述)"):
            s0 = FC.comp_score(pf0, cache_file=None)
            s = FC.comp_score(pf, cache_file=None)
            with FA_.lifted_guards():
                rm = RiskModel(cache_dir=TMP / "fwd_mvfix_stmt_em_risk").build(pf, "open10", force=True, verbose=False)
    finally:
        FU._load = saved
        FU._REP_CACHE.clear()
    rec("stmt_em", "em_h1_rows_spliced", value=int(n_add.get("rows", 0)), note="补入的(股票, 2026-06-30)利润表行;旧 4,382")
    a_, b_ = s0.to_numpy(np.float64), s.to_numpy(np.float64)
    rec("stmt_em", "score_identical_old_vs_new_panel", value=eq_nan(a_, b_), note="补丁下原面板 / 新口径面板的 COMP_v5 打分逐位比较(打分不读市值)")
    B0, B1 = pd.read_parquet(TMP / "fwd_bench.parquet"), pd.read_parquet(BENCH_FILE)
    p = MVPanel(returns_view=True)
    m = p.mask_from(SIG_START)
    cm = cost_model(MVPanel())
    RV = pd.read_csv(LEDGER / "fwd_review_fixes.csv")
    rv = lambda it: str(RV[(RV["section"] == "stmt_em") & (RV["item"] == it)]["value"].iloc[0])
    for plan, kw in RT.PLANS.items():
        for cost, cmx in (("old", None), ("Y1", cm)):
            r = ST.run_strategy(s, p, "V1000", "open10", date_mask=m, cost_model=cmx, **kw)["rets"]
            v = RC.wex(r, B1["pool5"])
            rec("stmt_em", f"{plan}|{cost}", new=v, new_pct_ref=M5.percentile_in(v, dist(f"{plan}|{cost}|pool5|n17")),
                note=f"旧口径(fwd_review_fixes.csv stmt_em)= {rv(f'{plan}|{cost}')};{PCT_NOTE}")
    W = FW.csi1000_close_weights(p)
    U = ST.universe(p, "V1000")
    # 管线自检:原面板 + 原 stmt_em 风险模型(只读)
    p0 = ForwardPanel(returns_view=True)
    rm0 = load_rm(TMP / "fwd_review_stmt_em_risk")
    from afactor.impact import CostModel
    with FA_.lifted_guards(raw_close=pf0.raw("raw_close")):
        cm0 = CostModel(pf0, impact_y=1.0)
    Wx0 = np.load(TMP / "fwd_csi1000_Wx.npy").astype(np.float64)
    r30, bd0 = RC._t3_t5(p0, pf0, s0, rm0, cm0, FW.csi1000_close_weights(p0), Wx0, ST.universe(p0, "V1000"), B0)
    v30 = RC.wex(r30, B0["csi1000_exec"])
    del rm0
    Wx = np.load(TMP / "fwd_mvfix_csi1000_Wx.npy").astype(np.float64)
    r3, bd = RC._t3_t5(p, pf, s, rm, cm, W, Wx, U, B1)
    v3 = RC.wex(r3, B1["csi1000_exec"])
    q5 = QREF[(QREF["target"] == "T5_V1") & (QREF["n"] == 17)].iloc[0]
    inb = lambda v: bool(q5["chi2_band_lo"] <= v <= q5["chi2_band_hi"])
    rec("stmt_em", "T3_V1", old=v30, new=v3, old_pct_ref=M5.percentile_in(v30, dist("T3_V1|Y1|proxy_exec|n17")),
        new_pct_ref=M5.percentile_in(v3, dist("T3_V1|Y1|proxy_exec|n17")),
        note=f"旧 = 原面板 + 原缓存重跑(fwd_review_fixes.csv 记 {rv('T3_V1')});{PCT_NOTE}")
    rec("stmt_em", "T5_V1_bd", old=bd0, new=bd, old_pct_ref=M5.percentile_in(bd0, dist("T5_V1|Y1|proxy_exec|n17|b_d")),
        new_pct_ref=M5.percentile_in(bd, dist("T5_V1|Y1|proxy_exec|n17|b_d")),
        note=f"旧 = 原面板 + 原缓存重跑(fwd_review_fixes.csv 记 {rv('T5_V1_bd')});χ² 区间内:旧 {inb(bd0)} / 新 {inb(bd)};{PCT_NOTE}")
    flush()


# ============================================================================================ 审查 P2 补充:§6 / §9 / §11 其余依赖市值的描述量
def stage_t3extra():
    """§6 T3 表里其余描述量(比值几何超额、日超额均值 / NW t、逐相位、求解记录、建仓 / 评估窗成本拆分)的 旧 → 新。
    旧 = 原 fwd_T3_V1 落盘(fwd_run_test.load_t3),新 = fwd_mvfix_T3_V1;函数用 fwd_report.window_stats / cost_block 原样。只读落盘结果。"""
    import fwd_run_test as RT
    import fwd_report as FR
    B0, B1 = pd.read_parquet(TMP / "fwd_bench.parquet"), pd.read_parquet(BENCH_FILE)
    sig = B0.index
    V0, V1 = RT.load_t3(), load_t3(f"{PRE}_T3_V1")
    st0, st1 = FR.window_stats(V0["rets"].reindex(sig), B0["csi1000_exec"]), FR.window_stats(V1["rets"].reindex(sig), B1["csi1000_exec"])
    for k in ("ratio_geo_excess", "strat_cum", "bench_cum", "daily_mean", "nw_t_L5", "te_realized"):
        rec("t3extra", f"P4|{k}", old=st0[k], new=st1[k], note="fwd_report.window_stats 原式;对中证 1000 实际成分(成交时点)")
    c0, c1 = FR.cost_block(V0["daily"]), FR.cost_block(V1["daily"])
    for k in ("cost_total_build", "cost_total_eval", "cost_spread_eval", "cost_impact_eval", "cost_commission_eval", "cost_stamp_eval",
              "turnover_mean_eval", "n_hold_mean_eval", "exposure_max"):
        rec("t3extra", f"cost|{k}", old=c0[k], new=c1[k], note="fwd_report.cost_block 原式(建仓 = 前 4 个信号日)")
    key = "T3_V1|Y1|proxy_exec|n17"
    ph = {}
    for tag, V, B in (("old", V0, B0), ("new", V1, B1)):
        v = {int(p_): FR.window_stats(g["ret"].reindex(sig), B["csi1000_exec"])["window_excess"] for p_, g in V["by_phase"].groupby("phase")}
        ph[tag] = v
    for p_ in sorted(ph["new"]):
        rec("t3extra", f"phase{p_}", old=ph["old"][p_], new=ph["new"][p_], old_pct_ref=M5.percentile_in(ph["old"][p_], dist(key)),
            new_pct_ref=M5.percentile_in(ph["new"][p_], dist(key)), note=f"逐相位窗口复利超额对中证 1000 实际成分;{PCT_NOTE}")
    s0, s1 = V0["solver"], V1["solver"]
    for c in ("te_ex_ante", "alpha_ex_ante", "n_hold", "bench_w_in"):
        rec("t3extra", f"solver|{c}|min_mean_max", old=f"{s0[c].min():.6g} / {s0[c].mean():.6g} / {s0[c].max():.6g}",
            new=f"{s1[c].min():.6g} / {s1[c].mean():.6g} / {s1[c].max():.6g}", note="21 次求解(5 相位)的 最小 / 均值 / 最大")
    flush()


def stage_cond():
    """§9 条件位置(fwd_review_checks.stage_cond 同式)只对 P4 重算:历史回归与相似状态子集都不变(研究期旧口径),
    只换前向 P4 值。旧值须复现 fwd_review_fixes.csv cond(残差百分位 56.1、子集内 44.4)。"""
    import statsmodels.api as sm
    import fwd_review_checks as RC
    d = pd.read_parquet(LEDGER / "freeze_ref_daily.parquet")
    Bf = pd.read_parquet(TMP / "fwd_bench.parquet").iloc[SKIP:]
    assert pd.read_parquet(BENCH_FILE).iloc[SKIP:].equals(Bf)
    f_pool = float(np.prod(1 + Bf["pool5"]) - 1)
    f_pmc = float(np.prod(1 + Bf["pool5"]) - np.prod(1 + Bf["csi1000_exec"]))
    L = pd.read_csv(OUT)
    rd = lambda it: L[(L["section"] == "reading") & (L["item"] == it)].iloc[0]
    p4 = rd("P4|T3_V1|Y1|csi1000_actual_exec")
    x = DIST[DIST["key"] == "T3_V1|Y1|proxy_exec|n17"].set_index("window_end")["value"]
    pool_w = RC._win(d["T3_V1|A|bench_pool5"]); pmc = RC._win(d["T3_V1|A|bench_pool5"], d["T3_V1|A|bench_proxy_exec"])
    df = pd.DataFrame({"x": x, "pool": pool_w.reindex(x.index), "pmc": pmc.reindex(x.index)}).dropna().astype(float)
    mdl = sm.OLS(df["x"], sm.add_constant(df[["pool", "pmc"]])).fit()
    pred = float(mdl.params @ np.array([1.0, f_pool, f_pmc]))
    sub = df[(df["pool"] <= -0.02) & (df["pmc"] >= 0.02)]
    rp = lambda fv: 100.0 * float((mdl.resid < fv - pred).mean())
    rec("cond", "P4|resid_pct", old=rp(float(p4["old"])), new=rp(float(p4["new"])),
        note=f"预测 {pred * 100:+.2f}%;回归与子集用研究期旧口径历史窗口(fwd_review_fixes.csv cond 同式),只换前向 P4 值")
    rec("cond", "P4|pct_in_similar_subset", old=M5.percentile_in(float(p4["old"]), sub["x"]), new=M5.percentile_in(float(p4["new"]), sub["x"]),
        note=f"相似状态子集 n={len(sub)}(池 ≤ −2% 且 池−中证 ≥ +2%)")
    flush()


def stage_may1314():
    """05-13 / 05-14 收盘补齐敏感性(fwd_review_checks.stage_may1314,§11)在新口径下重跑。补丁与原式相同,底面板换成新口径;
    风险模型重建,缓存写 cache/_tmp/fwd_mvfix_may1314_risk/。管线自检:原底面板 + 原缓存 fwd_review_may1314_risk(只读)重跑 T3 / T5。"""
    import fwd_review_checks as RC
    import fwd_run_test as RT
    import fwd_build_panel as FB
    from afactor import strategy as ST, fwd_adapters as FA_, fwd_weights as FW
    from afactor.barra import RiskModel
    from afactor.impact import CostModel
    base0 = ForwardPanel()
    days = [pd.Timestamp("2026-05-13"), pd.Timestamp("2026-05-14")]
    c = base0.raw("close_px")
    op = base0.raw("open_auction_px")
    tw = base0.raw("twap_open10")
    dbc = FB._aux("db_raw_close").reindex(index=days, columns=base0.codes).astype(np.float64)
    tgt = FB.rebase_target(base0.dates, base0.codes).reindex(index=days)
    fill = (c.loc[days].isna() & tw.loc[days].notna()).to_numpy() & np.isfinite((dbc * tgt).to_numpy())
    c2 = c.copy()
    c2.loc[days] = np.where(fill, (dbc * tgt).to_numpy(), c.loc[days].to_numpy()).astype(np.float32)
    ri2 = base0.raw("ret_intraday").copy()
    ri2.loc[days] = np.where(fill, (c2.loc[days] / op.loc[days] - 1.0).to_numpy(), ri2.loc[days].to_numpy()).astype(np.float32)
    Lb = FA_.statind_labels_for_quarters(c2, ["2025Q4", "2026Q1", "2026Q2", "2026Q3"])
    Lb.loc[Lb.index.year == 2025] = base0._derived["statind_L"].loc[Lb.index.year == 2025].to_numpy()
    Lb = Lb.astype(np.int16)
    raw_c = base0.raw("raw_close").copy()
    raw_c.loc[days] = np.where(fill, dbc.to_numpy(), raw_c.loc[days].to_numpy()).astype(np.float32)
    rec("may1314", "cells_filled", value=int(fill.sum()), note="原 fwd_review_fixes.csv may1314 同式")

    def mk(base, rv):
        class _P(base):
            def __init__(self):
                super().__init__(returns_view=rv)
                self._derived["statind_L"] = Lb
                self._cache.update({"close_px": c2, "ret_intraday": ri2, "raw_close": raw_c})
        return _P()
    pf0, p0 = mk(ForwardPanel, False), mk(ForwardPanel, True)
    pf, p = mk(MVPanel, False), mk(MVPanel, True)
    with forward_context("前向检验市值修复敏感性(fwd_mvfix):05-13/14 收盘补齐变体在新口径下重跑(只算评估窗读数,不进判定)"):
        s0 = FC.comp_score(pf0, cache_file=None)
        s = FC.comp_score(pf, cache_file=None)
        with FA_.lifted_guards():
            rm = RiskModel(cache_dir=TMP / "fwd_mvfix_may1314_risk").build(pf, "open10", force=True, verbose=False)
    rec("may1314", "score_identical_old_vs_new_panel", value=eq_nan(s0.to_numpy(np.float64), s.to_numpy(np.float64)),
        note="补丁下原底面板 / 新口径底面板的打分逐位比较")
    with FA_.lifted_guards(raw_close=pf0.raw("raw_close")):
        cm0 = CostModel(pf0, impact_y=1.0)
    with FA_.lifted_guards(raw_close=pf.raw("raw_close")):
        cm = CostModel(pf, impact_y=1.0)
    B0, B1 = pd.read_parquet(TMP / "fwd_bench.parquet"), pd.read_parquet(BENCH_FILE)
    RV = pd.read_csv(LEDGER / "fwd_review_fixes.csv")
    rv = lambda it: str(RV[(RV["section"] == "may1314") & (RV["item"] == it)]["value"].iloc[0])
    m = p.mask_from(SIG_START)
    for plan, kw in RT.PLANS.items():
        for cost, cmx in (("old", None), ("Y1", cm)):
            r = ST.run_strategy(s, p, "V1000", "open10", date_mask=m, cost_model=cmx, **kw)["rets"]
            v = RC.wex(r, B1["pool5"])
            rec("may1314", f"{plan}|{cost}", new=v, new_pct_ref=M5.percentile_in(v, dist(f"{plan}|{cost}|pool5|n17")),
                note=f"旧口径(fwd_review_fixes.csv may1314)= {rv(f'{plan}|{cost}')};{PCT_NOTE}")
    rm0 = load_rm(TMP / "fwd_review_may1314_risk")
    r30, bd0 = RC._t3_t5(p0, pf0, s0, rm0, cm0, FW.csi1000_close_weights(p0), np.load(TMP / "fwd_csi1000_Wx.npy").astype(np.float64),
                         ST.universe(p0, "V1000"), B0)
    v30 = RC.wex(r30, B0["csi1000_exec"])
    del rm0
    r3, bd = RC._t3_t5(p, pf, s, rm, cm, FW.csi1000_close_weights(p), np.load(TMP / "fwd_mvfix_csi1000_Wx.npy").astype(np.float64),
                       ST.universe(p, "V1000"), B1)
    v3 = RC.wex(r3, B1["csi1000_exec"])
    q5 = QREF[(QREF["target"] == "T5_V1") & (QREF["n"] == 17)].iloc[0]
    inb = lambda v: bool(q5["chi2_band_lo"] <= v <= q5["chi2_band_hi"])
    rec("may1314", "T3_V1", old=v30, new=v3, old_pct_ref=M5.percentile_in(v30, dist("T3_V1|Y1|proxy_exec|n17")),
        new_pct_ref=M5.percentile_in(v3, dist("T3_V1|Y1|proxy_exec|n17")),
        note=f"旧 = 原底面板 + 原缓存重跑(fwd_review_fixes.csv 记 {rv('T3_V1')});{PCT_NOTE}")
    rec("may1314", "T5_V1_bd", old=bd0, new=bd, old_pct_ref=M5.percentile_in(bd0, dist("T5_V1|Y1|proxy_exec|n17|b_d")),
        new_pct_ref=M5.percentile_in(bd, dist("T5_V1|Y1|proxy_exec|n17|b_d")),
        note=f"旧 = 原底面板 + 原缓存重跑(fwd_review_fixes.csv 记 {rv('T5_V1_bd')});χ² 区间内:旧 {inb(bd0)} / 新 {inb(bd)};{PCT_NOTE}")
    flush()


def stage_nofu():
    """去掉 3 个基本面成员的上包络式参考(fwd_review_checks.stage_stmt 的 no_fu|T3_V1,§11)在新口径风险模型下重跑。
    打分 s2 不读市值(MB._comp_score 同式);旧 = 原面板 + 原风险模型 fwd_risk(只读)重跑,须复现 +3.429% / 1.3752。"""
    import fwd_review_checks as RC
    import fwd_run_test as RT
    import m5_bridge as MB
    from afactor import strategy as ST, fwd_weights as FW
    from afactor.impact import CostModel
    from afactor import fwd_adapters as FA_
    members, signs, _ = MB.load_members()
    fu = [m_ for m_ in members if m_.startswith("fx_fu_")]
    keep = [m_ for m_ in members if m_ not in fu]
    pf0, p0 = ForwardPanel(), ForwardPanel(returns_view=True)
    pf, p = panels()
    with forward_context("前向检验市值修复敏感性(fwd_mvfix):去掉基本面成员的打分(上包络参考,不进判定)"):
        s20 = MB._comp_score(pf0, keep, {k: signs[k] for k in keep}).reindex(index=p0.dates, columns=p0.codes).astype(np.float32)
        s2 = MB._comp_score(pf, keep, {k: signs[k] for k in keep}).reindex(index=p.dates, columns=p.codes).astype(np.float32)
    rec("nofu", "score_identical_old_vs_new_panel", value=eq_nan(s20.to_numpy(np.float64), s2.to_numpy(np.float64)),
        note=f"去掉 {fu} 后的打分,原面板 / 新口径面板逐位比较")
    with FA_.lifted_guards(raw_close=pf0.raw("raw_close")):
        cm0 = CostModel(pf0, impact_y=1.0)
    cm = cost_model(pf)
    B0, B1 = pd.read_parquet(TMP / "fwd_bench.parquet"), pd.read_parquet(BENCH_FILE)
    RV = pd.read_csv(LEDGER / "fwd_review_fixes.csv")
    r30, bd0 = RC._t3_t5(p0, pf0, s20, RT.load_rm(), cm0, FW.csi1000_close_weights(p0),
                         np.load(TMP / "fwd_csi1000_Wx.npy").astype(np.float64), ST.universe(p0, "V1000"), B0)
    v30 = RC.wex(r30, B0["csi1000_exec"])
    r3, bd = RC._t3_t5(p, pf, s2, load_rm(RISK_DIR), cm, FW.csi1000_close_weights(p),
                       np.load(TMP / "fwd_mvfix_csi1000_Wx.npy").astype(np.float64), ST.universe(p, "V1000"), B1)
    v3 = RC.wex(r3, B1["csi1000_exec"])
    old_s = str(RV[(RV["section"] == "stmt") & (RV["item"] == "no_fu|T3_V1")]["value"].iloc[0])
    rec("nofu", "T3_V1", old=v30, new=v3, old_pct_ref=M5.percentile_in(v30, dist("T3_V1|Y1|proxy_exec|n17")),
        new_pct_ref=M5.percentile_in(v3, dist("T3_V1|Y1|proxy_exec|n17")),
        note=f"旧 = 原面板 + 原风险模型重跑(fwd_review_fixes.csv stmt 记 {old_s});{PCT_NOTE}")
    rec("nofu", "T5_V1_bd", old=bd0, new=bd, note="同上")
    flush()


if __name__ == "__main__":
    for st_ in sys.argv[1:]:
        t0 = time.time()
        log(f"===== {st_} =====")
        globals()[f"stage_{st_}"]()
        log(f"===== {st_} 完成 {time.time() - t0:.0f}s =====")
