"""压力测试的真实市值修复重跑(前缀 stress,2026-10-10)。

## 为什么要重跑
`research/TOTAL_MV_FIX.md` 修复了 `newdata.total_mv`(旧口径把限售解禁当成股本增加)。`STRESS_TEST.md` 经
`FA._real(p, "mv")` 用到市值的环节:
  1. Barra 十风格暴露(`cache/_tmp/stress_X_*.parquet`,由 run_barra_attribution.build_exposures 生成):
     Size / NonlinSize 直接是 log 市值,其余风格按市值加权去中心化 -> 归因、事前情景、事前风险、截尾诊断、N1/N3 中性化;
  2. N1r(池内 log 市值秩中性化)与 NS(池内市值五分位分层选股)两个反事实的打分;
  3. 分析阶段的市值分档:全 A 市值五分位(微盘权重、特质匹配拆分)、池内市值秩、加权中位总市值;
  4. 行业纯因子"市值加权和为零"的映射(capsum);
  5. Barra 纯因子收益 `research/ledger/barra_factor_returns.csv`(已由 barra 重跑代理换成新口径)。
策略打分(COMP_v5 冻结打分、B_gbt 冻结 walk-forward 打分)、决策池 V1000、池等权基准、中证1000、成本模型都不用
`ND.total_mv`,所以 5 个基础对象的收益序列不变;变的是 4 个反事实变体的收益和全部 Barra / 市值分档的拆解。
(B_gbt 打分含 `nd_mv` 特征,是旧市值训练的冻结打分,按纪律不重训,见 TOTAL_MV_FIX.md §6。)

## 与 run_stress_test.py / stress_analyze.py 的关系
两份脚本在前向冻结清单里,**不改一个字节**。本脚本只做:备份、强制重建暴露缓存(原脚本见缓存就复用)、
旧口径影子重跑(把同一套函数指到影子目录)、新旧对照。新口径的正式重跑直接调用原脚本四个阶段。

## 阶段
  backup          备份 research/ledger/stress_* -> research/ledger/_pre_mvfix/;reports/stress_2024q1.png ->
                  reports/_pre_mvfix/;cache/_tmp/stress_*.parquet 与 research/STRESS_TEST.md -> cache/_tmp/pre_mvfix_backup/。
                  逐个 sha256 核对,清单 research/ledger/stress_mvfix_backup_manifest.csv。已有同名备份且内容不同则拒绝。
  exposures       用当前代码(新市值)经原函数 run_stress_test.load_exposures 在暂存目录生成暴露,核对后 os.replace
                  覆盖 cache/_tmp/stress_X_*.parquet;新旧暴露差异写 research/ledger/stress_mvfix_exposure_delta.csv。
  (然后)          python3 scripts/run_stress_test.py engine / dims / analyze / figure   —— 原脚本,不改
  shadow_legacy   当前代码 + newdata.TOTAL_SHARE_METHOD="legacy"(诊断开关,只差 total_mv),纯因子收益取 barra 重跑代理的
                  cache/_tmp/barra2_legacy_{open10,tail1450}_F.parquet,全流程跑在 cache/_tmp/stress_mvfix_shadow/legacy/
                  {ledger,tmp,reports}。用来把"原报告 -> 新口径"拆成"原报告 -> 旧口径重跑"(代码 / 纯因子收益文件漂移)
                  与"旧口径重跑 -> 新口径"(纯市值修复)。--steps engine,dims,analyze,figure 可分步。
  compare         三列对照 -> research/ledger/stress_mvfix_files.csv(逐文件变化格数)、stress_mvfix_diff.csv(逐格长表)、
                  stress_mvfix_headlines.csv(报告引用的数字)。

纪律:只读研究期面板(<= 2025-12-31),不触碰 cache/sealed;不改任何预注册参数(窗口、对象、方法照 stress_preregistration.json)。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, TMP, REPORTS, ROOT          # noqa: E402

BK_LEDGER = LEDGER / "_pre_mvfix"
BK_REPORTS = REPORTS / "_pre_mvfix"
BK_CACHE = TMP / "pre_mvfix_backup"
SHADOW = TMP / "stress_mvfix_shadow" / "legacy"
STAGE_X = TMP / "stress_mvfix_stage_new"
STRESS_MD = ROOT / "research" / "STRESS_TEST.md"
FIG = REPORTS / "stress_2024q1.png"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ledger_files() -> list[Path]:
    return sorted([p for p in LEDGER.glob("stress_*") if p.is_file() and not p.name.startswith("stress_mvfix_")])


def cache_files() -> list[Path]:
    return sorted([p for p in TMP.glob("stress_*.parquet") if p.is_file()])


# ====================================================================== backup
def stage_backup():
    groups = [("ledger", f, BK_LEDGER) for f in ledger_files()] + \
             [("reports", FIG, BK_REPORTS)] + \
             [("cache", f, BK_CACHE) for f in cache_files()] + \
             [("md", STRESS_MD, BK_CACHE)]
    rows = []
    for g, src, dstdir in groups:
        dstdir.mkdir(parents=True, exist_ok=True)
        dst = dstdir / src.name
        h = sha(src)
        if dst.exists():
            hb = sha(dst)
            assert hb == h, f"备份已存在且内容不同,拒绝覆盖: {dst}"
            action = "已存在(相同)"
        else:
            tmp_ = dstdir / (src.name + ".copying")
            shutil.copy2(src, tmp_)
            assert sha(tmp_) == h, f"复制校验失败 {src}"
            os.replace(tmp_, dst)
            action = "新备份"
        rows.append(dict(group=g, path=str(src.relative_to(ROOT)) if str(src).startswith(str(ROOT)) else str(src),
                         backup_path=str(dst), sha256=h, bytes=src.stat().st_size, action=action,
                         verify=sha(dst) == h))
    df = pd.DataFrame(rows)
    assert df.verify.all()
    df.to_csv(LEDGER / "stress_mvfix_backup_manifest.csv", index=False)
    log("备份完成", df.groupby(["group", "action"]).size().to_dict())


# ====================================================================== exposures(新口径,强制重建)
def stage_exposures():
    import run_stress_test as RS
    from afactor.panel import Panel
    from afactor import newdata as ND
    assert ND.TOTAL_SHARE_METHOD == "songzhuan", ND.TOTAL_SHARE_METHOD
    man = pd.read_csv(LEDGER / "stress_mvfix_backup_manifest.csv")
    assert (man.group == "cache").sum() >= 10, "先跑 backup"
    STAGE_X.mkdir(parents=True, exist_ok=True)
    for f in STAGE_X.glob("stress_X_*.parquet"):
        raise SystemExit(f"暂存目录已有 {f},请先确认后手动处理")
    RS.TMP = STAGE_X                         # 原函数 load_exposures 写到暂存目录(它见缓存就复用,这里目录为空 -> 重建)
    p = Panel()
    t0 = time.time()
    X = RS.load_exposures(p)
    log(f"新口径暴露构建完成 {time.time() - t0:.0f}s")
    u = p["universe"].reindex(index=p.dates, columns=p.codes).fillna(False)
    rows = []
    for k in RS.STYLE_KEYS:
        short = RS.SHORT[k]
        new_f = STAGE_X / f"stress_X_{short}.parquet"
        old_f = BK_CACHE / f"stress_X_{short}.parquet"
        live_f = TMP / f"stress_X_{short}.parquet"
        assert sha(old_f) == sha(live_f), f"现行缓存与备份不一致 {live_f}"
        xo = pd.read_parquet(old_f).reindex(index=p.dates, columns=p.codes)
        xn = X[k].reindex(index=p.dates, columns=p.codes)
        m = u & xo.notna() & xn.notna()
        d = (xn - xo).where(m)
        ro = xo.where(m).rank(axis=1)
        rn = xn.where(m).rank(axis=1)
        rc = ro.corrwith(rn, axis=1).dropna()
        rows.append(dict(factor=k, n_cells_pool=int(m.to_numpy().sum()),
                         share_abs_diff_gt_0p5=float((d.abs() > 0.5).to_numpy().sum() / m.to_numpy().sum()),
                         max_abs_diff=float(np.nanmax(np.abs(d.to_numpy()))),
                         rank_corr_median=float(rc.median()), rank_corr_min=float(rc.min()),
                         rank_corr_2023_12_29=float(rc.get(pd.Timestamp("2023-12-29"), np.nan)),
                         sha_old=sha(old_f), sha_new=sha(new_f)))
        log(f"{short:14s} 秩相关中位 {rc.median():.4f} 最低 {rc.min():.4f} |Δ|>0.5 {rows[-1]['share_abs_diff_gt_0p5']:.3%}")
    pd.DataFrame(rows).to_csv(LEDGER / "stress_mvfix_exposure_delta.csv", index=False)
    for k in RS.STYLE_KEYS:                   # 原子替换(不删除:旧文件已在 pre_mvfix_backup)
        short = RS.SHORT[k]
        src, dst = STAGE_X / f"stress_X_{short}.parquet", TMP / f"stress_X_{short}.parquet"
        h = sha(src)
        os.replace(src, dst)
        assert sha(dst) == h
    log("stress_X_* 已替换为新口径")


# ====================================================================== shadow(旧口径 + 当前代码)
def _shadow_setup():
    from afactor import newdata as ND
    ND.TOTAL_SHARE_METHOD = "legacy"          # 诊断开关:只差 total_mv
    for sub in ("ledger", "tmp", "reports"):
        (SHADOW / sub).mkdir(parents=True, exist_ok=True)
    import run_stress_test as RS
    import stress_analyze as SA
    RS.TMP, RS.LEDGER, RS.REPORTS = SHADOW / "tmp", SHADOW / "ledger", SHADOW / "reports"
    SA.LEDGER = SHADOW / "ledger"
    # 旧口径纯因子收益:barra 重跑代理用当前代码 + legacy 开关算的 F(与正式账本同构:open10 / tail1450 两级表头)
    fo = TMP / "barra2_legacy_open10_F.parquet"
    ft = TMP / "barra2_legacy_tail1450_F.parquet"
    fr = pd.concat({"open10": pd.read_parquet(fo), "tail1450": pd.read_parquet(ft)}, axis=1)
    ref = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0)
    assert list(fr.columns) == list(ref.columns), "影子纯因子收益列与正式账本不同构"
    fr.to_csv(SHADOW / "ledger" / "barra_factor_returns.csv")
    # 只读输入:预注册(RS.PRE 在 import 时已从正式账本读入,这里只留副本)、维度方向用的 IC 汇总(dims 阶段读 LEDGER)
    for f in ("stress_preregistration.json", "factor_summary_all.csv"):
        shutil.copy2(LEDGER / f, SHADOW / "ledger" / f)
    return RS, SA, ND


def stage_shadow_legacy(steps: list[str]):
    RS, SA, ND = _shadow_setup()
    assert ND.TOTAL_SHARE_METHOD == "legacy"
    log("影子目录", SHADOW, "TOTAL_SHARE_METHOD =", ND.TOTAL_SHARE_METHOD, "steps", steps)
    if "engine" in steps:
        RS.stage_engine()
    if "dims" in steps:
        RS.stage_dims()
    if "analyze" in steps:
        SA.stage_analyze()
    if "figure" in steps:
        RS.stage_figure()
    log("影子完成")


# ====================================================================== compare
KEYS = {
    "stress_windows.csv": ["object", "window"],
    "stress_attribution.csv": ["object", "window", "component"],
    "stress_specific_breakdown.csv": ["object", "window", "dim", "bucket"],
    "stress_specific_matched.csv": ["kind", "object", "window", "matching", "n_bins", "bucket"],
    "stress_exante_vs_realized.csv": ["object", "window"],
    "stress_liquidity.csv": ["object", "window"],
    "stress_holdings_path.csv": ["object", "date"],
    "stress_counterfactual.csv": ["object", "variant", "period"],
    "stress_barra_clip.csv": ["window", "factor"],
    "stress_engine_checks.csv": ["object", "check"],
    "stress_participation_sensitivity.csv": ["object", "convention"],
    "stress_csi_alignment.csv": ["kind", "window"],
    "stress_fig_tail_values.csv": ["key"],
}


def _load(path: Path, keys: list[str]) -> pd.DataFrame:
    if path.name == "stress_fig_tail_values.csv":
        d = pd.read_csv(path)
        d.columns = ["key", "value"]
    else:
        d = pd.read_csv(path)
    for k in keys:
        if k not in d.columns:
            d[k] = ""
    d[keys] = d[keys].fillna("").astype(str)
    return d.set_index(keys)


def stage_compare():
    roots = {"orig": BK_LEDGER, "legacy": SHADOW / "ledger", "new": LEDGER}
    frows, drows = [], []
    for fname, keys in KEYS.items():
        D = {k: _load(r / fname, keys) for k, r in roots.items()}
        idx = D["orig"].index.union(D["legacy"].index).union(D["new"].index)
        cols = [c for c in D["orig"].columns if c in D["new"].columns]
        num = [c for c in cols if pd.api.types.is_numeric_dtype(D["orig"][c]) and pd.api.types.is_numeric_dtype(D["new"][c])]
        oth = [c for c in cols if c not in num]
        A = {k: v[~v.index.duplicated()].reindex(idx) for k, v in D.items()}
        stats = dict(file=fname, rows_orig=len(D["orig"]), rows_legacy=len(D["legacy"]), rows_new=len(D["new"]),
                     n_numeric_cols=len(num))
        for a, b in (("orig", "legacy"), ("legacy", "new"), ("orig", "new")):
            x, y = A[a][num].to_numpy(float), A[b][num].to_numpy(float)
            both = np.isfinite(x) & np.isfinite(y)
            ch = (both & (np.abs(x - y) > 1e-12 * np.maximum(1, np.abs(x)))) | (np.isfinite(x) != np.isfinite(y))
            s_ch = 0
            for c in oth:
                s_ch += int((A[a][c].astype(str) != A[b][c].astype(str)).sum())
            stats[f"cells_changed_{a}_vs_{b}"] = int(ch.sum()) + s_ch
            stats[f"max_abs_diff_{a}_vs_{b}"] = float(np.nanmax(np.where(both, np.abs(x - y), np.nan))) if both.any() else np.nan
        stats["n_numeric_cells"] = int(np.isfinite(A["new"][num].to_numpy(float)).sum())
        frows.append(stats)
        # 逐格长表(只记有变化的格)
        for c in num:
            o, l_, n = (A[k][c].astype(float) for k in ("orig", "legacy", "new"))
            m = ~(np.isclose(o, n, rtol=0, atol=1e-12, equal_nan=True) & np.isclose(o, l_, rtol=0, atol=1e-12, equal_nan=True))
            for key in idx[np.asarray(m)]:
                drows.append(dict(file=fname, key=" | ".join(key) if isinstance(key, tuple) else key, column=c,
                                  orig=o[key], legacy=l_[key], new=n[key]))
        for c in oth:
            o, l_, n = (A[k][c].astype(str) for k in ("orig", "legacy", "new"))
            m = (o != n) | (o != l_)
            for key in idx[m.to_numpy()]:
                drows.append(dict(file=fname, key=" | ".join(key) if isinstance(key, tuple) else key, column=c,
                                  orig=o[key], legacy=l_[key], new=n[key]))
    # 幸存者 JSON
    sv = {k: json.load(open(r / "stress_survivorship.json")) for k, r in roots.items()}
    frows.append(dict(file="stress_survivorship.json", rows_orig=len(sv["orig"]), rows_legacy=len(sv["legacy"]),
                      rows_new=len(sv["new"]),
                      cells_changed_orig_vs_legacy=int(sv["orig"] != sv["legacy"]),
                      cells_changed_legacy_vs_new=int(sv["legacy"] != sv["new"]),
                      cells_changed_orig_vs_new=int(sv["orig"] != sv["new"])))
    F = pd.DataFrame(frows)
    F.to_csv(LEDGER / "stress_mvfix_files.csv", index=False)
    Dd = pd.DataFrame(drows)
    Dd.to_csv(LEDGER / "stress_mvfix_diff.csv", index=False)
    log("stress_mvfix_files.csv")
    print(F.to_string())
    log("stress_mvfix_diff.csv", len(Dd))
    headlines()


def headlines():
    """报告引用的数字:原报告(_pre_mvfix 账本)/ 旧口径重跑(影子)/ 新口径(正式账本)。"""
    roots = {"orig": BK_LEDGER, "legacy": SHADOW / "ledger", "new": LEDGER}
    R = {k: {f: pd.read_csv(r / f) for f in KEYS if f != "stress_fig_tail_values.csv"} for k, r in roots.items()}
    T = {k: pd.read_csv(r / "stress_fig_tail_values.csv", index_col=0).iloc[:, 0] for k, r in roots.items()}
    out = []

    def add(section, name, fn, src, scale=1.0):
        v = {}
        for k in roots:
            try:
                v[k] = float(fn(R[k], T[k])) * scale
            except Exception:                 # noqa: BLE001
                v[k] = np.nan
        out.append(dict(section=section, item=name, orig=v["orig"], legacy=v["legacy"], new=v["new"],
                        d_drift=v["legacy"] - v["orig"], d_mv=v["new"] - v["legacy"], d_total=v["new"] - v["orig"],
                        source=src))

    def win(obj, w, col):
        return lambda r, t: r["stress_windows.csv"].set_index(["object", "window"]).loc[(obj, w), col]

    def att(obj, w, comp, col="value"):
        return lambda r, t: r["stress_attribution.csv"].set_index(["object", "window", "component"]).loc[(obj, w, comp), col]

    def ex(obj, w, col):
        return lambda r, t: r["stress_exante_vs_realized.csv"].set_index(["object", "window"]).loc[(obj, w), col]

    def liq(obj, w, col):
        return lambda r, t: r["stress_liquidity.csv"].set_index(["object", "window"]).loc[(obj, w), col]

    def cf(obj, var, per, col):
        return lambda r, t: r["stress_counterfactual.csv"].set_index(["object", "variant", "period"]).loc[(obj, var, per), col]

    def mt(obj, w, matching, nb, col):
        def f(r, t):
            d = r["stress_specific_matched.csv"]
            d = d[(d.kind == "匹配拆分") & (d.object == obj) & (d.window == w) & (d.matching == matching) & (d.n_bins == nb)]
            return d[col].iloc[0]
        return f

    def pb(w, matching, q):
        def f(r, t):
            d = r["stress_specific_matched.csv"]
            d = d[(d.kind == "池内分桶等权残差之和") & (d.window == w) & (d.matching == matching) & (d.bucket == q)]
            return d["value"].iloc[0]
        return f

    def clip(w, fac, col):
        return lambda r, t: r["stress_barra_clip.csv"].set_index(["window", "factor"]).loc[(w, fac), col]

    def path(obj, col, how, d0="2024-01-02", d1="2024-02-07"):
        def f(r, t):
            d = r["stress_holdings_path.csv"]
            d = d[d.object == obj]
            d = d[(d.te >= d0) & (d.te <= d1)]
            return {"first": d[col].iloc[0], "max": d[col].max(), "min": d[col].min()}[how]
        return f

    base = ["COMP_EXCL", "COMP_TOP20", "COMP_TOP50", "BGBT_TOP20"]
    pp = 100.0
    # §1 / §2.1 窗口(基础对象不依赖市值,应逐位不变)
    for o in base:
        for w in ("W1", "W1b", "W2", "W3", "W4", "W5", "W1+W1b"):
            add("§2.1 窗口超额", f"{o} {w} 超额 pp", win(o, w, "excess"), "stress_windows.csv", pp)
        add("§1 σ", f"{o} W1 σ 数", win(o, "W1", "excess_sigma"), "stress_windows.csv")
    for w in ("W1", "W1b", "W3"):
        add("§2.1 池 vs 指数", f"POOL_C2C {w} 超额 pp", win("POOL_C2C", w, "excess"), "stress_windows.csv", pp)
    # §3.1 纯因子收益
    for k in ["Size 市值", "NonlinSize 非线性市值", "EarningsYield 盈利", "Liquidity 流动性", "Leverage 杠杆",
              "ResidVol 残差波动", "Country", "BookToPrice 价值", "Momentum 动量", "Beta 市场敏感", "Growth 成长"]:
        add("§3.1 纯因子 W1", f"{k} W1 收益 %", att("FACTOR", "W1", k), "stress_attribution.csv object=FACTOR", pp)
        add("§3.1 纯因子 W1", f"{k} W1 冲击 z", att("FACTOR", "W1", k, "shock_z"), "stress_attribution.csv object=FACTOR")
    # §3.1 池等权绝对拆解
    for c in ["active", "Country", "style_total", "Size 市值", "Liquidity 流动性", "EarningsYield 盈利", "BookToPrice 价值",
              "ResidVol 残差波动", "NonlinSize 非线性市值", "Leverage 杠杆", "Momentum 动量", "Beta 市场敏感", "Growth 成长",
              "industry_total", "specific"]:
        add("§3.1 池等权绝对", f"POOL_EW W1 {c} pp", att("POOL_EW", "W1", c), "stress_attribution.csv object=POOL_EW", pp)
    # §3.2 主动拆解
    for o in base:
        for c in ["active", "cash(现金腿)", "style_total", "Size 市值", "NonlinSize 非线性市值", "Liquidity 流动性",
                  "EarningsYield 盈利", "ResidVol 残差波动", "BookToPrice 价值", "Leverage 杠杆", "Momentum 动量",
                  "Beta 市场敏感", "Growth 成长", "industry_total", "specific", "cost(负号=拖累)"]:
            add("§3.2 W1 主动拆解", f"{o} W1 {c} pp", att(o, "W1", c), "stress_attribution.csv", pp)
        add("§3.2 W1 暴露", f"{o} W1 Liquidity 平均暴露", att(o, "W1", "Liquidity 流动性", "mean_exposure"),
            "stress_attribution.csv mean_exposure")
        add("§3.2 W1 暴露", f"{o} W1 Size 平均暴露", att(o, "W1", "Size 市值", "mean_exposure"),
            "stress_attribution.csv mean_exposure")
        add("§3.2 W1 暴露", f"{o} W1 NonlinSize 平均暴露", att(o, "W1", "NonlinSize 非线性市值", "mean_exposure"),
            "stress_attribution.csv mean_exposure")
    # §3.3 匹配拆分
    for o in ["BGBT_TOP20", "COMP_TOP20", "COMP_TOP50", "COMP_EXCL"]:
        for matching, nb in (("全A市值", 5), ("全A市值", 50), ("全A市值", 100), ("池内过去20日收益", 5),
                             ("过去20日收益5档x全A市值5档", 25)):
            for col in ("specific_total", "bucket_part", "selection_part"):
                add("§3.3 匹配拆分 W1", f"{o} {matching}{nb} {col} pp", mt(o, "W1", matching, nb, col),
                    "stress_specific_matched.csv", pp)
            add("§3.3 匹配拆分 W1", f"{o} {matching}{nb} bucket_share", mt(o, "W1", matching, nb, "bucket_share"),
                "stress_specific_matched.csv")
        add("§3.3 已投资残差之和", f"{o} W1 port_invested_resid_sum %", mt(o, "W1", "全A市值", 5, "port_invested_resid_sum"),
            "stress_specific_matched.csv", pp)
        add("§3.3 匹配拆分 W1", f"{o} 收益档 unmatched_weight_mean", mt(o, "W1", "池内过去20日收益", 5, "unmatched_weight_mean"),
            "stress_specific_matched.csv")
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        add("§3.3 池内分档残差", f"全A市值 {q} %", pb("W1", "全A市值", q), "stress_specific_matched.csv", pp)
        add("§3.3 池内分档残差", f"过去20日收益 {q} %", pb("W1", "池内过去20日收益", q), "stress_specific_matched.csv", pp)
    add("§3.3 池内分档残差", "池等权残差之和 %", mt("COMP_TOP20", "W1", "全A市值", 5, "pool_ew_resid_sum"),
        "stress_specific_matched.csv", pp)
    # §3.4 持仓路径
    for o in ["POOL_EW"] + base:
        add("§3.4 持仓路径", f"{o} W1 首日微盘权重 %", path(o, "share_mvq1_allA", "first"), "stress_holdings_path.csv", pp)
        add("§3.4 持仓路径", f"{o} W1 内微盘权重最高 %", path(o, "share_mvq1_allA", "max"), "stress_holdings_path.csv", pp)
        add("§3.4 持仓路径", f"{o} W1 内过去20日跌最多权重最高 %", path(o, "share_past20_q1", "max"), "stress_holdings_path.csv", pp)
        add("§3.4 持仓路径", f"{o} W1 首日池内市值秩", path(o, "wavg_mv_rank_pool", "first"), "stress_holdings_path.csv")
        add("§3.4 持仓路径", f"{o} W1 内池内市值秩最低", path(o, "wavg_mv_rank_pool", "min"), "stress_holdings_path.csv")
    # §4.1 事前
    for o in base:
        for w in ("W1", "W3"):
            for col, sc in (("pred_total", pp), ("pred_style", pp), ("pred_cash", pp), ("realized_active", pp),
                            ("realized_style", pp), ("explained_ratio", 1), ("pred_te", pp), ("z_realized", 1),
                            ("hist_worst20d_before", pp), ("hist_worst_samelen_before", pp),
                            ("ratio_to_hist_worst_samelen", 1), ("visible_ex_ante", 1), ("pred_size_only", pp),
                            ("pred_nonlinsize_only", pp)):
                add(f"§4 事前 {w}", f"{o} {w} {col}", ex(o, w, col), "stress_exante_vs_realized.csv", sc)
        for k in ("Size", "NonlinSize", "BookToPrice", "EarningsYield", "Liquidity"):
            add("§4 事前 W1 暴露", f"{o} W1 pre_e_norm:{k}", ex(o, "W1", f"pre_e_norm:{k}"), "stress_exante_vs_realized.csv")
        for k in ("Size", "NonlinSize", "Liquidity"):
            add("§4 事前 W1 暴露", f"{o} W1 win_e_norm_mean:{k}", ex(o, "W1", f"win_e_norm_mean:{k}"),
                "stress_exante_vs_realized.csv")
    # §4.2 流动性 / 市值分布
    for o in base:
        for tag in ("port", "pool"):
            if tag == "pool" and o != "COMP_EXCL":
                continue
            nm = "POOL_EW" if tag == "pool" else o
            for col, sc in (("n_names", 1), ("wmed_adv20_wan", 1), ("share_adv_lt_5000wan", pp), ("share_adv_lt_3000wan", pp),
                            ("wmed_mv_yi", 1), ("share_mvq1_allA", pp), ("wavg_mv_rank_pool", 1)):
                add("§4.2 窗口前持仓", f"{nm} W1 {col}", liq(o, "W1", f"{tag}_{col}"), "stress_liquidity.csv", sc)
    # §8 反事实
    for o in base:
        for var in ("原始", "|N1", "|N3", "|N1r", "|NS"):
            if var != "原始":
                add("§8 反事实", f"{o}{var} W1 少亏 pp", cf(o, var, "W1", "avoided"), "stress_counterfactual.csv", pp)
                add("§8 反事实", f"{o}{var} W1b 少赚 pp", cf(o, var, "W1b", "avoided"), "stress_counterfactual.csv", pp)
                add("§8 反事实", f"{o}{var} W3 差 pp", cf(o, var, "W3", "avoided"), "stress_counterfactual.csv", pp)
                add("§8 反事实", f"{o}{var} W1 窗口超额 pp", cf(o, var, "W1", "window_excess"), "stress_counterfactual.csv", pp)
                for per in ("FULL_2021_2025", "DEV_2021_2023", "TEST_2024_2025"):
                    add("§8 反事实", f"{o}{var} {per} 年化差 pp", cf(o, var, per, "diff_vs_orig_ann"), "stress_counterfactual.csv", pp)
                    add("§8 反事实", f"{o}{var} {per} t", cf(o, var, per, "diff_vs_orig_nw_t"), "stress_counterfactual.csv")
            add("§8 反事实", f"{o}{var} 全期 池内市值秩", cf(o, var, "FULL_2021_2025", "mean_mv_rank_pool"), "stress_counterfactual.csv")
            add("§8 反事实", f"{o}{var} 全期 Size 归一化暴露", cf(o, var, "FULL_2021_2025", "mean_e_norm_size"), "stress_counterfactual.csv")
            add("§8 反事实", f"{o}{var} 全期 Liquidity 归一化暴露", cf(o, var, "FULL_2021_2025", "mean_e_norm_liq"), "stress_counterfactual.csv")
            add("§8 反事实", f"{o}{var} 全期 微盘权重", cf(o, var, "FULL_2021_2025", "mean_share_mvq1_allA"), "stress_counterfactual.csv")
    for k in ["COMP_TOP20", "COMP_TOP50", "BGBT_TOP20", "COMP_TOP20|NS", "COMP_TOP50|NS", "BGBT_TOP20|NS", "COMP_EXCL",
              "CSI1000_vs_POOL_C2C"]:
        add("§8 图线尾", f"{k} 2024-03-29 累计 %", lambda r, t, k=k: t[k], "stress_fig_tail_values.csv")
    # §7 截尾诊断
    for w in ("W1", "W1b", "W2", "W3", "W4", "W5"):
        add("§7 截尾", f"{w} Size 池内 ≤ -3 比例", clip(w, "Size 市值", "share_le_minus3"), "stress_barra_clip.csv", pp)
        add("§7 截尾", f"{w} Size 池等权均值", clip(w, "Size 市值", "pool_ew_mean"), "stress_barra_clip.csv")

    # ---- 审查补充(2026-10-10,P2"旧 → 新清单不全"):正文 §3.2 / §3.3 / §7.2 / §8 / §9 引用、原先只在逐格长表里的数字
    def bd(obj, w, dim, bucket, col):
        def f(r, t):
            d = r["stress_specific_breakdown.csv"]
            d = d[(d.object == obj) & (d.window == w) & (d.dim == dim) & (d.bucket == bucket)]
            return d[col].iloc[0]
        return f

    def path_at(obj, col, te):
        def f(r, t):
            d = r["stress_holdings_path.csv"]
            return d[(d.object == obj) & (d.te == te)][col].iloc[0]
        return f

    def path_argmax(obj, col, d0="2024-01-02", d1="2024-02-07"):
        """窗口内最大值所在的 te,编码成 yyyymmdd 数字。"""
        def f(r, t):
            d = r["stress_holdings_path.csv"]
            d = d[(d.object == obj) & (d.te >= d0) & (d.te <= d1)]
            return float(d.loc[d[col].idxmax(), "te"].replace("-", ""))
        return f

    comps9 = ["active", "cash(现金腿)", "style_total", "Size 市值", "NonlinSize 非线性市值", "Liquidity 流动性",
              "EarningsYield 盈利", "Beta 市场敏感", "ResidVol 残差波动", "industry_total", "specific", "cost(负号=拖累)"]
    for o in base:                                         # §9 / §7.2:其他窗口的主动拆解(B_gbt 无 W5 -> NaN)
        for w in ("W1b", "W2", "W3", "W4", "W5"):
            for c in comps9:
                add(f"§9 {w} 主动拆解", f"{o} {w} {c} pp", att(o, w, c), "stress_attribution.csv", pp)
        add("§9 W3 暴露", f"{o} W3 Beta 平均暴露", att(o, "W3", "Beta 市场敏感", "mean_exposure"),
            "stress_attribution.csv mean_exposure")
    for o in base:                                         # §8:变体的 W1 拆解(N3 / NS 读法)
        for var in ("|N1", "|N3", "|N1r", "|NS"):
            for c in ("active", "style_total", "Size 市值", "NonlinSize 非线性市值", "Liquidity 流动性",
                      "industry_total", "specific"):
                add("§8 变体 W1 拆解", f"{o}{var} W1 {c} pp", att(o + var, "W1", c), "stress_attribution.csv", pp)
    buckets = {"全A市值五分位": ["Q1最小", "Q2", "Q3", "Q4", "Q5最大"],
               "池内过去20日收益五分位": ["Q1跌最多", "Q2", "Q3", "Q4", "Q5涨最多"]}
    for o in base:                                         # §3.3 第一版分桶加总(描述性)
        for dim, qs in buckets.items():
            for q in qs:
                add("§3.3 特质分桶加总 W1", f"{o} W1 {dim} {q} specific_contrib pp", bd(o, "W1", dim, q, "specific_contrib"),
                    "stress_specific_breakdown.csv", pp)
                add("§3.3 特质分桶加总 W1", f"{o} W1 {dim} {q} port_weight_mean %", bd(o, "W1", dim, q, "port_weight_mean"),
                    "stress_specific_breakdown.csv", pp)
        for q in ("ST", "非ST"):                            # §9 W2 的 ST 一档
            add("§9 W2 ST", f"{o} W2 {q} specific_contrib pp", bd(o, "W2", "ST", q, "specific_contrib"),
                "stress_specific_breakdown.csv", pp)
            add("§9 W2 ST", f"{o} W2 {q} port_weight_mean %", bd(o, "W2", "ST", q, "port_weight_mean"),
                "stress_specific_breakdown.csv", pp)
    add("§9 W2 ST", "POOL W2 ST pool_weight_mean %", bd("COMP_EXCL", "W2", "ST", "ST", "pool_weight_mean"),
        "stress_specific_breakdown.csv", pp)
    for o in ("COMP_TOP20", "COMP_TOP50"):                 # §3.2 / §3.4:暴露时点与微盘峰值日期
        for te in ("2024-01-22", "2024-01-25", "2024-01-31", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07"):
            for col in ("e_norm_size", "e_norm_nls", "share_mvq1_allA"):
                add("§3.4 持仓路径(时点)", f"{o} {te} {col}", path_at(o, col, te), "stress_holdings_path.csv",
                    pp if col == "share_mvq1_allA" else 1)
        add("§3.4 持仓路径(时点)", f"{o} W1 微盘权重峰值日 yyyymmdd", path_argmax(o, "share_mvq1_allA"),
            "stress_holdings_path.csv")
    H = pd.DataFrame(out)
    H.to_csv(LEDGER / "stress_mvfix_headlines.csv", index=False)
    log("stress_mvfix_headlines.csv", len(H))


# ====================================================================== clip_mechanism(审查 P2:§7 第 1 条的机制)
def stage_clip_mechanism():
    """为什么新市值下 Size 截在 −3 的比例下降:逐窗口前一日,在估计池 u 上比较新 / 旧口径的 log 市值分布。

    Barra Size = (lncap − 市值加权均值) / 等权标准差,截尾 ±3(barra.zscore,u 上算矩)。截在 −3 的比例由
    "市值加权中心偏离等权中心多少个标准差"(gap_sd)决定:gap_sd 越大,等权分布里越多股票落在中心左侧 3σ 之外。
    输出 research/ledger/stress_mvfix_clip_mechanism.csv(长表):
      kind=moments  每个 (窗口, 口径):n、log 市值等权均值 / 市值加权均值 / 等权标准差、gap_sd、z ≤ −3 比例
                    (按定义重算,与 stress_barra_clip.csv 核对)、正态近似 Φ(gap_sd − 3)。
      kind=ratio_by_quintile  旧 / 新市值之比,按新口径池内市值五分位:均值、中位、> 1.5 的格子比例。
    只读研究期面板,不写缓存。"""
    from scipy.stats import norm
    from afactor.panel import Panel
    from afactor import newdata as ND, ops
    assert ND.TOTAL_SHARE_METHOD == "songzhuan"
    clipf = pd.read_csv(LEDGER / "stress_barra_clip.csv")
    clipf = clipf[clipf.factor == "Size 市值"].set_index("window")
    clip_legacy = pd.read_csv(SHADOW / "ledger" / "stress_barra_clip.csv")
    clip_legacy = clip_legacy[clip_legacy.factor == "Size 市值"].set_index("window")
    p = Panel()
    u = p["universe"].reindex(index=p.dates, columns=p.codes).fillna(False).astype(bool)
    mv = {}
    for meth in ("songzhuan", "legacy"):
        ND.TOTAL_SHARE_METHOD = meth
        t0 = time.time()
        mv[meth] = ND.total_mv(p.dates, p.codes).reindex(index=p.dates, columns=p.codes)
        log(f"total_mv[{meth}] {time.time() - t0:.0f}s")
    ND.TOTAL_SHARE_METHOD = "songzhuan"
    Xs = pd.read_parquet(TMP / "stress_X_Size.parquet")
    rows = []
    for w, r in clipf.iterrows():
        d = pd.Timestamp(r["date"])
        m = u.loc[d]
        for meth, tag in (("songzhuan", "new"), ("legacy", "legacy")):
            v = mv[meth].loc[d].where(m).astype(np.float64)
            x = ops.signed_log(v)
            ok = x.notna()
            x, wv = x[ok], v[ok]
            mu_e, sd = x.mean(), x.std()
            mu_w = float((x * wv).sum() / wv.sum())
            z = (x - mu_w) / sd
            gap = (mu_w - mu_e) / sd
            ledger_share = (clipf if tag == "new" else clip_legacy).loc[w, "share_le_minus3"]
            for k, val in (("n", int(ok.sum())), ("lncap_mean_ew", mu_e), ("lncap_mean_capw", mu_w), ("lncap_sd", sd),
                           ("gap_sd", gap), ("share_z_le_m3", float((z <= -3).mean())),
                           ("normal_approx_share", float(norm.cdf(gap - 3))),
                           ("ledger_share_le_minus3", float(ledger_share)),
                           ("pool_mean_z_unclipped", float(z.mean()))):
                rows.append(dict(kind="moments", window=w, date=str(d.date()), method=tag, bucket="", metric=k, value=val))
        if True:                                           # 新口径按定义重算的 z 与缓存暴露核对
            xs = Xs.loc[d].where(m).dropna().astype(np.float64)
            v = mv["songzhuan"].loc[d].where(m).astype(np.float64)
            x = ops.signed_log(v).dropna()
            zz = ((x - float((x * v[x.index]).sum() / v[x.index].sum())) / x.std()).clip(-3, 3)
            j = xs.index.intersection(zz.index)
            rows.append(dict(kind="check", window=w, date=str(d.date()), method="new", bucket="",
                             metric="max_abs_diff_vs_stress_X_Size", value=float((zz[j] - xs[j]).abs().max())))
        # 旧 / 新之比,按新口径池内市值五分位
        vn = mv["songzhuan"].loc[d].where(m).astype(np.float64)
        vl = mv["legacy"].loc[d].where(m).astype(np.float64)
        ok = vn.notna() & vl.notna() & (vn > 0)
        ratio = (vl / vn)[ok]
        q = pd.qcut(vn[ok].rank(method="first"), 5, labels=["Q1最小", "Q2", "Q3", "Q4", "Q5最大"])
        for b, g in ratio.groupby(q):
            for k, val in (("ratio_mean", g.mean()), ("ratio_median", g.median()), ("share_ratio_gt_1p5", (g > 1.5).mean()),
                           ("n", len(g))):
                rows.append(dict(kind="ratio_by_quintile", window=w, date=str(d.date()), method="legacy/new", bucket=str(b),
                                 metric=k, value=float(val)))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / "stress_mvfix_clip_mechanism.csv", index=False)
    piv = out[out.kind == "moments"].pivot_table(index=["window", "date", "method"], columns="metric", values="value")
    print(piv[["lncap_sd", "gap_sd", "share_z_le_m3", "ledger_share_le_minus3", "normal_approx_share"]].round(4).to_string())
    print(out[out.kind == "check"][["window", "value"]].to_string())
    rq = out[out.kind == "ratio_by_quintile"].pivot_table(index=["window", "bucket"], columns="metric", values="value")
    print(rq.round(3).to_string())
    log("stress_mvfix_clip_mechanism.csv", len(out))


# ====================================================================== coverage(审查 P1/P2:正文残留旧值与旧 → 新清单覆盖)
def stage_coverage():
    """对 STRESS_TEST.md §1 ~ §12 做两项文本检查,写 research/ledger/stress_mvfix_coverage.csv:
      kind=reverse_uncovered  §1、§3、§4、§7、§8、§9、§12 里的每个小数,按它显示的位数在 headlines 的 new 列
                              (含 ×100)或 stress_mvfix_clip_mechanism.csv 里找不到同值的 —— 需要人工确认它不依赖市值、
                              是明确标注的旧口径值,或是两个已覆盖数之和;
      kind=stale_candidate    headlines 中 orig 与 new 在 2 位小数上不同的行,其 orig 值出现在同对象的正文行里、
                              该行既没有对应 new 值也没有"旧口径 / 旧市值 / 第一版 / 原文"标注 —— 需要人工确认是巧合同值。
    只做文本匹配,结论靠人工复核(报告"审查处理"一节写明复核结果)。"""
    import re
    H = pd.read_csv(LEDGER / "stress_mvfix_headlines.csv")
    C = pd.read_csv(LEDGER / "stress_mvfix_clip_mechanism.csv")
    md = STRESS_MD.read_text(encoding="utf-8").split("\n")

    def rng(start, stop):
        i0 = next(i for i, l in enumerate(md) if l.startswith(start))
        i1 = next(i for i, l in enumerate(md) if i > i0 and l.startswith(stop))
        return range(i0, i1)
    body_end = next(i for i, l in enumerate(md) if l.startswith("## 13."))
    rows = []
    vals = np.concatenate([H["new"].dropna().to_numpy(), C["value"].to_numpy(), C["value"].to_numpy() * 100])
    vals = np.abs(vals[np.isfinite(vals)])
    num = re.compile(r"(?<![\w.])[−+-]?(\d+\.\d+)")
    for rg in (rng("## 1.", "## 2."), rng("## 3.", "## 5."), rng("## 7.", "## 10."), rng("## 12.", "## 13.")):
        for i in rg:
            for m in num.finditer(md[i]):
                t = m.group(1)
                d = len(t.split(".")[1])
                if not np.any(np.abs(np.round(vals, d) - float(t)) < 1e-9):
                    rows.append(dict(kind="reverse_uncovered", line=i + 1, token=t, item="", new="", text=md[i].strip()[:200]))
    kw = {"COMP_EXCL": ["排雷"], "COMP_TOP20": ["Top20", "Top-N", "选优"], "COMP_TOP50": ["Top50", "Top-N", "选优"],
          "BGBT_TOP20": ["B_gbt", "选优"], "POOL_EW": ["池等权", "池"], "POOL_C2C": ["池"]}
    old_mark = ("旧口径", "旧市值", "第一版", "原文")
    for _, r in H.iterrows():
        o, n = r["orig"], r["new"]
        if not (np.isfinite(o) and np.isfinite(n)) or abs(o) < 0.1:
            continue
        so, sn = f"{abs(o):.2f}", f"{abs(n):.2f}"
        if so == sn:
            continue
        keys = next((v for k, v in kw.items() if r["item"].startswith(k)), None)
        po = re.compile(r"(?<![\d.])" + re.escape(so) + r"(?![\d])")
        pn = re.compile(r"(?<![\d.])" + re.escape(sn) + r"(?![\d])")
        for i in range(body_end):
            l = md[i]
            if po.search(l) and not pn.search(l) and (keys is None or any(k in l for k in keys)) \
                    and not any(x in l for x in old_mark):
                rows.append(dict(kind="stale_candidate", line=i + 1, token=so, item=r["item"], new=sn, text=l.strip()[:200]))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / "stress_mvfix_coverage.csv", index=False)
    log("stress_mvfix_coverage.csv", out.groupby("kind").size().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["backup", "exposures", "shadow_legacy", "compare", "headlines", "clip_mechanism",
                                      "coverage"])
    ap.add_argument("--steps", default="engine,dims,analyze,figure")
    a = ap.parse_args()
    if a.stage == "backup":
        stage_backup()
    elif a.stage == "exposures":
        stage_exposures()
    elif a.stage == "shadow_legacy":
        stage_shadow_legacy(a.steps.split(","))
    elif a.stage == "compare":
        stage_compare()
    elif a.stage == "headlines":
        headlines()
    elif a.stage == "clip_mechanism":
        stage_clip_mechanism()
    elif a.stage == "coverage":
        stage_coverage()
