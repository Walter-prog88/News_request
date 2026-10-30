"""阶段3:截面模型 —— 特征集 A/B/C × 模型族 × 滚动 3 年验证 + 因子归因分层。

滚动:2020-22 训 -> 测 2023;2021-23 -> 2024;2022-24 -> 2025。三个 OOS 年份。
**样本内指标不报告**(因子是用这六年挑的,再用这六年训练,样本内无信息量)。

归因(用户明确要求):置换重要性 -> 按原始 |Rank IC| 分成
  「线性信号」(IC>=0.02,模型只是复现已知线性关系)
  「模型特有信号」(IC<0.02 但模型重用 -> 非线性/U型/交互,是本实验的核心产出)
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor import factors as FA, mlmodel as ML, screen, strategy as ST

pd.set_option("display.width", 250)
TRAIN_YEARS = 3
OOS_YEARS = [2023, 2024, 2025]


def ic_lookup(mode: str = "open") -> dict:
    """按口径取原始 IC。此前硬编码 "open",尾盘模式下归因会用错 IC
    (两口径分类不一致 9.4%,符号相反 6.7%)。"""
    led = screen.load_ledger("trials_[0-9]*.jsonl")
    led = led[led["mode"] == mode].drop_duplicates("factor", keep="last")
    return dict(zip(led["factor"], led["ic_mean"]))


def main():
    t0 = time.time()
    FA.build_all()
    # 两套 mode 命名空间:fwd_ret/EvalContext 用 "open"/"tail";strategy 用执行口径名
    mode = os.environ.get("ML_MODE", "open")
    exec_mode = {"open": "open10", "tail": "tail1450"}[mode]
    horizon = int(os.environ.get("ML_HORIZON", "5"))
    p = TailPanel() if mode == "tail" else Panel()
    u = p["universe"]
    icmap = ic_lookup(mode)

    # C 集的候选:过一致性关的因子
    led = screen.load_ledger("trials_[0-9]*.jsonl")
    led = led[led["mode"] == mode]
    g2 = screen.gate_consistency(screen.gate_stats(led))
    # 按最差滚动窗口 IC 排序(信息量优先),数量由 mlmodel.MAX_FEATURES 兜底
    g2 = g2.sort_values("worst_window_abs_ic", ascending=False)
    cnames = [n for n in g2["factor"].tolist() if n in FA.REGISTRY][:200]

    rows, imps = [], {}
    # ML_SETS 可只跑指定特征集(如 h=1 只需 C 集来比 1 日 vs 5 日的收益);
    # ML_NOIMP=1 跳过置换重要性(归因只需在主口径 h=5 做一次)。
    sets_env = os.environ.get("ML_SETS", "A,B,C").split(",")
    for fs_name in [x for x in ["A", "B", "C"] if x in sets_env]:
        print(f"\n{'='*120}\n特征集 {fs_name} | mode={mode} horizon={horizon}日", flush=True)
        feats = ML.feature_set(p, fs_name, cnames)
        X, Y, U, Ulab, names = ML.build_xy(feats, p, u, horizon, mode)
        print(f"  特征 {len(names)} 个,样本矩阵 {X.shape},构建耗时 {time.time()-t0:.0f}s", flush=True)
        years = np.asarray(p.dates.year)
        for kind in ["ridge", "gbt"]:
            for oos in OOS_YEARS:
                tr = ML.purge_train((years >= oos - TRAIN_YEARS) & (years < oos), horizon)
                te = years == oos
                if tr.sum() < 300 or te.sum() < 100:
                    continue
                Xtr, Ytr, dtr = ML.flatten(X, Y, Ulab, np.flatnonzero(tr))
                m = ML.fit_ts(ML.make_model(kind), Xtr, Ytr, dtr, purge=horizon + 1)
                score = ML.predict_panel(m, X[:, te], U[te], p.dates[te], p.codes)
                # 存盘:以后改策略配置(相位/持仓数/择时)只要 2 分钟重算,不必重训
                sd = LEDGER / "scores"; sd.mkdir(exist_ok=True)
                score.to_parquet(sd / f"{fs_name}_{kind}_{mode}_h{horizon}_{oos}.parquet")
                # OOS 年份的策略表现(用基线同口径)
                full = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
                full.loc[score.index] = score.to_numpy()
                dm = te
                for version in ["V1000", "V100"]:
                    bm = ST.benchmark(p, version, exec_mode, date_mask=dm)
                    for plan, kw in [("Top20周频", dict(scheme="top", n=20, rebalance=5)),
                                     ("排雷缓冲20/35", dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5))]:
                        r = ST.run_strategy(full, p, version, exec_mode, date_mask=dm, **kw)
                        if not np.isfinite(r.get("ann_return", np.nan)):
                            continue
                        pa = r.get("phase_ann") or [r["ann_return"]]
                        rows.append(dict(feature_set=fs_name, model=kind, oos_year=oos,
                                         version=version, plan=plan, horizon=horizon, mode=mode,
                                         ann=r["ann_return"], excess=r["ann_return"] - bm["ann_return"],
                                         sharpe=r["sharpe"], max_dd=r["max_dd"],
                                         turnover=r["turnover"], bench=bm["ann_return"],
                                         exposure=r.get("exposure"),
                                         phase_spread=float(max(pa) - min(pa)),
                                         phase_std=float(np.std(pa))))
                # IC 口径的 OOS 表现
                from afactor.fast_eval import EvalContext, eval_one
                ctx = EvalContext(p, mode, (horizon,))
                M = ctx.M[horizon].copy(); M[~te] = False
                ic, _, _, _ = eval_one(full.to_numpy(np.float32), ctx.R[horizon], M, want_layers=False)
                v = ic[np.isfinite(ic)]
                from afactor.fast_eval import _nw_t
                # 重叠标签必须用 Newey-West:朴素 t 在 h=5 下虚高约 2.1 倍
                tnw = _nw_t(v, max(5, 2 * horizon))
                nit = getattr(m, "_n_iter_chosen", getattr(m, "n_iter_", None))
                print(f"  {kind:6s} OOS {oos}: IC={v.mean():+.4f} t_NW={tnw:+.1f} "
                      f"(朴素t={v.mean()/v.std(ddof=1)*np.sqrt(v.size):+.1f}, 迭代={nit})", flush=True)
                # 归因(只在最后一个 OOS 年做,省时间)
                if oos == OOS_YEARS[-1] and kind == "gbt" and not os.environ.get("ML_NOIMP"):
                    imp = ML.permutation_importance(m, X, Y, Ulab, np.flatnonzero(te), names, seed=0)
                    imps[fs_name] = ML.attribute_importance(imp, icmap)
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"ml_results_{mode}_h{horizon}.csv", index=False)
    print(f"\n{'='*120}\n=== OOS 年份平均表现(按特征集 × 模型 × 用法)===")
    if len(df):
        g = df.groupby(["feature_set", "model", "version", "plan"]).agg(
            超额=("excess", "mean"), 夏普=("sharpe", "mean"), 回撤=("max_dd", "mean"),
            换手=("turnover", "mean"), 仓位=("exposure", "mean"),
            相位极差=("phase_spread", "mean")).round(4)
        print(g.to_string())
    for k, d in imps.items():
        d.to_csv(LEDGER / f"ml_importance_{k}_{mode}_h{horizon}.csv")
        print(f"\n=== 特征集 {k} 的重要性归因(前 20)===")
        print(d.head(20).to_string())
        print(f"  按类别汇总(重要性总和):")
        print(d.groupby("类别")["importance"].agg(["sum", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
