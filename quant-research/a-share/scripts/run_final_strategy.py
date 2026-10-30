"""阶段5:整合 —— 截面模型 × 时序仓位 -> 两个版本的完整策略 -> 逐年 OOS 报告。

流程:
  1. 选定截面模型(特征集 + 模型族,依据阶段3 的 OOS 表现,不看样本内)
  2. 滚动训练打分,拼成全期 OOS 打分面板
  3. 叠加时序仓位(规则版 / 小模型版 / 满仓对照)
  4. 两个资金版本 × 两种用法(集中选优 / 排雷)-> 与基线逐年对比
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor import factors as FA, mlmodel as ML, strategy as ST, timing as TM, screen

pd.set_option("display.width", 250)
OOS_YEARS = [2023, 2024, 2025]
TRAIN_YEARS = 3


def rolling_scores(p, feats, u, horizon, mode, kind):
    """滚动训练,拼出全期 OOS 打分面板(每年只用之前 3 年训练)。"""
    X, Y, U, Ulab, names = ML.build_xy(feats, p, u, horizon, mode)
    years = np.asarray(p.dates.year)
    out = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
    for oos in OOS_YEARS:
        tr = ML.purge_train((years >= oos - TRAIN_YEARS) & (years < oos), horizon)
        te = years == oos
        if tr.sum() < 300:
            continue
        Xtr, Ytr, dtr = ML.flatten(X, Y, Ulab, np.flatnonzero(tr))
        m = ML.fit_ts(ML.make_model(kind), Xtr, Ytr, dtr, purge=horizon + 1)
        sc = ML.predict_panel(m, X[:, te], U[te], p.dates[te], p.codes)
        out.loc[sc.index] = sc.to_numpy()
    return out


def main():
    FA.build_all()
    mode = os.environ.get("FS_MODE", "open")
    exec_mode = {"open": "open10", "tail": "tail1450"}[mode]
    horizon = int(os.environ.get("FS_HORIZON", "5"))
    fs_name = os.environ.get("FS_FEATURES", "B")
    kind = os.environ.get("FS_MODEL", "gbt")
    p = TailPanel() if mode == "tail" else Panel()
    u = p["universe"]
    print(f"整合:特征集 {fs_name} × {kind} × horizon {horizon} × {exec_mode}", flush=True)

    led = screen.load_ledger("trials_[0-9]*.jsonl")
    led = led[led["mode"] == mode]
    cnames = [n for n in screen.gate_consistency(screen.gate_stats(led))["factor"].tolist()
              if n in FA.REGISTRY][:400]
    feats = ML.feature_set(p, fs_name, cnames)
    score = rolling_scores(p, feats, u, horizon, mode, kind)
    score.to_parquet(LEDGER / f"oos_scores_{fs_name}_{kind}_{mode}_h{horizon}.parquet")

    feat = TM.market_features(p)
    is_tail = getattr(p, "_is_tail", False)
    pos_rule = TM.rule_position(feat, tail=is_tail)
    # 恒定仓位对照:择时的**净**贡献 = 择时后 - 恒定同均值仓位后。
    # 缺了这个对照,"降低回撤"里绝大部分其实只是去杠杆的功劳,不是择时。
    # ⚠ 必须用**年内**均值:用全期均值既含未来信息,又与年内实际暴露错配,
    #    把纯粹的杠杆水平差当成择时贡献(实测 2025 年约 -1.5pp 全由此产生)。
    print(f"规则仓位:全期均值 {pos_rule.mean():.3f},取值分布 "
          f"{pos_rule.value_counts().to_dict()},全期切换 {int((pos_rule.diff()!=0).sum())} 次")
    years = np.asarray(p.dates.year)
    rows = []
    for version in ["V1000", "V100"]:
        for plan, kw in [("集中 Top20周频", dict(scheme="top", n=20, rebalance=5)),
                         ("集中 Top10周频", dict(scheme="top", n=10, rebalance=5)),
                         ("排雷缓冲20/35", dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5))]:
            for oos in OOS_YEARS:
                dm = years == oos
                bm = ST.benchmark(p, version, exec_mode, date_mask=dm)
                r = ST.run_strategy(score, p, version, exec_mode, date_mask=dm, **kw)
                if not np.isfinite(r.get("ann_return", np.nan)):
                    continue
                dr = r["rets"]                       # 日收益序列(与基准同索引)
                def _stat(d):
                    nv = (1 + d).cumprod()
                    return (float(nv.iloc[-1] ** (242 / len(nv)) - 1),
                            float((nv / nv.cummax() - 1).min()),
                            float(d.mean() / d.std() * np.sqrt(242)) if d.std() > 0 else np.nan)
                pr = pos_rule.reindex(dr.index)
                pf = pd.Series(float(pr.mean()), index=dr.index)   # 年内均值
                a_t, dd_t, sh_t = _stat(TM.apply_position(dr, pr))
                a_f, dd_f, sh_f = _stat(TM.apply_position(dr, pf))
                rows.append(dict(version=version, plan=plan, oos_year=oos,
                                 bench=bm["ann_return"], ann=r["ann_return"],
                                 excess=r["ann_return"] - bm["ann_return"],
                                 sharpe=r["sharpe"], max_dd=r["max_dd"], turnover=r["turnover"],
                                 exposure=r.get("exposure", np.nan),
                                 ann_timed=a_t, dd_timed=dd_t, sharpe_timed=sh_t,
                                 ann_flat=a_f, dd_flat=dd_f, sharpe_flat=sh_f,
                                 择时净年化=a_t - a_f, 择时净回撤=dd_t - dd_f))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"final_strategy_{fs_name}_{kind}_{mode}_h{horizon}.csv", index=False)
    print("\n=== 逐年 OOS(模型打分,含成本)===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== 三年平均 ===")
    g = df.groupby(["version", "plan"]).agg(
        基准=("bench", "mean"), 年化=("ann", "mean"), 超额=("excess", "mean"),
        夏普=("sharpe", "mean"), 回撤=("max_dd", "mean"), 换手=("turnover", "mean"),
        仓位=("exposure", "mean"),
        择时后年化=("ann_timed", "mean"), 择时后回撤=("dd_timed", "mean"),
        恒定仓位年化=("ann_flat", "mean"), 恒定仓位回撤=("dd_flat", "mean"),
        择时净年化=("择时净年化", "mean"), 择时净回撤=("择时净回撤", "mean")).round(4)
    print(g.to_string())
    print("\n=== 择时的**净**贡献(相对同均值恒定仓位;正数才是择时真的有用)===")
    print(df.groupby("oos_year")[["择时净年化", "择时净回撤"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
