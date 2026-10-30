"""特征集选取标准的消融实验(由归因结论直接推出)。

归因发现:C 集按"最差滚动窗口 |IC| 最高"选 144 个因子,但置换重要性里
扛 61.4% 权重的是 |IC| 落在 0.010~0.025 的**边缘**特征,而 190 个 |IC|>=0.025 的
强线性因子合计只占 32.5%;Spearman(重要性, |IC|) = 0.079 ≈ 0。

若该现象成立,则"按 IC 高低选因子"对模型层是错误标准。三组对照(都在 B 集之上加 144 个):
  C  : 高 IC 组   —— 现行标准,|IC|>=0.025 中按 worst_window_abs_ic 取前 144
  D  : 中段 IC 组 —— 0.010<=|IC|<0.025(被关卡刷掉的 516 个)中按 |t| 取前 144
  E  : 多样性组   —— 跨 family 轮询,不看 IC 高低,每族内按 |t| 排序

对齐口径:同样的滚动 3 年训练、同样的 purge、同样的 fit_ts 早停、同样的策略回测。
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, IC_THRESHOLD
from afactor.panel import Panel, TailPanel
from afactor import factors as FA, mlmodel as ML, screen, strategy as ST
from afactor.fast_eval import EvalContext, eval_one, _nw_t

pd.set_option("display.width", 250)
OOS_YEARS = [2023, 2024, 2025]
TRAIN_YEARS = 3


def pick_sets(mode: str, n_add: int) -> dict[str, list[str]]:
    led = screen.load_ledger("trials_[0-9]*.jsonl")
    led = led[(led["mode"] == mode)].drop_duplicates("factor", keep="last")
    led = led[led["ic_mean"].notna()].copy()
    led["abs_ic"] = led["ic_mean"].abs()
    led["abs_t"] = led["ic_t"].abs()
    led = led[led["factor"].isin(FA.REGISTRY)]

    g2 = screen.gate_consistency(screen.gate_stats(led))
    hi = g2.sort_values("worst_window_abs_ic", ascending=False)["factor"].tolist()[:n_add]

    mid = led[(led["abs_ic"] >= 0.010) & (led["abs_ic"] < IC_THRESHOLD)]
    mid = mid.sort_values("abs_t", ascending=False)["factor"].tolist()[:n_add]

    # 多样性:跨 family 轮询(不看 IC 高低)
    div, pools = [], {}
    for fam, d in led.sort_values("abs_t", ascending=False).groupby("family"):
        pools[fam] = d["factor"].tolist()
    fams = sorted(pools, key=lambda f: -len(pools[f]))
    i = 0
    while len(div) < n_add and any(pools.values()):
        f = fams[i % len(fams)]
        if pools[f]:
            div.append(pools[f].pop(0))
        i += 1
    return {"C_高IC": hi, "D_中段IC": mid, "E_多样性": div}


def main():
    t0 = time.time()
    FA.build_all()
    mode = os.environ.get("AB_MODE", "open")
    exec_mode = {"open": "open10", "tail": "tail1450"}[mode]
    horizon = int(os.environ.get("AB_HORIZON", "5"))
    p = TailPanel() if mode == "tail" else Panel()
    u = p["universe"]
    n_base = len(ML.feature_set(p, "B", []))
    n_add = ML.MAX_FEATURES - n_base
    sets = pick_sets(mode, n_add)
    print(f"B 集基座 {n_base} 个,每组追加 {n_add} 个 | mode={mode} horizon={horizon}日\n")
    led = screen.load_ledger("trials_[0-9]*.jsonl")
    led = led[led["mode"] == mode].drop_duplicates("factor", keep="last")
    icm = dict(zip(led["factor"], led["ic_mean"].abs()))
    for k, v in sets.items():
        a = np.array([icm.get(x, np.nan) for x in v], float)
        fams = pd.Series([FA.REGISTRY[x][0] if x in FA.REGISTRY else "?" for x in v]).value_counts()
        print(f"  {k}: {len(v)} 个,|IC| 中位 {np.nanmedian(a):.4f} "
              f"[{np.nanmin(a):.4f},{np.nanmax(a):.4f}],涉及 {len(fams)} 个 family "
              f"(最大族 {fams.index[0]}×{fams.iloc[0]})")

    years = np.asarray(p.dates.year)
    rows = []
    for label, names in sets.items():
        print(f"\n{'='*110}\n特征集 {label}", flush=True)
        feats = ML.feature_set(p, "C", names)
        X, Y, U, Ulab, fnames = ML.build_xy(feats, p, u, horizon, mode)
        print(f"  特征 {len(fnames)} 个,矩阵 {X.shape},耗时 {time.time()-t0:.0f}s", flush=True)
        ctx = EvalContext(p, mode, (horizon,))
        for oos in OOS_YEARS:
            tr = ML.purge_train((years >= oos - TRAIN_YEARS) & (years < oos), horizon)
            te = years == oos
            Xtr, Ytr, dtr = ML.flatten(X, Y, Ulab, np.flatnonzero(tr))
            m = ML.fit_ts(ML.make_model("gbt"), Xtr, Ytr, dtr, purge=horizon + 1)
            sc = ML.predict_panel(m, X[:, te], U[te], p.dates[te], p.codes)
            sd = LEDGER / "scores"; sd.mkdir(exist_ok=True)
            sc.to_parquet(sd / f"AB{label.split('_')[0]}_gbt_{mode}_h{horizon}_{oos}.parquet")
            full = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
            full.loc[sc.index] = sc.to_numpy()
            M = ctx.M[horizon].copy(); M[~te] = False
            ic, _, _, _ = eval_one(full.to_numpy(np.float32), ctx.R[horizon], M, want_layers=False)
            v = ic[np.isfinite(ic)]
            rec = dict(feature_set=label, oos_year=oos, ic=float(v.mean()),
                       t_nw=_nw_t(v, max(5, 2 * horizon)))
            for version in ["V1000", "V100"]:
                bm = ST.benchmark(p, version, exec_mode, date_mask=te)
                r = ST.run_strategy(full, p, version, exec_mode, date_mask=te,
                                    scheme="top", n=20, rebalance=5)
                rec[f"{version}_ann"] = r["ann_return"]
                rec[f"{version}_excess"] = r["ann_return"] - bm["ann_return"]
                rec[f"{version}_sharpe"] = r["sharpe"]
                rec[f"{version}_dd"] = r["max_dd"]
                pa = r.get("phase_ann") or [r["ann_return"]]
                rec[f"{version}_phase_spread"] = float(max(pa) - min(pa))
            rows.append(rec)
            print(f"  OOS {oos}: IC={rec['ic']:+.4f} t_NW={rec['t_nw']:+.1f} | "
                  f"V1000 超额 {rec['V1000_excess']*100:+6.2f}% 夏普 {rec['V1000_sharpe']:.2f} | "
                  f"V100 超额 {rec['V100_excess']*100:+6.2f}%", flush=True)
        del X, Y, U, Ulab, feats
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"ablation_featureset_{mode}_h{horizon}.csv", index=False)
    print(f"\n{'='*110}\n=== 三年 OOS 平均 ===")
    print(df.groupby("feature_set").agg(
        IC=("ic", "mean"), t_NW=("t_nw", "mean"),
        V1000超额=("V1000_excess", "mean"), V1000夏普=("V1000_sharpe", "mean"),
        V1000回撤=("V1000_dd", "mean"),
        V100超额=("V100_excess", "mean"), V100夏普=("V100_sharpe", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
