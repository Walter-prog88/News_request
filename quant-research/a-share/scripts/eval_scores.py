"""从存盘的打分面板重算策略层结果(不重训模型)。

用途:改策略配置(相位平均/持仓数/用法/口径/择时)后几分钟出结果,
而不是重训 1.5 小时。打分面板由 run_ml_models / run_featureset_ablation 写入
`research/ledger/scores/{集}_{模型}_{口径}_h{horizon}_{年}.parquet`。
"""
from __future__ import annotations
import os, sys, glob, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor import strategy as ST

pd.set_option("display.width", 240)
SD = LEDGER / "scores"

PLANS = {
    "Top20周频": dict(scheme="top", n=20, rebalance=5),
    "Top10周频": dict(scheme="top", n=10, rebalance=5),
    "Top50周频": dict(scheme="top", n=50, rebalance=5),
    "排雷缓冲20/35": dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5),
}


def main():
    mode = os.environ.get("EV_MODE", "open")
    exec_mode = {"open": "open10", "tail": "tail1450"}[mode]
    horizon = os.environ.get("EV_HORIZON", "5")
    only = os.environ.get("EV_PLANS")
    plans = {k: v for k, v in PLANS.items() if not only or k in only.split(",")}
    p = TailPanel() if mode == "tail" else Panel()
    years = np.asarray(p.dates.year)
    files = sorted(glob.glob(str(SD / f"*_{mode}_h{horizon}_*.parquet")))
    print(f"找到 {len(files)} 个打分面板 | mode={mode} h={horizon} | 用法 {list(plans)}", flush=True)

    bench_cache = {}
    rows = []
    for f in files:
        stem = os.path.basename(f)[:-8]
        parts = stem.split("_")
        fs, kind, yr = parts[0], parts[1], int(parts[-1])
        sc = pd.read_parquet(f)
        full = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
        full.loc[sc.index] = sc.to_numpy()
        dm = years == yr
        for ver in ["V1000", "V100"]:
            key = (ver, yr)
            if key not in bench_cache:
                bench_cache[key] = ST.benchmark(p, ver, exec_mode, date_mask=dm)["ann_return"]
            bm = bench_cache[key]
            for plan, kw in plans.items():
                r = ST.run_strategy(full, p, ver, exec_mode, date_mask=dm, **kw)
                if not np.isfinite(r.get("ann_return", np.nan)):
                    continue
                pa = r.get("phase_ann") or [r["ann_return"]]
                rows.append(dict(集=fs, 模型=kind, 年=yr, 版本=ver, 用法=plan,
                                 基准=bm, 年化=r["ann_return"], 超额=r["ann_return"] - bm,
                                 夏普=r["sharpe"], 回撤=r["max_dd"], 换手=r["turnover"],
                                 仓位=r.get("exposure"), 相位极差=float(max(pa) - min(pa))))
        print(f"  {stem} 完成", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(LEDGER / f"score_eval_{mode}_h{horizon}.csv", index=False)
    print(f"\n=== 三年 OOS 平均(相位平均后)| {exec_mode} h={horizon} ===")
    print(d.groupby(["集", "模型", "版本", "用法"]).agg(
        超额=("超额", "mean"), 夏普=("夏普", "mean"), 回撤=("回撤", "mean"),
        换手=("换手", "mean"), 仓位=("仓位", "mean"),
        相位极差=("相位极差", "mean")).round(4).to_string())
    print("\n=== 逐年明细 ===")
    print(d.sort_values(["集", "模型", "版本", "用法", "年"]).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
