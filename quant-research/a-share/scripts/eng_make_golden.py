"""回测引擎改造前金标准(前缀 eng,2026-10-10)。

**只能在改造前的 strategy.py 上运行一次。** 生成 tests/golden_engine.json,供
tests/test_engine_regression.py 做逐位回归。文件里记录了生成时 strategy.py /
backtest.py 的 sha256,若检测到 strategy.py 已不是改造前版本且未加 --force,拒绝覆盖。

覆盖的配置(全期 + 2024 date_mask 各一份):
  a  COMP_v5 标准排雷方案 V1000 open10 exclude buffer=(0.20,0.35) rebalance=5
  b  COMP_v5 top n=20 rebalance=5
  c  COMP_v5 top n=50 rebalance=1
  d  COMP_v5 V100 top n=20 rebalance=5
  e  ST.benchmark V1000 open10
  f  V100 tail1450 top n=50 rebalance=5   —— 只为覆盖 lag=0 分支的引擎回归,
     COMP_v5 是主口径打分,用在尾盘口径没有策略含义
  g  COMP_v5 top n=20 rebalance=5 avg_phases=False —— 覆盖单相位分支
  h  ST.benchmark V100 tail1450
  i  backtest.run_standard(COMP_v5) —— backtest.py 的金标准(本次不改 backtest.py,锁住它)

每条记录存 run_strategy 返回的全部标量字段、相位年化、以及**逐日收益全序列**
(JSON 的 float 用 repr 输出,可逐位还原),回归测试按 1e-10 容差比较。
"""
from __future__ import annotations
import os, sys, json, hashlib, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import ROOT, TMP
from afactor.panel import Panel
from afactor import strategy as ST, backtest as BT

OUT = ROOT / "tests" / "golden_engine.json"
PRE_SHA = "56d74351ea1bf1e34cbc9f7f39ab2abda363634083b17421657baed01154e4b0"   # 改造前 strategy.py

CONFIGS = {
    "a_comp_exclude_std": dict(kind="strategy", version="V1000", mode="open10", scheme="exclude",
                               buffer=(0.20, 0.35), rebalance=5),
    "b_comp_top20_r5": dict(kind="strategy", version="V1000", mode="open10", scheme="top", n=20, rebalance=5),
    "c_comp_top50_r1": dict(kind="strategy", version="V1000", mode="open10", scheme="top", n=50, rebalance=1),
    "d_comp_v100_top20_r5": dict(kind="strategy", version="V100", mode="open10", scheme="top", n=20, rebalance=5),
    "e_bench_v1000_open10": dict(kind="benchmark", version="V1000", mode="open10"),
    "f_v100_tail_top50_r5": dict(kind="strategy", version="V100", mode="tail1450", scheme="top", n=50, rebalance=5),
    "g_comp_top20_r5_single": dict(kind="strategy", version="V1000", mode="open10", scheme="top", n=20,
                                   rebalance=5, avg_phases=False),
    "h_bench_v100_tail1450": dict(kind="benchmark", version="V100", mode="tail1450"),
    "i_backtest_run_standard": dict(kind="backtest_standard"),
}
WINDOWS = {"full": None, "y2024": 2024}


def sha_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_config(cfg: dict, score: pd.DataFrame, p: Panel, mask):
    kind = cfg["kind"]
    if kind == "strategy":
        kw = {k: v for k, v in cfg.items() if k not in ("kind", "version", "mode")}
        return ST.run_strategy(score, p, cfg["version"], cfg["mode"], date_mask=mask, **kw)
    if kind == "benchmark":
        return ST.benchmark(p, cfg["version"], cfg["mode"], date_mask=mask)
    if kind == "backtest_standard":
        return BT.run_standard(score, p, date_mask=mask)
    raise ValueError(kind)


def serialize(res: dict) -> dict:
    out = {}
    for k, v in res.items():
        if k in ("nav", "rets"):
            continue
        if isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, list):
            out[k] = [float(x) for x in v]
    r = res["rets"]
    out["rets"] = [float(x) for x in r.to_numpy(np.float64)]
    out["rets_index_first"] = str(r.index[0].date())
    out["rets_index_last"] = str(r.index[-1].date())
    out["rets_sha256"] = hashlib.sha256(r.to_numpy(np.float64).tobytes()).hexdigest()
    return out


def main():
    cur = sha_file(ROOT / "src" / "afactor" / "strategy.py")
    if OUT.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{OUT} 已存在;金标准只生成一次。确需重建请加 --force")
    if cur != PRE_SHA and "--force" not in sys.argv:
        raise SystemExit("strategy.py 已不是改造前版本,拒绝生成金标准")
    p = Panel()
    score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    years = p.dates.year
    gold = dict(meta=dict(
        created="2026-10-10", note="改造前金标准;逐日收益可逐位还原。COMP_v5 为信号样本内打分,仅用于引擎回归。",
        strategy_sha256=cur, backtest_sha256=sha_file(ROOT / "src" / "afactor" / "backtest.py"),
        score_file="cache/_tmp/comp_v5_score_open10.parquet",
        score_sha256=sha_file(TMP / "comp_v5_score_open10.parquet"),
        panel_first=str(p.dates[0].date()), panel_last=str(p.dates[-1].date()), n_codes=int(len(p.codes))),
        configs={k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in c.items()}
                 for k, c in CONFIGS.items()},
        results={})
    for name, cfg in CONFIGS.items():
        for wname, yr in WINDOWS.items():
            mask = None if yr is None else np.asarray(years == yr)
            t0 = time.time()
            res = run_config(cfg, score, p, mask)
            gold["results"][f"{name}|{wname}"] = serialize(res)
            print(f"{name:28s} {wname:6s} ann={res['ann_return']*100:8.4f}% sharpe={res.get('sharpe', np.nan):.4f} "
                  f"turn={res.get('turnover', res.get('avg_turnover', np.nan))} {time.time()-t0:.1f}s", flush=True)
    OUT.write_text(json.dumps(gold, ensure_ascii=False, indent=1))
    print("written", OUT)


if __name__ == "__main__":
    main()
