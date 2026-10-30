"""实验三:排雷工程化。

1. 雷区投票因子:每个信息簇代表因子各自标出决策池最差 10%,一只股票收到的
   票数(0..19)本身作为因子检验(负向:票越多未来越差)。
2. 缓冲带回测:排雷(剔最差10%进入/回到20%退出)vs 无缓冲,对比换手与超额。
3. walk-forward:2022-2025 每年用前两年选最优排雷比例,应用于当年,
   检验参数选择的稳定性(全程在研究期内,不触碰封存样本)。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
from scipy.stats import rankdata

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor.fast_eval import EvalContext, evaluate
from afactor import factors as FA, screen, backtest as BT


def build_votes(p: Panel, reps: dict[str, float], frac: float = 0.10) -> pd.DataFrame:
    """reps: {factor_name: ic_sign}。票 = 因子(方向对齐后)分位 <= frac。"""
    M = p["universe"].to_numpy(bool)
    T, N = M.shape
    votes = np.zeros((T, N), np.float32)
    counted = np.zeros((T, N), np.float32)
    for name, sgn in reps.items():
        f = FA.REGISTRY[name][1](p).reindex(index=p.dates, columns=p.codes) \
            .replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
        for t in range(T):
            m = M[t] & np.isfinite(f[t])
            n = int(m.sum())
            if n < 200:
                continue
            idx = np.flatnonzero(m)
            q = rankdata(f[t, idx] * sgn) / n
            votes[t, idx[q <= frac]] += 1.0
            counted[t, idx] += 1.0
        print(f"  投票完成: {name}", flush=True)
    out = np.where(counted >= max(len(reps) * 0.6, 3), votes, np.nan)
    return pd.DataFrame(out, index=p.dates, columns=p.codes)


def main():
    p = Panel()
    FA.build_all()
    ct = pd.read_csv(LEDGER / "clusters_open.csv")
    led = screen.load_ledger("trials_[0-9]*.jsonl")
    led = led[led["mode"] == "open"].set_index("factor")
    reps = {}
    for _, r in ct.iterrows():
        n = r["rep"]
        if n in FA.REGISTRY and n in led.index:
            reps[n] = float(np.sign(led.loc[n, "ic_mean"]))
    print(f"雷区投票:{len(reps)} 个簇代表")

    # ---- 1. 投票因子 ----
    votes = build_votes(p, reps, frac=0.10)
    ctx = EvalContext(p, "open", (1, 5))
    r = evaluate("VOTES_worst10", (-votes).to_numpy(np.float32), ctx, family="engineering",
                 note="雷区投票(负向:票多=雷)", ledger_path=LEDGER / "trials_engineering.jsonl")
    print("\n投票因子: IC=%.4f t=%.1f 单调=%s 分层=%s" % (
        r["ic_mean"], r["ic_t"], r.get("mono_spearman"), r.get("q_excess", "离散值分层不适用")))
    votes.to_pickle(LEDGER / "votes_panel.pkl")

    # ---- 2. 排雷方式对比:连续因子排雷 vs 投票排雷 vs 缓冲带 ----
    comp_members = json.load(open(LEDGER / "final_manifest.json"))["composite_members"]
    comp_signs = json.load(open(LEDGER / "final_manifest.json"))["composite_signs"]
    from afactor import combine
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in comp_members if n in FA.REGISTRY}
    z = combine.prepare(fac, comp_signs, p["universe"])
    comp = combine.combine_equal(z)

    bench = BT.benchmark_nav(p, "open")
    b_ann = bench.iloc[-1] ** (242 / len(bench)) - 1
    rows = []
    print(f"\n=== 排雷方案对比(研究期,基准年化 {b_ann*100:.2f}%)===")
    print("%-34s %9s %9s %8s %9s %8s" % ("方案", "年化", "超额", "夏普", "最大回撤", "日换手"))
    plans = [
        ("组合因子 剔最差10%(无缓冲)", dict(F=comp, scheme="exclude", top_frac=0.10)),
        ("组合因子 剔最差30%(无缓冲)", dict(F=comp, scheme="exclude", top_frac=0.30)),
        ("组合因子 缓冲带10/20", dict(F=comp, buffer=(0.10, 0.20))),
        ("组合因子 缓冲带20/35", dict(F=comp, buffer=(0.20, 0.35))),
        ("投票>=2票剔除", dict(F=-votes, scheme="exclude", top_frac=None)),
        ("投票>=3票剔除", dict(F=-votes, scheme="exclude", top_frac=None)),
    ]
    for label, kw in plans:
        if label.startswith("投票"):
            k = 2 if ">=2" in label else 3
            # 被 k 票及以上标雷者因子置 NaN(不可入选);票数缺失(计票因子不足)不视为雷
            fsel = comp.where((votes < k) | votes.isna())
            r = BT.run_backtest(fsel, p, sign=1, scheme="exclude", top_frac=0.0, mode="open")
        else:
            F = kw.pop("F")
            r = BT.run_backtest(F, p, sign=1, mode="open", **kw)
        rows.append({"plan": label, **{k: v for k, v in r.items() if k != "nav"}})
        print("%-34s %8.2f%% %8.2f%% %8.2f %8.1f%% %8.3f" % (
            label, r["ann_return"] * 100, (r["ann_return"] - b_ann) * 100,
            r["sharpe"], r["max_dd"] * 100, r["avg_turnover"]))
    pd.DataFrame(rows).to_csv(LEDGER / "mine_engineering.csv", index=False)

    # ---- 3. walk-forward 排雷比例 ----
    print("\n=== walk-forward:每年用前两年选最优排雷比例 ===")
    years = sorted(set(p.dates.year))
    fracs = [0.10, 0.20, 0.30, 0.40]
    px = p.raw("twap_open10")
    ret1 = (px.shift(-2) / px.shift(-1) - 1.0)
    M = p["universe"]
    wf = []
    for y in years:
        if y - 2 < years[0]:
            continue
        train = (p.dates.year >= y - 2) & (p.dates.year <= y - 1)
        test = p.dates.year == y
        best, best_v = None, -9e9
        perf = {}
        for fr in fracs:
            for mask, key in [(train, "train"), (test, "test")]:
                q = comp.rank(axis=1, pct=True)
                keep = (q > fr) & M
                r_ = ret1.where(keep)[mask].mean(axis=1) - ret1.where(M)[mask].mean(axis=1)
                perf[(fr, key)] = float(r_.mean() * 242)
            if perf[(fr, "train")] > best_v:
                best_v, best = perf[(fr, "train")], fr
        wf.append({"year": y, "chosen": best,
                   "test_excess_chosen": perf[(best, "test")],
                   **{f"test_{fr}": perf[(fr, 'test')] for fr in fracs}})
        print(f"  {y}: 选中剔除{int(best*100)}% -> 当年超额 {perf[(best,'test')]*100:+.2f}%  "
              f"(各比例当年: " + " ".join(f"{int(fr*100)}%:{perf[(fr,'test')]*100:+.1f}" for fr in fracs) + ")")
    pd.DataFrame(wf).to_csv(LEDGER / "walkforward_frac.csv", index=False)


if __name__ == "__main__":
    main()
