"""清点搜索规模 N —— 让噪音门槛的横轴可复算。

评审第一句话就是"656 是怎么数出来的"。这个脚本把三档 N 全部落到可复现的文件计数上,
产出 research/ledger/search_space_manifest.json,任何人跑一遍都能得到同样的数。

三档的定义与边界:
  N_strategy  策略层回测配置数 —— 在同一把尺子(IR/超额)上真正竞争过的配置。
              这是多重检验的"族"(family)定义:与被检验统计量同单位、同选择步骤。
  N_trials    因子层试验行数 —— append-only 账本 trials_*.jsonl 的总行数。
              绝大多数从未进入回测,用它当分母是保守方向上的单位错误。
  N_eff       有效独立试验数 —— IC 序列相关矩阵测出的有效秩(README §5.2)。
              ⚠ 它描述的是**因子库有多冗余**,不是**搜索程序有多少次独立机会**。
              只能当下界,不能当头条。
"""
from __future__ import annotations
import os, sys, glob, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
csv.field_size_limit(sys.maxsize)
from afactor.config import LEDGER

# 策略层:每一行都是一个被打过分、被比较过的回测配置
STRATEGY_FILES = [
    "baselines.csv", "backtest.csv", "usage_modes.csv",
    "ablation_featureset_open_h5.csv", "ablation_paired.csv",
    "comp_v3_backtest.csv", "dim_composite.csv",
    "midband_composite.csv", "blend_core_satellite.csv",
]


def rows(path) -> int:
    if not path.exists():
        return 0
    with open(path, newline="") as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


def main():
    strat = {}
    for f in STRATEGY_FILES:
        strat[f] = rows(LEDGER / f)
    n_strategy = sum(strat.values())

    trial_files = sorted(glob.glob(str(LEDGER / "trials_*.jsonl")))
    n_trials, facs = 0, set()
    for f in trial_files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            n_trials += 1
            try:
                facs.add(json.loads(line).get("factor"))
            except Exception:
                pass

    man = {
        "N_strategy": {
            "value": n_strategy,
            "definition": "策略层回测配置数(与被检验统计量 IR 同单位)",
            "sources": strat,
            "caveat": "只数了被记录的配置,未计入决定哪些因子有资格进入这些配置的因子层预筛",
        },
        "N_trials": {
            "value": n_trials,
            "definition": "因子层 append-only 账本总行数",
            "n_files": len(trial_files),
            "n_distinct_factors": len(facs),
            "caveat": "绝大多数试验从未进入回测,用作分母是保守方向上的单位错误",
        },
        "N_eff": {
            "value": 11.1,
            "definition": "IC 序列相关矩阵的有效独立试验数(README §5.2)",
            "caveat": "描述因子库冗余度,非搜索程序的独立机会数;只能当下界",
        },
    }
    out = LEDGER / "search_space_manifest.json"
    json.dump(man, open(out, "w"), ensure_ascii=False, indent=2)

    print("=" * 84)
    print("搜索规模清点")
    print("=" * 84)
    print(f"N_strategy = {n_strategy}  (策略层回测配置)")
    for k, v in strat.items():
        print(f"    {k:<40s} {v:>5d}")
    print(f"\nN_trials   = {n_trials}  ({len(trial_files)} 个 trials_*.jsonl,"
          f"{len(facs)} 个不重复因子名)")
    print(f"N_eff      = 11.1  (实测有效独立试验数,仅作下界)")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
