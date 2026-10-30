"""开关的正确用法检验:它是**股票池筛选**,不是因子。

`run_switch_factors.py` 的结果里,"硬门限"构造 sw_F__T__S = cs_rank(F).where(S 前 70%)
的 IC 比基础因子高 12~14%。但这个提升是**逻辑上必然**的,不是新信息:

  在保留的 70% 股票里,cs_rank(F).where(...) 是 F 的单调变换,
  所以它在这个子集上的 IC **恒等于** F 在这个子集上的 IC。
  换句话说,"硬门限因子"根本不是一个新因子,它就是**老因子换了个股票池**。
  与基础因子的 IC 序列相关 0.967~0.972 也印证了这一点。

所以正确的检验不是"它的 IC 高不高",而是"**当作池筛选用,回测能不能赚更多**",
而且基准必须仍用**全池**等权 —— 否则就是换基准作弊(缩小池子再和缩小的基准比)。

对照组(全部用 COMP_v5 + 排雷缓冲 20/35 + 5 日调仓,基准一律全池等权):
  1. 全池(基线)
  2. 剔除 idio_share 最低 30%   —— 开关扫描排名第 2~3
  3. 剔除 amount 最低 30%        —— 开关扫描排名第 1
  4. 剔除随机 30% × 5 次         —— **零假设**:随机缩池会不会也"变好"

第 4 组是关键:缩池本身会改变风格暴露(剔掉的往往是低波动大盘股),
不设随机对照就无法区分"开关有信息"和"缩池的机械效果"。

产出:research/ledger/switch_pool.csv
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor import factors as FA, ops, combine, backtest as BT

DROP = 0.30          # 剔除比例
N_RANDOM = 5
SEED = 20260825


def _comp_v5(p):
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members = [n for n in man["members"] if n in FA.REGISTRY]
    signs = {n: float(man["signs"][n]) for n in members}
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) for n in members}
    return combine.combine_equal(combine.prepare(fac, signs, p["universe"]))


def _keep_by(p, field, sign=1.0):
    """保留 field(方向调正后)最高的 1-DROP;返回布尔面板。"""
    r = ops.cs_rank(p.raw(field)) * sign          # [-0.5, 0.5]
    return r > (0.5 - (1 - DROP))


def _keep_random(p, seed):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.standard_normal((len(p.dates), len(p.codes))),
                     index=p.dates, columns=p.codes)
    return ops.cs_rank(x.where(p["universe"])) > (0.5 - (1 - DROP))


def _stats(res, bench, label, n_hold):
    nav = res["nav"]
    b = bench.reindex(nav.index).ffill()
    yrs = len(nav) / 242.0
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    bann = b.iloc[-1] ** (1 / yrs) - 1
    r = nav.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(242) if r.std() > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    return {"方案": label, "年化": ann, "基准年化": bann, "超额": ann - bann,
            "夏普": sharpe, "回撤": dd, "日换手": res.get("avg_turnover"), "成本bps": res.get("avg_cost_bps"),
            "平均持仓": res.get("avg_holdings"), "池内只数": n_hold}


def main():
    p = Panel(); FA.build_all()
    print("构造 COMP_v5 ...", flush=True)
    comp = _comp_v5(p)
    u = p["universe"]
    bench = BT.benchmark_nav(p, "open")

    cases = [("全池(基线)", None),
             ("剔除 idio_share 最低30%", _keep_by(p, "idio_share", +1.0)),
             ("剔除 mkt_corr 最高30%", _keep_by(p, "mkt_corr", -1.0)),
             ("剔除 amount 最低30%", _keep_by(p, "amount", +1.0))]
    cases += [(f"剔除随机30% #{i}", _keep_random(p, SEED + i)) for i in range(N_RANDOM)]

    rows = []
    for label, keep in cases:
        F = comp if keep is None else comp.where(keep.reindex_like(comp).fillna(False))
        n_hold = float((u & F.notna()).sum(axis=1).mean())
        res = BT.run_standard(F, p, sign=1, mode="open")
        rows.append(_stats(res, bench, label, n_hold))
        print(f"  {label:26s} 超额 {rows[-1]['超额']*100:+6.2f}%  "
              f"夏普 {rows[-1]['夏普']:.2f}  换手 {rows[-1]['日换手']:.3f}  "
              f"池内 {n_hold:.0f} 只", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "switch_pool.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("=== COMP_v5 + 排雷缓冲20/35 + 5日调仓;基准一律全池等权 ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    base = df[df.方案.str.startswith("全池")].iloc[0]
    rnd = df[df.方案.str.startswith("剔除随机")]
    print(f"\n基线超额 {base.超额*100:+.2f}% / 夏普 {base.夏普:.2f} / 换手 {base.日换手:.3f}")
    print(f"随机缩池 超额 {rnd.超额.mean()*100:+.2f}% ± {rnd.超额.std()*100:.2f}pp "
          f"(区间 {rnd.超额.min()*100:+.2f}% ~ {rnd.超额.max()*100:+.2f}%)")
    print("\n--- 判据:开关缩池是否显著优于随机缩池 ---")
    for _, r in df[df.方案.str.startswith("剔除") & ~df.方案.str.startswith("剔除随机")].iterrows():
        z = (r.超额 - rnd.超额.mean()) / rnd.超额.std() if rnd.超额.std() > 0 else np.nan
        print(f"  {r.方案:26s} 超额 {r.超额*100:+6.2f}%  vs 随机均值 "
              f"{rnd.超额.mean()*100:+6.2f}%  差 {(r.超额-rnd.超额.mean())*100:+5.2f}pp  "
              f"(= {z:+.1f} 个随机标准差)")
    print("\n产出:switch_pool.csv")


if __name__ == "__main__":
    main()
