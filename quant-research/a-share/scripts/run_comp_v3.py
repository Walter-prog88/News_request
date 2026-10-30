"""COMP_v3:把第二轮发现的新维度并入组合,双口径检验 + 标准回测配置对比。

新成员(每个新维度一个代表,方向取自 open 口径账本):
  mi_semibeta_mp_m20                半贝塔:市场涨我跌
  cv_ret_intraday_by_rvol_20        高波动日的日内收益(条件反转)
  si_gd_amt_vol_20                  统计行业内成交额波动
  cx_mom_overnight_20__P__amt_z_20  隔夜动量 + 缩量

诚实标注:新成员由 2020-2025 全样本筛出,本脚本的对比不构成样本外证据;
真正的检验只能等 2026-09 之后的新数据。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor.fast_eval import EvalContext, evaluate
from afactor import factors as FA, screen, combine, backtest as BT

NEW_MEMBERS = ["mi_semibeta_mp_m20", "cv_ret_intraday_by_rvol_20",
               "si_gd_amt_vol_20", "cx_mom_overnight_20__P__amt_z_20"]


def main():
    FA.build_all()
    man = json.load(open(LEDGER / "final_manifest.json"))
    led = screen.load_ledger("trials_[0-9]*.jsonl")
    led_o = led[led["mode"] == "open"].set_index("factor")
    members = [n for n in man["composite_members"] if n in FA.REGISTRY] + NEW_MEMBERS
    signs = dict(man["composite_signs"])
    for n in NEW_MEMBERS:
        signs[n] = float(np.sign(led_o.loc[n, "ic_mean"]))
    print(f"COMP_v3 成员 {len(members)} 个(原 {len(man['composite_members'])} + 新维度 {len(NEW_MEMBERS)})")

    out = {}
    for mode in ["open", "tail"]:
        p = TailPanel() if mode == "tail" else Panel()
        ctx = EvalContext(p, mode, (1, 5))
        fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
               .replace([np.inf, -np.inf], np.nan) for n in members}
        z = combine.prepare(fac, signs, p["universe"])
        v3 = combine.combine_equal(z)
        comp = combine.combine_equal({n: z[n] for n in man["composite_members"] if n in z})

        r = evaluate(f"COMP_v3__{mode}", v3.to_numpy(np.float32), ctx, family="composite",
                     note=f"COMP_final + 4个第二轮新维度", ledger_path=LEDGER / "trials_engineering.jsonl")
        base = led[(led["factor"] == f"COMP_indep__{mode}")]
        b_ic = float(base["ic_mean"].iloc[0]) if len(base) else np.nan
        print(f"\n[{mode}] COMP_v3  IC={r['ic_mean']:.4f} t={r['ic_t']:.1f} "
              f"最差窗={r['worst_window_abs_ic']:.4f} 单调={r['mono_spearman']:.2f} "
              f"换手={r['long_turnover']:.2f}  (对照 COMP_final IC={b_ic:.4f})")
        print(f"  分年 IC: " + " ".join(f"{k}:{v:+.4f}" for k, v in r["ic_by_year"].items()))

        # 标准回测配置对比
        bench = BT.benchmark_nav(p, mode)
        b_ann = bench.iloc[-1] ** (242 / len(bench)) - 1
        print(f"  标准回测(缓冲带20/35 + 5日调仓,基准年化 {b_ann*100:.2f}%):")
        rows = []
        for nm, f in [("COMP_final", comp), ("COMP_v3", v3)]:
            rr = BT.run_standard(f, p, sign=1, mode=mode)
            rows.append({"factor": nm, "mode": mode, **{k: v for k, v in rr.items() if k != "nav"}})
            print("    %-12s 年化 %6.2f%%  超额 %+6.2f%%  夏普 %5.2f  回撤 %6.1f%%  日换手 %.3f" % (
                nm, rr["ann_return"] * 100, (rr["ann_return"] - b_ann) * 100,
                rr["sharpe"], rr["max_dd"] * 100, rr["avg_turnover"]))
        out[mode] = rows
    pd.DataFrame([x for v in out.values() for x in v]).to_csv(LEDGER / "comp_v3_backtest.csv", index=False)
    json.dump({"members": members, "signs": signs, "new_members": NEW_MEMBERS},
              open(LEDGER / "comp_v3_members.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n已写 comp_v3_members.json / comp_v3_backtest.csv")


if __name__ == "__main__":
    main()
