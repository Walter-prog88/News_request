"""COMP_v5:并入 3 个不依赖市值的基本面交互因子(避开市值分红泄漏)。

对照三组:COMP_v3(22成员) / COMP_v5(25成员) / COMP_v5_capfree(v5 但成员也全部不依赖近似市值)
同时给出各成员对组合的边际贡献(留一法:抽掉该成员后组合 IC 的下降幅度)。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor.fast_eval import EvalContext, evaluate, eval_one
from afactor import factors as FA, screen, combine, backtest as BT, ops, statind as SI

NEW = ["fx_fu_rev_yoy__P__ivol_20", "fx_fu_roe__P__ivol_20",
       "fx_fu_roe__P__id_ret_intraday_m60"]


def main():
    FA.build_all()
    man = json.load(open(LEDGER / "comp_v3_members.json"))
    led = screen.load_ledger("trials_[0-9]*.jsonl")
    led_o = led[led["mode"] == "open"].drop_duplicates("factor", keep="last").set_index("factor")
    members3 = [n for n in man["members"] if n in FA.REGISTRY]
    members5 = members3 + NEW
    signs = dict(man["signs"])
    for n in NEW:
        signs[n] = float(np.sign(led_o.loc[n, "ic_mean"]))
    print(f"COMP_v3 {len(members3)} 成员 -> COMP_v5 {len(members5)} 成员(+{len(NEW)} 个不依赖市值的基本面交互)")

    out = {}
    for mode in ["open", "tail"]:
        p = TailPanel() if mode == "tail" else Panel()
        ctx = EvalContext(p, mode, (1, 5))
        u = p["universe"]
        fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
               .replace([np.inf, -np.inf], np.nan) for n in members5}
        z = combine.prepare(fac, signs, u)
        v3 = combine.combine_equal({n: z[n] for n in members3})
        v5 = combine.combine_equal(z)

        print(f"\n=== {mode} 口径 ===")
        res = {}
        for nm, F in [("COMP_v3", v3), ("COMP_v5", v5)]:
            r = evaluate(f"{nm}__{mode}_r3", F.to_numpy(np.float32), ctx, family="composite",
                         note="第三轮修复后", ledger_path=LEDGER / "trials_r3.jsonl")
            res[nm] = r
            print("  %-9s IC=%.4f t=%5.1f 最差窗=%.4f 单调=%+.2f 多头组=%2d 换手=%.2f h5IC=%.4f" % (
                nm, r["ic_mean"], r["ic_t"], r["worst_window_abs_ic"], r["mono_spearman"],
                r["long_group"], r["long_turnover"], r["h5_ic_mean"]))
        d = res["COMP_v5"]["ic_mean"] - res["COMP_v3"]["ic_mean"]
        print(f"  增量 IC {d:+.4f} ({100*d/res['COMP_v3']['ic_mean']:+.1f}%)")
        print("  分年 IC v3: " + " ".join(f"{k}:{v:+.4f}" for k, v in res["COMP_v3"]["ic_by_year"].items()))
        print("  分年 IC v5: " + " ".join(f"{k}:{v:+.4f}" for k, v in res["COMP_v5"]["ic_by_year"].items()))

        # 标准回测
        bench = BT.benchmark_nav(p, mode)
        b_ann = bench.iloc[-1] ** (242 / len(bench)) - 1
        print(f"  标准回测(缓冲带20/35+5日调仓,基准 {b_ann*100:.2f}%):")
        for nm, F in [("COMP_v3", v3), ("COMP_v5", v5)]:
            rr = BT.run_standard(F, p, sign=1, mode=mode)
            print("    %-9s 年化 %6.2f%%  超额 %+6.2f%%  夏普 %5.2f  回撤 %6.1f%%  换手 %.3f" % (
                nm, rr["ann_return"] * 100, (rr["ann_return"] - b_ann) * 100,
                rr["sharpe"], rr["max_dd"] * 100, rr["avg_turnover"]))
        out[mode] = res

        if mode == "open":
            # 留一法:各成员的边际贡献
            base = res["COMP_v5"]["ic_mean"]
            contrib = []
            for n in members5:
                loo = combine.combine_equal({k: v for k, v in z.items() if k != n})
                ic, _, _, _ = eval_one(loo.to_numpy(np.float32), ctx.R[1], ctx.M[1], want_layers=False)
                v = ic[np.isfinite(ic)]
                contrib.append((n, base - float(v.mean())))
            c = pd.DataFrame(contrib, columns=["member", "边际贡献IC"]).sort_values("边际贡献IC", ascending=False)
            c["is_new"] = c["member"].isin(NEW)
            c.to_csv(LEDGER / "comp_v5_contrib.csv", index=False)
            print("\n  === 成员边际贡献(留一法,抽掉后组合 IC 的下降)===")
            print(c.head(12).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
            print("  新增 3 个成员的贡献排名:",
                  [int(c.reset_index(drop=True).index[c.reset_index(drop=True)["member"] == n][0]) + 1 for n in NEW],
                  f"/ {len(members5)}")
    json.dump({"members": members5, "signs": signs, "new": NEW},
              open(LEDGER / "comp_v5_members.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
