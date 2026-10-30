"""封存样本冒烟测试 —— 只允许运行一次。

前置:research/ledger/final_manifest.json 必须已冻结(因子清单与方向在 2020-2025 内定稿)。
运行后写 smoke_used.lock,此后任何 SealedPanel 调用都会被拒绝。
结果如实记录,不得据此回头修改因子。
"""
from __future__ import annotations
import os, sys, json, warnings, time, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import DAILY, SEALED, SEAL_LOCK, SEAL_START, LEDGER
from afactor.panel import Panel, TailPanel
from afactor.fast_eval import EvalContext, eval_one, _ic_block, _layer_block
from afactor import factors as FA, ops, combine

MANIFEST = LEDGER / "final_manifest.json"


class SealedPanel(Panel):
    """研究集 + 封存集拼接:滚动窗口需要 2025 年历史,但只在 2026 日期上评估。"""

    def __init__(self, tail: bool = False):
        super().__init__(sealed=False)
        self._tail = tail

    def raw(self, name: str) -> pd.DataFrame:
        key = f"__cat__{name}"
        if key not in self._cache:
            a = pd.read_parquet(DAILY / f"{name}.parquet")
            b = pd.read_parquet(SEALED / f"{name}.parquet")
            self._cache[key] = pd.concat([a, b]).sort_index().astype(np.float32)
        return self._cache[key]


class SealedTailPanel(SealedPanel):
    _is_tail = True            # H-1 修正:口径判据改属性,否则 _fu/_nd/_real 全部退回主口径滞后
    _NO_SHIFT = TailPanel._NO_SHIFT

    def raw(self, name):
        if name == "close_px":
            return self.derived("px_1450")
        if name in ("twap_tail8", "amt_tail8"):
            return SealedPanel.raw(self, name)
        if (DAILY / f"pre_{name}.parquet").exists():
            return SealedPanel.raw(self, f"pre_{name}")
        if name in self._NO_SHIFT:
            return SealedPanel.raw(self, name)
        return SealedPanel.raw(self, name).shift(1)

    def derived(self, name):
        if name == "px_1450":
            if "px_1450" not in self._derived:
                self._derived["px_1450"] = (SealedPanel.raw(self, "open_auction_px") *
                                            (1.0 + SealedPanel.raw(self, "pre_ret_intraday")))
            return self._derived["px_1450"]
        return super().derived(name)


def _build_targets(p, man):
    FA.build_all()
    out = {}
    for n in man["single_factors"]:
        out[n] = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) \
            .replace([np.inf, -np.inf], np.nan)
    if man.get("composite_members"):
        fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
               .replace([np.inf, -np.inf], np.nan) for n in man["composite_members"]}
        z = combine.prepare(fac, man["composite_signs"], p["universe"])
        out["COMP_final"] = combine.combine_equal(z)
    return out


def main():
    if SEAL_LOCK.exists():
        print(f"拒绝执行:封存样本已于 {SEAL_LOCK.read_text().strip()[:60]} 使用过一次。")
        sys.exit(1)
    if not MANIFEST.exists():
        print(f"拒绝执行:未找到冻结清单 {MANIFEST}")
        sys.exit(1)
    man = json.load(open(MANIFEST))
    digest = hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()[:16]
    print(f"冻结清单 {MANIFEST.name} (sha256:{digest})")
    print(f"单因子 {len(man['single_factors'])} 个,组合成员 {len(man.get('composite_members', []))} 个\n")

    results = {}
    for mode in man.get("modes", ["open", "tail"]):
        p = SealedTailPanel() if mode == "tail" else SealedPanel()
        targets = _build_targets(p, man)
        ctx = EvalContext(p, mode, (1, 5))
        oos = np.asarray(p.dates >= pd.Timestamp(SEAL_START))
        print(f"=== 封存期({mode} 口径):{int(oos.sum())} 个交易日 "
              f"{p.dates[oos][0].date()} ~ {p.dates[oos][-1].date()} ===")
        print("%-30s %9s %8s %8s %9s %9s %s" % (
            "factor", "样本外IC", "t", "单调", "多头超额", "换手", "分层超额(1..10)"))
        for name, f in targets.items():
            A = f.to_numpy(np.float32)
            M = ctx.M[1].copy(); M[~oos] = False
            ic, q, cov, turn = eval_one(A, ctx.R[1], M, want_layers=True)
            ic_o = ic[oos]
            blk = _ic_block(ic, np.asarray(p.dates.year))
            lay = _layer_block(q[oos] if q is not None else q, turn)
            v = ic_o[np.isfinite(ic_o)]
            t = float(v.mean() / v.std(ddof=1) * np.sqrt(v.size)) if v.size > 20 else np.nan
            rec = {"mode": mode, "oos_ic": float(np.nanmean(ic_o)), "oos_t": t,
                   "n_days": int(v.size), "mono": lay.get("mono_spearman"),
                   "long_excess_ann": lay.get("long_excess_ann"),
                   "long_group": lay.get("long_group"),
                   "turnover": lay.get("long_turnover"), "q_excess": lay.get("q_excess"),
                   "is_ic": man["is_ic"].get(name, {}).get(mode)}
            results[f"{name}__{mode}"] = rec
            print("%-30s %9.4f %8.2f %8.2f %9.3f %9.2f %s" % (
                name, rec["oos_ic"], rec["oos_t"], rec["mono"] or 0,
                rec["long_excess_ann"] or 0, rec["turnover"] or 0, rec["q_excess"]))
        print()

    # ---- 封存期回测:选优 vs 排雷 ----
    from afactor import backtest as BT
    bt_rows = []
    for mode in man.get("modes", ["open", "tail"]):
        p = SealedTailPanel() if mode == "tail" else SealedPanel()
        targets = _build_targets(p, man)
        oos = np.asarray(p.dates >= pd.Timestamp(SEAL_START))
        bench = BT.benchmark_nav(p, mode)
        b = bench[oos[:len(bench)]]
        b_ann = (b.iloc[-1] / b.iloc[0]) ** (242 / len(b)) - 1
        print(f"=== 封存期回测({mode} 口径)  决策池等权基准年化 {b_ann*100:.2f}% ===")
        print("%-24s %-16s %9s %9s %8s %9s %8s" % ("factor", "用法", "年化", "超额", "夏普", "最大回撤", "换手"))
        for name, f in targets.items():
            sgn = int(np.sign(man["is_ic"].get(name, {}).get(mode) or 1)) or 1
            for label, kw in [("选优 top100", dict(scheme="top", top_n=100)),
                              ("选优 top10%", dict(scheme="top", top_n=None, top_frac=0.10)),
                              ("排雷 剔最差30%", dict(scheme="exclude", top_frac=0.30))]:
                r = BT.run_backtest(f, p, sign=sgn, mode=mode, date_mask=oos, **kw)
                if not np.isfinite(r["ann_return"]):
                    continue
                bt_rows.append({"factor": name, "mode": mode, "scheme": label,
                                "ann": r["ann_return"], "excess": r["ann_return"] - b_ann,
                                "sharpe": r["sharpe"], "max_dd": r["max_dd"],
                                "turnover": r["avg_turnover"]})
                print("%-24s %-16s %8.2f%% %8.2f%% %8.2f %8.1f%% %8.2f" % (
                    name if label == "选优 top100" else "", label, r["ann_return"] * 100,
                    (r["ann_return"] - b_ann) * 100, r["sharpe"], r["max_dd"] * 100,
                    r["avg_turnover"]))
        print()

    out = LEDGER / "sealed_results.json"
    pd.DataFrame(bt_rows).to_csv(LEDGER / "sealed_backtest.csv", index=False)
    json.dump({"manifest_sha": digest, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "results": results, "backtest": bt_rows},
              open(out, "w"), ensure_ascii=False, indent=1, default=float)
    SEAL_LOCK.write_text(f"used at {time.strftime('%Y-%m-%d %H:%M:%S')} manifest sha256:{digest}\n")
    print(f"结果已写入 {out};封存锁 {SEAL_LOCK} 已建立,不得重跑。")


if __name__ == "__main__":
    main()
