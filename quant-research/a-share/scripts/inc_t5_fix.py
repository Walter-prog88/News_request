"""T5 审查修复(前缀 inc_t5;审查发现 P1-1 / P1-2 / P2-1 ~ P2-4（原审查返回的四项 P2） 的补充诊断,全部在 <= 2025-12-31 数据上,复用 cache/_tmp/inc_t5_runs 的既有回测)。

阶段(日志 logs/inc_t5_run.log;新账本 research/ledger/inc_t5_fix_*.csv,不改动原判定账本):
  runs_min_depth  B-MIN 补跑"深度对齐"配置:LIQ@45/60、5 个风格篮子@20/35、S4/EW5/CORE5/EW6/CORE6@20/35
                  (原预注册 B-MIN 只跑 45/60;这是看到审查意见后的补充诊断,不进 P1-P7 与判定)
  diag            深度参照回归(B-LH / B-MIN)、E3 分段诊断、B-MIN 去掉开局 5 日、含成本稀释对照、开发窗变体 IR
  devs            偏离账本:重新标注"登记晚于首次回测"的条目,并登记本次补充诊断
用法: python3 scripts/inc_t5_fix.py runs_min_depth diag devs
"""
from __future__ import annotations

import pickle
import sys
import time

from inc_t5_common import *      # noqa: F401,F403
from inc_t5_common import ROOT, SC, RUNS, TMP, LEDGER, WIN, LE, log, log_dev, TD, DEVS

import numpy as np
import pandas as pd

from afactor import inc_t5_stats as S5

STYLE5 = ["LIQ", "LOWVOL", "VALUE", "DIV", "QUAL"]
STYLE6 = STYLE5 + ["REV"]
DEPTHS = {"45_60": (0.45, 0.60), "20_35": (0.20, 0.35)}
WINS = ("full", "dev", "test")


def fname(name: str) -> str:
    return name.replace("|", "__") + ".pkl"


def load(track: str, name: str) -> dict:
    with open(RUNS / track / fname(name), "rb") as f:
        result = pickle.load(f)
    if pd.to_datetime(result["rets"].index).max() > pd.Timestamp("2025-12-31"):
        raise ValueError(f"{track}/{name}: cached returns exceed frozen cutoff")
    return result


def b1_of(track: str) -> pd.Series:
    result = pd.read_pickle(RUNS / f"{track}_b1.pkl")
    if pd.to_datetime(result.index).max() > pd.Timestamp("2025-12-31"):
        raise ValueError(f"{track}: benchmark exceeds frozen cutoff")
    return result


# ================================================================ B-MIN 深度对齐补跑
def stage_runs_min_depth():
    from inc_t5_runs import MinCtx
    t0 = time.time()
    c = MinCtx()
    log(f"[fix_runs_min] Panel 载入 {time.time() - t0:.0f}s")
    d = RUNS / "min"

    def save(name, res):
        out = {k: res[k] for k in ("rets", "turnover", "cost_bps", "exposure", "holdings") if k in res}
        with open(d / fname(name), "wb") as f:
            pickle.dump(out, f)

    Wb = {}
    # LIQ@45/60:LIQ 本身放深到风格篮子的深度(深度参照)
    Wb[("LIQ", "45_60")] = c.W("LIQ", "45_60")
    save("S|LIQ|45_60", c.run(Wb[("LIQ", "45_60")]))
    log(f"[fix_runs_min] S|LIQ|45_60 {time.time() - t0:.0f}s")
    Wb[("LIQ", "20_35")] = c.W("LIQ", "20_35")
    for b in STYLE6[1:]:
        Wb[(b, "20_35")] = c.W(b, "20_35")
        save(f"S|{b}|20_35", c.run(Wb[(b, "20_35")]))
        n = (Wb[(b, "20_35")] > 0).sum(axis=1)
        log(f"[fix_runs_min] S|{b}|20_35 平均持仓 {n[n > 0].mean():.0f} {time.time() - t0:.0f}s")
    combos = {"S4": (STYLE5[1:], "EW"), "EW5": (STYLE5, "EW"), "CORE5": (STYLE5, "CORE"),
              "EW6": (STYLE6, "EW"), "CORE6": (STYLE6, "CORE")}
    for nm, (mem, sch) in combos.items():
        Ws = [Wb[(m, "20_35")] for m in mem]
        a = np.full(len(mem), 1.0 / len(mem)) if sch == "EW" else S5.core_weights(len(mem), mem.index("LIQ"), 0.5)
        save(f"C|{nm}|20_35", c.run(S5.combine_weights(Ws, a)))
        log(f"[fix_runs_min] C|{nm}|20_35 {time.time() - t0:.0f}s")
    log(f"[fix_runs_min] 完成 {time.time() - t0:.0f}s")


# ================================================================ 统计小工具
def excess(track: str, name: str) -> pd.Series:
    r = load(track, name)["rets"]
    b = b1_of(track)
    return (r - b.reindex(r.index)).dropna()


def reg(y: pd.Series, x: pd.Series, a: str, b: str, q: int = 5) -> dict:
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).loc[a:b].dropna()
    r = S5.hac_regress(df["y"].to_numpy(), df["x"].to_numpy(), q)
    return dict(n=r["n"], a_ann_pp=r["a"] * TD * 100, t_a=r["t_a"], b=r["b"], r2=r["r2"],
                rho=float(np.corrcoef(df["y"], df["x"])[0, 1]))


def ir_block(track: str, name: str, a: str, b: str, drop_first: int = 0) -> dict:
    rs = LE.slice_eval(load(track, name)["rets"], a, b)
    rb = LE.slice_eval(b1_of(track), a, b)
    rs, rb = rs.align(rb, join="inner")
    if drop_first:
        rs, rb = rs.iloc[drop_first:], rb.iloc[drop_first:]
    rel = LE.rel_block(rs, rb)
    return dict(excess_arith_pp=rel["excess_arith"] * 100, te_pct=rel["te"] * 100, ir=rel["ir"], nw_t5=rel["nw_t5"], n_days=len(rs))


# ================================================================ 诊断
def diag_depth_ref() -> pd.DataFrame:
    """P1-1:生成检验的深度混淆。y = 各篮子/组合对 LIQ@20/35 的回归;LIQ@45/60 是深度参照(它本身就是排雷)。"""
    rows = []
    cfgs = {
        "lh": ["S|LIQ|45_60", "C|S4|45_60", "C|S4|20_35", "C|S5|45_60", "C|S5|20_35"]
              + [f"S|{b}|{d}" for b in ("LOWVOL", "VALUE", "DIV", "QUAL", "REV") for d in ("45_60", "20_35")],
        "min": ["S|LIQ|45_60", "C|S4|45_60", "C|S4|20_35"]
               + [f"S|{b}|{d}" for b in ("LOWVOL", "VALUE", "DIV", "QUAL", "REV") for d in ("45_60", "20_35")],
    }
    for track, lst in cfgs.items():
        x = excess(track, "S|LIQ|20_35")
        for n in lst:
            try:
                y = excess(track, n)
            except FileNotFoundError:
                raise FileNotFoundError(f"Required review cache missing: {track}/{n}")
            for w in WINS:
                a, b = WIN[track][w]
                r = reg(y, x, a, b, 5)
                rows.append(dict(track=track, y=n, window=w, **r))
    return pd.DataFrame(rows)


def diag_depth_ir() -> pd.DataFrame:
    """深度对齐后的 IR(含 B-MIN 补跑):组合 vs 排雷,三窗;以及开发窗哪些变体高于排雷。"""
    rows = []
    for track, cfgs in (("lh", [f"C|{s}|{d}" for s in ("EW5", "CORE5", "ERC5", "EW6", "CORE6", "ERC6", "S4") for d in ("45_60", "20_35")]),
                        ("min", [f"C|{s}|{d}" for s in ("EW5", "CORE5", "EW6", "CORE6", "S4") for d in ("45_60", "20_35")]
                         + ["C|ERC5|45_60", "C|ERC6|45_60"])):
        for n in ["S|LIQ|20_35"] + cfgs:
            try:
                r = {w: ir_block(track, n, *WIN[track][w]) for w in WINS}
            except FileNotFoundError:
                raise FileNotFoundError(f"Required review cache missing: {track}/{n}")
            rows.append(dict(track=track, config=n, **{f"{k}_{w}": v for w in WINS for k, v in r[w].items() if k != "n_days"}))
    df = pd.DataFrame(rows)
    liq = df[df.config == "S|LIQ|20_35"].set_index("track")
    for w in WINS:
        df[f"d_ir_{w}"] = df.apply(lambda r: r[f"ir_{w}"] - liq.loc[r["track"], f"ir_{w}"], axis=1)
    df["dev_higher_than_liq"] = df["ir_dev"] > df.apply(lambda r: liq.loc[r["track"], "ir_dev"], axis=1)
    return df


def diag_e3() -> pd.DataFrame:
    """P1-2 / P2-1:E3(全池等权付成本 − B1 + 成本)分段与去掉开局日的诊断。几何超额 + 年成本,单位 pp/年。"""
    rows = []
    for track in ("lh", "a", "min"):
        fp = load(track, "FP")
        r = fp["rets"]
        cost = fp["rec_daily"]["cost_total"].reindex(r.index).fillna(0.0)
        b = b1_of(track).reindex(r.index)
        full = WIN[track]["full"]
        segs = {"full": (full[0], full[1])}
        if track == "a":
            segs["2006-2016"] = ("2006-01-04", "2016-12-30")
            segs["2017-2025"] = ("2017-01-03", "2025-12-31")
        if track == "min":
            segs["2021_excl_first5"] = None
        for sname, rng in segs.items():
            drop = 0
            if sname == "2021_excl_first5":
                rng, drop = full, 5
            a, bb = rng
            s, bm, cm = r.loc[a:bb], b.loc[a:bb], cost.loc[a:bb]
            if drop:
                s, bm, cm = s.iloc[drop:], bm.iloc[drop:], cm.iloc[drop:]
            ex = LE.ann_geo(s) - LE.ann_geo(bm)
            cst = float(cm.mean() * TD)
            rows.append(dict(track=track, segment=sname, drop_first_days=drop, n_days=len(s), geo_excess_pp=ex * 100,
                             ann_cost_pp=cst * 100, e3_pp=(ex + cst) * 100, gate_abs_le_0p3=bool(abs(ex + cst) <= 0.003)))
        if track == "min":
            # 开局若干天的 B1 / 全池等权
            d = (r + cost - b).dropna()
            rows.append(dict(track=track, segment="first2days_2021-01-05_06_sum", drop_first_days=0, n_days=2,
                             geo_excess_pp=float(d.loc["2021-01-05":"2021-01-06"].sum() * 100), ann_cost_pp=np.nan, e3_pp=np.nan,
                             gate_abs_le_0p3=True))
    return pd.DataFrame(rows)


def diag_min_x5() -> pd.DataFrame:
    """P2-1:B-MIN 去掉评价期开局 5 个信号日(建仓期 exposure 0.2->1.0,而 B1 满仓)后的超额 / IR / S4 截距。"""
    cfgs = ["S|LIQ|20_35", "S|LOWVOL|45_60", "S|VALUE|45_60", "S|DIV|45_60", "S|QUAL|45_60", "S|REV|45_60", "C|EW5|45_60", "C|CORE5|45_60",
            "C|ERC5|45_60", "C|EW6|45_60", "C|S4|45_60"]
    rows = []
    for n in cfgs:
        for drop in (0, 5):
            for w in WINS:
                a, b = WIN["min"][w]
                rows.append(dict(config=n, drop_first_days=drop, window=w, **ir_block("min", n, a, b, drop if w in ("full", "dev") else 0)))
    df = pd.DataFrame(rows)
    # S4 对 LIQ 的截距(去掉开局 5 日)
    gen = []
    x = excess("min", "S|LIQ|20_35")
    y = excess("min", "C|S4|45_60")
    for drop in (0, 5):
        for w in WINS:
            a, b = WIN["min"][w]
            xx, yy = x, y
            if drop and w in ("full", "dev"):
                start = x.loc[a:b].index[drop]
                a = str(start.date())
            gen.append(dict(config="S4_on_LIQ", drop_first_days=drop, window=w, **reg(yy, xx, a, b, 5)))
    return df, pd.DataFrame(gen)


def diag_dilution_cost() -> pd.DataFrame:
    """P2-2:稀释对照改为两条成本后收益序列的诊断混合 alpha*排雷 + (1-alpha)*全池等权（非合并权重重跑，不估计净额成交节省）(两腿都是引擎付成本后的收益),对比原不计成本的合成。"""
    rows = []
    for track in ("lh", "min", "a"):
        liq = load(track, "S|LIQ|20_35")["rets"]
        fp = load(track, "FP")["rets"].reindex(liq.index)
        b1 = b1_of(track).reindex(liq.index)
        for alpha in (0.2, 0.5, 1.0):
            mix_cost = alpha * liq + (1 - alpha) * fp
            mix_free = alpha * liq + (1 - alpha) * b1
            for w in WINS:
                a, b = WIN[track][w]
                for kind, mix in (("with_pool_cost", mix_cost), ("pool_leg_free", mix_free)):
                    rel = LE.rel_block(LE.slice_eval(mix, a, b), LE.slice_eval(b1, a, b))
                    rows.append(dict(track=track, alpha=alpha, window=w, kind=kind, excess_arith_pp=rel["excess_arith"] * 100,
                                     te_pct=rel["te"] * 100, ir=rel["ir"]))
    return pd.DataFrame(rows)


def diag_posthoc_rev() -> pd.DataFrame:
    """P2-3:事后 REV 20/35 在 A 轨 2006-2016(对该假说是样本外)的 ΔIR 与对排雷截距。"""
    rows = []
    x = excess("a", "S|LIQ|20_35")
    for n in ("S|REV|20_35", "C|LR_20_35"):
        y = excess("a", n)
        for lab, (a, b) in {"2006-2016": ("2006-01-04", "2016-12-30"), "2017-2025": ("2017-01-03", "2025-12-31"),
                            "full": ("2006-01-04", "2025-12-31")}.items():
            ib = ir_block("a", n, a, b)
            il = ir_block("a", "S|LIQ|20_35", a, b)
            rows.append(dict(config=n, segment=lab, ir=ib["ir"], ir_liq=il["ir"], d_ir=ib["ir"] - il["ir"],
                             rho_with_liq=reg(y, x, a, b)["rho"], **{f"gen_{k}": v for k, v in reg(y, x, a, b).items() if k in ("a_ann_pp", "t_a", "b")}))
    return pd.DataFrame(rows)


def stage_diag():
    t0 = time.time()
    dr = diag_depth_ref()
    dr.to_csv(LEDGER / "inc_t5_fix_depth_ref.csv", index=False)
    log(f"[fix_diag] 深度参照 {len(dr)} 行 {time.time() - t0:.0f}s")
    di = diag_depth_ir()
    di.to_csv(LEDGER / "inc_t5_fix_depth_ir.csv", index=False)
    e3 = diag_e3()
    e3.to_csv(LEDGER / "inc_t5_fix_e3.csv", index=False)
    x5, x5g = diag_min_x5()
    x5.to_csv(LEDGER / "inc_t5_fix_min_x5.csv", index=False)
    x5g.to_csv(LEDGER / "inc_t5_fix_min_x5_gen.csv", index=False)
    dc = diag_dilution_cost()
    dc.to_csv(LEDGER / "inc_t5_fix_dilution_cost.csv", index=False)
    ph = diag_posthoc_rev()
    ph.to_csv(LEDGER / "inc_t5_fix_posthoc_rev.csv", index=False)
    log(f"[fix_diag] 完成 {time.time() - t0:.0f}s")


# ================================================================ 偏离账本
def stage_devs():
    d = pd.read_csv(DEVS)
    if "review_note" not in d.columns:
        d["review_note"] = ""
    first_ret = pd.Timestamp("2026-10-11 04:38:00")      # 第一次回测(runs_a)开始
    ts = pd.to_datetime(d["logged_at"].str.replace(" JST", "", regex=False))
    late = (ts > first_ret) & d["stage"].isin(["runs", "clarify"])
    d.loc[late, "decided_before_returns"] = False
    d.loc[late, "review_note"] = "登记时点(" + d.loc[late, "logged_at"] + ")晚于首次回测(04:38);决定时点无法由日志证明,按事后登记的澄清处理"
    d.to_csv(DEVS, index=False)
    log(f"[fix_devs] 重新标注 {int(late.sum())} 条: {list(d.loc[late, 'item'])}")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        if s == "runs_min_depth":
            stage_runs_min_depth()
        elif s == "diag":
            stage_diag()
        elif s == "devs":
            stage_devs()
        else:
            raise SystemExit(f"unknown stage {s}")
