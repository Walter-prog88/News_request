"""中段因子组合:及格线下面的因子凑起来,能不能凑成一个过关的?

问题来源:
  因子层的收人门槛是 |IC| >= 0.025。账本里 |IC| 落在 0.010~0.025 的有 541 个,从未进过模型。
  而模型归因显示 Spearman(置换重要性, 原始|IC|) = 0.079,扛 70% 权重的两个特征
  IC 只有 0.0223 / 0.0241 —— 都在门槛下面。门槛可能画错了。

  先前粗测(C 高IC / D 中段IC / E 多样性 三组消融)D 组赢 8pp,但相位极差 28.6pp,未能定性。

  IC 序列相关显示中段因子**比过关因子多样得多**:两两相关中位 0.087(过关的高得多),
  有效独立维度 N_eff = 11.9(过关的 708 个只有 3.9),对 32 核心回归 R² 中位 0.652(过关的 0.921)。
  但**弱因子的 IC 序列本身噪声就大,两串纯噪声的相关也接近 0** ——
  "12 个独立信号"和"12 份互不相干的噪声"在相关矩阵上长得一模一样。
  唯一的区分办法就是真的做一次组合去测。

设计(避开 winner's curse):
  这 541 个是用**全期** IC 挑出来的,直接在全期做组合等于事后诸葛亮。必须分窗口:
  选择窗只用来挑因子和定方向,持出窗只用来评组合,两窗不重叠。

  主口径(用户 2026-08-25 指定):2020-2024 选 / 2025 测
    理由:① 2025H2 是用户认定的"趋势市样本",考的正是目标行情;
          ② 2024 含 9.24 政策行情,放进训练窗,选出的因子才可能扛住状态切换。
    ⚠ 局限:2025H2 在"小碎步法"实验里已当样本外看过一次,不是全新考卷;
       2026 已封存用尽,这是现有最好的选择,结果要打折读。
  稳健性:2020-2023 选 / 2024-2025 测;2020-2022 选 / 2023-2025 测
    三种切法结论一致才算数;若结论随切法翻转,那本身就是结论(状态依赖)。

对照组(同一持出窗,口径完全相同):
  D 中段组 : 选择窗 0.010 <= |IC| < 0.025 且分年方向一致
  C 过关组 : 选择窗 |IC| >= 0.025 且分年方向一致,按 |t| 取前 N(N = D 组大小)
  COMP_v5  : 现有默认组合
  另报 D 组对 COMP_v5 正交后的残差 IC —— 能不能给现有组合**加**东西,这才是最终判据。

产出:research/ledger/midband_composite.csv
"""
from __future__ import annotations
import os, sys, json, glob, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor.fast_eval import EvalContext, eval_one, _ic_block, _layer_block, IC_DIR
from afactor import factors as FA, ops, combine, screen, backtest as BT

SPLITS = [("主口径", (2020, 2024), (2025, 2025)),
          ("稳健A", (2020, 2023), (2024, 2025)),
          ("稳健B", (2020, 2022), (2023, 2025))]
MID_LO, MID_HI = 0.010, 0.025
MIN_DAYS = 400

_G = {}


def _load_ic_series(names):
    out = {}
    for n in names:
        f = IC_DIR / f"{n}__open.parquet"
        if f.exists():
            s = pd.read_parquet(f)
            out[n] = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    return out


def _select(ics, lo, hi, yrs):
    """在选择窗内按 |IC| 区间 + 分年方向一致挑因子;返回 {name: (ic, t, sign)}。"""
    a, b = yrs
    out = {}
    for n, s in ics.items():
        w = s[(s.index.year >= a) & (s.index.year <= b)].dropna()
        if len(w) < MIN_DAYS:
            continue
        m, sd = w.mean(), w.std(ddof=1)
        if sd <= 0:
            continue
        t = m / sd * np.sqrt(len(w))
        by_year = w.groupby(w.index.year).mean()
        if len(by_year) < 2 or not (np.sign(by_year) == np.sign(m)).all():
            continue
        if lo <= abs(m) < hi:
            out[n] = (float(m), float(t), float(np.sign(m)))
    return out


def _init():
    _G["p"] = Panel()
    FA.build_all()
    _G["ctx"] = EvalContext(_G["p"], "open", (1,))
    _G["u"] = _G["p"]["universe"]


def _accumulate(job):
    """一次构造,喂给多个累加器(避免 500+ 因子同时驻留内存)。"""
    names_signs, groups = job          # groups: {key: set(name)}
    p, u = _G["p"], _G["u"]
    shape = (len(p.dates), len(p.codes))
    acc = {k: [np.zeros(shape, np.float32), np.zeros(shape, np.int16)] for k in groups}
    for n, sg in names_signs:
        try:
            f = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
            z = ops.standardize(f, u).to_numpy(np.float32) * sg
        except Exception:
            continue
        ok = np.isfinite(z)
        for k, members in groups.items():
            if n in members:
                acc[k][0][ok] += z[ok]
                acc[k][1][ok] += 1
    return acc


def _eval_window(F, yrs, label):
    ctx, p = _G["ctx"], _G["p"]
    a, b = yrs
    mask = (p.dates.year >= a) & (p.dates.year <= b)
    A = F.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32)
    A = np.where(mask[:, None], A, np.nan)
    ic, q, cov, turn = eval_one(A, ctx.R[1], ctx.M[1], want_layers=True)
    blk = _ic_block(ic, ctx.years)
    if blk.get("ic_mean") is None:
        return None
    blk.update(_layer_block(q, turn))
    # date_mask 已在引擎内部限定区间(M &= date_mask),此处不必再遮 F,
    # 否则 where 的广播语义与引擎的池遮罩重复,反而可能改变可交易判定。
    bt = BT.run_standard(F, p, sign=1, mode="open", date_mask=mask)
    bench = BT.benchmark_nav(p, "open")
    nav = bt["nav"]
    # 基准是从样本起点算的**累计**净值,必须在窗口内重新归一,
    # 否则会拿 1 年的策略净值去比 6 年的累计基准(实测把超额压成 -40%)。
    bch = bench.reindex(nav.index).ffill()
    bch = bch / bch.iloc[0]
    yrs_n = max(len(nav) / 242.0, 1e-9)
    blk["bt_ann"] = float(nav.iloc[-1] ** (1 / yrs_n) - 1)
    blk["bt_excess"] = blk["bt_ann"] - float(bch.iloc[-1] ** (1 / yrs_n) - 1)
    blk["bt_sharpe"] = bt.get("sharpe")
    blk["bt_turnover"] = bt.get("avg_turnover")
    blk["组"] = label
    return blk


def main():
    FA.build_all()
    reg = set(FA.REGISTRY)
    names = [f.stem.replace("__open", "") for f in IC_DIR.glob("*__open.parquet")]
    names = [n for n in names if n in reg
             and FA.REGISTRY[n][0] not in ("control", "composite")]
    print(f"有 IC 序列且在注册表里的因子:{len(names)}", flush=True)
    ics = _load_ic_series(names)
    print(f"载入 IC 序列 {len(ics)} 条", flush=True)

    plan, groups = {}, {}
    for tag, tr, te in SPLITS:
        D = _select(ics, MID_LO, MID_HI, tr)
        Craw = _select(ics, MID_HI, 9.9, tr)
        C = dict(sorted(Craw.items(), key=lambda kv: -abs(kv[1][1]))[:len(D)])
        plan[tag] = {"train": tr, "test": te, "D": D, "C": C}
        groups[f"{tag}|D"] = set(D)
        groups[f"{tag}|C"] = set(C)
        print(f"  {tag}: 训练 {tr[0]}-{tr[1]} → 测试 {te[0]}-{te[1]} | "
              f"中段 D {len(D)} 个(|IC|中位 {np.median([abs(v[0]) for v in D.values()]):.4f})"
              f" | 过关 C {len(C)} 个(|IC|中位 {np.median([abs(v[0]) for v in C.values()]):.4f})"
              f" | D∩C={len(set(D)&set(C))}", flush=True)

    # 因子名 -> 各 split 下的方向(以 D/C 所属 split 的训练窗方向为准;同一因子在不同 split
    # 可能属不同组、方向也可能不同,故按 (name, sign) 分别累加)
    need = {}
    for tag in plan:
        for grp in ("D", "C"):
            for n, (_, _, sg) in plan[tag][grp].items():
                need.setdefault((n, sg), set()).add(f"{tag}|{grp}")
    items = [(n, sg, ks) for (n, sg), ks in need.items()]
    print(f"\n需要构造的 (因子,方向) 组合:{len(items)}", flush=True)

    W = 5
    chunks = [items[i::W] for i in range(W)]
    jobs = []
    for ch in chunks:
        ns = [(n, sg) for n, sg, _ in ch]
        gmap = {k: {n for n, _, ks in ch if k in ks} for k in groups}
        jobs.append((ns, gmap))

    total = {k: [None, None] for k in groups}
    with ProcessPoolExecutor(max_workers=W, initializer=_init) as ex:
        done = 0
        for fut in as_completed([ex.submit(_accumulate, j) for j in jobs]):
            acc = fut.result(); done += 1
            for k, (s, c) in acc.items():
                if total[k][0] is None:
                    total[k] = [s, c]
                else:
                    total[k][0] += s; total[k][1] += c
            print(f"  累加 {done}/{W}", flush=True)

    _init()
    p, u = _G["p"], _G["u"]
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    cm = [n for n in man["members"] if n in FA.REGISTRY]
    comp5 = combine.combine_equal(combine.prepare(
        {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) for n in cm},
        {n: float(man["signs"][n]) for n in cm}, u))

    rows = []
    for tag, tr, te in SPLITS:
        for grp, lab in (("D", "D中段组"), ("C", "C过关组")):
            s, c = total[f"{tag}|{grp}"]
            n_min = max(1, len(plan[tag][grp]) // 3)
            with np.errstate(invalid="ignore"):
                F = np.where(c >= n_min, s / np.maximum(c, 1), np.nan)
            F = pd.DataFrame(F, index=p.dates, columns=p.codes)
            r = _eval_window(F, te, lab)
            if r:
                r.update(切法=tag, 训练窗=f"{tr[0]}-{tr[1]}", 持出窗=f"{te[0]}-{te[1]}",
                         成员数=len(plan[tag][grp]))
                rows.append(r)
            if grp == "D":
                resid = ops.cs_neutralize_1d(ops.standardize(F, u), ops.standardize(comp5, u))
                r2 = _eval_window(resid, te, "D组对COMP_v5正交后")
                if r2:
                    r2.update(切法=tag, 训练窗=f"{tr[0]}-{tr[1]}", 持出窗=f"{te[0]}-{te[1]}",
                              成员数=len(plan[tag]["D"]))
                    rows.append(r2)
        r3 = _eval_window(comp5, te, "COMP_v5")
        if r3:
            r3.update(切法=tag, 训练窗=f"{tr[0]}-{tr[1]}", 持出窗=f"{te[0]}-{te[1]}", 成员数=25)
            rows.append(r3)

    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "midband_composite.csv", index=False)
    pd.set_option("display.width", 230)
    cols = ["切法", "持出窗", "组", "成员数", "ic_mean", "ic_t", "mono_spearman",
            "long_excess_ann", "long_turnover", "bt_excess", "bt_sharpe", "bt_turnover"]
    print("\n" + "=" * 130)
    print("=== 持出窗结果(选择窗只用来挑因子与定方向,两窗不重叠)===")
    print(df[[c for c in cols if c in df]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n--- 判据 ---")
    for tag, _, te in SPLITS:
        d = df[df.切法 == tag].set_index("组")
        if "D中段组" in d.index and "C过关组" in d.index:
            D, C = d.loc["D中段组"], d.loc["C过关组"]
            print(f"  {tag}({te[0]}-{te[1]}):D 组 IC {D.ic_mean:+.4f} vs C 组 {C.ic_mean:+.4f} "
                  f"| D 回测超额 {D.bt_excess*100:+.2f}% vs C {C.bt_excess*100:+.2f}%"
                  + (f" | D 是否过 0.025 关:{'是' if abs(D.ic_mean)>=0.025 else '否'}"))
        if "D组对COMP_v5正交后" in d.index:
            print(f"      D 对 COMP_v5 正交后的残差 IC = {d.loc['D组对COMP_v5正交后'].ic_mean:+.4f}"
                  f"(增量判据,接近 0 = 没给现有组合加东西)")
    print("\n产出:midband_composite.csv")


if __name__ == "__main__":
    main()
