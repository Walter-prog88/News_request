"""开关变量全扫描:哪些变量能标识出"因子什么时候管用"。

动机(2026-08-25):
  因子搜索在两个维度上密度极不对称 —— "被预测的量"搜了 51 原料 × 5 算子 × 3 窗口 = 728 个,
  "条件变量(开关)"一共只用过 8 个(amount / 量比 / rvol / 尾盘占比 / amtz / ep / roe / size),
  而且是手挑的。今天测出最强的开关(真实换手率)根本不在这 8 个里。

  更根本的问题:IC 是**全截面平均**,而开关的定义就是"在不同地方不一样",
  取平均恰好把它抹掉。市值本身作为因子 IC 0.0091 / t 1.8 不显著、被筛掉,
  却在模型里扛 24% 的置换重要性 —— 就是这个盲区的直接证据。

做法:把每个日频原料都当候选开关,按它把股票分 10 档,
      看核心因子的 IC 在各档之间散得多开。散得越开 = 越是好开关。

打分:disp = std(各档 IC) / |全样本 IC|,对所有核心因子取中位数。
零假设:8 个随机变量同样跑一遍,给出噪声地板;只有明显高出地板的才算数。

⚠ 纪律:价格水平类变量(前复权价、VWAP、TWAP、高低价)**禁止**当开关 ——
   绝对价格水平含未来分红信息,用它分桶会引入前视(与因子层同一条禁令)。

产出:research/ledger/switch_sweep.csv        每个候选开关的打分
      research/ledger/switch_sweep_detail.csv 前 N 名开关 × 因子 × 档位的明细
"""
from __future__ import annotations
import os, sys, csv, warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, DAILY
from afactor.panel import Panel
from afactor.fast_eval import EvalContext, eval_one
from afactor import factors as FA, ops

N_BUCKETS = 10
MIN_BUCKET_N = 30
MIN_BUCKET_DAYS = 250
N_RANDOM = 8                 # 零假设:随机分桶变量个数
SEED = 20260825

# 价格水平类:前复权价含未来分红信息,禁止作为分桶变量(与因子层同一条禁令)
_BANNED = {"close_px", "vwap", "twap_open10", "twap_tail8", "open_auction_px",
           "high_px", "low_px", "px_1450", "statind_labels"}

# 被切的因子:核心 32(已证明对 706 个过关因子的 IC 序列回归 R² 中位 0.921,足以代表全库)
def _targets():
    core = [r["factor"] for r in csv.DictReader(open(LEDGER / "neutralization.csv"))]
    extra = ["mx_rvol_d60", "pv_corr_5", "mx_amt_morning_share_std10", "rskew_5"]
    return [n for n in core + extra if n in FA.REGISTRY]


def _candidates(p):
    """候选开关 = 全部非 pre_ 日频原料(剔价格水平) + 真实市值/换手/流通市值。"""
    raws = sorted(f.stem for f in DAILY.glob("*.parquet")
                  if not f.stem.startswith("pre_") and f.stem not in _BANNED)
    out = [(n, ("raw", n)) for n in raws]
    out += [("REAL_mv", ("real", "mv")), ("REAL_turn", ("real", "turn")),
            ("REAL_circ", ("real", "circ")),
            ("DERIVED_adv20", ("derived", "adv20")),
            ("DERIVED_turnover_proxy", ("derived", "turnover_proxy"))]
    out += [(f"NULL_random{i}", ("null", i)) for i in range(N_RANDOM)]
    return out


_G = {}


def _init():
    p = Panel()
    FA.build_all()
    ctx = EvalContext(p, "open", (1,))
    _G.update(p=p, ctx=ctx, u=p["universe"], R=ctx.R[1], M=ctx.M[1])
    # 核心因子只构造一次
    F, base = {}, {}
    for n in _targets():
        try:
            f = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) \
                .replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
        except Exception:
            continue
        ic, _, _, _ = eval_one(f, ctx.R[1], ctx.M[1], want_layers=False)
        v = ic[np.isfinite(ic)]
        if v.size < 300 or abs(v.mean()) < 1e-6:
            continue
        F[n] = f
        base[n] = float(v.mean())
    _G["F"], _G["base"] = F, base


def _fetch(spec):
    p = _G["p"]
    kind, key = spec
    if kind == "raw":
        return p.raw(key).to_numpy(np.float32)
    if kind == "real":
        return FA._real(p, key).to_numpy(np.float32)
    if kind == "derived":
        return p[key].to_numpy(np.float32)
    if kind == "null":
        rng = np.random.default_rng(SEED + int(key))
        return rng.standard_normal(_G["M"].shape).astype(np.float32)
    raise ValueError(kind)


def _bucketize(X):
    """每日在决策池内按 X 分 N 档;1 = 最小。池外/缺值 = -1。"""
    M = _G["M"]
    x = np.where(M & np.isfinite(X), X, np.nan)
    out = np.full(x.shape, -1, np.int16)
    for t in range(x.shape[0]):
        v = x[t]
        m = np.isfinite(v)
        n = int(m.sum())
        if n < N_BUCKETS * MIN_BUCKET_N:
            continue
        r = rankdata(v[m], method="average") / n
        out[t, m] = np.clip(np.ceil(r * N_BUCKETS), 1, N_BUCKETS).astype(np.int16)
    return out


def _groups(B):
    g = []
    for t in range(B.shape[0]):
        m = _G["M"][t] & (B[t] >= 0)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        b = B[t, idx]
        o = np.argsort(b, kind="stable")
        idx, b = idx[o], b[o]
        parts = np.split(np.arange(len(idx)), np.flatnonzero(np.diff(b)) + 1)
        day = [(int(b[pp[0]]), idx[pp]) for pp in parts if len(pp) >= MIN_BUCKET_N]
        if len(day) >= N_BUCKETS - 1:          # 要求几乎所有档都成立,否则该日不计
            g.append((t, day))
    return g


def _bucket_ic(F, groups, Rrank):
    acc = {}
    for (t, day), rr_day in zip(groups, Rrank):
        ft = F[t]
        for (k, idx), (_, rr) in zip(day, rr_day):
            v = ft[idx]
            ok = np.isfinite(v)
            if int(ok.sum()) < MIN_BUCKET_N:
                continue
            fr = rankdata(v[ok]); r2 = rankdata(rr[ok])
            fm = fr - fr.mean(); rm = r2 - r2.mean()
            d = np.sqrt((fm * fm).sum() * (rm * rm).sum())
            if d > 0:
                acc.setdefault(k, []).append((fm * rm).sum() / d)
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) >= MIN_BUCKET_DAYS}


def _run(chunk):
    rows, detail = [], []
    R, F, base = _G["R"], _G["F"], _G["base"]
    for name, spec in chunk:
        try:
            B = _bucketize(_fetch(spec))
            g = _groups(B)
            if len(g) < 400:
                rows.append({"switch": name, "note": f"有效天数不足 ({len(g)})"})
                continue
            Rrank = [[(k, rankdata(R[t, idx])) for k, idx in day] for t, day in g]
            disps, ratios, shapes = [], [], []
            for fn, arr in F.items():
                res = _bucket_ic(arr, g, Rrank)
                if len(res) < N_BUCKETS - 1:
                    continue
                ks = sorted(res)
                ic = np.array([res[k] for k in ks])
                b = abs(base[fn])
                disps.append(ic.std(ddof=1) / b)
                ratios.append(np.abs(ic).max() / max(np.abs(ic).min(), 1e-6))
                shapes.append(pd.Series(ic).corr(pd.Series(ks), method="spearman"))
                for k, val in zip(ks, ic):
                    detail.append({"switch": name, "factor": fn, "decile": k, "ic": val})
            if not disps:
                rows.append({"switch": name, "note": "无因子成立"})
                continue
            rows.append({"switch": name, "n_factors": len(disps), "n_days": len(g),
                         "disp_median": float(np.median(disps)),
                         "disp_mean": float(np.mean(disps)),
                         "ratio_median": float(np.median(ratios)),
                         "shape_median": float(np.nanmedian(shapes)),
                         "shape_abs_median": float(np.nanmedian(np.abs(shapes)))})
        except Exception as e:
            rows.append({"switch": name, "note": f"{type(e).__name__}: {e}"})
    return rows, detail


def main():
    p = Panel(); FA.build_all()
    cand = _candidates(p)
    print(f"候选开关 {len(cand)} 个(含 {N_RANDOM} 个随机零假设);"
          f"被切因子 {len(_targets())} 个", flush=True)
    print(f"已禁用价格水平类变量:{sorted(_BANNED)}", flush=True)

    k = max(1, len(cand) // 12)
    chunks = [cand[i:i + k] for i in range(0, len(cand), k)]
    rows, detail = [], []
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        for fut in as_completed([ex.submit(_run, c) for c in chunks]):
            r, d = fut.result()
            rows += r; detail += d
            print(f"  [{len(rows)}/{len(cand)}]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "switch_sweep.csv", index=False)
    ok = df[df.get("disp_median").notna()] if "disp_median" in df else df
    null = ok[ok.switch.str.startswith("NULL_")]
    real = ok[~ok.switch.str.startswith("NULL_")].sort_values("disp_median", ascending=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    print("=== 零假设:随机变量分桶的离散度(噪声地板)===")
    print(null[["switch", "disp_median", "ratio_median", "n_days"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    floor = null.disp_median.max() if len(null) else np.nan
    print(f"\n噪声地板(随机变量里最高的 disp_median)= {floor:.4f}")

    print("\n" + "=" * 100)
    print("=== 开关排行榜(disp = 各档 IC 的标准差 ÷ 全样本|IC|,越大越是开关)===")
    show = real.head(30).copy()
    show["高出地板"] = show.disp_median / floor
    print(show[["switch", "disp_median", "ratio_median", "shape_median",
                "shape_abs_median", "高出地板", "n_factors"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n显著高出噪声地板(>1.5倍)的候选开关:{int((real.disp_median > floor * 1.5).sum())} / {len(real)}")

    if detail:
        dd = pd.DataFrame(detail)
        top = real.head(12).switch.tolist()
        dd[dd.switch.isin(top)].to_csv(LEDGER / "switch_sweep_detail.csv", index=False)
        print("\n=== 前 6 名开关:核心因子 IC 的档位形状(所有因子中位)===")
        for s in real.head(6).switch:
            sub = dd[dd.switch == s].pivot_table(index="factor", columns="decile", values="ic")
            med = sub.abs().median()
            print(f"  {s:26s} " + " ".join(f"D{int(c)}:{v:.3f}" for c, v in med.items()))
    print("\n产出:switch_sweep.csv / switch_sweep_detail.csv")


if __name__ == "__main__":
    main()
