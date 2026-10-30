"""行业 / 市值:中性化重测(A)+ 行业桶分层(B)+ 市值桶分层(C)。

背景与修复(2026-08-25):
  1. `run_neutralization.py:65` 的 "仅行业" 档传的是 `lc * 0.0` 而非 `None`,
     全零回归元 -> beta = 0/0 -> 残差全 NaN。两次运行日志的 `ind` 中位都是 nan,
     README §8.3 表里 "仅行业 0.931 / 最低 0.771" 无账本支撑。此处传 None 修复。
  2. 旧版用 `p["log_cap"]`(前复权价 x PIT 股本,函数自述含分红泄漏);
     此处改用 `FA._real(p, "mv")`(未复权价 x 股本,真实市值),并保留旧口径做对照。
  3. 旧版行业用统计行业(KMeans-30);此处主用申万一级(31 类,用户提供的
     `stock_sw_industry.csv`),统计行业作交叉验证。

口径限制:申万分类是**当前快照**,无 in_date/out_date,给 2020 年贴 2026 的标签
有轻微前视。故同时输出统计行业版本,两者接近则该偏差不重要。

产出:
  research/ledger/neutral_v2.csv          A 各中性化变体的 IC 与保留率
  research/ledger/bucket_industry_ic.csv  B 因子 x 申万一级 的 Rank IC
  research/ledger/bucket_size_ic.csv      C 因子 x 真实市值十分位 的 Rank IC
"""
from __future__ import annotations
import os, sys, json, csv, warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor.fast_eval import EvalContext, eval_one
from afactor import factors as FA, ops, statind as SI, newdata as ND

N_SIZE_BUCKETS = 10
MIN_BUCKET_N = 30          # 桶内当日最少股票数,低于此当日不计 IC
MIN_BUCKET_DAYS = 250      # 桶总天数少于此不报告

# 32 个核心因子(= 现有 neutralization.csv)+ 6 个"最张不住"的补充,
# 补充依据:708 个过关因子对 32 核心的 IC 序列回归,这 6 个 R^2 最低(0.41~0.63)。
EXTRA = ["lit_vola_overnight_vol_share_w5", "lit_vola_overnight_vol_share_w20",
         "mx_rvol_d60", "pv_corr_5", "mx_amt_morning_share_std10", "rskew_5"]

_G = {}


def _init():
    p = Panel()
    FA.build_all()
    u = p["universe"]
    ctx = EvalContext(p, "open", (1,))
    mv = FA._real(p, "mv")
    _G.update(
        p=p, u=u, ctx=ctx,
        lc_real=ops.signed_log(mv), lc_approx=p["log_cap"],
        ladv=ops.signed_log(p["adv20"]),
        L_sw=ND.sw_industry_labels(p.dates, p.codes),
        L_stat=SI.load_labels().reindex(index=p.dates, columns=p.codes),
        mv=mv,
    )
    u_nost = u & ~p["is_st"].reindex_like(u).fillna(False)
    ctx2 = EvalContext(p, "open", (1,))
    ctx2.M = {1: u_nost.to_numpy(bool) & np.isfinite(ctx.R[1])}
    _G["ctx_nost"] = ctx2
    # 三套分桶:市值 / 20日日均成交额 / 真实换手率。
    # ⚠ 池子本身已卡 MIN_ADV20 = 2000 万,故"成交额第 1 档"= 2000 万以上里最冷的,
    #   不是真正的死票;真实成交额分布的下尾已被股票池截断。
    _G["groups_ind"] = _build_groups(_G["L_sw"].to_numpy(np.int16), ctx.M[1])
    _G["groups_size"] = _build_groups(_decile_buckets(mv, u), ctx.M[1])
    _G["groups_liq"] = _build_groups(_decile_buckets(p["adv20"], u), ctx.M[1])
    _G["groups_turn"] = _build_groups(_decile_buckets(FA._real(p, "turn"), u), ctx.M[1])
    for k in ("ind", "size", "liq", "turn"):
        _G[f"Rrank_{k}"] = _rank_by_group(ctx.R[1], _G[f"groups_{k}"])


def _decile_buckets(x: pd.DataFrame, u: pd.DataFrame) -> np.ndarray:
    """每日在决策池内按 x 分 N 档;1 = 最小。池外/缺值 = -1。"""
    x = x.where(u.reindex_like(x).fillna(False))
    q = x.rank(axis=1, pct=True, na_option="keep")
    b = np.ceil(q.to_numpy(np.float64) * N_SIZE_BUCKETS)
    b = np.where(np.isfinite(b), np.clip(b, 1, N_SIZE_BUCKETS), -1)
    return b.astype(np.int16)


def _build_groups(B: np.ndarray, M: np.ndarray):
    """逐日预先切好桶,所有因子共用。返回 [(t, [(bucket_id, idx_array), ...]), ...]"""
    out = []
    for t in range(B.shape[0]):
        m = M[t] & (B[t] >= 0)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        b = B[t, idx]
        order = np.argsort(b, kind="stable")
        idx, b = idx[order], b[order]
        cuts = np.flatnonzero(np.diff(b)) + 1
        parts = np.split(np.arange(len(idx)), cuts)
        day = [(int(b[pp[0]]), idx[pp]) for pp in parts if len(pp) >= MIN_BUCKET_N]
        if day:
            out.append((t, day))
    return out


def _rank_by_group(R: np.ndarray, groups):
    """预先算好前瞻收益在每个桶内的秩(所有因子共用,省一半 rankdata)。"""
    out = []
    for t, day in groups:
        out.append([(k, rankdata(R[t, idx])) for k, idx in day])
    return out


def _bucket_ic(F: np.ndarray, groups, Rrank):
    """返回 {bucket_id: (ic_mean, t, n_days, avg_n)}。"""
    acc = {}
    for (t, day), rr_day in zip(groups, Rrank):
        f_t = F[t]
        for (k, idx), (_, rr) in zip(day, rr_day):
            v = f_t[idx]
            ok = np.isfinite(v)
            n = int(ok.sum())
            if n < MIN_BUCKET_N:
                continue
            fr = rankdata(v[ok])
            r2 = rankdata(rr[ok])            # 桶内秩的子集再排一次,保持同一样本
            fm = fr - fr.mean(); rm = r2 - r2.mean()
            d = np.sqrt((fm * fm).sum() * (rm * rm).sum())
            if d <= 0:
                continue
            acc.setdefault(k, []).append(((fm * rm).sum() / d, n))
    out = {}
    for k, v in acc.items():
        ic = np.array([a for a, _ in v]); ns = np.array([b for _, b in v])
        if len(ic) < MIN_BUCKET_DAYS:
            continue
        t = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)) if ic.std(ddof=1) > 0 else np.nan
        out[k] = (float(ic.mean()), float(t), int(len(ic)), float(ns.mean()))
    return out


def _ic(F: pd.DataFrame, ctx):
    A = F.reindex(index=_G["p"].dates, columns=_G["p"].codes).to_numpy(np.float32)
    ic, _, _, _ = eval_one(A, ctx.R[1], ctx.M[1], want_layers=False)
    v = ic[np.isfinite(ic)]
    if v.size < 100:
        return None, None
    return float(v.mean()), float(v.mean() / v.std(ddof=1) * np.sqrt(v.size))


def _exposure(F: np.ndarray, ctx):
    """因子对 log 真实市值的截面秩相关(日均),以及行业哑变量能解释因子的方差占比。"""
    lc = _G["lc_real"].to_numpy(np.float32)
    L = _G["L_sw"].to_numpy(np.int16)
    M = ctx.M[1]
    cs, r2s = [], []
    for t in range(0, F.shape[0], 5):                    # 每 5 天抽一次,够稳且快
        m = M[t] & np.isfinite(F[t]) & np.isfinite(lc[t]) & (L[t] >= 0)
        if m.sum() < 200:
            continue
        f = rankdata(F[t, m]); c = rankdata(lc[t, m])
        cs.append(np.corrcoef(f, c)[0, 1])
        v = F[t, m].astype(np.float64); lab = L[t, m]
        v = (v - v.mean()) / (v.std() + 1e-12)
        s = np.bincount(lab, weights=v, minlength=31)
        n = np.bincount(lab, minlength=31).astype(float)
        gm = np.divide(s, n, out=np.zeros_like(s), where=n > 0)[lab]
        r2s.append(1.0 - ((v - gm) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12))
    return (float(np.mean(cs)) if cs else None,
            float(np.mean(r2s)) if r2s else None)


def _run(names):
    p, u, ctx = _G["p"], _G["u"], _G["ctx"]
    lc_r, lc_a, ladv = _G["lc_real"], _G["lc_approx"], _G["ladv"]
    L_sw, L_st = _G["L_sw"], _G["L_stat"]
    rows, bi, bkt = [], [], {"size": [], "liq": [], "turn": []}
    for n in names:
        try:
            f = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) \
                .replace([np.inf, -np.inf], np.nan)
            base_ic, base_t = _ic(f, ctx)
            row = {"factor": n, "family": FA.REGISTRY[n][0], "ic": base_ic, "t": base_t}

            variants = {
                # 单维度
                "cap_real":   ops.cs_neutralize_cap_ind(f, lc_r, None, universe=u),
                "cap_approx": ops.cs_neutralize_cap_ind(f, lc_a, None, universe=u),
                "ind_sw":     ops.cs_neutralize_cap_ind(f, None, L_sw, universe=u),
                "ind_stat":   ops.cs_neutralize_cap_ind(f, None, L_st, universe=u),
                # 联合
                "cap_ind_sw": ops.cs_neutralize_cap_ind(f, lc_r, L_sw, universe=u),
            }
            liq = ops.cs_neutralize_cap_ind(ladv, lc_r, L_sw, universe=u)
            variants["cap_ind_sw_liq"] = ops.cs_neutralize_1d(variants["cap_ind_sw"], liq)
            for k, v in variants.items():
                ic, t = _ic(v, ctx)
                row[f"ic_{k}"] = ic
                row[f"t_{k}"] = t
                row[f"ret_{k}"] = (abs(ic) / abs(base_ic)) if (ic and base_ic) else None
            ic_ns, _ = _ic(f, _G["ctx_nost"])
            row["ic_no_st"] = ic_ns
            row["ret_no_st"] = (abs(ic_ns) / abs(base_ic)) if (ic_ns and base_ic) else None

            A = f.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32)
            row["corr_logmv"], row["ind_r2"] = _exposure(A, ctx)
            rows.append(row)

            for k, (ic, t, nd, an) in _bucket_ic(A, _G["groups_ind"], _G["Rrank_ind"]).items():
                bi.append({"factor": n, "ind_code": k, "ic": ic, "t": t,
                           "n_days": nd, "avg_n": an})
            for scheme in ("size", "liq", "turn"):
                res = _bucket_ic(A, _G[f"groups_{scheme}"], _G[f"Rrank_{scheme}"])
                for k, (ic, t, nd, an) in res.items():
                    bkt[scheme].append({"factor": n, "decile": k, "ic": ic, "t": t,
                                        "n_days": nd, "avg_n": an})
        except Exception as e:
            rows.append({"factor": n, "error": f"{type(e).__name__}: {e}"})
    return rows, bi, bkt


def main():
    FA.build_all()
    core = [r["factor"] for r in csv.DictReader(open(LEDGER / "neutralization.csv"))]
    targets = [n for n in core + EXTRA if n in FA.REGISTRY]
    miss = [n for n in core + EXTRA if n not in FA.REGISTRY]
    print(f"待检验 {len(targets)} 个因子(核心 {len(core)} + 补充 {len(EXTRA)})"
          + (f";注册表里没有:{miss}" if miss else ""), flush=True)

    chunks = [targets[i:i + 4] for i in range(0, len(targets), 4)]
    rows, bi = [], []
    bkt = {"size": [], "liq": [], "turn": []}
    with ProcessPoolExecutor(max_workers=5, initializer=_init) as ex:
        for fut in as_completed([ex.submit(_run, c) for c in chunks]):
            r, a, b = fut.result()
            rows += r; bi += a
            for k in bkt:
                bkt[k] += b[k]
            print(f"  [{len(rows)}/{len(targets)}]", flush=True)

    df = pd.DataFrame(rows); df.to_csv(LEDGER / "neutral_v2.csv", index=False)
    dbi = pd.DataFrame(bi)
    names = ND.sw_industry_names()
    if len(dbi):
        dbi["industry"] = dbi["ind_code"].map(names)
        dbi.to_csv(LEDGER / "bucket_industry_ic.csv", index=False)
    FILE = {"size": "bucket_size_ic.csv", "liq": "bucket_liq_ic.csv",
            "turn": "bucket_turn_ic.csv"}
    dbk = {}
    for k, v in bkt.items():
        dbk[k] = pd.DataFrame(v)
        if len(dbk[k]):
            dbk[k].to_csv(LEDGER / FILE[k], index=False)
    dbs = dbk["size"]

    ok = df[df.get("error").isna()] if "error" in df.columns else df
    pd.set_option("display.width", 260)

    print("\n" + "=" * 110)
    print("=== A. 中性化:保留率 = |残差IC| / |原始IC| ===")
    cols = ["factor", "ic", "ret_cap_real", "ret_cap_approx", "ret_ind_sw",
            "ret_ind_stat", "ret_cap_ind_sw", "ret_cap_ind_sw_liq", "ret_no_st",
            "corr_logmv", "ind_r2"]
    print(ok[cols].sort_values("ic", key=abs, ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n--- 各维度中位保留率 ---")
    for k, lab in [("cap_real", "仅真实市值"), ("cap_approx", "仅近似市值(旧口径)"),
                   ("ind_sw", "仅行业(申万一级)"), ("ind_stat", "仅行业(统计行业)"),
                   ("cap_ind_sw", "市值+行业"), ("cap_ind_sw_liq", "市值+行业+流动性"),
                   ("no_st", "剔除 ST")]:
        c = f"ret_{k}"
        if c in ok and ok[c].notna().any():
            print(f"  {lab:22s} 中位 {ok[c].median():.3f}  最低 {ok[c].min():.3f}  "
                  f"<50% 的因子数 {(ok[c] < 0.5).sum()}")

    if len(dbi):
        print("\n" + "=" * 110)
        print("=== B. 行业桶(申万一级):每个因子在各行业内的 Rank IC ===")
        piv = dbi.pivot_table(index="factor", columns="industry", values="ic")
        sm = pd.DataFrame({
            "全样本IC": ok.set_index("factor")["ic"],
            "行业内IC中位": piv.median(axis=1),
            "行业间标准差": piv.std(axis=1),
            "同号行业占比": (np.sign(piv).eq(np.sign(ok.set_index("factor")["ic"]), axis=0)
                              .sum(axis=1) / piv.notna().sum(axis=1)),
            "最强行业": piv.abs().idxmax(axis=1),
            "最弱行业": piv.abs().idxmin(axis=1),
        }).dropna(subset=["行业内IC中位"])
        print(sm.sort_values("行业间标准差", ascending=False)
                .to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\n  参与行业数 {piv.shape[1]},逐因子平均覆盖 {piv.notna().sum(axis=1).mean():.1f} 个行业")

    TITLE = {"size": "C. 市值桶(真实总市值)", "liq": "D. 成交额桶(20 日日均成交额)",
             "turn": "E. 换手率桶(真实换手率)"}
    for k in ("size", "liq", "turn"):
        d = dbk[k]
        if not len(d):
            continue
        print("\n" + "=" * 110)
        print(f"=== {TITLE[k]},1 = 最小档:每个因子在各档的 Rank IC ===")
        if k == "liq":
            print("    ⚠ 股票池已卡 ADV>=2000万,D1 = 2000万以上里最冷的一档,不是真正的死票")
        piv = d.pivot_table(index="factor", columns="decile", values="ic")
        piv.columns = [f"D{int(c)}" for c in piv.columns]
        piv["全样本"] = ok.set_index("factor")["ic"]
        cs = piv[[c for c in piv.columns if c.startswith("D")]]
        piv["小-大"] = cs.iloc[:, 0] - cs.iloc[:, -1]
        piv["峰值档"] = cs.abs().idxmax(axis=1)
        piv["单调性"] = [pd.Series(r.values).corr(pd.Series(range(len(r))), method="spearman")
                          for _, r in cs.iterrows()]
        print(piv.sort_values("小-大", key=abs, ascending=False)
                 .to_string(float_format=lambda x: f"{x:.4f}"))
        pk = cs.abs().idxmax(axis=1).value_counts().sort_index()
        print("  峰值档分布:", {i: int(v) for i, v in pk.items()})
        print(f"  D1÷(D2~D4均值) 中位 {(cs.abs()['D1'] / cs.abs()[['D2','D3','D4']].mean(axis=1)).median():.3f}"
              f" | D10÷峰值 中位 {(cs.abs()['D10'] / cs.abs().max(axis=1)).median():.3f}"
              f" | D10 与全样本同号 {int(np.sign(cs['D10']).eq(np.sign(piv['全样本'])).sum())}/{len(cs)}")

    print("\n产出:neutral_v2.csv / bucket_industry_ic.csv / bucket_{size,liq,turn}_ic.csv")


if __name__ == "__main__":
    main()
