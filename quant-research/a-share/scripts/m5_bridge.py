"""5 分钟口径的历史桥接校验(前缀 m5):用 2020-2025 研究期量化"5 分钟线近似 1 分钟线"的误差。

三种口径(V1000、open10、1000 万、旧成本 1bp + 印花税、5 相位平均):
  A 原版        1 分钟信号 + 1 分钟执行价(Panel)
  B 信号近似    5 分钟信号 + 1 分钟执行价(m5_panel(signal=True, exec_=False))—— 隔离信号近似误差
  C 全近似      5 分钟信号 + 5 分钟执行价(m5_panel(signal=True, exec_=True))—— 前向检验的真实处境
B 与 A 共用 1 分钟基准;C 的基准也用 5 分钟执行价(前向检验只能这样算)。
基准(审查 P1 修复后):两个方案都是 5 日调仓,超额一律对**同频**池等权 ST.benchmark(..., rebalance=5);
每日重置的 k=1 基准只作对照列(excess_k1)。滚动窗口分布去掉开头 4 个建仓日(审查 P0,见 minute5 模块注释)。

评估一律 2021-2025(2020 是预热年,同时是 minute5 候选近似的选定年)。COMP_v5 成员是在 2020-2025 选出的,
所以这里所有收益都是**信号样本内**的,只用来比较两个口径的差,不代表样本外能力。数据源有幸存者偏差(退市股不在列)。

分阶段运行(结果落盘,可续跑):
    python scripts/m5_bridge.py factors     # 成员因子两口径 + 秩相关 + COMP_v5 打分
    python scripts/m5_bridge.py raw         # (b) 类原料两口径秩相关
    python scripts/m5_bridge.py ic          # COMP_v5 打分秩相关、h1/h5 rank IC
    python scripts/m5_bridge.py backtest    # 三口径 x 两方案 x 窗口 + 引擎回归断言
    python scripts/m5_bridge.py exec        # 执行价近似误差
    python scripts/m5_bridge.py window      # 滚动窗口超额分布(给前向检验查百分位)
    python scripts/m5_bridge.py coldstart   # 冷启动读法(前 4 日只建仓)的历史验证,约 10 分钟
    python scripts/m5_bridge.py layer2      # baostock 第二层误差蒙特卡罗(上交所扰动 -> 打分 / Top20 / 窗口误差)
    python scripts/m5_bridge.py bj          # 北交所对 semibeta_mp 市场均值的影响
    python scripts/m5_bridge.py all
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, TMP, DAILY
from afactor.panel import Panel
from afactor import factors as FA, combine, ops, strategy as ST, minute5 as M5
from afactor.fast_eval import eval_one, _nw_t
from afactor.strategy import TRADING_DAYS

EVAL_Y0, EVAL_Y1 = 2021, 2025
SCORE1 = TMP / "m5_score_1m.parquet"
SCORE5 = TMP / "m5_score_5m.parquet"
MEMBER_CS = TMP / "m5_member_cs.parquet"
DAILY_OUT = LEDGER / "m5_bridge_daily.parquet"
PLANS = {"排雷缓冲20/35 r5": dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5),
         "Top20 周频": dict(scheme="top", n=20, rebalance=5)}
WINDOWS = {"2021-2025": (2021, 2025), "2021-2023": (2021, 2023), "2024-2025": (2024, 2025),
           **{str(y): (y, y) for y in range(2021, 2026)}}
CONFIGS = {"A 原版(1m信号+1m执行)": (False, False), "B 5m信号+1m执行": (True, False),
           "C 全近似(5m信号+5m执行)": (True, True)}
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
DAILY_LEVEL = {"close_px", "amount"}                       # 日级量(5 分钟可精确得到)
OPEN_LEVEL = {"open_auction_px", "ret_intraday"}           # 依赖开盘价口径(a')


# ---------------- 工具 ----------------
class _RecPanel(Panel):
    """记录被读取的原料名(核对成员因子的依赖)。"""
    def __init__(self):
        super().__init__()
        self.log = set()

    def raw(self, name):
        self.log.add(name)
        return super().raw(name)


def year_mask(dates, y0, y1) -> np.ndarray:
    y = np.asarray(dates.year)
    return (y >= y0) & (y <= y1)


def cs_spearman(a: pd.DataFrame, b: pd.DataFrame, mask: pd.DataFrame, min_n: int = 200) -> pd.Series:
    """逐日截面 Spearman,只在 mask 内、两者都有限的股票上算。"""
    a = a.replace([np.inf, -np.inf], np.nan); b = b.replace([np.inf, -np.inf], np.nan)
    ok = mask & a.notna() & b.notna()
    A = a.where(ok).rank(axis=1); B = b.where(ok).rank(axis=1)
    A = A.sub(A.mean(axis=1), axis=0); B = B.sub(B.mean(axis=1), axis=0)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    return (num / den.replace(0, np.nan)).where(ok.sum(axis=1) >= min_n)


def summarize(s: pd.Series, prefix: str = "") -> dict:
    s = s.dropna()
    sub = s[(s.index.year >= EVAL_Y0) & (s.index.year <= EVAL_Y1)]
    out = {f"{prefix}n_days": int(sub.size), f"{prefix}mean": float(sub.mean()), f"{prefix}min": float(sub.min())}
    for q in QS:
        out[f"{prefix}p{int(q * 100):02d}"] = float(sub.quantile(q))
    out[f"{prefix}frac_lt_0.8"] = float((sub < 0.8).mean())
    for y in range(EVAL_Y0, EVAL_Y1 + 1):
        out[f"{prefix}y{y}"] = float(s[s.index.year == y].mean())
    return out


def classify(deps: set, name: str) -> str:
    minute = deps - DAILY_LEVEL - OPEN_LEVEL
    if minute:
        return "(b) 分钟结构"
    if deps & OPEN_LEVEL:
        return "(a') 日级但依赖开盘价口径"
    return "(a) 只依赖日级字段"


def load_members():
    FA.build_all()
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members = [n for n in man["members"] if n in FA.REGISTRY]
    missing = [n for n in man["members"] if n not in FA.REGISTRY]
    return members, man["signs"], missing


# ---------------- 阶段 1:成员因子 + 打分 ----------------
def stage_factors():
    members, signs, missing = load_members()
    print(f"COMP_v5 成员 {len(members)} 个(不在 REGISTRY:{missing})", flush=True)
    p1 = _RecPanel()
    p5 = M5.m5_panel(signal=True, exec_=False)
    U = ST.universe(p1, "V1000")
    u1, u5 = p1["universe"], p5["universe"]
    z1, z5, cs, rows = {}, {}, {}, []
    keep_keys = lambda d: {k: v for k, v in d.items() if k.startswith("__") or k == "statind_L"}
    for n in members:
        t0 = time.time()
        p1._derived = keep_keys(p1._derived)          # 清掉派生缓存,让依赖记录完整(基本面/行业标签缓存保留)
        p1.log = set()
        fn = FA.REGISTRY[n][1]
        f1 = fn(p1).reindex(index=p1.dates, columns=p1.codes).replace([np.inf, -np.inf], np.nan)
        deps = set(p1.log)
        f5 = fn(p5).reindex(index=p5.dates, columns=p5.codes).replace([np.inf, -np.inf], np.nan)
        c = cs_spearman(f1, f5, U)
        cs[n] = c
        fam = FA.REGISTRY[n][0]
        extra = []
        if n.startswith("fx_") or fam == "fundamental":
            extra.append("基本面PIT")
        if n.startswith("si_"):
            extra.append("统计行业标签")
        deps_used = sorted(deps)
        rows.append(dict(member=n, family=fam, sign=signs.get(n, 1.0),
                         deps_raw="|".join(deps_used), deps_other="|".join(extra), cls=classify(deps, n),
                         **summarize(c)))
        sg = float(np.sign(signs.get(n, 1.0)))
        z1[n] = (ops.standardize(f1, u1) * sg).astype(np.float32)
        z5[n] = (ops.standardize(f5, u5) * sg).astype(np.float32)
        del f1, f5
        print(f"  {n:40s} {rows[-1]['cls']:16s} 秩相关均值 {rows[-1]['mean']:.4f}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values("mean")
    df.to_csv(LEDGER / "m5_member_rankcorr.csv", index=False)
    pd.DataFrame(cs).to_parquet(MEMBER_CS)
    s1 = combine.combine_equal(z1); s5 = combine.combine_equal(z5)
    s1.to_parquet(SCORE1); s5.to_parquet(SCORE5)
    ref = TMP / "comp_v5_score_open10.parquet"
    if ref.exists():
        r = pd.read_parquet(ref).reindex_like(s1)
        d = (r - s1).abs().to_numpy()
        print(f"  打分自检 vs {ref.name}: 最大绝对差 {np.nanmax(d):.2e},NaN 位置一致 {np.array_equal(np.isnan(r.to_numpy()), np.isnan(s1.to_numpy()))}")
    print(df[["member", "cls", "mean", "p05", "p50", "frac_lt_0.8"]].to_string(index=False))


# ---------------- 阶段 2:原料秩相关 ----------------
def stage_raw():
    p1 = Panel()
    U = ST.universe(p1, "V1000")
    rows = []
    for f in list(M5.M5_SIGNAL_FIELDS) + list(M5.M5_EXEC_FIELDS):
        a = p1.raw(f)
        b = pd.read_parquet(M5.M5_DIR / f"{f}.parquet").astype(np.float32).reindex_like(a)
        c = cs_spearman(a, b, U)
        cls = ("(a) 日级" if f in DAILY_LEVEL else "(a') 开盘价口径" if f in OPEN_LEVEL
               else "执行" if f in M5.M5_EXEC_FIELDS else "(b) 分钟结构")
        rows.append(dict(field=f, cls=cls, **summarize(c)))
        print(f"  {f:22s} {rows[-1]['mean']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(LEDGER / "m5_raw_rankcorr.csv", index=False)


# ---------------- 阶段 3:打分秩相关与 IC ----------------
def _topk_overlap(s1, s5, U, k=20, bottom_frac=None):
    """Top-k 名单重合率;bottom_frac 给定时为最差 frac 的 Jaccard(排雷方案的剔除区)。"""
    A = s1.where(U).to_numpy(np.float64); B = s5.where(U).to_numpy(np.float64)
    out = np.full(len(s1), np.nan)
    for t in range(len(s1)):
        m = np.isfinite(A[t]) & np.isfinite(B[t])
        n = int(m.sum())
        if n < 200:
            continue
        idx = np.flatnonzero(m)
        if bottom_frac is None:
            ta = set(idx[np.argsort(-A[t, idx])[:k]]); tb = set(idx[np.argsort(-B[t, idx])[:k]])
            out[t] = len(ta & tb) / k
        else:
            kk = int(n * bottom_frac)
            ta = set(idx[np.argsort(A[t, idx])[:kk]]); tb = set(idx[np.argsort(B[t, idx])[:kk]])
            out[t] = len(ta & tb) / len(ta | tb)
    return pd.Series(out, index=s1.index)


def stage_ic():
    p1 = Panel()
    p5f = M5.m5_panel(signal=True, exec_=True)
    U = ST.universe(p1, "V1000")
    s1 = pd.read_parquet(SCORE1).reindex(index=p1.dates, columns=p1.codes)
    s5 = pd.read_parquet(SCORE5).reindex(index=p1.dates, columns=p1.codes)
    rows = [dict(item="COMP_v5 打分 1m vs 5m 截面秩相关", **summarize(cs_spearman(s1, s5, U)))]
    rows.append(dict(item="Top20 名单重合率", **summarize(_topk_overlap(s1, s5, U, 20))))
    rows.append(dict(item="最差 20% 剔除区 Jaccard", **summarize(_topk_overlap(s1, s5, U, bottom_frac=0.20))))
    pd.DataFrame(rows).to_csv(LEDGER / "m5_score_rankcorr.csv", index=False)
    print(pd.DataFrame(rows)[["item", "mean", "p05", "p50", "min"]].to_string(index=False), flush=True)

    univ = p1["universe_tradable"].to_numpy(bool)
    yrs = np.asarray(p1.dates.year)
    out = []
    for h in (1, 5):
        R = {"1m执行价": p1.fwd_ret(h, "open").to_numpy(np.float32),
             "5m执行价": p5f.fwd_ret(h, "open").to_numpy(np.float32)}
        for sn, S in (("1m打分", s1), ("5m打分", s5)):
            F = S.to_numpy(np.float32)
            for rn, RR in R.items():
                M = univ & np.isfinite(RR)
                ic, _, _, _ = eval_one(F, RR, M, want_layers=False)
                ics = pd.Series(ic, index=p1.dates)
                for wn, (y0, y1) in WINDOWS.items():
                    v = ics[(yrs >= y0) & (yrs <= y1)].dropna().to_numpy()
                    out.append(dict(horizon=f"h{h}", score=sn, ret=rn, window=wn, n_days=v.size,
                                    ic_mean=float(v.mean()), ic_t_nw=_nw_t(v, max(5, 2 * h)),
                                    ic_ir=float(v.mean() / v.std(ddof=1))))
                print(f"  h{h} {sn} x {rn}: 2021-2025 IC {out[-len(WINDOWS)]['ic_mean']:.4f}", flush=True)
    pd.DataFrame(out).to_csv(LEDGER / "m5_ic.csv", index=False)


# ---------------- 阶段 4:回测 ----------------
# 审查 P1(2026-10-10 修复):5 日调仓的方案必须对**同频**池等权基准 ST.benchmark(..., rebalance=5) 算超额
# (ENGINE_IMPACT_MODEL §9 第 11 条、§8 ④b;本轮纪律 4)。旧版对每日重置的 k=1 基准,超额被再平衡频率差抬高约 0.5pp,
# 且"C−B 全为正、来自 mid4 含集合竞价价"的归因基本是 k=1 基准与 5 分钟执行价噪声的交互。
# k=1 的数字保留在 excess_k1 / bench_ann_k1 列与 m5_bridge_daily 的 benchmark_k1 列,只作对照与溯源。
def _corr_peak(rs, rb):
    cs = {k: float(rs.corr(rb.shift(k))) for k in (-2, -1, 0, 1, 2)}
    return max(cs, key=cs.get), cs


def _ann(r: pd.Series) -> float:
    """几何年化(与 strategy 的 ann_return 同式:nav[-1] ** (242/n) - 1)。"""
    r = r.dropna()
    return float(np.prod(1.0 + r.to_numpy(np.float64)) ** (TRADING_DAYS / len(r)) - 1.0)


def stage_backtest():
    p1 = Panel()
    p5f = M5.m5_panel(signal=True, exec_=True)
    s1 = pd.read_parquet(SCORE1)
    s5 = pd.read_parquet(SCORE5)
    panels = {"A 原版(1m信号+1m执行)": (s1, p1, "1m执行"), "B 5m信号+1m执行": (s5, p1, "1m执行"),
              "C 全近似(5m信号+5m执行)": (s5, p5f, "5m执行")}
    bcache = {}

    def bench(pn, pp, wn, m, k):
        key = (pn, wn, k)
        if key not in bcache:
            bcache[key] = ST.benchmark(pp, "V1000", "open10", date_mask=m, rebalance=k)
        return bcache[key]

    rows, checks, daily = [], [], {}
    # 引擎回归断言(纪律 4):全池等权 vs **同频**池等权基准,超额 ≈ -成本(两个执行面板 × 日调仓 / 5 日调仓)。
    # 5 日调仓另报两行对照:对 k=1 基准(旧口径,含再平衡频率差)与"去掉开头 4 个建仓日"(冷启动现金腿,审查 P0)。
    ones = pd.DataFrame(1.0, index=p1.dates, columns=p1.codes, dtype=np.float32)
    for pn, pp in (("1m执行", p1), ("5m执行", p5f)):
        m = year_mask(pp.dates, EVAL_Y0, EVAL_Y1)
        for k in (1, 5):
            s = ST.run_strategy(ones, pp, "V1000", "open10", scheme="exclude", exclude_frac=0.0, rebalance=k,
                                date_mask=m)
            b = bench(pn, pp, "2021-2025", m, k)
            assert s["rets"].index.equals(b["rets"].index)
            cost = s["cost_bps"] * 1e-4 * TRADING_DAYS
            kk, cs = _corr_peak(s["rets"], b["rets"])
            if kk != 0:
                raise AssertionError(f"全池等权 r{k} {pn} 与同频基准错位 k={kk}: {cs}")
            val = (s["ann_return"] - b["ann_return"] + cost) * 100
            checks.append(dict(check=f"fullpool_r{k}_vs_bench_k{k}_excess_plus_cost_pp", panel=pn, window="2021-2025",
                               value=val,
                               note=f"excess={100 * (s['ann_return'] - b['ann_return']):.3f}pp cost={100 * cost:.3f}pp "
                                    f"exposure_max<=1(引擎循环内断言) mean={s['exposure']:.4f} 峰值k={kk}"))
            checks.append(dict(check=f"fullpool_r{k}_align_peak_k", panel=pn, window="2021-2025", value=float(kk),
                               note=str({q: round(v, 4) for q, v in cs.items()})))
            print(f"  [引擎断言] {pn} 全池等权 r{k} vs 同频基准 k{k}: 超额+成本 = {val:+.3f}pp, 峰值 k={kk}, "
                  f"仓位 {s['exposure']:.4f}", flush=True)
            if k > 1:
                b1 = bench(pn, pp, "2021-2025", m, 1)
                v1 = (s["ann_return"] - b1["ann_return"] + cost) * 100
                checks.append(dict(check=f"fullpool_r{k}_vs_bench_k1_excess_plus_cost_pp(对照:旧口径)", panel=pn,
                                   window="2021-2025", value=v1, note="5 日调仓对每日重置基准:含再平衡频率差"))
                sk = M5.FWD_SKIP_HEAD
                v_sk = (_ann(s["rets"].iloc[sk:]) - _ann(b["rets"].iloc[sk:]) + cost) * 100
                checks.append(dict(check=f"fullpool_r{k}_vs_bench_k{k}_skip{sk}_excess_plus_cost_pp", panel=pn,
                                   window="2021-2025", value=v_sk,
                                   note=f"去掉 2021-01 开头 {sk} 个建仓日(相位冷启动现金腿);成本按全期日均"))
                print(f"      对照:vs k=1 基准 {v1:+.3f}pp;同频且去掉开头 {sk} 日 {v_sk:+.3f}pp", flush=True)
    for plan, kw in PLANS.items():
        kreb = int(kw["rebalance"])
        for cfg, (score, pp, pn) in panels.items():
            for wn, (y0, y1) in WINDOWS.items():
                t0 = time.time()
                m = year_mask(pp.dates, y0, y1)
                r = ST.run_strategy(score, pp, "V1000", "open10", date_mask=m, **kw)
                b = bench(pn, pp, wn, m, kreb)                  # 同频(主口径)
                b1 = bench(pn, pp, wn, m, 1)                    # 每日重置(旧口径,对照)
                assert r["rets"].index.equals(b["rets"].index), f"{plan}/{cfg}/{wn} 策略与基准索引不一致"
                assert r["rets"].index.equals(b1["rets"].index)
                k, cs = _corr_peak(r["rets"], b["rets"])
                if k != 0:
                    raise AssertionError(f"{plan}/{cfg}/{wn} 与同频基准错位 k={k}: {cs}")
                k1, _ = _corr_peak(r["rets"], b1["rets"])
                if k1 != 0:
                    raise AssertionError(f"{plan}/{cfg}/{wn} 与 k=1 基准错位 k={k1}")
                ex = ST.excess_stats(r["rets"], b["rets"])
                rows.append(dict(plan=plan, config=cfg, window=wn, bench_rebalance=kreb, ann_return=r["ann_return"],
                                 bench_ann=b["ann_return"], excess=r["ann_return"] - b["ann_return"],
                                 excess_arith=ex.get("excess_arith"), te=ex.get("te"), ir=ex.get("ir"),
                                 t_nw=ex.get("t_nw"), sharpe=r["sharpe"], max_dd=r["max_dd"],
                                 turnover=r["turnover"], cost_bps=r["cost_bps"], holdings=r["holdings"],
                                 exposure=r["exposure"], corr_peak_k=k, n_days=r["n_days"],
                                 bench_ann_k1=b1["ann_return"], excess_k1=r["ann_return"] - b1["ann_return"]))
                if wn == "2021-2025":
                    daily[(plan, cfg, "strategy")] = r["rets"]
                    daily[(plan, cfg, "benchmark")] = b["rets"]
                    daily[(plan, cfg, "benchmark_k1")] = b1["rets"]
                print(f"  {plan:16s} {cfg:22s} {wn:9s} 超额 {100 * rows[-1]['excess']:+7.2f}pp "
                      f"(策略 {100 * r['ann_return']:6.2f}% 同频基准 {100 * b['ann_return']:6.2f}%;"
                      f"对 k=1 基准 {100 * rows[-1]['excess_k1']:+6.2f}pp) "
                      f"换手 {r['turnover']:.3f} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "m5_backtest.csv", index=False)
    D = pd.DataFrame(daily)
    D.columns = ["|".join(c) for c in D.columns]
    D.to_parquet(DAILY_OUT)

    # 口径差:逐窗口差值 + 2021-2025 逐日超额差的 NW t(主口径 = 同频基准)
    diffs = []
    for plan in PLANS:
        for wn in WINDOWS:
            g = df[(df.plan == plan) & (df.window == wn)].set_index("config")
            a, b_, c = (g.at[k, "excess"] for k in CONFIGS)
            a1, b1_, c1 = (g.at[k, "excess_k1"] for k in CONFIGS)
            ba, bc = g.at[list(CONFIGS)[0], "bench_ann"], g.at[list(CONFIGS)[2], "bench_ann"]
            diffs.append(dict(plan=plan, window=wn, excess_A=a, excess_B=b_, excess_C=c,
                              B_minus_A=b_ - a, C_minus_A=c - a, C_minus_B=c - b_,
                              bench_C_minus_A=bc - ba,
                              k1_B_minus_A=b1_ - a1, k1_C_minus_A=c1 - a1, k1_C_minus_B=c1 - b1_))
        eA = D[f"{plan}|A 原版(1m信号+1m执行)|strategy"] - D[f"{plan}|A 原版(1m信号+1m执行)|benchmark"]
        eB = D[f"{plan}|B 5m信号+1m执行|strategy"] - D[f"{plan}|B 5m信号+1m执行|benchmark"]
        for lab, e, base in (("B-A", eB, eA),
                             ("C-A", D[f"{plan}|C 全近似(5m信号+5m执行)|strategy"]
                              - D[f"{plan}|C 全近似(5m信号+5m执行)|benchmark"], eA),
                             ("C-B", D[f"{plan}|C 全近似(5m信号+5m执行)|strategy"]
                              - D[f"{plan}|C 全近似(5m信号+5m执行)|benchmark"], eB)):
            d = (e - base).dropna()
            vr = ST.lr_variance_ratio(d.to_numpy(), 5)
            t = float(d.mean() / d.std() * np.sqrt(len(d)) / np.sqrt(vr))
            diffs.append(dict(plan=plan, window=f"2021-2025 逐日超额差 {lab}",
                              daily_diff_ann_arith=float(d.mean() * TRADING_DAYS),
                              daily_diff_te=float(d.std() * np.sqrt(TRADING_DAYS)), daily_diff_t_nw5=t,
                              corr_excess_with_A=float(e.corr(eA))))
    pd.DataFrame(diffs).to_csv(LEDGER / "m5_backtest_diff.csv", index=False)
    pd.DataFrame(checks).to_csv(LEDGER / "m5_engine_checks.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(diffs).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    _contslice(df, D)


def _contslice(df: pd.DataFrame, D: pd.DataFrame):
    """审查 P0 附带:主表的逐年 / 分段超额是 date_mask 冷启动(每段开头 4 天相位空仓,A/B/C 同一切法,
    口径差不受影响)。这里给"2021-2025 连续运行序列按年切片(去掉 2021 开头 4 个建仓日)"的同频超额作对照。"""
    out = []
    sk = M5.FWD_SKIP_HEAD
    for plan in PLANS:
        for cfg in CONFIGS:
            s = D[f"{plan}|{cfg}|strategy"].iloc[sk:]
            b = D[f"{plan}|{cfg}|benchmark"].iloc[sk:]
            for wn, (y0, y1) in WINDOWS.items():
                sel = (s.index.year >= y0) & (s.index.year <= y1)
                ec = _ann(s[sel]) - _ann(b[sel])
                cold = float(df[(df.plan == plan) & (df.config == cfg) & (df.window == wn)]["excess"].iloc[0])
                out.append(dict(plan=plan, config=cfg, window=wn, excess_cont_slice=ec, excess_cold_mask=cold,
                                cold_minus_cont=cold - ec, n_days=int(sel.sum())))
    O = pd.DataFrame(out)
    O.to_csv(LEDGER / "m5_backtest_contslice.csv", index=False)
    for plan in PLANS:
        g = O[O.plan == plan]
        piv = g.pivot(index="window", columns="config", values="excess_cont_slice")
        print(f"  [连续切片] {plan}: 冷启动 − 连续切片 范围 {100 * g.cold_minus_cont.min():+.2f} ~ "
              f"{100 * g.cold_minus_cont.max():+.2f}pp;连续切片 C−A 范围 "
              f"{100 * (piv[list(CONFIGS)[2]] - piv[list(CONFIGS)[0]]).min():+.2f} ~ "
              f"{100 * (piv[list(CONFIGS)[2]] - piv[list(CONFIGS)[0]]).max():+.2f}pp", flush=True)


# ---------------- 阶段 5:执行价误差 ----------------
def stage_exec():
    p1 = Panel()
    p5f = M5.m5_panel(signal=False, exec_=True)
    U = ST.universe(p1, "V1000")
    yrs = np.asarray(p1.dates.year)
    t1, t5 = p1.raw("twap_open10"), p5f.raw("twap_open10")
    a1, a5 = p1.raw("amt_open10"), p5f.raw("amt_open10")
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.log(t5 / t1) * 1e4
        ea = np.log(a5 / a1) * 1e4
    er = e.shift(-1) - e                                   # 日收益误差(回测只用相邻两日之比)
    d1 = (p1["one_word_up"] | p1["one_word_down"] | (a1 <= 0))
    d5 = (p5f["one_word_up"] | p5f["one_word_down"] | (a5 <= 0))
    rows = []
    for wn, (y0, y1) in WINDOWS.items():
        ym = pd.Series((yrs >= y0) & (yrs <= y1), index=p1.dates)
        m = U.mul(ym, axis=0).astype(bool)
        for nm, X, unit in (("twap_open10 价格 log(5m/1m)", e, "bp"), ("twap_open10 日收益误差", er, "bp"),
                            ("amt_open10 log(5m/1m)", ea, "bp")):
            v = X.where(m).to_numpy().ravel(); v = v[np.isfinite(v)]
            rows.append(dict(item=nm, window=wn, unit=unit, n=v.size, mean=float(v.mean()),
                             std=float(v.std()), abs_median=float(np.median(np.abs(v))),
                             p01=float(np.percentile(v, 1)), p05=float(np.percentile(v, 5)),
                             p50=float(np.median(v)), p95=float(np.percentile(v, 95)),
                             p99=float(np.percentile(v, 99)), abs_p99=float(np.percentile(np.abs(v), 99))))
        dd = (d1 != d5).where(m)
        rows.append(dict(item="不可成交判定不一致(一字板/开盘无量)", window=wn, unit="股票日比例",
                         n=int(m.to_numpy().sum()), mean=float(np.nanmean(dd.to_numpy(np.float64))),
                         p50=float(np.nanmean(d1.where(m).to_numpy(np.float64))),
                         p95=float(np.nanmean(d5.where(m).to_numpy(np.float64)))))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "m5_exec_error.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ---------------- 阶段 6:滚动窗口超额分布 ----------------
# 审查修复(2026-10-10):主口径 = 同频(5 日)池等权基准(P1)+ 去掉开头 FWD_SKIP_HEAD=4 个建仓日(P0),
# 与前向检验"冷启动、前 4 个信号日只建仓不计"的读法同一定义(minute5.reference_distribution)。
# 旧口径(k=1 基准、不跳过)的分位数保留在 m5_window_excess_quantiles.csv 的 bench="k1_skip0(旧)" 行作对照。
WIN_NS = (15, 17, 20, 25)    # 17 = 按 fwd_data 清单(forward_eval_start 2026-09-01、数据到 10-09)信号日 21 个 − 4;只是便利,窗口长度由前向协议定


def _win_df(D, plan, cfg, bcol="benchmark"):
    return pd.DataFrame({"strategy": D[f"{plan}|{cfg}|strategy"], "benchmark": D[f"{plan}|{cfg}|{bcol}"]})


def stage_window():
    D = pd.read_parquet(DAILY_OUT)
    sk = M5.FWD_SKIP_HEAD
    long, qrows = [], []
    for plan in PLANS:
        for cfg in CONFIGS:
            for bname, bcol, skh in (("k5_skip4", "benchmark", sk), ("k1_skip0(旧)", "benchmark_k1", 0)):
                df = _win_df(D, plan, cfg, bcol)
                for n in WIN_NS:
                    wc = M5.window_excess_distribution(df, n, how="compound", skip_head=skh)
                    ws = M5.window_excess_distribution(df, n, how="sum", skip_head=skh)
                    if cfg.startswith("C") and bname == "k5_skip4":
                        long.append(pd.DataFrame({"plan": plan, "config": cfg, "n_days": n, "end_date": wc.index,
                                                  "excess_compound": wc.to_numpy(),
                                                  "excess_sum": ws.reindex(wc.index).to_numpy()}))
                    for how, w in (("compound", wc), ("sum", ws)):
                        qrows.append(dict(plan=plan, config=cfg, bench=bname, n_days=n, how=how, n_windows=w.size,
                                          n_independent_approx=round(w.size / n, 1), mean=float(w.mean()),
                                          std=float(w.std()), frac_positive=float((w > 0).mean()),
                                          **{f"p{q:02d}": float(np.percentile(w, q))
                                             for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)}))
    L = pd.concat(long, ignore_index=True)
    L.to_csv(LEDGER / "m5_window_excess_dist.csv", index=False)
    Q = pd.DataFrame(qrows)
    Q.to_csv(LEDGER / "m5_window_excess_quantiles.csv", index=False)
    # 自检:账本分布 == minute5.reference_distribution(前向检验调用的接口)
    for plan in PLANS:
        ref = M5.reference_distribution(plan, 20)
        led = L[(L.plan == plan) & (L.n_days == 20)].set_index("end_date")["excess_compound"]
        assert np.allclose(ref.to_numpy(), led.to_numpy(), atol=0, rtol=0) and ref.index.equals(pd.DatetimeIndex(led.index))
    # 近似误差在 n 日窗口上的量级:B/C 与 A 的同窗口超额之差(主口径)
    drows = []
    for plan in PLANS:
        for n in WIN_NS:
            wa = M5.window_excess_distribution(_win_df(D, plan, "A 原版(1m信号+1m执行)"), n, skip_head=sk)
            for cfg in list(CONFIGS)[1:]:
                wx = M5.window_excess_distribution(_win_df(D, plan, cfg), n, skip_head=sk)
                d = (wx - wa).dropna()
                # 百分位漂移:同一窗口在各自分布里的百分位之差(前向检验报百分位时的近似误差)
                pa = wa.rank(pct=True) * 100; px_ = wx.reindex(wa.index).rank(pct=True) * 100
                ps = (px_ - pa).abs().dropna()
                drows.append(dict(plan=plan, config=cfg, bench="k5_skip4", n_days=n, n_windows=d.size,
                                  pct_shift_abs_median=float(ps.median()), pct_shift_abs_p90=float(ps.quantile(0.9)),
                                  mean=float(d.mean()),
                                  std=float(d.std()), abs_median=float(d.abs().median()),
                                  p05=float(d.quantile(0.05)), p95=float(d.quantile(0.95)),
                                  corr_with_A=float(wx.reindex(wa.index).corr(wa)),
                                  sign_agree=float((np.sign(wx.reindex(wa.index)) == np.sign(wa)).mean())))
        # 基准口径差的影响:同一 C 口径窗口值(同频)放进 k=1 分布 vs 同频分布的百分位差(审查 P1 的 −1.6 点)
        for n in WIN_NS:
            d5 = M5.window_excess_distribution(_win_df(D, plan, "C 全近似(5m信号+5m执行)"), n, skip_head=sk)
            d1 = M5.window_excess_distribution(_win_df(D, plan, "C 全近似(5m信号+5m执行)", "benchmark_k1"), n,
                                               skip_head=sk)
            v5, v1 = d5.to_numpy(), np.sort(d1.to_numpy())
            p_in1 = 100.0 * (np.searchsorted(v1, v5, "left") + np.searchsorted(v1, v5, "right")) / 2.0 / v1.size
            p_in5 = d5.rank(pct=True).to_numpy() * 100
            dd = p_in1 - p_in5
            drows.append(dict(plan=plan, config="C 同频窗口值查 k=1 分布 − 查同频分布", bench="k5 vs k1", n_days=n,
                              n_windows=dd.size, mean=float(dd.mean()), pct_shift_abs_median=float(np.median(np.abs(dd))),
                              pct_shift_abs_p90=float(np.percentile(np.abs(dd), 90)),
                              median_dist_k1_minus_k5=float(d1.median() - d5.median())))
    pd.DataFrame(drows).to_csv(LEDGER / "m5_window_diff.csv", index=False)
    pd.set_option("display.width", 250)
    print(Q[(Q.how == "compound")].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(pd.DataFrame(drows).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


# ---------------- 阶段 7:冷启动读法的历史验证(审查 P0)----------------
CS_STEP = 10                  # 每 10 个信号日取一个起点(窗口彼此重叠,只用于描述误差量级)


def stage_coldstart():
    """在 2021-2025 逐起点模拟前向检验的处境:从空仓冷启动(date_mask = 起点起 4+25 个信号日),
    比较三种读法与"连续运行的同一窗口":
      noskip  冷启动窗口 [s, s+n) 全部计入(旧报告的默认读法)
      skip4   冷启动后去掉前 4 个信号日,评估窗 [s+4, s+4+n)(预注册读法)
    再给前向读数的总测量误差:C 口径冷启动 skip4 vs A 口径(1 分钟真值)连续运行同一窗口。
    基准:同频(5 日)池等权;其相位平均与锚点无关、且在 mask 起点之前按日历预热,冷启动与连续运行的基准逐日相同
    (此处抽 10 个起点实算核对)。只读研究期(2021-2025)数据。"""
    p1 = Panel()
    p5f = M5.m5_panel(signal=True, exec_=True)
    s5 = pd.read_parquet(SCORE5)
    D = pd.read_parquet(DAILY_OUT)
    sk = M5.FWD_SKIP_HEAD
    cfgC, cfgA = "C 全近似(5m信号+5m执行)", "A 原版(1m信号+1m执行)"
    sig = D.index
    L = sk + max(WIN_NS)
    starts = list(range(0, len(sig) - L + 1, CS_STEP))
    # 基准核对:冷启动 mask 的同频基准 == 连续运行的同频基准(同一天)
    bchk = []
    for i in starts[::max(1, len(starts) // 10)][:10]:
        m = np.asarray(p5f.dates.isin(sig[i:i + L]))
        b = ST.benchmark(p5f, "V1000", "open10", date_mask=m, rebalance=M5.FWD_REBALANCE)["rets"]
        bchk.append(float((b - D[f"排雷缓冲20/35 r5|{cfgC}|benchmark"].reindex(b.index)).abs().max()))
    print(f"  同频基准 冷启动 vs 连续 逐日最大差 {max(bchk):.2e}(抽 {len(bchk)} 个起点)", flush=True)
    rows = []
    ex = lambda s, b: float(np.prod(1.0 + s.to_numpy()) - np.prod(1.0 + b.to_numpy()))
    t0 = time.time()
    for plan, kw in PLANS.items():
        cs_, cb = D[f"{plan}|{cfgC}|strategy"], D[f"{plan}|{cfgC}|benchmark"]
        as_, ab = D[f"{plan}|{cfgA}|strategy"], D[f"{plan}|{cfgA}|benchmark"]
        for j, i in enumerate(starts):
            win = sig[i:i + L]
            m = np.asarray(p5f.dates.isin(win))
            r = ST.run_strategy(s5, p5f, "V1000", "open10", date_mask=m, **kw)["rets"].reindex(win)
            bb = cb.reindex(win)
            for n in WIN_NS:
                a0, a1 = win[:n], win[sk:sk + n]
                rows.append(dict(plan=plan, start=win[0], n_days=n,
                                 cold_noskip=ex(r.loc[a0], bb.loc[a0]), cont_noskip=ex(cs_.loc[a0], cb.loc[a0]),
                                 cold_skip=ex(r.loc[a1], bb.loc[a1]), cont_skip=ex(cs_.loc[a1], cb.loc[a1]),
                                 contA_skip=ex(as_.loc[a1], ab.loc[a1]),
                                 bench_first4=float(np.prod(1.0 + bb.iloc[:sk].to_numpy()) - 1.0),
                                 cold_first4_mean_ret=float(r.iloc[:sk].mean()), bench_first4_mean_ret=float(bb.iloc[:sk].mean())))
            if j % 20 == 0:
                print(f"  {plan} 起点 {j + 1}/{len(starts)} {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    # 百分位:冷启动 / 连续读数放进同一条参考分布(C 主口径);总误差:C 冷启动 skip4 在 C 分布 vs A 连续在 A 分布
    for plan in PLANS:
        for n in WIN_NS:
            sel = (R.plan == plan) & (R.n_days == n)
            dC = M5.reference_distribution(plan, n, config="C")
            dA = M5.reference_distribution(plan, n, config="A")
            for col in ("cold_noskip", "cont_noskip", "cold_skip", "cont_skip"):
                R.loc[sel, f"pct_{col}"] = [M5.percentile_in(v, dC) for v in R.loc[sel, col]]
            R.loc[sel, "pctA_contA_skip"] = [M5.percentile_in(v, dA) for v in R.loc[sel, "contA_skip"]]
    R.to_csv(LEDGER / "m5_coldstart_check.csv", index=False)
    srows = []
    for (plan, n), g in R.groupby(["plan", "n_days"]):
        for lab, a, b, pa, pb in (("冷启动不跳过 − 连续(同窗)", "cold_noskip", "cont_noskip", "pct_cold_noskip", "pct_cont_noskip"),
                                  ("冷启动跳过4日 − 连续(同窗)", "cold_skip", "cont_skip", "pct_cold_skip", "pct_cont_skip"),
                                  ("前向总误差:C冷启动跳过4日 − A连续(同窗)", "cold_skip", "contA_skip", "pct_cold_skip", "pctA_contA_skip")):
            d = (g[a] - g[b]) * 100
            pdiff = (g[pa] - g[pb]).abs()
            slope = float(np.polyfit(g["bench_first4"], g[a] - g[b], 1)[0])
            srows.append(dict(plan=plan, n_days=n, compare=lab, n_starts=len(g), diff_mean_pp=float(d.mean()),
                              diff_std_pp=float(d.std()), diff_abs_median_pp=float(d.abs().median()),
                              diff_abs_max_pp=float(d.abs().max()),
                              corr_with_bench_first4=float(np.corrcoef(g["bench_first4"], g[a] - g[b])[0, 1]),
                              slope_on_bench_first4=slope,
                              pct_shift_abs_median=float(pdiff.median()), pct_shift_abs_p90=float(pdiff.quantile(0.9)),
                              pct_shift_abs_max=float(pdiff.max()),
                              std_a_pp=float(g[a].std() * 100), std_b_pp=float(g[b].std() * 100)))
    S = pd.DataFrame(srows)
    S.to_csv(LEDGER / "m5_coldstart_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


# ---------------- 阶段 8:baostock 第二层误差蒙特卡罗(审查 P2)----------------
# 第 5 节的桥接用 1 分钟精确聚合的 5 分钟 bar,不含 baostock 数据本身的误差(上交所 bar 边界 / 15:00 bar 缺收盘集合竞价)。
# 这里把 2025-12 30 只股票核对得到的上交所逐股票日误差(日线修正后,bs5r − ours5;scripts/m5_baostock_check.py --pairs)
# 按经验分布整行重抽样(保留同一股票日各原料误差的相关),加到 5 分钟面板全部上交所股票日上,重算 COMP_v5 打分,
# 看打分秩相关 / Top20 重合 / 20 日窗口误差在第一层之上还多多少。
#   加性原料(截面 sd 单位,sd = 当日 V1000 池内该 5 分钟原料的截面标准差):rvol、gap_share、smart_money、amt_gini、
#     amt_time_wgt、amt_first30_share、amt_openauc_share;乘性(log 误差):vwap、twap_open10、amt_open10。
#   不扰动:close_px / amount / open_auction_px / ret_intraday(日线修正后为 0)、rng(只用于一字板判定,平价 bar 两边相同)、
#     semibeta_mp(需全市场,未测;压力档用 gap_share 的 z 作代理)。
#   假设:误差逐股票日独立(样本内滞后 1 日自相关见输出);样本只有 11 只上交所股票 × 23 天 = 253 个股票日。
L2_PAIRS = TMP / "m5_bs_feature_pairs.parquet"
L2_DIR = TMP / "m5_l2mc"
L2_ADD = {"rvol": "rvol", "gap_share": "gap_share", "smart_money": "smart_money", "amt_gini": "amt_gini",
          "amt_time_wgt": "amt_time_wgt", "amt_first30_share": "amt_first30_share",
          "amt_openauc_share": "amt_openauc_share__bar0"}
L2_MUL = {"vwap": "vwap__mid_h", "twap_open10": "twap_open10__mid4", "amt_open10": "amt_open10__sum2"}
L2_CLIP01 = {"gap_share", "amt_gini", "amt_first30_share", "amt_openauc_share"}
L2_REPS = ((1.0, 101, False), (1.0, 202, False), (1.0, 303, False), (2.0, 404, True))   # (误差倍数, 种子, 是否扰动 semibeta 代理)


def _comp_score(panel, members, signs):
    z = {}
    u = panel["universe"]
    for n in members:
        f = FA.REGISTRY[n][1](panel).reindex(index=panel.dates, columns=panel.codes).replace([np.inf, -np.inf], np.nan)
        z[n] = (ops.standardize(f, u) * float(np.sign(signs.get(n, 1.0)))).astype(np.float32)
        del f
    return combine.combine_equal(z)


def stage_layer2():
    pairs = pd.read_parquet(L2_PAIRS)
    sh = pairs[pairs.code.str.endswith(".SH")].copy()
    p1 = Panel()
    U = ST.universe(p1, "V1000")
    Uv = U.reindex(index=p1.dates, columns=p1.codes).fillna(False).to_numpy(bool)
    codes = p1.codes
    is_sh = np.array([c.endswith(".SH") for c in codes])

    def cs_sd(X):                               # 逐日 V1000 池截面 sd(池内不足 200 只时用全部有限值)
        A = X.to_numpy(np.float64)
        M = np.where(Uv, A, np.nan)
        sd = np.nanstd(M, axis=1)
        n = np.isfinite(M).sum(axis=1)
        sd_all = np.nanstd(A, axis=1)
        return np.where(n >= 200, sd, sd_all)

    base = {f: pd.read_parquet(M5.M5_DIR / f"{f}.parquet").reindex(index=p1.dates, columns=codes)
            for f in list(L2_ADD) + list(L2_MUL) + ["semibeta_mp"]}
    sd = {f: cs_sd(base[f]) for f in list(L2_ADD) + ["semibeta_mp"]}
    # 经验误差矩阵(上交所股票日 × 原料):加性 = (bs5r − ours5) / 当日 sd;乘性 = log(bs5r / ours5)
    di = {d: i for i, d in enumerate(p1.dates)}
    rows_t = np.array([di[pd.Timestamp(d)] for d in sh["date"]])
    Z = {}
    for f, k in L2_ADD.items():
        Z[f] = ((sh[f"{k}|bs5r"] - sh[f"{k}|ours5"]).to_numpy() / sd[f][rows_t])
    for f, k in L2_MUL.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            Z[f] = np.log(sh[f"{k}|bs5r"].to_numpy() / sh[f"{k}|ours5"].to_numpy())
    Z = pd.DataFrame(Z).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    zinfo = []
    for f in Z.columns:
        v = Z[f]
        ac = sh.assign(z=v.to_numpy()).sort_values(["code", "date"]).groupby("code")["z"].apply(
            lambda x: x.autocorr(1) if x.std() > 0 else np.nan).mean()
        zinfo.append(dict(field=f, unit="截面sd" if f in L2_ADD else "log", mean=float(v.mean()), std=float(v.std()),
                          abs_median=float(v.abs().median()), abs_p90=float(v.abs().quantile(0.9)), lag1_autocorr=float(ac)))
    print(pd.DataFrame(zinfo).to_string(index=False, float_format=lambda x: f"{x:+.4f}"), flush=True)

    members, signs, _ = load_members()
    s1 = pd.read_parquet(SCORE1).reindex(index=p1.dates, columns=codes)
    s5 = pd.read_parquet(SCORE5).reindex(index=p1.dates, columns=codes)
    D = pd.read_parquet(DAILY_OUT)
    sk = M5.FWD_SKIP_HEAD
    L2_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    T, N = len(p1.dates), len(codes)
    sh_idx = np.flatnonzero(is_sh)
    for scale, seed, semi in L2_REPS:
        t0 = time.time()
        rng_ = np.random.default_rng(seed)
        pick = rng_.integers(0, len(Z), size=(T, sh_idx.size))       # 每个上交所股票日抽一整行误差
        for f in list(M5.M5_SIGNAL_FIELDS) + list(M5.M5_EXEC_FIELDS):
            dst = L2_DIR / f"{f}.parquet"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            pert = f in L2_ADD or f in L2_MUL or (semi and f == "semibeta_mp")
            if not pert:
                dst.symlink_to(M5.M5_DIR / f"{f}.parquet")
                continue
            A = base[f].to_numpy(np.float64).copy()
            zf = Z["gap_share" if f == "semibeta_mp" else f].to_numpy()[pick] * scale     # (T, n_sh)
            sub = A[:, sh_idx]
            if f in L2_MUL:
                sub = sub * np.exp(zf)
            else:
                sub = sub + zf * sd[f][:, None]
                if f in L2_CLIP01:
                    sub = np.clip(sub, 0.0, 1.0)
                elif f == "rvol":
                    sub = np.clip(sub, 0.0, None)
            A[:, sh_idx] = np.where(np.isfinite(A[:, sh_idx]), sub, A[:, sh_idx])
            pd.DataFrame(A.astype(np.float32), index=p1.dates, columns=codes).to_parquet(dst)
        pmc = M5.m5_panel(signal=True, exec_=True, m5_dir=L2_DIR)
        smc = _comp_score(pmc, members, signs).reindex(index=p1.dates, columns=codes)
        rec = dict(scale=scale, seed=seed, semibeta_proxy=semi)
        for lab, a, b in (("1m_vs_5mL2", s1, smc), ("5m_vs_5mL2", s5, smc), ("1m_vs_5m", s1, s5)):
            c = summarize(cs_spearman(a, b, U))
            rec[f"rankcorr_{lab}_mean"] = c["mean"]; rec[f"rankcorr_{lab}_p05"] = c["p05"]; rec[f"rankcorr_{lab}_min"] = c["min"]
            rec[f"top20_overlap_{lab}"] = summarize(_topk_overlap(a, b, U, 20))["mean"]
            rec[f"bottom20_jaccard_{lab}"] = summarize(_topk_overlap(a, b, U, bottom_frac=0.20))["mean"]
        m = year_mask(pmc.dates, EVAL_Y0, EVAL_Y1)
        bmc = ST.benchmark(pmc, "V1000", "open10", date_mask=m, rebalance=5)["rets"]
        for plan, kw in PLANS.items():
            r = ST.run_strategy(smc, pmc, "V1000", "open10", date_mask=m, **kw)
            assert r["rets"].index.equals(bmc.index)
            k, cs = _corr_peak(r["rets"], bmc)
            if k != 0:
                raise AssertionError(f"layer2 {plan} 错位 k={k}")
            dfL = pd.DataFrame({"strategy": r["rets"], "benchmark": bmc})
            rec[f"{plan}|excess"] = _ann(r["rets"]) - _ann(bmc)
            dfA = _win_df(D, plan, "A 原版(1m信号+1m执行)")
            dfC = _win_df(D, plan, "C 全近似(5m信号+5m执行)")
            rec[f"{plan}|excess_A"] = _ann(dfA["strategy"]) - _ann(dfA["benchmark"])
            rec[f"{plan}|excess_C"] = _ann(dfC["strategy"]) - _ann(dfC["benchmark"])
            for n in (20,):
                wa = M5.window_excess_distribution(dfA, n, skip_head=sk)
                for lab, dfx in (("C", dfC), ("CL2", dfL)):
                    wx = M5.window_excess_distribution(dfx, n, skip_head=sk).reindex(wa.index)
                    d = (wx - wa).dropna()
                    ps = (wx.rank(pct=True) * 100 - wa.rank(pct=True) * 100).abs().dropna()
                    rec[f"{plan}|{lab}-A|w{n}_std_pp"] = float(d.std() * 100)
                    rec[f"{plan}|{lab}-A|w{n}_sign_agree"] = float((np.sign(wx) == np.sign(wa)).mean())
                    rec[f"{plan}|{lab}-A|w{n}_pct_shift_med"] = float(ps.median())
                    rec[f"{plan}|{lab}-A|w{n}_pct_shift_p90"] = float(ps.quantile(0.9))
        out.append(rec)
        print(f"  [layer2] 倍数 {scale} 种子 {seed}: 打分秩相关 1m-vs-5mL2 {rec['rankcorr_1m_vs_5mL2_mean']:.4f} "
              f"(无第二层 {rec['rankcorr_1m_vs_5m_mean']:.4f});Top20 重合 {rec['top20_overlap_1m_vs_5mL2']:.3f} "
              f"(无 {rec['top20_overlap_1m_vs_5m']:.3f});Top20 20日窗口 std "
              f"{rec['Top20 周频|CL2-A|w20_std_pp']:.2f}pp (无 {rec['Top20 周频|C-A|w20_std_pp']:.2f});"
              f"排雷 {rec['排雷缓冲20/35 r5|CL2-A|w20_std_pp']:.3f}pp (无 {rec['排雷缓冲20/35 r5|C-A|w20_std_pp']:.3f})"
              f"  {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(out).to_csv(LEDGER / "m5_layer2_mc.csv", index=False)
    pd.DataFrame(zinfo).to_csv(LEDGER / "m5_layer2_err_dist.csv", index=False)


# ---------------- 阶段 9:北交所对 semibeta_mp 市场均值的影响(审查 P2)----------------
def stage_bj():
    """研究期 72 个月原始 1 分钟文件里的 .BJ 列数(只读 parquet schema),以及含 .BJ 的月份里
    "市场均值去掉北交所"对 5 分钟 semibeta_mp 的影响(V1000 池非北交所股票逐日截面秩相关)。
    前向:baostock 不含北交所,前向的市场均值不含 .BJ —— 与研究期 .BJ 数为 0 的月份同口径。"""
    import pyarrow.parquet as pq
    from afactor.config import RAW, raw_months, load_raw_minute, N_BARS
    from afactor.build_daily_5m import _day_features_5m
    months = [m for m in raw_months("close_adj") if "202001" <= m <= "202512"]
    rows = []
    for m in months:
        f = RAW / "close_adj" / f"{m}_close_adj.parquet"
        cols = pq.read_schema(f).names if f.exists() else list(load_raw_minute("close_adj", m).columns)
        rows.append(dict(item="month_bj_columns", month=m, value=float(sum(str(c).endswith(".BJ") for c in cols)),
                         n_columns=len(cols)))
    M = pd.DataFrame(rows)
    with_bj = M[M.value > 0].month.tolist()
    print(f"  研究期 {len(M)} 个月,含 .BJ 列的月份: {with_bj}", flush=True)
    p1 = Panel()
    U = ST.universe(p1, "V1000")
    for m in with_bj:
        C = load_raw_minute("close_adj", m); O = load_raw_minute("open_adj", m); V = load_raw_minute("amt", m)
        codes = np.asarray(C.columns.astype(str))
        isbj = np.array([c.endswith(".BJ") for c in codes])
        days = C.index.normalize()
        for d in pd.unique(days):
            sl = np.asarray(days == d)
            if sl.sum() != N_BARS or pd.Timestamp(d) not in U.index:
                continue
            P5, O5, V5 = M5.aggregate_day(*(X.to_numpy(np.float64)[sl] for X in (C, O, V)))
            f_all = _day_features_5m(P5, O5, V5)["semibeta_mp"]
            f_no = _day_features_5m(P5[:, ~isbj], O5[:, ~isbj], V5[:, ~isbj])["semibeta_mp"]
            a = pd.Series(f_all[~isbj], index=codes[~isbj]); b = pd.Series(f_no, index=codes[~isbj])
            u = U.loc[pd.Timestamp(d)].reindex(a.index).fillna(False).to_numpy(bool)
            ok = u & np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
            rc = float(a[ok].rank().corr(b[ok].rank()))
            rows.append(dict(item="semibeta_mp_5m_rankcorr_noBJ_vs_withBJ", month=m, date=pd.Timestamp(d),
                             value=rc, n_columns=int(ok.sum()), n_bj=int(isbj.sum())))
    R = pd.DataFrame(rows)
    R.to_csv(LEDGER / "m5_bj_check.csv", index=False)
    rr = R[R.item.str.startswith("semibeta")]
    if len(rr):
        print(f"  semibeta_mp(5m)去掉北交所 vs 含北交所 截面秩相关:均值 {rr.value.mean():.5f} 最低 {rr.value.min():.5f}"
              f"({len(rr)} 天)", flush=True)


STAGES = {"factors": stage_factors, "raw": stage_raw, "ic": stage_ic, "backtest": stage_backtest,
          "exec": stage_exec, "window": stage_window, "coldstart": stage_coldstart, "layer2": stage_layer2,
          "bj": stage_bj}

if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    if todo == ["all"]:
        todo = list(STAGES)
    for s in todo:
        t0 = time.time()
        print(f"\n===== 阶段 {s} =====", flush=True)
        STAGES[s]()
        print(f"===== 阶段 {s} 完成 {time.time() - t0:.0f}s =====", flush=True)
