"""INC-T3 审查修复:A 层"确切解禁日事前可知"假设的敏感性,追加承诺贡献,按来源拆解(审查后新增,看到收益之后)。

背景(审查发现 P1 / P2-2):预注册 G4 只验证了"锁定期条款在发行时已公开",没有验证"确切解禁日在 D-10 已知"。
本脚本把这个假设拆开量化,全部是**稳健性描述**,不改主格、不改 §8.2 判定:

  gaps    首发 ratio_plan>=10% 批次的解禁日与"上市日 + 6 的整数倍个月"规则日的偏离分布(日历日 + 交易日),
          以及新浪抽样(公告日期 + k 个月,±7 日)对首发 / 定增 / 追加承诺 / 其他的同口径对照。
          产出 inc_t3_review_ruledate.csv(摘要)、inc_t3_review_ipo_ruledate.csv.gz(首发事件级分类)。
  sens    同一引擎 / 同一口径(X=10%、N_pre=10、M=0、Y=1、1000 万、5 日调仓 5 相位)下的情景:
            main        复现主格(必须与 inc_t3_overlay_grid 的 X10_N10_M0 逐位一致)
            irrB        规则日不一致的首发批次降为 B 层(K = D-3)
            ruleD       同上批次,改为"只知道规则日 R":剔除 [R-10, R-1],另加 B 层 [D-3, D-1]
            strictIPO   只有规则日一致的首发保留 A 层,其余(定增 / 追加承诺 / 其他 A / 不规则首发)全部降为 B 层
            calib       首发按事件级分类降 B 层;定增、追加承诺按新浪抽样的"规则日不一致比例"随机降 B 层(R=20 次),
                        其他 A(股权分置)全部降 B 层
            noZhuijia   去掉追加承诺事件(审查 P2-2:现有追加承诺事件对增量的贡献)
            onlyIPO / onlyIPOreg / onlyIPOirr / onlyDZ / onlyZJ / onlyOther   按来源拆解(各自单独剔除)
          产出 inc_t3_review_sens.csv;日收益 cache/_tmp/inc_t3_review_sens_daily.parquet。
  events  首发 >=10% 在池事件按"规则日一致 / 不一致(提前 / 晚 8-30 日 / 晚 >30 日)"分组的 D 对齐 CAR(B1,D 聚类),
          描述性:不规则批次的解禁前漂移是否也出现在 [-20,-4](若日期事前不可知,D 对齐的 [-20,-4] 不应有漂移)。
          产出 inc_t3_review_regular_car.csv。

用法  python3 scripts/inc_t3_review_sens.py gaps | sens | events
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t3_common as C                                   # noqa: E402

KS = tuple(range(6, 61, 6))             # 规则日 = 锚点日期 + k 个月
TOL = (0, 7)                            # "规则日一致":解禁日落在规则日之后 0~7 个日历日(顺延到交易日)
OUT_RULE = C.LEDGER / "inc_t3_review_ruledate.csv"
OUT_IPO = C.LEDGER / "inc_t3_review_ipo_ruledate.csv.gz"
OUT_SENS = C.LEDGER / "inc_t3_review_sens.csv"
OUT_CAR = C.LEDGER / "inc_t3_review_regular_car.csv"
SENS_DAILY = C.TMP / "inc_t3_review_sens_daily.parquet"


# ====================================================================== 规则日
def nearest_rule_gap(anchor, free_date, ks=KS):
    """anchor + k 个月(k∈ks)里与 free_date 最近的规则日。返回 (gap_日历日 = free_date - R, k, R)。anchor 缺失 -> (nan, nan, NaT)。"""
    if pd.isna(anchor):
        return np.nan, np.nan, pd.NaT
    best = None
    for k in ks:
        r = anchor + pd.DateOffset(months=k)
        g = (free_date - r).days
        if best is None or abs(g) < abs(best[0]):
            best = (g, k, r)
    return best


def gap_bin(g):
    if pd.isna(g):
        return "无锚点日期"
    if g < -7:
        return "提前>7日"
    if g < 0:
        return "提前1-7日"
    if g <= 7:
        return "规则日后0-7日"
    if g <= 30:
        return "晚8-30日"
    return "晚>30日"


def is_regular(g) -> bool:
    return bool(pd.notna(g) and TOL[0] <= g <= TOL[1])


def classify_ipo(ev: pd.DataFrame) -> pd.DataFrame:
    """首发(含公开发行)且 ratio_plan >= 10% 的批次 -> 规则日分类(锚点 = 上市日)。"""
    e = ev[(ev["tg"] == "首发") & (ev["ratio_plan"] >= 0.10)].copy()
    res = [nearest_rule_gap(a, f) for a, f in zip(e["list_date"], e["FREE_DATE"])]
    e["gap_cal"] = [r[0] for r in res]
    e["k_months"] = [r[1] for r in res]
    e["R_date"] = [r[2] for r in res]
    e["gap_bin"] = [gap_bin(g) for g in e["gap_cal"]]
    e["regular"] = [is_regular(g) for g in e["gap_cal"]]
    cal = C.long_calendar()
    cal = cal[cal <= C.CUT].to_numpy()
    pf = np.searchsorted(cal, e["FREE_DATE"].to_numpy(dtype="datetime64[ns]"), side="left")
    pr = np.searchsorted(cal, e["R_date"].to_numpy(dtype="datetime64[ns]"), side="left")
    e["gap_td"] = np.where(e["R_date"].notna(), pf - pr, np.nan)
    return e


def sina_regular(sample: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """新浪抽样:公告日期(= 发行 / 上市公告日,与上市日相差 -5~+7 日)+ k 个月 vs 解禁日。只用日期匹配的批次。"""
    s = sample[sample["s_announce"].notna() & (sample["date_match"] == True)].copy()      # noqa: E712
    res = [nearest_rule_gap(a, f) for a, f in zip(s["s_announce"], s["FREE_DATE"])]
    s["gap_cal"] = [r[0] for r in res]
    s["gap_bin"] = [gap_bin(g) for g in s["gap_cal"]]
    s["reg_0_7"] = [is_regular(g) for g in s["gap_cal"]]
    s["reg_pm7"] = [bool(pd.notna(g) and -7 <= g <= 7) for g in s["gap_cal"]]
    return s


def cmd_gaps():
    ev = C.load_events()
    e = classify_ipo(ev)
    rows = []

    def add(sample, bins: pd.Series, n_total, extra=None):
        for b in ["提前>7日", "提前1-7日", "规则日后0-7日", "晚8-30日", "晚>30日", "无锚点日期"]:
            n = int((bins == b).sum())
            rows.append(dict(sample=sample, item="bin:" + b, n=n, n_total=n_total, share=n / max(n_total, 1)))

    for name, m in (("events_首发ge10_全部", np.ones(len(e), bool)), ("events_首发ge10_在池", e["in_pool"].to_numpy(bool))):
        x = e[m]
        add(name, x["gap_bin"], len(x))
        rows.append(dict(sample=name, item="regular_0_7", n=int(x["regular"].sum()), n_total=len(x), share=float(x["regular"].mean())))
        pm = x["gap_cal"].between(-7, 7)
        rows.append(dict(sample=name, item="regular_pm7", n=int(pm.sum()), n_total=len(x), share=float(pm.mean())))
        td = x["gap_td"]
        rows.append(dict(sample=name, item="td_gap==0(恰为周年日后首个交易日)", n=int((td == 0).sum()), n_total=len(x), share=float((td == 0).mean())))
        rows.append(dict(sample=name, item="td_gap in [0,2]", n=int(td.between(0, 2).sum()), n_total=len(x), share=float(td.between(0, 2).mean())))
        for k, g in x.groupby("k_months"):
            rows.append(dict(sample=name, item=f"k={int(k)}月:规则日一致份额", n=int(g["regular"].sum()), n_total=len(g), share=float(g["regular"].mean())))
    # 新浪抽样(与事件表同口径的相对关系;样本按类型分层抽取,不是 >=10% 子样本)
    sample = pd.read_csv(C.LEDGER / "inc_t3_sina_announce_sample.csv", parse_dates=["FREE_DATE", "D", "s_announce"])
    s = sina_regular(sample, ev)
    for tg, g in s.groupby("tg"):
        name = f"sina抽样_{tg}"
        add(name, g["gap_bin"], len(g))
        rows.append(dict(sample=name, item="regular_0_7", n=int(g["reg_0_7"].sum()), n_total=len(g), share=float(g["reg_0_7"].mean())))
        rows.append(dict(sample=name, item="regular_pm7", n=int(g["reg_pm7"].sum()), n_total=len(g), share=float(g["reg_pm7"].mean())))
    # G4 只证明"条款公开":新浪"公告日期"是发行 / 上市公告日,不是该批确切解禁日确定的时间(审查 P1 证据 1)
    s = s.merge(ev[["event_idx", "list_date"]], on="event_idx", how="left")
    s["lead"] = (s["FREE_DATE"] - s["s_announce"]).dt.days
    for tg, g in s.groupby("tg"):
        name = f"sina抽样_{tg}"
        rows.append(dict(sample=name, item="lead_days_median(解禁日 - 公告日期)", n=len(g), n_total=len(g), share=float(g["lead"].median())))
        if tg == "首发":
            dl = (g["list_date"] - g["s_announce"]).dt.days
            rows.append(dict(sample=name, item="上市日 - 公告日期 in [0,5] 日", n=int(dl.between(0, 5).sum()), n_total=len(g),
                             share=float(dl.between(0, 5).mean())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT_RULE, index=False)
    keep = ["event_idx", "code", "list_date", "FREE_DATE", "d_pos", "k_months", "R_date", "gap_cal", "gap_td", "gap_bin",
            "regular", "in_pool", "ratio_plan", "first_type"]
    e[keep].to_csv(OUT_IPO, index=False, compression="gzip")
    pd.set_option("display.width", 200)
    print(R[R["item"].isin(["regular_0_7", "regular_pm7", "td_gap==0(恰为周年日后首个交易日)", "td_gap in [0,2]"])
            | R["item"].str.startswith("bin:")].round(4).to_string())
    print("首发 ge10 批次数", len(e), "在池", int(e["in_pool"].sum()), "无上市日", int(e["list_date"].isna().sum()))


# ====================================================================== 情景
def calib_rates() -> dict:
    """定增 / 追加承诺的"规则日不一致比例" = 1 - 新浪抽样 ±7 日一致份额(摘要账本;读取前由 gaps 写入)。"""
    R = pd.read_csv(OUT_RULE)
    out = {}
    for tg in ("定增", "追加承诺"):
        r = R[(R["sample"] == f"sina抽样_{tg}") & (R["item"] == "regular_pm7")].iloc[0]
        out[tg] = 1.0 - float(r["share"])
    return out


def scenario_windows(OV, ev: pd.DataFrame, T: int, name: str, ipo: pd.DataFrame, rng=None, rates=None, dates=None) -> pd.DataFrame:
    """返回该情景的剔除窗口表(j, start, end)。X=10%、N_pre=10、M=0。"""
    e = ev[ev["ratio_plan"] >= 0.10].copy()
    reg = e["event_idx"].map(ipo.set_index("event_idx")["regular"]).eq(True).to_numpy(bool)
    is_ipo = (e["tg"] == "首发").to_numpy()
    irr = is_ipo & ~reg
    tg = e["tg"].to_numpy()
    tier = e["tier"].to_numpy().copy()
    extra = None
    if name == "main":
        pass
    elif name == "irrB":
        tier[irr] = "B"
    elif name == "ruleD":
        tier[irr] = "B"
        sel = e[irr].merge(ipo[["event_idx", "R_date"]], on="event_idx", how="left")
        rp = np.searchsorted(dates.to_numpy(), sel["R_date"].to_numpy(dtype="datetime64[ns]"), side="left")
        ok = sel["R_date"].notna().to_numpy() & (rp < T)
        s0 = np.maximum(rp[ok] - 10, 0)
        t0 = np.minimum(rp[ok] - 1, T - 1)
        extra = pd.DataFrame(dict(j=sel["j"].to_numpy()[ok].astype(np.int64), start=s0, end=t0))
        extra = extra[extra["end"] >= extra["start"]]
    elif name == "strictIPO":
        tier[~(is_ipo & reg)] = "B"
    elif name == "calib":
        tier[irr] = "B"
        tier[(tg == "其他") & (tier == "A")] = "B"
        for g, p in rates.items():
            m = tg == g
            tier[m & (rng.random(len(e)) < p)] = "B"
    elif name == "noZhuijia":
        e = e[tg != "追加承诺"]
        tier = e["tier"].to_numpy().copy()
    elif name == "onlyIPO":
        e = e[is_ipo]; tier = e["tier"].to_numpy().copy()
    elif name == "onlyIPOreg":
        e = e[is_ipo & reg]; tier = e["tier"].to_numpy().copy()
    elif name == "onlyIPOirr":
        e = e[irr]; tier = e["tier"].to_numpy().copy()
    elif name == "onlyDZ":
        e = e[tg == "定增"]; tier = e["tier"].to_numpy().copy()
    elif name == "onlyZJ":
        e = e[tg == "追加承诺"]; tier = e["tier"].to_numpy().copy()
    elif name == "onlyOther":
        e = e[~np.isin(tg, ["首发", "定增", "追加承诺"])]; tier = e["tier"].to_numpy().copy()
    else:
        raise ValueError(name)
    e = e.copy()
    e["tier"] = tier
    win = OV.exclusion_windows(e, 0.10, 10, 0, T, "main")[["j", "start", "end"]]
    if extra is not None and len(extra):
        win = pd.concat([win, extra], ignore_index=True)
    return win


DESC = {
    "main": "主格复现(X10 N10 M0,A 层 K=D-60 / B 层 K=D-3)",
    "irrB": "规则日不一致的首发批次降为 B 层",
    "ruleD": "不一致首发批次:只知规则日 R,剔除 [R-10,R-1] 加 B 层 [D-3,D-1]",
    "strictIPO": "只有规则日一致的首发保留 A 层,其余全部降 B 层",
    "calib": "校准情景:不一致首发降 B;定增 / 追加承诺按新浪抽样不一致比例随机降 B;其他 A 全降 B(R=20 次)",
    "noZhuijia": "去掉追加承诺事件",
    "onlyIPO": "仅首发事件",
    "onlyIPOreg": "仅规则日一致的首发事件",
    "onlyIPOirr": "仅规则日不一致的首发事件",
    "onlyDZ": "仅定增事件",
    "onlyZJ": "仅追加承诺事件",
    "onlyOther": "仅其他类型事件(股权分置 / 股权激励 / 其他)",
}
SCEN = ["main", "irrB", "ruleD", "strictIPO", "noZhuijia", "onlyIPO", "onlyIPOreg", "onlyIPOirr", "onlyDZ", "onlyZJ", "onlyOther"]
R_CALIB = 20


def cmd_sens():
    import inc_t3_overlay as OV
    import inc_t3_analyze as AN
    import cap_capacity_curve as CAP
    t0 = time.time()
    X = OV.Ctx()
    ev = X.ev
    ipo = pd.read_csv(OUT_IPO, parse_dates=["list_date", "FREE_DATE", "R_date"])
    rates = calib_rates()
    print("校准比例(规则日不一致):", rates, flush=True)
    base = pd.read_parquet(C.TMP / "inc_t3_base_daily.parquet")
    b = base[(base["variant"] == "Y1.0") & (base["aum"] == 1e7)].set_index("date").sort_index()
    B = pd.read_parquet(CAP.BENCH_CACHE.format(OV.MODE))
    grid = pd.read_csv(C.LEDGER / "inc_t3_overlay_grid.csv")
    gm = grid[(grid["cell"] == "X10_N10_M0") & (grid["variant"] == "Y1.0") & (grid["aum"] == 1e7)].iloc[0]

    rows, daily = [], []

    def run_one(name, rng=None, rep=None):
        t1 = time.time()
        win = scenario_windows(OV, ev, X.T, name, ipo, rng=rng, rates=rates, dates=X.p.dates)
        E = OV.panel_from_windows(win, X.T, X.N)
        W = X.weights(E)
        res, d, chk = X.run(W, "Y1.0", OV.AUM)
        dd = (d["ret"] - b["ret"].reindex(d.index)).dropna()
        rec = dict(scenario=name if rep is None else f"{name}_r{rep:02d}", kind="scenario" if rep is None else "draw",
                   desc=DESC[name], n_windows=len(win),
                   pool_share_excluded=float((X.U & E)[X.m].sum() / max(X.U[X.m].sum(), 1)))
        rec.update(AN.diff_stats(dd))
        rec["quarterly_t"], rec["n_quarters"] = AN.quarterly_t(dd)
        for y in range(2021, 2026):
            rec[f"ann_{y}"] = float(dd[dd.index.year == y].mean() * AN.TD)
        o_ = AN.arm_stats(d, B, "over")
        rec["d_turnover_ann"] = o_["over_turnover_ann"] - float(gm["base_turnover_ann"])
        rec["d_cost_total_bp"] = o_["over_cost_total_bp"] - float(gm["base_cost_total_bp"])
        rec["over_holdings"] = o_["over_holdings"]
        rec["over_max_dd"] = o_["over_max_dd"]
        for w in AN.YW:                        # 叠加臂对同频基准的超额,及相对基线臂的差(基线臂数字取自网格主格行,各情景共用)
            for kind in ("arith", "geo"):
                rec[f"over_excess_{kind}_{w}"] = o_[f"over_excess_{kind}_{w}"]
                rec[f"d_excess_{kind}_{w}"] = o_[f"over_excess_{kind}_{w}"] - float(gm[f"base_excess_{kind}_{w}"])
        rows.append(rec)
        daily.append(d[["ret"]].assign(scenario=rec["scenario"]).reset_index())
        print(f"  {rec['scenario']:<14} full {rec['ann_2021-2025']*100:+.3f}pp (t {rec['t_2021-2025']:+.2f}) dev {rec['ann_2021-2023']*100:+.3f} "
              f"test {rec['ann_2024-2025']*100:+.3f} pool_excl {rec['pool_share_excluded']:.5f} [{time.time()-t1:.0f}s]", flush=True)
        return rec

    for name in SCEN:
        run_one(name)
    main_rec = next(r for r in rows if r["scenario"] == "main")
    dmain = abs(main_rec["ann_2021-2025"] - float(gm["ann_2021-2025"]))
    print(f"main 复现 vs inc_t3_overlay_grid:差 {dmain:.2e}", flush=True)
    assert dmain < 1e-9, "主格未能逐位复现"
    for r in range(R_CALIB):
        run_one("calib", rng=np.random.default_rng([C.SEED, 34, r]), rep=r)
    S = pd.DataFrame(rows)
    dr = S[S["kind"] == "draw"]
    agg = {"scenario": "calib", "kind": "scenario", "desc": DESC["calib"], "n_windows": float(dr["n_windows"].mean())}
    for c in S.columns:
        if c not in agg and c not in ("scenario", "kind", "desc") and S[c].dtype != object:
            agg[c] = float(dr[c].mean())
    agg["draw_min_full"], agg["draw_max_full"] = float(dr["ann_2021-2025"].min()), float(dr["ann_2021-2025"].max())
    agg["draw_min_test"], agg["draw_max_test"] = float(dr["ann_2024-2025"].min()), float(dr["ann_2024-2025"].max())
    agg["note"] = f"R={R_CALIB} 次均值;校准比例 定增 {rates['定增']:.3f} / 追加承诺 {rates['追加承诺']:.3f}"
    S = pd.concat([S[S["kind"] == "scenario"], pd.DataFrame([agg]), dr], ignore_index=True)
    S["main_cell_ann_2021-2025"] = float(gm["ann_2021-2025"])
    # 按预注册 §8.2 的条件 ①②③ 逐情景套用(④ 网格 / ⑤ 安慰剂只对主格做过,这里不重算;calib 行用 20 次均值)
    S["c1_both_windows_pos"] = (S["ann_2021-2023"] > 0) & (S["ann_2024-2025"] > 0)
    S["c2_t_ge_2"] = S["t_2021-2025"] >= 2.0
    S["c3_ge_0p20pp"] = S["ann_2021-2025"] >= 0.0020
    no_inc = (S["ann_2021-2025"] <= 0.0010) | ((S["ann_2021-2023"] <= 0) & (S["ann_2024-2025"] <= 0)) | (S["t_2021-2025"] <= 0)
    S["verdict_123"] = np.where(no_inc, "无增量", np.where(S["c1_both_windows_pos"] & S["c2_t_ge_2"] & S["c3_ge_0p20pp"], "有增量", "证据不足"))
    S.loc[S["scenario"].str.startswith("only"), "verdict_123"] = "n/a(来源拆解行,不是规则设定)"
    S.to_csv(OUT_SENS, index=False)
    pd.concat(daily, ignore_index=True).to_parquet(SENS_DAILY)
    print(f"done sens {time.time()-t0:.0f}s")


# ====================================================================== 事件层:规则日一致 vs 不一致
def cmd_events():
    import inc_t3_analyze as AN
    L = pd.read_csv(C.LEDGER / "inc_t3_event_level.csv.gz", parse_dates=["D"])
    ipo = pd.read_csv(OUT_IPO, parse_dates=["list_date", "FREE_DATE", "R_date"])[["event_idx", "gap_cal", "gap_bin", "regular", "R_date"]]
    L = L.merge(ipo, on="event_idx", how="inner")
    L = L[L["tg"] == "首发"]
    L = L[L["ratio_plan"] >= 0.10]
    EV = {"2021-2025": ("2021-01-01", "2025-12-31"), "2021-2023": ("2021-01-01", "2023-12-31"), "2024-2025": ("2024-01-01", "2025-12-31")}
    groups = {
        "规则日一致": L["regular"].to_numpy(bool),
        "不一致(全部)": ~L["regular"].to_numpy(bool),
        "不一致:提前": L["gap_bin"].isin(["提前>7日", "提前1-7日"]).to_numpy(),
        "不一致:晚8-30日": (L["gap_bin"] == "晚8-30日").to_numpy(),
        "不一致:晚>30日": (L["gap_bin"] == "晚>30日").to_numpy(),
    }
    SW = ["[-20,-4]", "[-3,-1]", "[-20,-1]", "[0,20]", "[-3,20]"]
    rows = []
    for wn, (a, b) in EV.items():
        wm = ((L["D"] >= a) & (L["D"] <= b)).to_numpy()
        for gname, gm in groups.items():
            for sw in SW:
                col = f"car_B1_{sw}"
                m = wm & gm & L[col].notna().to_numpy()
                x = L.loc[m, col].to_numpy()
                if len(x) < 5:
                    continue
                mu, se, nG = C.cluster_mean_se(x, L.loc[m, "d_pos"].to_numpy())
                rows.append(dict(window=wn, group=gname, subwin=sw, n=len(x), judgeable=bool(len(x) >= 100), mean=float(mu[0]), se_D=float(se[0]),
                                 lo=float(mu[0] - 1.96 * se[0]), hi=float(mu[0] + 1.96 * se[0]), t_D=float(mu[0] / se[0]), nG_D=int(nG)))
        # 差:不一致(全部) - 一致
        for gname in ("不一致(全部)", "不一致:晚>30日"):
            for sw in SW:
                col = f"car_B1_{sw}"
                m1 = wm & groups[gname] & L[col].notna().to_numpy()
                m0 = wm & groups["规则日一致"] & L[col].notna().to_numpy()
                if m1.sum() < 5:
                    continue
                dif, se, nG = AN.ES_cluster_diff(L.loc[m1, col].to_numpy()[:, None], L.loc[m1, "d_pos"].to_numpy(),
                                                 L.loc[m0, col].to_numpy()[:, None], L.loc[m0, "d_pos"].to_numpy())
                rows.append(dict(window=wn, group=f"差:{gname} − 规则日一致", subwin=sw, n=int(m1.sum()), judgeable=bool(m1.sum() >= 100),
                                 mean=float(dif[0]), se_D=float(se[0]), lo=float(dif[0] - 1.96 * se[0]), hi=float(dif[0] + 1.96 * se[0]),
                                 t_D=float(dif[0] / se[0]), nG_D=int(nG)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT_CAR, index=False)
    pd.set_option("display.width", 220)
    print(R[R["window"] == "2021-2025"].round(4).to_string())

    # ---- 可交易回避收益 R(V_A = D-10 -> D+20, B1) 在"A 层去掉不规则首发"口径下是否仍成立 ----
    L2 = pd.read_csv(C.LEDGER / "inc_t3_event_level.csv.gz", parse_dates=["D"])
    L2 = L2[(L2["tier"] == "A") & (L2["ratio_plan"] >= 0.10)]
    reg_map = pd.read_csv(OUT_IPO)[["event_idx", "regular"]].set_index("event_idx")["regular"]
    is_ipo = (L2["tg"] == "首发").to_numpy()
    reg = L2["event_idx"].map(reg_map).eq(True).to_numpy(bool)
    tr_groups = {"A 层(预注册口径,全部)": np.ones(len(L2), bool), "A 层去掉不规则首发": ~(is_ipo & ~reg),
                 "仅不规则首发": is_ipo & ~reg, "仅定增": (L2["tg"] == "定增").to_numpy()}
    trows = []
    col = "R_V_A=D-10_h20_B1"
    for wn, (a, b) in EV.items():
        wm = ((L2["D"] >= a) & (L2["D"] <= b)).to_numpy()
        for gname, gm in tr_groups.items():
            m = wm & gm & L2[col].notna().to_numpy()
            x = L2.loc[m, col].to_numpy()
            mu, se, nG = C.cluster_mean_se(x, L2.loc[m, "d_pos"].to_numpy())
            trows.append(dict(window=wn, group=gname, V="V_A=D-10", h=20, control="B1", n=len(x), judgeable=bool(len(x) >= 100),
                              mean=float(mu[0]), se_D=float(se[0]), lo=float(mu[0] - 1.96 * se[0]), hi=float(mu[0] + 1.96 * se[0]), nG_D=int(nG)))
    TR = pd.DataFrame(trows)
    TR.to_csv(C.LEDGER / "inc_t3_review_tradable_regular.csv", index=False)
    print(TR.round(4).to_string())


# ====================================================================== 图
def cmd_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    S = pd.read_csv(OUT_SENS)
    S = S[S["kind"] == "scenario"].set_index("scenario")
    rows = [("主格(预注册:全部 A 层事前可知)", "main"), ("不规则首发降 B 层", "irrB"), ("不规则首发只知规则日 R", "ruleD"),
            ("校准:定增 / 追加承诺按抽样比例降 B", "calib"), ("最严:仅规则首发为 A,其余降 B", "strictIPO"), None,
            ("单独剔除:规则日一致的首发", "onlyIPOreg"), ("单独剔除:规则日不一致的首发", "onlyIPOirr"),
            ("单独剔除:定增", "onlyDZ"), ("单独剔除:追加承诺", "onlyZJ"), ("去掉追加承诺后的全部剔除", "noZhuijia")]
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#dcdbd5", "#fcfcfb"
    c_full, c_test = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor=surf)
    ax.set_facecolor(surf)
    y, ys, labs = 0, [], []
    for r in rows:
        if r is None:
            ax.axhline(y - 0.5, color=grid, lw=1.0)
            y += 1
            continue
        lab, key = r
        for off, w, col in ((-0.14, "2021-2025", c_full), (0.14, "2024-2025", c_test)):
            m, se = S.loc[key, f"ann_{w}"] * 100, S.loc[key, f"se_{w}"] * 100
            if key == "calib" and w == "2021-2025":
                lo, hi = S.loc[key, "draw_min_full"] * 100, S.loc[key, "draw_max_full"] * 100        # 校准情景:20 次抽样的范围
            elif key == "calib":
                lo, hi = S.loc[key, "draw_min_test"] * 100, S.loc[key, "draw_max_test"] * 100
            else:
                lo, hi = m - 1.96 * se, m + 1.96 * se
            ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot(m, y + off, "o", ms=7, color=col, mec=surf, mew=1.5, zorder=3)
        ys.append(y)
        labs.append(lab)
        y += 1
    ax.axvline(0.20, color=ink2, lw=1.0, ls="--", zorder=1)
    ax.text(0.205, -0.55, "+0.20pp 通过线", fontsize=9, color=ink2, va="center")
    ax.axvline(0, color=ink2, lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labs, fontsize=10, color=ink)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlabel("相对基线排雷的年化配对日差(pp,Y=1、1000 万;线 = 95% 区间,校准行 = 20 次抽样范围)", fontsize=10, color=ink2)
    ax.grid(axis="x", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, length=0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=c_full, lw=1.6, label="全期 2021-2025"),
                       Line2D([], [], marker="o", color=c_test, lw=1.6, label="检验窗 2024-2025")],
              loc="center right", bbox_to_anchor=(1.0, 0.52), frameon=False, fontsize=10, labelcolor=ink)
    ax.set_title("解禁剔除增量对\"确切解禁日是否事前可知\"的敏感性(审查后新增,描述性)", fontsize=12, color=ink, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(C.REPORTS / "inc_t3_review_sens.png", dpi=140, facecolor=surf)
    plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gaps"
    {"gaps": cmd_gaps, "sens": cmd_sens, "events": cmd_events, "plot": cmd_plot}[cmd]()
