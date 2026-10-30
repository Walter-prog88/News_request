"""E4:差距是选股还是择时 —— 同一守门集合里的"事后上限"(诊断,样本内,2024-09-24 ~ 2025-12-31)。

(a) 守门集合(strong ∩ universe,每日)内的前视上限:每 5 日调仓,选未来 5 日收益最高 5 只 / 前 q% / 随机 5 只 / 全部等权
(b) 同样的上限在全池里做一遍,对比池子缩小后天花板掉了多少
(c) 引擎 V1 实际交易清单逐笔归因:入场路径 / 退出原因 / 持有天 分桶
(d) 达到每月 +5% / +10% 所需的单笔期望收益与胜率,对比引擎实测

口径:尾盘 tail1450(T 日 14:50-14:57 TWAP 进,T+5 同窗口出);入场须可成交(非一字板、窗口有量);
成本:单边 1bp + 卖出印花税 5bp;参与率上限 10% × amt_tail8(对 top-5 线按当前净值生效)。
前视上限只作天花板,不是策略。"""
import os, sys, warnings, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.config import LEDGER, COST_BPS_ONEWAY, STAMP_TAX_BPS_AFTER, MAX_PARTICIPATION
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

version = "V100"; CAPITAL = ST.VERSIONS[version]["capital"]
W0, W1 = "2024-09-24", "2025-12-31"
H = 5
p = Panel(); d = p.dates; codes = p.codes; N = len(codes); T = len(d)
raw = pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code'] = raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$', r'\2.\1', regex=True)
lab = pd.Series(dict(zip(raw.code, raw.industry))).reindex(codes).replace('', np.nan)
u = ST.universe(p, version); R = EB.Rules
rules = R(rank_by="sector")
sig = EB.build_signals(p, rules, u, lab)
gate = (sig["strong"] & u).fillna(False)
m = np.asarray((d >= W0) & (d <= W1))
log("signals built; gate median", int(gate[m].sum(axis=1).median()), "pool median", int(u[m].sum(axis=1).median()))

# ---------------- 价格与可成交 ----------------
px = p.raw("twap_tail8").to_numpy(np.float64)            # 成交价
close = p.raw("close_px").to_numpy(np.float64)           # 日内计价
amt_tail = p.raw("amt_tail8").to_numpy(np.float64)
tradable = p["twap_tail8_f"].notna().to_numpy(bool)      # 非一字板且窗口有量
lab_arr = lab.to_numpy(object)
cost_rt = (2 * COST_BPS_ONEWAY + STAMP_TAX_BPS_AFTER) * 1e-4   # 往返 7bp

idx_win = np.flatnonzero(m)
reb = idx_win[::H]
log("rebalance days", len(reb), d[reb[0]].date(), d[reb[-1]].date())
for t in reb[:6]:
    log(f"  {d[t].date()} 守门 {int(gate.iloc[t].sum())} 只, 守门∩科技链(C39/I65/I64) {int((gate.iloc[t] & lab.astype(str).str[:3].isin(['C39','I65','I64'])).sum())} 只")


def period_paths(t, members):
    """t 日尾盘买入 members,t+H 日尾盘卖出;返回 (日期索引列表, 篮子逐日价值 B_k(k=0..H), 篮子期收益)。
    停牌日价值用前值填充;面板末尾不足 H 天用最后一天收盘价出。"""
    te = min(t + H, T - 1)
    E = px[t, members]
    seg = close[t:te + 1, members] / E                     # (H+1, k)
    seg = pd.DataFrame(seg).ffill().fillna(1.0).to_numpy()
    if te == t + H:
        X = px[te, members] / E
        X = np.where(np.isfinite(X), X, seg[-1])
        seg[-1] = X
    B = seg.mean(axis=1)                                     # 等权买入持有
    return list(range(t, te + 1)), B, float(B[-1] - 1.0)


def run_line(select_fn, expo=1.0):
    """select_fn(t, cand_idx, fwd_vals, nav_yuan) -> members(ndarray);返回日收益序列与期收益列表"""
    nav = 1.0; rets = {}; per = []
    for t in reb:
        te = min(t + H, T - 1)
        if te <= t: break
        f = np.where(np.isfinite(px[te]), px[te] / px[t] - 1.0, np.nan)
        members = select_fn(t, f, nav * CAPITAL)
        if members is None or len(members) == 0:
            for k in range(t, te + 1): rets[k] = 0.0
            per.append(np.nan); continue
        days, B, pr = period_paths(t, members)
        P = (1 - expo) + expo * B
        prev = 1.0
        for k, day in enumerate(days):
            r = P[k] / prev - 1.0
            if k == len(days) - 1: r -= expo * cost_rt
            rets[day] = r; prev = P[k]
        nav *= P[-1] - expo * cost_rt
        per.append(pr)
    s = pd.Series(rets).sort_index(); s.index = d[s.index]
    return s, np.array(per)


def make_selector(pool_mask, kind, q=None, seed=None, sector_top=None):
    rng = np.random.default_rng(seed)

    def sel(t, f, nav_yuan):
        ok = pool_mask[t] & tradable[t] & np.isfinite(f)
        cand = np.flatnonzero(ok)
        if len(cand) < (1 if kind == "all" else 5): return None     # 等权线只要有票就建仓;5 只线不足 5 只则空仓
        fv = f[cand]
        if kind in ("top5", "top5_fixedcap"):
            # 参与率上限:5 只各 20% 净值(fixedcap:按起始 100 万算,不随净值膨胀)
            cap_ok = MAX_PARTICIPATION * amt_tail[t, cand] >= 0.2 * (CAPITAL if kind == "top5_fixedcap" else nav_yuan)
            cand, fv = cand[cap_ok], fv[cap_ok]
            if len(cand) < 5: return None
            return cand[np.argsort(-fv)[:5]]
        if kind == "bottom5":
            return cand[np.argsort(fv)[:5]]
        if kind == "topq":
            k = max(5, int(np.ceil(q * len(cand))))
            return cand[np.argsort(-fv)[:k]]
        if kind == "all":
            return cand
        if kind == "rand5":
            return rng.choice(cand, 5, replace=False)
        if kind == "sector":      # 事后最强板块(按候选等权均值),板块内全部等权;sector_top 个板块
            labs = lab_arr[cand]
            df = pd.DataFrame(dict(l=labs, f=fv, j=cand)).dropna(subset=["l"])
            g = df.groupby("l").agg(f=("f", "mean"), n=("f", "size"))
            g = g[g.n >= 5].sort_values("f", ascending=False)
            if len(g) == 0: return None
            pick = g.index[:sector_top]
            return df[df.l.isin(pick)].j.to_numpy()
        if kind == "sector_rand5":   # 事后最强板块内随机 5 只(方向对、个股随机)
            labs = lab_arr[cand]
            df = pd.DataFrame(dict(l=labs, f=fv, j=cand)).dropna(subset=["l"])
            g = df.groupby("l").agg(f=("f", "mean"), n=("f", "size"))
            g = g[g.n >= 5].sort_values("f", ascending=False)
            if len(g) == 0: return None
            jj = df[df.l == g.index[0]].j.to_numpy()
            return rng.choice(jj, 5, replace=False)
        raise ValueError(kind)
    return sel


def summarize(s):
    nav = (1 + s).cumprod(); yrs = len(s) / EB.TRADING_DAYS
    return dict(ann=float(nav.iloc[-1] ** (1 / yrs) - 1), total=float(nav.iloc[-1] - 1),
                mdd=float((nav / nav.cummax() - 1).min()),
                sharpe=float(s.mean() / s.std() * np.sqrt(EB.TRADING_DAYS)) if s.std() > 0 else np.nan)


def monthly(s):
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1


gate_np = gate.to_numpy(bool); pool_np = u.to_numpy(bool)
lines = {}; per_ret = {}
Q = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90]
for pname, pm in [("守门", gate_np), ("全池", pool_np)]:
    s, pr = run_line(make_selector(pm, "top5")); lines[f"{pname}|前视Top5"] = s; per_ret[f"{pname}|前视Top5"] = pr
    s, pr = run_line(make_selector(pm, "top5_fixedcap")); lines[f"{pname}|前视Top5(参与率按起始100万)"] = s; per_ret[f"{pname}|前视Top5(参与率按起始100万)"] = pr
    for q in Q:
        s, pr = run_line(make_selector(pm, "topq", q=q)); lines[f"{pname}|前视前{int(q*100)}%"] = s; per_ret[f"{pname}|前视前{int(q*100)}%"] = pr
    s, pr = run_line(make_selector(pm, "all")); lines[f"{pname}|全部等权"] = s; per_ret[f"{pname}|全部等权"] = pr
    s, pr = run_line(make_selector(pm, "bottom5")); lines[f"{pname}|前视Bottom5"] = s; per_ret[f"{pname}|前视Bottom5"] = pr
    s, pr = run_line(make_selector(pm, "sector", sector_top=1)); lines[f"{pname}|前视最强板块等权"] = s; per_ret[f"{pname}|前视最强板块等权"] = pr
    s, pr = run_line(make_selector(pm, "sector", sector_top=3)); lines[f"{pname}|前视前3板块等权"] = s; per_ret[f"{pname}|前视前3板块等权"] = pr
    rs = []; rps = []
    for k in range(50):
        s, pr = run_line(make_selector(pm, "rand5", seed=k)); rs.append(s); rps.append(pr)
    RS = pd.concat(rs, axis=1); lines[f"{pname}|随机5只(50次均值)"] = RS.mean(axis=1)
    anns = np.array([summarize(x)["ann"] for x in rs]); lines[f"{pname}|随机5只__anns"] = anns
    per_ret[f"{pname}|随机5只(50次均值)"] = np.nanmean(np.vstack(rps), axis=0)
    rs2 = []
    for k in range(50):
        s, pr = run_line(make_selector(pm, "sector_rand5", seed=k)); rs2.append(s)
    RS2 = pd.concat(rs2, axis=1); lines[f"{pname}|前视最强板块内随机5只(50次均值)"] = RS2.mean(axis=1)
    lines[f"{pname}|前视最强板块内随机5只__anns"] = np.array([summarize(x)["ann"] for x in rs2])
    log(pname, "done")

# ---------------- 非前视"方向"线:AI/科技链 = 证监会 C39 电子 + I65 软件 + I64 互联网(用户定方向,系统守门)----------------
TECH = ("C39", "I65", "I64")
tech_mask = np.array([isinstance(x, str) and x[:3] in TECH for x in lab_arr])
c39_mask = np.array([isinstance(x, str) and x[:3] == "C39" for x in lab_arr])
for pname, pm in [("守门", gate_np), ("全池", pool_np)]:
    pmt = pm & tech_mask[None, :]
    s, pr = run_line(make_selector(pmt, "all")); lines[f"{pname}∩科技链|全部等权"] = s; per_ret[f"{pname}∩科技链|全部等权"] = pr
    s, pr = run_line(make_selector(pm & c39_mask[None, :], "all")); lines[f"{pname}∩C39电子|全部等权"] = s; per_ret[f"{pname}∩C39电子|全部等权"] = pr
    rs = []; rps = []
    for k in range(50):
        s, pr = run_line(make_selector(pmt, "rand5", seed=k)); rs.append(s); rps.append(pr)
    lines[f"{pname}∩科技链|随机5只(50次均值)"] = pd.concat(rs, axis=1).mean(axis=1)
    lines[f"{pname}∩科技链|随机5只__anns"] = np.array([summarize(x)["ann"] for x in rs])
    per_ret[f"{pname}∩科技链|随机5只(50次均值)"] = np.nanmean(np.vstack(rps), axis=0)
    for q in (0.5, 0.2):
        s, pr = run_line(make_selector(pmt, "topq", q=q)); lines[f"{pname}∩科技链|前视前{int(q*100)}%"] = s; per_ret[f"{pname}∩科技链|前视前{int(q*100)}%"] = pr
log("tech lines done; 守门∩科技链 median", int((gate_np & tech_mask[None, :])[m].sum(axis=1).mean()), "全池∩科技链", int((pool_np & tech_mask[None, :])[m].sum(axis=1).mean()))

# ---------------- 引擎实跑(V1 / V7)----------------
res_v1 = EB.run_event_bt(p, rules, version, date_mask=m, sector_labels=lab, universe=u, signals=sig)
sig7 = EB.build_signals(p, R(rank_by="sector", sl=9.0), u, lab)
res_v7 = EB.run_event_bt(p, R(rank_by="sector", sl=9.0), version, date_mask=m, sector_labels=lab, universe=u, signals=sig7)
lines["引擎V1(仓位≈55%)"] = res_v1["rets"]; lines["引擎V7不止损(仓位≈55%)"] = res_v7["rets"]
log("engine V1 ann", res_v1["ann_return"], "V7 ann", res_v7["ann_return"])
# 参照:项目既有口径
bench_pool = ST.benchmark(p, version, "tail1450", date_mask=m)
ew = pd.DataFrame(0.0, index=d, columns=codes).where(gate)
bench_gate = ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=m)
log("check: ST.benchmark pool ann", bench_pool["ann_return"], "ST gate EW (5 phases) ann", bench_gate["ann_return"])

# ---------------- (a)(b) 汇总 ----------------
rows = []
for k, s in lines.items():
    if k.endswith("__anns"): continue
    sm = summarize(s)
    pr = per_ret.get(k)
    rows.append(dict(线=k, 年化=sm["ann"], 总收益=sm["total"], 回撤=sm["mdd"], 夏普=sm["sharpe"],
                     五日期均收益=float(np.nanmean(pr)) if pr is not None else np.nan,
                     五日期中位=float(np.nanmedian(pr)) if pr is not None else np.nan,
                     月均=float(monthly(s).mean()), 月中位=float(monthly(s).median()),
                     月胜率=float((monthly(s) > 0).mean()), 月达5pct=float((monthly(s) >= 0.05).mean())))
summ = pd.DataFrame(rows)
print("\n=== (a)(b) 上限曲线汇总(2024-09-24~2025-12-31,样本内;前视线为 100% 仓位篮子,每 5 日换,含 7bp 往返成本)===")
print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
for pname in ("守门", "全池"):
    a = lines[f"{pname}|随机5只__anns"]; b = lines[f"{pname}|前视最强板块内随机5只__anns"]
    print(f"{pname} 随机5只 50次年化:均值 {a.mean():+.3f} 中位 {np.median(a):+.3f} [{a.min():+.3f}, {a.max():+.3f}] P10 {np.percentile(a,10):+.3f} P90 {np.percentile(a,90):+.3f}")
    print(f"{pname} 前视最强板块内随机5只 50次年化:均值 {b.mean():+.3f} 中位 {np.median(b):+.3f} [{b.min():+.3f}, {b.max():+.3f}]")
    c = lines[f"{pname}∩科技链|随机5只__anns"]
    print(f"{pname}∩科技链 随机5只 50次年化:均值 {c.mean():+.3f} 中位 {np.median(c):+.3f} [{c.min():+.3f}, {c.max():+.3f}] P10 {np.percentile(c,10):+.3f} P90 {np.percentile(c,90):+.3f}")

# 交易者尺子:每月 +5% / +10%,在 62~92% 仓位下需要的篮子月收益;并把守门集合 top-q 曲线按 q 插值
print("\n=== 交易者尺子:月 +5% / +10% 对应的前视分位(篮子月均收益 = 交易者月收益 / 仓位)===")
grid = [("前视Top5", None)] + [(f"前视前{int(q*100)}%", q) for q in Q] + [("全部等权", 1.0)]
for pname in ("守门", "全池"):
    print(f"-- {pname} --")
    for nm, q in grid:
        s = lines[f"{pname}|{nm}"]; mo = monthly(s)
        print(f"  {nm:>10s}: 月均 {mo.mean()*100:+6.2f}%  月中位 {mo.median()*100:+6.2f}%  5日期均 {np.nanmean(per_ret[f'{pname}|{nm}'])*100:+6.2f}%  年化 {summarize(s)['ann']*100:+8.1f}%")
    # 5 日篮子均值随 q 的曲线(q 越大越接近全部等权),按 log-q 线性插值求"所需前视分位"
    qs = sorted([(q, float(np.nanmean(per_ret[f"{pname}|{nm}"]))) for nm, q in grid if q is not None])
    qa = np.array([x[0] for x in qs]); ma_ = np.array([x[1] for x in qs])
    mu_all = float(np.nanmean(per_ret[f"{pname}|全部等权"])); mu20 = float(np.nanmean(per_ret[f"{pname}|前视前20%"])); mu10 = float(np.nanmean(per_ret[f"{pname}|前视前10%"]))
    for expo in (0.62, 0.77, 0.92):
        for tgt in (0.05, 0.10):
            g_p = (1 + tgt) ** (1 / (21 / H)) - 1; need = g_p / expo + cost_rt      # 每 5 日篮子需要的毛均值
            q_req = float(np.exp(np.interp(-need, -ma_, np.log(qa)))) if need <= ma_.max() else np.nan   # ma_ 随 q 递减
            p20 = (need - mu_all) / (mu20 - mu_all); p10 = (need - mu_all) / (mu10 - mu_all)
            print(f"  仓位 {expo:.2f} 目标月 {tgt*100:>2.0f}% -> 每5日篮子需 {need*100:.2f}%(等权 {mu_all*100:+.2f}%);≈ 前视前 {q_req*100:.0f}% 的等权 | 混合尺子:每笔以 {p20*100:.0f}% 概率命中前20%分位(其余随机),或 {p10*100:.0f}% 概率命中前10%")

# ---------------- 逐月表 ----------------
cols = {"引擎V1": "引擎V1(仓位≈55%)", "引擎V7": "引擎V7不止损(仓位≈55%)",
        "全池等权": "全池|全部等权", "守门等权": "守门|全部等权",
        "守门随机5": "守门|随机5只(50次均值)", "守门前50%": "守门|前视前50%", "守门前20%": "守门|前视前20%", "守门前5%": "守门|前视前5%", "守门Top5": "守门|前视Top5",
        "全池前20%": "全池|前视前20%", "全池Top5": "全池|前视Top5",
        "守门最强板块": "守门|前视最强板块等权",
        "守门前80%": "守门|前视前80%", "守门∩科技链等权": "守门∩科技链|全部等权", "全池∩科技链等权": "全池∩科技链|全部等权", "守门∩科技链随机5": "守门∩科技链|随机5只(50次均值)"}
MT = pd.DataFrame({k: monthly(lines[v]) for k, v in cols.items()})
MT.index = MT.index.astype(str)
print("\n=== 逐月收益表(自然月;引擎=实际仓位;其余=100% 仓位篮子;前视线=上限,不是策略)===")
print((MT * 100).to_string(float_format=lambda x: f"{x:+7.1f}"))
print("月均:", (MT.mean() * 100).round(2).to_dict())
print("月≥+5% 的比例:", (MT >= 0.05).mean().round(2).to_dict())
print("月<0 的比例:", (MT < 0).mean().round(2).to_dict())
summ.to_csv(LEDGER / "e4_upper_bound_summary.csv", index=False)
MT.to_csv(LEDGER / "e4_upper_bound_monthly.csv")

# ---------------- (c) 逐笔归因 ----------------
def attribute(res, sig_, name):
    tr = res["trades"].copy()
    buys = tr[tr.side == "buy"].reset_index(drop=True); sells = tr[tr.side == "sell"].reset_index(drop=True)
    # 配对:每笔 sell 对应同 code 最近一次 buy
    recs = []
    last_buy = {}; rem = {}                      # rem: 部分退出后剩余份额(否则残仓二次卖出会按整槽计贡献,虚增 5pp)
    navp = res["nav"].shift(1).fillna(1.0)      # 入场时的前一日净值(槽位 = slot_w × navp)
    for _, r in tr.iterrows():
        if r.side == "buy":
            last_buy[r.code] = r; rem[r.code] = 1.0
        else:
            b = last_buy.get(r.code)
            if b is None: continue
            ti = d.get_loc(b.date); j = codes.get_loc(r.code)
            lo_ = bool(sig_["path_lowopen"].iat[ti, j]); bo_ = bool(sig_["path_bo"].iat[ti, j])
            path = "both" if (lo_ and bo_) else ("low_open_red" if lo_ else ("breakout" if bo_ else "none"))
            te = min(ti + H, T - 1)
            f5 = px[te, j] / px[ti, j] - 1.0 if np.isfinite(px[te, j]) else np.nan   # 固定 5 日持有
            # 同日守门集合 / 全池 等权 5 日收益
            g_ok = gate_np[ti] & tradable[ti] & np.isfinite(px[te]); p_ok = pool_np[ti] & tradable[ti] & np.isfinite(px[te])
            fg = np.nanmean(px[te, g_ok] / px[ti, g_ok] - 1.0) if g_ok.sum() else np.nan
            fp = np.nanmean(px[te, p_ok] / px[ti, p_ok] - 1.0) if p_ok.sum() else np.nan
            w = rules.slot_w * b.frac * rem[r.code] * r.frac * float(navp.loc[b.date])   # 以初始资金计的仓位占比(与引擎 NAV 精确对账)
            rem[r.code] *= (1.0 - r.frac)
            recs.append(dict(code=r.code, entry=b.date, exit=r.date, held=int(r.held), ret=float(r.ret), why=r.why, path=path,
                             w=w, contrib=w * float(r.ret), fwd5=f5, gate5=fg, pool5=fp, month=str(pd.Timestamp(b.date).to_period("M"))))
    L = pd.DataFrame(recs)
    L.to_csv(LEDGER / f"e4_trades_{name}.csv", index=False)
    nav_total = res["nav"].iloc[-1] - 1
    n_buys = int((tr.side == "buy").sum()); cost_pp = (n_buys * rules.slot_w * 1e-4 + len(L) * rules.slot_w * 6e-4) * 100
    print(f"\n=== (c) {name} 逐笔归因:{len(L)} 笔卖出 / {n_buys} 笔买入,净值总收益 {nav_total*100:+.1f}% = 已实现贡献 {L.contrib.sum()*100:+.1f}pp − 成本≈{cost_pp:.1f}pp ± 期末未实现,仓位 {res['exposure']:.2f} ===")
    print(f"单笔:均值 {L.ret.mean()*100:+.2f}%  中位 {L.ret.median()*100:+.2f}%  胜率 {(L.ret>0).mean():.3f}  平均盈利 {L.ret[L.ret>0].mean()*100:+.2f}%  平均亏损 {L.ret[L.ret<=0].mean()*100:+.2f}%  持有 {L.held.mean():.1f} 天")
    print(f"同一入场若固定持有 5 日(tail->tail):均值 {L.fwd5.mean()*100:+.2f}%  胜率 {(L.fwd5>0).mean():.3f};同日守门等权 5 日 {L.gate5.mean()*100:+.2f}%;同日全池等权 5 日 {L.pool5.mean()*100:+.2f}%")
    print(f"  -> 入场选股相对守门等权的 5 日 alpha {(L.fwd5-L.gate5).mean()*100:+.2f}pp;退出规则相对固定 5 日持有 {(L.ret-L.fwd5).mean()*100:+.2f}pp(单笔均值差)")
    def bucket(col, order=None):
        g = L.groupby(col).agg(笔数=("ret", "size"), 均收益=("ret", "mean"), 胜率=("ret", lambda x: (x > 0).mean()),
                               持有天=("held", "mean"), 贡献pp=("contrib", lambda x: x.sum() * 100), 固定5日均=("fwd5", "mean"))
        if order is not None: g = g.reindex([o for o in order if o in g.index])
        return g
    print("\n-- 按入场路径 --"); print(bucket("path").to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n-- 按退出原因 --"); print(bucket("why", ["tp", "sl", "ma", "time", "am_gap", "swap"]).to_string(float_format=lambda x: f"{x:.3f}"))
    L["held_b"] = pd.cut(L.held, [-1, 2, 5, 10, 15, 99], labels=["1-2天", "3-5天", "6-10天", "11-15天", ">15天"])
    print("\n-- 按持有天 --"); print(bucket("held_b").to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n-- 路径 × 退出 (贡献pp) --"); print((L.pivot_table(index="path", columns="why", values="contrib", aggfunc="sum") * 100).round(2).to_string())
    print("\n-- 路径 × 退出 (笔数) --"); print(L.pivot_table(index="path", columns="why", values="ret", aggfunc="size").fillna(0).astype(int).to_string())
    mb = L.groupby("month").agg(笔数=("ret", "size"), 均收益=("ret", "mean"), 胜率=("ret", lambda x: (x > 0).mean()), 贡献pp=("contrib", lambda x: x.sum() * 100))
    print("\n-- 按入场月 --"); print(mb.to_string(float_format=lambda x: f"{x:.3f}"))
    return L

L1 = attribute(res_v1, sig, "V1")
L7 = attribute(res_v7, sig7, "V7")
print("\n引擎统计 stats V1:", res_v1["stats"], " 退出构成:", res_v1["exit_mix"])

# ---------------- (d) 所需单笔期望与胜率 ----------------
print("\n=== (d) 达到月 +5% / +10% 所需的单笔期望与胜率(5 槽、平均持有 5 天 ≈ 每月 4.2 轮)===")
Wn = L1.ret[L1.ret > 0].mean(); Ln = -L1.ret[L1.ret <= 0].mean(); wr = (L1.ret > 0).mean(); mu_eng = L1.ret.mean(); hold_eng = L1.held.mean()
print(f"引擎 V1 实测:胜率 {wr:.3f},平均盈利 {Wn*100:+.2f}%,平均亏损 {-Ln*100:+.2f}%,单笔均值 {mu_eng*100:+.2f}%,平均持有 {hold_eng:.1f} 天,仓位 {res_v1['exposure']:.2f}")
rows_d = []
for tgt in (0.05, 0.10):
    for expo in (0.55, 0.62, 0.77, 0.92):
        rounds = 21 / H
        g_p = (1 + tgt) ** (1 / rounds) - 1           # 每 5 日周期净值需涨
        mu = g_p / expo + cost_rt                     # 单笔(5 日持有)需要的毛均值
        w_star = (mu + Ln) / (Wn + Ln)                # 给定引擎盈亏幅度,所需胜率
        W_star = (mu + (1 - wr) * Ln) / wr            # 给定引擎胜率与亏损幅度,所需平均盈利
        rows_d.append(dict(目标月=tgt, 仓位=expo, 每5日净值需涨=g_p, 单笔需均值=mu, 所需胜率_给定引擎盈亏幅=w_star, 所需平均盈利_给定引擎胜率=W_star, 年化=(1 + tgt) ** 12 - 1))
D = pd.DataFrame(rows_d); print(D.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
# 守门集合 5 日收益分布:需要的 mu 落在哪个分位
f5_all = []
for t in reb:
    te = min(t + H, T - 1)
    ok = gate_np[t] & tradable[t] & np.isfinite(px[te]) & np.isfinite(px[t])
    f5_all.append(px[te, ok] / px[t, ok] - 1.0)
f5_all = np.concatenate(f5_all)
print(f"\n守门集合单只 5 日收益分布(全部调仓日池内):均值 {f5_all.mean()*100:+.2f}% 中位 {np.median(f5_all)*100:+.2f}% 标准差 {f5_all.std()*100:.2f}% P(>0) {(f5_all>0).mean():.3f}")
for mu in (0.010, 0.015, 0.020, 0.030, 0.050):
    print(f"  单只 5 日收益 > {mu*100:.1f}% 的比例 {(f5_all>mu).mean():.3f}")
D.to_csv(LEDGER / "e4_required_stats.csv", index=False)
log("ALL DONE")
