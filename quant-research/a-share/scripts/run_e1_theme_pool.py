"""E1:"熟悉方向"这一层值多少 —— 把守门从"证监会大类 + 个股 range_pos"换成热门主题的 PIT 代理。

主题代理(每月初只用过去信息):
  1) 取 ETF 簇图(effective_month=m,as_of=上月末)中的簇代表 ETF,限定 research_class == domestic_equity_etf
     (排除债券/货币/黄金/商品/港股与跨境——股票池是 A 股,这些簇映射不到 A 股个股);
  2) 按代表 ETF 过去 120 个交易日收益排序,取 K=3 个簇;
  3) 每只股票按过去 120 日日收益与这 3 个代表 ETF 的相关性映射:corr >= 0.6 归入,取最高者;
  4) 得到逐月变化的"热门主题股票池"。
引擎:与 V1 同的入场路径 / 仓位 / 退出,只把守门换成"在主题池内"。
对照:该池等权、守门集合等权(基线A)、全池等权、该池内随机入场 ×10、原 V1 / V7。
考场:2024-09-24 ~ 2025-12-31(全部已揭开,本次为诊断,结论标"样本内")。
"""
import os
from pathlib import Path, sys, glob, json, gc, warnings; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.config import LEDGER
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
version = os.environ.get("EB_VERSION", "V100")
K, LOOK, CORR_MIN = 3, 120, 0.6
# 外部 ETF 数据仓路径:用 ETF_STORE 环境变量覆盖
STORE = os.environ.get("ETF_STORE",
     str(Path(__file__).resolve().parents[2] / "ETF-Research" /
         "full-etf project2 " / "full-etf-universe-store"))
DAILY_VER = "daily-base-v2-60455ea6ab9f3f5469ea21fb"
T0, T1 = "2024-09-24", "2025-12-31"

# ---------------- A 股面板 ----------------
p = Panel(); d = p.dates; codes = p.codes
raw = pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code'] = raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$', r'\2.\1', regex=True)
lab = pd.Series(dict(zip(raw.code, raw.industry))).reindex(codes).replace('', np.nan)
u = ST.universe(p, version); R = EB.Rules
close = p.raw("close_px").astype(np.float64)
stock_ret = close.pct_change()

# ---------------- ETF 日线 + 簇图 ----------------
# 用 minute_qfq 的 15:00 bar 作日收盘。两套存储(daily_base / minute_qfq)都**不调整份额折算**
# (如 159363 2025-07-21 0.298->0.594、512800 2025-07-07 +101%),ETF 有 10%/20% 涨跌停,
# 单日 |r|>25% 只能是份额合并/拆分 -> 置 0 后重建复权净值。分红(~1-3%)忽略。
import pyarrow.parquet as pq
fs = sorted(glob.glob(f"{STORE}/minute_qfq/yr=202[45]/bkt=*/part.parquet"))
etf = pd.concat([pq.read_table(f, columns=["symbol", "session_date", "close"], filters=[("minute_of_day", "==", 900)]).to_pandas() for f in fs], ignore_index=True)
etf["session_date"] = pd.to_datetime(etf["session_date"])
etf_close = etf.pivot(index="session_date", columns="symbol", values="close").sort_index()
del etf; gc.collect()
print(f"ETF 日线(minute_qfq 15:00): {etf_close.shape}, {etf_close.index[0].date()} ~ {etf_close.index[-1].date()}")
missing_sess = [x for x in d[(d >= "2024-03-01") & (d <= T1)] if x not in etf_close.index]
print(f"A 股交易日不在 ETF 日线中的: {len(missing_sess)} {missing_sess[:5]}")
etf_close = etf_close.reindex(d)     # 对齐 A 股交易日
etf_ret = etf_close.pct_change(fill_method=None)
n_jump = int((etf_ret.abs() > 0.25).sum().sum()); print(f"份额折算类跳变(|r|>25%)置 0: {n_jump} 个 symbol-日")
etf_ret = etf_ret.mask(etf_ret.abs() > 0.25, 0.0)
sm = pd.read_parquet(glob.glob(f"{STORE}/universe/classification_versions/*/symbol_master.parquet")[0])
name_of = dict(zip(sm.symbol, sm.name)); idxname_of = dict(zip(sm.symbol, sm.tracked_index_name))
dom_eq = set(sm.loc[sm.research_class == "domestic_equity_etf", "symbol"])
cm_files = {os.path.basename(os.path.dirname(f)).split("=")[1]: f for f in glob.glob(f"{STORE}/clusters/versions/*/batch=cluster_map/effective_month=*/part.parquet")}

# ---------------- 逐月主题池 ----------------
months = pd.period_range("2024-09", "2025-12", freq="M")
pool = pd.DataFrame(False, index=d, columns=codes)
corr_panel = pd.DataFrame(np.nan, index=d, columns=codes)       # 与归入簇代表的相关性(用于 T4 排序)
month_info = []; zx = "300308.SZ"; zx_rows = []
for m in months:
    m0 = m.start_time; m_end = m.end_time
    idx_end = int(np.searchsorted(d.values, np.datetime64(m0)) - 1)       # 上月最后一个交易日
    win = d[idx_end - LOOK + 1: idx_end + 1]
    cm = pd.read_parquet(cm_files[str(m)], columns=["symbol", "cluster_id", "representative_symbol", "is_representative", "member_count"])
    reps = cm[cm.is_representative & cm.symbol.isin(dom_eq)]
    reps = reps[reps.symbol.isin(etf_close.columns)]
    Ew = etf_ret.loc[win, reps.symbol]
    nobs = Ew.count(); r120 = ((1 + Ew.fillna(0)).prod() - 1.0)[nobs >= 100]      # 窗口内至少 100 个有效日(新 ETF 自动排除)
    top = r120.sort_values(ascending=False).head(K)
    E = etf_ret.loc[win, top.index]
    Rs = stock_ret.loc[win]
    ok = Rs.count() >= 80
    C = pd.concat({s: Rs.corrwith(E[s]) for s in top.index}, axis=1)
    C = C.where(ok, np.nan)
    best = C.max(axis=1); best_s = C.idxmax(axis=1)
    member = (best >= CORR_MIN).fillna(False)
    in_m = (d >= m0) & (d <= m_end)
    pool.loc[in_m, member[member].index] = True
    corr_panel.loc[in_m, member[member].index] = best[member].values
    u_m = u.loc[in_m].any(axis=0) if in_m.any() else pd.Series(False, index=codes)
    sizes_by = member.groupby(best_s).sum().reindex(top.index).fillna(0).astype(int)
    month_info.append(dict(月=str(m), 池规模=int(member.sum()), 池并V100=int((member & u_m).sum()),
                           簇1=f"{name_of.get(top.index[0], top.index[0])}({top.iloc[0]*100:+.0f}%,{sizes_by.iloc[0]})",
                           簇2=f"{name_of.get(top.index[1], top.index[1])}({top.iloc[1]*100:+.0f}%,{sizes_by.iloc[1]})",
                           簇3=f"{name_of.get(top.index[2], top.index[2])}({top.iloc[2]*100:+.0f}%,{sizes_by.iloc[2]})",
                           候选簇数=int(len(r120))))
    zx_rows.append(dict(月=str(m), 在池=bool(member.get(zx, False)), 最高相关=float(best.get(zx, np.nan)),
                        归入=name_of.get(best_s.get(zx, ""), "") if member.get(zx, False) else "",
                        与各簇相关="/".join(f"{x:.2f}" for x in C.loc[zx].values) if zx in C.index else ""))
mi = pd.DataFrame(month_info); print("\n=== 逐月主题池(月初 PIT;括号 = 代表ETF 过去120日收益, 归入股票数)===")
print(mi.to_string(index=False))
print("\n=== 中际旭创 300308.SZ 归入情况 ===")
print(pd.DataFrame(zx_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# ---------------- 信号 ----------------
sigV1 = EB.build_signals(p, R(rank_by="sector"), u, lab)
c1 = close.shift(1); ma60 = c1.rolling(60, min_periods=30).mean()
paths = sigV1["path_lowopen"] | sigV1["path_bo"]
entry_ng = (u & (sigV1["px"] > ma60) & paths).fillna(False)          # 与 build_signals 同式,只去掉两道守门
score_sector = sigV1["score_all"].astype(np.float32); sigV1["score_all"] = score_sector; sigV1["score"] = score_sector.where(sigV1["entry"])
corr_panel = corr_panel.astype(np.float32)
for k_ in ("sec_rp", "rp"): sigV1.pop(k_, None)
del ma60, paths, c1, close, stock_ret, etf_close, etf_ret; gc.collect()

def mk(entry, score_all):
    s = dict(sigV1); s["entry"] = entry; s["score"] = score_all.where(entry); s["score_all"] = score_all; return s
pool_u = pool & u
variants = {
    "T1 主题池守门,板块排序":            (R(rank_by="sector"),          mk(entry_ng & pool, score_sector)),
    "T2 T1 不止损":                      (R(rank_by="sector", sl=9.0),  mk(entry_ng & pool, score_sector)),
    "T3 原V1守门 ∩ 主题池":              (R(rank_by="sector"),          mk(sigV1["entry"] & pool, score_sector)),
    "T4 主题池守门,按与主题ETF相关性排序": (R(rank_by="sector"),          mk(entry_ng & pool, corr_panel.fillna(0))),
    "T5 T4 不止损":                      (R(rank_by="sector", sl=9.0),  mk(entry_ng & pool, corr_panel.fillna(0))),
    "V1 原版(证监会大类+个股rp守门)":      (R(rank_by="sector"),          sigV1),
    "V7 原版不止损":                      (R(rank_by="sector", sl=9.0),  sigV1),
}
segs = {"924后A 2024-09-24~2025-06": (d >= T0) & (d <= "2025-06-30"),
        "2025H2": (d >= "2025-07-01") & (d <= T1),
        "924后全段 2024-09-24~2025-12": (d >= T0) & (d <= T1)}
full = np.asarray(segs["924后全段 2024-09-24~2025-12"])

rows = []; monthly = {}
def add(name, seg, r, extra=""):
    rows.append(dict(变体=name, 段=seg, 年化=r.get("ann_return"), 夏普=r.get("sharpe"), 回撤=r.get("max_dd"), 仓位=r.get("exposure"),
                     交易=r.get("n_trades", np.nan), 胜率=r.get("win_rate", np.nan), 持有天=r.get("avg_hold", np.nan),
                     退出=json.dumps(r.get("exit_mix", {}), ensure_ascii=False) if r.get("exit_mix") else extra))
def mret(rets): return rets.groupby(rets.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)

for name, (rules, sig) in variants.items():
    for sname, m in segs.items():
        r = EB.run_event_bt(p, rules, version, date_mask=np.asarray(m), sector_labels=lab, universe=u, signals=sig)
        add(name, sname, r)
        if sname.startswith("924后全段"): monthly[name] = mret(r["rets"])
    print("done", name, flush=True)

gate = (sigV1["strong"] & u); rng = np.random.default_rng(0)
t1_entry = variants["T1 主题池守门,板块排序"][1]["entry"]
for sname, m in segs.items():
    m = np.asarray(m)
    ew = pd.DataFrame(0.0, index=d, columns=codes).where(pool_u)
    r = ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=m)
    add("对照 主题池等权", sname, r, f"池∩V100 中位 {int(pool_u[m].sum(axis=1).median())} 只")
    if sname.startswith("924后全段"): monthly["主题池等权"] = mret(r["rets"])
    ew = pd.DataFrame(0.0, index=d, columns=codes).where(gate)
    r = ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=m)
    add("基线A 守门集合等权", sname, r, f"守门中位 {int(gate[m].sum(axis=1).median())} 只")
    if sname.startswith("924后全段"): monthly["守门集合等权"] = mret(r["rets"])
    r = ST.benchmark(p, version, "tail1450", date_mask=m)
    add("参照 全池等权", sname, r)
    if sname.startswith("924后全段"): monthly["全池等权"] = mret(r["rets"])
    anns = []; rnd_m = []
    for k in range(10):
        n_day = t1_entry.sum(axis=1)
        rnd = pd.DataFrame(rng.random(pool_u.shape, dtype=np.float32), index=d, columns=codes).where(pool_u)
        e_rnd = (rnd.rank(axis=1, method="first").le(n_day, axis=0)) & pool_u
        s2 = dict(sigV1); s2["entry"] = e_rnd; s2["score"] = rnd.where(e_rnd); s2["score_all"] = rnd
        rr = EB.run_event_bt(p, R(rank_by="none"), version, date_mask=m, sector_labels=lab, universe=u, signals=s2)
        anns.append(rr["ann_return"]); rnd_m.append(mret(rr["rets"]))
    a = np.array(anns)
    t1 = [x for x in rows if x["变体"].startswith("T1 ") and x["段"] == sname][0]["年化"]; t2 = [x for x in rows if x["变体"].startswith("T2 ") and x["段"] == sname][0]["年化"]
    add("随机对照 主题池内随机入场×10", sname, dict(ann_return=float(a.mean())), f"[{a.min()*100:+.1f}%, {a.max()*100:+.1f}%]; T1 第{int((a<t1).mean()*100)}百分位, T2 第{int((a<t2).mean()*100)}百分位")
    if sname.startswith("924后全段"): monthly["池内随机×10均值"] = pd.concat(rnd_m, axis=1).mean(axis=1)
    print("done controls", sname, flush=True)

df = pd.DataFrame(rows); df.to_csv(LEDGER / f"e1_theme_pool_{version}.csv", index=False)
for sname in segs:
    print(f"\n--- {sname} ---"); print(df[df.段 == sname].drop(columns="段").to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ---------------- 逐月收益表 ----------------
M = pd.DataFrame(monthly); M.index = M.index.astype(str)
M = M.join(mi.set_index("月")[["池规模", "池并V100"]])
M.to_csv(LEDGER / f"e1_theme_pool_monthly_{version}.csv")
print("\n=== 逐月收益表(自然月;2024-09 仅含 09-24 起;用户尺子:每月 +5~10%)===")
pct = M.drop(columns=["池规模", "池并V100"]).applymap(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "")
pct["池规模"] = M["池规模"]; pct["池并V100"] = M["池并V100"]
print(pct.to_string())
print("\n逐月统计(924后全段):")
for c in M.columns[:-2]:
    x = M[c].dropna(); print(f"  {c:28s} 月均 {x.mean()*100:+.2f}%  月中位 {x.median()*100:+.2f}%  ≥+5% 月数 {(x>=0.05).sum()}/{len(x)}  负月 {(x<0).sum()}/{len(x)}  累计 {((1+x).prod()-1)*100:+.1f}%")
h2 = M.loc["2025-07":"2025-12"]
print("\n2025H2 累计:", {c: f"{((1+h2[c].dropna()).prod()-1)*100:+.1f}%" for c in h2.columns[:-2]})
