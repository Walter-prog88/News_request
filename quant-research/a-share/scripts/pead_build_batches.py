"""PEAD 第 1 步:选股、匹配、平衡性(**不读任何收益**)。

按 research/PEAD_PREREG_ADDENDUM.md 第 2、4、5 节,在每个成交日 T(交易日历序号 T_pos,相位 φ = T_pos % 5)上:
  E      合格事件池内按 fu_op_yoy 降序取 ceil(n/5) 只(同分按代码升序),n<5 不建仓,w = min(1/n_sel, 5%)
  F      E 中剔除 COMP_v5 处在同一 t_signal 合格池最差 20% 的股票(rankdata(score)/n_scored <= 0.20;
         打分缺失视为不在最差 20%),被剔除槽位转现金,不递补
  E_rand 从 E 中随机删去与 F 同样多的股票(R=10,种子 [20261010, 3, φ, T_pos, r])
  B      候选 = 合格池 − E 入选股;每个 E 槽位逐级放宽(L1 同行业且三分位差<=0.10 / L2 同行业<=0.20 /
         L3 不限行业<=0.10 / L4 最近)匹配 3 个互不相同的对照,分别进 3 个复制组合(组合内不放回)
  C      候选 = 合格事件池 − E 入选股,其余同 B
  B_F    B 的复制组合只保留 F 留下的槽位
匹配变量(全部在 t_signal 收盘已知):申万一级(当前快照)、log 总市值(PIT,封存截断)、log ADV20、
20 日 ret_c2c 标准差(>=15 个有效日);连续变量换算为同日合格池内分位。

平衡性门槛(预注册 5.5,读收益之前落盘):|SMD| <= 0.10(三变量)、行业占比最大差 <= 5pp。
不达标 -> 只允许一次补救:L1、L3 容差 0.10 -> 0.05,重新匹配并记录;仍不达标则相关比较降级"风格未完全控制"。

产出:
  cache/_tmp/pead_batches.parquet       每个(臂, 复制, T, 槽位)一行的目标(整手前)
  cache/_tmp/pead_batch_meta.parquet    每个 T 的池规模、入选数、F 剔除数、打分缺失数
  cache/_tmp/pead_match_vars.parquet    E 槽位与对照的匹配变量(平衡性复核用)
  cache/_tmp/pead_signal_open10.parquet 信号面板(信号行 t x 股票,事件池外 NaN;parquet attrs 带口径说明)
  cache/_tmp/pead_signal_open10_events.parquet 信号面板的长表伴随文件(每个非空格的 P、V、T、known_at_t_close)
  research/ledger/pead_balance.csv      平衡性(含补救轮次)
  research/ledger/pead_batch_summary.csv 按年的批次统计
  research/ledger/pead_signal_panel_meta.csv 信号面板非空值按"P 相对信号行 t"的分布(lag 口径核对)
  research/ledger/pead_seal_audit.csv   封存核对(daily_basic 截断、因 2026 修订而排除的首次披露)

用法:python3 scripts/pead_build_batches.py              全部(选股、匹配、平衡性、信号面板)
      python3 scripts/pead_build_batches.py --signal-only 只重写信号面板及其伴随文件与封存核对(审查 P2 修复用;
                                                         不重写批次表 / 匹配变量 / 平衡性,保留它们"读收益之前"的落盘时间)
"""
from __future__ import annotations
import os, sys, json, time, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import TMP, LEDGER, ROOT
from afactor.panel import Panel
from afactor import earnings_events as EE, newdata as ND, fundamental as FU

SEED = 20261010
ARM_CODE = {"B": 1, "C": 2, "E_rand": 3}
TOL2 = 0.20
R_RAND = 10
K_CTRL = 3
WINDOWS = {"warmup2020": ("2020-01-01", "2020-12-31"), "dev": ("2021-01-01", "2023-12-31"),
           "test": ("2024-01-01", "2025-12-31")}


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ 封存截断后的辅助原料
def sealed_daily_basic() -> pd.DataFrame:
    """daily_basic.csv 读入后立即截到 <= 2025-12-31,并塞进 newdata 的缓存,
    使 ND.raw_close / float_share / total_mv 只看到研究期行。"""
    d = pd.read_csv(ND.DATA / "daily_basic.csv",
                    usecols=["ts_code", "trade_date", "close", "turnover_rate",
                             "turnover_rate_f", "float_share", "circ_mv"])
    d["trade_date"] = ND._ymd(d["trade_date"])
    d = d[d["trade_date"] <= EE.CUTOFF].copy()                     # 封存截断
    ND._CACHE["db"] = d
    return d


def sealed_total_mv(dates, codes) -> pd.DataFrame:
    """ND.total_mv 的封存版:财报按 avail <= 2025-12-31 截断(方法同 pead_data_acceptance.panel_consistency)。"""
    orig = FU._load

    def _load_sealed(name, items):
        d = orig(name, items)
        return d[d["avail"] <= EE.CUTOFF]
    FU._load = _load_sealed
    FU._REP_CACHE.clear()
    try:
        mv = ND.total_mv(dates, codes)
    finally:
        FU._load = orig
        FU._REP_CACHE.clear()
    return mv


def pool_quantile(X: pd.DataFrame, Q: pd.DataFrame) -> np.ndarray:
    """同日合格池内分位(0~1];池内缺失 -> 0.5;池外 NaN。"""
    r = X.where(Q).rank(axis=1, pct=True)
    return r.where(~(Q & r.isna()), 0.5).where(Q).to_numpy(np.float64)


# ------------------------------------------------------------------ 匹配
def match_arm(treat_j, cand_j, qv, ind, arm_code, phase, T_pos, tol1, tol2=TOL2, k=K_CTRL):
    """返回 ctrl (k, nT) 股票列号(-1 = 匹配失败)、lvl (k, nT)、dist (k, nT)。"""
    nT, nC = len(treat_j), len(cand_j)
    ctrl = np.full((k, nT), -1, np.int64)
    pos = np.full((k, nT), -1, np.int64)
    lvl = np.zeros((k, nT), np.int8)
    dist = np.full((k, nT), np.nan)
    if nC == 0:
        return ctrl, lvl, dist
    qT, qC = qv[treat_j], qv[cand_j]
    iT, iC = ind[treat_j], ind[cand_j]
    for r in range(1, k + 1):
        rng = np.random.default_rng([SEED, arm_code, phase, T_pos, r])
        order = rng.permutation(nT)
        used = np.zeros(nC, bool)
        for i in order:
            avail = ~used
            for rr in range(r - 1):
                if pos[rr, i] >= 0:
                    avail[pos[rr, i]] = False             # 同一处理股的三个对照互不相同
            if not avail.any():
                continue
            dq = np.abs(qC - qT[i])
            mx = dq.max(axis=1)
            d = np.sqrt((dq * dq).sum(axis=1))
            same = iC == iT[i]
            chosen_level = 4
            mm = avail
            for L, m in ((1, same & (mx <= tol1)), (2, same & (mx <= tol2)), (3, mx <= tol1)):
                cand = m & avail
                if cand.any():
                    chosen_level, mm = L, cand
                    break
            ci = np.flatnonzero(mm)
            dd = d[ci]
            best = dd.min()
            ties = ci[dd <= best + 1e-12]
            c = int(ties[rng.integers(len(ties))]) if len(ties) > 1 else int(ties[0])
            used[c] = True
            pos[r - 1, i] = c
            ctrl[r - 1, i] = cand_j[c]
            lvl[r - 1, i] = chosen_level
            dist[r - 1, i] = best
    return ctrl, lvl, dist


SIGNAL_DOC = (
    "PEAD 信号面板(前缀 pead)。行 = 引擎信号行 t,值 = fu_op_yoy(营业利润单季对称同比,±2,越高越好),"
    "只在合格事件池(V ∈ [t-3, t+1] 且 t 日在合格池)非空。"
    "【只能用于 lag>=1 的成交口径(open10 / open5 / vwap1)】:第 t 行约 20% 的值在 t 日收盘之后、"
    "t+1 日 09:25 之前才公开(P == t 的盘后公告,或 P 为 t 之后的周末/节假日),"
    "在 tail1450 / tail1445 / close(lag=0,W[t] 在 t 日成交)下直接使用是前视;"
    "也不得把第 t 行当作 't 日收盘已知' 的特征(如与 t 日收盘统计量对齐、同日 IC)。"
    "lag=0 口径请先整体下移一行(第 t 行改用 t-1 行),或只用伴随文件 pead_signal_open10_events.parquet 中 "
    "known_at_t_close == True 的格子。推荐用 afactor.pead_engine.load_signal_panel(p, mode) 读取,它会做这项检查。"
    "检验结论:同风格对照之上没有可交易增量(E−B 证据不足),不建议默认给正的 alpha 权重。"
)


def load_qualified_event_pool(p, t00=None):
    """合格事件池(长表,每个 (T, 股票) 一行)与合格池矩阵。冻结输入的 md5 在这里核对。"""
    t00 = time.time() if t00 is None else t00
    frozen = json.load(open(LEDGER / "pead_frozen_inputs.json"))
    ev_md5 = md5(EE.EVENTS_PATH)
    assert ev_md5 == frozen["md5"]["cache/_tmp/pead_events.parquet"], "事件表与冻结版本不一致"
    comp_path = TMP / "comp_v5_score_open10.parquet"
    assert md5(comp_path) == frozen["md5"]["cache/_tmp/comp_v5_score_open10.parquet"], "COMP_v5 打分与冻结版本不一致"
    dates, codes = p.dates, p.codes
    Q = EE.qualified_pool(p)
    Qn = Q.to_numpy(bool)
    print(f"[{time.time()-t00:.0f}s] 合格池 {Qn.sum(1).mean():.0f} 只/日")
    ev = EE.load_events()
    pool = EE.event_pool(EE.main_events(ev))
    pool["i"] = dates.get_indexer(pool["t_signal"])
    pool["j"] = codes.get_indexer(pool["stock_code"])
    pool["T_pos"] = dates.get_indexer(pool["T"])
    ok = (pool["i"] >= 0) & (pool["j"] >= 0) & (pool["T_pos"] >= 0)
    pool = pool[ok].copy()
    pool = pool[Qn[pool["i"].to_numpy(), pool["j"].to_numpy()]].copy()
    assert (pool["T_pos"] == pool["i"] + 1).all(), "t_signal 必须是 T 的前一交易日"
    print(f"[{time.time()-t00:.0f}s] 合格事件池 股票-T 对 {len(pool)}")
    return ev, pool, Q, Qn, comp_path


def write_signal_panel(pool: pd.DataFrame, dates, codes) -> pd.DataFrame:
    """信号面板(下游指增优化器用)+ 长表伴随文件 + lag 口径核对表。

    为什么要伴随文件(审查 P2):面板第 t 行 = "t+1 日 09:31 成交前已知",不等于"t 日收盘已知"。
    P(公告日)与 t 的关系:
      P <  t:t 日收盘前已公开(V <= t);
      P == t:按保守时钟视为 t 日盘后公开(V = t+1),t 日收盘时未知;
      P >  t:P 是 t 之后的周末/节假日(V = t+1),t 日收盘时未知。
    lag=1 口径(open10/open5/vwap1)决策点是 t+1 日 09:25,三类都合规;lag=0 口径(tail/close)只有第一类合规。"""
    T, N = len(dates), len(codes)
    sig = np.full((T, N), np.nan, np.float32)
    sig[pool["i"].to_numpy(), pool["j"].to_numpy()] = pool["value"].to_numpy(np.float32)
    S = pd.DataFrame(sig, index=dates, columns=codes)
    S.attrs = dict(doc=SIGNAL_DOC, valid_exec_modes="open10,open5,vwap1", min_lag="1",
                   companion="cache/_tmp/pead_signal_open10_events.parquet")
    S.to_parquet(TMP / "pead_signal_open10.parquet")
    del sig, S
    L = pool[["t_signal", "T", "stock_code", "value", "P", "V", "days_since_V", "i", "j"]].copy()
    L = L.rename(columns={"i": "t_pos", "j": "code_pos"})
    L["value"] = L["value"].astype(np.float32)
    P_ = pd.DatetimeIndex(L["P"]); t_ = pd.DatetimeIndex(L["t_signal"])
    L["P_rel_t"] = np.where(P_ < t_, "P<t", np.where(P_ == t_, "P==t", "P>t"))
    L["known_at_t_close"] = (pd.DatetimeIndex(L["V"]) <= t_)          # 与 P < t 等价(V = P 后首个交易日)
    assert (L["known_at_t_close"] == (L["P_rel_t"] == "P<t")).all()
    assert (pd.DatetimeIndex(L["P"]) < pd.DatetimeIndex(L["T"])).all(), "P 必须早于成交日 T"
    assert pd.DatetimeIndex(L["T"]).max() <= EE.CUTOFF
    L = L.sort_values(["t_pos", "code_pos"]).reset_index(drop=True)
    L.attrs = dict(doc="pead_signal_open10.parquet 的长表伴随文件:每个非空格一行;"
                       "known_at_t_close = V <= t_signal(= P < t),lag=0 口径只能用这些格子。")
    L.to_parquet(TMP / "pead_signal_open10_events.parquet", index=False)
    n = len(L)
    rows = [dict(item="nonnull_cells", n=n, share=1.0)]
    for k in ("P<t", "P==t", "P>t"):
        rows.append(dict(item=f"P_rel_t:{k}", n=int((L["P_rel_t"] == k).sum()),
                         share=float((L["P_rel_t"] == k).mean())))
    rows.append(dict(item="not_known_at_t_close(P>=t)", n=int((~L["known_at_t_close"]).sum()),
                     share=float((~L["known_at_t_close"]).mean())))
    for d in range(5):
        rows.append(dict(item=f"V_minus_t:{1 - d}", n=int((L["days_since_V"] == d).sum()),
                         share=float((L["days_since_V"] == d).mean())))
    rows.append(dict(item="max_T", n=np.nan, share=np.nan, value=str(pd.DatetimeIndex(L["T"]).max().date())))
    M = pd.DataFrame(rows)
    M.to_csv(LEDGER / "pead_signal_panel_meta.csv", index=False)
    print(M.to_string())
    return L


def seal_audit(ev: pd.DataFrame, Qn: np.ndarray, dates, codes, db: pd.DataFrame) -> pd.DataFrame:
    """封存核对(审查 P2):
      1) daily_basic 截断后 newdata 缓存的最大日期;
      2) 主指标首次披露里状态为 own_revised_sealed 的行(只有 2026 年修订版本,首次披露原值不可恢复)——
         它们不进入主事件:'被修订'这一事实本身是 2026 年的信息,属于事件表构造的固有选择(数值从未读入)。
         统计口径与 PEAD_DATA_ACCEPTANCE(pead_event_conflicts.csv)一致,另给 V 在 2021–2025 且
         V 前一交易日在合格池的子集。"""
    rows = [dict(item="daily_basic_rows_after_truncation", n=len(db), value=str(db["trade_date"].max().date()))]
    f = ev[(ev["source"] == "定期报告") & (ev["indicator"] == EE.MAIN_INDICATOR) & (ev["version"] == 1)]
    rs = f[f["value_status"] == "own_revised_sealed"]
    rows.append(dict(item="main_indicator_first_disclosure_rows", n=len(f)))
    rows.append(dict(item="own_revised_sealed_rows", n=len(rs), share=len(rs) / max(len(f), 1)))
    g = rs[rs["V"].notna()].copy()
    g["j"] = codes.get_indexer(g["stock_code"]); g["vpos"] = dates.get_indexer(g["V"])
    g = g[(g["j"] >= 0) & (g["vpos"] >= 1) & (g["V"] >= pd.Timestamp("2021-01-01")) & (g["V"] <= EE.CUTOFF)]
    q = Qn[g["vpos"].to_numpy() - 1, g["j"].to_numpy()]
    rows.append(dict(item="own_revised_sealed_V2021_2025_in_panel", n=len(g)))
    rows.append(dict(item="own_revised_sealed_V2021_2025_qualified_at_Vm1", n=int(q.sum())))
    A = pd.DataFrame(rows)
    A.to_csv(LEDGER / "pead_seal_audit.csv", index=False)
    print(A.to_string())
    return A


def main_signal_only():
    """只重写信号面板、伴随文件与封存核对(不读任何收益,不动批次表)。"""
    t00 = time.time()
    db = sealed_daily_basic()
    p = Panel()
    ev, pool, Q, Qn, _ = load_qualified_event_pool(p, t00)
    write_signal_panel(pool, p.dates, p.codes)
    seal_audit(ev, Qn, p.dates, p.codes, db)
    print(f"完成 [{time.time()-t00:.0f}s]")


def main():
    t00 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    T, N = len(dates), len(codes)
    ev, pool, Q, Qn, comp_path = load_qualified_event_pool(p, t00)

    # ---- 信号面板(下游指增优化器用):行 = 信号行 t,值 = fu_op_yoy ----
    write_signal_panel(pool, dates, codes)

    # ---- 匹配变量 ----
    db = sealed_daily_basic()
    seal_audit(ev, Qn, dates, codes, db)
    ind_lab = ND.sw_industry_labels(dates[:1], codes).iloc[0].to_numpy(np.int64)   # 当前快照,-1 = 未知
    mv = sealed_total_mv(dates, codes)
    circ = ND.circ_mv(dates, codes)
    logmv = np.log(mv.where(mv > 0)).fillna(np.log(circ.where(circ > 0)))
    logadv = np.log(p["adv20"].where(p["adv20"] > 0))
    vol = p["ret_c2c"].rolling(20, min_periods=15).std()
    qv = np.stack([pool_quantile(logmv, Q), pool_quantile(logadv, Q), pool_quantile(vol, Q)], axis=2)
    raw_vals = np.stack([logmv.to_numpy(np.float64), logadv.to_numpy(np.float64),
                         vol.to_numpy(np.float64)], axis=2)
    miss_mv = (Q & logmv.isna()).to_numpy().sum(); miss_vol = (Q & vol.isna()).to_numpy().sum()
    print(f"[{time.time()-t00:.0f}s] 匹配变量就绪;池内市值缺失 {miss_mv} 股票日,波动缺失 {miss_vol}")
    del mv, circ

    # ---- COMP_v5 最差 20% ----
    S = pd.read_parquet(comp_path).reindex(index=dates, columns=codes)
    Sq = S.where(Q)
    pct = Sq.rank(axis=1, method="average") .div(Sq.notna().sum(axis=1), axis=0)
    worst = (pct <= 0.20).fillna(False).to_numpy(bool)       # 缺失 -> 不在最差 20%
    score_ok = Sq.notna().to_numpy(bool)
    del S, Sq, pct

    # ---- 逐 T 选股与匹配 ----
    rows, meta, mvars = [], [], []
    pool = pool.sort_values(["T_pos", "value", "stock_code"], ascending=[True, False, True])
    for T_pos, g in pool.groupby("T_pos", sort=True):
        t = T_pos - 1
        phase = int(T_pos % 5)
        n = len(g)
        rec = dict(T_pos=T_pos, t_pos=t, date_T=dates[T_pos], phase=phase, n_pool=n, n_sel=0,
                   n_F_removed=0, n_score_missing=0)
        if n < 5:
            meta.append(rec); continue
        n_sel = -(-n // 5)
        sel = g.iloc[:n_sel]
        ej = sel["j"].to_numpy(np.int64)
        w = min(1.0 / n_sel, 0.05)
        rec.update(n_sel=n_sel, w=w, tgt_cash=1.0 - n_sel * w)
        for s, jj in enumerate(ej):
            rows.append(("E", 0, T_pos, t, phase, s, jj, w, 0, np.nan))
        # F
        rm = worst[t, ej]
        rec["n_F_removed"] = int(rm.sum()); rec["n_score_missing"] = int((~score_ok[t, ej]).sum())
        for s in np.flatnonzero(~rm):
            rows.append(("F", 0, T_pos, t, phase, int(s), ej[s], w, 0, np.nan))
        # E_rand
        k_rm = int(rm.sum())
        for r in range(R_RAND):
            rng = np.random.default_rng([SEED, ARM_CODE["E_rand"], phase, T_pos, r])
            drop = rng.choice(n_sel, size=k_rm, replace=False) if k_rm > 0 else np.array([], int)
            keep = np.setdiff1d(np.arange(n_sel), drop)
            for s in keep:
                rows.append(("E_rand", r, T_pos, t, phase, int(s), ej[s], w, 0, np.nan))
        # 候选
        candB = np.flatnonzero(Qn[t]); candB = np.setdiff1d(candB, ej)
        candC = np.setdiff1d(g["j"].to_numpy(np.int64), ej)
        q_t = qv[t]
        for arm, cand in (("B", candB), ("C", candC)):
            for tol_tag, tol1 in (("tol10", 0.10), ("tol05", 0.05)):
                ctrl, lvl, dist = match_arm(ej, cand, q_t, ind_lab, ARM_CODE[arm], phase, T_pos, tol1)
                arm_name = arm if tol_tag == "tol10" else f"{arm}@tol05"
                for r in range(K_CTRL):
                    for s in range(n_sel):
                        rows.append((arm_name, r + 1, T_pos, t, phase, s, int(ctrl[r, s]), w,
                                     int(lvl[r, s]), dist[r, s]))
                        if tol_tag == "tol10" and arm == "B" and not rm[s]:
                            rows.append(("B_F", r + 1, T_pos, t, phase, s, int(ctrl[r, s]), w,
                                         int(lvl[r, s]), dist[r, s]))
                    if tol_tag == "tol05" and arm == "B":
                        for s in np.flatnonzero(~rm):
                            rows.append(("B_F@tol05", r + 1, T_pos, t, phase, int(s), int(ctrl[r, s]), w,
                                         int(lvl[r, s]), dist[r, s]))
        meta.append(rec)
        if T_pos % 100 == 0:
            print(f"  T_pos {T_pos} n_pool {n} n_sel {n_sel} [{time.time()-t00:.0f}s]", flush=True)

    B = pd.DataFrame(rows, columns=["arm", "rep", "T_pos", "t_pos", "phase", "slot", "j", "w", "level", "dist"])
    B["date_T"] = dates[B["T_pos"].to_numpy()]
    B["t_signal"] = dates[B["t_pos"].to_numpy()]
    ok = B["j"] >= 0
    B["code"] = pd.Series(np.where(ok, codes.to_numpy()[B["j"].clip(lower=0)], None), index=B.index)
    B.to_parquet(TMP / "pead_batches.parquet", index=False)
    M = pd.DataFrame(meta)
    M.to_parquet(TMP / "pead_batch_meta.parquet", index=False)
    print(f"[{time.time()-t00:.0f}s] 批次表 {len(B)} 行,写 pead_batches.parquet")

    # ---- 匹配变量落盘(平衡性复核)----
    jj = B["j"].clip(lower=0).to_numpy(); ii = B["t_pos"].to_numpy()
    V = pd.DataFrame({"arm": B["arm"], "rep": B["rep"], "t_signal": B["t_signal"], "slot": B["slot"],
                      "T_pos": B["T_pos"], "ok": ok,
                      "ind": np.where(ok, ind_lab[jj], -99),
                      "logmv": np.where(ok, raw_vals[ii, jj, 0], np.nan),
                      "logadv": np.where(ok, raw_vals[ii, jj, 1], np.nan),
                      "vol": np.where(ok, raw_vals[ii, jj, 2], np.nan),
                      "level": B["level"]})
    V.to_parquet(TMP / "pead_match_vars.parquet", index=False)

    # ---- 平衡性 ----
    bal = balance_table(V)
    bal.to_csv(LEDGER / "pead_balance.csv", index=False)
    print(bal.to_string())

    # ---- 按年批次统计 ----
    M["year"] = pd.DatetimeIndex(M["date_T"]).year
    act = M[M["n_sel"] > 0]
    ys = act.groupby("year").agg(active_T=("T_pos", "size"), n_pool_mean=("n_pool", "mean"),
                                 n_sel_mean=("n_sel", "mean"), n_sel_max=("n_sel", "max"),
                                 tgt_cash_mean=("tgt_cash", "mean"),
                                 F_removed_share=("n_F_removed", lambda s: s.sum() / act.loc[s.index, "n_sel"].sum()),
                                 score_missing_share=("n_score_missing",
                                                      lambda s: s.sum() / act.loc[s.index, "n_sel"].sum()))
    ys["all_T"] = pd.Series(dates.year).value_counts().reindex(ys.index)      # 全部交易日(成交日)
    ys["active_T_share"] = ys["active_T"] / ys["all_T"]
    tot = dict(year="all", active_T=len(act), n_pool_mean=act["n_pool"].mean(), n_sel_mean=act["n_sel"].mean(),
               n_sel_max=act["n_sel"].max(), tgt_cash_mean=act["tgt_cash"].mean(),
               F_removed_share=act["n_F_removed"].sum() / act["n_sel"].sum(),
               score_missing_share=act["n_score_missing"].sum() / act["n_sel"].sum(),
               all_T=len(dates), active_T_share=len(act) / len(dates))
    ys = pd.concat([ys.reset_index(), pd.DataFrame([tot])], ignore_index=True)
    ys.to_csv(LEDGER / "pead_batch_summary.csv", index=False)
    print(ys.to_string())
    print(f"完成 [{time.time()-t00:.0f}s]")


def balance_table(V: pd.DataFrame) -> pd.DataFrame:
    """处理组 = E 槽位;对照 = 各臂全部复制组合的对照(池化)。按窗口算 SMD、行业占比最大差、层级占比、失败数。"""
    out = []
    tv = V[V["arm"] == "E"]
    for arm_t, arm_c in (("E", "B"), ("E", "C"), ("F", "B_F"), ("E", "B@tol05"), ("E", "C@tol05"),
                         ("F", "B_F@tol05")):
        trt = V[V["arm"] == arm_t]
        ctl = V[V["arm"] == arm_c]
        for wn, (a, b) in WINDOWS.items():
            mt = (trt["t_signal"] >= a) & (trt["t_signal"] <= b)
            mc = (ctl["t_signal"] >= a) & (ctl["t_signal"] <= b)
            x, y = trt[mt], ctl[mc]
            yo = y[y["ok"]]
            rec = dict(control=arm_c, treated=arm_t, window=wn, n_treated_slots=len(x),
                       n_control_slots=len(y), n_failed=int((~y["ok"]).sum()),
                       fail_share=float((~y["ok"]).mean()) if len(y) else np.nan)
            for v in ("logmv", "logadv", "vol"):
                xt, xc = x[v].dropna(), yo[v].dropna()
                s = np.sqrt((xt.var() + xc.var()) / 2.0)
                rec[f"smd_{v}"] = float((xt.mean() - xc.mean()) / s) if s > 0 else np.nan
                rec[f"mean_t_{v}"] = float(xt.mean()); rec[f"mean_c_{v}"] = float(xc.mean())
            st = x["ind"].value_counts(normalize=True)
            sc = yo["ind"].value_counts(normalize=True)
            dif = st.reindex(st.index.union(sc.index), fill_value=0) - sc.reindex(st.index.union(sc.index), fill_value=0)
            rec["ind_maxdiff_pp"] = float(dif.abs().max() * 100) if len(dif) else np.nan
            for L in (1, 2, 3, 4):
                rec[f"level{L}_share"] = float((yo["level"] == L).mean()) if len(yo) else np.nan
            rec["pass"] = bool(all(abs(rec[f"smd_{v}"]) <= 0.10 for v in ("logmv", "logadv", "vol"))
                               and rec["ind_maxdiff_pp"] <= 5.0)
            out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    if "--signal-only" in sys.argv[1:]:
        main_signal_only()
    else:
        main()
