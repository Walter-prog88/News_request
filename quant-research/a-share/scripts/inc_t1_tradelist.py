"""INC-T1 第一层:固定成交清单的逐笔重定价(预注册 §4.1、§5.2、§6、§7)。

步骤
  rebuild <strategy>  参考运行:现行口径 open10、恒定 AUM、Y=1 一跳、POV 截量,5 个相位分别 ST.simulate(record=True),
                      X ∈ {1e7, 1e8, 1e9}。逐笔成交清单 (phase, t, i, Δw) 落盘 cache/_tmp/inc_t1_trades_<strategy>_<X>.npz,
                      §3.3 第 5 条校验 -> research/ledger/inc_t1_tradelist_checks.csv(追加)。
  layer1 <strategy>   对清单逐笔重定价:按"腿"(窗口 W、成交日偏移 k、数量比例 φ、价格口径、截量口径)算逐笔分项,
                      按 (信号日 t, 买/卖) 汇总成日序列 -> cache/_tmp/inc_t1_l1legs_<strategy>.parquet;
                      方案 = 腿的线性组合(买卖分离 / 分批 / 延迟),在 summarize 里组装。
  paths               买卖清单相对决策价的日内路径(九个时点)减同窗池子路径 -> research/ledger/inc_t1_layer1_paths.csv
  summarize           方案网格与判定 -> research/ledger/inc_t1_layer1_grid.csv / inc_t1_layer1_sides.csv / inc_t1_summary.csv

审查 P0 修复(2026-10-11 审查后):窗口内整窗口封死在涨 / 跌停价的腿按"不可成交"处理(inc_t1_common.window_seal):
  seal = dir   买单被整窗口涨停封死挡住、卖单被整窗口跌停封死挡住(方向性,物理上正确;主口径)
  seal = any   涨停或跌停封死都挡买卖双方(与引擎 can 的一字板约定一致,也是第二层唯一能表达的口径)
  seal = tight 方向性,阈值收紧到距涨跌停幅度 0.1pp
  seal = none  修复前口径(只剔除缺价 / 零成交额),保留作前后对照(修复前的账本 inc_t1_l1legs_<strategy>.parquet)
被挡的腿与缺价的腿同样处理:整条腿顺延到次日同窗口(只顺延一次,预注册规则),分项按次日价、次日的价差与冲击。

清单捕获:引擎每个成交日调用 cost_model.extra_cost(te, |Δw|, capital, nav);包装成本模型在该调用里从调用帧读取
有符号的 delta 与信号日 t(只读,不改引擎状态,结果与不捕获时逐位相同 —— 由校验 (c) 复核)。这等价于预注册
§3.3 第 5 条的"在 record 里加 rec_delta",但不改库代码(见报告"偏离预注册"第 4 条)。

逐笔分项(全部以占净值比例计,正 = 成本;s = sign(Δw),w = |Δw|,决策价 Pd = close_px[t],池 = 信号日 t 的 V1000 池):
  一条腿 (W, k, φ):数量 φw 在成交日 t+k 的窗口 W 成交;可成交量 cap = 10% × amt_W[t+k] / X(POV)或事前估计;
    成交比例 f = min(1, cap/φw)(价格缺失或窗口成交额为 0 时 f = 0),未成交部分 (1-f) 在次日 t+k+1 同窗口成交(不再递归)。
    r_j = P_W[t+j]/Pd - 1,M_j = 池等权均值(P_W[t+j]/Pd - 1)
    px_rel  = φw·s·(r_k - M_k)           个股相对同窗池子的已实现成本(按全部当日成交计)  -> 可信增量 (A)
    px_mkt  = φw·s·M_k                   池子级日内漂移(不计入增量)
    unf_tot = φw·s·(1-f)·(r_{k+1} - r_k) 未成交成本(延后一日的价格差)                    -> 可信增量 (D)
    unf_mkt = φw·s·(1-f)·(M_{k+1} - M_k) 其中的池子级部分(判据 6 的严格口径再剔除)
    spread  = φw·[f·hs(t+k) + (1-f)·hs(t+k+1)]                                           -> (B)
    impact  = c(t+k)·(fφw)^1.5·s_W(t+k) + c(t+k+1)·((1-f)φw)^1.5·s_W(t+k+1),c = Y·σ·sqrt(X/ADV)  -> (B)
    当日价格缺失(不可成交)时整条腿按次日成交:px_* 用 r_{k+1}/M_{k+1},unf = 0(计数 n_unfillable)。
  可信增量 ΔC = 基线(W1 同口径)成本 − 方案成本 的 (px_rel + spread + impact + unf_tot);> 0 = 方案更省。
  佣金与印花税各方案相同,不进入差。
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t1_common as C                                   # noqa: E402
from inc_t1_common import ST, IM, Panel, LEDGER, TMP, TRADING_DAYS, REPORTS   # noqa: E402
import inc_t1_build_windows as BW                           # noqa: E402

MP = 0.10
N_PHASE = C.REBALANCE


def trades_path(strat, X):
    return TMP / f"inc_t1_trades_{strat}_{int(X)}.npz"


# ====================================================================== 1 清单重建
class CaptureCost:
    """恒定 AUM 成本(与 cap_capacity_curve.ConstAUMCost 同式:nav 固定为 1)+ 逐笔成交捕获。"""

    def __init__(self, base: IM.CostModel):
        self.base = base
        self.rows = []

    def extra_cost(self, te, abs_delta, capital, nav):
        L = sys._getframe(1).f_locals
        delta = L["delta"]
        assert int(L["te"]) == int(te)
        nz = np.flatnonzero(delta != 0.0)
        self.rows.append((int(L["t"]), nz.astype(np.int32), delta[nz].astype(np.float64)))
        return self.base.extra_cost(te, abs_delta, capital, 1.0)

    def describe(self):
        d = self.base.describe()
        d["aum_mode"] = "const"
        return d


def cmd_rebuild(strat: str):
    p = Panel()
    scores = C.load_scores(p)
    W = C.strategy_weights(p, scores, strat)
    mask = C.sim_mask(p, strat)
    cm = IM.CostModel(p, impact_y=1.0)
    px = p.raw("twap_open10").to_numpy(np.float64)
    T, N = px.shape
    checks = []
    for X in C.AUMS:
        t0 = time.time()
        PH, TT, II, DW = [], [], [], []
        for ph in range(N_PHASE):
            cap = CaptureCost(cm)
            res = ST.simulate(W, p, "open10", X, rebalance=C.REBALANCE, date_mask=mask, phase=ph, cost_model=cap,
                              participation_on_nav=False, record=True)
            # (c) 捕获不改变路径:同配置用 ConstAUMCost 再跑一遍逐位比对
            ref = ST.simulate(W, p, "open10", X, rebalance=C.REBALANCE, date_mask=mask, phase=ph,
                              cost_model=C.ConstAUMCost(cm), participation_on_nav=False, record=False)
            same = bool(np.array_equal(ref["rets"].to_numpy(), res["rets"].to_numpy()))
            daily = res["record"]["daily"]
            H = res["record"]["holdings"].astype(np.float64)
            idx = p.dates.get_indexer(daily.index)
            # 捕获清单 -> 每日 Σ|Δw|/2
            cap_turn = pd.Series(0.0, index=p.dates)
            for (t, nz, d) in cap.rows:
                cap_turn.iloc[t] += np.abs(d).sum() / 2
                PH.append(np.full(len(nz), ph, np.int8)); TT.append(np.full(len(nz), t, np.int32))
                II.append(nz); DW.append(d)
            diff_cap = float(np.max(np.abs(cap_turn.reindex(daily.index).to_numpy() - daily["turnover"].to_numpy())))
            # (b) 预注册公式:h_pre[t] = newh[t-1]·(1+r[te-1])/(1+port_r[t-1]),再按 hs>1 归一;Δw = newh[t] - h_pre[t]
            ret = daily["ret"].to_numpy()
            turn_formula = np.zeros(len(idx))
            max_dw_err = 0.0
            dw_cap = {t: (nz, d) for (t, nz, d) in cap.rows}
            for j in range(len(idx)):
                t = idx[j]
                if j == 0:
                    hpre = np.zeros(N)
                else:
                    te_prev = idx[j - 1] + 1
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ok = np.isfinite(px[te_prev]) & np.isfinite(px[te_prev + 1]) & (px[te_prev] > 0)
                        r = np.where(ok, px[te_prev + 1] / px[te_prev] - 1.0, 0.0)
                    g = H[j - 1] * (1.0 + np.nan_to_num(r))
                    hpre = g / (1.0 + ret[j - 1]) if (1.0 + ret[j - 1]) > 1e-9 else g
                    hs = hpre.sum()
                    if hs > 1.0 + 1e-12:
                        hpre = hpre / hs
                dwf = H[j] - hpre
                turn_formula[j] = np.abs(dwf).sum() / 2
                dd = np.zeros(N)
                if t in dw_cap:
                    nz, d = dw_cap[t]
                    dd[nz] = d
                max_dw_err = max(max_dw_err, float(np.max(np.abs(dwf - dd))))
            diff_formula = float(np.max(np.abs(turn_formula - daily["turnover"].to_numpy())))
            checks += [
                dict(strategy=strat, X=X, phase=ph, check="captured_turnover_vs_record_maxabs", value=diff_cap,
                     threshold=1e-9, passed=diff_cap <= 1e-9, note="捕获的 Σ|Δw|/2 vs record.turnover(主口径清单)"),
                dict(strategy=strat, X=X, phase=ph, check="formula_turnover_vs_record_maxabs", value=diff_formula,
                     threshold=1e-9, passed=diff_formula <= 1e-9,
                     note="预注册公式(record 持仓为 float32)重建的 Σ|Δw|/2 vs record.turnover;不过则用捕获清单"),
                dict(strategy=strat, X=X, phase=ph, check="formula_vs_captured_dw_maxabs", value=max_dw_err,
                     threshold=np.nan, passed=True, note="信息性:float32 持仓带来的逐笔 Δw 误差上界"),
                dict(strategy=strat, X=X, phase=ph, check="capture_path_bitwise_same", value=float(same),
                     threshold=1.0, passed=same, note="捕获包装不改变引擎路径(与 ConstAUMCost 逐位相同)"),
            ]
            print(f"  [{strat} X={X:.0e} ph{ph}] 交易日 {len(cap.rows)} 笔数 {sum(len(r[1]) for r in cap.rows)} "
                  f"cap差 {diff_cap:.1e} 公式差 {diff_formula:.1e} Δw误差 {max_dw_err:.1e} 同路径 {same}", flush=True)
        np.savez(trades_path(strat, X), phase=np.concatenate(PH), t=np.concatenate(TT), i=np.concatenate(II),
                 dw=np.concatenate(DW))
        print(f"[{strat} X={X:.0e}] 清单 {sum(len(x) for x in DW)} 笔 {time.time() - t0:.0f}s", flush=True)
    f = LEDGER / "inc_t1_tradelist_checks.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if len(old):
        old = old[old.strategy != strat]
    pd.concat([old, pd.DataFrame(checks)], ignore_index=True).to_csv(f, index=False)


# ====================================================================== 2 腿的逐笔分项
LEGS = sorted({("W%d" % k, 1, 1.0) for k in range(12)}
              | {(w, 1, 0.5) for b in C.BATCHES for w in b}
              | {("W1", 2, 1.0), ("W1", 2, 0.5)})
KINDS = ["vwap", "twap"]
CAPVARS = ["pov", "exante"]
COMPS = ["px_rel", "px_mkt", "unf_tot", "unf_mkt", "spread", "imp_pool", "imp_one", "imp_stock",
         "w", "w_capped", "w_unfilled", "n_unfillable", "n_sealed", "w_sealed"]


class Ctx:
    """一次 layer1 运行共用的面板(float64)与缓存。"""

    def __init__(self, p: Panel):
        self.p = p
        self.T, self.N = len(p.dates), len(p.codes)
        self.close = p.raw("close_px").to_numpy(np.float64)
        self.U = ST.universe(p, C.VERSION).to_numpy(bool)
        cm = IM.CostModel(p, impact_y=1.0)
        self.hs = cm._hs.astype(np.float64)
        self.sig = cm._sigma.astype(np.float64)
        self.adv = cm._adv.astype(np.float64)
        self.s_year = C.window_scales(p)
        self.years = np.asarray(p.dates.year)
        self._px, self._amt, self._amt_ex, self._M, self._sday = {}, {}, {}, {}, {}

    def px(self, w, kind):
        key = (w, "twap" if w in BW.SINGLE_BAR else kind)
        if key not in self._px:
            self._px[key] = C.win_px(self.p, w, key[1]).to_numpy(np.float32)      # 原料即 float32,不损失精度
        return self._px[key]

    def amt(self, w):
        if w not in self._amt:
            self._amt[w] = C.win_amt(self.p, w).to_numpy(np.float32)
        return self._amt[w]

    def amt_exante(self, w):
        """10% 上限的事前估计口径:截至 te-1 的 20 日窗口成交额中位数(min_periods 10;与 cap_capacity_curve 同)。
        估计缺失或 <= 0 时退回实际值(不改变可成交集合)。"""
        if w not in self._amt_ex:
            a = pd.DataFrame(self.amt(w).astype(np.float64))
            est = a.rolling(20, min_periods=10).median().shift(1).to_numpy()
            act = self.amt(w)
            good = np.isfinite(est) & (est > 0)
            self._amt_ex[w] = np.where(good & (act > 0), est, act).astype(np.float32)
        return self._amt_ex[w]

    def M(self, w, kind, k):
        """M_k[t] = 池(信号日 t 的 V1000)等权均值 (P_W[t+k]/close[t] - 1)。"""
        key = (w, kind if w not in BW.SINGLE_BAR else "twap", k)
        if key not in self._M:
            P = self.px(w, kind)
            out = np.full(self.T, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                R = P[k:].astype(np.float64) / self.close[:self.T - k] - 1.0
            Uk = self.U[:self.T - k] & np.isfinite(R)
            s = np.where(Uk, R, 0.0).sum(axis=1)
            n = Uk.sum(axis=1)
            out[:self.T - k] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            self._M[key] = out
        return self._M[key]

    def s_pool(self, w):
        return np.array([self.s_year.loc[y, w] for y in self.years], np.float64)

    def s_stock(self, w):
        if w not in self._sday:
            self._sday[w] = C.stockday_scale(self.p, w, self.s_year).astype(np.float32)
        return self._sday[w]


def leg_components(ctx: Ctx, tr: dict, X: float, leg: tuple, kind: str, capvar: str, seal: str = "none") -> dict:
    """逐笔分项 -> 按 (t, side) 的日汇总(已除以相位数 = 相位平均)。返回 {comp: ndarray (T, 2)},side 0 = 买, 1 = 卖。
    seal:none / dir / any / tight(见模块文档);被挡的腿与缺价的腿同样顺延到次日同窗口。"""
    w_name, k, phi = leg
    t, i, dw = tr["t"], tr["i"], tr["dw"]
    N, T = ctx.N, ctx.T
    s = np.sign(dw)
    w = np.abs(dw) * phi
    P = ctx.px(w_name, kind)
    A = ctx.amt_exante(w_name) if capvar == "exante" else ctx.amt(w_name)
    # 逐笔取值后转 float64 再运算
    Pd = ctx.close[t, i]
    tk, tk1 = np.minimum(t + k, T - 1), np.minimum(t + k + 1, T - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rk = P[tk, i].astype(np.float64) / Pd - 1.0
        rk1 = P[tk1, i].astype(np.float64) / Pd - 1.0
    Mk = ctx.M(w_name, kind, k)[t]
    Mk1 = ctx.M(w_name, kind, k + 1)[t]
    ak = A[tk, i].astype(np.float64)
    fill_ok = np.isfinite(rk) & np.isfinite(ak) & (ak > 0)
    blocked = C.blocked_mask(ctx.p, w_name, seal, dw > 0, tk, i) if seal != "none" else np.zeros(len(t), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        capw = MP * ak / X
        f = np.where(fill_ok, np.minimum(1.0, capw / np.maximum(w, 1e-300) * (1.0 + 1e-6)), 0.0)
    f = np.clip(f, 0.0, 1.0)
    nxt_ok = np.isfinite(rk1) & np.isfinite(Mk1)
    # 不可成交(缺价 / 零成交额 / 整窗口封板):整条腿按次日成交;次日也缺价 -> 记为当日成交(f=1)
    unfillable = ~fill_ok | blocked
    f = np.where(~fill_ok & ~nxt_ok, 1.0, f)
    f = np.where(fill_ok & ~nxt_ok, 1.0, f)
    r_same = np.where(unfillable & nxt_ok, rk1, rk)
    m_same = np.where(unfillable & nxt_ok, Mk1, Mk)
    f_eff = np.where(unfillable & nxt_ok, 1.0, f)               # 不可成交的腿:分项按次日价,unf 记 0
    both_nan = ~np.isfinite(r_same)
    r_same = np.where(both_nan, 0.0, r_same); m_same = np.where(both_nan | ~np.isfinite(m_same), 0.0, m_same)
    px_rel = w * s * (r_same - m_same)
    px_mkt = w * s * m_same
    d_r = np.where(f_eff < 1.0, np.nan_to_num(rk1 - rk), 0.0)
    d_m = np.where(f_eff < 1.0, np.nan_to_num(Mk1 - Mk), 0.0)
    unf_tot = w * s * (1.0 - f_eff) * d_r
    unf_mkt = w * s * (1.0 - f_eff) * d_m
    # 执行日:不可成交的腿全部在 t+k+1
    e1 = np.where(unfillable & nxt_ok, tk1, tk)
    fq = f_eff * w
    uq = (1.0 - f_eff) * w
    spread = fq * ctx.hs[e1, i] + uq * ctx.hs[tk1, i]
    c1 = ctx.sig[e1, i] * np.sqrt(X / ctx.adv[e1, i])
    c2 = ctx.sig[tk1, i] * np.sqrt(X / ctx.adv[tk1, i])
    base1 = c1 * fq ** 1.5
    base2 = c2 * uq ** 1.5
    sp = ctx.s_pool(w_name)
    ss = ctx.s_stock(w_name)
    imp_pool = base1 * sp[e1] + base2 * sp[tk1]
    imp_one = base1 + base2
    imp_stock = base1 * ss[e1, i].astype(np.float64) + base2 * ss[tk1, i].astype(np.float64)
    side = (dw < 0).astype(np.int64)
    key = t.astype(np.int64) * 2 + side
    out = {}
    vals = dict(px_rel=px_rel, px_mkt=px_mkt, unf_tot=unf_tot, unf_mkt=unf_mkt, spread=spread,
                imp_pool=imp_pool, imp_one=imp_one, imp_stock=imp_stock, w=w,
                w_capped=np.where(fill_ok & ~blocked & (f < 1.0), w, 0.0), w_unfilled=uq,
                n_unfillable=unfillable.astype(np.float64), n_sealed=blocked.astype(np.float64),
                w_sealed=np.where(blocked, w, 0.0))
    for c_, v in vals.items():
        out[c_] = (np.bincount(key, weights=v, minlength=2 * T)[:2 * T] / N_PHASE).reshape(T, 2)
    return out


SEAL_RUNS = [("dir", KINDS, CAPVARS), ("any", ["vwap"], ["pov"]), ("tight", ["vwap"], ["pov"])]


def legs_path(strat, seal_fix: bool = True):
    """修复后的腿账本(seal ∈ dir / any / tight);修复前的账本(seal = none)是 inc_t1_l1legs_<strategy>.parquet。"""
    return TMP / (f"inc_t1_l1legs_{strat}_seal.parquet" if seal_fix else f"inc_t1_l1legs_{strat}.parquet")


def cmd_layer1(strat: str, seal_fix: bool = True):
    p = Panel()
    ctx = Ctx(p)
    rows = []
    t00 = time.time()
    out_f = legs_path(strat, seal_fix)
    runs = SEAL_RUNS if seal_fix else [("none", KINDS, CAPVARS)]
    for seal, kinds, capvars in runs:
        for kind in kinds:
            ctx._px = {k_: v for k_, v in ctx._px.items() if k_[1] == kind or k_[0] in BW.SINGLE_BAR}
            for X in C.AUMS:
                z = np.load(trades_path(strat, X))
                tr = dict(t=z["t"].astype(np.int64), i=z["i"].astype(np.int64), dw=z["dw"])
                print(f"[{strat} seal={seal} {kind} X={X:.0e}] {len(tr['dw'])} 笔", flush=True)
                for capvar in capvars:
                    for leg in LEGS:
                        out = leg_components(ctx, tr, X, leg, kind, capvar, seal)
                        for side in (0, 1):
                            df = pd.DataFrame({c_: out[c_][:, side] for c_ in COMPS}, index=p.dates)
                            df["side"] = "buy" if side == 0 else "sell"
                            df["X"], df["kind"], df["capvar"], df["seal"] = X, kind, capvar, seal
                            df["leg"] = f"{leg[0]}|k{leg[1]}|{leg[2]:g}"
                            rows.append(df.reset_index().rename(columns={"index": "date"}))
                    print(f"  {seal}/{kind}/{capvar}/X={X:.0e} 完成 累计 {time.time() - t00:.0f}s", flush=True)
                del tr, z
    D = pd.concat(rows, ignore_index=True)
    for c_ in COMPS:
        D[c_] = D[c_].astype(np.float64)
    D.to_parquet(out_f)
    print(f"layer1 {strat} done {time.time() - t00:.0f}s -> {out_f}", flush=True)


# ====================================================================== 3 方案组装与统计
def scheme_legs(name: str) -> dict:
    """方案 -> {side: [leg 名]}。"""
    if name.startswith("S:"):
        a, b = name.split("|")
        sw, bw = a[2:], b[2:]
        return {"sell": [f"{sw}|k1|1"], "buy": [f"{bw}|k1|1"]}
    if "+" in name:
        a, b = name.split("+")
        legs = [f"{a}|k1|0.5", f"{b}|k1|0.5"]
        return {"buy": legs, "sell": legs}
    if name == "D1":
        return {"buy": ["W1|k2|1"], "sell": ["W1|k2|1"]}
    if name == "D2":
        return {"buy": ["W1|k1|0.5", "W1|k2|0.5"], "sell": ["W1|k1|0.5", "W1|k2|0.5"]}
    return {"buy": [f"{name}|k1|1"], "sell": [f"{name}|k1|1"]}


SCHEMES = [f"W{k}" for k in range(12)] + [f"S:{a}|B:{b}" for a in C.SPLIT_SET for b in C.SPLIT_SET] \
    + [f"{a}+{b}" for a, b in C.BATCHES] + ["D1", "D2"]
SCHEME_TYPE = {s: ("single" if s.startswith("W") and "+" not in s else "split" if s.startswith("S:") else
                   "batch" if "+" in s else "delay") for s in SCHEMES}
VARIANTS = {   # 名称 -> (kind, capvar, impact 列, Y, spread_mult, seal);主口径 main = 整窗口封板按不可成交顺延(seal = dir)
    "main": ("vwap", "pov", "imp_pool", 1.0, 1.0, "dir"),
    "Y0.5": ("vwap", "pov", "imp_pool", 0.5, 1.0, "dir"),
    "Y1.5": ("vwap", "pov", "imp_pool", 1.5, 1.0, "dir"),
    "twap": ("twap", "pov", "imp_pool", 1.0, 1.0, "dir"),
    "s_one": ("vwap", "pov", "imp_one", 1.0, 1.0, "dir"),
    "s_stock": ("vwap", "pov", "imp_stock", 1.0, 1.0, "dir"),
    "exante": ("vwap", "exante", "imp_pool", 1.0, 1.0, "dir"),
    "spread2": ("vwap", "pov", "imp_pool", 1.0, 2.0, "dir"),
    # 封板口径敏感性(审查 P0):any = 与引擎 can 一致(挡买卖双方);tight = 方向性 + 阈值收紧;none = 修复前口径
    "seal_any": ("vwap", "pov", "imp_pool", 1.0, 1.0, "any"),
    "seal_tight": ("vwap", "pov", "imp_pool", 1.0, 1.0, "tight"),
    "seal_none": ("vwap", "pov", "imp_pool", 1.0, 1.0, "none"),
}
MAIN_ONLY_SCHEMES = {"spread2"}      # 只对主候选 + W1 算的变体


def split_legs(L: pd.DataFrame) -> dict:
    return {k_: g.set_index("date") for k_, g in L.groupby(["X", "kind", "capvar", "seal", "leg", "side"])}


def assemble(L: dict, scheme: str, X: float, variant: str) -> pd.DataFrame:
    """方案的日成本分项(买、卖分列;正 = 成本)。L = split_legs(...)。返回 DataFrame(date × [comp_side])。"""
    kind, capvar, impc, Y, smult, seal = VARIANTS[variant]
    legs = scheme_legs(scheme)
    out = {}
    for side in ("buy", "sell"):
        acc = None
        for lg in legs[side]:
            g = L[(X, kind, capvar, seal, lg, side)]
            g = g[["px_rel", "px_mkt", "unf_tot", "unf_mkt", "spread", impc, "w", "w_capped",
                   "w_unfilled", "n_unfillable", "n_sealed", "w_sealed"]].rename(columns={impc: "impact"})
            acc = g if acc is None else acc.add(g, fill_value=0.0)
        acc = acc.copy()
        acc["impact"] = acc["impact"] * Y
        acc["spread"] = acc["spread"] * smult
        for c_ in acc.columns:
            out[f"{c_}_{side}"] = acc[c_]
    return pd.DataFrame(out)


def cost_cols(df: pd.DataFrame, side: str | None = None) -> dict:
    sides = ["buy", "sell"] if side is None else [side]
    g = lambda c_: sum(df[f"{c_}_{s_}"] for s_ in sides)
    A_rel = g("px_rel")
    B = g("spread") + g("impact")
    D = g("unf_tot")
    return dict(px_rel=A_rel, px_mkt=g("px_mkt"), spread=g("spread"), impact=g("impact"), unf_tot=D,
                unf_mkt=g("unf_mkt"), credible=A_rel + B + D, strict=A_rel + B + D - g("unf_mkt"),
                realized_A=A_rel + D, model_B=B, total_incl_mkt=A_rel + B + D + g("px_mkt"))


def ann_pp(x: pd.Series) -> float:
    return float(x.mean() * TRADING_DAYS * 100)


_PANEL = None


def Panel_cache():
    global _PANEL
    if _PANEL is None:
        _PANEL = Panel()
    return _PANEL


def cmd_summarize():
    p = Panel_cache()
    grid_rows, side_rows, year_rows = [], [], []
    for strat in C.STRAT_KEYS:
        f_old, f_new = legs_path(strat, False), legs_path(strat, True)
        if not f_old.exists() or not f_new.exists():
            print("缺", f_old, f_new, flush=True)
            continue
        L0 = pd.read_parquet(f_old)
        L0["seal"], L0["n_sealed"], L0["w_sealed"] = "none", 0.0, 0.0       # 修复前账本:无封板规则
        L = pd.concat([L0, pd.read_parquet(f_new)], ignore_index=True)
        del L0
        L["date"] = pd.to_datetime(L["date"])
        L = split_legs(L)
        mask = pd.Series(C.sim_mask(p, strat), index=p.dates)
        for X in C.AUMS:
            base_cache = {}
            for variant in VARIANTS:
                if variant in MAIN_ONLY_SCHEMES:
                    schemes = C.PRIMARY + ["W1"]
                else:
                    schemes = SCHEMES
                if variant not in base_cache:
                    base_cache[variant] = assemble(L, "W1", X, variant)
                B0 = base_cache[variant]
                c0 = cost_cols(B0)
                for sc in schemes:
                    S = assemble(L, sc, X, variant)
                    cs = cost_cols(S)
                    for pname, (y0, y1, role) in C.PERIODS[strat].items():
                        sel = mask & (p.dates.year >= y0) & (p.dates.year <= y1)
                        days = p.dates[sel.to_numpy()]
                        # 最后两个信号日没有 t+1/t+2 成交(引擎同样截断),只取有记录的日
                        days = days[days <= B0.index.max()] if len(B0) else days
                        row = dict(strategy=strat, X=X, variant=variant, scheme=sc, type=SCHEME_TYPE[sc],
                                   primary=sc in C.PRIMARY, period=pname, role=role, n_days=len(days))
                        for comp in ["credible", "strict", "px_rel", "px_mkt", "spread", "impact", "unf_tot", "unf_mkt",
                                     "realized_A", "model_B", "total_incl_mkt"]:
                            d = (c0[comp] - cs[comp]).reindex(days).fillna(0.0)     # 节省 = 基线 − 方案
                            row[f"d_{comp}_pp"] = ann_pp(d)
                            if comp in ("credible", "strict", "realized_A", "model_B", "total_incl_mkt"):
                                row[f"t_{comp}"] = C.nw_t(d.to_numpy())
                        row["base_credible_cost_pp"] = ann_pp(c0["credible"].reindex(days).fillna(0.0))
                        row["base_total_cost_incl_mkt_pp"] = ann_pp(c0["total_incl_mkt"].reindex(days).fillna(0.0))
                        row["scheme_credible_cost_pp"] = ann_pp(cs["credible"].reindex(days).fillna(0.0))
                        wsum = sum(S[f"w_{s_}"].reindex(days).fillna(0).sum() for s_ in ("buy", "sell"))
                        row["capped_share"] = float(sum(S[f"w_capped_{s_}"].reindex(days).fillna(0).sum()
                                                        for s_ in ("buy", "sell")) / wsum) if wsum > 0 else np.nan
                        row["unfilled_share"] = float(sum(S[f"w_unfilled_{s_}"].reindex(days).fillna(0).sum()
                                                          for s_ in ("buy", "sell")) / wsum) if wsum > 0 else np.nan
                        row["sealed_share"] = float(sum(S[f"w_sealed_{s_}"].reindex(days).fillna(0).sum()
                                                        for s_ in ("buy", "sell")) / wsum) if wsum > 0 else np.nan
                        row["sealed_legs_ann"] = float(sum(S[f"n_sealed_{s_}"].reindex(days).fillna(0).sum()
                                                           for s_ in ("buy", "sell")) / max(len(days), 1) * TRADING_DAYS)
                        row["turnover_ann"] = float(wsum / 2 / max(len(days), 1) * TRADING_DAYS)
                        grid_rows.append(row)
                        if variant == "main":
                            for side in ("buy", "sell"):
                                c0s, css = cost_cols(B0, side), cost_cols(S, side)
                                srow = dict(strategy=strat, X=X, scheme=sc, period=pname, role=role, side=side)
                                for comp in ["credible", "px_rel", "px_mkt", "spread", "impact", "unf_tot"]:
                                    d = (c0s[comp] - css[comp]).reindex(days).fillna(0.0)
                                    srow[f"d_{comp}_pp"] = ann_pp(d)
                                srow["t_credible"] = C.nw_t((c0s["credible"] - css["credible"]).reindex(days).fillna(0.0).to_numpy())
                                srow["base_px_rel_pp"] = ann_pp(c0s["px_rel"].reindex(days).fillna(0.0))
                                srow["base_px_mkt_pp"] = ann_pp(c0s["px_mkt"].reindex(days).fillna(0.0))
                                side_rows.append(srow)
                    if variant == "main":
                        sel = mask & (p.dates.year >= min(v[0] for v in C.PERIODS[strat].values()))
                        days = p.dates[sel.to_numpy()]
                        days = days[days <= B0.index.max()]
                        d = (c0["credible"] - cs["credible"]).reindex(days).fillna(0.0)
                        for y, g in d.groupby(d.index.year):
                            year_rows.append(dict(strategy=strat, X=X, scheme=sc, year=y, d_credible_pp=ann_pp(g),
                                                  t_credible=C.nw_t(g.to_numpy())))
                print(f"[{strat} X={X:.0e}] {variant} 完成", flush=True)
    G = pd.DataFrame(grid_rows)
    G.to_csv(LEDGER / "inc_t1_layer1_grid.csv", index=False)
    pd.DataFrame(side_rows).to_csv(LEDGER / "inc_t1_layer1_sides.csv", index=False)
    pd.DataFrame(year_rows).to_csv(LEDGER / "inc_t1_layer1_years.csv", index=False)
    print("summarize done", flush=True)


# ====================================================================== 4 日内路径(§7.5)
POINT_WINDOW = {0: "W0", 10: "W1", 30: "W2", 60: "W3", 120: "W4", 180: "W5", 210: "W6", 237: "W7", 240: "W8"}


def cmd_paths():
    """买卖清单相对决策价的日内路径(九个时点)减同时点池子路径。
    variant = raw     全部清单腿按 T+1 该时点的价格(修复前口径,含整窗口封死的腿);
    variant = rolled  该时点所在窗口(POINT_WINDOW)整窗口封板(dir 口径)的腿按"不可成交"顺延:用 T+2 同一时点的价格,
                      池子参照也换成 T+2 同一时点的池路径(与第一层对封死腿的处理同一规则,审查 P0)。"""
    p = Panel()
    close = p.raw("close_px").to_numpy(np.float64)
    U = ST.universe(p, C.VERSION).to_numpy(bool)
    T = len(p.dates)
    pts = {i: BW.load_extra(f"pt{i}").reindex(index=p.dates, columns=p.codes).to_numpy(np.float64) for i in BW.POINTS}
    # 池等权路径 PM[k][i][t] = mean_pool(pt_i[t+k]/close[t] - 1),k = 1, 2
    PM = {1: {}, 2: {}}
    for k in (1, 2):
        for i, P in pts.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                R = P[k:] / close[:T - k] - 1.0
            Uk = U[:T - k] & np.isfinite(R)
            out = np.full(T, np.nan)
            out[:T - k] = np.where(Uk.sum(1) > 0, np.where(Uk, R, 0).sum(1) / np.maximum(Uk.sum(1), 1), np.nan)
            PM[k][i] = out
    rows = []
    for strat in C.STRAT_KEYS:
        for X in C.AUMS:
            f = trades_path(strat, X)
            if not f.exists():
                continue
            z = np.load(f)
            t = z["t"].astype(np.int64); ii = z["i"].astype(np.int64); dw = z["dw"]
            w = np.abs(dw)
            side = (dw < 0).astype(np.int64)
            te = np.minimum(t + 1, T - 1)
            for sd, sname in ((0, "buy"), (1, "sell")):
                m = side == sd
                key = t[m]
                wm = w[m]
                for i, P in pts.items():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r1 = P[np.minimum(t[m] + 1, T - 1), ii[m]] / close[t[m], ii[m]] - 1.0
                        r2 = P[np.minimum(t[m] + 2, T - 1), ii[m]] / close[t[m], ii[m]] - 1.0
                    blk = C.blocked_mask(p, POINT_WINDOW[i], "dir", dw[m] > 0, te[m], ii[m])
                    pm1, pm2 = PM[1][i][t[m]], PM[2][i][t[m]]
                    roll = blk & np.isfinite(r2) & np.isfinite(pm2)
                    for variant in ("raw", "rolled"):
                        r = np.where(roll, r2, r1) if variant == "rolled" else r1
                        pl = np.where(roll, pm2, pm1) if variant == "rolled" else pm1       # 逐腿的池子参照
                        ok = np.isfinite(r) & np.isfinite(pl)
                        wsum = np.bincount(key[ok], weights=wm[ok], minlength=T)
                        has = wsum > 0
                        with np.errstate(invalid="ignore", divide="ignore"):
                            raw = np.bincount(key[ok], weights=wm[ok] * r[ok], minlength=T) / wsum   # 当日清单加权的 (pt/决策价 − 1)
                            rel = np.bincount(key[ok], weights=wm[ok] * (r[ok] - pl[ok]), minlength=T) / wsum
                        sign = 1.0 if sname == "buy" else -1.0     # 成本口径:买 = 价格上升为成本;卖 = 下降为成本
                        for pname, (y0, y1, role) in C.PERIODS[strat].items():
                            sel = has & (p.dates.year >= y0) & (p.dates.year <= y1) & np.isfinite(rel)
                            rows.append(dict(strategy=strat, X=X, side=sname, point=BW.POINT_LABELS[i], idx=i,
                                             variant=variant, period=pname, role=role, n_days=int(sel.sum()),
                                             rolled_w_share=float(wm[roll].sum() / wm.sum()) if variant == "rolled" else 0.0,
                                             raw_bp=float(np.nanmean(raw[sel]) * 1e4),
                                             pool_bp=float(np.nanmean(PM[1][i][sel]) * 1e4),
                                             rel_bp=float(np.mean(rel[sel]) * 1e4),
                                             rel_t_nw=C.nw_t(rel[sel]),
                                             rel_cost_bp=float(sign * np.mean(rel[sel]) * 1e4)))
            print(f"paths {strat} {X:.0e} done", flush=True)
    pd.DataFrame(rows).to_csv(LEDGER / "inc_t1_layer1_paths.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rebuild":
        cmd_rebuild(sys.argv[2])
    elif cmd == "layer1":
        cmd_layer1(sys.argv[2], seal_fix=(len(sys.argv) < 4 or sys.argv[3] != "old"))
    elif cmd == "summarize":
        cmd_summarize()
    elif cmd == "paths":
        cmd_paths()
    else:
        raise SystemExit(cmd)
