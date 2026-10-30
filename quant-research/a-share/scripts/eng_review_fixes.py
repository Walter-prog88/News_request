"""引擎审查修复的证据表(前缀 eng,2026-10-10)。research/ENGINE_IMPACT_MODEL.md "审查记录"的数字都出自这里。

产出(research/ledger/):
  eng_review_capw.csv     审查 P1-a:ctx['cap_w'] 改为事前估计
                          (A1) 新 ctx['cap_w'] == 0.1 × 截至 te-1 的 20 日窗口成交额中位数 / capital(独立重算)
                          (A2) 把 te 及以后的成交价/成交额乘随机扰动,新 ctx 的 cap_w / h_dec / nav_dec 逐位不变
                          (A3) 审查者的"截量后重分配"规则:cap 取 te 实际窗口成交额(= 修复前 ctx['cap_w'] 的定义)
                               vs 新 ctx['cap_w'] vs 手算事前中位数,年化 / 仓位 / 按仓位调整的超额
                          (A4) 事前估计与 te 实际窗口成交额之比的分布(V1000 池)
  eng_review_hdec.csv     审查 P2-a:ctx['h_dec'] 的精确性与"换手带"规则的前视量
                          (B1) ctx['h_dec'] 与按 px[te-1]/px[te] 回滚 h 的精确公式之差
                          (B2) 换手带规则:按 h(te 价)判定 vs 按 h_dec 判定(动作空间相同)vs h_dec + NaN 保持
  eng_vwap_autocorr.csv   审查 P1-b:vwap1 平均价口径的自相关与二阶矩偏差
                          池等权基准 / 个股 日收益一阶自相关、方差比、日频 vs 长期方差口径的波动、
                          COMP_v5 标准排雷的 Sharpe / 跟踪误差 / IR(日频 vs Newey-West 带宽 20)

⚠ COMP_v5 成员在 2020-2025 内选出,用它的组合数字都是**信号样本内**,这里只用于检验引擎/接口;
数据不含已退市股(约 209 只),收益有幸存者偏差(方向为高估)。研究期面板,未触碰封存数据。
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, TMP
from afactor.panel import Panel
from afactor import strategy as ST
from afactor.strategy import TRADING_DAYS

WINDOWS = {"2021-2025": (2021, 2025), "2021-2023": (2021, 2023), "2024-2025": (2024, 2025)}
MP = 0.10


def nw_t(d, bw: int = 5) -> float:
    d = np.asarray(d, np.float64); d = d[np.isfinite(d)]
    if len(d) < 30 or d.std() == 0:
        return np.nan
    vr = ST.lr_variance_ratio(d, bw)
    return float(d.mean() / d.std(ddof=1) * np.sqrt(len(d)) / np.sqrt(vr))


def wmask(p, y0, y1):
    yrs = p.dates.year
    return np.asarray((yrs >= y0) & (yrs <= y1))


class _Stop(Exception):
    pass


class _PerturbPanel(Panel):
    """把 rows >= k 的指定原料乘 U(0.5, 2) 随机扰动(保持有限且为正,不改变可成交性判定)。"""

    def __init__(self, base: Panel, k: int, fields, seed: int = 0):
        super().__init__()
        self._k, self._f, self._rng = k, set(fields), np.random.default_rng(seed)
        for f in ("one_word_up", "one_word_down"):             # 一字板判定不扰动,直接借用
            self._derived[f] = base[f]

    def raw(self, name):
        if name not in self._f:
            return super().raw(name)
        key = "__pert__" + name
        if key not in self._cache:
            df = super().raw(name)
            a = df.to_numpy(np.float32, copy=True)
            a[self._k:] *= self._rng.uniform(0.5, 2.0, size=a[self._k:].shape).astype(np.float32)
            self._cache[key] = pd.DataFrame(a, index=df.index, columns=df.columns)
        return self._cache[key]


def capture_ctx(p, W, mode, t_star, capital=1e8, nav_mode=True):
    """跑到信号日 t_star 截获 ctx 与 h(之前每天按 W 调仓),随即中止。"""
    got = {}

    def fn(t, h, ctx):
        if t == t_star:
            got.update(ctx=ctx, h=h)
            raise _Stop
        return W[t]

    try:
        ST.simulate(fn, p, mode, capital, rebalance=1, date_mask=wmask(p, 2023, 2025),
                    participation_on_nav=nav_mode)
    except _Stop:
        pass
    return got


# ------------------------------------------------------------------ A: cap_w
def part_capw(p, score) -> pd.DataFrame:
    rows = []
    W50 = ST.build_weights(score, p, "V1000", "top", 50)
    # (A1)(A2) 恒等式与扰动不变性
    te_star = 1000
    for mode in ("open10", "vwap1"):
        lag = ST.EXEC_MODES[mode]["lag"]
        px_f, amt_f = ("vwap", "amount") if mode == "vwap1" else ("twap_open10", "amt_open10")
        amt = p.raw(amt_f).astype(np.float64)
        med = amt.rolling(20, min_periods=10).median().to_numpy()       # 独立重算,行 d = 截至 d
        a = capture_ctx(p, W50, mode, te_star - lag, nav_mode=False)
        c = a["ctx"]; te = c["te"]
        exp_new = np.where(np.isfinite(med[te - 1]), MP * med[te - 1] / 1e8, 0.0)
        exp_old = np.where(np.isfinite(amt.to_numpy()[te]), MP * amt.to_numpy()[te] / 1e8, 0.0)
        rows.append(dict(part="A1", mode=mode, check="ctx_cap_w_equals_median_te_minus_1",
                         value=float(np.array_equal(c["cap_w"], exp_new)), date=str(c["trade_date"].date()),
                         note="新 ctx['cap_w'] == 0.1×截至 te-1 的 20 日窗口成交额中位数/capital(独立 pandas 重算)"))
        rows.append(dict(part="A1", mode=mode, check="ctx_cap_w_equals_realized_te",
                         value=float(np.array_equal(c["cap_w"], exp_old)), date=str(c["trade_date"].date()),
                         note="修复前的定义 0.1×te 实际窗口成交额/capital;修复后应为 0"))
        tp = _PerturbPanel(p, te_star, (px_f, amt_f), seed=7)
        b = capture_ctx(tp, W50, mode, te_star - lag, nav_mode=True)
        a = capture_ctx(p, W50, mode, te_star - lag, nav_mode=True)
        ca, cb = a["ctx"], b["ctx"]
        for f in ("cap_w", "h_dec", "nav_dec", "budget_dec", "can"):
            rows.append(dict(part="A2", mode=mode, check=f"perturb_te_onward_unchanged_{f}",
                             value=float(np.array_equal(np.asarray(ca[f]), np.asarray(cb[f]))),
                             date=str(ca["trade_date"].date()),
                             note=("te 及以后成交价/成交额乘 U(0.5,2) 后不变(1=逐位相等);participation_on_nav=True"
                                   + (";can 不变只说明它不依赖 te 价格数值,仍含 te/te+1 可成交性的旧约定" if f == "can" else ""))))
        held = a["h"] > 1e-12
        rows.append(dict(part="A2", mode=mode, check="perturb_te_onward_changed_share_h",
                         value=float(np.mean(a["h"][held] != b["h"][held])), date=str(ca["trade_date"].date()),
                         note="对照:按 px[te] 计价的 h 在持仓股上发生变化的比例(应接近 1)"))
        rows.append(dict(part="A2", mode=mode, check="perturb_te_onward_changed_nav",
                         value=float(ca["nav"] != cb["nav"]), date=str(ca["trade_date"].date()),
                         note="对照:按 px[te] 计价的 nav 是否变化(应为 1)"))
        del tp
        print(f"A1/A2 {mode} done", flush=True)

    # (A3) 截量后重分配规则(审查者 p6b 同法)
    m = wmask(p, 2021, 2025)
    for mode in ("open10", "vwap1"):
        amt_df = ST.exec_prices(p, mode)[1].astype(np.float64)
        amt = amt_df.to_numpy()
        med = amt_df.rolling(20, min_periods=10).median().to_numpy()
        b = ST.benchmark(p, "V1000", mode, date_mask=m)
        for cap_ in (1e8, 1e9):
            def make_fn(src):
                def fn(t, h, ctx):
                    te = ctx["te"]; w = W50[t].astype(np.float64); can = ctx["can"]
                    if src == "realized_te":                       # 修复前 ctx['cap_w'] 的定义
                        cap = np.where(np.isfinite(amt[te]), MP * amt[te] / cap_, 0.0)
                    elif src == "ctx_new":
                        cap = ctx["cap_w"]
                    else:                                          # 手算事前中位数
                        a_ = med[te - 1]
                        cap = np.where(np.isfinite(a_), MP * a_ / cap_, 0.0)
                    w = np.where(can, w, 0.0)
                    s = w.sum(); w = w / s * ctx["budget"] if s > 0 else w
                    d = w - h
                    dc = np.clip(d, -cap, cap)
                    for _ in range(5):
                        short = float(np.clip(d - dc, 0, None).sum())
                        room = np.where((w > 0) & can, np.clip(cap - dc, 0, None), 0.0)
                        rs = room.sum()
                        if short < 1e-9 or rs < 1e-12:
                            break
                        dc = dc + room * min(1.0, short / rs)
                        d = dc + np.clip(d - dc, None, 0)
                    tgt = np.clip(h + dc, 0, None)
                    tot = tgt.sum()
                    return tgt / tot * min(tot, 1.0) if tot > 0 else tgt
                return fn
            out = {}
            for src in ("realized_te", "ctx_new", "manual_median"):
                t0 = time.time()
                r = ST.run_weights(make_fn(src), p, mode, cap_, rebalance=5, date_mask=m, normalize=False, record=True)
                ex = r["record"]["daily"]["exposure"]
                br = b["rets"].reindex(r["rets"].index)
                out[src] = (r, ex, br)
                rows.append(dict(part="A3", mode=mode, check=f"policy_{src}_cap{cap_:.0e}",
                                 value=r["ann_return"], date="2021-2025",
                                 exposure=r["exposure"], alpha_adj_pp=float((r["rets"] - ex * br).mean() * TRADING_DAYS * 100),
                                 bench_ann=b["ann_return"], turnover=r["turnover"],
                                 note="COMP_v5 Top50 r5 五相位 normalize=False;截量后重分配;value=年化(信号样本内)"))
                print(f"A3 {mode} {cap_:.0e} {src}: ann {r['ann_return']*100:.3f}% exp {r['exposure']:.4f} {time.time()-t0:.0f}s", flush=True)
            base = out["manual_median"]
            for src in ("realized_te", "ctx_new"):
                ra, ea, br = out[src]; rb, eb, _ = base
                d = (ra["rets"] - ea * br) - (rb["rets"] - eb * br)
                rows.append(dict(part="A3", mode=mode, check=f"diff_alpha_adj_{src}_minus_manual_cap{cap_:.0e}",
                                 value=float(d.mean() * TRADING_DAYS * 100), date="2021-2025",
                                 nw_t=nw_t(d.to_numpy(), 5),
                                 bitwise_equal=float(np.array_equal(ra["rets"].to_numpy(), rb["rets"].to_numpy())),
                                 note="按仓位调整的超额之差 pp/年,NW t 带宽 5;ctx_new 应与手算逐位相等"))

    # (A4) 事前估计 vs te 实际
    U = ST.universe(p, "V1000").to_numpy(bool)
    for mode in ("open10", "vwap1"):
        est = ST.decision_window_amount(p, mode)
        amt = ST.exec_prices(p, mode)[1].to_numpy(np.float64)
        lag = ST.EXEC_MODES[mode]["lag"]
        ts = np.flatnonzero(m)
        ts = ts[ts + lag < len(p.dates)]
        R = amt[ts + lag] / est[ts + lag]
        sel = U[ts] & np.isfinite(R) & (est[ts + lag] > 0)
        r = R[sel]
        qs = np.quantile(r, [0.05, 0.25, 0.5, 0.75, 0.95])
        for qn, qv in zip(("q05", "q25", "q50", "q75", "q95"), qs):
            rows.append(dict(part="A4", mode=mode, check=f"realized_over_est_{qn}", value=float(qv), date="2021-2025",
                             note="te 实际窗口成交额 / 截至 te-1 的 20 日中位数,V1000 池股票日"))
        rows.append(dict(part="A4", mode=mode, check="share_realized_below_est", value=float(np.mean(r < 1)),
                         date="2021-2025", note="实际 < 估计(决策时高估了可成交量)的比例"))
        rows.append(dict(part="A4", mode=mode, check="share_realized_below_half_est", value=float(np.mean(r < 0.5)),
                         date="2021-2025", note="实际 < 0.5×估计 的比例"))
        rows.append(dict(part="A4", mode=mode, check="n_stock_days", value=float(len(r)), date="2021-2025", note=""))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B: h_dec
def part_hdec(p, score) -> pd.DataFrame:
    rows = []
    W50 = ST.build_weights(score, p, "V1000", "top", 50)
    # (B1) h_dec 精确性
    for mode in ("vwap1", "open10", "tail1450"):
        px = ST.exec_prices(p, mode)[0].to_numpy(np.float64)
        worst = [0.0, 0.0]

        def fn(t, h, ctx):
            te = ctx["te"]
            ok = np.isfinite(px[te - 1]) & np.isfinite(px[te]) & (px[te - 1] > 0)
            g = np.where(ok, px[te] / px[te - 1], 1.0)
            x = h / g
            man = x / (x.sum() + 1.0 - h.sum())
            worst[0] = max(worst[0], float(np.max(np.abs(man - ctx["h_dec"]))))
            worst[1] = max(worst[1], float(np.max(np.abs(h - ctx["h_dec"]))))
            return W50[t]

        ST.simulate(fn, p, mode, 1e7, rebalance=1, date_mask=wmask(p, 2024, 2024))
        rows.append(dict(part="B1", mode=mode, check="max_abs_hdec_minus_exact_rollback", value=worst[0],
                         note="2024 Top50 r1:ctx['h_dec'] 与 h 按 px[te-1]/px[te] 回滚并含现金腿归一的精确公式之差"))
        rows.append(dict(part="B1", mode=mode, check="max_abs_h_minus_hdec", value=worst[1],
                         note="对照:h(te 价)与 h_dec 的最大差,说明二者确实不同"))
        print(f"B1 {mode} {worst}", flush=True)

    # (B2) 换手带规则
    m = wmask(p, 2021, 2025)
    for mode in ("vwap1", "open10"):
        b = ST.benchmark(p, "V1000", mode, date_mask=m)
        for reb, bands in ((1, (0.0003, 0.0006)), (5, (0.001, 0.002))):
            for band in bands:
                def make_fn(arm):
                    def fn(t, h, ctx):
                        w = W50[t].astype(np.float64)
                        hh = h if arm == "decide_on_h_te" else ctx["h_dec"]
                        keep = (hh > 1e-12) & (w > 0) & (np.abs(hh - w) < band)
                        if arm == "decide_on_h_dec_nan_hold":
                            return np.where(keep, np.nan, w)
                        return np.where(keep, h, w)            # 保持 = 回传实际持仓(动作空间相同)
                    return fn
                out = {}
                for arm in ("decide_on_h_te", "decide_on_h_dec", "decide_on_h_dec_nan_hold"):
                    t0 = time.time()
                    r = ST.run_weights(make_fn(arm), p, mode, 1e7, rebalance=reb, date_mask=m, normalize=False)
                    out[arm] = r
                    rows.append(dict(part="B2", mode=mode, check=f"band_rule_{arm}", rebalance=reb, band=band,
                                     value=r["ann_return"], excess_pp=(r["ann_return"] - b["ann_return"]) * 100,
                                     turnover=r["turnover"], exposure=r["exposure"],
                                     note="COMP_v5 Top50 1000万 2021-2025 normalize=False;value=年化(信号样本内)"))
                    print(f"B2 {mode} r{reb} band={band} {arm}: ann {r['ann_return']*100:.3f}% {time.time()-t0:.0f}s", flush=True)
                for a_, b_ in (("decide_on_h_te", "decide_on_h_dec"), ("decide_on_h_dec_nan_hold", "decide_on_h_dec")):
                    d = out[a_]["rets"] - out[b_]["rets"]
                    rows.append(dict(part="B2", mode=mode, check=f"diff_{a_}_minus_{b_}", rebalance=reb, band=band,
                                     value=float(d.mean() * TRADING_DAYS * 100), nw_t=nw_t(d.to_numpy(), 5),
                                     note="日收益差均值×242(pp/年),NW t 带宽 5;h_te 组减 h_dec 组 = h 中 te 价信息的作用"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C: vwap1 自相关
def vr_overlap(r: np.ndarray, h: int) -> float:
    r = r[np.isfinite(r)]
    s = np.convolve(r, np.ones(h), mode="valid")
    return float(s.var(ddof=1) / (h * r.var(ddof=1)))


def vol_nonoverlap(r: np.ndarray, h: int = 20) -> float:
    r = r[np.isfinite(r)]
    n = len(r) // h
    s = r[:n * h].reshape(n, h).sum(axis=1)
    return float(s.std(ddof=1) * np.sqrt(TRADING_DAYS / h))


def part_autocorr(p, score) -> pd.DataFrame:
    rows = []
    for wname, (y0, y1) in WINDOWS.items():
        m = wmask(p, y0, y1)
        for mode in ("open10", "tail1450", "vwap1"):
            b = ST.benchmark(p, "V1000", mode, date_mask=m)
            r = b["rets"].to_numpy(np.float64)
            base = dict(part="C1", window=wname, mode=mode, series="pool_ew_benchmark")
            rows += [dict(base, metric="ac1", value=b["ret_ac1"]),
                     dict(base, metric="vr5_overlap", value=vr_overlap(r, 5)),
                     dict(base, metric="vr20_overlap", value=vr_overlap(r, 20)),
                     dict(base, metric="vol_daily", value=b["vol"]),
                     dict(base, metric="vol_adj_nw20", value=b["vol_adj"]),
                     dict(base, metric="vol_nonoverlap20", value=vol_nonoverlap(r, 20)),
                     dict(base, metric="sharpe_daily", value=b["sharpe"]),
                     dict(base, metric="sharpe_adj_nw20", value=b["sharpe_adj"])]
            for L in (5, 10, 20, 40):
                rows.append(dict(base, metric=f"vr_nw{L}", value=ST.lr_variance_ratio(r, L)))
            # COMP_v5 标准排雷(旧成本)
            s = ST.run_strategy(score, p, "V1000", mode, scheme="exclude", buffer=(0.20, 0.35), rebalance=5, date_mask=m)
            es = ST.excess_stats(s["rets"], b["rets"])
            base = dict(part="C2", window=wname, mode=mode, series="comp_v5_exclude_std_legacy_cost")
            rows += [dict(base, metric="ann_return", value=s["ann_return"]),
                     dict(base, metric="bench_ann", value=b["ann_return"]),
                     dict(base, metric="ac1", value=s["ret_ac1"]),
                     dict(base, metric="vol_daily", value=s["vol"]),
                     dict(base, metric="vol_adj_nw20", value=s["vol_adj"]),
                     dict(base, metric="vol_nonoverlap20", value=vol_nonoverlap(s["rets"].to_numpy(), 20)),
                     dict(base, metric="sharpe_daily", value=s["sharpe"]),
                     dict(base, metric="sharpe_adj_nw20", value=s["sharpe_adj"])]
            rows += [dict(base, metric=f"excess_{k}", value=v) for k, v in es.items()]
            print(f"C {wname} {mode}: bench ac1 {b['ret_ac1']:.3f} vol {b['vol']*100:.2f}->{b['vol_adj']*100:.2f}; "
                  f"comp IR {es['ir']:.2f}->{es['ir_adj']:.2f}", flush=True)
    # 个股日收益一阶自相关(V1000 池,2021-2025,逐股 >= 200 个有效对)
    m = wmask(p, 2021, 2025)
    U = ST.universe(p, "V1000")
    for fld in ("vwap", "twap_open10", "twap_tail8", "close_px"):
        x = p.raw(fld).astype(np.float64)
        rr = (x / x.shift(1) - 1).where(U)[m]
        ac = []
        for c in rr.columns:
            s = rr[c]
            if s.notna().sum() < 200:
                continue
            a, b_ = s.iloc[1:].to_numpy(), s.iloc[:-1].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b_)
            if ok.sum() < 200:
                continue
            ac.append(np.corrcoef(a[ok], b_[ok])[0, 1])
        ac = np.asarray(ac)
        rows.append(dict(part="C3", window="2021-2025", mode=fld, series="stock_daily_returns",
                         metric="ac1_median", value=float(np.median(ac))))
        rows.append(dict(part="C3", window="2021-2025", mode=fld, series="stock_daily_returns",
                         metric="n_stocks", value=float(len(ac))))
        print(f"C3 {fld}: median ac1 {np.median(ac):.3f} n={len(ac)}", flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    p = Panel()
    score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    parts = sys.argv[1:] or ["C", "B", "A"]
    if "C" in parts:
        c = part_autocorr(p, score); c.to_csv(LEDGER / "eng_vwap_autocorr.csv", index=False)
        print(f"C {time.time()-t0:.0f}s", flush=True)
    if "B" in parts:
        bb = part_hdec(p, score); bb.to_csv(LEDGER / "eng_review_hdec.csv", index=False)
        print(bb.to_string(), flush=True); print(f"B {time.time()-t0:.0f}s", flush=True)
    if "A" in parts:
        a = part_capw(p, score); a.to_csv(LEDGER / "eng_review_capw.csv", index=False)
        print(a.to_string(), flush=True); print(f"A {time.time()-t0:.0f}s", flush=True)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
