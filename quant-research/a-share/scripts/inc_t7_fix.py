"""INC-T7 审查修复计算(前缀 inc_t7_fix)。针对 2026-10-11 审查的 P0 / P1:

  verify   自写逐日模拟器 MinSim 对官方引擎已落盘序列的逐位复现(固定 k 相位平均、E2b 五个起点、1e8 恒定 AUM)。任一不过即停。
  price    P0:成交价口径。同一信号、同一池、同一成本,k=1 减 k=5 在 {开盘 10 分钟 TWAP, 开盘集合竞价单点价(窗口额取 amt_open10 / 取真实集合竞价额),
           收盘价} 下的配对差;信号 = COMP_v5 与长历史 S1=-turn20(映射到分钟面板);另跑全池等权对照与长历史引擎 2021-2025 对照。
  lock     P1:一字板可成交规则。引擎口径(涨跌停一字板双向不可成交,标记用 te 日收盘涨跌幅 + 全日振幅)
           vs 开盘窗口可观察的封板标记 vs 方向性规则(封涨停可卖不可买、封跌停可买不可卖;dir=被挡的买单留现金,dir2=被挡名字冻结、其余目标重新归一)。
           对 k=1 / 5 / 10 与 E2b_10 / E2b_20(五个起点平均)报对同规则 k=5 的配对差。
  lots     P1:整手与最低佣金。1e7 / 1e8 / 1e9 下 k=1 / 5(单相位)/ E2b_10 的逐笔订单不足一手的占比、5 元最低佣金的额外成本;
           再把逐笔订单取整到 100 股(清仓允许零股)后重跑 k=1 / 5 / E2b_10 / E2b_20,报对同口径 k=5 的配对差。
  e2bx     E2b 补充:收盘价成交 / 长历史信号 S1 映射到分钟面板后 E2b 与 k=1 是否仍有增量(信号特异性)
  xcheck   MinSim 的换价选项对官方引擎(经 PricePanel 换价)逐日核对
  finalize 汇总 inc_t7_fix_summary.csv、登记偏离 D18-D24、写审查记录 inc_t7_review.csv
  all      依次 verify -> price -> lock -> lots -> e2bx -> xcheck -> finalize(各自独立落盘,可单独重跑)

本脚本全部是**审查之后的事后诊断**(登记为 D18-D22,不改预注册、不改 inc_t7_grid 的结果);只读 <= 2025-12-31 的缓存,不读 sealed / forward。
用法:PYTHONPATH=src python3 scripts/inc_t7_fix.py <stage> [--workers 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor import strategy as ST
from afactor import impact as IM
from afactor import lh_engine as LE
from afactor import newdata as ND
from afactor.config import (LEDGER, TMP, RESEARCH_END, COST_BPS_ONEWAY, STAMP_TAX_BPS_BEFORE,
                            STAMP_TAX_BPS_AFTER, STAMP_TAX_CHANGE_DATE, MAX_PARTICIPATION)
from afactor.panel import Panel, _limit_frame, _LIMIT_TOL, _ONE_WORD_RNG

import inc_t7_report as R

TD = 242
MIN_START = "2020-07-01"
BAND = (0.20, 0.35)
WIN = {"full": ("2021-01-01", "2025-12-31"), "dev": ("2021-01-01", "2023-12-31"), "test": ("2024-01-01", "2025-12-31")}
OFFSETS = [0, 1, 2, 3, 4]
# 价格口径 -> (成交价键, 窗口成交额键)
PRICE_SPEC = {"twap": ("twap", "twap"), "auction": ("auction", "twap"), "auction_real": ("auction", "auction_real"),
              "close": ("close", "close")}
LOGDIR = ROOT / "logs"
_LOGF: Path | None = None


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    if _LOGF is not None:
        with open(_LOGF, "a") as f:
            f.write(s + "\n")


# ====================================================================== 模拟器(与 strategy.simulate 同一逻辑,可选三处修改)
class MinSim:
    """分钟面板(V1000,open10 类成交,恒定 AUM)逐日模拟器。

    与 ST.simulate(callable / ndarray 两条路径)逐位等价(stage verify 核对),并在其上加三个可选项:
      price   成交价口径(PRICE_SPEC)
      lock    一字板可成交规则:sym(引擎,双向不可成交)/ dir(方向性,被挡买单留现金)/ dir2(方向性,被挡名字冻结、其余重新归一);
              后缀 _ol = 用开盘窗口可观察的封板标记(open10 内振幅 < 0.4% 且触板)代替引擎的 te 日收盘涨跌幅 + 全日振幅标记
      lots    把逐笔订单取整到 100 股(按前一日未复权收盘价;清仓允许零股);min_comm = 每笔最低佣金(元,0 = 不设)
    """

    def __init__(self):
        p = Panel()
        self.p = p
        d, codes = p.dates, p.codes
        self.d, self.codes = d, codes
        self.T, self.N = len(d), len(codes)
        self.mask0 = np.asarray((d >= pd.Timestamp(MIN_START)) & (d <= pd.Timestamp(RESEARCH_END)))
        self.t_start = int(np.flatnonzero(self.mask0)[0])

        def arr(name):
            return p.raw(name).reindex(index=d, columns=codes).to_numpy(np.float64)
        self.PX = {"twap": arr("twap_open10"), "auction": arr("open_auction_px"), "close": arr("close_px")}
        self.WIN = {"twap": arr("amt_open10"), "auction_real": arr("amt_open_auction"), "close": arr("amt_close_auction")}
        self.owu = p["one_word_up"].reindex(index=d, columns=codes).fillna(False).to_numpy(bool)
        self.owd = p["one_word_down"].reindex(index=d, columns=codes).fillna(False).to_numpy(bool)
        lim = _limit_frame(p).reindex(index=d, columns=codes).to_numpy(np.float64)
        hi, lo = arr("open10_high"), arr("open10_low")
        cl = self.PX["close"]
        prevc = np.vstack([np.full((1, self.N), np.nan), cl[:-1]])
        with np.errstate(all="ignore"):
            rr = self.PX["twap"] / prevc - 1.0
            flat = (hi / lo - 1.0) < _ONE_WORD_RNG
            self.owu_ol = flat & (rr >= lim * _LIMIT_TOL)
            self.owd_ol = flat & (rr <= -lim * _LIMIT_TOL)
        self.RC = ND.raw_close(d, codes).reindex(index=d, columns=codes).to_numpy(np.float64)
        self.U = ST.universe(p, "V1000").reindex(index=d, columns=codes).fillna(False).to_numpy(bool)
        self.CM = {"Y1": IM.CostModel(p, impact_y=1.0, spread_mult=1.0),
                   "pess": IM.CostModel(p, impact_y=1.5, spread_mult=2.0), "old": None}
        self.stamp_switch = int(np.searchsorted(d.values, np.datetime64(STAMP_TAX_CHANGE_DATE)))
        self.i2021 = int(np.searchsorted(d.values, np.datetime64("2021-01-01")))

    # ---- 目标权重 ----
    def comp_score(self) -> pd.DataFrame:
        return pd.read_parquet(TMP / "comp_v5_score_open10.parquet")

    def s1_score(self) -> tuple[pd.DataFrame, dict]:
        """长历史 S1=-turn20 映射到分钟面板(同一日历,代码取交集)。"""
        X = LE.load_inputs()
        li = X.dates.get_indexer(self.d)
        assert (li >= 0).all(), "分钟面板日期不全在长历史日历内"
        s1 = pd.DataFrame(np.asarray(X.arr("score_S1"))[li], index=self.d, columns=X.codes).reindex(columns=self.codes)
        ok = np.isfinite(s1.to_numpy(np.float64))
        cov = float((ok & self.U & self.mask0[:, None]).sum() / max((self.U & self.mask0[:, None]).sum(), 1))
        return s1, dict(pool_coverage=cov)

    def weights(self, score: pd.DataFrame, band=BAND, mask=None) -> np.ndarray:
        m = self.mask0 if mask is None else mask
        return ST.build_weights(score, self.p, "V1000", scheme="exclude", buffer=band, date_mask=m)

    def fp_weights(self) -> np.ndarray:
        Ua = self.U & self.mask0[:, None]
        n = Ua.sum(1, keepdims=True)
        return np.divide(Ua.astype(np.float32), n, out=np.zeros(Ua.shape, np.float32), where=n > 0)

    def offset_mask(self, o: int) -> np.ndarray:
        return np.arange(self.T) >= (self.t_start + int(o))

    # ---- 单路径 ----
    def run_path(self, W, mode="fixed", k=1, phase=0, keq=10, off=0, cost="Y1", price="twap", lock="sym",
                 cap=1e7, lots=False, min_comm=0.0, stat=None) -> dict:
        T, N = self.T, self.N
        pk, wk = PRICE_SPEC[price]
        px, win = self.PX[pk], self.WIN[wk]
        ol = lock.endswith("_ol")
        owu, owd = (self.owu_ol, self.owd_ol) if ol else (self.owu, self.owd)
        kind = lock[:-3] if ol else lock               # sym / dir / dir2
        directional = kind in ("dir", "dir2")
        dead = owu | owd
        cm = self.CM[cost]
        h = np.zeros(N)
        hdec = np.zeros(N)
        rets = np.full(T - 2, np.nan)
        costs = np.zeros(T - 2)
        grosses = np.zeros(T - 2)
        turns = np.zeros(T - 2)
        exps = np.zeros(T - 2)
        ntr = np.zeros(T - 2)
        t0 = self.t_start + (int(phase) if mode == "fixed" else int(off))
        tol = 1e-10
        for t in range(T - 2):
            te = t + 1
            pn, p1 = px[te], px[te + 1]
            okp = np.isfinite(pn) & np.isfinite(p1) & (pn > 0)
            can0 = okp & (win[te] > 0)
            can = can0 & ~dead[te] if not directional else can0
            delta = np.zeros(N)
            tgt = None
            trade = False
            if mode == "fixed":
                if k == 1 or ((t - t0) % k == 0):
                    trade = True
                    w = W[t].astype(np.float64)
                    tgt = self._tgt_fixed(w, can, h)
            else:   # e2b:t0 之前不交易(W 已按起点掩码)
                if t >= t0:
                    w = W[t].astype(np.float64)
                    full = ((t - t0) % keq == 0)
                    tgt = self._tgt_e2b(w, can, h, hdec, full)
                    trade = tgt is not None
            if trade:
                desired = self._finish(tgt - h, can, te, win, cap)
                if directional:
                    blk = (owu[te] & (desired > 1e-12)) | (owd[te] & (desired < -1e-12))
                    if kind == "dir2" and blk.any():
                        can2 = can & ~blk                      # 被挡名字冻结,其余目标在剩余预算里重新归一
                        tgt = (self._tgt_fixed(w, can2, h) if mode == "fixed"
                               else self._tgt_e2b(w, can2, h, hdec, full))
                        desired = np.zeros(N) if tgt is None else self._finish(tgt - h, can2, te, win, cap)
                        blk = (owu[te] & (desired > 1e-12)) | (owd[te] & (desired < -1e-12))
                    delta = np.where(blk, 0.0, desired)
                else:
                    delta = desired
                if lots:
                    delta = self._round_lots(delta, tgt, h, te, cap)
                over = float(np.clip(h + delta, 0, None).sum()) - 1.0
                if over > 1e-12:
                    b = np.clip(delta, 0, None)
                    bs = b.sum()
                    if bs > 0:
                        delta = delta - b * min(1.0, over / bs)
            newh = np.clip(h + delta, 0, None)
            buy = float(np.clip(delta, 0, None).sum())
            sell = float(np.clip(-delta, 0, None).sum())
            stamp = (STAMP_TAX_BPS_BEFORE if te < self.stamp_switch else STAMP_TAX_BPS_AFTER) * 1e-4
            c = (buy + sell) * COST_BPS_ONEWAY * 1e-4 + sell * stamp
            if min_comm > 0 and (buy + sell) > 0:
                ad = np.abs(delta)
                nz = ad > 1e-12
                c += float(np.clip(min_comm / cap - ad[nz] * COST_BPS_ONEWAY * 1e-4, 0, None).sum())
            if cm is not None and (buy + sell) > 0:
                cs, ci = cm.extra_cost(te, np.abs(delta), cap, 1.0)
                c += cs + ci
            if stat is not None and te >= self.i2021 and (buy + sell) > 0:
                self._stat(stat, delta, te, cap)
            with np.errstate(invalid="ignore"):
                r = np.where(okp, p1 / pn - 1.0, 0.0)
            gross = float(np.nansum(newh * r))
            pr = gross - c
            grown = newh * (1.0 + np.nan_to_num(r))
            h = grown / (1.0 + pr) if (1.0 + pr) > 1e-9 else grown
            hs = h.sum()
            if hs > 1.0 + 1e-12:
                h = h / hs
            assert newh.sum() <= 1.0 + 1e-9, f"exposure>1: {newh.sum():.6f} at t={t}"
            rets[t], costs[t], grosses[t] = pr, c, gross
            turns[t] = np.abs(delta).sum() / 2
            exps[t] = float(newh.sum())
            ntr[t] = int((np.abs(delta) > 1e-12).sum())
            den = 1.0 - c
            hdec = newh / den if den > 1e-9 else newh.copy()
            hsd = hdec.sum()
            if hsd > 1.0 + 1e-12:
                hdec = hdec / hsd
        idx = self.d[:T - 2]
        return dict(ret=pd.Series(rets, index=idx), cost=pd.Series(costs, index=idx),
                    gross=pd.Series(grosses, index=idx), turnover=pd.Series(turns, index=idx),
                    exposure=pd.Series(exps, index=idx), n_orders=pd.Series(ntr, index=idx))

    # ---- 目标构造(与 strategy.simulate 的 ndarray / E2bTarget 路径逐字对应) ----
    @staticmethod
    def _tgt_fixed(w, can, h):
        stuck = (~can) & (h > 1e-12)
        budget = max(0.0, 1.0 - float(h[stuck].sum()))
        tgt = np.where(can, w, 0.0)
        s = tgt.sum()
        if s > 0:
            tgt = tgt / s * budget
        return tgt + np.where(stuck, h, 0.0)

    @staticmethod
    def _tgt_e2b(w, can, h, hdec, full):
        stuck = (~can) & (h > 1e-12)
        budget = max(0.0, 1.0 - float(h[stuck].sum()))
        if full:
            bdec = max(0.0, 1.0 - float(hdec[stuck].sum()))
            tg = np.where(can, w, 0.0)
            s = tg.sum()
            wt = tg / s * bdec if s > 0 else tg
            tgt = np.where(can, wt, 0.0)
            s = tgt.sum()
            if s > budget:                                   # normalize=False:只防超预算
                tgt = tgt / s * budget
            return tgt + np.where(stuck, h, 0.0)
        held = hdec > 1e-10
        want = w > 0
        sell = held & ~want
        buy = ~held & want
        if not (sell.any() or buy.any()):
            return None
        wt = np.full(w.shape, np.nan)
        wt[sell] = 0.0
        wt[buy] = 1.0 / max(int(want.sum()), 1)
        hold = np.isnan(wt) & can
        free = max(0.0, budget - float(h[hold].sum()))
        tgt = np.where(can & ~hold, np.nan_to_num(wt, nan=0.0), 0.0)
        s = tgt.sum()
        if s > free:
            tgt = tgt / s * free
        return tgt + np.where(stuck | hold, h, 0.0)

    @staticmethod
    def _finish(delta, can, te, win, cap):
        """参与率截量 + 不可成交置零(与 strategy.simulate 同序)。"""
        cap_w = np.where(np.isfinite(win[te]), MAX_PARTICIPATION * win[te] / cap, 0.0)
        delta = np.clip(delta, -cap_w, cap_w)
        return np.where(can, delta, 0.0)

    def _round_lots(self, delta, tgt, h, te, cap):
        """订单取整到 100 股(按 te-1 日未复权收盘价);目标为 0 的整仓卖出允许零股。"""
        lotw = 100.0 * self.RC[te - 1] / cap
        fin = np.isfinite(lotw) & (lotw > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dr = np.where(fin, np.rint(delta / lotw) * lotw, delta)
        full_exit = (tgt <= 1e-12) & (delta < 0) & (np.abs(delta + h) <= 1e-12)
        return np.where(full_exit, delta, dr)

    def _stat(self, st, delta, te, cap):
        ad = np.abs(delta)
        nz = ad > 1e-12
        val = ad[nz] * cap
        lot = 100.0 * self.RC[te - 1][nz]
        ok = np.isfinite(lot)
        val, lot = val[ok], lot[ok]
        sub = val < lot
        st["n"] = st.get("n", 0) + int(ok.sum())
        st["n_sub"] = st.get("n_sub", 0) + int(sub.sum())
        st["v"] = st.get("v", 0.0) + float(val.sum())
        st["v_sub"] = st.get("v_sub", 0.0) + float(val[sub].sum())
        st["floor5"] = st.get("floor5", 0.0) + float(np.clip(5.0 - val * COST_BPS_ONEWAY * 1e-4, 0, None).sum())
        st["days"] = st.get("days", 0) + 1


# ====================================================================== 统计工具
def avg(series_list) -> pd.Series:
    return pd.concat(series_list, axis=1).mean(axis=1)


def wslice(s: pd.Series, win: str) -> pd.Series:
    a, b = WIN[win]
    return s.loc[pd.Timestamp(a):pd.Timestamp(b)]


def paired_row(a: pd.Series, b: pd.Series, win: str) -> dict:
    """a - b 的配对差:几何年化差、算术年化差、NW t(q=20 / 5)、±1 NW se(年化,算术)。"""
    x, y = wslice(a, win).dropna(), wslice(b, win).dropna()
    idx = x.index.intersection(y.index)
    x, y = x.loc[idx], y.loc[idx]
    dd = (x - y).to_numpy()
    t20, se20 = R.nw(dd, 20)
    t5, _ = R.nw(dd, 5)
    return dict(window=win, n=len(idx), a_ann=R.ann_geo(x), b_ann=R.ann_geo(y), d_geo=R.ann_geo(x) - R.ann_geo(y),
                d_arith=float(dd.mean() * TD), t_nw20=t20, t_nw5=t5, se_nw20_ann=se20 * TD)


def pair_rows(a: pd.Series, b: pd.Series, **meta) -> list[dict]:
    return [dict(meta, **paired_row(a, b, w)) for w in ("full", "dev", "test")]


def ann_cost(c: pd.Series, win="full") -> float:
    return float(wslice(c, win).mean() * TD)


# ====================================================================== 作业
def _worker_init():
    pass


def stage_verify(S: MinSim):
    comp = S.comp_score()
    D = pd.read_parquet(TMP / "inc_t7_grid_daily_minute.parquet")
    E = pd.read_parquet(TMP / "inc_t7_event_daily_minute.parquet")
    SN = pd.read_parquet(TMP / "inc_t7_sens_daily_minute.parquet")
    W = S.weights(comp)
    rows = []

    def cached(df, cell):
        return df[df.cell == cell].set_index("date")["ret"].sort_index()

    def cmp(name, mine: pd.Series, ref: pd.Series):
        idx = ref.index
        diff = float((mine.reindex(idx) - ref).abs().max())
        rows.append(dict(item=name, max_abs_diff_daily_ret=diff, n_days=len(idx), ok=bool(diff < 1e-12)))
        log(f"  verify {name}: max|diff| = {diff:.3e}")
        return diff

    for k in (1, 5):
        for cost in ("Y1", "old"):
            r = avg([S.run_path(W, "fixed", k=k, phase=ph, cost=cost)["ret"] for ph in range(k)])
            cmp(f"fixed k{k} 20/35 {cost}", r, cached(D, f"k{k}|20/35|{cost}"))
    r = avg([S.run_path(W, "fixed", k=10, phase=ph, cost="Y1")["ret"] for ph in range(10)])
    cmp("fixed k10 20/35 Y1", r, cached(D, "k10|20/35|Y1"))
    r = S.run_path(W, "fixed", k=1, cost="Y1", cap=1e8)["ret"]
    cmp("fixed k1 20/35 Y1 cap1e8", r, cached(SN, "k1|20/35|Y1|aum1e+08c"))
    for keq in (10, 20):
        per = []
        for off in OFFSETS:
            Wo = S.weights(comp, mask=S.offset_mask(off))
            ro = S.run_path(Wo, "e2b", keq=keq, off=off, cost="Y1")["ret"]
            cmp(f"e2b_{keq} offset{off} Y1", ro, cached(E, f"E2b_{keq}|o{off}|Y1"))
            per.append(ro)
        if keq == 10:
            pass
    ok = all(r["ok"] for r in rows)
    pd.DataFrame(rows).to_csv(LEDGER / "inc_t7_fix_verify.csv", index=False)
    log("verify", "ALL OK" if ok else "FAILED")
    if not ok:
        raise SystemExit("MinSim 与引擎不一致,停止")


def stage_price(S: MinSim):
    comp = S.comp_score()
    s1, aux = S.s1_score()
    SIG = {"COMP_v5": S.weights(comp), "S1_turn20": S.weights(s1), "pool_EW": S.fp_weights()}
    rows, series = [], {}
    for sn, W in SIG.items():
        for price in ("twap", "auction", "auction_real", "close"):
            for cost in ("Y1", "old", "pess"):
                t1 = time.time()
                r1 = S.run_path(W, "fixed", k=1, cost=cost, price=price)
                rs = [S.run_path(W, "fixed", k=5, phase=ph, cost=cost, price=price) for ph in range(5)]
                r5 = avg([x["ret"] for x in rs])
                c1, c5 = r1["cost"], avg([x["cost"] for x in rs])
                series[f"{sn}|{price}|{cost}|k1"] = r1["ret"]
                series[f"{sn}|{price}|{cost}|k5"] = r5
                for rw in pair_rows(r1["ret"], r5, signal=sn, price=price, cost=cost, test="k1_minus_k5", panel="minute"):
                    rw["k1_cost_ann"] = ann_cost(c1, rw["window"])
                    rw["k5_cost_ann"] = ann_cost(c5, rw["window"])
                    rows.append(rw)
                log(f"  price {sn} {price} {cost} done {time.time() - t1:.1f}s")
        pd.DataFrame(rows).to_csv(LEDGER / "inc_t7_fix_price.csv", index=False)
    pd.DataFrame(series).to_parquet(TMP / "inc_t7_fix_price_series.parquet")

    # ---- LH 引擎同窗口对照(已落盘日序列,不重跑)----
    DL = pd.read_parquet(TMP / "inc_t7_grid_daily_lh.parquet")

    def lh(cell):
        return DL[DL.cell == cell].set_index("date")["ret"].sort_index()
    a, b = lh("k1|20/35|Y1"), lh("k5|20/35|Y1")
    lh_rows = []
    for lab, (s0, e0) in {"2021-2025": ("2021-01-01", "2025-12-31"), "2021-2023": ("2021-01-01", "2023-12-31"),
                          "2024-2025": ("2024-01-01", "2025-12-31"), "2006-2019": ("2006-01-01", "2019-12-31"),
                          "2020-2025": ("2020-01-01", "2025-12-31"), "2006-2025": ("2006-01-01", "2025-12-31")}.items():
        x, y = a.loc[s0:e0], b.loc[s0:e0]
        t20, se = R.nw((x - y).to_numpy(), 20)
        lh_rows.append(dict(panel="lh_engine", signal="S1_turn20", price="日开盘价(集合竞价单点)", cost="Y1",
                            test="k1_minus_k5", window=lab, n=len(x), d_geo=R.ann_geo(x) - R.ann_geo(y),
                            d_arith=float((x - y).mean() * TD), t_nw20=t20, se_nw20_ann=se * TD))
    pd.DataFrame(rows + lh_rows).to_csv(LEDGER / "inc_t7_fix_price.csv", index=False)

    # ---- 辅助:集合竞价额占比、S1 覆盖 ----
    d = S.d
    amt = S.p.raw("amount").reindex(index=d, columns=S.codes).to_numpy(np.float64)
    m = S.U & S.mask0[:, None] & (d >= pd.Timestamp("2021-01-01"))[:, None] & np.isfinite(amt) & (amt > 0)
    with np.errstate(all="ignore"):
        sh_auc = (S.WIN["auction_real"] / amt)[m]
        sh_o10 = (S.WIN["twap"] / amt)[m]
    prow = dict(item="V1000 池-日(2021-2025):集合竞价额 / 全日额 中位数", value=float(np.nanmedian(sh_auc)))
    aux_rows = [prow,
                dict(item="V1000 池-日(2021-2025):开盘 10 分钟额 / 全日额 中位数", value=float(np.nanmedian(sh_o10))),
                dict(item="V1000 池-日(2021-2025):集合竞价额 / 全日额 均值", value=float(np.nanmean(sh_auc))),
                dict(item="长历史引擎假设的开盘窗口额 / 全日额", value=0.114),
                dict(item="S1 在 V1000 池-日的覆盖率", value=aux["pool_coverage"])]
    # 横截面秩相关 COMP_v5 vs S1(池内,逐日均值)
    from scipy.stats import rankdata
    cs = comp.reindex(index=d, columns=S.codes).to_numpy(np.float64)
    ss = s1.to_numpy(np.float64)
    rho = []
    for t in np.flatnonzero((d >= pd.Timestamp("2021-01-01")) & (d <= pd.Timestamp(RESEARCH_END)))[::5]:
        mm = S.U[t] & np.isfinite(cs[t]) & np.isfinite(ss[t])
        if mm.sum() > 100:
            rho.append(np.corrcoef(rankdata(cs[t, mm]), rankdata(ss[t, mm]))[0, 1])
    aux_rows.append(dict(item="池内逐日秩相关 COMP_v5 vs S1(每 5 日抽样均值)", value=float(np.mean(rho))))
    pd.DataFrame(aux_rows).to_csv(LEDGER / "inc_t7_fix_price_aux.csv", index=False)
    log("price done")


def stage_lock(S: MinSim):
    comp = S.comp_score()
    W = S.weights(comp)
    Wo = {off: S.weights(comp, mask=S.offset_mask(off)) for off in OFFSETS}
    Wfp = S.fp_weights()
    RULES = ["sym", "sym_ol", "dir", "dir_ol", "dir2", "dir2_ol"]
    rows, lev = [], []
    series = {}
    for rule in RULES:
        for cost in ("Y1", "old"):
            t1 = time.time()
            res = {}
            res["k5"] = [S.run_path(W, "fixed", k=5, phase=ph, cost=cost, lock=rule) for ph in range(5)]
            res["k1"] = [S.run_path(W, "fixed", k=1, cost=cost, lock=rule)]
            res["k10"] = [S.run_path(W, "fixed", k=10, phase=ph, cost=cost, lock=rule) for ph in range(10)]
            res["E2b_10"] = [S.run_path(Wo[o], "e2b", keq=10, off=o, cost=cost, lock=rule) for o in OFFSETS]
            res["E2b_20"] = [S.run_path(Wo[o], "e2b", keq=20, off=o, cost=cost, lock=rule) for o in OFFSETS]
            if cost == "Y1":
                res["pool_EW_k5"] = [S.run_path(Wfp, "fixed", k=5, phase=ph, cost=cost, lock=rule) for ph in range(5)]
                res["pool_EW_k1"] = [S.run_path(Wfp, "fixed", k=1, cost=cost, lock=rule)]
            agg = {c: avg([x["ret"] for x in v]) for c, v in res.items()}
            aggc = {c: avg([x["cost"] for x in v]) for c, v in res.items()}
            aggx = {c: avg([x["exposure"] for x in v]) for c, v in res.items()}
            for c, s in agg.items():
                series[f"{rule}|{cost}|{c}"] = s
            base = agg["k5"]
            for c in ("k1", "k10", "E2b_10", "E2b_20"):
                for rw in pair_rows(agg[c], base, rule=rule, cost=cost, cell=c, test="cell_minus_k5_same_rule",
                                    panel="minute"):
                    rw["cell_cost_ann"] = ann_cost(aggc[c], rw["window"])
                    rw["k5_cost_ann"] = ann_cost(aggc["k5"], rw["window"])
                    rw["cell_exposure"] = float(wslice(aggx[c], rw["window"]).mean())
                    rows.append(rw)
            if cost == "Y1":
                for rw in pair_rows(agg["pool_EW_k1"], agg["pool_EW_k5"], rule=rule, cost=cost, cell="pool_EW_k1",
                                    test="cell_minus_k5_same_rule", panel="minute"):
                    rows.append(rw)
            for c, s in agg.items():
                for w in ("full", "dev", "test"):
                    lev.append(dict(rule=rule, cost=cost, cell=c, window=w, ann_net=R.ann_geo(wslice(s, w)),
                                    cost_ann=ann_cost(aggc[c], w), exposure=float(wslice(aggx[c], w).mean())))
            log(f"  lock {rule} {cost} done {time.time() - t1:.1f}s")
            pd.DataFrame(rows).to_csv(LEDGER / "inc_t7_fix_lock.csv", index=False)
            pd.DataFrame(lev).to_csv(LEDGER / "inc_t7_fix_lock_levels.csv", index=False)
    # 交叉:k=1 / E2b 在某规则下 vs k=5 在引擎规则下(把"规则本身的价值"与"频率"分开)
    cross = []
    for rule in RULES:
        for c in ("k1", "E2b_10"):
            a = series[f"{rule}|Y1|{c}"]
            b = series["sym|Y1|k5"]
            cross += pair_rows(a, b, rule=rule, cost="Y1", cell=c, test="cell_under_rule_minus_k5_under_sym",
                               panel="minute")
        a = series[f"{rule}|Y1|k5"]
        cross += pair_rows(a, series["sym|Y1|k5"], rule=rule, cost="Y1", cell="k5", test="k5_under_rule_minus_k5_under_sym",
                           panel="minute")
    pd.DataFrame(cross).to_csv(LEDGER / "inc_t7_fix_lock_cross.csv", index=False)
    pd.DataFrame(series).to_parquet(TMP / "inc_t7_fix_lock_series.parquet")
    # 封板标记计数
    d = S.d
    m = (d >= pd.Timestamp("2021-01-01")) & S.mask0
    cnt = pd.DataFrame([dict(flag="引擎 one_word_up(te 日收盘涨跌幅 + 全日振幅)", n=int(S.owu[m].sum())),
                        dict(flag="引擎 one_word_down", n=int(S.owd[m].sum())),
                        dict(flag="开盘窗口封涨停(open10 振幅<0.4% 且触板)", n=int(S.owu_ol[m].sum())),
                        dict(flag="开盘窗口封跌停", n=int(S.owd_ol[m].sum())),
                        dict(flag="引擎双向标记在 V1000 池内的股票日", n=int(((S.owu | S.owd) & S.U)[m].sum())),
                        dict(flag="开盘窗口双向标记在 V1000 池内的股票日", n=int(((S.owu_ol | S.owd_ol) & S.U)[m].sum()))])
    cnt.to_csv(LEDGER / "inc_t7_fix_lock_flags.csv", index=False)
    log("lock done")


def stage_lots(S: MinSim):
    comp = S.comp_score()
    W = S.weights(comp)
    Wo = {off: S.weights(comp, mask=S.offset_mask(off)) for off in OFFSETS}
    # ---- 逐笔订单大小(未取整,引擎口径)----
    srows = []
    for cap in (1e7, 1e8, 1e9):
        for cell, fn in (("k1", lambda st, c=cap: S.run_path(W, "fixed", k=1, cap=c, stat=st)),
                         ("k5_phase0", lambda st, c=cap: S.run_path(W, "fixed", k=5, phase=0, cap=c, stat=st)),
                         ("E2b_10_off0", lambda st, c=cap: S.run_path(Wo[0], "e2b", keq=10, off=0, cap=c, stat=st)),
                         ("E2b_20_off0", lambda st, c=cap: S.run_path(Wo[0], "e2b", keq=20, off=0, cap=c, stat=st))):
            st = {}
            fn(st)
            yrs = 5.0
            srows.append(dict(capital=cap, cell=cell, orders_per_year=st["n"] / yrs, sublot_share_orders=st["n_sub"] / st["n"],
                              sublot_share_value=st["v_sub"] / st["v"], mean_order_yuan=st["v"] / st["n"],
                              trade_days_per_year=st["days"] / yrs,
                              floor5_extra_cost_pct_per_year=st["floor5"] / yrs / cap * 100))
            log(f"  lots stat cap={cap:.0e} {cell}: orders/yr {srows[-1]['orders_per_year']:,.0f} "
                f"sublot {srows[-1]['sublot_share_orders']:.1%} floor5 {srows[-1]['floor5_extra_cost_pct_per_year']:.2f}%/yr")
        pd.DataFrame(srows).to_csv(LEDGER / "inc_t7_fix_lots_stats.csv", index=False)
    # ---- 取整后重跑 ----
    rows, lev = [], []
    for cap in (1e7, 1e8, 1e9):
        for rule in ("sym", "dir2"):
            for lots, mc in ((False, 0.0), (True, 0.0), (True, 5.0)):
                t1 = time.time()
                kw = dict(cost="Y1", lock=rule, cap=cap, lots=lots, min_comm=mc)
                res = {"k5": [S.run_path(W, "fixed", k=5, phase=ph, **kw) for ph in range(5)],
                       "k1": [S.run_path(W, "fixed", k=1, **kw)],
                       "E2b_10": [S.run_path(Wo[o], "e2b", keq=10, off=o, **kw) for o in OFFSETS],
                       "E2b_20": [S.run_path(Wo[o], "e2b", keq=20, off=o, **kw) for o in OFFSETS]}
                agg = {c: avg([x["ret"] for x in v]) for c, v in res.items()}
                aggc = {c: avg([x["cost"] for x in v]) for c, v in res.items()}
                aggx = {c: avg([x["exposure"] for x in v]) for c, v in res.items()}
                for c in ("k1", "E2b_10", "E2b_20"):
                    for rw in pair_rows(agg[c], agg["k5"], capital=cap, rule=rule, lots=lots, min_comm=mc, cell=c,
                                        test="cell_minus_k5_same_config", panel="minute"):
                        rw["cell_cost_ann"] = ann_cost(aggc[c], rw["window"])
                        rw["k5_cost_ann"] = ann_cost(aggc["k5"], rw["window"])
                        rw["cell_exposure"] = float(wslice(aggx[c], rw["window"]).mean())
                        rows.append(rw)
                for c, s in agg.items():
                    for w in ("full", "dev", "test"):
                        lev.append(dict(capital=cap, rule=rule, lots=lots, min_comm=mc, cell=c, window=w,
                                        ann_net=R.ann_geo(wslice(s, w)), cost_ann=ann_cost(aggc[c], w),
                                        exposure=float(wslice(aggx[c], w).mean())))
                log(f"  lots sim cap={cap:.0e} {rule} lots={lots} mc={mc} done {time.time() - t1:.1f}s")
                pd.DataFrame(rows).to_csv(LEDGER / "inc_t7_fix_lots_sim.csv", index=False)
                pd.DataFrame(lev).to_csv(LEDGER / "inc_t7_fix_lots_levels.csv", index=False)
    log("lots done")


def stage_e2bx(S: MinSim):
    """E2b 的稳健性补充:(i) 换成收盘价成交;(ii) 把长历史 S1 映射到分钟面板后 E2b / k=1 是否也有增量(信号特异性)。
    都对同信号、同价格、同规则的 k=5 配对;Y1 成本,1000 万恒定 AUM。"""
    comp = S.comp_score()
    s1, _ = S.s1_score()
    rows = []
    for sn, sc in (("COMP_v5", comp), ("S1_turn20", s1)):
        W = S.weights(sc)
        Wo = {off: S.weights(sc, mask=S.offset_mask(off)) for off in OFFSETS}
        for price in ("twap", "close", "auction"):
            for rule in ("sym", "dir2"):
                t1 = time.time()
                kw = dict(cost="Y1", price=price, lock=rule)
                k5 = avg([S.run_path(W, "fixed", k=5, phase=ph, **kw)["ret"] for ph in range(5)])
                res = {"k1": S.run_path(W, "fixed", k=1, **kw)["ret"],
                       "E2b_10": avg([S.run_path(Wo[o], "e2b", keq=10, off=o, **kw)["ret"] for o in OFFSETS]),
                       "E2b_20": avg([S.run_path(Wo[o], "e2b", keq=20, off=o, **kw)["ret"] for o in OFFSETS])}
                for c, r in res.items():
                    rows += pair_rows(r, k5, signal=sn, price=price, rule=rule, cost="Y1", cell=c,
                                      test="cell_minus_k5_same_config", panel="minute")
                log(f"  e2bx {sn} {price} {rule} done {time.time() - t1:.1f}s")
                pd.DataFrame(rows).to_csv(LEDGER / "inc_t7_fix_e2bx.csv", index=False)
    log("e2bx done")


class PricePanel:
    """只替换成交价 / 窗口额字段的面板代理,其余原样转发;用来让官方引擎 ST.run_weights 在别的价格上跑,核对 MinSim 的 price 选项。"""

    def __init__(self, p, repl: dict):
        self._p, self._repl = p, repl

    def raw(self, name):
        return self._p.raw(self._repl.get(name, name))

    def __getitem__(self, k):
        return self._p[k]

    def __getattr__(self, k):
        return getattr(self._p, k)


def stage_xcheck(S: MinSim):
    """MinSim 的 price 选项(集合竞价 / 收盘)对官方引擎 ST.run_weights(经 PricePanel 换价)的逐日核对。"""
    import cap_capacity_curve as CAP
    ser = pd.read_parquet(TMP / "inc_t7_fix_price_series.parquet")
    s1, _ = S.s1_score()
    comp = S.comp_score()
    rows = []
    for sn, sc, price, repl in (("S1_turn20", s1, "twap", {}),
                                ("S1_turn20", s1, "auction", {"twap_open10": "open_auction_px"}),
                                ("COMP_v5", comp, "close", {"twap_open10": "close_px", "amt_open10": "amt_close_auction"})):
        W = S.weights(sc)
        pp = PricePanel(S.p, repl)
        cm = CAP.ConstAUMCost(IM.CostModel(S.p, impact_y=1.0, spread_mult=1.0))
        for k in (1, 5):
            res = ST.run_weights(W, pp, "open10", 1e7, rebalance=k, date_mask=S.mask0, cost_model=cm, record=False)
            mine = ser[f"{sn}|{price}|Y1|k{k}"]
            dd = float((res["rets"].reindex(mine.index) - mine).abs().max())
            rows.append(dict(item=f"engine(换价) vs MinSim {sn} {price} k{k} Y1", max_abs_diff_daily_ret=dd, n_days=len(mine), ok=bool(dd < 1e-12)))
            log(f"  xcheck {sn} {price} k{k}: max|diff| = {dd:.3e}")
    v = pd.read_csv(LEDGER / "inc_t7_fix_verify.csv")
    v = v[~v["item"].str.startswith("engine(换价)")]
    pd.concat([v, pd.DataFrame(rows)], ignore_index=True).to_csv(LEDGER / "inc_t7_fix_verify.csv", index=False)
    if not all(r["ok"] for r in rows):
        raise SystemExit("MinSim 价格选项与引擎不一致")
    log("xcheck ALL OK")


# ====================================================================== 收尾:汇总账本 / 偏离登记 / 审查记录
def _q(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def stage_finalize(S=None):
    """把 fix 各阶段的账本汇总成 inc_t7_fix_summary.csv(报告引用的数字都在这里),登记偏离 D18-D24,写审查记录 inc_t7_review.csv。
    幂等:重复运行只覆盖同 id 的行。不做任何模拟。"""
    pr = pd.read_csv(LEDGER / "inc_t7_fix_price.csv")
    lk = pd.read_csv(LEDGER / "inc_t7_fix_lock.csv")
    lc = pd.read_csv(LEDGER / "inc_t7_fix_lock_cross.csv")
    ls = pd.read_csv(LEDGER / "inc_t7_fix_lots_stats.csv")
    lm = pd.read_csv(LEDGER / "inc_t7_fix_lots_sim.csv")
    ex = pd.read_csv(LEDGER / "inc_t7_fix_e2bx.csv")
    aux = pd.read_csv(LEDGER / "inc_t7_fix_price_aux.csv")
    rows = []

    def add(section, key, src, **v):
        rows.append(dict(section=section, key=key, src=src, **v))

    # P0
    for sig in ("S1_turn20", "COMP_v5", "pool_EW"):
        for price in ("twap", "auction", "auction_real", "close"):
            for cost in ("Y1", "old", "pess"):
                for w in ("full", "dev", "test"):
                    r = _q(pr, panel="minute", signal=sig, price=price, cost=cost, window=w)
                    if len(r):
                        r = r.iloc[0]
                        add("P0_price", f"{sig}|{price}|{cost}|k1-k5", "inc_t7_fix_price.csv", window=w,
                            d_geo_pp=r.d_geo * 100, d_arith_pp=r.d_arith * 100, t_nw20=r.t_nw20)
    for w in ("2021-2025", "2021-2023", "2024-2025", "2006-2019", "2020-2025", "2006-2025"):
        r = _q(pr, panel="lh_engine", window=w).iloc[0]
        add("P0_price", "LH引擎(日开盘价)|k1-k5", "inc_t7_fix_price.csv", window=w, d_geo_pp=r.d_geo * 100,
            d_arith_pp=r.d_arith * 100, t_nw20=r.t_nw20)
    for _, r in aux.iterrows():
        add("P0_price_aux", r["item"], "inc_t7_fix_price_aux.csv", value=r["value"])
    # P1 lock
    for rule in ("sym", "sym_ol", "dir", "dir_ol", "dir2", "dir2_ol"):
        for cell in ("k1", "k10", "E2b_10", "E2b_20", "pool_EW_k1"):
            for cost in ("Y1", "old"):
                for w in ("full", "dev", "test"):
                    r = _q(lk, rule=rule, cost=cost, cell=cell, window=w)
                    if len(r):
                        r = r.iloc[0]
                        add("P1_lock", f"{rule}|{cell}|{cost}|minus_k5_same_rule", "inc_t7_fix_lock.csv", window=w,
                            d_geo_pp=r.d_geo * 100, d_arith_pp=r.d_arith * 100, t_nw20=r.t_nw20)
    for _, r in lc[lc.window == "full"].iterrows():
        add("P1_lock_cross", f"{r.rule}|{r.cell}|{r.test}", "inc_t7_fix_lock_cross.csv", window="full",
            d_geo_pp=r.d_geo * 100, d_arith_pp=r.d_arith * 100, t_nw20=r.t_nw20)
    # P1 lots
    for _, r in ls.iterrows():
        add("P1_lots_stats", f"cap{r.capital:.0e}|{r.cell}", "inc_t7_fix_lots_stats.csv",
            orders_per_year=r.orders_per_year, sublot_share_orders=r.sublot_share_orders,
            sublot_share_value=r.sublot_share_value, floor5_cost_pct_per_year=r.floor5_extra_cost_pct_per_year)
    for _, r in lm.iterrows():
        add("P1_lots_sim", f"cap{r.capital:.0e}|{r.rule}|lots={r.lots}|mc={r.min_comm}|{r.cell}-k5", "inc_t7_fix_lots_sim.csv",
            window=r.window, d_geo_pp=r.d_geo * 100, d_arith_pp=r.d_arith * 100, t_nw20=r.t_nw20)
    for _, r in ex.iterrows():
        add("E2b_extra", f"{r.signal}|{r.price}|{r.rule}|{r.cell}-k5", "inc_t7_fix_e2bx.csv", window=r.window,
            d_geo_pp=r.d_geo * 100, d_arith_pp=r.d_arith * 100, t_nw20=r.t_nw20)
    pd.DataFrame(rows).to_csv(LEDGER / "inc_t7_fix_summary.csv", index=False)

    # ---- 偏离登记 ----
    now = time.strftime("%Y-%m-%d %H:%M JST")
    dev_new = [
        dict(id="D18", stage="post_review(判定修订)", written_at=now,
             item="C5 两面板一致的价格口径有效性 / 判定降级",
             prereg="§7.1 C5:同一格子在两面板同时满足 C1;§7.2 有增量 = 存在格子满足 C1-C5",
             actual=("机械执行(D03 口径)结果为“有增量”(k1|20/30、k1|20/35、k1|20/45)。审查 P0 后加一条有效性检验:长历史信号 S1=-turn20 映射到分钟面板"
                     "(同池 V1000、同成本 Y1、1000 万恒定 AUM),开盘 10 分钟 TWAP 下 k1-k5 全期 +0.02pp、开发 -0.08、检验 +0.19(不过 C1 / C3),"
                     "换成开盘集合竞价单点价为 +0.95pp(inc_t7_fix_price.csv);长历史引擎同窗口 2021-2025 为 +0.74pp。结论:长历史侧 k=1 的正号是集合竞价价格的再平衡溢价,"
                     "C5 不成立,k=1 各格仅分钟面板通过,判定改为“证据不足”;inc_t7_verdict.json 保留 verdict_mechanical_prereg"),
             reason=("预注册 §9.6 与 H5 解释候选 (b) 已把“长历史用日开盘价”列为已知局限,但没有预设针对它的有效性检验;"
                     "审查给出的检验在同一样本、同一池、同一成本下只换成交价,足以说明 C5 想要的“两面板检验同一件事”并不成立"),
             affects_verdict="是(有增量 -> 证据不足)"),
        dict(id="D19", stage="post_review(判定修订)", written_at=now,
             item="§7.3 触发顺序(对 D04 的更正)",
             prereg="§7.3:若长历史 2006-2019 上有格子明显优于现行而分钟面板上没有,按 §7.2 记“证据不足”",
             actual=("D04 曾把该规则限定为“只在‘有增量’不成立时生效”,这是对判定有利方向的解释。按 §7.3 字面,rule73 格子 E1_6、k1|30/30、k5|30/30、k5|30/40 已触发,"
                     "同样记“证据不足”。修订后的判定不再依赖 D04"),
             reason="审查 P2", affects_verdict="是(与 D18 同向)"),
        dict(id="D20", stage="post_review(事后诊断)", written_at=now,
             item="审查后新增的事后诊断集合(scripts/inc_t7_fix.py)",
             prereg="§6.3 只允许机制诊断;§5.3 只列 5 项敏感性",
             actual=("新增:MinSim 自写逐日模拟器(对官方引擎 16 条已落盘序列逐位核对,最大差 <1e-17,inc_t7_fix_verify.csv)、成交价口径(TWAP / 集合竞价 / 收盘)、"
                     "一字板可成交规则(引擎 / 开盘窗口可观察 / 方向性)、整手与 5 元最低佣金、E2b 的收盘价与 S1 信号补充。全部只读已落盘缓存与面板(<=2025-12-31),不调参;"
                     "只用于判定修订(D18)与披露敏感性,不改任何预注册格子的结果"),
             reason="回应 2026-10-11 审查 P0/P1", affects_verdict="是(D18 的依据)"),
        dict(id="D21", stage="post_review(披露)", written_at=now,
             item="一字板“双向不可成交”口径对 k=1 的影响",
             prereg="§3.3 不可成交:一字板、停牌、窗口无量(沿用引擎,未披露其对 k=1 的影响)",
             actual=("引擎的 one_word 标记用 te 日收盘涨跌幅与全日振幅,双向不可成交。方向性规则(封涨停可卖不可买、封跌停可买不可卖)下分钟 k1-k5 全期 +0.30pp"
                     "(t 1.42,开发 -0.04,检验 +0.90),引擎规则下 +0.86pp;改用开盘窗口可观察的封板标记(双向不交易)为 +0.85pp,所以不是前视而是“封板股双向不交易”这条规则的价值。"
                     "E2b_10 方向性规则下仍 +0.92pp(t 4.43),开盘可观察规则下 +1.39pp"),
             reason="审查 P1(inc_t7_fix_lock.csv / inc_t7_fix_lock_cross.csv)",
             affects_verdict="是(k=1 幅度区间改为 +0.30~+0.86pp,E2b 为首选假设)"),
        dict(id="D22", stage="post_review(披露)", written_at=now,
             item="整手与最低佣金(引擎成本按比例计费)",
             prereg="§3.3 / §9 未写整手与最低佣金",
             actual=("1000 万持约 3000 只,单名约 3300 元;k=1 有 97.1% 订单不足一手,5 元最低佣金额外成本 k1 37.4%/年、k5 7.9%/年、E2b_10 4.8%/年(1000 万)。"
                     "订单取整到 100 股后(无最低佣金)k=1 与 E2b 的 Δ 基本不变(1e8:k1 +0.80 / E2b_10 +1.35pp,引擎规则),加 5 元最低佣金后 k=1 在 1e7 / 1e8 转负"
                     "(-0.44 / -0.24pp),E2b_10 仍正(+1.24 / +1.50pp);方向性规则下 E2b_10 在 1e8 为 +0.86(无最低佣金)/ +1.02pp(有)。"
                     "1000 万的“叠加后”数字不可按模型执行,k=1 不作为 1000 万的实施建议"),
             reason="审查 P1(inc_t7_fix_lots_stats.csv / inc_t7_fix_lots_sim.csv)", affects_verdict="否(不改判定,改实施建议)"),
        dict(id="D23", stage="post_review(披露)", written_at=now,
             item="偏离登记时间戳的可核实性",
             prereg="§10:偏离登记写明时间",
             actual=("D01-D11 的 written_at 为自述,本文件 mtime 是最后一次追加的时间,无法用元数据核实 04:08。可核实的时间线:PREREG mtime 04:01:55、冻结清单 04:01:58、"
                     "首个计算日志 repro 04:12:10(logs/inc_t7_repro.log);freeze verify 在本次修复结束时仍通过(15 个文件哈希一致)"),
             reason="审查 P2", affects_verdict="否"),
        dict(id="D24", stage="post_review(交付形式)", written_at=now,
             item="修订后报告仍未落盘为 research/INC_T7_REPORT.md",
             prereg="§10:报告 research/INC_T7_REPORT.md", 
             actual="执行环境规则仍禁止写报告 .md;修订后报告全文(含“审查记录”)随返回值交付,审查记录的结构化版本写入 research/ledger/inc_t7_review.csv,数字全部在 inc_t7_fix_summary.csv",
             reason="执行环境规则,非方法偏离(延续 D17)", affects_verdict="否"),
    ]
    f = LEDGER / "inc_t7_deviations.csv"
    old = pd.read_csv(f)
    old = old[~old["id"].isin([d["id"] for d in dev_new])]
    pd.concat([old, pd.DataFrame(dev_new)], ignore_index=True).to_csv(f, index=False)

    # ---- 审查记录 ----
    rv = [
        dict(id="R1", severity="P0", finding="“有增量”依赖的 C5 两面板一致,其长历史一侧是开盘集合竞价价格伪影",
             disposition="采纳(实测复现,判定改为证据不足)",
             evidence=("S1 映射到分钟面板、同池同成本:TWAP 下 k1-k5 +0.02pp(开发 -0.08 / 检验 +0.19,t 0.22),集合竞价价 +0.95pp(t 6.82),收盘价 -0.03pp;"
                       "COMP_v5 TWAP +0.86 / 收盘价 +0.86 / 集合竞价价 +1.84;全池等权零信号对照 TWAP -0.46 / 收盘 -0.59 / 集合竞价 +0.64pp;"
                       "长历史引擎同窗口 2021-2025 +0.74pp;集合竞价额中位仅占全日 0.47%(开盘 10 分钟 11.6%,引擎假设 11.4%),但把窗口额换成真实集合竞价额结果不变(+0.91pp),说明问题在价格而非流动性"),
             ledger="inc_t7_fix_price.csv; inc_t7_fix_price_aux.csv", change="inc_t7_verdict.json、inc_t7_criteria_revised.csv、inc_t7_hypotheses.csv(H5 修订行)、报告判定行与一句话结论"),
        dict(id="R2", severity="P1", finding="分钟 k=1 的通过依赖“封板股双向不交易”;方向性规则下 +0.30pp 不过线,E2b 稳健",
             disposition="采纳(实测复现)",
             evidence=("引擎规则 +0.86(t 3.45)/ 开盘可观察标记 +0.85(t 3.40)/ 方向性 +0.30(t 1.42,开发 -0.04,检验 +0.90)/ 方向性+再归一 +0.30;"
                       "E2b_10 +1.40 / +1.39 / +0.92(t 4.43,开发 +0.67,检验 +1.35);E2b_20 +1.44 / +1.42 / +0.96。以引擎规则下的 k=5 为统一基准:k1 方向性 +0.08pp(t 0.52),E2b_10 方向性 +0.70pp(t 3.18)。"
                       "补充:E2b 在收盘价成交下仍 +1.15(引擎规则)/ +0.74pp(方向性),对 TWAP 不特异;长历史信号 S1 的 E2b 在分钟面板 TWAP 下只有 +0.35pp(t 1.54)/ +0.13pp(方向性),增益是 COMP_v5 特异的"),
             ledger="inc_t7_fix_lock.csv; inc_t7_fix_lock_cross.csv; inc_t7_fix_lock_flags.csv; inc_t7_fix_e2bx.csv",
             change="D21;报告 §7 把 k=1 区间改为 +0.30~+0.86,E2b 为前向检验首选,并要求前向规则写死封板股处理"),
        dict(id="R3", severity="P1", finding="1000 万下 k=1 的绝大多数订单不足一手,“叠加后 +6.18pp”不可按模型执行",
             disposition="采纳(实测复现并扩展到取整重跑)",
             evidence=("订单不足一手占比 1e7:k1 97.1% / k5 87.3% / E2b_10 76.8%;1e8:85.3% / 57.6% / 41.3%;1e9:26.5% / 10.5% / 6.5%。5 元最低佣金额外成本(占 AUM/年)1e7:37.4% / 7.9% / 4.8%;1e8:3.6% / 0.7% / 0.4%。"
                       "订单取整到 100 股(无最低佣金)后 Δ 几乎不变(1e8 引擎规则:k1 +0.80、E2b_10 +1.35pp),取整相当于免费的不交易带;加 5 元最低佣金后 k=1 转负(1e7 -0.44 / 1e8 -0.24pp)而 E2b_10 为正(+1.24 / +1.50pp)。"
                       "1000 万取整后仓位降到 95.5%~97.4%(高价股持不起一手)"),
             ledger="inc_t7_fix_lots_stats.csv; inc_t7_fix_lots_sim.csv; inc_t7_fix_lots_levels.csv", change="D22;报告 §7 的“叠加后”数字加注“按比例成本、无整手 / 无最低佣金”,实施建议改为 >=1e8 且以 E2b 为准"),
        dict(id="R4", severity="P2", finding="通过的只有探索格(三个主候选全败),实质是一个发现;开发窗幅度偏弱",
             disposition="采纳(不需重跑,一句话结论补写)",
             evidence="P1 k10 -0.93pp(t -4.31)、P2 k5 30/45 +0.31pp(t 0.61)、P3 E1_6 +0.46pp(t 1.80)均不过;k=1 三格互为邻居(C6 区分力弱);k1|20/35 开发窗 +0.355pp(t 1.59)< 0.5pp,pess 开发窗 +0.18pp",
             ledger="inc_t7_paired.csv; inc_t7_criteria.csv", change="一句话结论"),
        dict(id="R5", severity="P2", finding="D04 把 §7.3 的降级触发改为只在“有增量”不成立时生效;D01-D11 的 04:08 时间戳无法核实",
             disposition="采纳(登记 D19 / D23,与 R1 合并后判定不再依赖 D04)",
             evidence="rule73_cells = E1_6, k1|30/30, k5|30/30, k5|30/40;PREREG mtime 04:01:55,冻结清单 04:01:58,首个计算 04:12:10,freeze verify 仍通过",
             ledger="inc_t7_deviations.csv; inc_t7_verdict.json", change="D19、D23"),
    ]
    pd.DataFrame(rv).to_csv(LEDGER / "inc_t7_review.csv", index=False)
    log("finalize done:summary", len(rows), "行;偏离 D18-D24;审查记录 R1-R5")


STAGES = {"verify": stage_verify, "price": stage_price, "lock": stage_lock, "lots": stage_lots, "e2bx": stage_e2bx, "xcheck": stage_xcheck, "finalize": stage_finalize}


def main():
    global _LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    a = ap.parse_args()
    LOGDIR.mkdir(exist_ok=True)
    _LOGF = LOGDIR / f"inc_t7_fix_{a.stage}.log"
    t0 = time.time()
    S = MinSim() if a.stage != "finalize" else None
    if S is not None:
        log(f"MinSim 载入 {time.time() - t0:.1f}s,T={S.T} N={S.N}")
    for st in (list(STAGES) if a.stage == "all" else [a.stage]):
        t1 = time.time()
        STAGES[st](S)
        log(f"stage {st} 完成 {time.time() - t1:.1f}s")


if __name__ == "__main__":
    main()
