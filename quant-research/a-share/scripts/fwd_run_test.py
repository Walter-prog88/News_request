"""前向检验 T1 / T2 / T3 / T5 的运行与读数(前缀 fwd;严格按 research/FORWARD_TEST_PREREG.md)。

纪律:
  - 只有信号日 >= 2026-09-01 的收益被计算 / 落盘:引擎与基准吃"收益面板"(twap_open10 在 09-02 之前为 NaN),
    预热段持仓收益恒为 0;date_mask 首日 = 09-01(冷启动,前 FWD_SKIP_HEAD = 4 个信号日只建仓、不计入);
  - 参数全部是冻结值:COMP_v5 打分(m5_bridge._comp_score)、V1000 / open10 / 1000 万 / 缓冲 20/35 / 5 日 5 相位;
    成本 cost_model=None(旧成本)与 CostModel(impact_y=1.0);T3 = OptConfig() 默认、IC 冻结 0.11196633374948033、
    中证 1000 实际成分(fwd_decisions:有效条件全部满足);
  - 冻结代码只经 fwd_adapters 放宽日期守卫 / 换数据源(研究期等价性检验见 fwd_adapter_equiv.csv)。

阶段(结果落盘 cache/_tmp/fwd_*,可续跑):
    python3 scripts/fwd_run_test.py score t12 risk t3 t5 report
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor.config import TMP, LEDGER, REPORTS, ROOT
from afactor.fwd_panel import ForwardPanel, forward_context, assert_forward_index, SIG_START
from afactor import fwd_adapters as FA_, fwd_weights as FW
import fwd_common as FC

log = FC.log
IC_FROZEN = 0.11196633374948033
RISK_DIR = TMP / "fwd_risk"
BENCH_FILE = TMP / "fwd_bench.parquet"
CAPITAL = 1e7
PLANS = {"T1_excl": dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5),
         "T2_top20": dict(scheme="top", n=20, rebalance=5)}
_P: dict = {}


def panels():
    if "ret" not in _P:
        _P["feat"] = ForwardPanel()
        _P["ret"] = ForwardPanel(returns_view=True)
    return _P["feat"], _P["ret"]


def cost_model(p):
    from afactor.impact import CostModel
    key = "cm"
    if key not in _P:
        with FA_.lifted_guards(raw_close=p.raw("raw_close")):
            _P[key] = CostModel(p, impact_y=1.0)
    return _P[key]


def score5():
    pf, _ = panels()
    with forward_context("前向检验 T1-T3:COMP_v5 5 分钟口径打分(信号日 t 收盘可知;预热段打分只作特征)"):
        return FC.comp_score(pf)


def _save_series(df: pd.DataFrame | pd.Series, path, what: str):
    assert_forward_index(df, what)
    (df.to_frame() if isinstance(df, pd.Series) else df).to_parquet(path)


# ================================================================ score
def stage_score():
    s = score5()
    pf, _ = panels()
    u = pf["universe"]
    sig = pf.signal_days()
    cov = float((s.loc[sig].notna() & u.loc[sig]).to_numpy().sum() / u.loc[sig].to_numpy().sum())
    log(f"[score] 前向信号日打分覆盖(决策池内有限值比例)= {cov:.4f}")


# ================================================================ t12:T1 / T2 / 全池等权判据 + 基准
def stage_t12():
    from afactor import strategy as ST, index_proxy as IP
    pf, p = panels()
    s = score5()
    m = p.mask_from(SIG_START)
    cm = cost_model(pf)
    t0 = time.time()
    b5 = ST.benchmark(p, "V1000", "open10", date_mask=m, rebalance=5)["rets"]
    W = FW.csi1000_close_weights(p)
    rb_csi, Wx = IP.exec_time_bench_returns(p, "open10", W, return_weights=True)
    rb_csi = rb_csi.reindex(b5.index)
    real = FW.real_index_on_signal_days(p.dates, "open10").reindex(b5.index)
    B = pd.DataFrame({"pool5": b5, "csi1000_exec": rb_csi, "real000852": real})
    _save_series(B, BENCH_FILE, "基准")
    np.save(TMP / "fwd_csi1000_Wx.npy", Wx[np.asarray(p.dates >= SIG_START)].astype(np.float32))
    W.loc[W.index >= SIG_START].to_parquet(TMP / "fwd_csi1000_W.parquet")
    log(f"[t12] 基准:{len(B)} 个信号日 {B.index[0].date()}~{B.index[-1].date()} [{time.time()-t0:.0f}s]")
    runs = {}
    for plan, kw in PLANS.items():
        for cost, cmx in (("old", None), ("Y1", cm)):
            t1 = time.time()
            r = ST.run_strategy(s, p, "V1000", "open10", date_mask=m, cost_model=cmx, record=True, **kw)
            assert r["rets"].index.equals(b5.index), f"{plan}/{cost} 索引与同频基准不一致"
            runs[f"{plan}|{cost}"] = r
            ST.save_record(r, f"fwd_{plan}_{cost}")
            log(f"[t12] {plan} {cost}:{r['n_days']} 天,仓位均值 {r['exposure']:.3f},换手 {r['turnover']:.4f} "
                f"[{time.time()-t1:.0f}s]")
    const = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype=np.float32)
    r = ST.run_strategy(const, p, "V1000", "open10", scheme="top", n=100000, rebalance=5, date_mask=m,
                        cost_model=cm, record=True)
    ST.save_record(r, "fwd_CHK_poolew_Y1")
    runs["CHK_poolew|Y1"] = r
    D = pd.DataFrame({k: v["rets"] for k, v in runs.items()})
    _save_series(D, TMP / "fwd_strategy_rets.parquet", "策略日收益")
    for k, v in runs.items():
        assert_forward_index(v["record"]["daily"], k)
    log(f"[t12] 完成 [{time.time()-t0:.0f}s]")


# ================================================================ risk:前向特征面板上的风险模型
def stage_risk():
    from afactor.barra import RiskModel
    pf, _ = panels()
    t0 = time.time()
    stamp = RISK_DIR / "_panel_stamp.json"
    fresh = not FC.stamp_ok(stamp)                    # 面板重建过(审查 P0 复权基准修正)-> 风险模型缓存失效,重估
    with forward_context("前向检验 T3/T5:风险模型逐日估计(2026-01~08 的因子收益只作估计输入,不报告)"):
        with FA_.lifted_guards():
            rm = RiskModel(cache_dir=RISK_DIR).build(pf, "open10", force=fresh)
    FC.write_stamp(stamp)
    sig = pf.signal_days()
    av = [rm.available(d) for d in sig]
    log(f"[risk] 首个预测日 {rm.meta.get('first_forecast')};前向信号日可用 {sum(av)}/{len(sig)};"
        f"因子收益 {rm.meta.get('F_first')}~{rm.meta.get('F_last')}(只作估计输入)[{time.time()-t0:.0f}s]")


def load_rm():
    from afactor.barra import RiskModel
    with FA_.lifted_guards():
        return RiskModel.load("open10", cache_dir=RISK_DIR)


# ================================================================ t3:中证 1000 指增 V1(5 相位)
def stage_t3(phases=(0, 1, 2, 3, 4)):
    from afactor import strategy as ST, optimizer as OPT
    pf, p = panels()
    s = score5()
    rm = load_rm()
    W = FW.csi1000_close_weights(p)
    U = ST.universe(p, "V1000")
    icd = pd.Series(IC_FROZEN, index=p.dates)
    cm = cost_model(pf)
    m = p.mask_from(SIG_START)
    for ph in phases:
        t0 = time.time()
        enh = OPT.IndexEnhancer(p, rm, W, s, icd, U, OPT.OptConfig(), tag="fwd_V1")
        res = ST.simulate(enh, p, "open10", CAPITAL, rebalance=5, date_mask=m, phase=ph, cost_model=cm, record=True)
        ST.save_record(res, f"fwd_T3_V1_ph{ph}")
        lg = enh.log_frame()
        lg.to_parquet(TMP / f"fwd_T3_V1_ph{ph}_solver.parquet")
        assert_forward_index(res["rets"], f"T3 ph{ph}")
        log(f"[t3] ph{ph}:{res['n_days']} 天,求解 {len(lg)} 次,状态 {lg['status'].value_counts().to_dict()},"
            f"仓位均值 {res['exposure']:.3f} [{time.time()-t0:.0f}s]")


def load_t3() -> dict:
    """5 相位合并(与 run_index_enhance.load_variant 同口径:日收益 = 相位均值,持仓 = 相位均值)。"""
    from afactor.strategy import _REC_COLS
    ds, Hs, sl = [], [], []
    for ph in range(5):
        pre = TMP / f"fwd_T3_V1_ph{ph}"
        ds.append(pd.read_parquet(f"{pre}_daily.parquet").assign(phase=ph))
        Hs.append(pd.read_parquet(f"{pre}_holdings.parquet"))
        sl.append(pd.read_parquet(f"{pre}_solver.parquet").assign(phase=ph))
    byp = pd.concat(ds)
    idx = ds[0].index
    num = [c for c in _REC_COLS if c != "rebalance"]
    daily = byp.groupby(level=0)[num].mean().reindex(idx)
    cols = Hs[0].columns
    for h in Hs[1:]:
        cols = cols.union(h.columns)
    H = sum(h.reindex(index=idx, columns=cols).fillna(0.0).astype(np.float64) for h in Hs) / len(Hs)
    return dict(daily=daily, rets=daily["ret"], by_phase=byp, H=H, solver=pd.concat(sl, ignore_index=True))


# ================================================================ t5:事前 TE vs 实现(freeze_reference_dists.stage_te 同式)
def te_pred_series(H: pd.DataFrame, Wv: np.ndarray, rm, dates, codes) -> pd.Series:
    from afactor.barra import psd_sqrt
    TD = 242
    H = H.reindex(columns=codes).fillna(0.0)
    idx = H.index
    Hv = H.to_numpy(np.float64)
    pos = dates.get_indexer(idx)
    out = np.full(len(idx), np.nan)
    for j, i in enumerate(pos):                          # 逐行照搬 run_index_enhance.report() 的 te_pred
        if not np.isfinite(rm.cov[i, 0, 0]):
            continue
        a = Hv[j] - Wv[j]
        m = np.asarray(rm.mask[i])
        am = a[m]
        X = np.nan_to_num(np.asarray(rm.X[i])[m].astype(np.float64))
        Hm = rm._onehot(rm.ind[m])
        d = np.asarray(rm.spec[i])[m].astype(np.float64)
        L = psd_sqrt(rm.cov[i])[1]
        with np.errstate(all="ignore"):
            e = np.concatenate([[am.sum()], np.dot(X.T, am), np.dot(Hm.T, am)])
            y = np.dot(L.T, e)
        out[j] = np.sqrt(np.sum(y * y) + np.sum(np.nan_to_num(d) * am * am)) * np.sqrt(TD / 5)
    return pd.Series(out, index=idx)


def stage_t5():
    pf, p = panels()
    rm = load_rm()
    B = pd.read_parquet(BENCH_FILE)
    Wx = np.load(TMP / "fwd_csi1000_Wx.npy")                  # 行 = 面板日期 >= 09-01
    wdates = p.dates[p.dates >= SIG_START]
    out = {}
    V = load_t3()
    H3 = V["H"]
    for nm, H, r in (("T5_V1", H3, V["rets"]),
                     ("T5_poolew", pd.read_parquet(TMP / "fwd_CHK_poolew_Y1_holdings.parquet"),
                      pd.read_parquet(TMP / "fwd_CHK_poolew_Y1_daily.parquet")["ret"])):
        Wv = pd.DataFrame(Wx, index=wdates, columns=p.codes).reindex(H.index).to_numpy(np.float64)
        tp = te_pred_series(H, Wv, rm, p.dates, p.codes)
        act = (r - B["csi1000_exec"]).reindex(tp.index)
        z = act / (tp / np.sqrt(242))
        out[f"{nm}|te_pred_ann"] = tp
        out[f"{nm}|active"] = act
        out[f"{nm}|z"] = z
    T = pd.DataFrame(out)
    _save_series(T, TMP / "fwd_te_daily.parquet", "T5")
    log(f"[t5] 事前 TE 与 z:{len(T)} 天;V1 事前 TE 中位 {T['T5_V1|te_pred_ann'].median():.4f}")


if __name__ == "__main__":
    for st in sys.argv[1:]:
        t0 = time.time()
        log(f"===== {st} =====")
        globals()[f"stage_{st}"]()
        log(f"===== {st} 完成 {time.time() - t0:.0f}s =====")
