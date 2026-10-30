"""压力测试 —— "你的策略什么时候会死"(前缀 stress,2026-10-10)。

回答面试官会问的问题:2024 年初小微盘流动性危机里,这套因子/组合亏了多少、为什么亏、
事前能不能看出来。窗口、对象、口径与判据全部**先写进** research/ledger/stress_preregistration.json
(以及 research/STRESS_TEST.md §0)再计算,本脚本只读那份预注册,不在代码里另设窗口。

## 分阶段运行(每阶段把中间结果落到 cache/_tmp/stress_*,可单独重跑)

    python3 scripts/run_stress_test.py engine    # 回测(带持仓记录的引擎复刻 + 与 ST.run_strategy 逐位核对)
    python3 scripts/run_stress_test.py dims      # 11 个独立维度代表因子的十分位多空 / 排除最差 20%
    python3 scripts/run_stress_test.py analyze   # Barra 归因、事前情景、流动性、反事实、幸存者统计,写 ledger
    python3 scripts/run_stress_test.py figure    # reports/stress_2024q1.png

审查修复(2026-10-10)后新增:engine 阶段另存同钟池等权收盘口径 POOL_C2C、中证1000 te+1 对齐
CSI1000_TE1、开盘 10 分钟钉跌停的持仓诊断(pin_hold/pin_sell)、参与率按净值计的敏感性
(stress_rets_navcap / stress_cash_navcap);analyze 阶段另写 stress_specific_matched.csv、
stress_participation_sensitivity.csv、stress_survivorship.json。

## 口径要点(为什么这么做)

1. **为什么要复刻引擎而不是直接调 ST.run_strategy**:压力归因需要逐日持仓,而 run_strategy 只返回收益。
   复刻版逐行照抄 `_run_one_phase` 的浮点运算顺序,只在旁边多记几个诊断量(跌停锁仓、参与率截断),
   并对每一个被测方案断言"复刻收益 == ST.run_strategy 收益"(最大绝对差 <= 1e-12)。
   不修改 strategy.py(另一个代理正在原子替换它)。
2. **相位平均的持仓**:run_strategy 的 5 相位平均 = 5 个子组合日收益等权平均,
   所以逐日的"有效持仓" = 5 个相位成交后持仓的简单平均,归因用它可以精确复原策略毛收益。
3. **日期**:所有收益序列按**信号日 t** 索引(引擎原生口径),收益在 t+1 开盘 -> t+2 开盘实现;
   窗口按成交日 te = t+1 归属(预注册口径)。
4. **事前**:事前情景只用"窗口首个信号日 t_s 的前一日 t_s-1 的成交后持仓"与 t_s 日收盘可得的暴露;
   风险模型的协方差/特质方差只用截至 t_s-2 的已实现纯因子收益(t_s-2 的收益在 t_s 开盘实现,
   是 t_s 收盘时点已知的最后一期)。
5. **封存**:中证1000 指数序列读入后立即截到 2025-12-31;面板本身只到 2025-12-31。
6. **样本属性**:COMP_v5 的成员是 2020-2025 内选出的,COMP 系列结果是**信号样本内**;
   B_gbt 打分在 2023-2025 是 walk-forward 样本外。数据缺约 209 只已退市股。
"""
from __future__ import annotations
import os, sys, json, time, glob, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from afactor.config import (LEDGER, TMP, REPORTS, RESEARCH_END, COST_BPS_ONEWAY,
                            STAMP_TAX_BPS_BEFORE, STAMP_TAX_BPS_AFTER, STAMP_TAX_CHANGE_DATE,
                            MAX_PARTICIPATION)
from afactor.panel import Panel
from afactor import strategy as ST, backtest as BT, factors as FA, newdata as ND
from afactor.strategy import TRADING_DAYS

pd.set_option("display.width", 240)
VERSION, MODE = "V1000", "open10"
PRE = json.load(open(LEDGER / "stress_preregistration.json"))
WINDOWS = {k: (pd.Timestamp(v["start"]), pd.Timestamp(v["end"]), v["desc"])
           for k, v in PRE["windows"].items()}
DIM_REPS = PRE["dim_reps"]
SCORE_COMP = TMP / "comp_v5_score_open10.parquet"
SCORES_DIR = LEDGER / "scores"
DEV = ("2021-01-01", "2023-12-31")
TEST = ("2024-01-01", "2025-12-31")
FULL = ("2021-01-01", "2025-12-31")      # 2020 只作预热
STYLE_KEYS = ["Size 市值", "Beta 市场敏感", "Momentum 动量", "ResidVol 残差波动",
              "NonlinSize 非线性市值", "BookToPrice 价值", "Liquidity 流动性",
              "EarningsYield 盈利", "Growth 成长", "Leverage 杠杆"]
SHORT = {k: k.split(" ")[0] for k in STYLE_KEYS}

# 被测组合(预注册):name -> (打分来源, 方案)
PLANS = {
    "COMP_EXCL": dict(scheme="exclude", buffer=(0.20, 0.35), k=5),
    "COMP_TOP20": dict(scheme="top", n=20, k=5),
    "COMP_TOP50": dict(scheme="top", n=50, k=5),
    "BGBT_TOP20": dict(scheme="top", n=20, k=5),
}
NEUTRAL = {"N1": ["Size 市值"],                                          # 主反事实(预注册)
           "N3": ["Size 市值", "NonlinSize 非线性市值", "Liquidity 流动性"],   # 敏感性(预注册)
           # 事后追加(第一轮结果出来后才加,报告中单独标注):Barra Size 按 ±3σ 截尾且以市值加权均值
           # 为中心,2023-12-28 池内 37.6% 的股票挤在 -3,线性 Size 中性化分不清小盘与微盘。
           # N1r 改为对**池内 log 市值的秩**(不截尾、池内均匀分布)中性化。
           "N1r": ["__lncap_rank__"]}
# 事后追加 NS:**组合层**市值中性 —— 打分换成"池内市值五分位内的名次",Top-N 因此在 5 个市值档等量入选,
# 排雷也在每档内各剔最差 20%,组合市值分布按构造 ≈ 池等权。追加原因:N1/N1r 只中性化**打分**,
# 实测 COMP Top-N 的持仓池内市值秩反而从 0.47 降到 0.35/0.30(更小),打分中性 ≠ 组合中性。
VARIANTS = ["", "|N1", "|N3", "|N1r", "|NS"]


def size_stratified(score: pd.DataFrame, mv: pd.DataFrame, u: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """池内市值 k 分位内各自的打分名次(0~1)。只用信号日 t 的市值与打分。"""
    s = score.reindex_like(u).where(u)
    q = np.ceil(mv.reindex_like(u).where(u).rank(axis=1, pct=True) * k).clip(1, k)
    out = pd.DataFrame(np.nan, index=u.index, columns=u.columns, dtype=np.float32)
    for g in range(1, k + 1):
        m = (q == g)
        out = out.mask(m, s.where(m).rank(axis=1, pct=True))
    return out.where(s.notna()).astype(np.float32)
SAMPLE_FLAG = {"COMP": "信号样本内", "BGBT": "样本外(walk-forward)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tmp(name: str):
    return TMP / f"stress_{name}.parquet"


# ====================================================================== 工具
def nw_t(x, q: int = 10) -> float:
    """Newey-West t(Bartlett 核)。周频调仓的日超额有自相关,带宽取 max(5, 2h)=10。"""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = float((e ** 2).sum() / n)
    for L in range(1, q + 1):
        s += 2.0 * (1.0 - L / (q + 1.0)) * float((e[L:] * e[:-L]).sum() / n)
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def corr_peak(rs: pd.Series, rb: pd.Series, ks=(-2, -1, 0, 1, 2)) -> tuple[int, dict]:
    idx = rs.index.intersection(rb.index)
    rs, rb = rs.reindex(idx), rb.reindex(idx)
    cs = {k: float(rs.corr(rb.shift(k))) for k in ks}
    return max(ks, key=lambda k: cs[k]), cs


def te_map(dates: pd.DatetimeIndex) -> pd.Series:
    """信号日 t -> 成交日 te = t 的下一交易日。"""
    return pd.Series(dates[1:], index=dates[:-1])


def win_signal_days(index: pd.DatetimeIndex, te: pd.Series, d0, d1) -> pd.DatetimeIndex:
    t = te.reindex(index)
    return index[((t >= d0) & (t <= d1)).to_numpy()]


# ====================================================================== 引擎复刻(带持仓记录)
def _phase_rec(W, U, px, win_amt, dead, ldown, dates, k, date_mask, phase, capital, lag, pin=None):
    """逐行复刻 strategy._run_one_phase 的成交循环,并额外记录:
      H[t]        成交后持仓 newh(信号日 t 行)
      ld_hold     成交前持仓中当日(te)一字跌停的权重 —— "想卖也卖不掉"的暴露
      ld_block    调仓日:一字跌停股中目标权重低于持仓的部分(想减仓但被锁住)
      clip_vol / want_vol  调仓日:参与率上限截掉的成交量 / 意图成交量(只算可交易股)
      clip_any    调仓日是否有任何一笔被参与率截断
      pin_hold    成交前持仓中 te 日**开盘 10 分钟整段钉在跌停价**的权重(含一字跌停)。
                  审查补充(2026-10-10):全天一字跌停的口径漏掉"开盘钉死跌停、盘中才打开"的票,
                  这些票在 09:31-09:40 的 TWAP 窗口里同样只能排队,卖单大概率成交不了。
      pin_sell    引擎按 TWAP 实际成交了的、落在上述开盘钉跌停股上的卖出量(引擎视其可成交,
                  现实中可能排队卖不出 —— 这是引擎对危机的乐观偏差上限)
    ⚠ 主循环的浮点运算顺序与原引擎完全一致,诊断量只在旁路计算,保证收益逐位相同。"""
    T, N = W.shape
    h = np.zeros(N)
    H = np.zeros((T, N), np.float32)
    rets, ridx, turns, costs, exps = [], [], [], [], []
    diag = {key: [] for key in ("ld_hold", "ld_block", "clip_vol", "want_vol", "clip_any", "reb",
                                "pin_hold", "pin_sell")}
    stamp_switch = np.searchsorted(dates.values, np.datetime64(STAMP_TAX_CHANGE_DATE))
    t0 = int(np.flatnonzero(date_mask)[0]) if date_mask is not None and date_mask.any() else 0
    t0 += int(phase)
    rebalance = k
    for t in range(T - lag - 1):
        te = t + lag
        if te >= T - 1:
            break
        p_now, p_next = px[te], px[te + 1]
        ok_px = np.isfinite(p_now) & np.isfinite(p_next) & (p_now > 0)
        can = ok_px & ~dead[te] & (win_amt[te] > 0)
        held_dn = ldown[te] & (h > 1e-12)
        ld_hold = float(h[held_dn].sum())
        pin_te = pin[te] if pin is not None else np.zeros(N, bool)
        pin_hold = float(h[pin_te & (h > 1e-12)].sum())
        ld_block = clip_vol = want_vol = 0.0
        clip_any = 0.0
        if rebalance > 1 and ((t - t0) % rebalance):
            delta = np.zeros(N)
            reb = 0.0
        else:
            reb = 1.0
            stuck = (~can) & (h > 1e-12)
            budget = max(0.0, 1.0 - float(h[stuck].sum()))
            tgt = np.where(can, W[t].astype(np.float64), 0.0)
            s = tgt.sum()
            if s > 0:
                tgt = tgt / s * budget
            tgt = tgt + np.where(stuck, h, 0.0)
            delta = tgt - h
            cap_w = np.where(np.isfinite(win_amt[te]),
                             MAX_PARTICIPATION * win_amt[te] / capital, 0.0)
            delta_pre = delta
            delta = np.clip(delta, -cap_w, cap_w)
            delta = np.where(can, delta, 0.0)
            # ---- 旁路诊断(不影响主路径)----
            want = np.where(can, np.abs(delta_pre), 0.0)
            want_vol = float(want.sum())
            clip_vol = float((want - np.abs(delta)).clip(min=0).sum())
            clip_any = float(clip_vol > 1e-12)
            ld_block = float(np.clip(h[held_dn] - W[t][held_dn], 0, None).sum())
            # ---- 主路径继续 ----
            over = float(np.clip(h + delta, 0, None).sum()) - 1.0
            if over > 1e-12:
                buys = np.clip(delta, 0, None); bs = buys.sum()
                if bs > 0:
                    delta = delta - buys * min(1.0, over / bs)
        newh = np.clip(h + delta, 0, None)
        buy = np.clip(delta, 0, None).sum(); sell = np.clip(-delta, 0, None).sum()
        stamp = (STAMP_TAX_BPS_BEFORE if te < stamp_switch else STAMP_TAX_BPS_AFTER) * 1e-4
        cost = (buy + sell) * COST_BPS_ONEWAY * 1e-4 + sell * stamp
        pin_sell = float(np.clip(-delta, 0, None)[pin_te].sum())      # 旁路诊断
        with np.errstate(invalid="ignore"):
            r = np.where(ok_px, p_next / p_now - 1.0, 0.0)
        port_r = float(np.nansum(newh * r)) - cost
        grown = newh * (1.0 + np.nan_to_num(r))
        h = grown / (1.0 + port_r) if (1.0 + port_r) > 1e-9 else grown
        hs = h.sum()
        if hs > 1.0 + 1e-12:
            h = h / hs
        rets.append(port_r); ridx.append(dates[t])
        turns.append(np.abs(delta).sum() / 2); costs.append(cost); exps.append(float(newh.sum()))
        H[t] = newh
        for key, v in (("ld_hold", ld_hold), ("ld_block", ld_block), ("clip_vol", clip_vol),
                       ("want_vol", want_vol), ("clip_any", clip_any), ("reb", reb),
                       ("pin_hold", pin_hold), ("pin_sell", pin_sell)):
            diag[key].append(v)
        assert newh.sum() <= 1.0 + 1e-9, f"exposure>1: {newh.sum():.6f} at t={t}"
    idx = pd.DatetimeIndex(ridx)
    r_s = pd.Series(rets, index=idx)
    D = pd.DataFrame(diag, index=idx)
    D["cost"] = costs; D["turnover"] = turns; D["exposure"] = exps
    if date_mask is not None:
        keep = pd.Series(date_mask[:len(dates)], index=dates).reindex(idx).fillna(False).to_numpy(bool)
        if keep.sum() < 10:
            return None
        r_s, D = r_s[keep], D[keep]
    return dict(rets=r_s, H=H, diag=D)


_PIN_CACHE: dict = {}


def pinned_down(p: Panel) -> pd.DataFrame:
    """te 日开盘 10 分钟(09:31-09:40)整段钉在跌停价附近:open10_high / 前收 - 1 <= -涨跌幅上限 x 0.98。
    并显式并上全天一字跌停(p["one_word_down"])。只用 te 日的盘口数据描述 te 日的成交环境,不进入任何信号。
    价格口径:open10_high 与 close_px 同为前复权分钟数据,比值不受除权影响。
    ⚠ 为什么要显式并上一字跌停:前复权价的分位取整让 2020-2025 有 207 个一字跌停股票日(共 5401 个)的
    open10_high/前收 只到 -9.6% 左右,达不到 0.98 x 10% 的门槛;不并上就不是"含一字跌停"的超集。"""
    key = id(p)
    if key not in _PIN_CACHE:
        from afactor.panel import _limit_frame
        lim = _limit_frame(p)
        pc = p.raw("close_px").shift(1)
        pin = ((p.raw("open10_high") / pc - 1.0) <= -lim * 0.98).fillna(False)
        _PIN_CACHE[key] = pin | p["one_word_down"].reindex_like(pin).fillna(False).astype(bool)
    return _PIN_CACHE[key]


def run_rec(score: pd.DataFrame, p: Panel, scheme: str, n: int = 20, k: int = 5,
            buffer=None, date_mask=None, check: bool = True, tag: str = "") -> dict:
    """带持仓记录的 run_strategy 复刻(5 相位平均)。check=True 时与 ST.run_strategy 逐位核对。"""
    cfg = ST.VERSIONS[VERSION]
    lag = ST.EXEC_MODES[MODE]["lag"]
    dates, codes = p.dates, p.codes
    U = ST.universe(p, VERSION).to_numpy(bool)
    if date_mask is not None:
        U = U & date_mask[:, None]
    S = score.reindex(index=dates, columns=codes).to_numpy(np.float32) * 1
    if scheme == "top":
        W = ST.topn_weights(S, U, n)
    elif scheme == "vw":
        W = np.where(U & np.isfinite(S), np.clip(S, 0, None), 0.0).astype(np.float32)
        rs = W.sum(axis=1, keepdims=True)
        W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    else:
        W = BT.weights_exclude_buffer(S, U, 1, *buffer)
    px_df, amt_df = ST.exec_prices(p, MODE)
    px = px_df.to_numpy(np.float64)
    win_amt = amt_df.to_numpy(np.float64)
    ow_dn = p["one_word_down"].to_numpy(bool)
    dead = (p["one_word_up"] | p["one_word_down"]).to_numpy(bool)
    pin = pinned_down(p).reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
    subs = []
    for ph in (range(k) if k > 1 else [0]):
        x = _phase_rec(W, U, px, win_amt, dead, ow_dn, dates, k, date_mask, ph, cfg["capital"], lag, pin)
        if x is not None and len(x["rets"]) > 5:
            subs.append(x)
    R = pd.concat([x["rets"] for x in subs], axis=1)
    r_s = R.mean(axis=1)
    H = np.mean(np.stack([x["H"] for x in subs]), axis=0).astype(np.float32)
    Dm = sum(x["diag"] for x in subs) / len(subs)            # 相位平均的诊断量
    Dm["clip_any_cnt"] = sum(x["diag"]["clip_any"] for x in subs)   # 当日有截断的相位数
    Dm["reb_cnt"] = sum(x["diag"]["reb"] for x in subs)              # 当日调仓的相位数
    out = dict(rets=r_s, H=H, diag=Dm, n_phases=len(subs))
    if check:
        ref = ST.run_strategy(score, p, VERSION, MODE, scheme=scheme, n=n, rebalance=k,
                              buffer=buffer, date_mask=date_mask)
        rr = ref["rets"].reindex(r_s.index)
        out["max_abs_diff"] = float(np.nanmax(np.abs(rr.to_numpy() - r_s.to_numpy())))
        out["ref_ann"] = ref["ann_return"]
        assert len(ref["rets"]) == len(r_s) and out["max_abs_diff"] <= 1e-12, \
            f"[{tag}] 复刻引擎与 ST.run_strategy 不一致: {out['max_abs_diff']}"
    return out


# ====================================================================== Barra 暴露 / 中性化
def load_exposures(p: Panel) -> dict[str, pd.DataFrame]:
    """Barra 十风格暴露(scripts/run_barra_attribution.build_exposures,不改其口径),缓存到 _tmp。"""
    f0 = tmp("X_Size")
    if f0.exists():
        return {k: pd.read_parquet(tmp(f"X_{SHORT[k]}")) for k in STYLE_KEYS}
    import run_barra_attribution as RBA
    FA.build_all()
    log("构建 Barra 暴露度…")
    X = RBA.build_exposures(p, p["universe"])
    missing = [k for k in STYLE_KEYS if k not in X]
    assert not missing, f"暴露度缺失 {missing}"
    for k in STYLE_KEYS:
        X[k].astype(np.float32).to_parquet(tmp(f"X_{SHORT[k]}"))
    return {k: X[k].astype(np.float32) for k in STYLE_KEYS}


def neutralize(score: pd.DataFrame, Xs: list[pd.DataFrame], u: pd.DataFrame) -> pd.DataFrame:
    """逐日截面 OLS:score ~ 1 + Xs,决策池内取残差。
    暴露缺失的股票按暴露 0(=截面加权均值)处理而不是丢掉——否则中性化会顺带改变可选股票集合,
    反事实就混入了"股票池变小"的效应。"""
    dates, codes = u.index, u.columns
    S = score.reindex(index=dates, columns=codes).to_numpy(np.float64)
    Xa = [x.reindex(index=dates, columns=codes).to_numpy(np.float64) for x in Xs]
    U = u.to_numpy(bool)
    out = np.full(S.shape, np.nan)
    for t in range(len(dates)):
        m = U[t] & np.isfinite(S[t])
        if m.sum() < 50:
            continue
        A = np.column_stack([np.ones(m.sum())] + [np.nan_to_num(x[t, m]) for x in Xa])
        b, *_ = np.linalg.lstsq(A, S[t, m], rcond=None)
        out[t, m] = S[t, m] - A @ b
    return pd.DataFrame(out.astype(np.float32), index=dates, columns=codes)


def load_bgbt(p: Panel) -> tuple[pd.DataFrame, np.ndarray]:
    fs = sorted(glob.glob(str(SCORES_DIR / "B_gbt_open_h5_*.parquet")))
    assert [os.path.basename(f)[-12:-8] for f in fs] == ["2023", "2024", "2025"], fs
    sc = pd.concat([pd.read_parquet(f) for f in fs]).sort_index()
    assert sc.index.max() <= pd.Timestamp(RESEARCH_END)
    full = sc.reindex(index=p.dates, columns=p.codes).astype(np.float32)
    mask = np.isin(np.asarray(p.dates.year), [2023, 2024, 2025])
    return full, mask


def csi1000_aligned(p: Panel, bench: pd.Series) -> tuple[pd.Series, dict, pd.Series, pd.Series]:
    """中证1000 只有收盘价。open10 信号日 t 的收益主体是 te=t+1 的盘中,故取 te 日收盘对收盘。
    先用**池等权的收盘口径**校准对齐方式(te vs te+1),不用任何策略收益校准。

    返回 (预注册对齐的中证1000, 校准量, te+1 对齐的中证1000, 池等权同钟收盘对收盘)。

    审查修正(2026-10-10):池等权 open10 收益覆盖 [te 开盘, te+1 开盘],中证1000 只有收盘价,
    两种对齐(te 收盘对收盘 / te+1 收盘对收盘)都覆盖不了这一区间,差半天。在开盘跳空大的日子
    (2024-09-24、2024-10-08)这半天单日就能制造 ±15pp 的"超额",窗口级比较会被单日钟点错位主导。
    所以"池等权 vs 中证1000"的主口径改为**同钟**:同一个池(信号日 t 的 V1000 池)按
    close[te]/close[te-1]-1 等权,与中证1000 的 te 日收盘对收盘完全同一时钟;
    open10 池等权对两种中证1000 对齐的结果只作为区间报告。策略 vs 池等权的比较全部同钟,不受影响。"""
    u = ST.universe(p, VERSION)
    c = p["close_px"]
    c2c = c / c.shift(1) - 1.0
    cal = {}
    pool_c = {}
    for lead in (1, 2):
        pool_c[lead] = c2c.shift(-lead).where(u).mean(axis=1).reindex(bench.index)
        cal[f"corr_pool_open10_vs_pool_c2c_te+{lead - 1}"] = float(bench.corr(pool_c[lead]))
    idx = ND.index_daily()["000852"]
    idx = idx[idx.index <= pd.Timestamp(RESEARCH_END)].astype(float)   # 封存:立即截断
    ic2c = idx.pct_change()
    lead = 1 if cal["corr_pool_open10_vs_pool_c2c_te+0"] >= cal["corr_pool_open10_vs_pool_c2c_te+1"] else 2
    r = ic2c.reindex(p.dates).shift(-lead).reindex(bench.index)
    r_alt = ic2c.reindex(p.dates).shift(-(3 - lead)).reindex(bench.index)    # 另一种对齐(敏感性)
    cal["csi_lead_used"] = lead - 1
    cal["csi_missing_days"] = int(r.isna().sum())
    pool_same = pool_c[lead]                       # 与所用中证1000 对齐同钟的池等权收盘口径
    cal["corr_pool_c2c_sameclock_vs_csi"] = float(pool_same.corr(r))
    # 钟点错位的量化:日超额的标准差与一阶自相关(2021 起)。错位噪声会让 sd 偏大、自相关显著为负。
    for tag, a, b in (("open10_vs_csi_te", bench, r), ("open10_vs_csi_te+1", bench, r_alt),
                      ("sameclock_c2c_vs_csi_te", pool_same, r)):
        e = (a - b).loc["2021-01-01":]
        cal[f"daily_excess_sd[{tag}]"] = float(e.std())
        cal[f"daily_excess_ac1[{tag}]"] = float(e.autocorr(1))
    return r.fillna(0.0), cal, r_alt.fillna(0.0), pool_same


# ====================================================================== 阶段 1:引擎
def stage_engine():
    p = Panel()
    dates = p.dates
    u = ST.universe(p, VERSION)
    X = load_exposures(p)
    bench = ST.benchmark(p, VERSION, MODE)["rets"]
    csi, cal, csi_alt, pool_c2c = csi1000_aligned(p, bench)
    log("中证1000 对齐校准", cal)

    comp = pd.read_parquet(SCORE_COMP)
    bg, bg_mask = load_bgbt(p)
    scores = {"COMP": comp, "BGBT": bg}
    # 池内 log 市值的秩(0~1),只用 t 日市值,N1r 用
    lncap_rank = np.log(FA._real(p, "mv").where(u)).rank(axis=1, pct=True)
    X = dict(X, __lncap_rank__=lncap_rank)
    for nk, keys in NEUTRAL.items():
        for sk in ("COMP", "BGBT"):
            log(f"中性化 {sk} | {nk} {keys}")
            scores[f"{sk}|{nk}"] = neutralize(scores[sk], [X[k] for k in keys], u)
    mv_t = FA._real(p, "mv")
    for sk in ("COMP", "BGBT"):
        log(f"市值分层 {sk} | NS")
        scores[f"{sk}|NS"] = size_stratified(scores[sk], mv_t, u)

    rets = {"POOL_EW": bench, "CSI1000": csi, "CSI1000_TE1": csi_alt, "POOL_C2C": pool_c2c}
    checks = []
    for name, plan in PLANS.items():
        src = "BGBT" if name.startswith("BGBT") else "COMP"
        for var in VARIANTS:
            obj = name + var
            sc = scores[src + var]
            dm = bg_mask if src == "BGBT" else None
            t1 = time.time()
            res = run_rec(sc, p, plan["scheme"], n=plan.get("n", 20), k=plan["k"],
                          buffer=plan.get("buffer"), date_mask=dm, tag=obj)
            r = res["rets"]
            rets[obj] = r
            pd.DataFrame(res["H"], index=dates, columns=p.codes).to_parquet(tmp(f"H_{obj}"))
            res["diag"].to_parquet(tmp(f"diag_{obj}"))
            peak, cs = corr_peak(r, bench)
            assert peak == 0, f"{obj} 与基准错位 {peak}"
            checks.append(dict(object=obj, check="replica_vs_ST_max_abs_diff", value=res["max_abs_diff"]))
            checks.append(dict(object=obj, check="ann_return_ST", value=res["ref_ann"]))
            checks.append(dict(object=obj, check="max_exposure", value=float(res["diag"]["exposure"].max())))
            checks.append(dict(object=obj, check="corr_peak_k", value=peak))
            for kk, v in cs.items():
                checks.append(dict(object=obj, check=f"corr_k{kk:+d}", value=v))
            log(f"{obj:22s} 年化 {res['ref_ann']:+.4f} 复刻差 {res['max_abs_diff']:.1e} "
                f"峰值 k={peak} 用时 {time.time() - t1:.0f}s")

    # 回归断言 ①:全池等权 vs 池等权基准,超额 ≈ -成本
    ones = u.astype(np.float32).where(u)
    res = run_rec(ones, p, "vw", k=5, check=True, tag="FULLPOOL_EW")
    r = res["rets"]
    ex = float(((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1) - ((1 + bench.reindex(r.index)).prod() ** (TRADING_DAYS / len(r)) - 1))
    checks += [dict(object="FULLPOOL_EW", check="ann_excess_vs_pool", value=ex),
               dict(object="FULLPOOL_EW", check="ann_cost", value=float(res["diag"]["cost"].mean() * TRADING_DAYS)),
               dict(object="FULLPOOL_EW", check="replica_vs_ST_max_abs_diff", value=res["max_abs_diff"]),
               dict(object="FULLPOOL_EW", check="corr_peak_k", value=corr_peak(r, bench)[0])]
    peak, cs = corr_peak(csi, bench)
    checks.append(dict(object="CSI1000", check="corr_peak_k_vs_pool", value=peak))
    peak, cs = corr_peak(pool_c2c.dropna(), csi)
    checks.append(dict(object="POOL_C2C", check="corr_peak_k_vs_csi1000", value=peak))
    for kk, v in cal.items():
        checks.append(dict(object="CSI1000", check=kk, value=v))
    pd.DataFrame(rets).to_parquet(tmp("rets"))

    # 审查补充(2026-10-10):参与率上限的资金口径敏感性。引擎默认按**初始资金** 1000 万计参与率,
    # 不随净值增长;W1 开始时净值约 1.18,真实 AUM 已超过 1000 万,默认口径低估了参与率。
    # 这里用 ST.run_strategy(participation_on_nav=True) 重跑 4 个基础方案(只作敏感性,不改主口径)。
    nav_rets, nav_cash = {}, {}
    for name, plan in PLANS.items():
        sc = scores["BGBT" if name.startswith("BGBT") else "COMP"]
        dm = bg_mask if name.startswith("BGBT") else None
        res = ST.run_strategy(sc, p, VERSION, MODE, scheme=plan["scheme"], n=plan.get("n", 20),
                              rebalance=plan["k"], buffer=plan.get("buffer"), date_mask=dm,
                              participation_on_nav=True, record=True)
        nav_rets[name] = res["rets"]
        nav_cash[name] = res["record"]["daily"]["cash"]
        log(f"{name:12s} participation_on_nav 年化 {res['ann_return']:+.4f}")
    pd.DataFrame(nav_rets).to_parquet(tmp("rets_navcap"))
    pd.DataFrame(nav_cash).to_parquet(tmp("cash_navcap"))
    pd.DataFrame(checks).to_csv(LEDGER / "stress_engine_checks.csv", index=False)
    log("engine 阶段完成")


# ====================================================================== 阶段 2:维度因子
def stage_dims():
    FA.build_all()
    p = Panel()
    u = ST.universe(p, VERSION)
    R = p.fwd_ret(1, "open")          # 信号日 t: twap_open10[t+2]/twap_open10[t+1]-1,与引擎同索引
    fs = pd.read_csv(LEDGER / "factor_summary_all.csv")
    fs = fs[fs["mode"] == "open"].drop_duplicates("factor").set_index("factor")["ic_mean"]
    Ra = R.to_numpy(np.float64)
    out = {}
    for name in DIM_REPS:
        t1 = time.time()
        sign = float(np.sign(fs[name]))
        f = FA.REGISTRY[name][1](p).reindex(index=p.dates, columns=p.codes)\
              .replace([np.inf, -np.inf], np.nan) * sign
        m = u.to_numpy(bool) & np.isfinite(f.to_numpy()) & np.isfinite(Ra)
        fm = f.where(m)
        pct = fm.rank(axis=1, pct=True).to_numpy()
        grp = np.ceil(pct * 10).clip(1, 10)
        dec = {}
        for g in (1, 10):
            sel = m & (grp == g)
            dec[g] = np.where(sel, Ra, 0).sum(1) / np.maximum(sel.sum(1), 1)
        allm = np.where(m, Ra, 0).sum(1) / np.maximum(m.sum(1), 1)
        top80 = m & (pct > 0.2)
        ex = np.where(top80, Ra, 0).sum(1) / np.maximum(top80.sum(1), 1)
        ok = m.sum(1) >= 100
        out[f"LS:{name}"] = np.where(ok, dec[10] - dec[1], np.nan)
        out[f"EX20:{name}"] = np.where(ok, ex, np.nan)
        out[f"EX20BM:{name}"] = np.where(ok, allm, np.nan)
        out[f"D1EX:{name}"] = np.where(ok, dec[1] - allm, np.nan)
        log(f"维度 {name:36s} sign {sign:+.0f} 用时 {time.time() - t1:.0f}s")
        del f, fm, pct, grp
        for key in list(p._derived):          # 释放该因子的派生缓存,控制内存
            if key not in ("valid", "days_listed", "adv20", "universe"):
                p._derived.pop(key, None)
    D = pd.DataFrame(out, index=p.dates)
    D = D.iloc[:len(p.dates) - 2]            # 与引擎收益同长度(最后 2 天没有实现收益)
    D.to_parquet(tmp("dims"))
    log("dims 阶段完成", D.shape)


# ====================================================================== 阶段 4:图
def stage_figure():
    """reports/stress_2024q1.png:2024Q1 窗口内的累计超额路径、持仓微盘占比、组合层市值中性反事实。
    静态图:三个上下排列的小图共享时间轴(各自一个 y 轴,不做双轴);色序取 dataviz 参考调色板前 5 槽,
    同一对象在三个小图里同色;W1/W1b 用浅灰底标出;直接标注线尾 + 图例。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    COL = {"COMP_EXCL": "#2a78d6", "COMP_TOP20": "#eb6834", "COMP_TOP50": "#1baf7a",
           "BGBT_TOP20": "#eda100", "CSI1000": "#e87ba4"}
    NAME = {"COMP_EXCL": "COMP 排雷(缓冲20/35)", "COMP_TOP20": "COMP Top20 周频",
            "COMP_TOP50": "COMP Top50 周频", "BGBT_TOP20": "B_gbt Top20(样本外)",
            "CSI1000": "中证1000(对同钟池等权)"}
    rets = pd.read_parquet(tmp("rets"))
    dates = rets.index
    te = pd.Series(list(dates[1:]) + [pd.NaT], index=dates)     # 信号日 -> 成交日
    a0, a1 = pd.Timestamp("2023-12-01"), pd.Timestamp("2024-03-29")
    base = pd.Timestamp("2024-01-02")                            # W1 首个成交日
    sel = (te >= a0) & (te <= a1)

    def cum_excess(col, bcol="POOL_EW"):
        """col 相对 bcol 的累计超额。中证1000 只能与**同钟**的池等权收盘口径(POOL_C2C)比
        (审查修正:与 open10 池等权比差半天,单日跳空会制造假超额)。"""
        r = rets[col][sel].fillna(0.0)
        b = rets[bcol][sel].fillna(0.0)
        rel = (1 + r).cumprod() / (1 + b).cumprod()
        x = te[sel]
        k = int(np.flatnonzero((x >= base).to_numpy())[0])
        rel = rel / (rel.iloc[k - 1] if k > 0 else 1.0)      # 以 W1 开始前为 0
        return pd.Series((rel - 1).to_numpy() * 100, index=x.to_numpy())

    path = pd.read_csv(LEDGER / "stress_holdings_path.csv", parse_dates=["te"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 11.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.35, 1.0, 1.0], hspace=0.28))
    fig.patch.set_facecolor(SURF)
    W = PRE["windows"]
    for ax in axes:
        ax.set_facecolor(SURF)
        for wk in ("W1", "W1b"):
            ax.axvspan(pd.Timestamp(W[wk]["start"]), pd.Timestamp(W[wk]["end"]),
                       color="#f0efec", zorder=0, lw=0)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.grid(axis="x", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)

    def endlabels(ax, items, gap_frac=0.06):
        """线尾直接标注,按 y 排序后强制最小间距,避免文字重叠(只移动文字,不移动数据点)。"""
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * gap_frac
        items = sorted(items, key=lambda t: t[0].iloc[-1])
        ys = [float(t[0].iloc[-1]) for t in items]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for (s, txt), y in zip(items, ys):
            ax.annotate(txt, (s.index[-1], s.iloc[-1]), xytext=(s.index[-1] + pd.Timedelta(days=2), y),
                        textcoords="data", va="center", fontsize=8.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))

    # --- A 累计超额
    ax = axes[0]
    items = []
    for col in ["COMP_EXCL", "COMP_TOP20", "COMP_TOP50", "BGBT_TOP20", "CSI1000"]:
        s = cum_excess(col, "POOL_C2C" if col == "CSI1000" else "POOL_EW")
        ax.plot(s.index, s.values, color=COL[col], lw=2, label=NAME[col], zorder=3)
        items.append((s, f"{NAME[col]} {s.iloc[-1]:+.1f}%"))
    endlabels(ax, items)
    ax.set_title("A. 相对池等权的累计超额(%,W1 开始前 = 0;V1000 open10,含成本;中证1000 对同钟收盘口径池等权)",
                 loc="left", fontsize=10.5, color=INK)
    ymax = ax.get_ylim()[1]
    for wk, lab in (("W1", "W1 小微盘流动性危机"), ("W1b", "W1b 反弹")):
        ax.text(pd.Timestamp(W[wk]["start"]) + pd.Timedelta(days=1), ymax * 0.92, lab,
                fontsize=9, color=INK2, va="top")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2)

    # --- B 持仓中全 A 市值最小 20% 的占比
    ax = axes[1]
    for col in ["COMP_EXCL", "COMP_TOP20", "COMP_TOP50", "BGBT_TOP20"]:
        d = path[path.object == col].set_index("te")["share_mvq1_allA"] * 100
        d = d[(d.index >= a0) & (d.index <= a1)]
        ax.plot(d.index, d.values, color=COL[col], lw=2, label=NAME[col], zorder=3)
    d = path[path.object == "POOL_EW"].set_index("te")["share_mvq1_allA"] * 100
    d = d[(d.index >= a0) & (d.index <= a1)]
    ax.plot(d.index, d.values, color=INK2, lw=1.5, ls=(0, (4, 3)), label="池等权(参照)", zorder=2)
    ax.set_title("B. 已投资持仓中『全 A 市值最小 20%』股票的权重(%)", loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower left", bbox_to_anchor=(0.52, 0.36), fontsize=8.5, frameon=False, ncol=1)
    ax.set_ylim(-3, 105)

    # --- C 组合层市值中性(NS,事后追加)反事实
    ax = axes[2]
    items = []
    for col in ["COMP_TOP20", "BGBT_TOP20"]:
        s0 = cum_excess(col)
        s1 = cum_excess(col + "|NS")
        ax.plot(s0.index, s0.values, color=COL[col], lw=2, label=f"{NAME[col]} 原始", zorder=3)
        ax.plot(s1.index, s1.values, color=COL[col], lw=2, ls=(0, (5, 3)),
                label=f"{NAME[col]} 市值五分位分层选股(NS)", zorder=3)
        short = "Top20" if col == "COMP_TOP20" else "B_gbt"
        items += [(s0, f"{short} 原始 {s0.iloc[-1]:+.1f}%"), (s1, f"{short} NS {s1.iloc[-1]:+.1f}%")]
    endlabels(ax, items)
    ax.set_title("C. 反事实:组合层市值中性(池内市值五分位分层选股,事后追加)的累计超额(%)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02), fontsize=8, frameon=False, ncol=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(a0, a1 + pd.Timedelta(days=30))
    fig.text(0.01, 0.005, "注:横轴为成交日 te。COMP 系列为信号样本内(成员在 2020-2025 选出);B_gbt 为 walk-forward 样本外;"
             "数据缺已退市股(幸存者偏差)。数据:research/ledger/stress_holdings_path.csv、cache/_tmp/stress_rets.parquet",
             fontsize=7.5, color=INK2)
    out = REPORTS / "stress_2024q1.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURF)
    # 图中线尾数字落盘,保证报告里引用的每个数都能在 ledger 找到
    tail = {c: float(cum_excess(c).iloc[-1]) for c in
            ["COMP_EXCL", "COMP_TOP20", "COMP_TOP50", "BGBT_TOP20", "COMP_TOP20|NS", "COMP_TOP50|NS", "BGBT_TOP20|NS"]}
    # 中证1000 与同钟池等权(收盘口径)比;旧版与 open10 池等权比的数值一并保留,供审查记录对照
    tail["CSI1000_vs_POOL_C2C"] = float(cum_excess("CSI1000", "POOL_C2C").iloc[-1])
    tail["CSI1000_vs_POOL_EW_open10(旧,钟点错位)"] = float(cum_excess("CSI1000", "POOL_EW").iloc[-1])
    pd.Series(tail, name="cum_excess_pct_2024-03-29_since_W1").to_csv(LEDGER / "stress_fig_tail_values.csv")
    log("图已写", out)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("engine", "all"):
        stage_engine()
    if stage in ("dims", "all"):
        stage_dims()
    if stage in ("analyze", "all"):
        from stress_analyze import stage_analyze   # noqa
        stage_analyze()
    if stage in ("figure", "all"):
        stage_figure()
