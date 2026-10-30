"""压力测试分析阶段(由 scripts/run_stress_test.py analyze 调用,前缀 stress)。

输入:cache/_tmp/stress_{rets,dims,H_*,diag_*,X_*}.parquet(engine / dims 阶段产物)、
      research/ledger/barra_factor_returns.csv(open10 纯因子收益,信号日索引)。
输出(research/ledger/):
  stress_windows.csv              每个对象 x 窗口:收益、超额、回撤、恢复天数、sigma 数
  stress_attribution.csv          事后归因长表(现金腿/风格/行业/特质/成本)+ 窗口内纯因子收益与冲击 z
  stress_specific_breakdown.csv   特质项按 持仓股票的 全A市值五分位 / 过去20日收益五分位 / ST 拆开
  stress_exante_vs_realized.csv   事前情景预测 vs 实际、事前风险模型 z、历史最差 20 日
  stress_liquidity.csv            窗口前持仓的 ADV/市值分布、窗口内一字跌停锁仓、参与率截断、换手
  stress_holdings_path.csv        2023-11 ~ 2024-03 逐日持仓特征(微盘占比、过去20日收益分位等)
  stress_counterfactual.csv       市值中性化反事实(N1/N3 预注册,N1r/NS 事后追加)
  stress_barra_clip.csv           Barra Size/NonlinSize 截尾诊断(窗口前一日池内被截在下限的比例)
  —— 审查修复(2026-10-10)后新增 ——
  stress_specific_matched.csv     特质项"档位效应 vs 档内选股"匹配拆分(分桶加总是同义反复,见 matched_split)
  stress_participation_sensitivity.csv  参与率按初始资金 vs 按净值
  stress_survivorship.json        退市股缺失统计(此前不可复现,现由 write_survivorship 生成)
  stress_csi_alignment.csv        池等权 vs 中证1000 的半天钟点错位量化(csi_alignment_table)

## 归因口径(为什么这样拆)

主动收益 = 现金腿 + Σ风格 + Σ行业 + 特质 − 成本,其中
    现金腿 = (ΣH − 1) × 池等权收益        参与率上限没买进的部分留现金 = 隐性的"未满仓/低 Beta"头寸:
                                          跌时保护、急涨时拖累(线性,无凸性,不是"做多波动率")
    主动权重 A' = H − ΣH × B               只看**已投资部分**相对池等权的倾斜,ΣA' = 0 → 国家项为 0
第一版把现金也放进 A = H − B,结果现金在 Barra 框架里表现为"Size 暴露 0 = 市值加权平均 = 大盘",
B_gbt Top20 窗口前 29% 现金、94% 持仓在全 A 最小 20% 市值,Size 主动暴露却是 +0.30 —— 不可解读。
现金腿单列后,风格暴露只描述"买了什么",现金的作用单独可见。

其他容易错的地方:
  * 基准 = 池等权(与 ST.benchmark 同一组股票:信号日池内且 te/te+1 有价),不是 Barra 的 sqrt 市值加权。
  * 行业纯因子收益用"市值加权和为零"识别,最后一个行业 = 对其余行业因子 -cap_j/cap_last 的暴露,
    归因与逐股残差都按同一映射还原。申万行业为 2026 快照(轻微前视),与 run_barra_attribution 同口径。
  * **Barra Size 按 ±3σ 截尾、以市值加权均值为中心**:2023-12-28 池内 37.6% 的股票 Size 暴露都等于 -3,
    线性 Size 看不见"小盘 vs 微盘"。所以本模块另用**全 A 市值五分位**与**池内市值秩**描述持仓,
    特质项也按市值五分位拆开 —— 截尾之下的微盘效应会落进特质项。
  * 事前情景只用 t_s-1 的成交后持仓与 t_s 的暴露;协方差/特质方差只用到 t_s-2(决策时点已实现)。
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from afactor.config import LEDGER, RESEARCH_END
from afactor.panel import Panel
from afactor import strategy as ST, factors as FA, newdata as ND
from afactor.strategy import TRADING_DAYS
from run_stress_test import (WINDOWS, PLANS, STYLE_KEYS, SHORT, DIM_REPS, FULL, DEV, TEST,
                             MODE, SAMPLE_FLAG, VARIANTS, tmp, log, nw_t, te_map, win_signal_days,
                             load_exposures, pinned_down)

N_HIST = 250          # 事前风险模型回看天数
MIN_HIST = 60
PATH_RANGE = ("2023-11-01", "2024-03-29")
VAR_KIND = {"": "策略", "|N1": "反事实:Barra Size 中性(预注册)",
            "|N3": "反事实:Size+NonlinSize+Liquidity 中性(预注册敏感性)",
            "|N1r": "反事实:池内市值秩中性(事后追加)",
            "|NS": "反事实:池内市值五分位分层选股=组合层市值中性(事后追加)"}


# ====================================================================== 窗口统计
def _nav(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()


def window_stats(rs: pd.Series, rb: pd.Series | None, idx: pd.DatetimeIndex, sd_daily: float) -> dict:
    """窗口内:复利收益、超额、窗口内最大回撤(绝对/相对)、从谷底回到窗口起点水平的交易日数。"""
    s = rs.reindex(idx).fillna(0.0)
    b = rb.reindex(idx).fillna(0.0) if rb is not None else s * 0.0
    n = len(idx)
    ret, bret = float((1 + s).prod() - 1), float((1 + b).prod() - 1)
    out = dict(n_days=n, ret=ret, bench_ret=bret, excess=ret - bret, excess_sum=float((s - b).sum()))
    out["excess_sigma"] = out["excess_sum"] / (sd_daily * np.sqrt(n)) if sd_daily > 0 else np.nan
    for tag, path in (("", (1 + s).cumprod()), ("_rel", (1 + s).cumprod() / (1 + b).cumprod())):
        ext = np.r_[1.0, path.to_numpy()]
        out[f"max_dd{tag}"] = float((ext / np.maximum.accumulate(ext) - 1).min())
    full_s = rs.dropna()
    full_b = rb.reindex(full_s.index).fillna(0.0) if rb is not None else full_s * 0.0
    for tag, navF in (("", _nav(full_s)), ("_rel", _nav(full_s) / _nav(full_b))):
        pos = navF.index.get_indexer(idx)
        pos = pos[pos >= 0]
        start = float(navF.iloc[pos[0] - 1]) if pos[0] > 0 else 1.0
        wv = navF.iloc[pos].to_numpy()
        tp = int(pos[np.argmin(wv)])
        out[f"trough_t{tag}"] = navF.index[tp].date()
        if navF.iloc[tp] >= start:
            out[f"recovery_days{tag}"], out[f"recovered{tag}"] = 0, True
            continue
        after = np.flatnonzero(navF.iloc[tp + 1:].to_numpy() >= start)
        if after.size:
            out[f"recovery_days{tag}"], out[f"recovered{tag}"] = int(after[0] + 1), True
        else:
            out[f"recovery_days{tag}"], out[f"recovered{tag}"] = np.nan, False
    return out


def worst_windows(e: pd.Series, k: int = 3, L: int = 20) -> list[tuple[int, int]]:
    """互不重叠的最差 k 个 L 日窗口(滚动和)。返回 [(起点位置, 终点位置)]。"""
    roll = e.rolling(L).sum().to_numpy()
    order = np.argsort(np.where(np.isfinite(roll), roll, np.inf))
    picked = []
    for j in order:
        if not np.isfinite(roll[j]):
            break
        if all(abs(j - q) >= L for q in picked):
            picked.append(int(j))
        if len(picked) == k:
            break
    return [(j - L + 1, j) for j in picked]


# ====================================================================== Barra 帧
class BarraFrame:
    """把 Barra 纯因子收益、暴露、行业映射、逐股残差整理成逐日可乘的数组。"""

    def __init__(self, p: Panel):
        self.p = p
        self.dates, self.codes = p.dates, p.codes
        T, N = len(self.dates), len(self.codes)
        X = load_exposures(p)
        self.X = {k: X[k].reindex(index=self.dates, columns=self.codes).to_numpy(np.float32) for k in STYLE_KEYS}
        allfin = np.ones((T, N), bool)
        for k in STYLE_KEYS:
            allfin &= np.isfinite(self.X[k])
        self.allfin = allfin
        self.Xz = {k: np.nan_to_num(v) for k, v in self.X.items()}
        u = p["universe"].reindex(index=self.dates, columns=self.codes).fillna(False).to_numpy(bool)
        self.u = u
        px, _ = ST.exec_prices(p, MODE)
        g = (px.shift(-1) / px - 1.0).shift(-ST.EXEC_MODES[MODE]["lag"])
        self.R = g.to_numpy(np.float64)                     # 信号日 t 的已实现收益
        self.r_eng = np.nan_to_num(self.R)                  # 引擎口径:无价记 0
        mv = FA._real(p, "mv").reindex(index=self.dates, columns=self.codes).to_numpy(np.float64)
        self.mv = mv
        L = ND.sw_industry_labels(self.dates, self.codes)
        lab = L.iloc[0].to_numpy().astype(int)              # 快照:各日相同
        inds = sorted({int(c) for c in pd.unique(L.where(p["universe"]).values.ravel())
                       if pd.notna(c) and int(c) >= 0})
        names = ND.sw_industry_names()
        self.inds, self.last, self.lab = inds, inds[-1], lab
        self.ind_cols = [f"行业:{names.get(c, c)}" for c in inds[:-1]]
        OH = np.zeros((N, len(inds)))
        for j, c in enumerate(inds):
            OH[lab == c, j] = 1.0
        self.OH = OH
        base = u & np.isfinite(mv) & (mv > 0) & allfin
        self.capsum_post = np.where(base & np.isfinite(self.R), mv, 0.0) @ OH
        self.capsum_pre = np.where(base, mv, 0.0) @ OH
        bm = u & np.isfinite(self.R)
        self.B = bm / np.maximum(bm.sum(1, keepdims=True), 1)
        fr = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0)
        F = fr[MODE].copy()
        F.index = pd.to_datetime(F.index)
        assert F.index.max() <= pd.Timestamp("2025-12-31")
        assert [c for c in F.columns if c.startswith("行业:")] == self.ind_cols, "行业因子列与重建编码不一致"
        assert [c for c in F.columns if not c.startswith("行业:") and c != "Country"] == STYLE_KEYS
        self.F = F.reindex(self.dates).astype(float)
        self.fcols = ["Country"] + STYLE_KEYS + self.ind_cols
        self.xcols = STYLE_KEYS + self.ind_cols           # 现金腿口径下的暴露列(国家项恒为 0)

    def ind_expo(self, A: np.ndarray, capsum: np.ndarray) -> np.ndarray:
        """权重 A (T x N 或 1 x N) -> 行业纯因子暴露(含最后一个行业的 -cap_j/cap_last 映射)。"""
        w = A @ self.OH
        cl = capsum[..., -1:]
        ratio = np.divide(capsum[..., :-1], cl, out=np.zeros_like(capsum[..., :-1]), where=cl > 0)
        return w[..., :-1] - w[..., -1:] * ratio

    def exposures(self, A: np.ndarray, capsum: np.ndarray) -> pd.DataFrame:
        e = {"Country": A.sum(1)}
        for k in STYLE_KEYS:
            e[k] = (A * self.Xz[k]).sum(1)
        E = pd.DataFrame(e, index=self.dates)
        E[self.ind_cols] = self.ind_expo(A, capsum)
        return E

    def residuals(self, rows: np.ndarray) -> np.ndarray:
        """指定信号日行的逐股 Barra 残差 u = r - (国家 + 风格 + 行业)。暴露缺失按 0(与归因同口径)。"""
        F = self.F.iloc[rows]
        fit = np.repeat(F["Country"].to_numpy()[:, None], len(self.codes), 1)
        for k in STYLE_KEYS:
            fit = fit + self.Xz[k][rows] * F[k].to_numpy()[:, None]
        Fi = F[self.ind_cols].to_numpy()                       # T' x (J-1)
        jmap = {c: j for j, c in enumerate(self.inds[:-1])}
        col = np.array([jmap.get(int(c), -1) for c in self.lab])
        reg = col >= 0
        fit[:, reg] += Fi[:, col[reg]]
        cs = self.capsum_post[rows]
        ratio = np.divide(cs[:, :-1], cs[:, -1:], out=np.zeros_like(cs[:, :-1]), where=cs[:, -1:] > 0)
        lastv = -(ratio * Fi).sum(1)
        fit[:, self.lab == self.last] += lastv[:, None]
        return self.r_eng[rows] - fit


def attribution_daily(bf: BarraFrame, H: np.ndarray, rs: pd.Series, bench: pd.Series,
                      diag: pd.DataFrame) -> pd.DataFrame:
    """逐日事后归因(现金腿单列):主动 = 现金腿 + 风格 + 行业 + 特质 - 成本。
    暴露列 e:* 为**已投资部分归一化**后的主动暴露(h/ΣH - B),贡献列 c:* 为精确贡献。"""
    Hd = H.astype(np.float64)
    inv = Hd.sum(1)
    A = Hd - inv[:, None] * bf.B
    E = bf.exposures(A, bf.capsum_post)
    assert float(E["Country"].abs().max()) < 1e-6
    C = E[bf.xcols] * bf.F[bf.xcols]
    En = E.div(pd.Series(np.where(inv > 1e-9, inv, np.nan), index=bf.dates), axis=0)
    idx = rs.index
    out = pd.DataFrame(index=idx)
    for k in bf.xcols:
        out[f"e:{k}"] = En[k].reindex(idx)
        out[f"c:{k}"] = C[k].reindex(idx)
    out["invested"] = pd.Series(inv, index=bf.dates).reindex(idx)
    out["active"] = rs - bench.reindex(idx)
    out["cost"] = diag["cost"].reindex(idx)
    out["c:cash"] = (out["invested"] - 1.0) * bench.reindex(idx)
    out["c:style_total"] = out[[f"c:{k}" for k in STYLE_KEYS]].sum(1, min_count=1)
    out["c:industry_total"] = out[[f"c:{k}" for k in bf.ind_cols]].sum(1, min_count=1)
    out["specific"] = out["active"] + out["cost"] - out["c:cash"] - out["c:style_total"] - out["c:industry_total"]
    noexp = (H > 1e-10) & ~bf.allfin
    out["w_noexpo"] = pd.Series(np.where(noexp, Hd, 0).sum(1), index=bf.dates).reindex(idx)
    return out


def attribution_pool_abs(bf: BarraFrame, bench: pd.Series) -> pd.DataFrame:
    """池等权基准自身的**绝对**收益拆解(国家暴露 = 1)。"""
    E = bf.exposures(bf.B, bf.capsum_post)
    C = E[bf.fcols] * bf.F[bf.fcols]
    idx = bench.index
    out = pd.DataFrame(index=idx)
    for k in bf.fcols:
        out[f"e:{k}"] = E[k].reindex(idx)
        out[f"c:{k}"] = C[k].reindex(idx)
    out["active"] = bench
    out["cost"] = 0.0
    out["c:style_total"] = out[[f"c:{k}" for k in STYLE_KEYS]].sum(1, min_count=1)
    out["c:industry_total"] = out[[f"c:{k}" for k in bf.ind_cols]].sum(1, min_count=1)
    out["specific"] = out["active"] - out[[f"c:{k}" for k in bf.fcols]].sum(1, min_count=1)
    return out


def _bucket_means(key: np.ndarray, val: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """单日:ref(池内且残差有限)股票按 key 分桶求等权均值,再映射回所有 key 有限的股票;
    桶内没有池内股票 / key 缺失的股票返回 NaN。"""
    out = np.full(key.shape, np.nan)
    fin = np.isfinite(key)
    m = fin & ref & np.isfinite(val)
    if not m.any():
        return out
    ug, inv = np.unique(key[m], return_inverse=True)
    means = np.bincount(inv, weights=val[m]) / np.bincount(inv)
    kf = key[fin]
    pos = np.clip(np.searchsorted(ug, kf), 0, len(ug) - 1)
    hit = ug[pos] == kf
    tmp_ = np.full(kf.shape, np.nan)
    tmp_[hit] = means[pos[hit]]
    out[fin] = tmp_
    return out


def matched_split(bf: BarraFrame, Hs: dict, att: dict, objs_: list, sig_index, te, mv_pct_all, mvq, p20q):
    """审查补充(2026-10-10):特质项的"档位效应 vs 档内选股"匹配拆分 -> stress_specific_matched.csv。

    为什么要做:原 §3.3 把特质贡献按持仓所在的桶加总("B_gbt 的 -17.13pp 来自全 A 最小 20%")。
    B_gbt 99.8% 的权重就在那一档,任何亏损都会落在那里 —— 这是分桶的同义反复,不能说明亏损是
    "微盘这一档整体的额外下跌(Barra 截尾盲区)"还是"在微盘里选错了股"。

    拆法:对每只持仓,期望残差 m_i = 当日池内同桶股票的等权平均残差。
      特质合计   = Σ_t Σ_i (H_i - inv·B_i) · u_i                (与归因的 specific 完全一致)
      档位效应   = Σ_t [Σ_i H_i · m_i - inv · Σ_i B_i · u_i]   持仓的桶分布 x 各桶池内平均残差,减池平均
      档内选股   = Σ_t Σ_i H_i · (u_i - m_i)                   同桶里选到的股票相对桶平均的残差
    两者之和 = 特质合计(recon_err 列核对)。桶越细,档位效应越接近"纯市值档"能解释的上限。
    无桶 / 桶内无池内股票的持仓(权重见 unmatched_weight_mean)全部记入档位效应、选股为 0
    —— 对"档位效应能解释多少"偏宽松,即对原结论有利的方向。"""
    keyfns = {
        ("全A市值", 5): lambda r: mvq[r],
        ("全A市值", 50): lambda r: np.ceil(mv_pct_all[r] * 50).clip(1, 50),
        ("全A市值", 100): lambda r: np.ceil(mv_pct_all[r] * 100).clip(1, 100),
        ("池内过去20日收益", 5): lambda r: p20q[r],
        ("过去20日收益5档x全A市值5档", 25): lambda r: p20q[r] * 10 + mvq[r],
    }
    dates = bf.dates
    rows = []
    for w, (d0, d1, desc) in WINDOWS.items():
        idx = win_signal_days(sig_index, te, d0, d1)
        idx = idx.intersection(att["POOL_EW"].dropna(subset=["c:style_total"]).index)
        if len(idx) < 2:
            continue
        r_ = np.array([dates.get_loc(t) for t in idx])
        U_ = bf.residuals(r_)
        Bw = bf.B[r_]
        inpool = Bw > 0
        Uz = np.nan_to_num(U_)
        pool_sum = float((Bw * Uz).sum())
        for (mname, nb), kf in keyfns.items():
            key = kf(r_)
            M = np.vstack([_bucket_means(key[j], U_[j], inpool[j]) for j in range(len(r_))])
            Mz = np.where(np.isfinite(M), M, Uz)
            if nb == 5:      # 池内各桶的等权残差之和(说明"这一档整体跌了多少")
                for q in range(1, 6):
                    mq = (key == q) & inpool
                    v = float((np.where(mq, Uz, 0).sum(1) / np.maximum(mq.sum(1), 1)).sum())
                    rows.append(dict(kind="池内分桶等权残差之和", object="POOL_EW", window=w, matching=mname,
                                     n_bins=nb, bucket=f"Q{q}", value=v, pool_ew_resid_sum=pool_sum))
            for o in objs_:
                H = Hs[o][r_].astype(np.float64)
                inv = H.sum(1, keepdims=True)
                if float(inv.min()) < 1e-6:          # 窗口内未建仓(B_gbt 在 2022)
                    continue
                hn = H / inv
                total = float(((H - inv * Bw) * Uz).sum())
                bucket = float((H * Mz).sum() - (inv * Bw * Uz).sum())
                sel = float((H * (Uz - Mz)).sum())
                spec_attr = float(att[o]["specific"].reindex(idx).sum())
                rows.append(dict(kind="匹配拆分", object=o, window=w, matching=mname, n_bins=nb,
                                 specific_total=total, bucket_part=bucket, selection_part=sel,
                                 bucket_share=bucket / total if total != 0 else np.nan,
                                 recon_err=bucket + sel - spec_attr,
                                 unmatched_weight_mean=float(np.where(~np.isfinite(M), hn, 0).sum(1).mean()),
                                 port_invested_resid_sum=float((hn * Uz).sum()),
                                 port_bucketmix_resid_sum=float((hn * Mz).sum()),
                                 pool_ew_resid_sum=pool_sum))
    D = pd.DataFrame(rows)
    chk = D.loc[D.kind == "匹配拆分", "recon_err"].abs().max()
    assert chk < 1e-6, f"匹配拆分不自洽 {chk}"
    D.to_csv(LEDGER / "stress_specific_matched.csv", index=False)


def csi_alignment_table(rets: pd.DataFrame, te) -> None:
    """审查补充:池等权 vs 中证1000 的半天钟点错位量化 -> stress_csi_alignment.csv。
    每个窗口给出:open10 池等权、同钟收盘口径池等权、中证1000 两种对齐的复利收益与差距;
    offset_artifact = open10 池等权 - 同钟池等权(同一个池、同一窗口,只差时钟)。
    另列 2021 年起错位最大的 5 个单日,以及原版事后第二差窗口 2024-10-08 ~ 11-04。"""
    b, c2c, csi, csi1 = rets["POOL_EW"], rets["POOL_C2C"], rets["CSI1000"], rets["CSI1000_TE1"]
    wins = {k: (v[0], v[1]) for k, v in WINDOWS.items()}
    wins["W1+W1b"] = (WINDOWS["W1"][0], WINDOWS["W1b"][1])
    wins["原版事后第二差 2024-10-08~11-04"] = (pd.Timestamp("2024-10-08"), pd.Timestamp("2024-11-04"))
    base = b.dropna().index
    rows = []
    for w, (d0, d1) in wins.items():
        idx = win_signal_days(base, te, d0, d1)

        def f(s):
            return float((1 + s.reindex(idx).fillna(0.0)).prod() - 1)
        po, pc, ci, ci1 = f(b), f(c2c), f(csi), f(csi1)
        g = (b - csi).reindex(idx)
        rows.append(dict(kind="窗口", window=w, te_start=te[idx[0]].date(), te_end=te[idx[-1]].date(),
                         n_days=len(idx), pool_open10=po, pool_c2c_sameclock=pc, csi_te=ci, csi_te1=ci1,
                         gap_sameclock=pc - ci, gap_open10_vs_csi_te=po - ci, gap_open10_vs_csi_te1=po - ci1,
                         offset_artifact=po - pc,
                         sum_daily_gap_sameclock=float((c2c - csi).reindex(idx).sum()),
                         sum_daily_gap_open10_vs_csi_te=float(g.sum()),
                         first_day_gap_open10_vs_csi_te=float(g.iloc[0])))
    d = (b - c2c).loc["2021-01-01":].dropna()
    for t in d.abs().sort_values(ascending=False).index[:5]:
        rows.append(dict(kind="错位最大单日", window=str(te[t].date()), te_start=te[t].date(), te_end=te[t].date(),
                         n_days=1, pool_open10=float(b[t]), pool_c2c_sameclock=float(c2c[t]),
                         csi_te=float(csi[t]), csi_te1=float(csi1[t]),
                         gap_sameclock=float(c2c[t] - csi[t]), gap_open10_vs_csi_te=float(b[t] - csi[t]),
                         gap_open10_vs_csi_te1=float(b[t] - csi1[t]), offset_artifact=float(b[t] - c2c[t])))
    pd.DataFrame(rows).to_csv(LEDGER / "stress_csi_alignment.csv", index=False)


def write_survivorship(p: Panel):
    """幸存者偏差统计 -> stress_survivorship.json(审查补充:此前是临时脚本算的,不可复现)。
    封存:先把 stock_basic 截到退市日 <= RESEARCH_END(2026 及以后的退市记录一律不读入计算)。"""
    sb = pd.read_csv(ND.DATA / "stock_basic.csv")
    sb["dl"] = pd.to_datetime(sb["delist_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    sb["ld"] = pd.to_datetime(sb["list_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    sb = sb[sb["dl"].notna() & (sb["dl"] <= pd.Timestamp(RESEARCH_END))]      # 封存截断在前
    d = sb[sb["dl"] >= pd.Timestamp("2020-01-01")]
    inpanel = d["ts_code"].isin(p.codes)
    miss = d[~inpanel]
    out = {"source": f"data_8:21/stock_basic.csv(只用 delist_date <= {RESEARCH_END} 的记录)",
           "generated_by": "scripts/run_stress_test.py analyze -> stress_analyze.write_survivorship",
           "delisted_2020_2025_total": int(len(d)),
           "delisted_2020_2025_in_panel": int(inpanel.sum()),
           "delisted_2020_2025_missing": int((~inpanel).sum())}
    for y in range(2020, 2026):
        out[f"missing_delisted_{y}"] = int((miss["dl"].dt.year == y).sum())
    for w, (d0, d1, _) in WINDOWS.items():
        out[f"missing_listed_through_{w}"] = int(((miss["ld"] <= d0) & (miss["dl"] > d1)).sum())
    with open(LEDGER / "stress_survivorship.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def participation_sensitivity(rets: pd.DataFrame, diags: dict, base_objs: list, te, full0) -> None:
    """审查补充:参与率上限按初始资金(默认)vs 按 capital x 当前净值 -> stress_participation_sensitivity.csv。
    默认口径不随净值增长,净值 > 1 时低估真实参与率、低估现金腿;这里只作敏感性,不改主口径。"""
    navr, navc = pd.read_parquet(tmp("rets_navcap")), pd.read_parquet(tmp("cash_navcap"))
    bench = rets["POOL_EW"]
    out = []
    for o in base_objs:
        for conv, r, cash in (("初始资金(默认)", rets[o].dropna(), 1.0 - diags[o]["exposure"]),
                              ("按净值(participation_on_nav)", navr[o].dropna(), navc[o])):
            row = dict(object=o, convention=conv)
            rf = r.loc[full0:]
            bf_ = bench.reindex(rf.index)
            n = len(rf)
            row["ann_excess_full"] = float((1 + rf).prod() ** (TRADING_DAYS / n) - (1 + bf_).prod() ** (TRADING_DAYS / n))
            row["full_cash_mean"] = float(cash.reindex(rf.index).mean())
            for w in ("W1", "W1b", "W3"):
                d0, d1, _ = WINDOWS[w]
                idx = win_signal_days(r.index, te, d0, d1)
                row[f"{w}_excess"] = float((1 + r.reindex(idx)).prod() - (1 + bench.reindex(idx)).prod())
                row[f"{w}_cash_mean"] = float(cash.reindex(idx).mean())
                pos = r.index.get_loc(idx[0])
                row[f"{w}_start_nav"] = float((1 + r.iloc[:pos]).prod())
            out.append(row)
    pd.DataFrame(out).to_csv(LEDGER / "stress_participation_sensitivity.csv", index=False)


# ====================================================================== 主流程
def stage_analyze():
    t_all = time.time()
    p = Panel()
    dates = p.dates
    T = len(dates)
    te = te_map(dates)
    rets = pd.read_parquet(tmp("rets"))
    dims = pd.read_parquet(tmp("dims"))
    bench, csi = rets["POOL_EW"], rets["CSI1000"]
    bf = BarraFrame(p)
    log("Barra 帧就绪")
    full0 = pd.Timestamp(FULL[0])

    # ---- 描述持仓用的分桶(全部只用信号日 t 及之前的信息)
    valid = p["valid"].reindex(index=dates, columns=p.codes).fillna(False).to_numpy(bool)
    mvdf = pd.DataFrame(bf.mv, index=dates, columns=p.codes)
    mv_pct_all = mvdf.where(valid & np.isfinite(bf.mv)).rank(axis=1, pct=True).to_numpy()
    mvq = np.ceil(mv_pct_all * 5).clip(1, 5)                       # 全 A 市值五分位,1=最小
    mv_rank_u = mvdf.where(bf.u).rank(axis=1, pct=True).to_numpy()  # 池内市值秩
    close = p["close_px"]
    past20 = (close / close.shift(20) - 1.0).where(bf.u)
    p20_pct = past20.rank(axis=1, pct=True).to_numpy()
    p20q = np.ceil(p20_pct * 5).clip(1, 5)                         # 池内过去 20 日收益五分位,1=跌最多
    st = p["is_st"].reindex(index=dates, columns=p.codes).fillna(False).to_numpy(bool)
    turn_rt = FA._real(p, "turn").reindex(index=dates, columns=p.codes).to_numpy(np.float64)
    log("分桶就绪")

    csi_alt, pool_c2c = rets["CSI1000_TE1"], rets["POOL_C2C"]
    strat_objs = [c for c in rets.columns if c not in ("POOL_EW", "CSI1000", "CSI1000_TE1", "POOL_C2C")]
    base_objs = [c for c in strat_objs if "|" not in c]
    # 审查修正(2026-10-10):池等权 open10 与中证1000 收盘口径差半天(见 run_stress_test.csi1000_aligned)。
    #   POOL_C2C            同钟:池等权 te 收盘对收盘 vs 中证1000 te 收盘对收盘 —— "池 vs 指数"的主口径
    #   POOL_EW             open10 池等权 vs 中证1000(te 对齐,预注册)—— 钟点错位,只作区间一端
    #   POOL_EW|CSI_te+1    open10 池等权 vs 中证1000(te+1 对齐)—— 钟点错位,区间另一端
    #   CSI1000             中证1000 vs 同钟池等权(原先对 open10 池等权,已改)
    # 钟点错位的对象:日超额被单日跳空噪声主导(sd 翻倍、一阶自相关 -0.46),sigma 数不可解读,置 NaN;
    # 也不在其上找"事后最差窗口"(会被单日错位挑中)。
    MISALIGNED = {"POOL_EW", "POOL_EW|CSI_te+1"}
    objs = {"POOL_EW": (bench, csi, "基准 open10(对照=中证1000 te 收盘,预注册;钟点错位半天)", "—"),
            "POOL_EW|CSI_te+1": (bench, csi_alt, "基准 open10(对照=中证1000 te+1 收盘,敏感性;钟点错位半天)", "—"),
            "POOL_C2C": (pool_c2c, csi, "池等权同钟收盘口径(对照=中证1000 te 收盘,同钟)", "—"),
            "CSI1000": (csi, pool_c2c, "指数(对照=同钟池等权收盘口径)", "真实指数")}
    for o in strat_objs:
        var = o[o.index("|"):] if "|" in o else ""
        objs[o] = (rets[o].dropna(), bench, VAR_KIND[var], SAMPLE_FLAG["BGBT" if o.startswith("BGBT") else "COMP"])
    for n in DIM_REPS:
        objs[f"DIM_LS:{n}"] = (dims[f"LS:{n}"], None, "维度多空 D10-D1(理论、无成本)", "信号样本内")
        objs[f"DIM_EX20:{n}"] = (dims[f"EX20:{n}"], dims[f"EX20BM:{n}"], "维度排除最差20%(理论、无成本)", "信号样本内")
        objs[f"DIM_D1:{n}"] = (dims[f"D1EX:{n}"], None, "维度最差十分位相对池(理论)", "信号样本内")

    # ---------- 1. 窗口表
    # 派生窗口(审查补充):W1 起点到 W1b 终点,回答"W1 的相对亏损在 W1b 里收复了没有"。
    # 不同窗口的 pp 超额基数不同(复利),不能相减判断是否收复,要看相对净值 rel_nav_change。
    DERIVED = {"W1+W1b": (WINDOWS["W1"][0], WINDOWS["W1b"][1], "派生:W1 起点到 W1b 终点(W1 亏损是否在 W1b 收复)")}
    rows, win_index = [], {}
    for o, (rs, rb, kind, flag) in objs.items():
        rs = rs.dropna()
        e = rs - (rb.reindex(rs.index).fillna(0.0) if rb is not None else 0.0)
        e = e[e.index >= full0]
        sd = float(e.std())
        wins = []
        for w, (d0, d1, desc) in WINDOWS.items():
            idx = win_signal_days(rs.index, te, d0, d1)
            if len(idx) >= 2:
                wins.append((w, "预注册", desc, idx))
        if o not in MISALIGNED:
            for i, (a, b) in enumerate(worst_windows(e)):
                wins.append((f"H{i + 1}", "事后描述", "超额最差20日", e.index[a:b + 1]))
        win_index[o] = wins
        der = []
        for w, (d0, d1, desc) in DERIVED.items():
            idx = win_signal_days(rs.index, te, d0, d1)
            if len(idx) >= 2:
                der.append((w, "派生(审查补充)", desc, idx))
        for w, wtype, desc, idx in wins + der:
            st_ = window_stats(rs, rb, idx, sd)
            if o in MISALIGNED:
                st_["excess_sigma"] = np.nan
            rows.append(dict(object=o, kind=kind, sample=flag, window=w, window_type=wtype, desc=desc,
                             te_start=te[idx[0]].date(), te_end=te[idx[-1]].date(),
                             sd_daily_excess_full=sd, clock_misaligned=o in MISALIGNED,
                             rel_nav_change=(1 + st_["ret"]) / (1 + st_["bench_ret"]) - 1, **st_))
    pd.DataFrame(rows).to_csv(LEDGER / "stress_windows.csv", index=False)
    log("stress_windows.csv", len(rows))

    # ---------- 2. 逐日归因
    att = {"POOL_EW": attribution_pool_abs(bf, bench)}
    Hs, diags = {}, {}
    for o in strat_objs:
        H = pd.read_parquet(tmp(f"H_{o}")).reindex(index=dates, columns=p.codes).to_numpy(np.float32)
        dg = pd.read_parquet(tmp(f"diag_{o}"))
        Hs[o], diags[o] = H, dg
        rs = rets[o].dropna()
        att[o] = attribution_daily(bf, H, rs, bench, dg)
        gross = pd.Series((H.astype(np.float64) * bf.r_eng).sum(1), index=dates).reindex(rs.index)
        err = float((gross - dg["cost"].reindex(rs.index) - rs).abs().max())
        # 持仓以 float32 落盘,复原误差 ~1e-8/日,远小于任何被报告的数字;误差并入特质项
        assert err < 1e-7, f"{o} 持仓复原收益误差 {err}"
    log("逐日归因完成")

    attr_rows, spec_rows, ex_rows, liq_rows = [], [], [], []
    pin = pinned_down(p).reindex(index=dates, columns=p.codes).fillna(False).to_numpy(bool)
    # 纯因子收益:窗口内之和与冲击 z(用窗口前 250 日日波动标定)
    for w, (d0, d1, desc) in WINDOWS.items():
        idx = win_signal_days(bf.F.dropna(how="all").index, te, d0, d1)
        s = dates.get_loc(idx[0])
        hist = bf.F.iloc[max(0, s - 1 - N_HIST):s - 1].dropna(how="all")
        for k in ["Country"] + STYLE_KEYS:
            fsum = float(bf.F[k].reindex(idx).sum())
            sd = float(hist[k].std())
            attr_rows.append(dict(object="FACTOR", window=w, window_type="预注册", basis="纯因子收益",
                                  component=k, value=fsum, factor_ret_sum=fsum,
                                  shock_z=fsum / (sd * np.sqrt(len(idx))) if sd > 0 else np.nan,
                                  n_days=len(idx)))

    for o, A in att.items():
        for w, wtype, desc, idx in win_index[o]:
            idx = idx.intersection(A.dropna(subset=["c:style_total"]).index)
            if len(idx) < 2:
                continue
            a = A.reindex(idx)
            if o == "POOL_EW":
                comps = ["active", "specific", "c:style_total", "c:industry_total", "c:Country"] + \
                        [f"c:{k}" for k in STYLE_KEYS] + [f"c:{k}" for k in bf.ind_cols]
            else:
                comps = ["active", "cost", "c:cash", "specific", "c:style_total", "c:industry_total"] + \
                        [f"c:{k}" for k in STYLE_KEYS] + [f"c:{k}" for k in bf.ind_cols]
            for c in comps:
                key = c.replace("c:", "")
                val = float(a[c].sum()) * (-1 if c == "cost" else 1)
                attr_rows.append(dict(object=o, window=w, window_type=wtype,
                                      basis="绝对(池等权自身)" if o == "POOL_EW" else "主动(对池等权,现金腿单列)",
                                      component={"cost": "cost(负号=拖累)", "cash": "cash(现金腿)"}.get(key, key),
                                      value=val,
                                      mean_exposure=float(a[f"e:{key}"].mean()) if f"e:{key}" in a else np.nan,
                                      factor_ret_sum=float(bf.F[key].reindex(idx).sum()) if key in bf.F.columns else np.nan,
                                      n_days=len(idx)))
            if o == "POOL_EW":
                continue
            rows_ = np.array([dates.get_loc(t) for t in idx])
            H = Hs[o]

            # ---- 特质项按持仓股票分桶
            U_ = bf.residuals(rows_)
            Hd = H[rows_].astype(np.float64)
            inv = Hd.sum(1, keepdims=True)
            Ap = Hd - inv * bf.B[rows_]
            contrib = Ap * U_
            spec_chk = float(contrib.sum()) - float(a["specific"].sum())
            hn = np.divide(Hd, inv, out=np.zeros_like(Hd), where=inv > 1e-9)
            noexp = ~bf.allfin[rows_]
            dims_b = {"全A市值五分位": mvq[rows_], "池内过去20日收益五分位": p20q[rows_],
                      "ST": np.where(st[rows_], 1, 0)}
            for dname, bk in dims_b.items():
                labels = {"全A市值五分位": {1: "Q1最小", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5最大"},
                          "池内过去20日收益五分位": {1: "Q1跌最多", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5涨最多"},
                          "ST": {0: "非ST", 1: "ST"}}[dname]
                for bv, bl in labels.items():
                    m = (bk == bv)
                    spec_rows.append(dict(object=o, window=w, window_type=wtype, dim=dname, bucket=bl,
                                          specific_contrib=float(np.where(m, contrib, 0).sum()),
                                          port_weight_mean=float(np.where(m, hn, 0).sum(1).mean()),
                                          pool_weight_mean=float(np.where(m, bf.B[rows_], 0).sum(1).mean()),
                                          window_specific_total=float(a["specific"].sum()),
                                          recon_err=spec_chk))
                miss = ~np.isin(bk, list(labels))
                if (np.where(miss, hn, 0).sum() > 1e-6):
                    spec_rows.append(dict(object=o, window=w, window_type=wtype, dim=dname, bucket="无分桶数据",
                                          specific_contrib=float(np.where(miss, contrib, 0).sum()),
                                          port_weight_mean=float(np.where(miss, hn, 0).sum(1).mean()),
                                          pool_weight_mean=float(np.where(miss, bf.B[rows_], 0).sum(1).mean()),
                                          window_specific_total=float(a["specific"].sum()), recon_err=spec_chk))
            spec_rows.append(dict(object=o, window=w, window_type=wtype, dim="暴露缺失(池外持仓)", bucket="无暴露",
                                  specific_contrib=float(np.where(noexp, contrib, 0).sum()),
                                  port_weight_mean=float(np.where(noexp, hn, 0).sum(1).mean()),
                                  pool_weight_mean=0.0, window_specific_total=float(a["specific"].sum()),
                                  recon_err=spec_chk))

            # ---- 事前情景(现金腿单列)
            s = rows_[0]
            H_pre = H[s - 1].astype(np.float64)
            inv_pre = float(H_pre.sum())
            if inv_pre < 0.5:              # 窗口前尚未建仓(B_gbt 2023 年初)
                continue
            Bp = bf.u[s] / max(bf.u[s].sum(), 1)
            Ap_pre = H_pre - inv_pre * Bp
            e_pre = {k: float((Ap_pre * bf.Xz[k][s]).sum()) for k in STYLE_KEYS}
            ie = bf.ind_expo(Ap_pre[None, :], bf.capsum_pre[s][None, :])[0]
            e_pre.update({c: float(ie[j]) for j, c in enumerate(bf.ind_cols)})
            Fw = bf.F.reindex(idx)[bf.xcols].sum()
            bsum = float(bench.reindex(idx).sum())
            pred = {k: e_pre[k] * float(Fw[k]) for k in bf.xcols}
            pred_cash = (inv_pre - 1.0) * bsum
            pred_style = sum(pred[k] for k in STYLE_KEYS)
            pred_ind = sum(pred[k] for k in bf.ind_cols)
            pred_tot = pred_style + pred_ind + pred_cash
            real = float(a["active"].sum())
            hist_end = dates[s - 2]
            Fh = pd.concat([bench.rename("__cash__"), bf.F[bf.xcols]], axis=1).loc[:hist_end].dropna(how="any").iloc[-N_HIST:]
            sp = A["specific"].loc[:hist_end].dropna().iloc[-N_HIST:]
            ev = np.array([inv_pre - 1.0] + [e_pre[k] for k in bf.xcols])
            if len(Fh) >= MIN_HIST and len(sp) >= MIN_HIST:
                Sig = np.cov(Fh.to_numpy().T)
                vf, vs = float(ev @ Sig @ ev), float(sp.var())
                te_pred, te_fac = np.sqrt(len(idx) * (vf + vs)), np.sqrt(len(idx) * vf)
            else:
                te_pred = te_fac = np.nan
            act_hist = A["active"].loc[full0:hist_end].dropna()
            hist_w20 = float(act_hist.rolling(20).sum().min()) if len(act_hist) >= MIN_HIST else np.nan
            # 审查补充:与窗口同长度的历史最差(W1 27 日不应与 20 日比)
            hist_wL = float(act_hist.rolling(len(idx)).sum().min()) if len(act_hist) >= MIN_HIST else np.nan
            row = dict(object=o, window=w, window_type=wtype, te_start=te[idx[0]].date(),
                       te_end=te[idx[-1]].date(), n_days=len(idx), pre_holding_date=dates[s - 1].date(),
                       pre_invested=inv_pre,
                       realized_active=real, realized_cash=float(a["c:cash"].sum()),
                       realized_style=float(a["c:style_total"].sum()),
                       realized_industry=float(a["c:industry_total"].sum()),
                       realized_specific=float(a["specific"].sum()), realized_cost=-float(a["cost"].sum()),
                       pred_total=pred_tot, pred_style=pred_style, pred_industry=pred_ind, pred_cash=pred_cash,
                       pred_size_only=pred["Size 市值"], pred_nonlinsize_only=pred["NonlinSize 非线性市值"],
                       explained_ratio=pred_tot / real if real != 0 else np.nan,
                       same_sign=bool(np.sign(pred_tot) == np.sign(real)),
                       visible_ex_ante=bool(np.sign(pred_tot) == np.sign(real) and abs(pred_tot) >= 0.5 * abs(real)),
                       pred_te=te_pred, pred_te_factor_only=te_fac,
                       z_realized=real / te_pred if te_pred and np.isfinite(te_pred) else np.nan,
                       beyond_risk_model=bool(abs(real / te_pred) > 2) if te_pred and np.isfinite(te_pred) else None,
                       hist_worst20d_before=hist_w20, hist_worst_samelen_before=hist_wL,
                       ratio_to_hist_worst_samelen=real / hist_wL if hist_wL and np.isfinite(hist_wL) else np.nan)
            for k in STYLE_KEYS:
                row[f"pre_e_norm:{SHORT[k]}"] = e_pre[k] / inv_pre
                row[f"pred:{SHORT[k]}"] = pred[k]
            # 窗口内实际(归一化)暴露均值,用来对照"事前暴露"是否在窗口内被调仓改变
            for k in ("Size 市值", "NonlinSize 非线性市值", "Liquidity 流动性", "ResidVol 残差波动"):
                row[f"win_e_norm_mean:{SHORT[k]}"] = float(a[f"e:{k}"].mean())
            ex_rows.append(row)

            # ---- 流动性
            hn_pre = H_pre / inv_pre
            adv = p["adv20"].iloc[s].reindex(p.codes).to_numpy(np.float64)
            mvs = bf.mv[s]

            def wstats(wt, tag):
                m = wt > 1e-10
                ww, a_, q_ = wt[m], adv[m], mvq[s][m]

                def wmed(x, wgt):
                    ok = np.isfinite(x)
                    x, wgt = x[ok], wgt[ok]
                    o_ = np.argsort(x)
                    c = np.cumsum(wgt[o_]) / wgt[o_].sum()
                    return float(x[o_][np.searchsorted(c, 0.5)])
                return {f"{tag}_n_names": int(m.sum()),
                        f"{tag}_wmed_adv20_wan": wmed(a_, ww) / 1e4,
                        f"{tag}_share_adv_lt_5000wan": float(ww[a_ < 5e7].sum() / ww.sum()),
                        f"{tag}_share_adv_lt_3000wan": float(ww[a_ < 3e7].sum() / ww.sum()),
                        f"{tag}_wmed_mv_yi": wmed(mvs[m], ww) / 1e8,
                        f"{tag}_share_mvq1_allA": float(ww[q_ == 1].sum() / ww.sum()),
                        f"{tag}_wavg_mv_rank_pool": float(np.nansum(ww * mv_rank_u[s][m]) / ww.sum()),
                        f"{tag}_wavg_turn_rate_pct": float(np.nansum(ww * turn_rt[s][m]) / ww[np.isfinite(turn_rt[s][m])].sum())}
            lr = dict(object=o, window=w, window_type=wtype, pre_holding_date=dates[s - 1].date(),
                      pre_invested=inv_pre)
            lr.update(wstats(hn_pre, "port")); lr.update(wstats(Bp, "pool"))
            dg = diags[o].reindex(idx)
            dgf = diags[o].loc[full0:]
            lr.update(win_ld_hold_mean=float(dg["ld_hold"].mean()), win_ld_hold_max=float(dg["ld_hold"].max()),
                      win_days_ld_hold_gt_0p5pct=int((dg["ld_hold"] > 0.005).sum()),
                      win_ld_block_sum=float(dg["ld_block"].sum()),
                      win_cap_bind_share_of_rebalances=float(dg["clip_any_cnt"].sum() / max(dg["reb_cnt"].sum(), 1)),
                      win_cap_clip_ratio=float(dg["clip_vol"].sum() / dg["want_vol"].sum()) if dg["want_vol"].sum() > 0 else np.nan,
                      win_cash_mean=float(1 - dg["exposure"].mean()), win_turnover_mean=float(dg["turnover"].mean()),
                      full_cap_bind_share_of_rebalances=float(dgf["clip_any_cnt"].sum() / max(dgf["reb_cnt"].sum(), 1)),
                      full_cap_clip_ratio=float(dgf["clip_vol"].sum() / dgf["want_vol"].sum()),
                      full_cash_mean=float(1 - dgf["exposure"].mean()), full_turnover_mean=float(dgf["turnover"].mean()),
                      full_ld_hold_mean=float(dgf["ld_hold"].mean()))
            dn = p["one_word_down"].to_numpy(bool)
            pos = rows_ + 1                                      # te 行
            lr["win_pool_oneword_down_share_mean"] = float(np.mean([(dn[q] & bf.u[q - 1]).sum() / max(bf.u[q - 1].sum(), 1) for q in pos]))
            lr["win_pool_oneword_down_share_max"] = float(np.max([(dn[q] & bf.u[q - 1]).sum() / max(bf.u[q - 1].sum(), 1) for q in pos]))
            lr["win_allA_oneword_down_count_max"] = int(np.max([dn[q].sum() for q in pos]))
            lr["win_pool_oneword_down_count_max"] = int(np.max([(dn[q] & bf.u[q - 1]).sum() for q in pos]))
            # 审查补充:开盘 10 分钟整段钉在跌停价(含一字跌停)的第二口径 —— open10 TWAP 窗口里同样卖不掉
            lr["win_pool_pinned_down_count_max"] = int(np.max([(pin[q] & bf.u[q - 1]).sum() for q in pos]))
            lr["win_pool_pinned_down_share_max"] = float(np.max([(pin[q] & bf.u[q - 1]).sum() / max(bf.u[q - 1].sum(), 1) for q in pos]))
            lr["win_allA_pinned_down_count_max"] = int(np.max([pin[q].sum() for q in pos]))
            lr.update(win_pin_hold_mean=float(dg["pin_hold"].mean()), win_pin_hold_max=float(dg["pin_hold"].max()),
                      win_days_pin_hold_gt_0p5pct=int((dg["pin_hold"] > 0.005).sum()),
                      win_pin_sell_sum=float(dg["pin_sell"].sum()),
                      full_pin_hold_mean=float(dgf["pin_hold"].mean()))
            liq_rows.append(lr)
    pd.DataFrame(attr_rows).to_csv(LEDGER / "stress_attribution.csv", index=False)
    pd.DataFrame(spec_rows).to_csv(LEDGER / "stress_specific_breakdown.csv", index=False)
    matched_split(bf, Hs, att, base_objs, rets.index, te, mv_pct_all, mvq, p20q)
    log("stress_specific_matched.csv")
    pd.DataFrame(ex_rows).to_csv(LEDGER / "stress_exante_vs_realized.csv", index=False)
    pd.DataFrame(liq_rows).to_csv(LEDGER / "stress_liquidity.csv", index=False)
    log("归因/特质分桶/事前/流动性 写出")

    # ---------- 3. 2024Q1 前后的逐日持仓特征
    a0, a1 = (pd.Timestamp(x) for x in PATH_RANGE)
    prow = [i for i, d in enumerate(dates[:T - 2]) if a0 <= d <= a1]
    path = []
    for o in base_objs + [f"{b}|N1r" for b in base_objs]:
        H = Hs[o][prow].astype(np.float64)
        inv = H.sum(1, keepdims=True)
        hn = np.divide(H, inv, out=np.zeros_like(H), where=inv > 1e-9)
        A = att[o]
        for j, r_ in enumerate(prow):
            d = dates[r_]
            if inv[j, 0] < 1e-6:
                continue
            path.append(dict(object=o, date=d.date(), te=te[d].date(), invested=float(inv[j, 0]),
                             share_mvq1_allA=float(hn[j][mvq[r_] == 1].sum()),
                             share_past20_q1=float(hn[j][p20q[r_] == 1].sum()),
                             wavg_mv_rank_pool=float(np.nansum(hn[j] * mv_rank_u[r_])),
                             wavg_past20=float(np.nansum(hn[j] * np.nan_to_num(past20.iloc[r_].to_numpy()))),
                             e_norm_size=float(A["e:Size 市值"].get(d, np.nan)),
                             e_norm_nls=float(A["e:NonlinSize 非线性市值"].get(d, np.nan)),
                             e_norm_liq=float(A["e:Liquidity 流动性"].get(d, np.nan)),
                             turnover=float(diags[o]["turnover"].get(d, np.nan)),
                             active=float(A["active"].get(d, np.nan))))
    pool_path = [dict(object="POOL_EW", date=dates[r_].date(), te=te[dates[r_]].date(), invested=1.0,
                      share_mvq1_allA=float(bf.B[r_][mvq[r_] == 1].sum()),
                      share_past20_q1=float(bf.B[r_][p20q[r_] == 1].sum()),
                      wavg_mv_rank_pool=float(np.nansum(bf.B[r_] * mv_rank_u[r_])),
                      wavg_past20=float(np.nansum(bf.B[r_] * np.nan_to_num(past20.iloc[r_].to_numpy()))),
                      active=0.0) for r_ in prow]
    pd.DataFrame(pool_path + path).to_csv(LEDGER / "stress_holdings_path.csv", index=False)
    log("stress_holdings_path.csv")

    # ---------- 4. 反事实
    cf = []
    for base in PLANS:
        r0 = rets[base].dropna()
        e0 = r0 - bench.reindex(r0.index)
        for var in VARIANTS:
            o = base + var
            r1 = rets[o].dropna()
            e1 = r1 - bench.reindex(r1.index)
            Hn = Hs[o].astype(np.float64)
            inv = Hn.sum(1, keepdims=True)
            hn = np.divide(Hn, inv, out=np.zeros_like(Hn), where=inv > 1e-9)
            mvr = pd.Series(np.nansum(hn * np.nan_to_num(mv_rank_u, nan=0.0), 1), index=dates)
            mq1 = pd.Series(np.where(mvq == 1, hn, 0).sum(1), index=dates)
            for per, (a_, b_) in {"FULL_2021_2025": FULL, "DEV_2021_2023": DEV, "TEST_2024_2025": TEST}.items():
                ee = e1.loc[a_:b_]
                if len(ee) < 60:
                    continue
                rr, bb = r1.loc[a_:b_], bench.reindex(r1.index).loc[a_:b_]
                n = len(rr)
                ann = float((1 + rr).prod() ** (TRADING_DAYS / n) - 1)
                annb = float((1 + bb).prod() ** (TRADING_DAYS / n) - 1)
                d = (e1 - e0.reindex(e1.index)).loc[a_:b_]
                cf.append(dict(object=base, variant=var or "原始", kind=VAR_KIND[var], period=per, n_days=n,
                               ann_return=ann, ann_excess=ann - annb,
                               sharpe_excess=float(ee.mean() / ee.std() * np.sqrt(TRADING_DAYS)),
                               diff_vs_orig_ann=float(d.mean() * TRADING_DAYS) if var else 0.0,
                               diff_vs_orig_nw_t=nw_t(d.to_numpy(), 10) if var else np.nan,
                               mean_e_norm_size=float(att[o]["e:Size 市值"].loc[a_:b_].mean()),
                               mean_e_norm_liq=float(att[o]["e:Liquidity 流动性"].loc[a_:b_].mean()),
                               mean_mv_rank_pool=float(mvr.loc[a_:b_].reindex(rr.index).mean()),
                               mean_share_mvq1_allA=float(mq1.loc[a_:b_].reindex(rr.index).mean())))
            for w, (d0, d1, desc) in WINDOWS.items():
                idx = win_signal_days(r1.index, te, d0, d1)
                if len(idx) < 2:
                    continue
                s1, s0, b = r1.reindex(idx), r0.reindex(idx), bench.reindex(idx)
                ex1 = float((1 + s1).prod() - (1 + b).prod())
                ex0 = float((1 + s0).prod() - (1 + b).prod())
                cf.append(dict(object=base, variant=var or "原始", kind=VAR_KIND[var], period=w, n_days=len(idx),
                               window_excess=ex1, window_excess_orig=ex0, avoided=ex1 - ex0 if var else 0.0,
                               mean_e_norm_size=float(att[o]["e:Size 市值"].reindex(idx).mean()),
                               mean_e_norm_liq=float(att[o]["e:Liquidity 流动性"].reindex(idx).mean()),
                               mean_mv_rank_pool=float(mvr.reindex(idx).mean()),
                               mean_share_mvq1_allA=float(mq1.reindex(idx).mean())))
    pd.DataFrame(cf).to_csv(LEDGER / "stress_counterfactual.csv", index=False)

    # ---------- 5. Barra Size 截尾诊断:窗口前一日池内有多少股票的 Size 暴露被截在 -3
    clip = []
    for w, (d0, d1, desc) in WINDOWS.items():
        s = dates.get_loc(win_signal_days(dates[:T - 2], te, d0, d1)[0])
        for k in ("Size 市值", "NonlinSize 非线性市值"):
            x = bf.X[k][s][bf.u[s]]
            x = x[np.isfinite(x)]
            clip.append(dict(window=w, date=dates[s].date(), factor=k, n_pool=int(x.size),
                             share_at_min=float((x <= x.min() + 1e-6).mean()), min_value=float(x.min()),
                             share_le_minus3=float((x <= -2.999).mean()), pool_ew_mean=float(x.mean())))
    pd.DataFrame(clip).to_csv(LEDGER / "stress_barra_clip.csv", index=False)

    # ---------- 6. 审查补充:参与率资金口径敏感性、幸存者统计
    participation_sensitivity(rets, diags, base_objs, te, full0)
    csi_alignment_table(rets, te)
    sv = write_survivorship(p)
    log("stress_participation_sensitivity.csv / stress_survivorship.json", sv["delisted_2020_2025_missing"])
    log(f"analyze 完成,用时 {time.time() - t_all:.0f}s")
