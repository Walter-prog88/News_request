"""路径 B:监督学习直接吃 60 日原始量价序列。第 1 步 —— 构造样本。

特征(全部只用 T 日及之前收盘数据,全部 scale-free,不依赖本项目任何已有因子):
  cols 0..59   : 归一化日 log 收益 r_k / sigma_60   (训练前 in-place cumsum -> 标准化 log 价格路径)
  cols 60..119 : 窗口内 log(成交额) 的 z-score      (标准化 log 成交额路径)
辅助列(汇总统计量对照 & 条件版用):
  s1 = 60 日收益, s2 = 60 日日收益标准差, s3 = log(近5日均额/60日均额), s4 = 距60日最高收盘
  mom120 = 120 日收益
标签 y = 未来 20 日收益(twap_open10 口径)在当日决策池内的截面百分位 (0,1)
"""
from __future__ import annotations
import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
from afactor import strategy as ST

OUT = Path("/private/tmp/claude-501/-Users-walter-Desktop-A-stock-market/6780c3ab-65c2-4a54-ae1e-2b1cdff53a2d/scratchpad/pathb")
OUT.mkdir(parents=True, exist_ok=True)

H = 20
WIN = 60
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TRAIN_STRIDE = 3          # 训练段每 3 个交易日取一个截面(控内存)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class Builder:
    """把面板转成 numpy,按日期切窗口。内存:几个 1455x5565 float32 矩阵 ~ 32MB each。"""

    def __init__(self):
        p = Panel()
        self.dates = p.dates
        self.codes = p.codes
        c = p.raw("close_px").to_numpy(np.float32)
        a = p.raw("amount").to_numpy(np.float32)
        tw = p.raw("twap_open10").to_numpy(np.float32)
        univ = ST.universe(p, "V1000").to_numpy(bool)
        del p

        with np.errstate(divide="ignore", invalid="ignore"):
            self.LC = np.where(c > 0, np.log(np.maximum(c, 1e-12)), np.nan).astype(np.float32)
            self.LA = np.where(a > 0, np.log(np.maximum(a, 1e-12)), np.nan).astype(np.float32)
        self.C = c
        self.A = a
        self.DR = np.empty_like(self.LC)
        self.DR[0] = np.nan
        self.DR[1:] = self.LC[1:] - self.LC[:-1]
        self.univ = univ

        # 前瞻收益: tw[t+1+H]/tw[t+1]-1
        T = len(self.dates)
        fwd = np.full_like(tw, np.nan)
        hi, lo = 1 + H, 1
        fwd[: T - hi] = tw[hi:] / tw[lo: T - hi + lo] - 1.0
        self.FWD = fwd
        del c, a, tw

    def label_rank(self, t: int) -> np.ndarray:
        """当日决策池内的前瞻收益百分位 (0,1);池外/缺失为 nan。"""
        f = self.FWD[t]
        m = self.univ[t] & np.isfinite(f)
        out = np.full(f.shape, np.nan, np.float32)
        n = int(m.sum())
        if n < 200:
            return out
        v = f[m]
        r = pd.Series(v).rank(method="average").to_numpy(np.float64)
        out[m] = (r / (n + 1.0)).astype(np.float32)
        return out

    def build(self, t: int):
        """返回 (X[n,120], y[n], aux[n,5], idx[n]) —— idx 是股票列号。X 前 60 列是归一化日收益(未 cumsum)。"""
        if t < 120 or t + 1 + H >= len(self.dates):
            return None
        s, e = t - WIN + 1, t + 1           # 窗口 [s, e)
        dr = self.DR[s:e]                   # 60 x N,需要 LC[s-1] 有效
        la = self.LA[s:e]
        y = self.label_rank(t)

        ok = self.univ[t] & np.isfinite(y)
        ok &= np.isfinite(dr).all(0) & np.isfinite(la).all(0)
        ok &= np.isfinite(self.LC[t - 120])
        if ok.sum() < 200:
            return None

        idx = np.flatnonzero(ok)
        dr = dr[:, idx]                     # 60 x n
        la = la[:, idx]
        sig = dr.std(0)
        vsd = la.std(0)
        g = (sig > 1e-8) & (vsd > 1e-8)
        if g.sum() < 200:
            return None
        idx = idx[g]
        dr = dr[:, g]; la = la[:, g]; sig = sig[g]; vsd = vsd[g]

        n = len(idx)
        X = np.empty((n, 2 * WIN), np.float32)
        X[:, :WIN] = (dr / sig).T
        X[:, WIN:] = ((la - la.mean(0)) / vsd).T

        lc_t = self.LC[t, idx]
        aux = np.empty((n, 5), np.float32)
        aux[:, 0] = np.expm1(lc_t - self.LC[t - WIN, idx])                       # s1 60日收益
        aux[:, 1] = sig                                                          # s2 波动
        amt_w = self.A[s:e, idx]
        aux[:, 2] = np.log(amt_w[-5:].mean(0) / amt_w.mean(0))                   # s3 成交额比
        aux[:, 3] = np.expm1(lc_t - self.LC[s:e, idx].max(0))                    # s4 距60日高
        aux[:, 4] = np.expm1(lc_t - self.LC[t - 120, idx])                       # mom120
        return X, y[idx].astype(np.float32), aux, idx


def main():
    b = Builder()
    log("panel loaded", b.LC.shape)
    dates = b.dates
    tr_mask = (dates <= TRAIN_END)
    all_t = np.arange(len(dates))
    # 训练段:留出最后 H+1 个交易日做 purge,避免标签窗口伸进检验段
    tr_last = int(np.flatnonzero(tr_mask)[-1]) - (H + 1)
    tr_dates = [t for t in all_t if 120 <= t <= tr_last][::TRAIN_STRIDE]
    log(f"train sections={len(tr_dates)}  from {dates[tr_dates[0]].date()} to {dates[tr_dates[-1]].date()}")

    nmax = len(tr_dates) * int(b.univ.sum(1).max())
    X = np.empty((nmax, 120), np.float32)
    Y = np.empty(nmax, np.float32)
    AUX = np.empty((nmax, 5), np.float32)
    DT = np.empty(nmax, np.int32)
    pos = 0
    t0 = time.time()
    for i, t in enumerate(tr_dates):
        r = b.build(t)
        if r is None:
            continue
        x, y, aux, idx = r
        k = len(y)
        X[pos:pos + k] = x; Y[pos:pos + k] = y; AUX[pos:pos + k] = aux; DT[pos:pos + k] = t
        pos += k
        if i % 40 == 0:
            log(f"  {i}/{len(tr_dates)} rows={pos} {time.time()-t0:.0f}s")
    log(f"train rows={pos}")

    np.save(OUT / "train_X.npy", X[:pos])
    np.save(OUT / "train_Y.npy", Y[:pos])
    np.save(OUT / "train_AUX.npy", AUX[:pos])
    np.save(OUT / "train_DT.npy", DT[:pos])
    meta = {"n": int(pos), "train_dates": [int(t) for t in tr_dates],
            "dates": [str(d.date()) for d in dates], "H": H, "WIN": WIN,
            "stride": TRAIN_STRIDE}
    (OUT / "meta.json").write_text(json.dumps(meta))
    log("saved to", OUT)


if __name__ == "__main__":
    main()
