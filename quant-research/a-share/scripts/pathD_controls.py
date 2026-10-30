"""路径 D 的对照实验:
A. 随机零模型 —— 观察到的"久趋势更易续"是真惯性,还是随机游走+个股异质性的机械产物?
   NULL-1: 个股日对数收益整体随机重排(毁掉一切时序结构,保留个股均值/波动)
   NULL-2: r = mkt[t] + eps[i,t],只重排 eps(保留大盘节奏与共振,毁掉个股特有的序列相关)
B. 个股异质性剥离 —— 按"过去 250 日处于该状态的比例"(无前视)分五组,组内看 hazard 是否还随 k 上升。
C. 中性化收益 —— 相对同日"成交额十分位"/"申万一级行业"/"行业×成交额三分位"的超额。
D. 行业层趋势 —— 等权行业指数上的同一状态,持续性与收益;以及个股所属行业趋势 age 对个股收益的作用。
E. 中际旭创 300308.SZ 的实际 spell 明细(直觉校准)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST

OUT = "/path/to/user/Desktop/A stock market/reports/pathD_controls.txt"
_buf = []


def w(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _buf.append(s)


def run_age(cond, ok):
    T, N = cond.shape
    age = np.zeros((T, N), np.int32)
    cur = np.zeros(N, np.int32)
    for t in range(T):
        cur = np.where(ok[t], np.where(cond[t], cur + 1, 0), cur)
        age[t] = cur
    return age


def fut_run(B, h):
    T, N = B.shape
    nxt = np.zeros((T, N), np.int16)
    acc = np.zeros(N, np.int16)
    for t in range(T - 1, -1, -1):
        nxt[t] = acc
        acc = np.where(B[t], np.minimum(acc + 1, h), 0).astype(np.int16)
    return nxt


def state_from_close(C):
    df = pd.DataFrame(C)
    M20 = df.rolling(20, min_periods=18).mean().values.astype(np.float32)
    M60 = df.rolling(60, min_periods=50).mean().values.astype(np.float32)
    ok = np.isfinite(C) & np.isfinite(M20) & np.isfinite(M60)
    up = ok & (C > M60) & (M20 > M60)
    dn = ok & (C < M60) & (M20 < M60)
    return up, dn, ok


def spell_len(B):
    T, N = B.shape
    out = []
    for j in range(N):
        col = B[:, j]
        if not col.any():
            continue
        d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        k = (s > 60) & (e < T)
        if k.any():
            out.append((e - s)[k])
    return np.concatenate(out) if out else np.zeros(0, int)


KS = [1, 5, 10, 20, 30, 40, 60, 80, 120]


def hazard_row(age, F, U, h, mask2d=None, minn=300):
    r = []
    for k in KS:
        sel = (age >= max(k - 2, 1)) & (age <= k + 2) & U
        if mask2d is not None:
            sel = sel & mask2d
        r.append(np.nan if sel.sum() < minn else (F[sel] >= h).mean() * 100)
    return r


def fmt(r):
    return "".join("  --  " if not np.isfinite(x) else f"{x:6.1f}" for x in r)


# ================= main =================
def main():
    p = Panel()
    close = p.raw("close_px")
    dates, codes = close.index, close.columns
    T, N = close.shape
    U = ST.universe(p, "V1000").values
    C = close.values.astype(np.float32)
    okpx = np.isfinite(C)
    tail = np.zeros(T, bool); tail[:T - 20] = True
    tail2d = tail[:, None]

    up, dn, ok = state_from_close(C)
    age_up, age_dn = run_age(up, okpx), run_age(dn, okpx)
    F_up, F_dn = fut_run(age_up > 0, 20), fut_run(age_dn > 0, 20)

    # ---------------- A. 零模型 ----------------
    w("=" * 110)
    w("A. 随机零模型对照:P(状态再连续存活 20 天 | age≈k) %,以及 spell 长度分布")
    w("=" * 110)
    w(f"{'序列':<30}" + "".join(f"k={k:<4}" for k in KS) + "   中位spell  P90  P(>=60d)%")
    for lbl, ag, F, B in (("实际 UP", age_up, F_up, up), ("实际 DOWN", age_dn, F_dn, dn)):
        L = spell_len(ag > 0)
        w(f"{lbl:<30}{fmt(hazard_row(ag, F, U, 20, tail2d))}"
          f"{np.percentile(L,50):>10.0f}{np.percentile(L,90):>6.0f}{(L>=60).mean()*100:>9.1f}")

    lr = np.log(C[1:] / C[:-1])
    lr = np.where(np.isfinite(lr), lr, np.nan)
    with np.errstate(invalid="ignore"):
        mkt = np.nanmean(np.where(U[1:], lr, np.nan), axis=1)
    mkt = np.nan_to_num(mkt, nan=0.0)
    eps = lr - mkt[:, None]
    for nm, base, add in (("NULL-1 全收益重排", lr, None), ("NULL-2 只重排残差(保大盘)", eps, mkt)):
        for seed in (0, 1):
            rng = np.random.default_rng(seed)
            X = base.copy()
            for j in range(N):
                v = np.flatnonzero(np.isfinite(X[:, j]))
                if len(v) > 2:
                    X[v, j] = X[rng.permutation(v), j]
            if add is not None:
                X = X + add[:, None]
            Cs = np.vstack([np.ones(N, np.float32),
                            np.exp(np.nancumsum(np.where(np.isfinite(X), X, 0.0), 0))
                            ]).astype(np.float32)
            Cs[~okpx] = np.nan
            u2, d2, _ = state_from_close(Cs)
            a2 = run_age(u2, okpx)
            f2 = fut_run(a2 > 0, 20)
            L = spell_len(a2 > 0)
            w(f"{nm+f' seed{seed} UP':<30}{fmt(hazard_row(a2, f2, U, 20, tail2d))}"
              f"{np.percentile(L,50):>10.0f}{np.percentile(L,90):>6.0f}{(L>=60).mean()*100:>9.1f}")
            del Cs, u2, d2, a2, f2
    del lr, eps
    w("  解读:若 NULL 的 k 梯度与实际相当,则'久趋势更易续'并非真惯性。")

    # ---------------- B. 个股异质性剥离 ----------------
    w("\n" + "=" * 110)
    w("B. 剥离个股异质性:按'过去250日处于UP的比例'(t-1 及之前,无前视)分五组,组内 hazard")
    w("=" * 110)
    freq = pd.DataFrame(up.astype(np.float32)).rolling(250, min_periods=200).mean().shift(1).values
    q = np.full((T, N), -1, np.int8)
    for t in range(T):
        m = U[t] & np.isfinite(freq[t])
        if m.sum() < 200:
            continue
        v = freq[t, m]
        cut = np.quantile(v, [.2, .4, .6, .8])
        q[t, m] = np.searchsorted(cut, v, side="right")
    w(f"{'组(过去UP占比)':<30}" + "".join(f"k={k:<4}" for k in KS) + "   组内均值UP占比")
    for g in range(5):
        m2 = (q == g)
        w(f"{'Q'+str(g+1)+(' 最少趋势' if g==0 else ' 最常趋势' if g==4 else ''):<30}"
          f"{fmt(hazard_row(age_up, F_up, U, 20, m2 & tail2d))}"
          f"{np.nanmean(freq[m2 & U])*100:>12.1f}%")
    del freq, q

    # ---------------- C. 中性化收益 ----------------
    w("\n" + "=" * 110)
    w("C. age 分桶的中性化超额(h=20):相对全池 / 成交额十分位 / 申万一级 / 行业×成交额三分位")
    w("=" * 110)
    tw = p.raw("twap_open10")
    R = (tw.shift(-21) / tw.shift(-1) - 1.0).values.astype(np.float32)
    del tw
    adv = np.log(p["adv20"].values + 1.0)
    ind = load_ind(codes)
    neutral(age_up, "UP", R, U, adv, ind, dates)
    neutral(age_dn, "DOWN", R, U, adv, ind, dates)

    # ---------------- D. 行业层 ----------------
    w("\n" + "=" * 110)
    w("D. 行业层(申万一级等权指数)趋势的持续性与收益")
    w("=" * 110)
    industry_level(C, U, ind, dates, R, age_up)

    # ---------------- E. 案例 ----------------
    w("\n" + "=" * 110)
    w("E. 中际旭创 300308.SZ 的实际 UP spell(close>MA60 且 MA20>MA60)")
    w("=" * 110)
    case(C, age_up, dates, codes, "300308.SZ")
    case(C, age_up, dates, codes, "002230.SZ")

    with open(OUT, "w") as f:
        f.write("\n".join(_buf))
    w(f"\n[written] {OUT}")


def load_ind(codes):
    df = pd.read_csv("/path/to/user/Desktop/A stock market/stock_sw_industry.csv",
                     encoding="utf-8-sig")
    m = df.set_index("ts_code")["sw_industry"]
    s = pd.Series(codes).map(m).fillna("NA")
    cats, inv = np.unique(s.values, return_inverse=True)
    return inv.astype(np.int16), cats


AGE_BINS = [(1, 4), (5, 9), (10, 19), (20, 39), (40, 59), (60, 119), (120, 10 ** 6)]


def cell_demean(R, cells, U):
    """按 (T,N) 整数 cell 标签在每日池内做去均值。cell<0 视为无效。"""
    T, N = R.shape
    out = np.full_like(R, np.nan)
    ok = np.isfinite(R) & U & (cells >= 0)
    K = int(cells.max()) + 1
    for t in range(T):
        m = ok[t]
        if m.sum() < 50:
            continue
        cl = cells[t, m]
        rv = R[t, m]
        cnt = np.bincount(cl, minlength=K)
        ssum = np.bincount(cl, weights=rv, minlength=K)
        good = cnt >= 5
        mu = np.where(good, ssum / np.maximum(cnt, 1), np.nan)
        out[t, m] = rv - mu[cl]
    return out


def neutral(age, lbl, R, U, adv, ind, dates):
    T, N = R.shape
    inv, cats = ind
    IND = np.repeat(inv[None, :], T, 0).astype(np.int16)
    IND = np.where(U, IND, -1)
    # 成交额十分位
    DEC = np.full((T, N), -1, np.int16)
    for t in range(T):
        m = U[t] & np.isfinite(adv[t])
        if m.sum() < 100:
            continue
        v = adv[t, m]
        DEC[t, m] = np.searchsorted(np.quantile(v, np.arange(1, 10) / 10), v, side="right")
    TER = np.where(DEC >= 0, DEC // 4, -1).astype(np.int16)          # 三分位近似
    CELL = np.where((IND >= 0) & (TER >= 0), IND * 3 + TER, -1).astype(np.int16)
    ALL = np.where(U, 0, -1).astype(np.int16)

    variants = {"全池": ALL, "成交额10分位": DEC, "申万一级": IND, "行业x额三分位": CELL}
    D = {k: cell_demean(R, v, U) for k, v in variants.items()}
    y = dates.year.values
    segs = {"TRAIN 2020-23": (y <= 2023), "TEST 2024-25": (y >= 2024),
            "2024-09-24起": dates.values >= np.datetime64("2024-09-24")}
    w(f"\n  --- {lbl} ---")
    w(f"{'段':<16}{'age桶':<10}{'日均只数':>8}" + "".join(f"{k:>14}" for k in variants))
    for lab, sm in segs.items():
        for lo, hi in AGE_BINS:
            sel = (age >= lo) & (age <= hi) & U & sm[:, None]
            if sel.sum() < 2000:
                continue
            ns = sel.sum(1)
            row = f"{lab:<16}{f'{lo}-{hi if hi<10**5 else chr(105)+chr(110)+chr(102)}':<10}{ns[ns>0].mean():>8.0f}"
            for k in variants:
                x = D[k][sel]
                x = x[np.isfinite(x)]
                row += f"{x.mean()*100:>14.2f}" if len(x) > 500 else f"{'--':>14}"
            w(row)
        w("")


def industry_level(C, U, ind, dates, R, age_up_stk):
    inv, cats = ind
    T, N = C.shape
    ret = np.full((T, N), np.nan, np.float32)
    ret[1:] = C[1:] / C[:-1] - 1.0
    ret = np.where(U & np.isfinite(ret) & (np.abs(ret) < 0.5), ret, np.nan)
    K = len(cats)
    IR = np.full((T, K), np.nan, np.float32)
    for k in range(K):
        m = inv == k
        if m.sum() < 8:
            continue
        IR[:, k] = np.nanmean(ret[:, m], axis=1)
    IC_ = np.nancumprod(np.where(np.isfinite(IR), 1 + IR, 1.0), 0).astype(np.float32)
    okI = np.isfinite(IR).astype(bool)
    okI[0] = okI[1]
    up_i, dn_i, _ = state_from_close(IC_)
    a_i = run_age(up_i, np.ones((T, K), bool))
    f_i = fut_run(a_i > 0, 20)
    Ui = np.ones((T, K), bool)
    tail = np.zeros(T, bool); tail[:T - 20] = True
    w(f"  行业数={K}  申万一级等权指数")
    w(f"{'':<22}" + "".join(f"k={k:<4}" for k in KS))
    w(f"{'P(再撑20天|age≈k) UP':<22}{fmt(hazard_row(a_i, f_i, Ui, 20, tail[:,None], minn=60))}")
    a_d = run_age(dn_i, np.ones((T, K), bool)); f_d = fut_run(a_d > 0, 20)
    w(f"{'P(再撑20天|age≈k) DN':<22}{fmt(hazard_row(a_d, f_d, Ui, 20, tail[:,None], minn=60))}")
    L = spell_len(a_i > 0)
    w(f"  行业 UP spell:n={len(L)} 中位={np.percentile(L,50):.0f} P75={np.percentile(L,75):.0f} "
      f"P90={np.percentile(L,90):.0f} P(>=60d)={(L>=60).mean()*100:.1f}%")

    # 行业指数自身未来 20 日收益(收盘口径)
    fwdI = np.full((T, K), np.nan, np.float32)
    fwdI[:T - 21] = IC_[21:] / IC_[1:T - 20] - 1.0
    y = dates.year.values
    segs = {"TRAIN 2020-23": (y <= 2023), "TEST 2024-25": (y >= 2024),
            "2024-09-24起": dates.values >= np.datetime64("2024-09-24")}
    w(f"\n{'段':<16}{'行业age桶':<12}{'行业指数fwd20%':>16}{'全行业均值%':>14}{'超额%':>10}{'样本':>8}")
    for lab, sm in segs.items():
        base = np.nanmean(np.where(sm[:, None], fwdI, np.nan))
        for lo, hi in AGE_BINS:
            sel = (a_i >= lo) & (a_i <= hi) & sm[:, None] & np.isfinite(fwdI)
            if sel.sum() < 200:
                continue
            m = fwdI[sel].mean()
            w(f"{lab:<16}{f'{lo}-{hi if hi<10**5 else 999}':<12}{m*100:>16.2f}{base*100:>14.2f}"
              f"{(m-base)*100:>10.2f}{int(sel.sum()):>8}")
        w("")

    # 个股收益 vs 其所属行业的 trend age(行业中性后)
    IND2 = np.where(U, np.repeat(inv[None, :], T, 0), -1).astype(np.int16)
    ind_age = np.zeros((T, N), np.int32)
    for k in range(K):
        ind_age[:, inv == k] = a_i[:, [k]]
    w(f"{'段':<16}{'行业age桶':<12}{'个股fwd20%':>13}{'全池%':>10}{'超额%':>9}{'日均只数':>9}")
    for lab, sm in segs.items():
        b = U & np.isfinite(R) & sm[:, None]
        base = R[b].mean()
        for lo, hi in AGE_BINS:
            sel = (ind_age >= lo) & (ind_age <= hi) & b
            if sel.sum() < 2000:
                continue
            w(f"{lab:<16}{f'{lo}-{hi if hi<10**5 else 999}':<12}{R[sel].mean()*100:>13.2f}"
              f"{base*100:>10.2f}{(R[sel].mean()-base)*100:>9.2f}{sel.sum(1)[sel.any(1)].mean():>9.0f}")
        w("")


def case(C, age, dates, codes, code):
    if code not in set(codes):
        w(f"  {code} 不在样本"); return
    j = list(codes).index(code)
    a = age[:, j]
    B = a > 0
    d = np.diff(np.concatenate(([0], B.astype(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    w(f"  {code}: 共 {len(s)} 段 UP;长度>=20 日的段:")
    for a0, b0 in zip(s, e):
        L = b0 - a0
        if L < 20:
            continue
        p0, p1 = C[a0, j], C[min(b0 - 1, len(C) - 1), j]
        w(f"    {dates[a0].date()} -> {dates[min(b0-1,len(dates)-1)].date()}  {L:>4}日  "
          f"段内涨幅 {(p1/p0-1)*100:>7.1f}%")


if __name__ == "__main__":
    main()
