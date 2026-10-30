"""Sequence-model chapter: what does a recurrent encoder look at?

The 312-factor matrix hands the model a pre-digested snapshot of day t. This
chapter removes that intermediation: the input is the raw 60-session x 20-channel
path ending at day t, and the model has to discover its own summary of it. Two
encoders read the same tensor so that "sequence helps" cannot be confused with
"this particular architecture helps".

The centrepiece is not the curve, it is the attention profile. An additive
temporal attention head over the 60 steps is forced to publish where it spends
its mass, so the lags the model leans on become an observable rather than a
story told after the fact.

What that observable actually says here is a negative result, and the chapter
must report it as one: the profile comes out flat (1.66%-1.75% per lag against
a 1.67% uniform baseline), identical across both seeds, all eight years, and
even across windows whose time order has been scrambled. Three independent
diagnostics are computed and written next to the profile so the flatness cannot
be quietly dressed up as a discovered horizon -- see input_dependence in
results/seq_attention.json.

Same house conventions as the rest of the package: information up to day t,
fill at t+1 continuous open, open-to-open marking, 0.5bp per executed side,
Top-4 equal weight, full-sample fit (upper-bound research 口径).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sweep_dial import CACHE, ZOO, load_env, run_topk, stats  # noqa: E402

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"

SEQ_LEN = 60
STD_WIN = 252
STD_MIN = 60
Z_CLIP = 5.0
D_MODEL = 64
BATCH = 4096
LR = 1e-3
WD = 1e-4

# Channel order is load-bearing: the first N_RAW are per-symbol series that get
# a causal rolling z-score, the rest are already cross-sectional ranks in [0,1].
RAW_CHANNELS = [
    "ret_cc", "overnight_ret", "ret_oc", "open30_ret", "last30_ret",
    "rv_closepath", "up_rv", "down_rv", "log_hl_range",
    "high_time_ordinal", "zr_share", "n_price_moves",
    "dlog_amount", "flow_ratio",
]
RANK_CHANNELS = [
    "cs_rank_amount", "cs_rank_ret_cc", "cs_rank_rv", "cs_rank_overnight",
    "cs_rank_flow", "cs_rank_mom21",
]
CHANNELS = RAW_CHANNELS + RANK_CHANNELS
N_RAW = len(RAW_CHANNELS)
N_CH = len(CHANNELS)


# ---------------------------------------------------------------- panel build

def _rank01_rows(m, mask):
    """Row-wise percentile rank in [0,1] over `mask`; elsewhere NaN."""
    out = np.full(m.shape, np.nan, dtype=np.float32)
    ok = mask & np.isfinite(m)
    cnt = ok.sum(axis=1)
    order = np.argsort(np.where(ok, m, np.inf), axis=1, kind="stable")
    ranks = np.empty(m.shape, dtype=np.int32)
    rng = np.arange(m.shape[1], dtype=np.int32)
    np.put_along_axis(ranks, order, rng[None, :], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = ranks / np.maximum(cnt - 1, 1)[:, None]
    out[ok] = vals[ok].astype(np.float32)
    return out


def _causal_z(a):
    """Rolling 252-session z-score ending at t. Uses only days <= t."""
    df = pd.DataFrame(a)
    mu = df.rolling(STD_WIN, min_periods=STD_MIN).mean()
    sd = df.rolling(STD_WIN, min_periods=STD_MIN).std()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (df - mu) / sd.where(sd > 0)
    return np.clip(z.to_numpy(dtype=np.float32), -Z_CLIP, Z_CLIP)


def build_panel(syms):
    """(T_zoo, N, 20) float32 on the zoo calendar, NaN already filled."""
    meta = json.load(open(f"{ZOO}/meta.json"))
    zsyms = list(meta["symbols"])
    zi = {s: i for i, s in enumerate(zsyms)}
    col = np.array([zi[s] for s in syms])
    T = len(meta["dates"])

    def z(name):
        return np.load(f"{ZOO}/{name}.npy")[:T][:, col].astype(np.float32)

    valid = np.load(f"{ZOO}/daily_ohlc_valid.npy")[:T][:, col].astype(bool)
    gate = np.load(f"{ZOO}/liq_gate_v2.npy")[:T][:, col].astype(bool)
    live = valid & gate

    hi, lo = z("high_continuous"), z("low_continuous")
    with np.errstate(invalid="ignore", divide="ignore"):
        hl = np.log(np.where((hi > 0) & (lo > 0), hi / lo, np.nan))

    amt = pd.read_parquet(f"{ZOO}/daily_amount_all.parquet")[syms].to_numpy(
        dtype=np.float32)
    flow = pd.read_parquet(f"{ZOO}/flow_ratio_all.parquet")[syms].to_numpy(
        dtype=np.float32)
    amt = np.where(amt > 0, amt, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        damt = np.diff(np.log(amt), axis=0, prepend=np.nan)

    ret = z("ret_cc")
    rv = z("rv_closepath_continuous")
    onr = z("overnight_ret_continuous")
    raw = {
        "ret_cc": ret,
        "overnight_ret": onr,
        "ret_oc": z("ret_oc_continuous"),
        "open30_ret": z("open30_ret_continuous"),
        "last30_ret": z("last30_ret"),
        "rv_closepath": rv,
        "up_rv": z("up_rv_continuous"),
        "down_rv": z("down_rv_continuous"),
        "log_hl_range": hl,
        "high_time_ordinal": z("high_time_ordinal_continuous"),
        "zr_share": z("zr_share_continuous"),
        "n_price_moves": z("n_price_moves"),
        "dlog_amount": damt,
        "flow_ratio": flow,
    }

    # 21-session momentum, causal, from log returns so it survives gaps.
    lr = np.log1p(np.clip(np.where(valid, ret, np.nan), -0.25, 0.25))
    mom21 = pd.DataFrame(lr).rolling(21, min_periods=15).sum().to_numpy(
        dtype=np.float32)

    P = np.empty((T, len(syms), N_CH), dtype=np.float32)
    for k, name in enumerate(RAW_CHANNELS):
        a = np.where(valid, raw[name], np.nan) if name != "flow_ratio" \
            else raw[name]
        P[:, :, k] = np.nan_to_num(_causal_z(a), nan=0.0,
                                   posinf=0.0, neginf=0.0)
    rank_src = {"cs_rank_amount": amt, "cs_rank_ret_cc": ret,
                "cs_rank_rv": rv, "cs_rank_overnight": onr,
                "cs_rank_flow": flow, "cs_rank_mom21": mom21}
    for k, name in enumerate(RANK_CHANNELS):
        P[:, :, N_RAW + k] = np.nan_to_num(
            _rank01_rows(rank_src[name], live), nan=0.5)
    return P


# --------------------------------------------------------------------- models

def make_models(torch, nn):
    class LSTMAttn(nn.Module):
        """2-layer LSTM + additive attention. Publishes its attention."""

        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(N_CH, D_MODEL, num_layers=2, batch_first=True)
            self.att_w = nn.Linear(D_MODEL, D_MODEL)
            self.att_v = nn.Linear(D_MODEL, 1, bias=False)
            self.head = nn.Linear(D_MODEL, 1)

        def forward(self, x, want_att=False):
            h, _ = self.rnn(x)
            e = self.att_v(torch.tanh(self.att_w(h))).squeeze(-1)
            a = torch.softmax(e, dim=1)
            ctx = (a.unsqueeze(-1) * h).sum(1)
            y = self.head(ctx).squeeze(-1)
            return (y, a) if want_att else (y, None)

    class TFMEnc(nn.Module):
        """2-layer encoder, learned positions, mean-pool. Same head budget."""

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(N_CH, D_MODEL)
            self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=4, dim_feedforward=2 * D_MODEL,
                dropout=0.1, batch_first=True, norm_first=True,
                activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(D_MODEL, 1)

        def forward(self, x, want_att=False):
            h = self.enc(self.proj(x) + self.pos)
            return self.head(h.mean(1)).squeeze(-1), None

    return LSTMAttn, TFMEnc


# ------------------------------------------------------------------- training

class Gather:
    """Batched window gather. (N, T, C) layout keeps a symbol's path local."""

    def __init__(self, panel, zrow, day, sym):
        self.p = np.ascontiguousarray(panel.transpose(1, 0, 2))
        self.z = zrow[day].astype(np.int64)
        self.s = sym.astype(np.int64)
        self.off = np.arange(-SEQ_LEN + 1, 1, dtype=np.int64)

    def __call__(self, idx):
        li = np.clip(self.z[idx][:, None] + self.off, 0, None)
        return self.p[self.s[idx][:, None], li]


def train_one(torch, nn, Model, gat, y, seed, epochs, dev, log):
    torch.manual_seed(seed)
    net = Model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    n = len(y)
    yt = torch.as_tensor(y)
    for ep in range(epochs):
        net.train()
        perm = np.random.default_rng(seed * 100 + ep).permutation(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            xb = torch.from_numpy(gat(b)).to(dev)
            yb = yt[b].to(dev)
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb)[0], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        log(f"    seed {seed} epoch {ep} mse {tot/n:.3e} "
            f"({time.time()-t0:.0f}s)")
    return net


def predict(torch, net, gat, n, dev, want_att, year_idx, n_years):
    """Scores plus, for the attention model, lag mass accumulated on the fly.

    Second moments are accumulated alongside the mean so the profile can be
    tested for input dependence: an attention head that ignores its input
    produces the same vector for every sample, i.e. zero cross-sample spread.
    """
    net.eval()
    out = np.empty(n, dtype=np.float32)
    by_lag = np.zeros(SEQ_LEN, dtype=np.float64)
    sq_lag = np.zeros(SEQ_LEN, dtype=np.float64)
    by_year = np.zeros((n_years, SEQ_LEN), dtype=np.float64)
    yr_cnt = np.zeros(n_years, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, 16384):
            idx = np.arange(i, min(i + 16384, n))
            xb = torch.from_numpy(gat(idx)).to(dev)
            p, a = net(xb, want_att=want_att)
            out[idx] = p.float().cpu().numpy()
            if want_att:
                aw = a.float().cpu().numpy().astype(np.float64)
                by_lag += aw.sum(0)
                sq_lag += (aw ** 2).sum(0)
                np.add.at(by_year, year_idx[idx], aw)
                np.add.at(yr_cnt, year_idx[idx], 1.0)
    if not want_att:
        return out, None
    # lag 0 = day t: the encoder's last step is the most recent session.
    mean = by_lag / n
    sd = np.sqrt(np.maximum(sq_lag / n - mean ** 2, 0.0))
    att = {"by_lag": mean[::-1].copy(), "sd_by_lag": sd[::-1].copy(),
           "by_year": (by_year / np.maximum(yr_cnt, 1)[:, None]
                       )[:, ::-1].copy()}
    return out, att


def attention_shuffle_control(torch, net, gat, n, dev, seed, n_sub=60000):
    """Attention profile when the window's time order is destroyed.

    If the head reads content, scrambling the sequence must move the profile.
    If the profile survives intact, what we are plotting is the encoder's
    positional dynamics, not a timescale the model found in the data.
    """
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, min(n_sub, n), replace=False)
    net.eval()
    acc = np.zeros(SEQ_LEN, dtype=np.float64)
    seen = 0
    with torch.no_grad():
        for i in range(0, len(sub), 16384):
            idx = sub[i:i + 16384]
            xb = gat(idx)
            perm = rng.permutation(SEQ_LEN)
            xb = np.ascontiguousarray(xb[:, perm, :])
            _, a = net(torch.from_numpy(xb).to(dev), want_att=True)
            acc += a.float().cpu().numpy().astype(np.float64).sum(0)
            seen += len(idx)
    return (acc / seen)[::-1].copy()


# ------------------------------------------------------------------- analysis

def local_maxima(v, min_prom_frac=0.05):
    """Strict local maxima with a prominence floor, endpoints included.

    A peak at lag 0 or lag 59 has only one flank, so its prominence is scored
    against that flank alone; scoring it against both would zero it out and
    silently drop the most interesting case (mass piling up at the edge).
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    rng = float(v.max() - v.min())
    floor = min_prom_frac * rng if rng > 0 else 0.0
    out = []
    for i in range(n):
        lo = v[i - 1] if i > 0 else -np.inf
        hi = v[i + 1] if i < n - 1 else -np.inf
        if not (v[i] > lo and v[i] > hi):
            continue
        keys = []
        for step in (-1, 1):
            j, m = i + step, np.inf
            while 0 <= j < n and v[j] < v[i]:
                m = min(m, v[j])
                j += step
            if np.isfinite(m):
                keys.append(m)
        prom = v[i] - max(keys) if keys else rng
        if prom >= floor:
            out.append({"lag": int(i), "weight": round(float(v[i]), 5),
                        "prominence": round(float(prom), 5)})
    return sorted(out, key=lambda d: -d["weight"])


def win_stats(r):
    return {"2019-2021": stats(r.loc["2019":"2021"]),
            "2022-2023": stats(r.loc["2022":"2023"]),
            "2024-2026": stats(r.loc["2024":]),
            "full": stats(r)}


# ---------------------------------------------------------------------- charts

def svg_attention_profile(v, maxima, sd=None, ctrl=None):
    W, H = 880, 300
    L, R, TP, BT = 62, 68, 34, 46
    v = np.asarray(v, dtype=float)
    unif = 1.0 / SEQ_LEN
    lo_b = v - sd if sd is not None else v
    hi_b = v + sd if sd is not None else v
    y0 = min(v.min(), unif, float(np.min(lo_b))) * 0.94
    y1 = max(v.max(), unif, float(np.max(hi_b))) * 1.06

    def px(i):
        return L + i / (SEQ_LEN - 1) * (W - L - R)

    def py(u):
        return H - BT - (u - y0) / (y1 - y0) * (H - TP - BT)

    body = []
    if sd is not None:
        # The band is the cross-sample spread: if it is invisible, the head
        # returned the same vector for every one of the 495k samples.
        ring = ([f"{px(i):.1f},{py(u):.1f}" for i, u in enumerate(hi_b)]
                + [f"{px(i):.1f},{py(u):.1f}"
                   for i, u in reversed(list(enumerate(lo_b)))])
        body.append(f'<polygon points="{" ".join(ring)}" fill="var(--up)" '
                    f'opacity="0.18"/>')
    for g in np.linspace(y0, y1, 5):
        gy = py(g)
        body.append(f'<line x1="{L}" x2="{W-R}" y1="{gy:.0f}" y2="{gy:.0f}" '
                    f'stroke="var(--hair)"/>'
                    f'<text x="{L-8}" y="{gy+4:.0f}" text-anchor="end" '
                    f'fill="var(--dim)" font-size="11">{g*100:.2f}%</text>')
    uy = py(unif)
    body.append(f'<line x1="{L}" x2="{W-R}" y1="{uy:.1f}" y2="{uy:.1f}" '
                f'stroke="var(--dim)" stroke-width="1.2" '
                f'stroke-dasharray="4 4"/>'
                f'<text x="{W-R+6}" y="{uy+4:.0f}" fill="var(--dim)" '
                f'font-size="11">均匀 {unif*100:.2f}%</text>')
    if ctrl is not None:
        cpts = " ".join(f"{px(i):.1f},{py(u):.1f}"
                        for i, u in enumerate(np.asarray(ctrl, dtype=float)))
        body.append(f'<polyline points="{cpts}" fill="none" '
                    f'stroke="var(--muted)" stroke-width="1.6" '
                    f'stroke-dasharray="5 3" opacity=".9"/>')
    pts = " ".join(f"{px(i):.1f},{py(u):.1f}" for i, u in enumerate(v))
    body.append(f'<polyline points="{pts}" fill="none" stroke="var(--up)" '
                f'stroke-width="2.4"/>')
    for m in maxima[:6]:
        i, u = m["lag"], m["weight"]
        body.append(f'<circle cx="{px(i):.1f}" cy="{py(u):.1f}" r="4" '
                    f'fill="var(--gold)"/>'
                    f'<text x="{px(i):.1f}" y="{py(u)-10:.0f}" '
                    f'text-anchor="middle" fill="var(--gold)" font-size="11">'
                    f'lag {i}</text>')
    for i in (0, 10, 20, 30, 40, 50, 59):
        body.append(f'<text x="{px(i):.0f}" y="{H-12}" text-anchor="middle" '
                    f'fill="var(--dim)" font-size="11">{i}</text>')
    body.append(f'<text x="{L}" y="16" font-size="12" fill="var(--muted)">'
                f'注意力权重按滞后期分布（lag 0 = 当日 t，lag 59 = 60 个交易日前）'
                f'</text>')
    body.append(
        f'<text x="{L}" y="{H-16}" font-size="11">'
        f'<tspan fill="var(--up)">■</tspan>'
        f'<tspan fill="var(--muted)"> 实测均值（浅带 = 样本间 ±1σ）   </tspan>'
        f'<tspan fill="var(--muted)">- - 打乱时序对照   </tspan>'
        f'<tspan fill="var(--dim)">- - 均匀基准</tspan></text>')
    body.append(f'<text x="{W//2}" y="{H-1}" text-anchor="middle" '
                f'fill="var(--muted)" font-size="11">滞后交易日</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img">{"".join(body)}</svg>'


def svg_attention_heatmap(years, M):
    W = 880
    L, R, TP = 62, 78, 30
    rh = 26
    H = TP + len(years) * rh + 30
    M = np.asarray(M, dtype=float)
    lo, hi = float(M.min()), float(M.max())
    cw = (W - L - R) / SEQ_LEN
    body = []
    for j, yr in enumerate(years):
        y = TP + j * rh
        body.append(f'<text x="{L-8}" y="{y+rh/2+4:.0f}" text-anchor="end" '
                    f'fill="var(--ink)" font-size="12">{yr}</text>')
        for i in range(SEQ_LEN):
            o = 0.06 + 0.94 * ((M[j, i] - lo) / (hi - lo) if hi > lo else 0.5)
            body.append(f'<rect x="{L+i*cw:.2f}" y="{y+2}" '
                        f'width="{cw+0.4:.2f}" height="{rh-4}" '
                        f'fill="var(--gold)" opacity="{o:.3f}"/>')
        pk = int(np.argmax(M[j]))
        body.append(f'<text x="{W-R+6}" y="{y+rh/2+4:.0f}" '
                    f'fill="var(--muted)" font-size="11">峰值 lag {pk}</text>')
    for i in (0, 10, 20, 30, 40, 50, 59):
        body.append(f'<text x="{L+(i+0.5)*cw:.0f}" y="{H-12}" '
                    f'text-anchor="middle" fill="var(--dim)" font-size="11">'
                    f'{i}</text>')
    # Shared absolute scale across rows on purpose: rescaling each row would
    # turn a year-on-year difference of ~1e-5 into fake visible structure.
    body.append(f'<text x="{L}" y="16" font-size="12" fill="var(--muted)">'
                f'逐年注意力热力图 — 统一绝对色标（{lo*100:.2f}% → '
                f'{hi*100:.2f}%），逐年剖面几乎完全重合</text>')
    body.append(f'<text x="{W//2}" y="{H-1}" text-anchor="middle" '
                f'fill="var(--muted)" font-size="11">滞后交易日</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img">{"".join(body)}</svg>'


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    def log(m):
        print(m, flush=True)

    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    y = np.load(f"{CACHE}/y_res.npy")

    meta = json.load(open(f"{ZOO}/meta.json"))
    zt = {np.datetime64(d, "D"): i for i, d in enumerate(meta["dates"])}
    zrow = np.array([zt[d] for d in dates])

    t0 = time.time()
    panel = build_panel(syms)
    log(f"panel {panel.shape} built in {time.time()-t0:.0f}s "
        f"({panel.nbytes/1e6:.0f}MB)")

    if a.smoke:
        keep = np.arange(0, len(y), 20)
        day, sym, y = day[keep], sym[keep], y[keep]
    gat = Gather(panel, zrow, day, sym)
    n = len(y)

    row_year = pd.DatetimeIndex(dates[day]).year.to_numpy()
    years = sorted(set(row_year.tolist()))
    yidx = np.searchsorted(np.array(years), row_year)

    LSTMAttn, TFMEnc = make_models(torch, nn)
    log(f"device {dev}  rows {n}  channels {N_CH}  seq {SEQ_LEN}")

    results, panels = {}, {}
    att_seeds, ctrl_seeds, nparams = [], [], {}
    for tag, Model, want_att in (("lstm", LSTMAttn, True),
                                 ("tfm", TFMEnc, False)):
        acc = np.zeros(n, dtype=np.float64)
        nparams[tag] = sum(p.numel() for p in Model().parameters())
        log(f"  {tag}: {nparams[tag]} params")
        for s in range(a.seeds):
            net = train_one(torch, nn, Model, gat, y, s, a.epochs, dev, log)
            sc, att = predict(torch, net, gat, n, dev, want_att,
                              yidx, len(years))
            acc += sc
            if want_att:
                att_seeds.append(att)
                ctrl_seeds.append(attention_shuffle_control(
                    torch, net, gat, n, dev, s))
        P = np.full((len(dates), len(syms)), np.nan, np.float32)
        P[day, sym] = (acc / a.seeds).astype(np.float32)
        panels[tag] = P
        r = run_topk(P, G, O2O, dates)
        results[tag] = win_stats(r.dropna())
        log(f"  {tag} full {results[tag]['full']}")

    by_lag = np.mean([d["by_lag"] for d in att_seeds], axis=0)
    sd_lag = np.mean([d["sd_by_lag"] for d in att_seeds], axis=0)
    by_year = np.mean([d["by_year"] for d in att_seeds], axis=0)
    ctrl = np.mean(ctrl_seeds, axis=0)
    maxima = local_maxima(by_lag)

    # Does the head read content, or only position? Two independent tests.
    cv = float(np.mean(sd_lag / by_lag))
    yr_spread = float(np.abs(by_year - by_lag[None, :]).max())
    ctrl_tv = float(0.5 * np.abs(ctrl - by_lag).sum())
    log(f"  attention cross-sample CV {cv:.4f}  year spread {yr_spread:.2e}  "
        f"shuffle-control TV {ctrl_tv:.4f}")

    os.makedirs(OUT, exist_ok=True)
    att = {
        "lag_convention": "lag 0 = day t (latest session), lag 59 = 60 "
                          "sessions back",
        "n_samples": int(n), "seeds": a.seeds, "epochs": a.epochs,
        "uniform_weight": 1.0 / SEQ_LEN,
        "by_lag": [round(float(v), 6) for v in by_lag],
        "sd_by_lag": [round(float(v), 7) for v in sd_lag],
        "by_lag_per_seed": [[round(float(v), 6) for v in d["by_lag"]]
                            for d in att_seeds],
        "years": [int(v) for v in years],
        "by_year": [[round(float(v), 6) for v in row] for row in by_year],
        "local_maxima": maxima,
        "mass_lag0_4": round(float(by_lag[:5].sum()), 4),
        "mass_lag0_20": round(float(by_lag[:21].sum()), 4),
        "channels": CHANNELS,
        "input_dependence": {
            "cross_sample_cv": round(cv, 5),
            "max_abs_year_deviation": yr_spread,
            "time_shuffle_control_by_lag": [round(float(v), 6) for v in ctrl],
            "time_shuffle_control_tv_distance": round(ctrl_tv, 5),
            "seed_correlation": round(float(np.corrcoef(
                att_seeds[0]["by_lag"], att_seeds[-1]["by_lag"])[0, 1]), 4)
            if len(att_seeds) > 1 else None,
            "reading": "cross_sample_cv near 0, a year deviation at the "
                       "rounding floor and a shuffle-control TV near 0 all "
                       "say the same thing: the profile is the encoder's "
                       "positional dynamics, not a timescale read off the "
                       "data. Do not narrate it as a discovered horizon.",
        },
    }
    with open(f"{OUT}/seq_attention.json", "w") as fh:
        json.dump(att, fh, ensure_ascii=False, indent=1)
    pd.DataFrame(by_year, index=pd.Index(years, name="year"),
                 columns=[f"lag{i}" for i in range(SEQ_LEN)]
                 ).to_parquet(f"{OUT}/seq_attention.parquet")

    curves = {}
    navs = {}
    for tag, P in panels.items():
        r = run_topk(P, G, O2O, dates).dropna()
        curves[tag] = r
        navs[tag] = (1 + r).cumprod()
    pd.DataFrame(navs).to_parquet(f"{OUT}/seq_nav.parquet")
    with open(f"{OUT}/seq_curves.json", "w") as fh:
        json.dump({"convention": "full-sample fit, signal t, fill t+1 open, "
                                 "open-to-open marks, 0.5bp/side, Top-4 EW",
                   "seq_len": SEQ_LEN, "channels": CHANNELS,
                   "seeds": a.seeds, "epochs": a.epochs, "device": dev,
                   "params": nparams,
                   "lstm": results["lstm"], "tfm": results["tfm"],
                   "attention_is_input_dependent": bool(cv > 0.02),
                   "attention_cross_sample_cv": round(cv, 5)},
                  fh, ensure_ascii=False, indent=1)

    with open(f"{OUT}/seq_charts.svg", "w") as fh:
        fh.write(svg_attention_profile(by_lag, maxima, sd_lag, ctrl)
                 + "\n<!--SPLIT-->\n"
                 + svg_attention_heatmap(years, by_year))

    summary = {"lstm": results["lstm"], "tfm": results["tfm"],
               "attention_local_maxima_lags": [m["lag"] for m in maxima],
               "attention_by_lag": [round(float(v), 4) for v in by_lag],
               "attention_input_dependence": {
                   "cross_sample_cv": round(cv, 5),
                   "max_abs_year_deviation": yr_spread,
                   "time_shuffle_control_tv_distance": round(ctrl_tv, 5)}}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
