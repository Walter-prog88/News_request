"""Gradient saliency for the sequence model, replacing the dead attention read.

The attention head trained in seq_models.py emits a near-constant vector for
every sample (cross-sample CV 0.6%, time-shuffle TV distance 3e-5), so its
profile describes the LSTM's positional dynamics, not anything read off the
data. Input gradients do not have that failure mode: d(score)/d(input) is a
function of the sample by construction, and a shuffle control can be run
against it the same way.

Reports, per lag and per channel, mean |d score / d input| times the input's
own scale, which is the standard input-times-gradient attribution. Saves the
score panel too — seq_models.py kept it in memory only.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1")

import seq_models as S  # noqa: E402
from sweep_dial import CACHE, load_env, run_topk, stats  # noqa: E402

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
EPOCHS = 8
SEED = 0
SAL_SAMPLES = 40000


def main():
    import torch
    import torch.nn as nn

    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    y = np.load(f"{CACHE}/y_res.npy")
    meta = json.load(open(f"{S.ZOO}/meta.json"))
    zt = {np.datetime64(d, "D"): i for i, d in enumerate(meta["dates"])}
    zrow = np.array([zt[d] for d in dates])
    P = S.build_panel(syms)                       # (T, N, C) float32
    gat = S.Gather(P, zrow, day, sym)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"panel {P.shape} rows {len(y)} device {dev}", flush=True)

    LSTMAttn, _ = S.make_models(torch, nn)
    net = S.train_one(torch, nn, LSTMAttn, gat, y, SEED, EPOCHS, dev,
                      lambda m: print(m, flush=True))

    # persist what the builder left in memory, so the panel is auditable
    # predict() allocates its per-year attention buffers unconditionally, so
    # it needs a real n_years even when attention is not wanted
    scores = S.predict(torch, net, gat, len(y), dev, False,
                       np.zeros(len(y), dtype=np.int64), 1)
    s = scores[0] if isinstance(scores, tuple) else scores
    panel = np.full((len(dates), len(syms)), np.nan, np.float32)
    panel[day, sym] = np.asarray(s, dtype=np.float32).ravel()
    np.save(f"{CACHE}/score_lstm_panel.npy", panel)
    r = run_topk(panel, G, O2O, dates)
    print("panel sharpe:", stats(r.dropna())["sharpe"], flush=True)

    # input x gradient attribution on a random subset
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(y), SAL_SAMPLES, replace=False))
    net.eval()
    acc = np.zeros((S.SEQ_LEN, len(S.CHANNELS)), dtype=np.float64)
    ctrl = np.zeros_like(acc)
    n_done = 0
    for i in range(0, len(idx), 2048):
        b = idx[i:i + 2048]
        xb = torch.tensor(gat(b), dtype=torch.float32, device=dev)
        xb.requires_grad_(True)
        out = net(xb)
        out = out[0] if isinstance(out, tuple) else out
        out.sum().backward()
        acc += (xb.grad * xb).abs().sum(dim=0).detach().cpu().numpy()
        # control: same magnitudes with the time axis scrambled. If the model
        # were time-blind this would match the real profile.
        perm = torch.randperm(S.SEQ_LEN, device=dev)
        xs = xb.detach()[:, perm, :].clone().requires_grad_(True)
        o2 = net(xs)
        o2 = o2[0] if isinstance(o2, tuple) else o2
        o2.sum().backward()
        ctrl += (xs.grad * xs).abs().sum(dim=0).detach().cpu().numpy()
        n_done += len(b)
    acc /= n_done
    ctrl /= n_done

    by_lag = acc.sum(axis=1)
    by_lag = by_lag / by_lag.sum()
    ctrl_lag = ctrl.sum(axis=1)
    ctrl_lag = ctrl_lag / ctrl_lag.sum()
    by_ch = acc.sum(axis=0)
    by_ch = by_ch / by_ch.sum()

    order = np.argsort(-by_ch)
    tv = 0.5 * np.abs(by_lag - ctrl_lag).sum()
    uni = 1.0 / S.SEQ_LEN
    out = {
        "lag_convention": "lag 0 = day t (latest session), lag 59 = 60 back",
        "attribution": "mean |input x gradient|, normalised over lags",
        "by_lag": [round(float(v), 5) for v in by_lag],
        "shuffle_control_by_lag": [round(float(v), 5) for v in ctrl_lag],
        "shuffle_tv_distance": round(float(tv), 4),
        "uniform_baseline": round(uni, 5),
        "max_over_uniform_x": round(float(by_lag.max() / uni), 2),
        "mass_lag0_4": round(float(by_lag[:5].sum()), 4),
        "mass_lag0_20": round(float(by_lag[:21].sum()), 4),
        "by_channel": [{"channel": S.CHANNELS[j],
                        "share": round(float(by_ch[j]), 4)} for j in order],
        "n_samples": int(n_done),
    }
    with open(f"{OUT}/seq_saliency.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    pd.DataFrame(acc, columns=S.CHANNELS).to_parquet(
        f"{OUT}/seq_saliency_matrix.parquet")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("by_lag", "shuffle_control_by_lag",
                                   "by_channel")}, indent=1))
    print("top channels:", [(c["channel"], c["share"])
                            for c in out["by_channel"][:8]])
    print("by_lag head:", out["by_lag"][:8], "tail:", out["by_lag"][-4:])


if __name__ == "__main__":
    main()
