"""Train one model family on the showcase matrix and emit its score panel.

  --model gbdt       HistGradientBoosting on residual returns, 5-seed bagging
  --model gbdt_rank  same machinery, label = within-day percentile rank
  --model mlp        torch MLP, per-day-ranked features, 3-seed ensemble

Full-sample fit by design (upper-bound convention, documented in the report).
Scores are written as a (T, N) panel aligned to the matrix manifest.
"""

import argparse
import json

import numpy as np
import os as _os
# Default to a persistent root: the old fallback pointed at the research
# session's scratch dir, which the OS purges, taking every cached panel
# with it. rebuild_caches.sh regenerates this tree bit-identically.
_SCRATCH = _os.environ.get(
    "INDEX_CACHE_ROOT", _os.path.expanduser("~/index-cache"))

SCRATCH = _SCRATCH
CACHE = f"{SCRATCH}/showcase_cache"


def load():
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    man = json.load(open(f"{CACHE}/manifest.json"))
    return X, man


def panelize(scores: np.ndarray, man) -> np.ndarray:
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    P = np.full((len(man["dates"]), len(man["symbols"])), np.nan,
                dtype=np.float32)
    P[day, sym] = scores
    return P


def train_gbdt(X, y, seeds=5, frac=0.8):
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.float64)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        idx = rng.choice(n, int(n * frac), replace=False)
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=200, l2_regularization=1.0, random_state=s)
        m.fit(np.asarray(X[idx]), y[idx])
        preds += m.predict(np.asarray(X))
        print(f"seed {s} done", flush=True)
    return (preds / seeds).astype(np.float32)


def train_mlp(X, y, seeds=3, epochs=8):
    import torch
    import torch.nn as nn
    dev = ("mps" if torch.backends.mps.is_available() else "cpu")
    Xf = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.5)
    yt = torch.tensor(y, dtype=torch.float32)
    Xt = torch.tensor(Xf)
    n = Xt.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for s in range(seeds):
        torch.manual_seed(s)
        net = nn.Sequential(
            nn.Linear(Xt.shape[1], 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, 8192):
                b = perm[i:i + 8192]
                xb, yb = Xt[b].to(dev), yt[b].to(dev)
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            print(f"seed {s} epoch {ep} loss {tot/n:.6f}", flush=True)
        net.eval()
        with torch.no_grad():
            for i in range(0, n, 65536):
                out[i:i + 65536] += (net(Xt[i:i + 65536].to(dev))
                                     .squeeze(-1).cpu().numpy())
    return (out / seeds).astype(np.float32)


def oof(train_fn, X, y, day, folds=5, embargo=5, **kw):
    """Cross-fitted scores: each row predicted by a model that never saw its
    fold, with an embargo strip at the fold edges. Contiguous time blocks."""
    days = np.unique(day)
    edges = np.array_split(days, folds)
    out = np.full(X.shape[0], np.nan, dtype=np.float32)
    for k, block in enumerate(edges):
        lo, hi = block[0], block[-1]
        test = (day >= lo) & (day <= hi)
        train = ~((day >= lo - embargo) & (day <= hi + embargo))
        print(f"fold {k}: train {train.sum()} test {test.sum()}", flush=True)
        Xtr = np.asarray(X[train])
        s = train_fn(Xtr, y[train], X_pred=np.asarray(X[test]), **kw)
        out[test] = s
    return out


def train_gbdt_fold(Xtr, ytr, X_pred=None, seeds=3, frac=0.85):
    from sklearn.ensemble import HistGradientBoostingRegressor
    preds = np.zeros(X_pred.shape[0], dtype=np.float64)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        idx = rng.choice(len(Xtr), int(len(Xtr) * frac), replace=False)
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=200, l2_regularization=1.0, random_state=s)
        m.fit(Xtr[idx], ytr[idx])
        preds += m.predict(X_pred)
    return (preds / seeds).astype(np.float32)


def train_mlp_fold(Xtr, ytr, X_pred=None, seeds=2, epochs=6):
    import torch
    import torch.nn as nn
    dev = ("mps" if torch.backends.mps.is_available() else "cpu")
    Xtr = torch.tensor(np.nan_to_num(Xtr, nan=0.5))
    Xp = torch.tensor(np.nan_to_num(X_pred, nan=0.5))
    yt = torch.tensor(ytr, dtype=torch.float32)
    n = Xtr.shape[0]
    out = np.zeros(Xp.shape[0], dtype=np.float64)
    for s in range(seeds):
        torch.manual_seed(s)
        net = nn.Sequential(
            nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 8192):
                b = perm[i:i + 8192]
                xb, yb = Xtr[b].to(dev), yt[b].to(dev)
                opt.zero_grad()
                nn.functional.mse_loss(net(xb).squeeze(-1), yb).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            for i in range(0, len(Xp), 65536):
                out[i:i + 65536] += (net(Xp[i:i + 65536].to(dev))
                                     .squeeze(-1).cpu().numpy())
    return (out / seeds).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["gbdt", "gbdt_rank", "mlp"])
    ap.add_argument("--oof", action="store_true")
    a = ap.parse_args()
    X, man = load()
    day = np.load(f"{CACHE}/row_day.npy")
    if a.oof:
        y = np.load(f"{CACHE}/y_rank.npy" if a.model == "gbdt_rank"
                    else f"{CACHE}/y_res.npy")
        fn = train_mlp_fold if a.model == "mlp" else train_gbdt_fold
        s = oof(fn, X, y, day)
        tag = f"{a.model}_oof"
    elif a.model == "gbdt":
        y = np.load(f"{CACHE}/y_res.npy")
        s = train_gbdt(X, y)
        tag = a.model
    elif a.model == "gbdt_rank":
        y = np.load(f"{CACHE}/y_rank.npy")
        s = train_gbdt(X, y)
        tag = a.model
    else:
        y = np.load(f"{CACHE}/y_res.npy")
        s = train_mlp(X, y)
        tag = a.model
    np.save(f"{CACHE}/score_{tag}.npy", panelize(s, man))
    # quick in-sample rank IC by day
    day = np.load(f"{CACHE}/row_day.npy")
    yy = np.load(f"{CACHE}/y_res.npy")
    ics = []
    for d in np.unique(day)[::10]:
        m = day == d
        if m.sum() > 30:
            a1 = np.argsort(np.argsort(s[m]))
            a2 = np.argsort(np.argsort(yy[m]))
            ics.append(np.corrcoef(a1, a2)[0, 1])
    print(f"{a.model}: mean in-sample rank IC = {np.mean(ics):.4f}")


if __name__ == "__main__":
    main()
