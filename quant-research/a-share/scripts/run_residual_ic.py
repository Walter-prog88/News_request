"""对所有通过一致性关的因子计算"对规模/流动性代理正交化后"的残差 IC。

结论用于把候选因子分为「独立信号」与「规模马甲」两类;按用户口径,残差 IC 只报告、不淘汰。
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor.fast_eval import EvalContext, evaluate
from afactor import factors as FA, ops, screen

_G = {}


def _init(mode):
    p = Panel()
    _G["p"] = p
    _G["ctx"] = EvalContext(p, mode, (1,))
    _G["size"] = ops.cs_rank(ops.signed_log(p["adv20"]))
    FA.build_all()


def _run(names):
    p, ctx, size = _G["p"], _G["ctx"], _G["size"]
    out = []
    lp = LEDGER / f"resid_{os.getpid()}.jsonl"
    for n in names:
        try:
            fam, fn, note = FA.REGISTRY[n]
            f = fn(p).reindex(index=p.dates, columns=p.codes).replace([np.inf, -np.inf], np.nan)
            resid = ops.cs_neutralize_1d(f, size)
            r = evaluate(f"RESID__{n}", resid.to_numpy(np.float32), ctx, family=fam,
                         note=f"对adv20规模代理正交化 | {note}", save_ic=False, ledger_path=lp)
            out.append((n, r.get("ic_mean"), r.get("ic_t"), r.get("mono_spearman")))
        except Exception as e:
            out.append((n, None, None, f"ERR {type(e).__name__}: {e}"))
    return out


def main():
    mode = "open"
    df = screen.load_ledger("trials_[0-9]*.jsonl")
    cand = screen.gate_consistency(screen.gate_stats(df))
    names = [n for n in cand["factor"].tolist() if n in FA.build_all()]
    print(f"待算残差 IC:{len(names)} 个", flush=True)
    chunks = [names[i:i + 10] for i in range(0, len(names), 10)]
    rows = []
    with ProcessPoolExecutor(max_workers=5, initializer=_init, initargs=(mode,)) as ex:
        futs = [ex.submit(_run, c) for c in chunks]
        for i, fu in enumerate(as_completed(futs), 1):
            for n, ic, t, mono in fu.result():
                rows.append({"factor": n, "resid_ic": ic, "resid_t": t, "resid_mono": mono})
            print(f"  [{len(rows)}/{len(names)}]", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / "residual_ic.csv", index=False)
    print(out.describe().to_string())


if __name__ == "__main__":
    main()
