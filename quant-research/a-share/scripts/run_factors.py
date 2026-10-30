"""批量因子检验运行器(多进程,append-only 账本)。

    python scripts/run_factors.py --workers 5 [--family X] [--limit N] [--mode open]
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor.fast_eval import EvalContext, evaluate
from afactor import factors as FA
from afactor import ops

_G = {}


def _init(mode, horizons, tag=""):
    # 尾盘口径必须用 TailPanel:它把全日字段自动滞后一天、当日字段换成 14:50 截断版,
    # 否则用收盘价算出的因子在 14:50 成交就是前视。
    p = TailPanel() if mode == "tail" else Panel()
    _G["panel"] = p
    _G["ctx"] = EvalContext(p, mode, horizons, tag=tag)
    FA.build_all()


def _run(names):
    p, ctx = _G["panel"], _G["ctx"]
    out = []
    lp = LEDGER / f"trials_{os.getpid()}.jsonl"
    for name in names:
        family, fn, note = FA.REGISTRY[name]
        try:
            t0 = time.time()
            f = fn(p)
            if not isinstance(f, pd.DataFrame):
                raise TypeError(f"返回类型 {type(f)}")
            f = f.reindex(index=p.dates, columns=p.codes)
            f = f.replace([np.inf, -np.inf], np.nan)
            arr = f.to_numpy(np.float32)
            build_s = time.time() - t0
            r = evaluate(name, arr, ctx, family=family, note=note, ledger_path=lp)
            r["build_s"] = round(build_s, 2)
            out.append((name, r.get("ic_mean"), r.get("ic_t"), None))
        except Exception as e:
            with open(lp, "a") as fh:
                fh.write(json.dumps({"factor": name, "family": family, "error":
                                     f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]},
                                    ensure_ascii=False) + "\n")
            out.append((name, None, None, f"{type(e).__name__}: {e}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--family", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="open")
    ap.add_argument("--horizons", default="1,5,10")
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--tag", default="", help="账本 mode 标签后缀(变体运行防覆盖)")
    a = ap.parse_args()
    hor = tuple(int(x) for x in a.horizons.split(","))

    FA.build_all()
    names = [n for n, (fam, _, _) in FA.REGISTRY.items() if a.family is None or fam in a.family.split(",")]
    if a.limit:
        names = names[:a.limit]
    print(f"待检验 {len(names)} 个因子 | mode={a.mode} tag={a.tag} horizons={hor} workers={a.workers}", flush=True)

    chunks = [names[i:i + a.chunk] for i in range(0, len(names), a.chunk)]
    t0 = time.time(); done = 0; errs = 0; passed = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.mode, hor, a.tag)) as ex:
        futs = [ex.submit(_run, c) for c in chunks]
        for fu in as_completed(futs):
            for name, ic, t, err in fu.result():
                done += 1
                if err:
                    errs += 1
                    print(f"  [ERR] {name}: {err}", flush=True)
                elif ic is not None and abs(ic) >= 0.025:
                    passed += 1
            el = time.time() - t0
            print(f"[{done}/{len(names)}] {el:.0f}s  过初筛={passed} 报错={errs} "
                  f"(剩余约 {el/max(done,1)*(len(names)-done):.0f}s)", flush=True)
    print(f"完成 {done} 个,{time.time()-t0:.0f}s,过 IC 初筛 {passed},报错 {errs}")


if __name__ == "__main__":
    main()
