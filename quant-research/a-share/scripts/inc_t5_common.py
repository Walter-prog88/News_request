"""T5 公共设施:路径、日志、窗口、偏离登记、数据加载上下文(前缀 inc_t5;预注册 research/INC_T5_PREREG.md)。"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor import lh_engine as LE
from afactor.config import LEDGER, TMP

LOG = ROOT / "logs" / "inc_t5_run.log"
SC = TMP / "inc_t5_scores"
RUNS = TMP / "inc_t5_runs"
FETCH = TMP / "inc_t5_fetch"
DEVS = LEDGER / "inc_t5_deviations.csv"
QA = LEDGER / "inc_t5_data_qa.csv"
TD = 242

# ---------------- 窗口(预注册 §8)----------------
WIN = {
    "lh": dict(rows_from="2016-01-04", full=("2017-01-03", "2025-12-31"), dev=("2017-01-03", "2019-12-31"),
               test=("2020-01-01", "2025-12-31"), dev2=("2017-01-03", "2020-12-31"), test2=("2021-01-01", "2025-12-31")),
    "a": dict(rows_from=None, full=("2006-01-04", "2025-12-31"), dev=("2006-01-04", "2019-12-31"),
              test=("2020-01-01", "2025-12-31"), dev2=("2006-01-04", "2014-12-31"), test2=("2015-01-01", "2025-12-31")),
    "min": dict(rows_from=None, full=("2021-01-04", "2025-12-31"), dev=("2021-01-04", "2023-12-31"),
                test=("2024-01-01", "2025-12-31")),
    "c": dict(rows_from=None, full=("2007-01-04", "2016-12-30")),
}
WIN_NAMES = ("full", "dev", "test")


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def log_dev(stage: str, item: str, prereg_text: str, decision: str, reason: str, before_returns: bool = True):
    """偏离 / 口径澄清登记(追加,按 item 去重)。before_returns=False 的必须写明"看到结果后",且不进判定。"""
    cols = ["stage", "item", "prereg_text", "decision", "reason", "decided_before_returns", "logged_at"]
    old = pd.read_csv(DEVS) if DEVS.exists() and DEVS.stat().st_size > 5 else pd.DataFrame(columns=cols)
    if "logged_at" not in old.columns:
        old["logged_at"] = ""
    if (old["item"] == item).any():
        return
    row = dict(stage=stage, item=item, prereg_text=prereg_text, decision=decision, reason=reason,
               decided_before_returns=bool(before_returns), logged_at=time.strftime("%Y-%m-%d %H:%M:%S JST"))
    pd.concat([old, pd.DataFrame([row])], ignore_index=True)[cols].to_csv(DEVS, index=False)


def qa_row(gate: str, item: str, value, threshold: str, ok, note: str = "") -> dict:
    return dict(gate=gate, item=item, value=value, threshold=threshold, ok=ok, note=note)


def merge_csv(path: Path, df: pd.DataFrame, key: str | list):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    keys = [key] if isinstance(key, str) else list(key)
    if len(old) and all(k in old.columns for k in keys):
        m = old.merge(df[keys].drop_duplicates(), on=keys, how="left", indicator=True)
        old = old[(m["_merge"] == "left_only").to_numpy()]
    pd.concat([old, df], ignore_index=True).to_csv(path, index=False)


def verify_prereg() -> bool:
    man = json.loads((LEDGER / "inc_t5_prereg_manifest.json").read_text())
    bad = [f for f, h in man["sha256"].items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    return not bad


# ---------------- 数据上下文 ----------------
class LHCtx:
    """长历史上下文:LongPanel 落盘输入 + 适配器 + 池 + B1。rows_from=None -> 全 5101 行(A 轨)。"""

    def __init__(self, rows_from: str | None):
        self.X = LE.load_inputs()
        self.dates_all = self.X.dates
        self.codes = self.X.codes
        i0 = 0 if rows_from is None else int(self.dates_all.searchsorted(pd.Timestamp(rows_from)))
        self.i0 = i0
        self.rows = slice(i0, None) if i0 > 0 else None
        self.ad = LE.adapter(self.X, rows=self.rows)
        self.dates = self.ad.dates
        self.U = LE.pool(self.X, "q92", self.rows)
        self._b1 = None

    def b1(self) -> pd.Series:
        if self._b1 is None:
            self._b1 = LE.bench(self.ad, self.U, 5)
        return self._b1

    def score(self, name: str) -> pd.DataFrame:
        a = np.load(SC / f"lh_{name}.npy", mmap_mode="r")
        a = a[self.i0:] if self.i0 else a
        return pd.DataFrame(np.asarray(a), index=self.dates, columns=self.codes, copy=False)

    def weights(self, score: pd.DataFrame, buffer) -> np.ndarray:
        return LE.weights(self.X, score=score, buffer=buffer, rows=self.rows)

    def weights_U(self, score: pd.DataFrame, buffer, U: pd.DataFrame) -> np.ndarray:
        """同 weights,但决策池由调用方给定(C 轨的 EM 覆盖池)。"""
        from afactor import strategy as ST
        shape = LE._Shape(U.index, U.columns)
        return ST.build_weights(score, shape, "V1000", scheme="exclude", exclude_frac=0.3, buffer=buffer,
                                universe_override=U)

    def liq_weights(self, buffer=(0.20, 0.35)) -> np.ndarray:
        return LE.weights(self.X, "S1", buffer=buffer, rows=self.rows)

    def rand_score(self, seed: int) -> pd.DataFrame:
        s = np.random.default_rng(seed).random(len(self.codes)).astype(np.float32)
        return pd.DataFrame(np.tile(s, (len(self.dates), 1)), index=self.dates, columns=self.codes)

    def run(self, W: np.ndarray, cost: str = "Y1", record: bool = False) -> dict:
        return LE.run(W, self.ad, rebalance=5, cost=cost, record=record)

    def fullpool_W(self) -> np.ndarray:
        Ua = self.U.to_numpy(bool)
        n = Ua.sum(1, keepdims=True)
        return np.divide(Ua.astype(np.float32), n, out=np.zeros(Ua.shape, np.float32), where=n > 0)


def slim(res: dict) -> dict:
    """落盘用的精简结果。"""
    keep = ("rets", "phase_rets", "turnover", "cost_bps", "exposure", "holdings", "n_days")
    out = {k: res[k] for k in keep if k in res}
    if "record" in res:
        rec = res["record"]
        cols = ["ret", "gross_ret", "turnover", "cost_total", "cost_commission", "cost_stamp", "cost_spread",
                "cost_impact", "exposure", "cash", "n_hold", "stuck_w", "unfilled", "part_cut"]
        out["rec_daily"] = rec["daily"][[c for c in cols if c in rec["daily"].columns]]
        out["max_exposure_phase"] = float(rec["daily_by_phase"]["exposure"].max())
    return out
