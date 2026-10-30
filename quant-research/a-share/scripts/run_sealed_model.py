"""模型层封存期最终考试 —— 每个模型版本只允许考一次。

与因子层的 run_sealed_smoke.py 是**两次独立的考试**:
  - 因子层锁 SEAL_LOCK 已于 2026-08-20 用掉(冻结因子清单的一次性验证)
  - 模型层用独立台账 model_seal_ledger.jsonl,按"模型版本指纹"记账

模型版本指纹 = sha256(特征集 + 模型族 + horizon + 执行口径 + 策略配置 + 是否带择时)。
同一指纹只能考一次;考完写台账,此后拒绝重跑。**看完成绩不得回头调参**——
若要改参数,那是一个新版本,需要用户显式批准再消耗一次机会。

用法(必须显式传 --confirm,防止误触):
    python scripts/run_sealed_model.py --confirm
"""
from __future__ import annotations
import os, sys, json, time, hashlib, argparse, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import DAILY, SEALED, SEAL_START, LEDGER
from afactor.panel import Panel, TailPanel
from afactor import factors as FA, mlmodel as ML, strategy as ST, screen, timing as TM

MODEL_LEDGER = LEDGER / "model_seal_ledger.jsonl"
SPEC_FILE = LEDGER / "model_final_spec.json"


class SealedPanel(Panel):
    """研究集 + 封存集拼接:滚动窗口需要 2025 年历史,但只在 2026 日期上评估。"""

    def __init__(self):
        super().__init__(sealed=False)

    def raw(self, name: str) -> pd.DataFrame:
        key = f"__cat__{name}"
        if key not in self._cache:
            a = pd.read_parquet(DAILY / f"{name}.parquet")
            f = SEALED / f"{name}.parquet"
            b = pd.read_parquet(f) if f.exists() else None
            self._cache[key] = (pd.concat([a, b]).sort_index() if b is not None
                                else a).astype(np.float32)
        return self._cache[key]


class SealedTailPanel(SealedPanel):
    _is_tail = True
    _NO_SHIFT = TailPanel._NO_SHIFT

    def raw(self, name):
        if name == "close_px":
            return self.derived("px_1450")
        if name in ("twap_tail8", "amt_tail8"):
            return SealedPanel.raw(self, name)
        if (DAILY / f"pre_{name}.parquet").exists():
            return SealedPanel.raw(self, f"pre_{name}")
        if name in self._NO_SHIFT:
            return SealedPanel.raw(self, name)
        return SealedPanel.raw(self, name).shift(1)

    def derived(self, name):
        if name == "px_1450":
            if "px_1450" not in self._derived:
                self._derived["px_1450"] = (SealedPanel.raw(self, "open_auction_px") *
                                            (1.0 + SealedPanel.raw(self, "pre_ret_intraday")))
            return self._derived["px_1450"]
        return super().derived(name)


def fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def used_before(fp: str) -> dict | None:
    if not MODEL_LEDGER.exists():
        return None
    for line in MODEL_LEDGER.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("fingerprint") == fp:
                return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="确认消耗一次封存期考试机会")
    args = ap.parse_args()

    if not SPEC_FILE.exists():
        print(f"拒绝执行:未找到冻结的模型规格 {SPEC_FILE}。\n"
              f"必须先在研究期(2020-2025)定稿并写入该文件,才允许考试。")
        sys.exit(1)
    spec = json.load(open(SPEC_FILE))
    fp = fingerprint(spec)
    prev = used_before(fp)
    if prev:
        print(f"拒绝执行:该模型版本(指纹 {fp})已于 {prev['ts']} 考过一次。\n"
              f"结果见 {LEDGER / ('sealed_model_' + fp + '.json')}。\n"
              f"改参数 = 新版本 = 需要用户显式批准后再消耗一次机会。")
        sys.exit(1)
    print(f"模型规格(指纹 {fp}):")
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    if not args.confirm:
        print("\n未传 --confirm,只做干跑校验,不消耗考试机会。")

    mode = spec["mode"]; horizon = spec["horizon"]; kind = spec["model"]
    exec_mode = spec["exec_mode"]; fs_name = spec.get("feature_set", "C")
    p = SealedTailPanel() if mode == "tail" else SealedPanel()
    FA.build_all()
    u = p["universe"]
    oos = np.asarray(p.dates >= pd.Timestamp(SEAL_START))
    print(f"\n封存期 {p.dates[oos][0].date()} ~ {p.dates[oos][-1].date()},"
          f"{int(oos.sum())} 个交易日")

    names = spec.get("screen_names") or []
    feats = ML.feature_set(p, "C" if names else fs_name, names)
    X, Y, U, Ulab, fnames = ML.build_xy(feats, p, u, horizon, mode)
    print(f"特征 {len(fnames)} 个(冻结清单 {len(spec.get('feature_names', []))} 个)")
    if spec.get("feature_names") and sorted(fnames) != sorted(spec["feature_names"]):
        miss = set(spec["feature_names"]) - set(fnames)
        extra = set(fnames) - set(spec["feature_names"])
        print(f"⚠ 特征清单与冻结规格不一致!缺 {len(miss)} 个 {sorted(miss)[:5]},"
              f"多 {len(extra)} 个 {sorted(extra)[:5]}")
        if args.confirm:
            print("拒绝执行:特征清单必须与冻结规格完全一致。")
            sys.exit(1)

    years = np.asarray(p.dates.year)
    tr = ML.purge_train((years >= 2026 - spec["train_years"]) & (years < 2026), horizon)
    print(f"训练窗 {int(tr.sum())} 天({p.dates[tr][0].date()} ~ {p.dates[tr][-1].date()})")
    if not args.confirm:
        print("\n干跑校验通过:数据、特征、训练窗均就绪。加 --confirm 正式考试。")
        return

    Xtr, Ytr, dtr = ML.flatten(X, Y, Ulab, np.flatnonzero(tr))
    m = ML.fit_ts(ML.make_model(kind), Xtr, Ytr, dtr, purge=horizon + 1)
    sc = ML.predict_panel(m, X[:, oos], U[oos], p.dates[oos], p.codes)
    full = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
    full.loc[sc.index] = sc.to_numpy()

    pos = None
    if spec.get("timing"):
        feat = TM.market_features(p, index_code=None)   # 宽基快照缺 22 天,用自建等权
        if spec["timing"] == "rule":
            pos = TM.rule_position(feat, tail=(mode == "tail"))
        else:
            fwd = TM.forward_market(p, spec.get("timing_horizon", 20))
            mm = TM.fit_position_model(feat, fwd, tr, horizon=spec.get("timing_horizon", 20))
            pos = TM.model_position(mm, feat, tail=(mode == "tail"))

    rows = []
    for version in spec["versions"]:
        bm = ST.benchmark(p, version, exec_mode, date_mask=oos)
        for plan, kw in spec["plans"].items():
            r = ST.run_strategy(full, p, version, exec_mode, date_mask=oos, **kw)
            if not np.isfinite(r.get("ann_return", np.nan)):
                continue
            rec = dict(version=version, plan=plan, bench=bm["ann_return"],
                       ann=r["ann_return"], excess=r["ann_return"] - bm["ann_return"],
                       sharpe=r["sharpe"], max_dd=r["max_dd"], turnover=r["turnover"],
                       exposure=r.get("exposure"), n_days=r["n_days"])
            if pos is not None:
                d2 = TM.apply_position(r["rets"], pos)
                d3 = TM.apply_position(r["rets"], pd.Series(pos.mean(), index=pos.index))
                for tag, d in [("timed", d2), ("flat", d3)]:
                    nv = (1 + d).cumprod()
                    rec[f"ann_{tag}"] = float(nv.iloc[-1] ** (242 / len(nv)) - 1)
                    rec[f"dd_{tag}"] = float((nv / nv.cummax() - 1).min())
            rows.append(rec)
    df = pd.DataFrame(rows)
    print("\n=== 封存期(2026)最终考试结果 —— 如实记录,不得据此调参 ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out = LEDGER / f"sealed_model_{fp}.json"
    json.dump({"fingerprint": fp, "spec": spec, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "results": rows}, open(out, "w"), ensure_ascii=False, indent=1, default=float)
    with open(MODEL_LEDGER, "a") as f:
        f.write(json.dumps({"fingerprint": fp, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "spec": spec, "result_file": out.name}, ensure_ascii=False) + "\n")
    print(f"\n结果写入 {out};台账 {MODEL_LEDGER} 已记录,该版本不得重考。")


if __name__ == "__main__":
    main()
