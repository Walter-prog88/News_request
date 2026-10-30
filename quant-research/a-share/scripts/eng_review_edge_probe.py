"""引擎审查修复的"修复前 vs 修复后"边界用例(前缀 eng,2026-10-10)-> research/ledger/eng_review_edge.csv。

用法:python3 scripts/eng_review_edge_probe.py <修复前 strategy.py 快照路径>
  修复前版本(sha256 cdf31cb1b345…,即引擎改造后、审查修复前)从未提交进仓库,
  2026-10-10 在会话 scratchpad 里留存后一次性运行;仓库 HEAD 里的 strategy.py 是更早的改造前版本,不能代用。
内容:
  (1) 审查 P2-b:目标含负权重(20 只未持有股票 -1%)、target_fn 原地改 ctx['can'] —— 修复前 / 修复后对比;
  (2) 审查 P1-a:修复前 ctx['cap_w'] 驱动的"截量后重分配"规则,与 eng_review_capw.csv 的 realized_te 组
      (当前引擎 + 手算 te 实际窗口成交额)逐位相等 —— 所以 realized_te 组就是"修复前"的数字。
COMP_v5 信号样本内;数据不含已退市股。"""
import hashlib, importlib.util, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import strategy as NEW
from afactor.config import TMP, LEDGER
OLD_PATH = sys.argv[1]
sha_old = hashlib.sha256(open(OLD_PATH, "rb").read()).hexdigest()
sha_new = hashlib.sha256(open(os.path.join(ROOT, "src/afactor/strategy.py"), "rb").read()).hexdigest()
spec = importlib.util.spec_from_file_location("st_old", OLD_PATH)
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
p = Panel(); dates = p.dates
score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
m24 = np.asarray(dates.year == 2024)
W = NEW.build_weights(score, p, "V1000", "top", 20).astype(np.float64)
rows = []
src_old = f"修复前快照 sha256 {sha_old[:12]}…"; src_new = f"修复后 strategy.py sha256 {sha_new[:12]}…"


def fn_neg(t, h, ctx):
    w = W[t].copy()
    cand = np.flatnonzero(ctx["can"] & (w == 0) & (h < 1e-12))[:20]
    w[cand] = -0.01
    return w


def fn_mut(t, h, ctx):
    ctx["can"][:] = True
    return W[t]


for lab, M, src in (("before", OLD, src_old), ("after", NEW, src_new)):
    r0 = M.simulate(lambda t, h, ctx: W[t], p, "open10", 1e7, rebalance=5, date_mask=m24, record=True)
    d0 = r0["record"]["daily"]
    rows.append(dict(stage=lab, case="baseline_top20_r5_2024", metric="ann_return", value=r0["ann_return"], source=src))
    rows.append(dict(stage=lab, case="baseline_top20_r5_2024", metric="exposure", value=r0["exposure"], source=src))
    rows.append(dict(stage=lab, case="baseline_top20_r5_2024", metric="sell_value_sum_million", value=d0.sell_value.sum() / 1e6, source=src))
    rows.append(dict(stage=lab, case="baseline_top20_r5_2024", metric="stamp_sum_bp", value=d0.cost_stamp.sum() * 1e4, source=src))
    try:
        r1 = M.simulate(fn_neg, p, "open10", 1e7, rebalance=5, date_mask=m24, record=True)
        d1 = r1["record"]["daily"]
        rows += [dict(stage=lab, case="neg_target_-1pct_x20", metric="raised_ValueError", value=0.0, source=src),
                 dict(stage=lab, case="neg_target_-1pct_x20", metric="ann_return", value=r1["ann_return"], source=src),
                 dict(stage=lab, case="neg_target_-1pct_x20", metric="exposure", value=r1["exposure"], source=src),
                 dict(stage=lab, case="neg_target_-1pct_x20", metric="sell_value_sum_million", value=d1.sell_value.sum() / 1e6, source=src),
                 dict(stage=lab, case="neg_target_-1pct_x20", metric="stamp_sum_bp", value=d1.cost_stamp.sum() * 1e4, source=src)]
    except ValueError:
        rows.append(dict(stage=lab, case="neg_target_-1pct_x20", metric="raised_ValueError", value=1.0, source=src))
    r2 = M.simulate(fn_mut, p, "open10", 1e7, rebalance=5, date_mask=m24)
    rows += [dict(stage=lab, case="mutate_ctx_can_inplace", metric="ann_return", value=r2["ann_return"], source=src),
             dict(stage=lab, case="mutate_ctx_can_inplace", metric="bitwise_equal_baseline",
                  value=float(np.array_equal(r2["rets"].to_numpy(), r0["rets"].to_numpy())), source=src)]
    print(lab, "done", flush=True)

# P1-a:修复前 ctx['cap_w'] 驱动的重分配策略 == eng_review_capw.csv 的 realized_te 组(新模块 + 手算 te 实际成交额)
yrs = dates.year; m = np.asarray((yrs >= 2021) & (yrs <= 2025))
W50 = NEW.build_weights(score, p, "V1000", "top", 50)
for mode, cap_ in (("open10", 1e8), ("vwap1", 1e9)):
    amt = NEW.exec_prices(p, mode)[1].to_numpy(np.float64)
    def make(src):
        def fn(t, h, ctx):
            te = ctx["te"]; w = W50[t].astype(np.float64); can = ctx["can"]
            cap = ctx["cap_w"] if src == "old_ctx" else np.where(np.isfinite(amt[te]), 0.10 * amt[te] / cap_, 0.0)
            w = np.where(can, w, 0.0); s = w.sum(); w = w / s * ctx["budget"] if s > 0 else w
            d = w - h; dc = np.clip(d, -cap, cap)
            for _ in range(5):
                short = float(np.clip(d - dc, 0, None).sum())
                room = np.where((w > 0) & can, np.clip(cap - dc, 0, None), 0.0); rs = room.sum()
                if short < 1e-9 or rs < 1e-12: break
                dc = dc + room * min(1.0, short / rs); d = dc + np.clip(d - dc, None, 0)
            tgt = np.clip(h + dc, 0, None); tot = tgt.sum()
            return tgt / tot * min(tot, 1.0) if tot > 0 else tgt
        return fn
    a = OLD.run_weights(make("old_ctx"), p, mode, cap_, rebalance=5, date_mask=m, normalize=False)
    b = NEW.run_weights(make("manual_realized"), p, mode, cap_, rebalance=5, date_mask=m, normalize=False)
    rows += [dict(stage="before", case=f"capw_policy_{mode}_cap{cap_:.0e}", metric="ann_return_old_ctx", value=a["ann_return"], source=src_old),
             dict(stage="check", case=f"capw_policy_{mode}_cap{cap_:.0e}", metric="bitwise_equal_realized_te_arm",
                  value=float(np.array_equal(a["rets"].to_numpy(), b["rets"].to_numpy())),
                  source="修复前 ctx['cap_w'] 驱动 == eng_review_capw.csv realized_te 组(当前引擎+手算 te 实际成交额)")]
df = pd.DataFrame(rows)
df.to_csv(LEDGER / "eng_review_edge.csv", index=False)
print(df.to_string())
