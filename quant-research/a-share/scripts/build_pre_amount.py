"""一次性:从分钟 amt 构建 pre_amount(09:30-14:49 累计成交额)日频面板 —— XSB2 线(2026-08-30)。

口径与 build_daily 的 pre_* 家族一致:PRE_CUTOFF=230 截断(09:30-14:49)。
只建研究期 2020-01 ~ 2025-12;2026 属封存期,本脚本不触碰(封存重建留给考试流程)。
输出 cache/daily/pre_amount.parquet,宽表 date×ts_code,float32,与 merge_daily 同规格。
"""
from __future__ import annotations
import numpy as np, pandas as pd
from afactor.config import PRE_CUTOFF, N_BARS, DAILY, load_raw_minute, raw_months

months = [m for m in raw_months("amt") if m <= "202512"]
frames = []
for m in months:
    df = load_raw_minute("amt", m)
    df = df.sort_index()
    days = pd.DatetimeIndex(df.index.normalize().unique())
    arr = df.to_numpy(np.float64)
    if arr.shape[0] == len(days) * N_BARS:
        pre = np.nansum(arr.reshape(len(days), N_BARS, -1)[:, :PRE_CUTOFF, :], axis=1)
        out = pd.DataFrame(pre, index=days, columns=df.columns)
    else:  # 罕见:某日 bar 数不齐,逐日回退
        parts = {}
        for d, g in df.groupby(df.index.normalize()):
            parts[d] = g.iloc[:PRE_CUTOFF].sum(min_count=1)
        out = pd.DataFrame(parts).T
    frames.append(out)
    print(m, out.shape, flush=True)
panel = pd.concat(frames).sort_index().astype(np.float32)
ref = pd.read_parquet(DAILY / "amount.parquet")
panel = panel.reindex(index=ref.index, columns=ref.columns)
# 有分钟数据但全 NaN 的日子沿用 NaN;完整性对账:pre_amount ≤ amount(容差 1%)
chk = (panel.to_numpy() <= ref.to_numpy() * 1.01) | np.isnan(panel.to_numpy()) | np.isnan(ref.to_numpy())
assert chk.mean() > 0.999, f"pre_amount > amount 比例异常: {1-chk.mean():.4%}"
panel.to_parquet(DAILY / "pre_amount.parquet")
cov = float(np.isfinite(panel.to_numpy()).mean())
print("written", panel.shape, "coverage", round(cov, 4))
