"""影子根目录里的补充重算(前缀 mvfix,2026-10-10):CSI1000_PROXY §7 / 容量报告引用的推荐基准跟踪统计,
原样调用 scripts/fixa_benchmark_fix.proxy_rows,写 <影子根>/research/ledger/mvfix_shadow_proxy_rows.csv。
只在 cache/_tmp/mvfix_shadow/{legacy,new} 里运行(由 mvfix_total_mv.py --stage shadow_run 调用),不写本项目账本。"""
from __future__ import annotations
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, ROOT          # noqa: E402

if __name__ == "__main__":
    assert "mvfix_shadow" in str(ROOT), f"只在影子根目录里运行(当前 ROOT = {ROOT})"
    import fixa_benchmark_fix as FX               # noqa: E402
    from afactor.panel import Panel               # noqa: E402
    FX.proxy_rows(Panel()).to_csv(LEDGER / "mvfix_shadow_proxy_rows.csv", index=False)
    print("写出", LEDGER / "mvfix_shadow_proxy_rows.csv")
