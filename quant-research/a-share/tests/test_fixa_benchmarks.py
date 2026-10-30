"""第二轮修订(前缀 fixa,2026-10-10)的基准接口测试:推荐的中证1000 代理基准搬到 index_proxy。

运行:  PYTHONPATH=src python3 -m pytest tests/test_fixa_benchmarks.py -q

  1 IP.exec_time_bench_returns 是推荐基准;optimizer 的旧名是同一个函数(调用方式与数值不变)
  2 W=None 时自动取代理收盘权重,与显式传入 proxy_weights 逐位相同;索引与 strategy.benchmark 相同
  3 IP.engine_consistent_bench_returns 仍可用(复现旧账本),但发 DeprecationWarning
  4 新基准对齐:与池基准的相关峰值在 k=0;且比旧基准高(旧基准每天重置回过时的收盘权重,系统性偏低)
同频池等权基准(strategy.benchmark(rebalance=k))的测试在 tests/test_engine_regression.py(4b)。
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.config import TMP                                    # noqa: E402
from afactor import index_proxy as IP, optimizer as OPT, strategy as ST   # noqa: E402

pytestmark = pytest.mark.skipif(not (TMP / "proxy_csi1000_weights.parquet").exists(),
                                reason="需要真实数据面板与代理权重缓存")


@pytest.fixture(scope="module")
def p():
    from afactor.panel import Panel
    return Panel()


def test_optimizer_names_are_index_proxy_functions():
    assert OPT.exec_time_bench_returns is IP.exec_time_bench_returns
    assert OPT.exec_time_weights is IP.exec_time_weights
    assert OPT.exec_time_bench_arrays is IP.exec_time_bench_arrays


@pytest.mark.parametrize("mode", ["open10", "vwap1"])
def test_exec_time_default_weights_and_index(p, mode):
    W = IP.proxy_weights(p, "000852")
    a = IP.exec_time_bench_returns(p, mode)
    b, Wx = OPT.exec_time_bench_returns(p, mode, W, return_weights=True)
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)
    bench = ST.benchmark(p, "V1000", mode)["rets"]
    assert a.index.equals(bench.index)
    s = Wx.sum(axis=1)
    assert np.allclose(s[s > 0], 1.0)
    with pytest.raises(NotImplementedError):
        IP.exec_time_bench_returns(p, "tail1450")


@pytest.mark.parametrize("mode", ["open10", "vwap1"])
def test_legacy_proxy_deprecated_and_lower(p, mode):
    with pytest.warns(DeprecationWarning):
        old = IP.engine_consistent_bench_returns(p, mode, "000852")
    new = IP.exec_time_bench_returns(p, mode)
    pool = ST.benchmark(p, "V1000", mode, rebalance=5)["rets"]
    m = new.index.year >= 2021
    cs = {k: float(new[m].corr(pool.shift(k)[m])) for k in (-2, -1, 0, 1, 2)}
    assert max(cs, key=cs.get) == 0, cs
    # 旧基准系统性偏低(fixa_bench_compare.csv:2021-2025 open10 +0.80pp、vwap1 +2.52pp)
    geo = lambda r: float((1 + r.dropna()).prod() ** (242 / r.dropna().size) - 1)
    gap = geo(new[m]) - geo(old[m])
    assert gap > (0.003 if mode == "open10" else 0.015), gap
