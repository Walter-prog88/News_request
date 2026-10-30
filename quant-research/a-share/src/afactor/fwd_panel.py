"""前向检验面板 ForwardPanel(前缀 fwd):接口与 Panel 一致(raw / __getitem__ / dates / codes / 派生量),
现有因子、引擎、风险模型、优化器代码可以直接用。

数据:cache/forward/panel/<字段>.parquet(scripts/fwd_build_panel.py 构建),日期 2025-01-02 ~ 2026-10-09,
代码 = 研究面板 5565 只(原顺序)+ 2026-08-18 以后上市的 16 只沪深新股。字段全部是 MINUTE5_BRIDGE 的 5 分钟口径
(桥接 C 口径:5 分钟信号 + 5 分钟执行价),来源分段见构建脚本文档与面板里的 _src_code。

**两个视图**(预注册 FORWARD_TEST_PREREG §3.5):
  ForwardPanel()                    特征面板:执行价完整,用于因子、风险模型、成本模型;
  ForwardPanel(returns_view=True)   收益面板:只把 twap_open10 在 2026-09-02(首个前向成交日)之前置为 NaN,
                                    用于引擎、strategy.benchmark、指数基准、PEAD 账户 —— 引擎在预热段算不出任何持仓收益。
**纪律**:2026-01-01 ~ 2026-08-30 的数据只作特征回看/预热,不得计算或报告这段时间的策略收益、IC、超额;
落盘的任何收益序列首个索引必须 >= 2026-09-01(assert_forward_index)。

**派生量的数据源替换**(公式不变,只把冻结代码里会越过数据截止日的读取换成面板里的 PIT 字段):
  is_st        -> 面板 is_st(研究期 ST 名单 <= 2026-06-30,07-01 起 baostock 日线 isST)。
                  不用 fundamental.build_st_panel(源只到 07-03,越界会报 SourceCoverageError)。
  market_cap   -> 面板 mv = 未复权收盘 × PIT 总股本(tushare 资产负债表 + baostock 季频 totalShare,按报告期末之后的
                  除权因子放大)。**不用** fundamental.build_market_cap(送转还原依赖 2026-08 前复权基准),也不用
                  超出 08-21 的 newdata.total_mv(预注册 §4.3)。这个市值与研究期 ND.total_mv 不是同一口径
                  (后者有限售解禁外推高估),只影响风格暴露与匹配分位。
  _real(mv/turn/circ/fs) -> 面板 mv / turnover_rate / circ_mv / float_share(daily_basic <= 08-17,之后 baostock 日线)。
  statind_L    -> 面板 statind_labels(2025 年行 = 研究期标签文件;2026 各季由 fwd_adapters 同一算法计算)。
财报(_fu)仍走 fundamental.build_report_level,必须在 forward_context() 内:截止日放宽到 2026-10-09,并按预注册
§4.3 的原规则(fwd_decisions:东财营业收入一致率 98.5% < 99%)在 accept_source_gap(FU.REPORT_TABLES) 内沿用最后已知报告期。
"""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

from afactor.config import CACHE
from afactor.panel import Panel

PANEL_DIR = CACHE / "forward" / "panel"
FWD_CUTOFF = "2026-10-09"
SIG_START = pd.Timestamp("2026-09-01")      # 首个前向信号日
EXEC_START = pd.Timestamp("2026-09-02")     # 首个前向成交日(收益面板执行价从这天起才有值)
DATA_END = pd.Timestamp("2026-10-09")


class ForwardPanel(Panel):
    """前向检验面板。returns_view=True 为"收益面板"(twap_open10 在 2026-09-02 之前置 NaN)。"""

    def __init__(self, returns_view: bool = False):
        super().__init__(sealed=False)
        self.dir = PANEL_DIR
        self.returns_view = bool(returns_view)
        self._seed()

    def raw(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            f = self.dir / f"{name}.parquet"
            if not f.exists():
                raise KeyError(f"前向面板原料不存在: {name}")
            df = pd.read_parquet(f).astype(np.float32)
            if self.returns_view and name == "twap_open10":
                df = df.copy()
                df.loc[df.index < EXEC_START] = np.nan
            self._cache[name] = df
        return self._cache[name]

    @property
    def features(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.parquet") if not p.stem.startswith("_"))

    def _seed(self) -> None:
        """派生量的数据源替换(见模块文档)。_real 的缓存键带截止日后缀(factors._cut_tag),两种都放。"""
        rd = lambda n: pd.read_parquet(self.dir / f"{n}.parquet")
        self._derived["is_st"] = rd("is_st").astype(np.float32) > 0.5
        mv = rd("mv").astype(np.float32)
        self._derived["market_cap"] = mv
        self._derived["statind_L"] = rd("statind_labels")
        real = {"mv": mv, "turn": rd("turnover_rate").astype(np.float32),
                "circ": rd("circ_mv").astype(np.float32), "fs": rd("float_share").astype(np.float32)}
        for k, v in real.items():
            for tag in ("", "@" + pd.Timestamp(FWD_CUTOFF).strftime("%Y%m%d")):
                self._derived[f"__real__{k}__0{tag}"] = v

    # ---- 窗口工具 ----
    def signal_days(self) -> pd.DatetimeIndex:
        """前向信号日:>= 09-01 且 t+2 个交易日在面板内(open10 收益在 t+2 实现)。"""
        d = self.dates
        return d[(d >= SIG_START) & (np.arange(len(d)) + 2 < len(d))]

    def mask_from(self, start=SIG_START) -> np.ndarray:
        """date_mask:从 start 起的全部面板日(引擎 / 基准的冷启动锚点 = mask 首日)。"""
        return np.asarray(self.dates >= pd.Timestamp(start))


@contextlib.contextmanager
def forward_context(reason: str = "前向检验(fwd):冻结参数检验 2026-09 窗口"):
    """放宽数据截止日到 2026-10-09(fixb 接口,打印横幅并写 logs/data_cutoff_switch.log),
    并按预注册 §4.3 原规则在 accept_source_gap(FU.REPORT_TABLES) 内沿用最后已知报告期。"""
    from afactor import config as C, fundamental as FU
    with C.set_data_cutoff(FWD_CUTOFF, reason=reason):
        with C.accept_source_gap(list(FU.REPORT_TABLES),
                                 reason="预注册 §4.3 原规则:两源一致率未全部达 99%(revenue 98.5%),沿用最后已知报告期"):
            yield


def assert_forward_index(s, what: str) -> None:
    """落盘的收益 / 超额序列首个索引必须 >= 2026-09-01(预注册 §3.5、§5.5 第 4 条)。"""
    idx = pd.DatetimeIndex(s.index)
    if len(idx) and idx.min() < SIG_START:
        raise AssertionError(f"{what}:序列首个索引 {idx.min().date()} 早于 {SIG_START.date()}(纪律 1)")
