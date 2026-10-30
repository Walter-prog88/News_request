"""全局配置:路径、口径常量、封存边界。

数据事实(2026-08-20 审计确认):
- new_data/{open_adj,close_adj,amt}/YYYYMM_*,宽表 [分钟 x 股票],80 个月 2020-01~2026-08
  存储:价格字段 parquet+zstd+float32(6.6x),amt 为 pkl.gz+float32;统一经 load_raw_minute() 读取。
  float32 相对误差 ~6e-8(分钟收益误差 ~1e-7),相对 IC 门槛 0.025 可忽略。
- 每日 241 根 bar:idx0=09:30(开盘集合竞价),idx1..120=09:31..11:30,idx121..240=13:01..15:00
- 14:58/14:59 几乎无成交(收盘集合竞价期间),15:00 bar = 收盘集合竞价成交
- 价格为**前复权**(基准 2026-08):收益率无前视,但**绝对价格水平含未来分红信息 → 禁用绝对价格类因子**
- NaN 为整日级(未上市/停牌),不存在日内部分缺失;amt=0 分钟价格为前值填充
- **数据源存在幸存者偏差**:已退市股票(乐视网/康得新/信威等)完全不在列中
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "new_data"
CACHE = ROOT / "cache"
DAILY = CACHE / "daily"          # 2020-2025 研究期原料
SEALED = CACHE / "sealed"        # 2026 封存期原料(研究流程禁止读取)
TMP = CACHE / "_tmp"
LEDGER = ROOT / "research" / "ledger"
REPORTS = ROOT / "reports"

for _p in (DAILY, SEALED, TMP, LEDGER, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---- 封存边界 ----
SEAL_START = "2026-01-01"          # >= 此日期的数据封存,最终定稿后只跑一次
RESEARCH_START = "2020-01-01"
RESEARCH_END = "2025-12-31"
SEAL_LOCK = LEDGER / "smoke_used.lock"

# ---- 外部表数据截止日(2026-10-10,前缀 fixb;见 research/DATA_CUTOFF_DISCIPLINE.md)----
# data_8:21/ 下的 daily_basic、财报三表、业绩预告/快报、龙虎榜、ST 名单、stock_basic 与宽基指数快照
# 都含 2026 行。原则是"**先截断再读**":各加载器(newdata / fundamental)读入后**立即**丢掉截止日之后
# 的行,然后才做去重、透视、ffill。此前是整表读入、靠 reindex 到研究期日期"看不见"2026 行 ——
# 但"同一报告期保留最晚公告版本"这类去重会被 2026 年的更正公告改写(选择效应),reindex 挡不住。
# 默认截止日 = 研究期末;只有前向检验代码(文件名前缀 fwd)能用 set_data_cutoff() 临时放宽。
DATA_CUTOFF = RESEARCH_END
# 价格面板(new_data / cache/daily)的**前复权基准月末**。前复权价本身已经含有基准日之前全部送转的
# 信息,fundamental.build_market_cap 还原送转时必须用同一基准的股本序列 —— 这是唯一允许越过
# DATA_CUTOFF 读取股本的地方(理由见该函数文档),不服务任何其它字段。
PRICE_ADJ_BASE_END = "2026-08-31"
FWD_CALLER_PREFIXES = ("fwd", "test_")      # 允许放宽截止日的调用者文件名前缀(前向检验脚本 / 单元测试)
CUTOFF_LOG = ROOT / "logs" / "data_cutoff_switch.log"
_CUTOFF_STACK: list = []                    # [(Timestamp, 原因, 调用者)],栈顶为当前生效值


def data_cutoff():
    """当前生效的外部表数据截止日(pd.Timestamp)。加载器每次读取时调用,不要在 import 时缓存它。"""
    import pandas as pd
    if _CUTOFF_STACK:
        return _CUTOFF_STACK[-1][0]
    return pd.Timestamp(DATA_CUTOFF)


class set_data_cutoff:
    """临时切换数据截止日的上下文管理器 —— **只供前向检验脚本使用**。

        with set_data_cutoff("2026-10-09", reason="前向检验:2026-09 起冻结参数"):
            p = <前向面板>; mv = ND.total_mv(p.dates, p.codes)

    纪律(为什么这样设计):
      - 放宽(截止日晚于 DATA_CUTOFF)只允许文件名以 fwd / test_ 开头的调用者,其它代码路径直接报错,
        防止研究流程"顺手"读到 2026 数据;收紧(更早的截止日)任何代码都可以用。
      - 进入与退出都向 stderr 打印醒目横幅,并向 logs/data_cutoff_switch.log 追加一行审计记录。
      - 加载器的进程内缓存都按截止日分键(newdata._CACHE / fundamental._REP_CACHE / factors 的面板缓存),
        切换后不会串用另一个截止日的结果;退出时恢复上一层截止日,可嵌套。
      - 放宽只解决"能读到"。2026-01-01 ~ 2026-08-30 的数据只能作特征回看/预热输入,
        不得计算或报告这段时间的策略收益、IC、超额 —— 这一条由前向检验脚本自己遵守。
      - 放宽也不等于有数据:源文件覆盖不到的请求日期由 check_source_coverage 拦截(默认报错),
        见下方"源覆盖守卫"。
    """

    def __init__(self, date, reason: str):
        import os
        import sys
        import pandas as pd
        self.date = pd.Timestamp(date).normalize()
        self.reason = (reason or "").strip()
        if len(self.reason) < 4:
            raise ValueError("set_data_cutoff 必须写明切换原因(reason)")
        fr = sys._getframe(1)
        self.caller = f"{fr.f_code.co_filename}:{fr.f_lineno}"
        base = os.path.basename(fr.f_code.co_filename)
        if self.date > pd.Timestamp(DATA_CUTOFF) and not base.startswith(FWD_CALLER_PREFIXES):
            raise PermissionError(
                f"数据截止日只能由前向检验代码放宽(文件名前缀 {FWD_CALLER_PREFIXES}),"
                f"调用者 {base} 试图放宽到 {self.date.date()}。研究流程只能用 <= {DATA_CUTOFF} 的数据。")
        self._entry = None

    def _announce(self, what: str, old, new) -> None:
        import json
        import sys
        import time
        bar = "=" * 96
        msg = (f"\n{bar}\n!!! [DATA_CUTOFF] 数据截止日{what}: {old.date()} -> {new.date()}\n"
               f"!!!   原因: {self.reason}\n!!!   调用者: {self.caller}\n"
               f"!!!   2026-01-01~2026-08-30 只作特征回看/预热,不得报告其收益/IC/超额\n{bar}\n")
        print(msg, file=sys.stderr, flush=True)
        try:
            CUTOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(CUTOFF_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), action=what,
                                         old=str(old.date()), new=str(new.date()), reason=self.reason,
                                         caller=self.caller), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def __enter__(self):
        old = data_cutoff()
        self._entry = (self.date, self.reason, self.caller)
        _CUTOFF_STACK.append(self._entry)
        self._announce("切换", old, self.date)
        return self.date

    def __exit__(self, *exc):
        if not _CUTOFF_STACK or _CUTOFF_STACK[-1] is not self._entry:
            raise RuntimeError("set_data_cutoff 嵌套顺序错误:退出的不是最内层截止日")
        _CUTOFF_STACK.pop()
        self._announce("恢复", self.date, data_cutoff())
        return False


def assert_within_cutoff(dates, what: str) -> None:
    """请求的面板日期不得晚于当前数据截止日 —— 否则截止日之后的行会被静默 ffill 成旧值或留空。

    前向检验面板必须在 set_data_cutoff(...) 上下文内计算;研究面板(<= RESEARCH_END)不受影响。
    """
    import pandas as pd
    if len(dates) == 0:
        return
    last = pd.Timestamp(pd.DatetimeIndex(dates).max())
    cut = data_cutoff()
    if last > cut:
        raise PermissionError(
            f"{what}:面板日期到 {last.date()},晚于数据截止日 {cut.date()}。"
            f"前向检验请在 config.set_data_cutoff(...) 上下文内计算;研究流程只能用 <= {cut.date()} 的数据。")


# ---- 源覆盖守卫(2026-10-10 第二轮审查 P1,前缀 fixb)----
# assert_within_cutoff 只管"请求日期 <= 截止日",管不了"源文件本身只到哪天"。放宽截止日后(前向检验),
# 源文件覆盖不到的日期会被各加载器**静默**填成旧值:ST 名单 ffill(引擎据此判定 5% 涨跌停 / 一字板 /
# 不可成交)、龙虎榜 fillna(0)(全市场"未上榜")、财报 / 预告 ffill(停在源文件最后一期)。
# 修法:每个源在读入时登记**原始文件**的最晚日期(截断之前取 max,只是一个日期标量,不进入任何数值);
# 面板请求日期晚于它时默认报 SourceCoverageError。研究期(<= RESEARCH_END)所有源文件都覆盖到 2026 年,
# 守卫永远不触发,数值与之前逐位相同。前向检验要么补数(例如自建 ST 面板放进 p._derived["is_st"]),
# 要么在 accept_source_gap(...) 内显式接受缺口(此时只发 RuntimeWarning,见各加载器的"接受后行为")。
_SOURCE_END: dict = {}                      # 源名 -> 原始文件最晚日期(pd.Timestamp)
_ACCEPTED_GAPS: list = []                   # [(frozenset(源名), 原因, 调用者)]


class SourceCoverageError(RuntimeError):
    """面板请求日期晚于源文件实际覆盖日(放宽截止日 != 有数据)。"""


def record_source_end(source: str, end) -> None:
    """加载器读入原始文件后、截断前调用:登记该源的原始最晚日期。"""
    import pandas as pd
    _SOURCE_END[source] = pd.Timestamp(end) if end is not None and pd.notna(end) else pd.NaT


def source_end(source: str):
    """源文件原始最晚日期;本进程尚未读过该源时返回 None。"""
    return _SOURCE_END.get(source)


class accept_source_gap:
    """显式接受"源文件覆盖不到请求日期"——只在前向检验里、确认缺口已另行处理或可以接受时使用。

        with C.set_data_cutoff("2026-10-09", reason=...):
            with C.accept_source_gap(FU.REPORT_TABLES, reason="财报停在 2026Q1,前向窗口沿用最后一期"):
                x = FA._fu(p, "roe")

    上下文内,所列源的覆盖缺口只发 RuntimeWarning、不报错;进入时打印并写入 CUTOFF_LOG 审计行。
    """

    def __init__(self, sources, reason: str):
        import sys
        if isinstance(sources, str):
            sources = [sources]
        self.sources = frozenset(sources)
        self.reason = (reason or "").strip()
        if len(self.reason) < 4:
            raise ValueError("accept_source_gap 必须写明原因(reason)")
        fr = sys._getframe(1)
        self.caller = f"{fr.f_code.co_filename}:{fr.f_lineno}"
        self._entry = None

    def __enter__(self):
        import json
        import sys
        import time
        self._entry = (self.sources, self.reason, self.caller)
        _ACCEPTED_GAPS.append(self._entry)
        print(f"!!! [SOURCE_GAP] 显式接受源覆盖缺口: {sorted(self.sources)};原因: {self.reason};调用者: {self.caller}",
              file=sys.stderr, flush=True)
        try:
            CUTOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(CUTOFF_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), action="接受源缺口",
                                         sources=sorted(self.sources), cutoff=str(data_cutoff().date()),
                                         reason=self.reason, caller=self.caller), ensure_ascii=False) + "\n")
        except OSError:
            pass
        return self

    def __exit__(self, *exc):
        if not _ACCEPTED_GAPS or _ACCEPTED_GAPS[-1] is not self._entry:
            raise RuntimeError("accept_source_gap 嵌套顺序错误")
        _ACCEPTED_GAPS.pop()
        return False


def _gap_accepted(source: str) -> bool:
    return any(source in s for s, _, _ in _ACCEPTED_GAPS)


def check_source_coverage(sources, dates, what: str):
    """面板请求日期 vs 源文件原始最晚日期。

    返回 None:请求的日期全部在覆盖范围内(研究期恒为此);
    返回 pd.Timestamp(各源最晚日期的最小值):有缺口但所有缺口源都已在 accept_source_gap 内接受(同时发 RuntimeWarning);
    否则抛 SourceCoverageError。源尚未登记(本进程没读过)时不检查该源 —— 调用方应先读源再检查。
    """
    import warnings
    import pandas as pd
    if len(dates) == 0:
        return None
    if isinstance(sources, str):
        sources = [sources]
    d = pd.DatetimeIndex(dates)
    last = d.max()
    gaps = {s: _SOURCE_END[s] for s in sources
            if s in _SOURCE_END and pd.notna(_SOURCE_END[s]) and last > _SOURCE_END[s]}
    if not gaps:
        return None
    end = min(gaps.values())
    n_after = int((d > end).sum())
    detail = ";".join(f"{s} 只到 {e.date()}" for s, e in sorted(gaps.items()))
    msg = (f"{what}:请求日期到 {last.date()},但 {detail}(当前截止日 {data_cutoff().date()})。"
           f"{n_after} 个请求日期在源覆盖之外,加载器会把它们填成最后已知值或缺失。"
           f"前向检验请补数(例如自建 ST 面板放进 p._derived['is_st']),"
           f"或在 config.accept_source_gap({sorted(gaps)}, reason=...) 内显式接受。")
    if all(_gap_accepted(s) for s in gaps):
        warnings.warn("[已接受] " + msg, RuntimeWarning, stacklevel=3)
        return end
    raise SourceCoverageError(msg)

# ---- 分钟 bar 索引常量(241 根)----
N_BARS = 241
IDX_OPEN_AUCTION = 0             # 09:30 开盘集合竞价
IDX_AM_START, IDX_AM_END = 1, 120        # 09:31..11:30 连续竞价上午
IDX_PM_START, IDX_PM_END = 121, 240      # 13:01..15:00
IDX_1000 = 30                    # 10:00
IDX_1130 = 120
IDX_1301 = 121
IDX_1400 = 180
IDX_1430 = 210                   # 14:30
IDX_1450 = 230                   # 14:50  <- 尾盘口径因子截断点(含)
IDX_1457 = 237                   # 14:57  连续竞价最后一分钟
IDX_CLOSE_AUCTION = 240          # 15:00 收盘集合竞价

# 成交窗口
OPEN_TWAP = (1, 11)              # 09:31-09:40,主口径 T+1 开盘成交
TAIL_TWAP = (230, 238)           # 14:50-14:57,尾盘口径 T 日成交
PRE_CUTOFF = 230                 # 尾盘口径:因子只能用 idx < 230 的数据(09:30-14:49)

# ---- 回测口径 ----
COST_BPS_ONEWAY = 1.0            # 单边交易成本 1bp(与 ETF 项目一致)
STAMP_TAX_BPS_BEFORE = 10.0      # 2023-08-28 前 卖出印花税 10bp
STAMP_TAX_BPS_AFTER = 5.0        # 之后 5bp
STAMP_TAX_CHANGE_DATE = "2023-08-28"
CAPITAL = 1e7                    # 1000 万
MAX_PARTICIPATION = 0.10         # TWAP 窗口内最大成交额参与率
MIN_ADV20 = 2e7                  # 股票池最低 20 日日均成交额(2000万)
MAX_TURNOVER = 1.0               # 日换手上限 100%
N_QUANTILES = 10
IC_THRESHOLD = 0.025

# ---- 标准回测配置(2026-08-20 第二轮实验后固化)----
# 排雷缓冲带:分位跌入最差 20% 剔除,回升到 35% 以上才重新纳入;每 5 个交易日调仓。
# 依据:缓冲带 20/35 超额 6.70% vs 无缓冲剔30% 的 6.11%,换手 0.033 vs 0.059;
#       5 日调仓再将换手砍半(~0.02)而超额只降 0.3-0.5pp;walk-forward 四年参数稳定。
STANDARD_BUFFER = (0.20, 0.35)
STANDARD_REBALANCE = 5
MIN_LIST_DAYS = 6                # 上市后 N 个交易日内不进入股票池(首日无涨跌幅限制,二级市场拿不到)

# 涨跌停幅度(按代码前缀)
GEM_20PCT_START = "2020-08-24"   # 创业板注册制改革,涨跌幅由 10% 放宽到 20%


def price_limit(code: str, date=None) -> float:
    """涨跌幅上限。date 为 None 时按当前规则(仅用于粗略场景)。

    2026-08-22 修正:创业板(300/301)的 20% 自 2020-08-24 起才生效,
    此前为 10%;之前全期按 20% 判定,使 2020 年前 8 个月漏检 3494 个涨停日
    (其中 699 个一字板被误当作可成交)。科创板(688/689)自 2019 年开板即 20%。
    """
    import pandas as pd
    if code.startswith(("688", "689")):
        return 0.20
    if code.startswith(("300", "301")):
        if date is None or pd.Timestamp(date) >= pd.Timestamp(GEM_20PCT_START):
            return 0.20
        return 0.10
    if code.startswith(("8", "4", "92", "43")):   # 北交所
        return 0.30
    return 0.10


def assert_not_sealed(dates) -> None:
    """研究流程守卫:任何触及 2026 数据的调用直接报错。"""
    import pandas as pd
    d = pd.to_datetime(pd.Index(dates))
    if (d >= pd.Timestamp(SEAL_START)).any():
        raise PermissionError(
            f"封存违规:研究流程试图读取 {SEAL_START} 之后的数据。"
            f"封存样本只能在最终定稿后由 scripts/run_sealed_smoke.py 使用一次。"
        )


def load_raw_minute(field: str, month: str):
    """统一读取原始分钟宽表,自动适配存储格式(压缩前后都可用)。

    parquet 按扩展名识别;pickle 按**文件头字节**判断是否 gzip,
    而不是靠扩展名——历史上出现过 .pkl.gz 里装裸 pickle 的情况。
    """
    import pandas as pd
    from pathlib import Path as _P
    base = _P(str(RAW / field / f"{month}_{field}"))
    pq = _P(str(base) + ".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    for ext in (".pkl.gz", ".pkl"):
        p = _P(str(base) + ext)
        if p.exists():
            with open(p, "rb") as fh:
                gz = fh.read(2) == b"\x1f\x8b"
            return pd.read_pickle(p, compression="gzip" if gz else None)
    raise FileNotFoundError(f"未找到 {base}.(parquet|pkl.gz|pkl)")


def raw_months(field: str = "close_adj") -> list[str]:
    """列出可用月份,与存储格式无关。"""
    import re
    seen = set()
    for p in (RAW / field).iterdir():
        m = re.match(r"(\d{6})_", p.name)
        if m:
            seen.add(m.group(1))
    return sorted(seen)
