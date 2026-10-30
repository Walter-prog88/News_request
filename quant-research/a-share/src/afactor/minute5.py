"""5 分钟 bar 口径(baostock 口径)工具:1 分钟 -> 5 分钟精确聚合、执行价近似、baostock 抓取、窗口超额分布。

背景(2026-10-10,前缀 m5):原始 1 分钟数据只到 2026-08-17,前向检验(2026-09 以后)只能拿到
baostock 免费 5 分钟线。本模块定义"baostock 口径 5 分钟 bar",并让历史 1 分钟数据可以**精确**聚合成同一口径,
从而在 2020-2025 研究期量化"用 5 分钟线近似 1 分钟线"的误差(报告 research/MINUTE5_BRIDGE.md)。

## baostock 口径(每天 48 根,时间戳 = bar 结束时刻)
  bar 0   09:35 = 1 分钟 idx0..5   (idx0 = 09:30 开盘集合竞价 + 09:31..09:35)   <- 6 根合 1
  bar k   k=1..23  = idx 5k+1 .. 5k+5      (09:40 = idx6..10, ..., 11:30 = idx116..120)
  bar 24  13:05 = idx121..125, ..., bar 47  15:00 = idx236..240(含 15:00 收盘集合竞价)
  open   = 组内第一分钟 open_adj(bar 0 即 idx0 的 open = 开盘集合竞价价)
  close  = 组内最后一分钟 close_adj
  amount = 组内成交额求和
  原始 1 分钟数据**没有成交量、没有分钟高低价**,所以 5 分钟 bar 也只有 open/close/amount 三个字段;
  baostock 给的 high/low/volume 在桥接校验里一律不用(否则历史上无法复现同一口径,误差无法量化)。
  2025-12 30 只股票逐 bar 核对见 scripts/m5_baostock_check.py 与 research/ledger/m5_baostock_*.csv。

## 1 分钟数据的一个事实(决定 open_auction_px 的映射)
  1 分钟 idx0 这根 bar 的 open ≠ close 的股票日约占 46%(2025-12-01 抽样):idx0 = 集合竞价成交 + 09:30 这一分钟,
  open_adj[idx0] 才是开盘集合竞价价,close_adj[idx0] 是 09:30 这一分钟末的价格。build_daily 的
  open_auction_px 取的是 close_adj[idx0];5 分钟口径拿不到它,只能取 bar0 的 open(= open_adj[idx0],
  与 baostock 第一根 open 一致)。差异中位 0bp、90% 分位约 40bp(2025-12-01),会传导到 ret_intraday / overnight。

## 测量工具的选择准则(预先写定,不是策略参数)
  执行价 twap_open10、开盘窗口成交额 amt_open10、全日 vwap、开盘集合竞价占比 amt_openauc_share 的 5 分钟近似
  各有若干候选(见 *_CANDIDATES)。**只用 2020 年**(不在 2021-2025 评估窗内)的 V1000 决策池股票日选定,
  评估窗 2021-2025 只报告误差,不再改选:
    价格类(twap_open10 / vwap):|log(候选/1 分钟真值)| 的中位数最小(bp);并列时看 99% 分位。
    成交额类(amt_open10):|log(候选/真值)| 的中位数最小。
    占比类(amt_openauc_share):成员因子 id_amt_openauc_share_m5 的逐日截面秩相关均值最大。
  选定结果写在 TWAP5_METHOD 等常量里,依据见 research/ledger/m5_candidate_selection.csv。
"""
from __future__ import annotations
import signal
import time
import numpy as np
import pandas as pd

from afactor.config import CACHE, N_BARS

M5_DIR = CACHE / "m5"                      # 5 分钟口径日频原料(2020-2025),与 cache/daily 同形状
M5_TMP = CACHE / "_tmp" / "m5_months"      # 月度中间结果(放子目录:merge_daily 会 glob cache/_tmp/*.npz)

N_BARS5 = 48
EPS = 1e-12

# ---- bar 定义:bar k 覆盖 1 分钟 idx [BAR5_LO[k], BAR5_HI[k]](闭区间)----
BAR5_LO = np.array([0] + [5 * k + 1 for k in range(1, N_BARS5)])
BAR5_HI = np.array([5] + [5 * k + 5 for k in range(1, N_BARS5)])
assert BAR5_HI[-1] == N_BARS - 1 and BAR5_LO[24] == 121 and BAR5_HI[23] == 120
BAR5_TIMES = ([f"{9 + (35 + 5 * k) // 60:02d}:{(35 + 5 * k) % 60:02d}" for k in range(24)] +
              [f"{13 + (5 + 5 * k) // 60:02d}:{(5 + 5 * k) % 60:02d}" for k in range(24)])
assert BAR5_TIMES[0] == "09:35" and BAR5_TIMES[23] == "11:30" and BAR5_TIMES[24] == "13:05" \
    and BAR5_TIMES[47] == "15:00"
# bar 内各分钟在 1 分钟网格(0..240)上的中心位置 —— 成交额时间重心的 5 分钟映射用
BAR5_CENTER = (BAR5_LO + BAR5_HI) / 2.0

IDX5_0935, IDX5_0940 = 0, 1
IDX5_1000 = 5            # 09:56-10:00(idx26..30)
IDX5_1130 = 23
IDX5_1305 = 24
IDX5_1430 = 41           # 14:26-14:30(idx206..210)
IDX5_1450 = 45           # 14:46-14:50(idx226..230)
IDX5_CLOSE = 47          # 14:56-15:00(idx236..240,含收盘集合竞价)


# ================= 1 分钟 -> 5 分钟 =================
def aggregate_day(P: np.ndarray, O: np.ndarray, V: np.ndarray):
    """单日 1 分钟 (241, N) -> 5 分钟 (48, N)。返回 (P5, O5, V5) = (close, open, amount)。

    整日 NaN 的列(未上市/停牌)保持整日 NaN;1 分钟数据不存在日内部分缺失(config 文档),
    所以组内求和用 nansum 不会把缺失伪装成 0 —— 但仍按"整日无数据"把 V5 置回 NaN。
    """
    assert P.shape[0] == N_BARS, f"需要 {N_BARS} 根 1 分钟 bar,得到 {P.shape[0]}"
    N = P.shape[1]
    P5 = np.empty((N_BARS5, N), P.dtype)
    O5 = np.empty((N_BARS5, N), O.dtype)
    V5 = np.empty((N_BARS5, N), np.float64)
    P5[0] = P[5]
    O5[0] = O[0]
    V5[0] = np.nansum(V[0:6], axis=0)
    rest = slice(6, N_BARS)                                  # idx6..240 = 47 组 x 5
    P5[1:] = P[rest].reshape(N_BARS5 - 1, 5, N)[:, -1]
    O5[1:] = O[rest].reshape(N_BARS5 - 1, 5, N)[:, 0]
    V5[1:] = np.nansum(V[rest].reshape(N_BARS5 - 1, 5, N), axis=1)
    dead = np.isnan(P).all(axis=0)
    V5[:, dead] = np.nan
    return P5, O5, V5


def aggregate_wide(C: pd.DataFrame, O: pd.DataFrame, V: pd.DataFrame):
    """1 分钟宽表(分钟 x 股票,多日)-> 5 分钟宽表三件套,索引为 bar 结束时刻。只处理恰好 241 根的交易日。"""
    days = C.index.normalize()
    outP, outO, outV, idx = [], [], [], []
    Ca, Oa, Va = C.to_numpy(np.float64), O.to_numpy(np.float64), V.to_numpy(np.float64)
    uniq, starts, counts = np.unique(days.values, return_index=True, return_counts=True)
    for d, s, n in sorted(zip(uniq, starts, counts), key=lambda x: x[1]):
        if n != N_BARS:
            continue
        p5, o5, v5 = aggregate_day(Ca[s:s + n], Oa[s:s + n], Va[s:s + n])
        outP.append(p5); outO.append(o5); outV.append(v5)
        d = pd.Timestamp(d)
        idx += [d + pd.Timedelta(hours=int(t[:2]), minutes=int(t[3:])) for t in BAR5_TIMES]
    ix = pd.DatetimeIndex(idx)
    mk = lambda a: pd.DataFrame(np.vstack(a) if a else np.zeros((0, C.shape[1])), index=ix, columns=C.columns)
    return mk(outP), mk(outO), mk(outV)


def price_path(P5: np.ndarray, O5: np.ndarray) -> np.ndarray:
    """5 分钟口径的"收盘价路径":[开盘价, 48 根 bar 收盘] 共 49 点,对应 1 分钟口径的 P[0..240](241 点)。

    1 分钟口径的路径起点 P[0] 是 09:30 这一分钟的收盘(≈ 开盘价,54% 的股票日完全相等);
    5 分钟口径只能以 bar0 的 open(开盘集合竞价价)作起点,于是第一段收益 = 开盘价 -> 09:35 收盘,
    覆盖 1 分钟口径 P0->P5 的五段收益(外加 open->P0 那一小段)。"""
    return np.vstack([O5[:1], P5])


# ================= 执行价 / 窗口成交额 / VWAP 的候选 =================
def _first_traded_mean(px: np.ndarray, amt: np.ndarray) -> np.ndarray:
    """与 build_daily._twap 同一回退规则:只平均有成交的 bar;都无成交时退化为价格均值。"""
    trad = amt > EPS
    num = np.where(trad, px, np.nan)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(num, axis=0)
        return np.where(trad.sum(axis=0) == 0, np.nanmean(px, axis=0), m)


def twap_open10_candidates(P5: np.ndarray, O5: np.ndarray, V5: np.ndarray) -> dict:
    """1 分钟真值 twap_open10 = 09:31..09:40 十根 1 分钟 bar 中价 (open+close)/2 的等时均值(只算有成交分钟)。
    5 分钟只有 09:35(含集合竞价)与 09:40 两根 bar,候选:
      c2      两根收盘均值 (P[5] + P[10]) / 2(无成交 bar 不计,同 _twap 回退规则)
      c2amt   两根收盘的成交额加权(无量,只能用收盘价近似 bar 均价;bar0 的成交额含集合竞价)
      mid4    两根 bar 中价的均值 (O5[0]+P5[0]+O5[1]+P5[1]) / 4
      trap    梯形积分:把价格路径看成 开盘价(t=0) -> P[5](t=5) -> P[10](t=10) 的折线,
              10 分钟时间均值 = (O + 2*P[5] + P[10]) / 4
    """
    p0, p1 = P5[IDX5_0935], P5[IDX5_0940]
    v0, v1 = V5[IDX5_0935], V5[IDX5_0940]
    two_p = np.vstack([p0, p1]); two_v = np.vstack([v0, v1])
    with np.errstate(invalid="ignore", divide="ignore"):
        vs = v0 + v1
        c2amt = np.where(vs > EPS, (v0 * p0 + v1 * p1) / np.where(vs > EPS, vs, np.nan), (p0 + p1) / 2.0)
    return {
        "c2": _first_traded_mean(two_p, two_v),
        "c2amt": c2amt,
        "mid4": (O5[IDX5_0935] + p0 + O5[IDX5_0940] + p1) / 4.0,
        "trap": (O5[IDX5_0935] + 2.0 * p0 + p1) / 4.0,
    }


def amt_open10_candidates(V5: np.ndarray) -> dict:
    """1 分钟真值 amt_open10 = Σ amt[idx1..10](09:31-09:40,不含集合竞价)。5 分钟候选:
      sum2   bar0 + bar1(= 真值 + 集合竞价成交额,系统性偏高)
      twice1 2 * bar1(假设 09:31-09:35 与 09:36-09:40 等速,开盘更活跃 -> 系统性偏低)
    """
    return {"sum2": V5[IDX5_0935] + V5[IDX5_0940], "twice1": 2.0 * V5[IDX5_0940]}


def vwap_candidates(P5: np.ndarray, O5: np.ndarray, V5: np.ndarray) -> dict:
    """1 分钟真值 vwap = Σamt / Σ(amt/mid),mid = 1 分钟 (open+close)/2,全日 241 根(近似 VWAP,无成交量)。
      mid_h   同一公式换 5 分钟 bar:Σamt / Σ(amt/mid5),mid5 = (O5+P5)/2(与 1 分钟公式只差 bar 粒度)
      close_h Σamt / Σ(amt/P5)(bar 收盘价代表 bar 均价)
      mid_a   Σ(amt*mid5) / Σamt(算术加权)
    """
    mid5 = (O5 + P5) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        tot = np.nansum(V5, axis=0)
        f = lambda den: np.where(np.abs(den) > EPS, tot / np.where(np.abs(den) > EPS, den, np.nan), np.nan)
        return {"mid_h": f(np.nansum(V5 / mid5, axis=0)),
                "close_h": f(np.nansum(V5 / P5, axis=0)),
                "mid_a": np.where(tot > EPS, np.nansum(V5 * mid5, axis=0) / np.where(tot > EPS, tot, np.nan), np.nan)}


def openauc_share_candidates(V5: np.ndarray) -> dict:
    """1 分钟真值 amt_openauc_share = amt[idx0] / 全日成交额。5 分钟无法拆出集合竞价,候选:
      bar0       bar0 占比(集合竞价 + 09:31-09:35)
      bar0_m1    max(bar0 - bar1, 0) 占比(用 09:36-09:40 的成交速度扣掉连续竞价部分的粗估)
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        tot = np.nansum(V5, axis=0)
        d = np.where(tot > EPS, tot, np.nan)
        return {"bar0": V5[IDX5_0935] / d,
                "bar0_m1": np.clip(V5[IDX5_0935] - V5[IDX5_0940], 0.0, None) / d}


# ---- 选定结果(只用 2020 年 V1000 池选定,准则见模块文档;依据 research/ledger/m5_candidate_selection.csv,
#      scripts/m5_select_candidates.py)。2020 年 |log 误差| 中位数(bp):
#      twap_open10  mid4 12.05 < trap 12.23 < c2 15.63 < c2amt 15.97(mid4 与 trap 几乎并列,99% 分位也最低 86.2)
#      amt_open10   sum2 480(偏高 = 集合竞价额)< twice1 2244
#      vwap         mid_h 2.73 < mid_a 2.79 < close_h 3.94
#      amt_openauc_share  成员因子 5 日均值截面秩相关 bar0 0.691 > bar0_m1 0.634
TWAP5_METHOD = "mid4"
AMT_OPEN10_METHOD = "sum2"
VWAP5_METHOD = "mid_h"
OPENAUC5_METHOD = "bar0"


# ================= 5 分钟口径面板 =================
# 由 5 分钟 bar 重算的 (b) 类原料 + 由 5 分钟 bar 得到的 (a) 类原料(收盘/成交额/开盘价),
# 前向检验时这些字段全部只能来自 5 分钟线。
M5_SIGNAL_FIELDS = ("close_px", "amount", "open_auction_px", "ret_intraday", "rvol", "gap_share",
                    "amt_first30_share", "amt_gini", "smart_money", "vwap", "amt_openauc_share",
                    "amt_time_wgt", "semibeta_mp")
# 执行相关:成交价、成交窗口成交额、一字板判定用的日内振幅
M5_EXEC_FIELDS = ("twap_open10", "amt_open10", "rng")


def m5_panel(signal: bool = True, exec_: bool = True, m5_dir=None, base_cls=None):
    """返回一个 Panel:指定字段改读 5 分钟口径原料,其余(基本面、统计行业标签、ST 等)照旧。

    signal=True  -> M5_SIGNAL_FIELDS 读 5 分钟口径(打分用)
    exec_=True   -> M5_EXEC_FIELDS 读 5 分钟口径(成交价 / 参与率 / 一字板判定)
    两者都 False 即原面板。研究期面板只含 2020-2025(cache/daily 与 cache/m5 都不含 2026)。
    """
    from afactor.panel import Panel
    base = Panel if base_cls is None else base_cls
    over = set()
    if signal:
        over |= set(M5_SIGNAL_FIELDS)
    if exec_:
        over |= set(M5_EXEC_FIELDS)
    d = M5_DIR if m5_dir is None else m5_dir

    class _M5Panel(base):
        _m5_fields = frozenset(over)
        _m5_dir = d

        def raw(self, name: str) -> pd.DataFrame:
            if name in self._m5_fields:
                if name not in self._cache:
                    f = self._m5_dir / f"{name}.parquet"
                    if not f.exists():
                        raise KeyError(f"5 分钟口径原料不存在: {f}")
                    self._cache[name] = pd.read_parquet(f).astype(np.float32)
                return self._cache[name]
            return super().raw(name)

    _M5Panel.__name__ = f"M5Panel_sig{int(signal)}_exe{int(exec_)}"
    return _M5Panel()


# ================= baostock 抓取(带超时、行数保护、失败重登与响应校验)=================
# 审查 P1(2026-10-10,前缀 m5 修复):baostock 所有请求共用一个 socket(baostock.common.context.default_socket),
# 而 baostock.util.socketutil.send_msg 用 `except Exception` 包住 recv。旧版超时异常继承 Exception,会被它吞掉
# (返回 None -> 10002007),服务器迟到的响应留在 socket 缓冲区里,**下一次请求读到的是上一只股票的数据**,
# 且不报错(本地假服务器复现:600000 超时后 000001 拿到 600000 的数据,之后持续错一位)。修复:
#   1. 超时异常 BsTimeout 继承 BaseException,不会被 baostock 内部吞掉;
#   2. 任何失败(超时 / 错误码 / 行数超限 / 校验不过)后先 bs_relogin():显式关闭旧 socket、重新 login 换新 socket;
#   3. 返回前校验响应属于本次请求:请求字段强制含 code,要求响应头 rs.code 与逐行 code 都等于请求代码,
#      日期都落在 [start, end] 内;不符抛 BsDesyncError(同样触发重登)。
# 历史桥接(m5_baostock_check,2025-12 30 只)当时 0 次失败,缓存数据不受影响(逐股收盘吻合率见 m5_baostock_by_stock.csv)。
class BsTimeout(BaseException):
    """baostock 请求超时。继承 BaseException:baostock 的 send_msg 用 `except Exception` 吞异常。"""


_Timeout = BsTimeout          # 旧名(scripts/m5_baostock_check.py 曾直接引用)


class BsDesyncError(RuntimeError):
    """响应不属于本次请求(代码或日期对不上):共享 socket 错位的症状,必须重登后再请求。"""


def _alarm(_s, _f):
    raise BsTimeout("baostock 请求超时")


class _deadline:
    """signal 定时器上下文:超时抛 BsTimeout。用 setitimer(支持小数秒);只能在主线程使用。"""

    def __init__(self, seconds: float):
        self.seconds = float(seconds)

    def __enter__(self):
        self.old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old)
        return False


def bs_code(ts_code: str) -> str:
    """'600000.SH' -> 'sh.600000'(baostock 不含北交所)。"""
    c, ex = ts_code.split(".")
    return f"{ex.lower()}.{c}"


def bs_login(timeout: float = 30):
    """带超时的 bs.login();失败抛 RuntimeError。login 内部会新建 socket(SocketUtil.connect)。"""
    import baostock as bs
    with _deadline(timeout):
        lg = bs.login()
    if lg is None or lg.error_code != "0":
        raise RuntimeError(f"baostock login 失败: {getattr(lg, 'error_code', None)} {getattr(lg, 'error_msg', '')}")
    return lg


def bs_relogin(timeout: float = 30):
    """失败后的标准恢复:logout(尽力而为)-> **显式关闭旧 socket**(丢弃缓冲区里迟到的响应)-> login 新 socket。

    baostock.logout() 只有在收到回包时才会关 socket;socket 已错位或断开时它会提前返回,所以这里自己关。"""
    import baostock as bs
    import baostock.common.context as ctx
    try:
        with _deadline(timeout):
            bs.logout()
    except (BsTimeout, Exception):
        pass
    s = getattr(ctx, "default_socket", None)
    if s is not None:
        try:
            s.close()
        except Exception:
            pass
        setattr(ctx, "default_socket", None)
    return bs_login(timeout)


def _on_fail(on_fail):
    if on_fail is None:
        return
    try:
        bs_relogin() if on_fail == "relogin" else on_fail()
    except (BsTimeout, Exception) as e:          # 重登失败不掩盖原异常;下一次请求会再失败并被记录
        print(f"baostock 重登失败: {type(e).__name__}: {e}", flush=True)


def bs_query_rows(code: str, fields: str, start: str, end: str, frequency: str, adjustflag: str = "3",
                  timeout: float = 60, max_rows: int = 20000, on_fail="relogin") -> pd.DataFrame:
    """baostock query_history_k_data_plus 的安全包装,返回字符串 DataFrame(列 = 响应 fields)。

    code        baostock 代码('sh.600000');fields 自动补上 date、code 两列用于校验。
    timeout     整个请求(含全部翻页)的秒数上限,超时抛 BsTimeout(BaseException)。
    max_rows    行数上限(rs.next() 空转保护)。
    on_fail     'relogin'(默认)= 任何失败后 bs_relogin() 换新 socket 再把原异常抛出;None = 不重登(仅测试用);
                也可传无参 callable。**失败后不重登就继续请求,下一只股票可能拿到上一只的数据**(见上方 P1 说明)。
    校验        翻页结束 error_code 必须为 "0";响应头 rs.code 与每行 code 都必须等于 code;每行 date ∈ [start, end]。
    只能在主线程调用(signal 定时器)。"""
    import baostock as bs
    flist = [f.strip() for f in fields.split(",") if f.strip()]
    for need in ("code", "date"):
        if need not in flist:
            flist.insert(0, need)
    try:
        with _deadline(timeout):
            rs = bs.query_history_k_data_plus(code, ",".join(flist), start_date=start, end_date=end,
                                              frequency=frequency, adjustflag=adjustflag)
            if rs is None or rs.error_code != "0":
                raise RuntimeError(f"baostock {code}: {getattr(rs, 'error_code', None)} {getattr(rs, 'error_msg', '')}")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
                if len(rows) > max_rows:
                    raise RuntimeError(f"baostock {code}: 超过最大行数 {max_rows}")
            if rs.error_code != "0":
                raise RuntimeError(f"baostock {code}: 翻页失败 {rs.error_code} {rs.error_msg}")
        if rs.code and rs.code != code:
            raise BsDesyncError(f"baostock 响应代码 {rs.code} ≠ 请求 {code}(socket 错位)")
        df = pd.DataFrame(rows, columns=rs.fields)
        if not df.empty:
            bad = df["code"] != code
            if bad.any():
                raise BsDesyncError(f"baostock 响应行代码 {sorted(set(df.loc[bad, 'code']))[:3]} ≠ 请求 {code}")
            dmin, dmax = df["date"].min(), df["date"].max()
            if dmin < start or dmax > end:
                raise BsDesyncError(f"baostock {code} 响应日期 [{dmin}, {dmax}] 超出请求 [{start}, {end}]")
        return df
    except (BsTimeout, Exception):
        _on_fail(on_fail)
        raise


def fetch_bs_5m(ts_code: str, start: str, end: str, adjustflag: str = "3", timeout: float = 60,
                max_rows: int = 20000, on_fail="relogin") -> pd.DataFrame:
    """抓一只股票 [start, end] 的 baostock 5 分钟线。调用方负责首次 login(bs_login())。

    超时 / 错误码 / 行数超限 / 响应不属于本次请求 -> 先重登(on_fail='relogin')再抛异常,由调用方记录、重试。
    adjustflag: '3' 不复权(默认;与前复权价只差每日常数因子,比价时按日归一)、'2' 前复权、'1' 后复权。
    返回列 time(bar 结束时刻)、open、high、low、close、volume、amount(float)。只能在主线程调用。
    """
    df = bs_query_rows(bs_code(ts_code), "date,time,code,open,high,low,close,volume,amount", start, end,
                       frequency="5", adjustflag=adjustflag, timeout=timeout, max_rows=max_rows, on_fail=on_fail)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"].str[:12], format="%Y%m%d%H%M")
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop(columns=["date", "code"]).set_index("time")


def reconcile_with_daily(P5: np.ndarray, O5: np.ndarray, V5: np.ndarray, close_d: np.ndarray,
                         open_d: np.ndarray, amount_d: np.ndarray):
    """用 baostock **日线**修正 5 分钟线的已知缺陷(2025-12 30 只股票核对,research/ledger/m5_baostock_*.csv)。

    事实:baostock 日线的收盘/开盘/成交额与 1 分钟数据逐日一致(690/690 股票日在 1e-4 内);但上交所股票的
    5 分钟线 15:00 那根常缺一部分收盘集合竞价(46% 的 SH 股票日 5 分钟加总 < 日成交额,最多少 3.1%;
    缺口集中在 15:00 bar,15:00 之前的累计在 0.1% 内吻合的占 95%),且 24% 的 SH 股票日最后一根收盘 ≠ 日线收盘。
    深交所逐 bar 吻合(收盘 99.9%、成交额 99.5% 在 0.1% 内)。
    修正:bar47 收盘 := 日线收盘;bar0 开盘 := 日线开盘;bar47 成交额 := 日成交额 - 前 47 根之和(为负时不动)。
    输入 (48, N) 与 (N,);价格须与 P5 同一复权口径。返回修正后的副本。"""
    P5, O5, V5 = P5.copy(), O5.copy(), V5.astype(np.float64).copy()
    P5[-1] = np.where(np.isfinite(close_d), close_d, P5[-1])
    O5[0] = np.where(np.isfinite(open_d), open_d, O5[0])
    rest = amount_d - np.nansum(V5[:-1], axis=0)
    V5[-1] = np.where(np.isfinite(rest) & (rest >= 0), rest, V5[-1])
    return P5, O5, V5


# ================= 前向检验辅助:历史 n 日滚动窗口超额分布 =================
# 审查 P0(2026-10-10,前缀 m5 修复):前向检验只能从空仓起步(2026-01 ~ 08-30 不得计算任何策略收益)。
# 引擎把 k 个调仓相位锚在 date_mask 首日 + phase(strategy.simulate `t0 += int(phase)`),相位 ph 第 ph 个信号日
# 才首次建仓,所以 k=5 相位平均后前 4 个信号日的仓位依次约 0.2 / 0.4 / 0.6 / 0.8,而池基准从首日满仓。
# 这条"现金腿"让冷启动窗口的超额 ≈ 连续运行的超额 − 0.5 × 前 4 日基准累计收益,误差远大于 5 分钟近似误差
# (旧文档称"只多一笔 1bp 建仓成本、可忽略",是错的)。约定(预注册,前向与历史分布同一口径):
#   - 前向从首个前向信号日冷启动,**前 FWD_SKIP_HEAD = k-1 = 4 个信号日只建仓、不计入窗口超额**;
#     评估窗 = 第 5 个信号日起的 n 天(此时 5 个相位都已建仓;同频基准的 5 个相位也都已在窗口内重置过,
#     评估窗内的基准与 mask 起点之前的预热无关)。
#   - 历史参考分布 = 连续运行序列去掉开头 4 个信号日(其中含 2021-01 的建仓爬坡)后的全部 n 日窗口。
#   - 基准必须是**同频**池等权 strategy.benchmark(..., rebalance=5)(ENGINE_IMPACT_MODEL §9 第 11 条);
#     审查 P1:旧版分布对每日重置的 k=1 基准,与同频口径的中位差 ≈ 1.6 个百分位点,已大于近似误差。
# 历史验证(m5_bridge.py coldstart 阶段,research/ledger/m5_coldstart_*.csv):逐窗口冷启动 + 跳过 4 日,与连续运行
# 同一评估窗的超额差、百分位漂移见报告 MINUTE5_BRIDGE.md §5.5。
FWD_REBALANCE = 5
FWD_SKIP_HEAD = FWD_REBALANCE - 1


def window_excess_distribution(rets_excess, n_days: int, how: str | None = None,
                               skip_head: int = 0) -> pd.Series:
    """历史上所有 n 日滚动窗口的累计超额(按窗口末日索引),供前向检验算百分位。

    rets_excess
      - pd.Series:逐日超额 = 策略日收益 - 基准日收益(两者同为信号日索引,strategy.benchmark 口径);
        默认 how='sum':窗口超额 = Σ 日超额(算术累计)。
      - pd.DataFrame,含 'strategy' 与 'benchmark' 两列逐日收益:默认 how='compound':
        窗口超额 = Π(1+r_策略) - Π(1+r_基准)(与"窗口内策略累计收益减基准累计收益"一致);也可 how='sum'。
    n_days       窗口长度(交易日,按序列行数计)。
    skip_head    先去掉序列开头 skip_head 行再滚动(窗口只从第 skip_head 行起)。历史序列来自 date_mask 冷启动
                 的回测时,开头 k-1 天是建仓爬坡,应传 FWD_SKIP_HEAD(默认 0 = 旧行为)。
    前向检验必须用**同一定义**计算自己的 n 日超额再查百分位:forward_window_excess() + percentile_in(),
    或直接用 reference_distribution() 取账本里的历史分布。

    ⚠ 读法:滚动窗口高度重叠,~1200 个窗口只相当于约 1200/n 个独立样本;分布只用于描述"历史上同长度窗口
    落在哪里",不能当显著性检验。
    ⚠ 冷启动(审查 P0):前向从空仓起步时,k=5 相位平均的前 4 个信号日平均仓位只有 0.2→0.8,基准满仓;
    这条现金腿的误差远大于近似误差,**不是**"一笔 1bp 建仓成本"。前向窗口必须跳过前 FWD_SKIP_HEAD 天
    (forward_window_excess),历史分布用 skip_head=FWD_SKIP_HEAD 并对同频基准算(见上方模块注释)。
    """
    n = int(n_days)
    if n < 1:
        raise ValueError("n_days 必须 >= 1")
    sk = int(skip_head)
    if sk < 0:
        raise ValueError("skip_head 必须 >= 0")
    if isinstance(rets_excess, pd.DataFrame):
        if not {"strategy", "benchmark"} <= set(rets_excess.columns):
            raise ValueError("DataFrame 输入需含 'strategy' 与 'benchmark' 两列")
        how = how or "compound"
        df = rets_excess[["strategy", "benchmark"]].dropna().iloc[sk:]
        if how == "compound":
            ls = np.log1p(df["strategy"]).rolling(n, min_periods=n).sum()
            lb = np.log1p(df["benchmark"]).rolling(n, min_periods=n).sum()
            out = np.expm1(ls) - np.expm1(lb)
        elif how == "sum":
            out = (df["strategy"] - df["benchmark"]).rolling(n, min_periods=n).sum()
        else:
            raise ValueError(how)
    else:
        how = how or "sum"
        if how != "sum":
            raise ValueError("Series 输入只支持 how='sum'(复利口径需要策略与基准两条序列)")
        out = pd.Series(rets_excess).dropna().iloc[sk:].rolling(n, min_periods=n).sum()
    out = out.dropna()
    out.name = f"excess_{how}_{n}d"
    return out


def forward_window_excess(df: pd.DataFrame, skip_head: int = FWD_SKIP_HEAD, n_days: int | None = None,
                          how: str = "compound") -> tuple[float, int]:
    """前向检验的窗口超额(与 window_excess_distribution 同一定义),返回 (超额, 评估天数 n)。

    df        冷启动前向回测的逐日收益,列 'strategy'、'benchmark'(信号日索引,benchmark 为同频
              strategy.benchmark(..., rebalance=5)),从首个前向信号日起。
    skip_head 前几个信号日只建仓、不计入(默认 FWD_SKIP_HEAD = 4,预注册,不得事后改)。
    n_days    None = 跳过之后的全部天数;给定时只取跳过之后的前 n_days 天(不足则报错)。
    查百分位时历史分布要用同一个 n:reference_distribution(plan, n) 或
    window_excess_distribution(hist, n, skip_head=FWD_SKIP_HEAD)。"""
    if not {"strategy", "benchmark"} <= set(df.columns):
        raise ValueError("需含 'strategy' 与 'benchmark' 两列")
    d = df[["strategy", "benchmark"]].dropna().iloc[int(skip_head):]
    if n_days is not None:
        if len(d) < int(n_days):
            raise ValueError(f"跳过 {skip_head} 天后只剩 {len(d)} 天 < n_days={n_days}")
        d = d.iloc[:int(n_days)]
    if d.empty:
        raise ValueError("跳过建仓期后没有可评估的信号日")
    if how == "compound":
        v = float(np.prod(1.0 + d["strategy"].to_numpy()) - np.prod(1.0 + d["benchmark"].to_numpy()))
    elif how == "sum":
        v = float((d["strategy"] - d["benchmark"]).sum())
    else:
        raise ValueError(how)
    return v, int(len(d))


def reference_distribution(plan: str, n_days: int, how: str = "compound", config: str = "C",
                           daily_path=None) -> pd.Series:
    """前向检验查百分位用的历史参考分布(预注册口径):从 research/ledger/m5_bridge_daily.parquet 读
    2021-2025 相位平均回测的逐日收益,对**同频**(5 日)池等权基准,去掉开头 FWD_SKIP_HEAD 天后取全部 n 日窗口。

    plan    '排雷缓冲20/35 r5' 或 'Top20 周频';config 'A' / 'B' / 'C'(前向处境 = C 全近似)。
    ⚠ 信号样本内(COMP_v5 在 2020-2025 选出),前向百分位应预期偏低;数据有幸存者偏差。"""
    from afactor.config import LEDGER
    D = pd.read_parquet(LEDGER / "m5_bridge_daily.parquet" if daily_path is None else daily_path)
    cols = [c for c in D.columns if c.startswith(f"{plan}|{config}")]
    s = [c for c in cols if c.endswith("|strategy")]
    b = [c for c in cols if c.endswith("|benchmark")]
    if len(s) != 1 or len(b) != 1:
        raise KeyError(f"m5_bridge_daily 里找不到 {plan} / {config} 的 strategy、benchmark 列: {cols}")
    df = pd.DataFrame({"strategy": D[s[0]], "benchmark": D[b[0]]})
    return window_excess_distribution(df, n_days, how=how, skip_head=FWD_SKIP_HEAD)


def percentile_in(value: float, dist: pd.Series) -> float:
    """value 在历史分布中的百分位(0~100,≤ value 的比例;平局取中)。"""
    v = np.asarray(dist, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite(value):
        return np.nan
    return float(100.0 * ((v < value).sum() + 0.5 * (v == value).sum()) / v.size)
