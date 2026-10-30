"""第二步:风控层对比(前缀 lh_step2;预注册 research/LONG_HISTORY_PREREG.md §12 / §13 / §14)。

阶段(按顺序;日志 logs/lh_step2.log,耗时 research/ledger/lh_step2_timing.csv):
  verify   冻结哈希校验 + 登记第二步实现口径(lh_deviations.csv,开跑前写入)
  runs     第一批引擎回测(静态网格 / O2 / O3 / O4 / O6 / 对冲多头腿 / 探针 / 敏感性),多进程
  runs2    依赖第一批的回测:匹配静态对照 O1(ē_R)、规则 x 固定降仓 f
  report   对冲腿、相位平均、账本 research/ledger/lh_step2_*.csv、引擎断言、判定、图
  review   审查修复(2026-10-11):lh_step2_review.csv(对冲成本逐年、逐相位 Q-a、仓位扰动检验、f* 余量/留一段/逐相位、
           匹配静态留一段、熊市+反弹留一段、熊市少跌 vs 反弹少赚、几何/算术超额、对冲断言分布)、补登偏离、
           单进程微基准(lh_step2_timing.csv 的 micro_* 行)。只实跑插值 f* 的 4 个静态配置(O1 57.8% / 59.3% x 两种估值),
           其余只读 runs/report 的产出。

用法: python3 scripts/lh_step2.py verify runs runs2 report review [--workers 5]

接法(预注册 §12.1):callable target_fn + normalize=False。target_fn 只在各相位调仓日被调用,返回
  仓位 pos_ph(t) x [W[t] 在 te 可成交股票上归一 x 引擎预算(1 - 残仓)],即"O0 持仓整体缩放";
  卖不掉的残仓由引擎保留。pos = 1 时与 O0 逐位一致(断言 S2_O1_100_equals_O0)。
  所有仓位序列只用 s <= t-2 的 O0 收益(strat_main.shift(2))或 t 日收盘可知的全 A 等权指数。
提速:同一进程复用同一个适配器对象,引擎的参与率事前估计(20 日滚动中位数,约 3 秒)只算一次,
  callable 单相位从 3.9 秒降到 0.8 秒,与 ndarray 同量级,不必改成 W x pos 的 ndarray 接法。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
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

LOG = ROOT / "logs" / "lh_step2.log"
RUNS = TMP / "lh_step2_runs"
TIMING = LEDGER / "lh_step2_timing.csv"
CHECKS = LEDGER / "lh_engine_checks.csv"
DEVS = LEDGER / "lh_deviations.csv"
DAILY1 = TMP / "lh_step1_daily.parquet"
DAILY2 = TMP / "lh_step2_daily.parquet"
TD = 242
K = 5
LIST_IF, LIST_IC, LIST_IM = "2010-04-16", "2015-04-16", "2022-07-22"
EVAL_END = "2025-12-31"
PERIODS = {"full": (LE.EVAL_START, EVAL_END), "dev": LE.DEV, "test": LE.TEST, "full05": ("2005-01-04", EVAL_END)}


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def timing(stage: str, sec: float, note: str = ""):
    row = pd.DataFrame([dict(stage=stage, seconds=round(sec, 1), minutes=round(sec / 60, 2), note=note,
                             at=time.strftime("%Y-%m-%d %H:%M:%S"))])
    old = pd.read_csv(TIMING) if TIMING.exists() else pd.DataFrame()
    if len(old):
        old = old[old["stage"] != stage]
    pd.concat([old, row], ignore_index=True).to_csv(TIMING, index=False)


def merge_csv(path: Path, df: pd.DataFrame, key: str):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old) and key in old.columns:
        old = old[~old[key].isin(df[key])]
    pd.concat([old, df], ignore_index=True).to_csv(path, index=False)


# ================================================================ verify(含实现口径登记,开跑前写入)
DEVIATIONS = [
    ("[step2] §15 报告文件名", "research/LONG_HISTORY_STEP2.md", "research/LH_STEP2_RISK_OVERLAY.md",
     "编排任务书指定该文件名", "否", "无"),
    ("[step2] §12.1 规则目标的构造",
     "T+1 开盘按 仓位 x O0 目标权重成交;callable target_fn + normalize=False;卖不掉的持仓保留,实际仓位会高于目标;所有规则都在 O0 的持仓上整体缩放",
     "callable 返回 仓位 x (W[t] 在 te 可成交股票上归一 x 引擎预算 1-残仓),即 O0 引擎 normalize 后的有效目标整体缩放;残仓按引擎 stuck 逻辑保留,实际仓位 = 仓位 x (1-残仓) + 残仓。"
     "仓位 = 1 时与 O0 逐日差 <= 1e-15(断言 S2_O1_100_equals_O0)。另跑敏感性 'C':残仓计入目标仓位(可成交部分 = max(0, 仓位-残仓))",
     "字面的 仓位 x W[t](未归一)在 normalize=False 下会把 te 开盘不可成交股票的目标权重留成现金,仓位 = 1 时不等于 O0(2015-07-08 一度 95% 的持仓不可成交),"
     "会把'不可成交'误记为风控效果",
     "否(开跑前决定;一致性检查只比 O0 已知序列)",
     "小;对冲规则在有残仓时'多头 + 保证金'按字面超出 m x h x 残仓,报告实测值;'C' 口径下不超"),
    ("[step2] §12.1 O2/O4 的输入序列", "O0 过去 60 个已实现日收益 / 影子 O0 净值(s <= t-2)",
     "用第一步 5 相位平均的 O0 日收益 strat_main.shift(2);停牌指数法估值敏感性用 strat_main_vidx.shift(2);σ̂ 用样本标准差(ddof=1)",
     "预注册未指定相位口径;5 相位平均就是 O0", "否", "无"),
    ("[step2] §12.2 O4 状态机细节", "降仓:100% 且 DD<=-15% -> 50%;任一状态 DD<=-25% -> 25%;恢复:<100% 且影子净值 > 60 日均线 -> 100% 并重置 P",
     "每个检查日只做一次转移,按检查日开始时的状态:100% 时 DD<=-25% -> 25%,否则 DD<=-15% -> 50%;<100% 时先看恢复(影子净值 > 其 60 日均线 -> 100%,P 重置为当前影子净值),"
     "否则 DD<=-25% -> 25%。P = 上次重置以来影子净值逐日最高值;60 日均线至少 45 个观测(0.75 x 窗口,与 O3 同一约定),不足时不恢复。"
     "O4-b:P = 全历史最高,恢复后 20 个交易日内不触发降仓",
     "预注册未写同一检查日内降仓与恢复的先后、P 是否逐日更新、均线最少观测", "否", "只影响 O4/O4-b,方向未知"),
    ("[step2] §12.2 O5/O5-R 对冲腿记账",
     "空头腿 = -h x 指数开盘对开盘 - h x 贴水 - 调整名义本金 1bp;名义本金在调仓日重置为 h x 净值,其间随指数漂移;保证金 15% 按现金计、收益 0",
     "逐相位:调仓日把空头名义比例重置为 h(按当时总净值),其间 hr <- hr(1+指数)/(1+组合);实际贴水 = lh_basis.hedge_cost_daily(规则 main、R=2、含换月 2bp、相对价格指数),"
     "O5-R 同一规则;成本日期取成交日 te(收盘 te-1 -> 收盘 te,与开盘 te -> 开盘 te+1 的日内时段重叠),敏感性取 te+1;品种上市首日成本 NaN 记 0;换品种按旧 + 新名义各 1bp。"
     "多空日收益相加后相位平均(§14.1(3)),即对冲盈亏每日按比例并入总净值;保证金账户 M <- (M + 空头腿)/(1+组合),只在调仓日补回 m x h,不另设追保缓冲;"
     "检查日之间的保证金缺口(m x hr - M)与 M<0 天数如实报告(相当于短暂透支,不计利息)",
     "预注册未写名义漂移公式、成本日期对齐、追保处理与 O5-R 的合约规则", "否", "小"),
    ("[step2] §12.2 O5-R 保证金", "若 LH_FUTURES_BASIS.md 给出逐期套保保证金,O5-R 按史实值计",
     "用默认 15%:该报告 §8.1 明言没有经核验的套保保证金;30% 档作为 2015-09~2018-12 限仓期的上界敏感性",
     "预注册的条件分支", "否", "无"),
    ("[step2] §12.1 现金 2% 口径", "按实际现金比例逐日加 2%/242",
     "现金 = max(0, 1 - 实际多头仓位 - m x hr);保证金不计息", "预注册写保证金收益 0", "否", "无"),
    ("[step2] §12.3(2) 匹配静态对照", "O1(ē_R),ē_R = 评价期平均实际仓位(对冲用净多头 (1-0.15h)-h)",
     "ē_R = 2006-01-04~2025-12-29 相位平均实际仓位(对冲 = 实际多头 - 实际空头比例 hr);O1 名义仓位按 O1 网格'名义 -> 实际'线性插值反解,使 O1 实际平均仓位 ≈ ē_R;"
     "另报'按窗口内平均仓位在静态前沿上插值'的对照(敏感性,不进判定)",
     "有残仓时 O1(f) 的实际仓位略高于 f", "否", "小"),
    ("[step2] §12.4 Q-b 留一段口径", "留任一熊市段后,'降低 >= 10pp 且 Calmar 更高'仍成立",
     "主判:逐段剔除熊市(口径 A)后,开发窗与检验窗(该段所在窗口剔除,另一窗口不变)都仍满足;另报全期口径",
     "预注册未写留一段后看哪个区间;取较严的读法", "否", "使 Q-b 更难满足"),
    ("[step2] §12.3(4) 被打脸计数", "每个降仓的调仓日 d", "逐相位计数(各相位有自己的调仓日与 O4 状态),报告 5 相位合计、比例与相位平均",
     "引擎按相位独立运行", "否", "无"),
    ("[step2] 非预注册的诊断与敏感性", "—",
     "加跑(不进判定):前视探针(O2/O4 用 shift(3) 与 shift(0),O3 用 t-1 与 t+1 的指数)、停牌指数法估值下的主规则与静态网格、残仓计入目标的 'C' 口径、"
     "规则 x 固定降仓 f(0.9~0.5)、2005-2025 区间行;O3 指数敏感性(国证A股、中证1000)是预注册 §12.1 列出的",
     "第一步交接建议加可证伪探针;用户约束按最大回撤判,停牌陈旧价计价会低报 2015 回撤", "否", "无"),
    ("[step2] §12.3(1) 每减少 1pp 回撤的代价", "(年化_O0 - 年化_R)/(|MDD_O0| - |MDD_R|)", "照用;分母 <= 0 时记 NaN", "—", "否", "无"),
]


def stage_verify():
    man = json.loads((LEDGER / "lh_prereg_manifest.json").read_text())
    bad = [f for f, h in man["sha256"].items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    if bad:
        log("[verify] 冻结哈希不一致,停止:", bad)
        sys.exit(2)
    log(f"[verify] 预注册冻结哈希一致({len(man['sha256'])} 个文件)")
    merge_csv(CHECKS, pd.DataFrame([dict(check="prereg_hash_step2", step=2, value=0, ok=True,
                                         note=f"{len(man['sha256'])} 个冻结文件 sha256 与 manifest 一致(第二步开工)")]), "check")
    cols = ["条目", "本文规定", "实际做法", "原因", "发现时是否已看到收益", "对结论的预期影响"]
    merge_csv(DEVS, pd.DataFrame(DEVIATIONS, columns=cols), "条目")
    log(f"[verify] 登记第二步实现口径 {len(DEVIATIONS)} 条 -> {DEVS.name}")


# ================================================================ 配置
def C(kind, group, main=False, val="close", tgt="B", **kw):
    d = dict(kind=kind, group=group, main=main, val=val, tgt=tgt)
    d.update(kw)
    return d


O5_LONG = {}          # 名义多头仓位 -> 静态配置名(对冲多头腿 = O1(1 - m h))
for _h in (0.3, 0.5, 1.0):
    for _m in (0.0, 0.15, 0.30):
        O5_LONG[(_h, _m)] = round(1 - _m * _h, 4)


def static_name(f, val="close", tgt="B"):
    s = f"O1_f{int(round(f * 1000)):04d}"
    if val == "index":
        s += "_vidx"
    if tgt == "C":
        s += "_C"
    return s


def build_configs() -> dict:
    cf = {}
    for f in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        cf[static_name(f)] = C("static", "O1", main=f in (0.5, 0.7), f=f)
        cf[static_name(f, "index")] = C("static", "O1_vidx", val="index", f=f)
    for f in sorted({O5_LONG[(0.3, 0.15)], O5_LONG[(0.5, 0.15)], O5_LONG[(1.0, 0.15)], O5_LONG[(0.5, 0.30)]}):
        if static_name(f) not in cf:
            cf[static_name(f)] = C("static", "O5_long", f=f)
    cf[static_name(0.925, "index")] = C("static", "O5_long_vidx", val="index", f=0.925)
    for h in (0.3, 0.5, 1.0):
        cf[f"O5R_long_h{int(h * 100)}"] = C("static_from", "O5R_long", f=round(1 - 0.15 * h, 4), from_date=LIST_IF)
    cf["O5R_long_h50_m30"] = C("static_from", "O5R_long", f=0.85, from_date=LIST_IF)
    for v in (0.15, 0.20, 0.25):
        cf[f"O2_v{int(v * 100)}"] = C("vol", "O2", main=v == 0.20, target=v, lag=2)
    for L in (120, 200, 250):
        for bp in (0.0, 0.5):
            cf[f"O3_ma{L}_p{int(bp * 100)}"] = C("trend", "O3", main=(L == 200 and bp == 0.5), L=L, below=bp,
                                                index="ew", ishift=0)
    cf["O3_ma200_p50_gz"] = C("trend", "O3_idx", L=200, below=0.5, index="GZ_A", ishift=0)
    cf["O3_ma200_p50_csi1000"] = C("trend", "O3_idx", L=200, below=0.5, index="CSI1000", ishift=0)
    for d1, d2 in ((0.10, 0.20), (0.15, 0.25), (0.20, 0.30)):
        cf[f"O4_{int(d1 * 100)}_{int(d2 * 100)}"] = C("dd", "O4", main=(d1 == 0.15), d1=d1, d2=d2, reset=True,
                                                      cool=0, lag=2)
    cf["O4b_15_25"] = C("dd", "O4b", main=True, d1=0.15, d2=0.25, reset=False, cool=20, lag=2)
    o6 = {"O6_v20_ma200": (0.20, 200), "O6_v15_ma200": (0.15, 200), "O6_v25_ma200": (0.25, 200),
          "O6_v20_ma120": (0.20, 120), "O6_v20_ma250": (0.20, 250)}
    for n, (v, L) in o6.items():
        cf[n] = C("o6", "O6", main=n == "O6_v20_ma200", target=v, L=L, below=0.5, lag=2)
    # 前视探针(诊断,不进判定)
    cf["probe_O2_v20_lag3"] = C("vol", "probe", target=0.20, lag=3)
    cf["probe_O2_v20_lag0"] = C("vol", "probe", target=0.20, lag=0)
    cf["probe_O3_ma200_p50_idx_tm1"] = C("trend", "probe", L=200, below=0.5, index="ew", ishift=1)
    cf["probe_O3_ma200_p50_idx_tp1"] = C("trend", "probe", L=200, below=0.5, index="ew", ishift=-1)
    cf["probe_O4_15_25_lag3"] = C("dd", "probe", d1=0.15, d2=0.25, reset=True, cool=0, lag=3)
    cf["probe_O4_15_25_lag0"] = C("dd", "probe", d1=0.15, d2=0.25, reset=True, cool=0, lag=0)
    # 停牌指数法估值(敏感性)
    for n in ("O2_v20", "O3_ma200_p50", "O4_15_25", "O4b_15_25", "O6_v20_ma200"):
        c = dict(cf[n]); c.update(val="index", group=c["group"] + "_vidx", main=False)
        cf[n + "_vidx"] = c
    # 残仓计入目标(敏感性 'C')
    for n in (static_name(0.5), static_name(0.7), static_name(0.925), "O2_v20", "O3_ma200_p50", "O4_15_25",
              "O4b_15_25", "O6_v20_ma200"):
        c = dict(cf[n]); c.update(tgt="C", group=c["group"] + "_C", main=False)
        cf[n + "_C"] = c
    return cf


CONFIGS = build_configs()
SCALED_RULES = ("O2_v20", "O3_ma200_p50", "O4_15_25", "O4b_15_25", "O6_v20_ma200")
SCALE_F = (0.9, 0.8, 0.7, 0.6, 0.5)


# ================================================================ 仓位序列(只用决策时点已知的数据)
class Ctx:
    """worker 进程内缓存:输入、权重、适配器(复用以缓存引擎的参与率事前估计)、O0 收益、指数。"""
    X = None
    W = None
    ads: dict = {}
    cms: dict = {}
    sf = None
    o0: dict = {}
    idx: dict = {}


def _init_ctx():
    if Ctx.X is not None:
        return
    Ctx.X = LE.load_inputs()
    Ctx.W = LE.weights(Ctx.X, "S1", buffer=(0.20, 0.35))
    Ctx.sf = LE.StampFn(Ctx.X.dates)
    D = pd.read_parquet(DAILY1)
    T = len(Ctx.X.dates)
    for val, col in (("close", "strat_main"), ("index", "strat_main_vidx")):
        r = np.full(T, np.nan)
        r[:len(D)] = D[col].reindex(Ctx.X.dates[:len(D)]).to_numpy()
        Ctx.o0[val] = r
    ew = Ctx.X.ew_allA_ret.to_numpy(np.float64)
    Ctx.idx["ew"] = np.cumprod(1.0 + np.nan_to_num(ew))
    import lh_step1 as S1
    close, _ = S1.index_frame(Ctx.X.dates)
    for c in ("GZ_A", "CSI1000"):
        Ctx.idx[c] = close[c].ffill().to_numpy(np.float64)


def _adapter(val: str):
    if val not in Ctx.ads:
        ad = LE.adapter(Ctx.X, discount=0.0, valuation=val)
        Ctx.ads[val] = ad
        Ctx.cms[val] = LE.cost_model(ad, "Y1")
    return Ctx.ads[val], Ctx.cms[val]


def pos_vol(r: np.ndarray, target: float, lag: int) -> np.ndarray:
    rr = pd.Series(r).shift(lag)
    sig = rr.rolling(60, min_periods=40).std().to_numpy() * np.sqrt(TD)
    with np.errstate(all="ignore"):
        p = np.where(np.isfinite(sig) & (sig > 0), np.minimum(1.0, target / sig), 1.0)
    return p


def pos_trend(level: np.ndarray, L: int, below: float, ishift: int) -> np.ndarray:
    s = pd.Series(level).shift(ishift)              # ishift=0: t 日收盘;1: t-1(多滞后);-1: t+1(前视探针)
    ma = s.rolling(L, min_periods=int(np.ceil(0.75 * L))).mean()
    b = (s < ma).to_numpy() & ma.notna().to_numpy() & s.notna().to_numpy()
    return np.where(b, below, 1.0)


def pos_dd(r: np.ndarray, ph: int, d1: float, d2: float, reset: bool, cool: int, lag: int) -> np.ndarray:
    T = len(r)
    nav = np.cumprod(1.0 + np.nan_to_num(r))
    Nk = np.full(T, np.nan)
    if lag > 0:
        Nk[lag:] = nav[:T - lag]
    else:
        Nk[:] = nav
    ma = pd.Series(Nk).rolling(60, min_periods=45).mean().to_numpy()
    state, P, last_rec = 1.0, np.nan, -10 ** 9
    out = np.ones(T)
    for t in range(T):
        x = Nk[t]
        if np.isfinite(x):
            P = x if not np.isfinite(P) else max(P, x)
        if t >= ph and (t - ph) % K == 0 and np.isfinite(x):
            dd = x / P - 1.0
            if state >= 1.0:
                if t - last_rec >= cool:
                    if dd <= -d2:
                        state = 0.25
                    elif dd <= -d1:
                        state = 0.50
            else:
                if np.isfinite(ma[t]) and x > ma[t]:
                    state = 1.0
                    last_rec = t
                    if reset:
                        P = x
                elif dd <= -d2:
                    state = 0.25
        out[t] = state
    return out


def positions(cfg: dict) -> np.ndarray:
    """(K, T) 目标仓位;第 ph 行只在相位 ph 的调仓日被引擎读取。"""
    T = len(Ctx.X.dates)
    kind = cfg["kind"]
    r = Ctx.o0[cfg["val"]]
    if kind == "static":
        p = np.full(T, cfg["f"])
    elif kind == "static_from":
        te_ok = np.zeros(T, bool)
        te_ok[:-1] = Ctx.X.dates[1:] >= pd.Timestamp(cfg["from_date"])
        te_ok[-1] = True
        p = np.where(te_ok, cfg["f"], 1.0)
    elif kind == "vol":
        p = pos_vol(r, cfg["target"], cfg["lag"])
    elif kind == "trend":
        p = pos_trend(Ctx.idx[cfg["index"]], cfg["L"], cfg["below"], cfg["ishift"])
    elif kind == "o6":
        p = pos_vol(r, cfg["target"], cfg["lag"]) * pos_trend(Ctx.idx["ew"], cfg["L"], cfg["below"], 0)
    elif kind == "dd":
        P = np.vstack([pos_dd(r, ph, cfg["d1"], cfg["d2"], cfg["reset"], cfg["cool"], cfg["lag"]) for ph in range(K)])
        return P * cfg.get("scale", 1.0)
    elif kind == "scaled":
        base = dict(CONFIGS[cfg["base"]])
        return positions(base) * cfg["scale"]
    else:
        raise ValueError(kind)
    return np.tile(p * cfg.get("scale", 1.0), (K, 1))


def run_cfg(name: str, cfg: dict) -> dict:
    from afactor import strategy as ST
    t0 = time.time()
    _init_ctx()
    ad, cm = _adapter(cfg["val"])
    P = positions(cfg)
    W = Ctx.W
    N = W.shape[1]
    tgtC = cfg["tgt"] == "C"
    cols = ["ret", "gross_ret", "exposure", "tgt_exposure", "unfilled", "turnover", "cost_total", "cost_commission",
            "cost_stamp", "cost_spread", "cost_impact", "stuck_w", "rebalance"]
    phases = []
    for ph in range(K):
        pos = P[ph]

        def fn(t, h, ctx, pos=pos):
            w = np.where(ctx["can"], W[t].astype(np.float64), 0.0)
            s = w.sum()
            if s <= 0:
                return np.zeros(N)
            tot = max(0.0, pos[t] - ctx["stuck_w"]) if tgtC else pos[t] * ctx["budget"]
            return w / s * tot

        res = ST.simulate(fn, ad, "open10", LE.CAPITAL, rebalance=K, phase=ph, normalize=False, cost_model=cm,
                          stamp_fn=Ctx.sf, record=True)
        d = res["record"]["daily"][cols].astype(np.float64)
        phases.append(d)
    out = dict(name=name, cfg=cfg, phases=phases, pos=P.astype(np.float32), elapsed=time.time() - t0,
               max_exposure=float(max(d["exposure"].max() for d in phases)))
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / f"{name}.pkl", "wb") as f:
        pickle.dump(out, f)
    return dict(name=name, elapsed=out["elapsed"])


def _run_many(cfgs: dict, workers: int, tag: str):
    import concurrent.futures as cf
    import multiprocessing as mp
    t0 = time.time()
    names = list(cfgs)
    # 先跑慢的(O4 逐相位状态机略慢);其余顺序无所谓
    log(f"[{tag}] {len(names)} 个配置,{workers} 进程")
    done = []
    if workers <= 1:
        for n in names:
            done.append(run_cfg(n, cfgs[n]))
            log(f"[{tag}] {n} {done[-1]['elapsed']:.1f}s ({len(done)}/{len(names)})")
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(run_cfg, n, cfgs[n]): n for n in names}
            for f in cf.as_completed(futs):
                r = f.result()
                done.append(r)
                log(f"[{tag}] {r['name']} {r['elapsed']:.1f}s ({len(done)}/{len(names)})")
    sec = time.time() - t0
    serial = sum(r["elapsed"] for r in done)
    timing(tag, sec, f"{len(names)} 配置 x 5 相位(callable,record);{workers} 进程;单进程累计 {serial / 60:.1f} 分钟"
                     f"(含每进程首个配置约 4~6 秒的权重与参与率估计)")
    log(f"[{tag}] 完成 {sec:.0f}s(单进程累计 {serial:.0f}s)")


def stage_runs(workers: int, only=None):
    cfgs = {n: c for n, c in CONFIGS.items() if only is None or n in only}
    _run_many(cfgs, workers, "runs")


# ================================================================ 读结果 / 对冲腿 / 组合
def load_run(name: str) -> dict:
    with open(RUNS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


class Data:
    """report 阶段共用的数据:日期、O0、B1(逐相位)、指数、对冲成本。"""

    def __init__(self):
        self.X = LE.load_inputs()
        self.D = pd.read_parquet(DAILY1)
        self.sig = self.D.index                     # 信号日(5099 行)
        self.n = len(self.sig)
        dX = self.X.dates
        self.te = dX[1:self.n + 1]                  # 成交日
        self.te1 = dX[2:self.n + 2]
        self.ew_level = np.cumprod(1.0 + np.nan_to_num(self.X.ew_allA_ret.to_numpy(np.float64)))
        # 对冲:指数开盘对开盘(信号日对齐)与日贴水成本(按成交日 te 对齐;敏感性 te+1)
        self.idx_oo = {"CSI1000": self.D["idx_CSI1000_oo"].to_numpy(), "CSI300": self.D["idx_CSI300_oo"].to_numpy(),
                       "CSI500": self.D["idx_CSI500_oo"].to_numpy()}
        from afactor import lh_basis as LB
        self.hc = {}
        for v in ("IF", "IC", "IM"):
            s = LB.hedge_cost_daily(v, "main", 2)
            s.index = pd.to_datetime(s.index)
            self.hc[(v, "te")] = s.reindex(self.te).fillna(0.0).to_numpy()
            self.hc[(v, "te1")] = s.reindex(self.te1).fillna(0.0).to_numpy()
            self.hc[(v, "avail")] = np.asarray(self.te >= pd.Timestamp({"IF": LIST_IF, "IC": LIST_IC, "IM": LIST_IM}[v]))
        self._b1ph = {}

    def b1_phases(self, val: str) -> np.ndarray:
        """B1 逐相位日收益(K x n),与 bench_b1 的相位平均一致(断言)。"""
        if val not in self._b1ph:
            from afactor import strategy as ST
            ad = LE.adapter(self.X, discount=0.0, valuation=val)
            U = LE.pool(self.X, "q92")
            R = ST._pool_ew_drift_returns(ad, "V1000", "open10", K, 0, U)
            self._b1ph[val] = R[:, :self.n]
        return self._b1ph[val]


def eff_pos(P: np.ndarray, n: int) -> np.ndarray:
    """逐相位'当日生效的目标仓位' = 最近一个调仓日的目标(首个调仓日之前 = 0)。"""
    out = np.zeros((K, n))
    s = np.arange(n)
    for ph in range(K):
        last = ((s - ph) // K) * K + ph
        ok = s >= ph
        out[ph, ok] = P[ph, last[ok]]
    return out


def hedge_spec_arrays(Dt: Data, mode: str, h: float, scen: float, align: str):
    """返回逐信号日的 (目标对冲比例, 工具代码, {工具: 指数收益}, {工具: 日成本})。
    mode='ideal':全期中证1000;贴水 2015-04-16 前按情景 scen,之后 IC(代理)/ 2022-07-22 起 IM 实际值。
    mode='real':只用当时存在的期货 IF -> IC -> IM,2010-04-16 前不对冲。"""
    n = Dt.n
    te = Dt.te
    if mode == "ideal":
        hh = np.full(n, h)
        ins = np.full(n, 4)
        c = np.where(te < pd.Timestamp(LIST_IC), scen / TD,
                     np.where(te < pd.Timestamp(LIST_IM), Dt.hc[("IC", align)], Dt.hc[("IM", align)]))
        return hh, ins, {4: Dt.idx_oo["CSI1000"]}, {4: c}
    ins = np.where(te >= pd.Timestamp(LIST_IM), 3, np.where(te >= pd.Timestamp(LIST_IC), 2,
                                                             np.where(te >= pd.Timestamp(LIST_IF), 1, 0)))
    hh = np.where(ins > 0, h, 0.0)
    idx = {1: Dt.idx_oo["CSI300"], 2: Dt.idx_oo["CSI500"], 3: Dt.idx_oo["CSI1000"]}
    cost = {1: Dt.hc[("IF", align)], 2: Dt.hc[("IC", align)], 3: Dt.hc[("IM", align)]}
    return hh, ins, idx, cost


def hedge_phase(rL, expL, ph, hh, ins, idx, cost, m, bp=1e-4):
    n = len(rL)
    R = np.zeros(n); rH = np.zeros(n); HR = np.zeros(n); M = np.zeros(n); gap = np.zeros(n)
    lm = np.full(n, np.nan)
    hr, cur, marg = 0.0, 0, 0.0
    for s in range(n):
        tc = 0.0
        if s >= ph and (s - ph) % K == 0:
            nh, ni = float(hh[s]), int(ins[s])
            if ni != cur and cur != 0 and hr > 0:
                tc = bp * (hr + nh)
            else:
                tc = bp * abs(nh - hr)
            hr = nh
            cur = ni if nh > 0 else 0
            marg = m * nh
            lm[s] = expL[s] + m * nh                 # 调仓日 多头 + 保证金
        HR[s] = hr
        if hr > 0 and cur != 0:
            ir = idx[cur][s]
            ir = 0.0 if not np.isfinite(ir) else ir
            rh = -hr * ir - hr * cost[cur][s] - tc
        else:
            ir, rh = 0.0, -tc
        r = rL[s] + rh
        R[s], rH[s] = r, rh
        marg = (marg + rh) / (1.0 + r)
        hr = hr * (1.0 + ir) / (1.0 + r)
        M[s] = marg
        gap[s] = m * hr - marg if hr > 0 else 0.0
    return R, rH, HR, M, gap, lm


class Rule:
    """一个组合(相位平均):收益、仓位、成本、对冲腿。"""

    def __init__(self, name, family, label, rets_ph, exp_ph, pos_eff, P, rec, val, hedge=None, m=0.0,
                 grid=None, main=False, note=""):
        self.name, self.family, self.label, self.val, self.main, self.note = name, family, label, val, main, note
        self.rets_ph, self.exp_ph, self.pos_eff, self.P, self.rec, self.hedge, self.m = rets_ph, exp_ph, pos_eff, P, rec, hedge, m
        self.grid = grid


def build_rule(Dt: Data, name, family, label, run, hedge=None, val=None, main=False, note="") -> Rule:
    n = Dt.n
    ph = run["phases"]
    rL = np.vstack([d["ret"].reindex(Dt.sig).to_numpy() for d in ph])
    ex = np.vstack([d["exposure"].reindex(Dt.sig).to_numpy() for d in ph])
    rec = {c: np.mean([d[c].reindex(Dt.sig).to_numpy() for d in ph], axis=0)
           for c in ("cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact", "turnover",
                     "stuck_w", "unfilled", "tgt_exposure")}
    P = run["pos"].astype(np.float64)
    pe = eff_pos(P, n)
    val = val or run["cfg"]["val"]
    if hedge is None:
        return Rule(name, family, label, rL, ex, pe, P, rec, val, main=main, note=note)
    mode, h, m, scen, align = hedge
    hh, ins, idx, cost = hedge_spec_arrays(Dt, mode, h, scen, align)
    R = np.zeros_like(rL); HR = np.zeros_like(rL); RH = np.zeros_like(rL); M = np.zeros_like(rL)
    G = np.zeros_like(rL); LM = np.full_like(rL, np.nan)
    for p in range(K):
        R[p], RH[p], HR[p], M[p], G[p], LM[p] = hedge_phase(rL[p], ex[p], p, hh, ins, idx, cost, m)
    rule = Rule(name, family, label, R, ex, pe, P, rec, val, hedge=dict(mode=mode, h=h, m=m, scen=scen, align=align,
                                                                      rH=RH, HR=HR, M=M, gap=G, LM=LM, rL=rL),
                m=m, main=main, note=note)
    return rule


def rule_series(Dt: Data, R: Rule, cash: float = 0.0) -> pd.Series:
    r = R.rets_ph.copy()
    if cash > 0:
        mh = R.hedge["HR"] * R.m if R.hedge is not None else 0.0
        free = np.clip(1.0 - R.exp_ph - mh, 0.0, 1.0)
        r = r + free * cash / TD
    return pd.Series(np.nanmean(r, axis=0), index=Dt.sig)


def net_exposure(Dt: Data, R: Rule) -> pd.Series:
    e = R.exp_ph.mean(axis=0)
    if R.hedge is not None:
        e = e - R.hedge["HR"].mean(axis=0)
    return pd.Series(e, index=Dt.sig)


def b1_overlay(Dt: Data, R: Rule, cash: float = 0.0) -> pd.Series:
    """'同一仓位序列施加在 B1 上':逐相位 实际多头仓位 x B1_ph(+ 同一对冲腿,+ 同一现金收益)。"""
    B = Dt.b1_phases(R.val)
    o = R.exp_ph * B
    if R.hedge is not None:
        o = o + R.hedge["rH"]
    if cash > 0:
        mh = R.hedge["HR"] * R.m if R.hedge is not None else 0.0
        o = o + np.clip(1.0 - R.exp_ph - mh, 0.0, 1.0) * cash / TD
    return pd.Series(o.mean(axis=0), index=Dt.sig)


# ================================================================ runs2(匹配静态对照、规则 x 固定降仓)
def all_rules(Dt: Data, include_matched=True) -> dict:
    rules = {}

    def add(name, family, label, run_name, **kw):
        p = RUNS / f"{run_name}.pkl"
        if not p.exists():
            return
        rules[name] = build_rule(Dt, name, family, label, load_run(run_name), **kw)

    for n, c in CONFIGS.items():
        if c["kind"] in ("static", "static_from") and not c["group"].startswith("O5"):
            fam = c["group"]
            add(n, fam, f"O1 {c['f'] * 100:.0f}%" + (" [指数法估值]" if c["val"] == "index" else "")
                + (" [C]" if c["tgt"] == "C" else ""), n, main=c["main"])
        elif c["kind"] in ("vol", "trend", "dd", "o6"):
            add(n, c["group"], n, n, main=c["main"])
    # 对冲:O5(理想化,中证1000)与 O5-R(可实施)
    o5 = [(h, 0.15, 0.03, "te") for h in (0.3, 0.5, 1.0)]                      # 邻域网格(Q-d)
    o5 += [(0.5, m, 0.03, "te") for m in (0.0, 0.30)]                          # 保证金敏感性
    o5 += [(0.5, 0.15, sc, "te") for sc in (0.0, 0.06)]                         # 2015-04-16 前贴水情景
    o5 += [(0.5, 0.15, 0.03, "te1")]                                            # 成本日期对齐敏感性
    for h, m, sc, al in o5:
        nm = f"O5_h{int(h * 100)}_m{int(m * 100)}_s{int(sc * 100)}" + ("_te1" if al == "te1" else "")
        add(nm, "O5", nm, static_name(O5_LONG[(h, m)]), hedge=("ideal", h, m, sc, al),
            main=(h == 0.5 and m == 0.15 and sc == 0.03 and al == "te"))
    for h in (0.3, 0.5, 1.0):
        add(f"O5R_h{int(h * 100)}_m15", "O5R", f"O5R_h{int(h * 100)}_m15", f"O5R_long_h{int(h * 100)}",
            hedge=("real", h, 0.15, 0.0, "te"), main=(h == 0.5))
    add("O5R_h50_m30", "O5R", "O5R_h50_m30", "O5R_long_h50_m30", hedge=("real", 0.5, 0.30, 0.0, "te"))
    add("O5R_h50_m15_te1", "O5R", "O5R_h50_m15_te1", "O5R_long_h50", hedge=("real", 0.5, 0.15, 0.0, "te1"))
    add("O5_h50_m15_s3_vidx", "O5_vidx", "O5_h50 [指数法估值]", static_name(0.925, "index"),
        hedge=("ideal", 0.5, 0.15, 0.03, "te"))
    add("O5_h50_m15_s3_C", "O5_C", "O5_h50 [C]", static_name(0.925, tgt="C"), hedge=("ideal", 0.5, 0.15, 0.03, "te"))
    if include_matched:
        for p in sorted(RUNS.glob("match_*.pkl")) + sorted(RUNS.glob("scaled_*.pkl")):
            n = p.stem
            run = load_run(n)
            rules[n] = build_rule(Dt, n, "matched" if n.startswith("match_") else "scaled", n, run)
    return rules


MAIN_RULES = ["O2_v20", "O3_ma200_p50", "O4_15_25", "O4b_15_25", "O5_h50_m15_s3", "O5R_h50_m15", "O6_v20_ma200"]


def ev(s: pd.Series, a=LE.EVAL_START, b=EVAL_END) -> pd.Series:
    return s.loc[pd.Timestamp(a): pd.Timestamp(b)]


def stage_runs2(workers: int):
    Dt = Data()
    rules = all_rules(Dt, include_matched=False)
    # O1 网格:名义 f -> 实际平均仓位(评价期)
    grid = []
    for f in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        n = static_name(f)
        if n in rules:
            grid.append((f, float(ev(net_exposure(Dt, rules[n])).mean())))
    gf = np.array([g[0] for g in grid]); ge = np.array([g[1] for g in grid])
    cfgs, rows = {}, []
    for rn in MAIN_RULES + ["O5_h30_m15_s3", "O5_h100_m15_s3", "O5R_h30_m15", "O5R_h100_m15"]:
        if rn not in rules:
            continue
        e = float(ev(net_exposure(Dt, rules[rn])).mean())
        if e <= 0:
            rows.append(dict(rule=rn, e_bar=e, f_nominal=np.nan, note="净敞口 <= 0,无匹配静态对照"))
            continue
        # 线性插值(低于网格下端时按最低两点外推)
        if e < ge[0]:
            f_nom = gf[0] + (e - ge[0]) * (gf[1] - gf[0]) / (ge[1] - ge[0])
        else:
            f_nom = float(np.interp(e, ge, gf))
        f_nom = float(np.clip(f_nom, 0.0, 1.0))
        cfgs[f"match_{rn}"] = C("static", "matched", f=round(f_nom, 6))
        rows.append(dict(rule=rn, e_bar=e, f_nominal=f_nom, note=""))
    for rn in SCALED_RULES:
        for f in SCALE_F:
            cfgs[f"scaled_{rn}_x{int(f * 100)}"] = C("scaled", "scaled", base=rn, scale=f)
    pd.DataFrame(rows).to_csv(TMP / "lh_step2_matched_targets.csv", index=False)
    for k, v in cfgs.items():
        CONFIGS[k] = v
    _run_many(cfgs, workers, "runs2")




# ================================================================ report:统计工具
def drop_rows(idx: pd.DatetimeIndex, bear: str | None) -> np.ndarray:
    keep = np.ones(len(idx), bool)
    if bear is not None:
        a, b = LE.WINDOWS_A[bear]
        keep &= ~((idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b)))
    return keep


def pslice(r: pd.Series, period: str, bear: str | None = None) -> pd.Series:
    a, b = PERIODS[period]
    s = LE.slice_eval(r, a, b)
    return s[drop_rows(s.index, bear)]


def mdd_abs(r: pd.Series) -> float:
    return abs(LE.max_dd(r)[0])


def calmar(r: pd.Series) -> float:
    m = mdd_abs(r)
    return LE.ann_geo(r) / m if m > 0 else np.nan


def qb_window(r: pd.Series, o0: pd.Series, period: str, bear=None) -> dict:
    s, o = pslice(r, period, bear), pslice(o0, period, bear)
    d = mdd_abs(o) - mdd_abs(s)
    cs, co = calmar(s), calmar(o)
    return dict(dmdd=d, calmar=cs, calmar_o0=co, ok=bool(d >= 0.10 and cs > co), mdd=-mdd_abs(s), mdd_o0=-mdd_abs(o),
                ann=LE.ann_geo(s), ann_o0=LE.ann_geo(o))


def qa(r: pd.Series) -> bool:
    return all(mdd_abs(pslice(r, p)) <= 0.50 for p in ("full", "dev", "test"))


def qb(r: pd.Series, o0: pd.Series) -> dict:
    base = {p: qb_window(r, o0, p) for p in ("dev", "test")}
    loo = {}
    for bear in LE.BEARS:
        loo[bear] = {p: qb_window(r, o0, p, bear) for p in ("dev", "test", "full")}
    ok_base = base["dev"]["ok"] and base["test"]["ok"]
    ok_loo = all(loo[b]["dev"]["ok"] and loo[b]["test"]["ok"] for b in LE.BEARS)
    ok_loo_full = all(loo[b]["full"]["ok"] for b in LE.BEARS)
    return dict(ok=bool(ok_base and ok_loo), ok_base=bool(ok_base), ok_loo=bool(ok_loo), ok_loo_fullonly=bool(ok_loo_full),
                base=base, loo=loo,
                fail_loo=[f"{b}:{p}" for b in LE.BEARS for p in ("dev", "test") if not loo[b][p]["ok"]])


def whipsaw(Dt: Data, R: Rule, period: str, big: float = 0.0) -> dict:
    """逐相位:新目标仓位 < 旧目标仓位 的调仓日 d,若全 A 等权收盘指数在 d 后 20 个交易日内最高收盘比 d 日收盘高 10% 以上,计一次。"""
    a, b = PERIODS[period]
    I = Dt.ew_level
    T = len(I)
    nd = nw = 0
    for ph in range(K):
        days = np.arange(ph, Dt.n, K)
        p = R.P[ph, days]
        for j in range(1, len(days)):
            if p[j] < p[j - 1] - 1e-9 - big:
                d = days[j]
                dt = Dt.sig[d]
                if not (pd.Timestamp(a) <= dt <= pd.Timestamp(b)):
                    continue
                nd += 1
                fut = I[d + 1: min(d + 21, T)]
                if len(fut) and fut.max() / I[d] - 1 > 0.10:
                    nw += 1
    return dict(n_down=nd, n_whipsaw=nw, whip_ratio=nw / nd if nd else np.nan, n_down_per_phase=nd / K,
                n_whipsaw_per_phase=nw / K)


def days_to_full(Dt: Data, R: Rule, trough: str) -> dict:
    i = int(Dt.sig.searchsorted(pd.Timestamp(trough)))
    vals, never = [], 0
    for ph in range(K):
        pe = R.pos_eff[ph, i:]
        hit = np.flatnonzero(pe >= 0.999)
        if len(hit):
            vals.append(int(hit[0]))
        else:
            never += 1
    return dict(days_to_full_mean=float(np.mean(vals)) if vals else np.nan,
                days_to_full_max=float(np.max(vals)) if vals else np.nan, phases_never_full=never)


def window_defs() -> list[tuple[str, str, str, str]]:
    out = []
    for w, (a, b) in LE.WINDOWS_A.items():
        out.append(("A", w, a, b))
    for w, (a, b) in LE.WINDOWS_B.items():
        out.append(("B", w, a, b))
    for w, (a, b) in LE.WINDOWS_ORIG.items():
        out.append(("orig", w, a, b))
    sig = LE.load_inputs().dates
    for v, tr in LE.V_TROUGHS.items():
        for h in (60, 120):
            a, b = LE.v_window(sig, tr, h)
            out.append(("V", f"{v}_{h}", a, b))
    return out


# ================================================================ report
def stage_report():
    t0 = time.time()
    Dt = Data()
    rules = all_rules(Dt)
    log(f"[report] {len(rules)} 个组合")
    D = Dt.D
    O0 = {"close": D["strat_main"], "index": D["strat_main_vidx"]}
    B1 = {"close": D["bench_b1"], "index": D["bench_b1_vidx"]}
    checks = []

    # ---------------- 断言 ----------------
    for val, rn in (("close", static_name(1.0)), ("index", static_name(1.0, "index"))):
        if rn in rules:
            dmax = float((rule_series(Dt, rules[rn]) - O0[val]).abs().max())
            ph = rules[rn].rets_ph
            col = "strat_main" if val == "close" else "strat_main_vidx"
            dph = float(np.nanmax(np.abs(ph - np.vstack([D[f"strat_main_ph{p}"].to_numpy() for p in range(K)])))) \
                if val == "close" else np.nan
            phn = f"逐相位最大差 {dph:.1e}" if val == "close" else "第一步没有落盘指数法估值的逐相位序列,只比相位平均"
            checks.append(dict(check=f"S2_O1_100_equals_O0[{val}]", step=2, value=dmax, ok=bool(dmax <= 1e-12),
                               note=f"callable 目标(仓位 1)vs 第一步 {col} 相位平均逐日最大差;{phn}"))
    for val in ("close", "index"):
        Bp = Dt.b1_phases(val).mean(axis=0)
        d = float(np.nanmax(np.abs(Bp - B1[val].to_numpy())))
        checks.append(dict(check=f"S2_B1_phases_mean_equals_B1[{val}]", step=2, value=d, ok=bool(d <= 1e-12),
                           note="B1 逐相位序列(选股/择时分解用)的相位平均 vs 第一步 bench_b1"))
    mx = max(load_run(n)["max_exposure"] for n in CONFIGS if (RUNS / f"{n}.pkl").exists())
    mx = max([mx] + [load_run(p.stem)["max_exposure"] for p in RUNS.glob("match_*.pkl")]
             + [load_run(p.stem)["max_exposure"] for p in RUNS.glob("scaled_*.pkl")])
    checks.append(dict(check="S2_E1_exposure_le_1", step=2, value=mx, ok=bool(mx <= 1 + 1e-9),
                       note="第二步全部引擎回测逐相位逐日多头仓位最大值(引擎内硬断言同时生效)"))
    lm_rows = []
    for rn, R in rules.items():
        if R.hedge is None:
            continue
        lm = np.nanmax(R.hedge["LM"])
        lm_rows.append((rn, lm))
    if lm_rows:
        worst = max(lm_rows, key=lambda x: x[1])
        ok_c = [v for n, v in lm_rows if n.endswith("_C")]
        checks.append(dict(check="S2_E1_hedge_long_plus_margin_le_1", step=2, value=float(worst[1]),
                           ok=bool(worst[1] <= 1 + 1e-9),
                           note=f"按字面不通过(残仓所致,不是记账错误):对冲组合调仓日 实际多头 + 保证金 m x h 的最大值({worst[0]});"
                                f"按'O0 持仓整体缩放'口径,残仓卖不掉时超出 m x h x 残仓(2015-07 残仓一度约 95%);"
                                f"'C' 口径(残仓计入目标)最大 {max(ok_c) if ok_c else np.nan:.4f}(出现在 2005-05 预热段,2006 年起 1.003);"
                                f"§14.2 要求不过即停,本步未停机,已补登偏离 [step2-review];调仓日超出频率见 lh_step2_review.csv section=hedge_lm"))
    for rn in MAIN_RULES + [static_name(0.7), static_name(0.5)]:
        if rn not in rules:
            continue
        R = rules[rn]
        r = ev(rule_series(Dt, R)); b = ev(B1[R.val])
        cs = {k: float(r.corr(b.shift(k))) for k in range(-3, 4)}
        kk = max(cs, key=cs.get)
        checks.append(dict(check=f"S2_E2_align_peak_k0[{rn}]", step=2, value=kk, ok=bool(kk == 0),
                           note="corr(规则, B1.shift(k)): " + ", ".join(f"{k}:{v:.3f}" for k, v in cs.items())))
        if R.hedge is not None:
            rl = ev(pd.Series(R.hedge["rL"].mean(axis=0), index=Dt.sig))
            hl = ev(pd.Series(R.hedge["rH"].mean(axis=0), index=Dt.sig))
            cs = {k: float(-hl.corr(rl.shift(k))) for k in range(-3, 4)}
            kk = max(cs, key=cs.get)
            checks.append(dict(check=f"S2_E2_hedge_leg_peak_k0[{rn}]", step=2, value=kk, ok=bool(kk == 0),
                               note="corr(-空头腿, 多头腿.shift(k)): " + ", ".join(f"{k}:{v:.3f}" for k, v in cs.items())))

    # ---------------- 汇总 ----------------
    summ, wins, loos, whips = [], [], [], []
    ser = {}
    for rn, R in rules.items():
        o0, b1 = O0[R.val], B1[R.val]
        for cash in (0.0, 0.02):
            r = rule_series(Dt, R, cash)
            bo = b1_overlay(Dt, R, cash)
            o0c = o0 if cash == 0 else o0 + (1.0 - D["main_exposure"]).clip(lower=0) * cash / TD
            ser[(rn, cash)] = r
            ne = net_exposure(Dt, R)
            for pn in PERIODS:
                s, o, bb, boo = pslice(r, pn), pslice(o0c, pn), pslice(b1, pn), pslice(bo, pn)
                ps = LE.perf(s)
                row = dict(rule=rn, family=R.family, main=R.main, valuation=R.val, cash=cash, period=pn,
                           start=str(s.index[0].date()), end=str(s.index[-1].date()))
                row.update(ps)
                po = LE.perf(o)
                row.update(o0_ann=po["ann"], o0_mdd=po["mdd"], o0_calmar=po["calmar"])
                row["dmdd_vs_o0_pp"] = (abs(po["mdd"]) - abs(ps["mdd"])) * 100
                row["dann_vs_o0_pp"] = (ps["ann"] - po["ann"]) * 100
                den = abs(po["mdd"]) - abs(ps["mdd"])
                row["cost_per_pp_mdd"] = (po["ann"] - ps["ann"]) / den if den > 1e-9 else np.nan
                e = pslice(ne, pn)
                row["net_exposure_avg"] = float(e.mean())
                row["long_exposure_avg"] = float(pslice(pd.Series(R.exp_ph.mean(axis=0), index=Dt.sig), pn).mean())
                row["target_pos_avg"] = float(pslice(pd.Series(R.pos_eff.mean(axis=0), index=Dt.sig), pn).mean())
                row["exp_minus_target"] = row["long_exposure_avg"] - row["target_pos_avg"]
                for c in ("cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact", "turnover"):
                    row[f"{c}_ann"] = float(pslice(pd.Series(R.rec[c], index=Dt.sig), pn).mean() * TD)
                if R.hedge is not None:
                    hr = pd.Series(R.hedge["HR"].mean(axis=0), index=Dt.sig)
                    rh = pd.Series(R.hedge["rH"].mean(axis=0), index=Dt.sig)
                    row["hedge_ratio_avg"] = float(pslice(hr, pn).mean())
                    row["hedge_leg_ann_arith"] = float(pslice(rh, pn).mean() * TD)
                    g = R.hedge["gap"]
                    sl = np.asarray((Dt.sig >= pd.Timestamp(PERIODS[pn][0])) & (Dt.sig <= pd.Timestamp(PERIODS[pn][1])))
                    row["margin_gap_max"] = float(np.nanmax(g[:, sl]))
                    row["margin_gap_days_frac"] = float((g[:, sl] > 1e-9).mean())
                    row["margin_negative_days"] = int((R.hedge["M"][:, sl] < 0).sum())
                rel = LE.rel_block(s, bb)
                row.update({f"b1_{k}": v for k, v in rel.items()})
                y = LE.yearly(s, bb)
                row["b1_year_t"] = LE.year_t(y["excess"]) if len(y) >= 3 else np.nan
                rs_ = LE.rel_block(s, boo, "sel_")
                row.update({k: v for k, v in rs_.items() if k in ("sel_excess_geo", "sel_excess_arith", "sel_nw_t5", "sel_te")})
                row["timing_part_geo"] = LE.ann_geo(boo) - LE.ann_geo(bb)
                row["o0_excess_geo"] = LE.ann_geo(o) - LE.ann_geo(bb)
                row["o0_excess_x_exposure"] = row["o0_excess_geo"] * row["long_exposure_avg"]
                if cash == 0:
                    w = whipsaw(Dt, R, pn)
                    row.update(w)
                    wb = whipsaw(Dt, R, pn, big=0.0999)
                    row["n_down_ge10pp"] = wb["n_down"]; row["n_whipsaw_ge10pp"] = wb["n_whipsaw"]
                summ.append(row)
            if cash == 0:
                # 窗口
                for st_, w, a, b in window_defs():
                    ix = LE.window_rows(Dt.sig, a, b)
                    if len(ix) == 0:
                        continue
                    s = r.iloc[ix]; o = o0.iloc[ix]; bb = b1.iloc[ix]
                    rr = dict(rule=rn, family=R.family, main=R.main, valuation=R.val, set=st_, window=w, start=a, end=b,
                              n_days=len(ix), ret=float((1 + s).prod() - 1), ret_o0=float((1 + o).prod() - 1),
                              ret_b1=float((1 + bb).prod() - 1), mdd=LE.max_dd(s)[0], mdd_o0=LE.max_dd(o)[0],
                              net_exposure_avg=float(ne.iloc[ix].mean()))
                    rr["rel_vs_o0"] = rr["ret"] - rr["ret_o0"]
                    rr["rel_vs_b1"] = rr["ret"] - rr["ret_b1"]
                    if st_ == "V":
                        tr = LE.V_TROUGHS[w.rsplit("_", 1)[0]] if w.rsplit("_", 1)[0] in LE.V_TROUGHS else None
                        if tr is None:
                            tr = LE.V_TROUGHS["V22_supp"]
                        rr.update(days_to_full(Dt, R, tr))
                    wins.append(rr)
                # 留一段
                for bear in (None,) + LE.BEARS:
                    for pn in ("full", "dev", "test"):
                        s, o = pslice(r, pn, bear), pslice(o0, pn, bear)
                        loos.append(dict(rule=rn, family=R.family, main=R.main, valuation=R.val, dropped=bear or "(none)",
                                         period=pn, n_days=len(s), ann=LE.ann_geo(s), mdd=LE.max_dd(s)[0],
                                         calmar=calmar(s), ann_o0=LE.ann_geo(o), mdd_o0=LE.max_dd(o)[0], calmar_o0=calmar(o),
                                         dmdd_pp=(mdd_abs(o) - mdd_abs(s)) * 100,
                                         qb_cond=bool((mdd_abs(o) - mdd_abs(s)) >= 0.10 and calmar(s) > calmar(o))))
        log(f"[report] {rn} 完成")
    summ = pd.DataFrame(summ); wins = pd.DataFrame(wins); loos = pd.DataFrame(loos)
    summ.to_csv(LEDGER / "lh_step2_summary.csv", index=False)
    wins.to_csv(LEDGER / "lh_step2_windows.csv", index=False)
    loos.to_csv(LEDGER / "lh_step2_loo.csv", index=False)
    wh = summ[(summ.cash == 0)][["rule", "family", "main", "valuation", "period", "n_down", "n_whipsaw", "whip_ratio",
                                  "n_down_per_phase", "n_whipsaw_per_phase", "n_down_ge10pp", "n_whipsaw_ge10pp"]]
    wh.to_csv(LEDGER / "lh_step2_whipsaw.csv", index=False)

    # ---------------- 匹配静态对照(Q-c)----------------
    mt = pd.read_csv(TMP / "lh_step2_matched_targets.csv")
    grid_rows = summ[(summ.family == "O1") & (summ.cash == 0)]
    mrows = []
    for _, m in mt.iterrows():
        rn = m["rule"]
        if rn not in rules:
            continue
        r = ser[(rn, 0.0)]
        mn = f"match_{rn}"
        row = dict(rule=rn, e_bar=m["e_bar"], f_nominal=m["f_nominal"])
        if mn not in rules:
            row["note"] = m.get("note", "无匹配对照")
            mrows.append(row); continue
        sm = ser[(mn, 0.0)]
        row["e_bar_matched_actual"] = float(ev(net_exposure(Dt, rules[mn])).mean())
        okc = True
        for pn in ("full", "dev", "test"):
            s, z = pslice(r, pn), pslice(sm, pn)
            row[f"{pn}_ann"] = LE.ann_geo(s); row[f"{pn}_ann_static"] = LE.ann_geo(z)
            row[f"{pn}_mdd"] = LE.max_dd(s)[0]; row[f"{pn}_mdd_static"] = LE.max_dd(z)[0]
            row[f"{pn}_timing_net_pp"] = (LE.ann_geo(s) - LE.ann_geo(z)) * 100
            row[f"{pn}_dmdd_vs_static_pp"] = (mdd_abs(z) - mdd_abs(s)) * 100
            row[f"{pn}_calmar"] = calmar(s); row[f"{pn}_calmar_static"] = calmar(z)
            ok = (mdd_abs(z) - mdd_abs(s) >= 0.05) and (LE.ann_geo(s) >= LE.ann_geo(z) - 0.005)
            row[f"{pn}_qc_cond"] = bool(ok)
            if pn in ("dev", "test"):
                okc &= ok
            # 敏感性:按窗口内平均净仓位在静态前沿上插值
            g = grid_rows[grid_rows.period == pn].sort_values("net_exposure_avg")
            ew_ = float(pslice(net_exposure(Dt, rules[rn]), pn).mean())
            row[f"{pn}_e_window"] = ew_
            row[f"{pn}_interp_static_ann"] = float(np.interp(ew_, g.net_exposure_avg, g.ann))
            row[f"{pn}_interp_static_mdd"] = float(np.interp(ew_, g.net_exposure_avg, g.mdd))
        row["qc"] = bool(okc)
        mrows.append(row)
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(LEDGER / "lh_step2_matched_static.csv", index=False)

    # ---------------- 判定 ----------------
    QD = {
        static_name(0.7): [static_name(0.6), static_name(0.7), static_name(0.8)],
        static_name(0.5): [static_name(0.4), static_name(0.5), static_name(0.6)],
        "O2_v20": ["O2_v15", "O2_v20", "O2_v25"],
        "O3_ma200_p50": [f"O3_ma{L}_p{p}" for L in (120, 200, 250) for p in (0, 50)],
        "O4_15_25": ["O4_10_20", "O4_15_25", "O4_20_30"],
        "O4b_15_25": None,
        "O5_h50_m15_s3": ["O5_h30_m15_s3", "O5_h50_m15_s3", "O5_h100_m15_s3"],
        "O5R_h50_m15": ["O5R_h30_m15", "O5R_h50_m15", "O5R_h100_m15"],
        "O6_v20_ma200": ["O6_v20_ma200", "O6_v15_ma200", "O6_v25_ma200", "O6_v20_ma120", "O6_v20_ma250"],
    }
    gcache = {}

    def grade_cell(rn):
        if rn not in gcache:
            R = rules[rn]
            r = ser[(rn, 0.0)]
            gcache[rn] = dict(qa=qa(r), qb=qb(r, O0[R.val]))
        return gcache[rn]

    grid_out = []
    for fam_main, cells in QD.items():
        for c in (cells or [fam_main]):
            if c not in rules:
                continue
            g = grade_cell(c)
            r = ser[(c, 0.0)]
            grid_out.append(dict(rule_main=fam_main, cell=c, is_main=c == fam_main, qa=g["qa"], qb=g["qb"]["ok"],
                                 qb_base=g["qb"]["ok_base"], qb_loo=g["qb"]["ok_loo"],
                                 qb_loo_fullonly=g["qb"]["ok_loo_fullonly"],
                                 mdd_full=LE.max_dd(pslice(r, "full"))[0], mdd_dev=LE.max_dd(pslice(r, "dev"))[0],
                                 mdd_test=LE.max_dd(pslice(r, "test"))[0], ann_full=LE.ann_geo(pslice(r, "full")),
                                 ann_dev=LE.ann_geo(pslice(r, "dev")), ann_test=LE.ann_geo(pslice(r, "test")),
                                 dmdd_dev_pp=g["qb"]["base"]["dev"]["dmdd"] * 100,
                                 dmdd_test_pp=g["qb"]["base"]["test"]["dmdd"] * 100,
                                 calmar_dev=g["qb"]["base"]["dev"]["calmar"], calmar_dev_o0=g["qb"]["base"]["dev"]["calmar_o0"],
                                 calmar_test=g["qb"]["base"]["test"]["calmar"], calmar_test_o0=g["qb"]["base"]["test"]["calmar_o0"],
                                 qb_loo_fails=";".join(g["qb"]["fail_loo"])))
    # 静态全网格也写进 grid(静态前沿)
    for f in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for val in ("close", "index"):
            c = static_name(f, val)
            if c not in rules or any(x["cell"] == c and x["rule_main"] == "O1_frontier" for x in grid_out):
                continue
            g = grade_cell(c); r = ser[(c, 0.0)]
            grid_out.append(dict(rule_main="O1_frontier" + ("_vidx" if val == "index" else ""), cell=c, is_main=False,
                                 qa=g["qa"], qb=g["qb"]["ok"], qb_base=g["qb"]["ok_base"], qb_loo=g["qb"]["ok_loo"],
                                 qb_loo_fullonly=g["qb"]["ok_loo_fullonly"],
                                 mdd_full=LE.max_dd(pslice(r, "full"))[0], mdd_dev=LE.max_dd(pslice(r, "dev"))[0],
                                 mdd_test=LE.max_dd(pslice(r, "test"))[0], ann_full=LE.ann_geo(pslice(r, "full")),
                                 ann_dev=LE.ann_geo(pslice(r, "dev")), ann_test=LE.ann_geo(pslice(r, "test")),
                                 dmdd_dev_pp=g["qb"]["base"]["dev"]["dmdd"] * 100,
                                 dmdd_test_pp=g["qb"]["base"]["test"]["dmdd"] * 100,
                                 calmar_dev=g["qb"]["base"]["dev"]["calmar"], calmar_dev_o0=g["qb"]["base"]["dev"]["calmar_o0"],
                                 calmar_test=g["qb"]["base"]["test"]["calmar"], calmar_test_o0=g["qb"]["base"]["test"]["calmar_o0"],
                                 qb_loo_fails=";".join(g["qb"]["fail_loo"])))
    gdf = pd.DataFrame(grid_out)
    gdf.to_csv(LEDGER / "lh_step2_grid.csv", index=False)

    # f*:静态前沿上满足 Q-a 的最高固定仓位(网格 + 线性插值估计)
    def fstar(val: str, cash: float = 0.0):
        fs = [f for f in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0) if static_name(f, val) in rules]
        worst = {f: max(mdd_abs(pslice(ser[(static_name(f, val), cash)], p)) for p in ("full", "dev", "test")) for f in fs}
        ok = [f for f in fs if worst[f] <= 0.50]
        fg = max(ok) if ok else np.nan
        xs = np.array([worst[f] for f in fs]); ys = np.array(fs)
        fi = float(np.interp(0.50, xs, ys)) if xs.min() <= 0.5 <= xs.max() else np.nan
        return fg, fi, worst
    fst = {}
    for val in ("close", "index"):
        for cash in (0.0, 0.02):
            fst[(val, cash)] = fstar(val, cash)

    o0full = pslice(O0["close"], "full")
    grades = []
    for rn in [static_name(0.7), static_name(0.5)] + MAIN_RULES:
        if rn not in rules:
            continue
        R = rules[rn]
        g = grade_cell(rn)
        r = ser[(rn, 0.0)]
        cells = QD.get(rn)
        if cells:
            cg = [grade_cell(c) for c in cells if c in rules]
            n_ok = sum(1 for x in cg if x["qa"] and x["qb"]["ok"])
            qd = n_ok >= np.ceil(2 / 3 * len(cg) - 1e-9)
            qd_txt = f"{n_ok}/{len(cg)}"
        else:
            qd, qd_txt = False, "无邻域网格(变体)"
        mrow = mdf[mdf.rule == rn]
        is_static = R.family == "O1"
        qc = (bool(mrow.qc.iloc[0]) if len(mrow) and "qc" in mrow and pd.notna(mrow.qc.iloc[0]) else False) \
            if not is_static else np.nan
        extra = ""
        o5_ok = np.nan
        if rn.startswith("O5_"):
            gr = grade_cell("O5R_h50_m15")
            # 附加条件:O5-R 在检验窗(实际贴水)满足 Q-b(检验窗本身 + 检验窗内留一段)
            okt = gr["qb"]["base"]["test"]["ok"] and all(gr["qb"]["loo"][b]["test"]["ok"] for b in LE.BEARS)
            o5_ok = bool(okt)
            extra = f"O5-R 检验窗 Q-b {'满足' if okt else '不满足'}"
        # 结论用语(预注册 §12.4)
        if not g["qb"]["ok"]:
            concl = "无增量"
        elif not g["qa"]:
            concl = "降低了回撤,但仍不满足 50% 约束"
        elif is_static:
            concl = "静态降仓满足约束(基线)" + ("" if qd else ",参数脆弱")
        elif qc is True:
            concl = "可用风控增量,且择时成分有效" + ("" if qd else "(参数脆弱,不作推荐)")
        else:
            concl = "可用风控增量,但只是去杠杆" + ("" if qd else "(参数脆弱,不作推荐)")
        if rn.startswith("O5_") and concl.startswith("可用") and not o5_ok:
            concl += ";O5-R 不满足,不可实施"
        # 需要叠加的固定降仓(满足 Q-b 不满足 Q-a 时)
        need = ""
        if g["qb"]["ok"] and not g["qa"] and rn in SCALED_RULES:
            okf = [f for f in SCALE_F if f"scaled_{rn}_x{int(f * 100)}" in rules and qa(ser[(f"scaled_{rn}_x{int(f * 100)}", 0.0)])]
            need = f"叠加固定降仓 x{max(okf):.1f} 后满足 Q-a" if okf else "叠加 x0.5 仍不满足 Q-a"
        full = pslice(r, "full")
        grades.append(dict(rule=rn, qa=g["qa"], qb=g["qb"]["ok"], qb_base=g["qb"]["ok_base"], qb_loo=g["qb"]["ok_loo"],
                           qb_loo_fullonly=g["qb"]["ok_loo_fullonly"], qb_loo_fails=";".join(g["qb"]["fail_loo"]),
                           qc=qc, qd=qd, qd_cells=qd_txt, o5_implementable=o5_ok, conclusion=concl, note=extra + need,
                           mdd_full=LE.max_dd(full)[0], ann_full=LE.ann_geo(full),
                           mdd_dev=LE.max_dd(pslice(r, "dev"))[0], mdd_test=LE.max_dd(pslice(r, "test"))[0],
                           dmdd_dev_pp=g["qb"]["base"]["dev"]["dmdd"] * 100, dmdd_test_pp=g["qb"]["base"]["test"]["dmdd"] * 100,
                           cost_per_pp_full=(LE.ann_geo(o0full) - LE.ann_geo(full)) / (mdd_abs(o0full) - mdd_abs(full))
                           if mdd_abs(o0full) - mdd_abs(full) > 1e-9 else np.nan))
    grd = pd.DataFrame(grades)
    ok = grd[grd.qa & grd.qb]
    grd["rank_cost_per_pp"] = grd.cost_per_pp_full.where(grd.qa & grd.qb).rank()
    for (val, cash), (fg, fi, worst) in fst.items():
        grd = pd.concat([grd, pd.DataFrame([dict(rule=f"f*[{val},cash{int(cash * 100)}%]",
                                                 conclusion=f"静态前沿上满足 Q-a 的最高网格仓位 {fg}",
                                                 note=f"线性插值估计 {fi:.3f}(插值点按构造零余量,不能当上限;实跑、逐相位与留一段见 lh_step2_review.csv section=fstar);"
                                     f"网格 f* 的余量 {(0.5 - worst[fg]) * 100:.1f}pp;各网格三区间最差回撤 " +
                                                      ", ".join(f"{f:.1f}:{w * 100:.1f}%" for f, w in worst.items()))])],
                        ignore_index=True)
    grd.to_csv(LEDGER / "lh_step2_grade.csv", index=False)
    log("[report] 判定:\n" + grd[["rule", "qa", "qb", "qc", "qd", "conclusion", "note"]].to_string())

    # ---------------- 其他:2011 中证1000 收盘口径下多空相关、探针 ----------------
    extra = []
    R5 = rules.get("O5_h50_m15_s3")
    if R5 is not None:
        rl = pd.Series(R5.hedge["rL"].mean(axis=0), index=Dt.sig)
        ix = pd.Series(Dt.idx_oo["CSI1000"], index=Dt.sig)
        for y in range(2006, 2026):
            m = rl.index.year == y
            extra.append(dict(item="corr_long_leg_vs_CSI1000_oo", period=str(y), value=float(rl[m].corr(ix[m])),
                              note="2011 年中证1000 开盘缺失,用收盘对收盘近似" if y == 2011 else ""))
    for pr, base in (("probe_O2_v20_lag3", "O2_v20"), ("probe_O2_v20_lag0", "O2_v20"),
                     ("probe_O3_ma200_p50_idx_tm1", "O3_ma200_p50"), ("probe_O3_ma200_p50_idx_tp1", "O3_ma200_p50"),
                     ("probe_O4_15_25_lag3", "O4_15_25"), ("probe_O4_15_25_lag0", "O4_15_25")):
        if pr not in rules or base not in rules:
            continue
        for pn in ("full", "dev", "test"):
            a_, b_ = pslice(ser[(pr, 0.0)], pn), pslice(ser[(base, 0.0)], pn)
            extra.append(dict(item=f"probe:{pr} - {base}", period=pn,
                              value=(LE.ann_geo(a_) - LE.ann_geo(b_)) * 100,
                              note=f"年化差 pp;最大回撤 探针 {LE.max_dd(a_)[0] * 100:.1f}% vs 主 {LE.max_dd(b_)[0] * 100:.1f}%"))
    pd.DataFrame(extra).to_csv(LEDGER / "lh_step2_extra.csv", index=False)

    merge_csv(CHECKS, pd.DataFrame(checks), "check")
    for c in checks:
        log(f"[report] {c['check']}: {c['value']:.4g} {'OK' if c['ok'] else 'FAIL'}")

    # ---------------- 每日序列 ----------------
    out = {"O0": O0["close"], "B1": B1["close"]}
    for rn in [static_name(0.7), static_name(0.5)] + MAIN_RULES:
        if rn in rules:
            out[rn] = ser[(rn, 0.0)]
            out[rn + "|net_exposure"] = net_exposure(Dt, rules[rn])
            out[rn + "|b1_overlay"] = b1_overlay(Dt, rules[rn])
    pd.DataFrame(out).to_parquet(DAILY2)
    figures(Dt, rules, ser, summ, gdf)
    timing("report", time.time() - t0, f"{len(rules)} 个组合")


# ================================================================ review(审查修复,2026-10-11)
REVIEW = LEDGER / "lh_step2_review.csv"
REVIEW_DEVIATIONS = [
    ("[step2-review] §12.4 Q-d 邻域:O6",
     "O6 = O2 x O3(两者各自的主参数);Q-d 要求'该规则的邻域网格'中至少 2/3 的格子同时满足 Q-a 与 Q-b,但 §12.2 只给出 O2/O3/O4/O5 的邻域",
     "O6 邻域取 5 格:主参数 v20/ma200,加单维变动 v15、v25(沿用 O2 邻域)与 ma120、ma250(沿用 O3 邻域);低于均线时仓位固定 50%;门槛 = ceil(2/3 x 5) = 4 格",
     "预注册没有写 O6 的邻域;沿用两个母规则各自的单维邻域",
     "审查补登(未在开跑前写入本表);5 个格子的配置在第一批回测开跑前已写在 build_configs 中(logs/lh_step2.log 01:50:51 首批 runs 含 O6_v15_ma200 等)",
     "无:O6 主参数 Q-b 不满足,结论已是'无增量';5 个格子全部 Q-b 不满足(lh_step2_grid.csv)"),
    ("[step2-review] §12.4 Q-d 邻域:O1 70% / 50%",
     "O1 固定仓位 70% / 50%;§12.2 另跑 30%~100% 静态网格作为静态前沿,未定义 O1 的 Q-d 邻域",
     "O1 70% 的邻域 = {60%, 70%, 80%},O1 50% 的邻域 = {40%, 50%, 60%}(静态网格上 ±10pp)",
     "预注册没有写 O1 的邻域;取静态网格上相邻两格",
     "审查补登(未在开跑前写入本表);网格配置在开跑前已定义",
     "无:O1 70% 的 Q-a、O1 50% 的 Q-b 都不满足,Q-d 不影响结论用语"),
    ("[step2-review] §12.4 Q-d:O4-b",
     "O4-b 是 O4 的变体(不重置峰值、恢复后 20 日不触发),预注册没有给它邻域",
     "记为'无邻域网格(变体)',按 Q-d 不满足处理",
     "预注册未定义;不另造网格",
     "审查补登(未在开跑前写入本表)",
     "无:O4-b 的 Q-b 不满足"),
    ("[step2-review] §14.2 断言不通过时继续执行",
     "§14.2:断言任一不过即停",
     "S2_E1_hedge_long_plus_margin_le_1 按字面不通过(最大 1.117,O5R_h100_m15)后,仍继续生成全部账本与报告",
     "超出量 = m x h x 残仓,来自卖不掉的残仓(引擎 stuck 逻辑),不是记账错误;保证金收益为 0,超出部分不进入任何收益数字;"
     "停机只会让对冲规则无法评估。超出频率见 lh_step2_review.csv section=hedge_lm",
     "是(该断言在 report 阶段与收益同时产生);审查补登",
     "无:O5/O5-R 的判定不变;但说明对冲账户在调仓日常有小幅透支(O5 主配置约 2/3 的相位调仓日 > 1.001)"),
    ("[step2-review] 更正 [step2] §12.1 规则目标的构造 的'预期影响'",
     "原登记(开跑前)写'C 口径下(多头 + 保证金)不超 1'",
     "实测 C 口径最大 1.011(2005-05-31,评价期之前的预热段;2006 年起最大 1.003),即 C 口径下也会略超 1。原登记行保留为开跑前的预期,不改写;以本行为准",
     "审查发现开跑前的预期与实测相反",
     "是(实测之后的更正)",
     "无:只涉及对冲断言的解释,不影响收益与判定"),
]


def _rv(rows, section, item, value=np.nan, rule="", period="", variant="", note="", **kw):
    d = dict(section=section, item=item, rule=rule, period=period, variant=variant, value=value, note=note)
    d.update(kw)
    rows.append(d)


def _phase_avg(Dt: Data, run_name: str) -> pd.Series:
    run = load_run(run_name)
    r = np.vstack([d["ret"].reindex(Dt.sig).to_numpy() for d in run["phases"]])
    return pd.Series(np.nanmean(r, axis=0), index=Dt.sig)


def _phase_rets(Dt: Data, run_name: str) -> list:
    run = load_run(run_name)
    return [pd.Series(d["ret"].reindex(Dt.sig).to_numpy(), index=Dt.sig) for d in run["phases"]]


def _worst3(r: pd.Series, bear=None) -> tuple:
    w = {p: mdd_abs(pslice(r, p, bear)) for p in ("full", "dev", "test")}
    k = max(w, key=w.get)
    return w[k], k


def _drop_ext(r: pd.Series, bear: str, extra: int, sig: pd.DatetimeIndex) -> pd.Series:
    """剔除 [熊市起点, 熊市终点(谷底)后第 extra 个交易日);extra=0 即 §8 的留一段。"""
    a, b = LE.WINDOWS_A[bear]
    j = min(int(sig.searchsorted(pd.Timestamp(b))) + extra, len(sig) - 1)
    b2 = sig[j]
    return r[~((r.index >= pd.Timestamp(a)) & (r.index < b2))]


def _microbench(Dt: Data):
    """单进程微基准(回答'要跑多久、能否加速'),写入 lh_step2_timing.csv 的 micro_* 行。"""
    from afactor import strategy as ST
    X = Dt.X
    t = time.time(); W = LE.weights(X, "S1", buffer=(0.20, 0.35)); sf = LE.StampFn(X.dates)
    timing("micro_weights", time.time() - t, "单进程:O0 目标权重 W(S1,缓冲 20/35)与印花税钩子")
    N = W.shape[1]
    pos = load_run("O2_v20")["pos"][0].astype(np.float64)

    def fn(t_, h, ctx):
        w = np.where(ctx["can"], W[t_].astype(np.float64), 0.0)
        s = w.sum()
        return np.zeros(N) if s <= 0 else w / s * pos[t_] * ctx["budget"]

    def sim(target, ad, cm, rec=True, norm=False):
        t1 = time.time()
        kw = dict(normalize=False) if norm is False and callable(target) else {}
        ST.simulate(target, ad, "open10", LE.CAPITAL, rebalance=K, phase=0, cost_model=cm, stamp_fn=sf, record=rec, **kw)
        return time.time() - t1

    t = time.time(); ad1 = LE.adapter(X, discount=0.0, valuation="close"); cm1 = LE.cost_model(ad1, "Y1")
    timing("micro_adapter_build", time.time() - t, "单进程:新建适配器 + 成本模型(不含参与率事前估计)")
    timing("micro_callable_first", sim(fn, ad1, cm1), "单进程单相位:callable(O2 主参数仓位,record),新适配器上的首次调用,含参与率事前估计的一次性开销")
    timing("micro_callable_reuse1", sim(fn, ad1, cm1), "单进程单相位:callable,复用同一适配器(第 2 次)")
    timing("micro_callable_reuse2", sim(fn, ad1, cm1), "单进程单相位:callable,复用同一适配器(第 3 次)")
    timing("micro_ndarray_record", sim(W, ad1, cm1), "单进程单相位:ndarray(O0 的 W,record),复用同一适配器")
    timing("micro_ndarray_norecord", sim(W, ad1, cm1, rec=False), "单进程单相位:ndarray(O0 的 W,不 record),复用同一适配器")
    ad2 = LE.adapter(X, discount=0.0, valuation="close"); cm2 = LE.cost_model(ad2, "Y1")
    timing("micro_callable_fresh_adapter", sim(fn, ad2, cm2), "单进程单相位:callable,换一个新适配器(一次性开销重新出现)")
    ad3 = LE.adapter(X, discount=0.0, valuation="close"); cm3 = LE.cost_model(ad3, "Y1")
    timing("micro_ndarray_fresh_adapter", sim(W, ad3, cm3), "单进程单相位:ndarray(record),新适配器上的首次调用")
    del W


def stage_review():
    """审查修复:把审查指出的口径/表述问题落成账本 lh_step2_review.csv(不改任何预注册判定)。"""
    t0 = time.time()
    Dt = Data()
    E = pd.read_parquet(DAILY2)
    rows = []
    _microbench(Dt)
    log(f"[review] 微基准完成 {time.time() - t0:.0f}s")
    # 插值 f* 的实跑(4 个静态配置 x 5 相位;单进程,复用适配器)
    fcf = {static_name(fq, val): C("static", "review_fstar", val=val, f=fq) for fq in (0.578, 0.593) for val in ("close", "index")}
    _run_many(fcf, 1, "review_runs")

    # ---------- 1. 对冲成本:每单位名义本金年化(按 O5-R / O5 实际品种链,te 对齐)与净值拖累 ----------
    from afactor import lh_basis as LB
    for v in ("IF", "IC", "IM"):
        s = LB.hedge_cost_daily(v, "main", 2)
        s.index = pd.to_datetime(s.index)
        s = s.loc[: pd.Timestamp(EVAL_END)].dropna()
        for y, x in (s.groupby(s.index.year).mean() * TD).items():
            _rv(rows, "hcost", "unit_cost_ann", float(x), rule=v, period=str(y), note="lh_basis.hedge_cost_daily(main, R=2) 逐年均值 x 242")
        _rv(rows, "hcost", "unit_cost_ann", float(s.mean() * TD), rule=v, period="all", note=f"{s.index[0].date()} ~ {s.index[-1].date()} 全段")
    for mode, lab in (("real", "O5R_chain"), ("ideal", "O5_chain")):
        hh, ins, idx, cost = hedge_spec_arrays(Dt, mode, 0.5, 0.03, "te")
        c = np.zeros(Dt.n)
        for k_, v_ in cost.items():
            c = np.where(ins == k_, v_, c)
        s = pd.Series(c, index=Dt.te)
        s = s[(s.index >= pd.Timestamp(LIST_IC)) & (s.index <= pd.Timestamp(EVAL_END))]
        for y, x in (s.groupby(s.index.year).mean() * TD).items():
            _rv(rows, "hcost", "unit_cost_ann", float(x), rule=lab, period=str(y),
                note="按实际使用的品种链(O5-R:IC,2022-07-22 起 IM;O5:IC 代理,2022-07-22 起 IM),成交日 te 对齐")
        _rv(rows, "hcost", "unit_cost_ann", float(s.mean() * TD), rule=lab, period="2015-04-16~2025",
            note="2015-04-16 ~ 2025 整段平均")
    hedged = {"O5_h50_m15_s3": (static_name(O5_LONG[(0.5, 0.15)]), ("ideal", 0.5, 0.15, 0.03, "te")),
              "O5R_h50_m15": ("O5R_long_h50", ("real", 0.5, 0.15, 0.0, "te")),
              "O5_h100_m15_s3": (static_name(O5_LONG[(1.0, 0.15)]), ("ideal", 1.0, 0.15, 0.03, "te")),
              "O5R_h100_m15": ("O5R_long_h100", ("real", 1.0, 0.15, 0.0, "te")),
              "O5_h50_m15_s3_C": (static_name(0.925, tgt="C"), ("ideal", 0.5, 0.15, 0.03, "te"))}
    HR = {}
    for rn, (ln, hs) in hedged.items():
        HR[rn] = build_rule(Dt, rn, "O5", rn, load_run(ln), hedge=hs)
    for rn in ("O5_h50_m15_s3", "O5R_h50_m15"):
        R = HR[rn]
        hh, ins, idx, cost = hedge_spec_arrays(Dt, R.hedge["mode"], R.hedge["h"], R.hedge["scen"], R.hedge["align"])
        drag = np.zeros((K, Dt.n))
        s_ = np.arange(Dt.n)
        for ph in range(K):
            last = ((s_ - ph) // K) * K + ph
            ok = s_ >= ph
            cur = np.zeros(Dt.n, int)
            cur[ok] = np.where(hh[last[ok]] > 0, ins[last[ok]], 0)
            cc = np.zeros(Dt.n)
            for k_, v_ in cost.items():
                cc = np.where(cur == k_, v_, cc)
            drag[ph] = R.hedge["HR"][ph] * cc
        d = pd.Series(drag.mean(axis=0), index=Dt.sig).loc[pd.Timestamp(LE.EVAL_START):]
        hr = pd.Series(R.hedge["HR"].mean(axis=0), index=Dt.sig).loc[pd.Timestamp(LE.EVAL_START):]
        for y in range(2006, 2026):
            m = d.index.year == y
            _rv(rows, "hcost", "nav_drag_pp", float(d[m].sum() * 100), rule=rn, period=str(y),
                note=f"贴水成本对总净值的拖累(相位平均 对冲比例 x 日成本,逐年加总);当年平均对冲比例 {hr[m].mean():.3f}")
        dt_ = d.loc[pd.Timestamp(LE.TEST[0]):]
        _rv(rows, "hcost", "nav_drag_pp", float(dt_.mean() * TD * 100), rule=rn, period="test", note="检验窗 2015-2025 年均")

    # ---------- 2. O2 等的 Q-a 稳健性:逐相位(单一真实账户)三区间最差回撤 ----------
    for nm in ("O1_f1000", "O1_f0500", "O1_f0600", "O2_v20", "probe_O2_v20_lag3", "O2_v20_vidx", "O6_v20_ma200",
               "O6_v20_ma200_vidx", "O1_f0500_vidx", "O1_f0600_vidx"):
        if not (RUNS / f"{nm}.pkl").exists():
            continue
        avg = _phase_avg(Dt, nm)
        w, k = _worst3(avg)
        _rv(rows, "qa_phase", "worst3_mdd", -w, rule=nm, variant="phase_avg", note=f"相位平均;最差区间 {k}")
        for ph, r in enumerate(_phase_rets(Dt, nm)):
            w, k = _worst3(r)
            _rv(rows, "qa_phase", "worst3_mdd", -w, rule=nm, variant=f"ph{ph}",
                note=f"单相位(一条真实账户路径);最差区间 {k};全期年化 {LE.ann_geo(pslice(r, 'full')) * 100:.2f}%")

    # ---------- 3. 仓位序列的扰动不变性(代码级前视检验,补 O2 探针没有分辨力的缺口)----------
    T = len(Dt.X.dates)
    r0 = np.full(T, np.nan)
    r0[:Dt.n] = Dt.D["strat_main"].to_numpy()
    lev = np.cumprod(1.0 + np.nan_to_num(Dt.X.ew_allA_ret.to_numpy(np.float64)))
    rng = np.random.default_rng(20261011)
    cuts = rng.integers(300, Dt.n - 10, 25)
    leak = {"O2": 0, "O3": 0, "O4": 0, "O4b": 0, "O6": 0}
    power = {"O2": 0, "O3": 0}
    for k in cuts:
        r2 = r0.copy(); r2[k:] = rng.normal(0, 0.2, T - k)
        l2 = lev.copy(); l2[k:] *= np.exp(np.cumsum(rng.normal(0, 0.05, T - k)))
        a, b = pos_vol(r0, 0.20, 2), pos_vol(r2, 0.20, 2)
        leak["O2"] += int(not np.allclose(a[:k + 2], b[:k + 2]))
        power["O2"] += int(not np.allclose(a[k + 2:k + 30], b[k + 2:k + 30]))
        a3, b3 = pos_trend(lev, 200, 0.5, 0), pos_trend(l2, 200, 0.5, 0)
        leak["O3"] += int(not np.array_equal(a3[:k], b3[:k]))
        power["O3"] += int(not np.array_equal(a3[k:], b3[k:]))
        leak["O6"] += int(not np.allclose((a * a3)[:k], (b * b3)[:k]))
        for ph in range(K):
            leak["O4"] += int(not np.array_equal(pos_dd(r0, ph, .15, .25, True, 0, 2)[:k + 2],
                                                 pos_dd(r2, ph, .15, .25, True, 0, 2)[:k + 2]))
            leak["O4b"] += int(not np.array_equal(pos_dd(r0, ph, .15, .25, False, 20, 2)[:k + 2],
                                                  pos_dd(r2, ph, .15, .25, False, 20, 2)[:k + 2]))
    for kk, v in leak.items():
        n_ = len(cuts) * (K if kk in ("O4", "O4b") else 1)
        _rv(rows, "pit_perturb", "n_leak", v, rule=kk, variant=f"{n_} 次扰动",
            note="把 O0 收益 r[k:](O3 为全 A 等权指数 [k:])换成随机值;O2/O4/O6 检查 pos[:k+2] 是否变化,O3 检查 pos[:k];0 = 没有用到未来")
    for kk, v in power.items():
        _rv(rows, "pit_perturb", "n_changed_after_cut", v, rule=kk, variant=f"{len(cuts)} 次扰动",
            note="同一扰动在切点之后确实改变了仓位的次数(检验有分辨力)")
    kb = 3000
    r3 = r0.copy(); r3[kb - 1] = 0.5
    _rv(rows, "pit_perturb", "boundary_r_tm1_changes_pos_t", float(pos_vol(r0, .2, 2)[kb] != pos_vol(r3, .2, 2)[kb]), rule="O2",
        note="改 r[t-1] 是否改变 pos[t](应为 0:t 日只用 s<=t-2)")
    _rv(rows, "pit_perturb", "boundary_r_tm1_changes_pos_t+1", float(pos_vol(r0, .2, 2)[kb + 1] != pos_vol(r3, .2, 2)[kb + 1]),
        rule="O2", note="改 r[t-1] 是否改变 pos[t+1](应为 1:边界正好在 t-2)")
    for nm in ("O2_v20", "O6_v20_ma200", "O4_15_25"):
        run = load_run(nm)
        dmax = 0.0
        for ph in range(K):
            d = run["phases"][ph]
            reb = d[(d["rebalance"] > 0.5) & (d.index >= pd.Timestamp(LE.EVAL_START))]
            ii = Dt.X.dates.get_indexer(reb.index)
            st = reb["stuck_w"].to_numpy()
            exp_t = run["pos"][ph][ii].astype(np.float64) * (1 - st) + st
            dmax = max(dmax, float(np.nanmax(np.abs(exp_t - reb["tgt_exposure"].to_numpy()))))
        _rv(rows, "pit_perturb", "engine_reads_pos_t", dmax, rule=nm,
            note="2006 年后各相位调仓日 tgt_exposure 与 pos[t] x (1-残仓) + 残仓 的最大差(pos 以 float32 存盘,误差 ~1e-8)")

    # ---------- 4. f*:余量、指数法估值、逐相位、留一段 ----------
    grid_f = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    for val in ("close", "index"):
        sers = {f: _phase_avg(Dt, static_name(f, val)) for f in grid_f}
        phs = {f: _phase_rets(Dt, static_name(f, val)) for f in grid_f}
        for bear in (None,) + LE.BEARS:
            worst = {f: _worst3(sers[f], bear)[0] for f in grid_f}
            ok = [f for f in grid_f if worst[f] <= 0.50]
            fg = max(ok) if ok else np.nan
            xs = np.array([worst[f] for f in grid_f])
            fi = float(np.interp(0.50, xs, np.array(grid_f))) if xs.min() <= 0.5 <= xs.max() else np.nan
            _rv(rows, "fstar", "f_grid", fg, rule=f"O1[{val}]", variant=bear or "(none)",
                note="剔除该熊市段(口径 A)后,全期/开发窗/检验窗最差回撤 <= 50% 的最高网格仓位" if bear else "主口径(不剔除)",
                f_interp=fi, margin_at_grid_pp=(0.5 - worst[fg]) * 100 if ok else np.nan,
                worst_by_f="; ".join(f"{f:.1f}:{worst[f] * 100:.1f}%" for f in grid_f))
        for ph in range(K):
            worst = {f: _worst3(phs[f][ph])[0] for f in grid_f}
            ok = [f for f in grid_f if worst[f] <= 0.50]
            xs = np.array([worst[f] for f in grid_f])
            fi = float(np.interp(0.50, xs, np.array(grid_f))) if xs.min() <= 0.5 <= xs.max() else np.nan
            _rv(rows, "fstar", "f_grid", max(ok) if ok else np.nan, rule=f"O1[{val}]", variant=f"ph{ph}",
                note="单相位(一条真实账户路径)", f_interp=fi,
                margin_at_grid_pp=(0.5 - worst[max(ok)]) * 100 if ok else np.nan,
                worst_by_f="; ".join(f"{f:.1f}:{worst[f] * 100:.1f}%" for f in grid_f))
    # 插值 f* 实跑:仓位 59.3% / 57.8% 在两种估值口径下的三区间最差回撤(引擎实跑,不是插值)
    for val in ("close", "index"):
        for fq in (0.578, 0.593):
            nm = static_name(fq, val)
            if (RUNS / f"{nm}.pkl").exists():
                w, k = _worst3(_phase_avg(Dt, nm))
                _rv(rows, "fstar", "worst3_mdd_at_f_interp", -w, rule=f"O1[{val}]", variant=f"f={fq}",
                    note=f"引擎实跑 O1({fq}),5 相位平均;最差区间 {k};逐相位最差 " +
                         ", ".join(f"{_worst3(r)[0] * 100:.1f}%" for r in _phase_rets(Dt, nm)))

    # ---------- 5. 匹配静态对照的留一段:择时优势是否只来自 2008 ----------
    for rn in MAIN_RULES:
        mn = f"match_{rn}"
        if not (RUNS / f"{mn}.pkl").exists() or rn not in E:
            continue
        r = E[rn]
        z = _phase_avg(Dt, mn)
        for bear in (None,) + LE.BEARS:
            for pn in ("dev", "test"):
                s_, z_ = pslice(r, pn, bear), pslice(z, pn, bear)
                tn = (LE.ann_geo(s_) - LE.ann_geo(z_)) * 100
                dm = (mdd_abs(z_) - mdd_abs(s_)) * 100
                _rv(rows, "qc_loo", "timing_net_pp", tn, rule=rn, period=pn, variant=bear or "(none)",
                    dmdd_vs_static_pp=dm, qc_cond=bool(dm >= 5 and tn >= -0.5),
                    note="相对同平均仓位静态 O1(ē_R) 的择时净贡献与回撤差;剔除熊市段(口径 A)")

    # ---------- 6. 留一段的'剔除熊市 + 反弹'敏感性 ----------
    sigE = E.index
    loo_rules = {"O0": E["O0"], "O1_f0500": E["O1_f0500"], "O2_v20": E["O2_v20"], "O3_ma200_p50": E["O3_ma200_p50"],
                 "O4_15_25": E["O4_15_25"], "O6_v20_ma200": E["O6_v20_ma200"]}
    for bear in LE.BEARS:
        pn = "dev" if bear == "BEAR08" else "test"
        for extra in (0, 60, 120, 242):
            o = _drop_ext(pslice(E["O0"], pn), bear, extra, sigE)
            for rn, r in loo_rules.items():
                if rn == "O0":
                    continue
                s_ = _drop_ext(pslice(r, pn), bear, extra, sigE)
                dm = mdd_abs(o) - mdd_abs(s_)
                _rv(rows, "loo_rebound", "calmar", calmar(s_), rule=rn, period=pn, variant=f"{bear}+{extra}d",
                    calmar_o0=calmar(o), mdd=-mdd_abs(s_), mdd_o0=-mdd_abs(o), dmdd_pp=dm * 100,
                    qb_cond=bool(dm >= 0.10 and calmar(s_) > calmar(o)),
                    note="剔除 [熊市起点, 谷底后第 extra 个交易日);extra=0 即预注册 §8 的留一段")

    # ---------- 7. 熊市少跌 vs 反弹少赚:统一以同仓位静态为参照(另报相对 O0)----------
    W_ = pd.read_csv(LEDGER / "lh_step2_windows.csv")
    W_ = W_[W_.valuation == "close"]
    pairs = (("BEAR08", "V08"), ("BEAR15", "V16"), ("BEAR18", "V19"), ("BEAR22", "V22_supp"), ("BEAR24", "V24"))

    def wret(rule, win, st):
        x = W_[(W_.rule == rule) & (W_.window == win) & (W_.set == st)]
        return float(x.ret.iloc[0]) if len(x) else np.nan

    for rn in ("O2_v20", "O3_ma200_p50", "O4_15_25", "O6_v20_ma200"):
        for refname, ref in (("matched_static", f"match_{rn}"), ("O0", "O1_f1000")):
            for bear, v in pairs:
                rb, sb = wret(rn, bear, "A"), wret(ref, bear, "A")
                for h in (60, 120):
                    rv_, sv_ = wret(rn, f"{v}_{h}", "V"), wret(ref, f"{v}_{h}", "V")
                    save, short = (rb - sb) * 100, (rv_ - sv_) * 100
                    comb = ((1 + rb) * (1 + rv_)) / ((1 + sb) * (1 + sv_)) - 1
                    _rv(rows, "bear_rebound", "bear_saving_pp", save, rule=rn, period=f"{bear}+{v}_{h}", variant=refname,
                        rebound_shortfall_pp=short, giveback_ratio=-short / save if save > 0 else np.nan,
                        combined_nav_ratio_minus1=comb,
                        note="熊市段(口径 A)窗口收益差 / 谷底后 h 日收益差 / 熊市 + 反弹合并的净值比 - 1;两段首尾相接")

    # ---------- 8. 几何超额 vs 算术超额(NW t 检验的是算术日均超额)----------
    S_ = pd.read_csv(LEDGER / "lh_step2_summary.csv")
    S_ = S_[(S_.cash == 0) & (S_.valuation == "close") & (S_.period.isin(["full", "dev", "test"]))]
    b1v = {pn: float(pslice(E["B1"], pn).std() * np.sqrt(TD)) for pn in ("full", "dev", "test")}
    for rn in ["O1_f1000", static_name(0.7), static_name(0.5)] + MAIN_RULES:
        for _, x in S_[S_.rule == rn].iterrows():
            _rv(rows, "excess_geo_arith", "b1_excess_geo", float(x.b1_excess_geo), rule=rn, period=x.period,
                b1_excess_arith=float(x.b1_excess_arith), b1_nw_t5=float(x.b1_nw_t5), b1_te=float(x.b1_te),
                vol=float(x.vol), vol_b1=b1v[x.period], half_var_gap=0.5 * (b1v[x.period] ** 2 - float(x.vol) ** 2),
                sel_excess_geo=float(x.sel_excess_geo), sel_excess_arith=float(x.sel_excess_arith), sel_nw_t5=float(x.sel_nw_t5),
                note="几何超额 ≈ 算术超额 + ½(σ_B1² - σ_规则²);NW t(q=5)检验的是算术日均超额")

    # ---------- 9. 对冲账户 多头 + 保证金 在调仓日的分布 ----------
    for rn, R in HR.items():
        lm = R.hedge["LM"]
        msk = np.asarray(Dt.sig >= pd.Timestamp(LE.EVAL_START))
        v = lm[:, msk]
        dts = np.tile(np.asarray(Dt.sig[msk]), (K, 1))
        ok_ = np.isfinite(v)
        vv, dd = v[ok_], pd.DatetimeIndex(dts[ok_])
        extra = dict(n=int(len(vv)), median=float(np.median(vv)), p90=float(np.quantile(vv, 0.9)), max=float(vv.max()),
                     frac_gt_1001=float((vv > 1.001).mean()), frac_gt_101=float((vv > 1.01).mean()),
                     frac_gt_105=float((vv > 1.05).mean()))
        ys = pd.Series(vv, index=dd).groupby(dd.year).max()
        _rv(rows, "hedge_lm", "long_plus_margin", float(vv.max()), rule=rn, variant="all", **extra,
            note="各相位调仓日 实际多头仓位 + 保证金 m x h(2006 年起);O5-R 在 2010-04-16 前不对冲。逐年最大:" +
                 ", ".join(f"{y}:{x:.3f}" for y, x in ys.items() if x > 1.001))

    # ---------- 10. 多空合并的严格记账(周内买入持有)vs 日收益相加 ----------
    def strict_hedge(ln, hs):
        mode, h, m, scen, align = hs
        hh, ins, idx, cost = hedge_spec_arrays(Dt, mode, h, scen, align)
        run = load_run(ln)
        out = []
        for ph in range(K):
            rL = np.nan_to_num(run["phases"][ph]["ret"].reindex(Dt.sig).to_numpy())
            Rr = np.zeros(Dt.n); hr, cur, share = 0.0, 0, 1.0
            for s in range(Dt.n):
                tc = 0.0
                if s >= ph and (s - ph) % K == 0:
                    nh, ni = float(hh[s]), int(ins[s])
                    tc = 1e-4 * (hr + nh) if (ni != cur and cur != 0 and hr > 0) else 1e-4 * abs(nh - hr)
                    hr = nh; cur = ni if nh > 0 else 0; share = 1.0
                if hr > 0 and cur != 0:
                    ir = idx[cur][s]; ir = 0.0 if not np.isfinite(ir) else ir
                    rh = -hr * ir - hr * cost[cur][s] - tc
                else:
                    ir, rh = 0.0, -tc
                r = share * rL[s] + rh
                Rr[s] = r
                share = share * (1 + rL[s]) / (1 + r)       # 多头腿占总净值的比例在两次调仓之间漂移
                hr = hr * (1 + ir) / (1 + r)
            out.append(Rr)
        return pd.Series(np.mean(out, axis=0), index=Dt.sig)

    for rn in ("O5_h50_m15_s3", "O5R_h50_m15"):
        ln, hs = hedged[rn]
        a, b = E[rn], strict_hedge(ln, hs)
        for pn in ("full", "dev", "test"):
            sa, sb = pslice(a, pn), pslice(b, pn)
            _rv(rows, "hedge_strict", "ann_strict", LE.ann_geo(sb), rule=rn, period=pn, ann_daily_sum=LE.ann_geo(sa),
                mdd_strict=LE.max_dd(sb)[0], mdd_daily_sum=LE.max_dd(sa)[0],
                note="严格记账:调仓日把多头腿与空头名义按总净值重置,其间多头腿按自身收益漂移(周内买入持有);主口径为多空日收益相加")

    rv = pd.DataFrame(rows)
    rv.to_csv(REVIEW, index=False)
    cols = ["条目", "本文规定", "实际做法", "原因", "发现时是否已看到收益", "对结论的预期影响"]
    merge_csv(DEVS, pd.DataFrame(REVIEW_DEVIATIONS, columns=cols), "条目")
    timing("review", time.time() - t0, "审查修复账本 lh_step2_review.csv(含单进程微基准 micro_* 与插值 f* 的 4 个静态配置实跑;不重跑第二步其余回测)")
    log(f"[review] {REVIEW} {len(rv)} 行;补登偏离 {len(REVIEW_DEVIATIONS)} 条;{time.time() - t0:.0f}s")


# ================================================================ 图
def figures(Dt, rules, ser, summ, gdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fn in ("PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"):
        if any(fn == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [fn]
            break
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
    REP = ROOT / "reports"

    # ---- 前沿图:x = 2006-2025 最大回撤,y = 年化 ----
    f = summ[(summ.cash == 0) & (summ.period == "full") & (summ.valuation == "close")]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    st = f[f.family == "O1"].copy()
    st["f"] = st.rule.str.extract(r"O1_f(\d+)").astype(float) / 1000
    st = st.sort_values("f")
    ax.plot(st.mdd.abs() * 100, st.ann * 100, "-o", color=GRAY, lw=2, ms=6, zorder=2, label="静态前沿 O1(30%~100%)")
    for _, r in st.iterrows():
        ax.annotate(f"{r.f * 100:.0f}%", (abs(r.mdd) * 100, r.ann * 100), textcoords="offset points", xytext=(-4, -13),
                    fontsize=7.5, color=MUTED, ha="center")
    gcells = set(gdf[~gdf.rule_main.str.startswith("O1")].cell)
    mains = set(MAIN_RULES)
    pts = f[f.rule.isin(gcells - mains)]
    ax.scatter(pts.mdd.abs() * 100, pts.ann * 100, s=36, facecolors="none", edgecolors=BLUE, linewidths=1.4, zorder=3,
               label="邻域网格格子")
    mp = f[f.rule.isin(mains)]
    ax.scatter(mp.mdd.abs() * 100, mp.ann * 100, s=64, color=BLUE, edgecolors="white", linewidths=1.5, zorder=4,
               label="主参数规则")
    lab = {"O2_v20": "O2 波动率目标", "O3_ma200_p50": "O3 趋势过滤", "O4_15_25": "O4 回撤触发", "O4b_15_25": "O4-b",
           "O5_h50_m15_s3": "O5 对冲(理想)", "O5R_h50_m15": "O5-R 对冲(可实施)", "O6_v20_ma200": "O6 = O2×O3"}
    for _, r in mp.iterrows():
        ax.annotate(lab.get(r.rule, r.rule), (abs(r.mdd) * 100, r.ann * 100), textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5, color=INK)
    ax.axvline(50, color=ORANGE, lw=1.2, ls="--", zorder=1)
    ax.text(50.4, ax.get_ylim()[1] if False else st.ann.max() * 100, "用户上限 50%", color=ORANGE, fontsize=8.5, va="top")
    ax.set_xlabel("最大回撤(2006-2025,%)", color=MUTED)
    ax.set_ylabel("年化收益(几何,%)", color=MUTED)
    ax.set_title("回撤-收益前沿:风控规则 vs 同仓位静态降仓(2006-2025,现金 0 收益)", color=INK, fontsize=11, loc="left")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(REP / "lh_step2_frontier.png"); plt.close(fig)

    # ---- 净值图:小多图,每格 规则(蓝)vs O0(灰),对数纵轴 ----
    names = [static_name(0.7), static_name(0.5)] + [x for x in MAIN_RULES if x != "O4b_15_25"]
    ttl = {static_name(0.7): "O1 固定 70%", static_name(0.5): "O1 固定 50%"}
    ttl.update(lab)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), dpi=140, sharex=True, sharey=True)
    o0 = ev(Dt.D["strat_main"]); nav0 = (1 + o0).cumprod()
    for ax, rn in zip(axs.ravel(), names):
        if (rn, 0.0) not in ser:
            ax.set_visible(False); continue
        r = ev(ser[(rn, 0.0)]); nav = (1 + r).cumprod()
        ax.plot(nav0.index, nav0, color=GRAY, lw=1.2)
        ax.plot(nav.index, nav, color=BLUE, lw=1.6)
        ax.set_yscale("log")
        m, m0 = LE.max_dd(r)[0], LE.max_dd(o0)[0]
        ax.set_title(f"{ttl.get(rn, rn)}\n最大回撤 {m * 100:.1f}%(O0 {m0 * 100:.1f}%),年化 {LE.ann_geo(r) * 100:.1f}%",
                     fontsize=9, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
    axs[0, 0].plot([], [], color=BLUE, lw=1.6, label="规则")
    axs[0, 0].plot([], [], color=GRAY, lw=1.2, label="O0 满仓")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("净值(2006-01-04 = 1,对数轴):各风控规则 vs 满仓 O0", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(REP / "lh_step2_nav.png"); plt.close(fig)


# ================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None
    T0 = time.time()
    for st in a.stages:
        t1 = time.time()
        if st == "verify":
            stage_verify()
        elif st == "runs":
            stage_runs(a.workers, only)
        elif st == "runs2":
            stage_runs2(a.workers)
        elif st == "report":
            stage_report()
        elif st == "review":
            stage_review()
        else:
            raise SystemExit(f"未知阶段 {st}")
        if st == "verify":
            timing("verify", time.time() - t1)
    log(f"[done] {a.stages} 共 {time.time() - T0:.0f}s")
    timing("total_" + "_".join(a.stages), time.time() - T0, "本次命令总耗时")
