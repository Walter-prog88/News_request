"""INC-T3(限售解禁)共用:路径、快照校验、事件表读取、类型 / 层映射、聚类与 NW 统计。

预注册:research/INC_T3_PREREG.md(md5 4a78c40376e5a6ee2a32a0f676017363)。本模块只放口径与工具函数,
不计算任何收益。

⚠ 东财快照里有两列事后收益(B20_ADJCHRATE / A20_ADJCHRATE,解禁前后 20 日涨跌幅),
   load_em() 读入后立即丢掉,本轨道任何地方都不得使用。
"""
from __future__ import annotations
import os, sys, re, json, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import ROOT, TMP, LEDGER, REPORTS, RESEARCH_END   # noqa: E402

PREREG = ROOT / "research" / "INC_T3_PREREG.md"
PREREG_MD5 = "4a78c40376e5a6ee2a32a0f676017363"
EM_RAW_DIR = TMP / "inc_t3_em_lift_stage_raw"
EM_RAW_MD5 = "b763cb351ae9c58b6a97ef8961cd7c1c"
EM_PARQUET = TMP / "inc_t3_em_lift_stage_2020_2025.parquet"
EM_PARQUET_MD5 = "8c2fcad88c57ddc8331cbb304941ccad"
COMP_PATH = TMP / "comp_v5_score_open10.parquet"
COMP_MD5 = "d011468feacc9a71f463d3f7c13c2e05"
CAPGRID = LEDGER / "capacity_grid.csv"
CAPGRID_MD5 = "2f73a540d5ef5a2b7e536ef815e7962d"
JUMPS = LEDGER / "mvfix_songzhuan_events.csv"
JUMPS_MD5 = "f182a2d460669435f69ec47a47d4704f"

EVENTS_CSV = LEDGER / "inc_t3_unlock_events.csv"
SINA_DIR = TMP / "inc_t3_sina_xsjj"
SEED = 20261011
VERSION = "V1000"
CUT = pd.Timestamp(RESEARCH_END)

DEV = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31"))
TEST = (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"))
FULL = (pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31"))
WINDOWS = {"2021-2025": FULL, "2021-2023": DEV, "2024-2025": TEST}

SIZE_BINS = [0.05, 0.10, 0.20, 0.50, np.inf]
SIZE_LABELS = ["[5,10)", "[10,20)", "[20,50)", ">=50"]
AGE_EDGES = (250, 750)              # 上市 < 250 / 250~750 / > 750 个交易日
K_A, K_B = 60, 3                    # 已知日:A 层 D-60、B 层 D-3(交易日)


def md5(path, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def raw_concat_md5() -> str:
    def key(f):
        m = re.match(r"lift_(\d+)_p(\d+)\.json", f)
        return (int(m.group(1)), int(m.group(2)))
    fs = sorted([f for f in os.listdir(EM_RAW_DIR) if f.endswith(".json")], key=key)
    h = hashlib.md5()
    for f in fs:
        with open(EM_RAW_DIR / f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


# ---------------------------------------------------------------- 类型 / 层
def first_type(s: str) -> str:
    return str(s).split(",")[0].strip()


def tier_of(s: str) -> str:
    """§3.2:首发*、公开发行*、定向增发*、追加承诺*、股权分置* -> A;其余(股权激励*、期权行权、其他类型、
    以及预注册未列出的债转股)-> B(保守:K = D-3)。逗号分隔时取第一个类型。"""
    f = first_type(s)
    if f.startswith(("首发", "公开发行", "定向增发", "追加承诺", "股权分置")):
        return "A"
    return "B"


def type_group(s: str) -> str:
    """§3.3 第 4 条:首发(含公开发行)/ 定增 / 追加承诺 / 股权激励 / 其他。"""
    f = first_type(s)
    if f.startswith(("首发", "公开发行")):
        return "首发"
    if f.startswith("定向增发"):
        return "定增"
    if f.startswith("追加承诺"):
        return "追加承诺"
    if f.startswith("股权激励"):
        return "股权激励"
    return "其他"


def ts_code(sec: str, market: str) -> str:
    return str(sec).zfill(6) + (".SH" if market in ("sh", "kcb") else ".SZ")


def load_em() -> pd.DataFrame:
    """东财快照(G1 校验过的 parquet),丢掉事后收益列,截到 <= 2025-12-31。"""
    d = pd.read_parquet(EM_PARQUET)
    d = d.drop(columns=[c for c in ("B20_ADJCHRATE", "A20_ADJCHRATE") if c in d.columns])
    d = d[d["FREE_DATE"] <= CUT].copy()
    return d


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_CSV, dtype={"SECURITY_CODE": str},
                     parse_dates=["FREE_DATE", "D", "t_ref_date", "K_date", "list_date"])
    return ev


def size_bucket(x: pd.Series) -> pd.Series:
    return pd.cut(x, bins=SIZE_BINS, labels=SIZE_LABELS, right=False).astype(object)


def age_bucket(age: np.ndarray) -> np.ndarray:
    a = np.asarray(age, np.float64)
    out = np.full(a.shape, -1, np.int8)
    out[a < AGE_EDGES[0]] = 0
    out[(a >= AGE_EDGES[0]) & (a <= AGE_EDGES[1])] = 1
    out[a > AGE_EDGES[1]] = 2
    out[~np.isfinite(a)] = -1
    return out


# ---------------------------------------------------------------- 交易日历 / 上市年龄
def long_calendar() -> pd.DatetimeIndex:
    """2005-2025 交易日历(长历史面板的日期索引;只读索引,不读任何价格)。"""
    import pyarrow.parquet as pq
    f = ROOT / "cache" / "long_daily" / "panel" / "close_px.parquet"
    pf = pq.ParquetFile(f)
    idx_name = None
    md = pf.schema_arrow.pandas_metadata or {}
    for c in md.get("index_columns", []):
        if isinstance(c, str):
            idx_name = c
    tbl = pq.read_table(f, columns=[idx_name] if idx_name else [pf.schema_arrow.names[-1]])
    s = tbl.column(0).to_pandas()
    return pd.DatetimeIndex(pd.to_datetime(s)).sort_values()


def listing_age_panel(dates: pd.DatetimeIndex, codes: pd.Index) -> np.ndarray:
    """上市年龄(交易日)T x N:日 t 时,自 list_date 起的交易日数(list_date 当日计 0)。
    list_date 在长历史日历之前的,按日历起点之前的工作日数 × 0.95 近似(只影响 > 750 桶,无影响)。"""
    from afactor import newdata as ND
    cal = long_calendar()
    cal = cal[cal <= CUT]
    li = ND.listing_info(codes)
    ld = li["list_date"]
    pos_t = cal.get_indexer(dates)
    assert (pos_t >= 0).all(), "研究面板日期不在长历史日历里"
    pos_l = np.searchsorted(cal.to_numpy(), ld.to_numpy(dtype="datetime64[ns]"), side="left").astype(np.float64)
    early = (ld < cal[0]).to_numpy()
    extra = np.where(early, np.busday_count(ld.fillna(cal[0]).to_numpy(dtype="datetime64[D]"),
                                            np.datetime64(cal[0].date())) * 0.95, 0.0)
    pos_l = np.where(early, -extra, pos_l)
    pos_l[ld.isna().to_numpy()] = np.nan
    return pos_t[:, None].astype(np.float64) - pos_l[None, :]


# ---------------------------------------------------------------- 统计
def cluster_mean_se(x: np.ndarray, g: np.ndarray):
    """x (n, k) 的列均值与以 g 为聚类的稳健标准误(CR1)。同 scripts/pead_event_level.cluster_mean_se。"""
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    mu = x.mean(axis=0)
    e = x - mu
    G = pd.Series(np.arange(n)).groupby(np.asarray(g)).indices
    S = np.vstack([e[ix].sum(axis=0) for ix in G.values()])
    nG = len(G)
    se = np.sqrt((S ** 2).sum(axis=0) * nG / max(nG - 1, 1)) / n
    return mu, se, nG


def nw_se(x, L: int) -> float:
    """均值的 Newey-West(Bartlett)标准误。同 afactor.pead_engine.nw_se。"""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, int(L) + 1):
        s += 2.0 * (1.0 - k / (L + 1.0)) * float(e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(s, 0.0) / n))


def win_slice(idx: pd.DatetimeIndex, w) -> np.ndarray:
    a, b = WINDOWS[w] if isinstance(w, str) else w
    return np.asarray((idx >= a) & (idx <= b))


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
