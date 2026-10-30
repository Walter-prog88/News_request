"""真实市值日频外推修复(前缀 mvfix,2026-10-10;research/TOTAL_MV_FIX.md)的回归测试。

合成用例(不读数据,秒级):替换 newdata / fundamental 的加载器,逐条锁住 newdata.total_share 的口径——
解禁不外推、送转外推(深市同日 / 沪市隔日)、按报告期末锚定、已含期后送转不重复计入、新股加法、流通股本下界、
不前视(截断不变性),以及旧口径(legacy)确实会累积解禁(证明用例能抓到原 bug)。
真实数据用例(需要 data_8:21 与 cache/daily,约 2 分钟):审查例 600339.SH / 600390.SH、比亚迪送转、
全面板高估比例上限、daily_basic 截断不变性。

    python3 -m pytest tests/test_total_mv_fix.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor import newdata as ND, fundamental as FU   # noqa: E402

DATES = pd.bdate_range("2022-12-01", "2023-12-29")
HAS_DATA = (ROOT / "data_8:21" / "daily_basic.csv").exists()


# ---------------------------------------------------------------- 合成数据
def _qi(e):
    e = pd.Timestamp(e)
    return e.year * 4 + e.quarter - 1


class Synth:
    """逐只股票描述:流通股本(股)、未复权价、财报 [(报告期末, 公告日, 总股本)]、上市日。"""

    def __init__(self):
        self.fs, self.px, self.rep, self.lst = {}, {}, [], {}

    def stock(self, code, fs, px, reports, list_date=None, start=None):
        idx = DATES if start is None else DATES[DATES >= start]
        self.fs[code] = pd.Series(fs, index=DATES).reindex(idx) if np.ndim(fs) else pd.Series(float(fs), index=idx)
        self.px[code] = pd.Series(px, index=DATES).reindex(idx) if np.ndim(px) else pd.Series(float(px), index=idx)
        for e, a, s in reports:
            self.rep.append(dict(stock_code=code, end_date=pd.Timestamp(e), avail=pd.Timestamp(a), qi=_qi(e),
                                 total_share=float(s), pit_vintage=np.int16(FU.PIT_FINAL)))
        self.lst[code] = pd.Timestamp(list_date) if list_date else pd.Timestamp("2000-01-04")

    def daily_basic(self, upto=None):
        rows = []
        for c in self.fs:
            f, p = self.fs[c], self.px[c]
            d = pd.DataFrame({"ts_code": c, "trade_date": f.index, "close": p.to_numpy(),
                              "turnover_rate": 1.0, "turnover_rate_f": 1.0, "float_share": f.to_numpy() / 1e4,
                              "circ_mv": (f * p).to_numpy() / 1e4})
            rows.append(d)
        d = pd.concat(rows, ignore_index=True)
        return d if upto is None else d[d["trade_date"] <= pd.Timestamp(upto)].reset_index(drop=True)

    def report(self, upto=None):
        r = pd.DataFrame(self.rep)
        return r if upto is None else r[r["avail"] <= pd.Timestamp(upto)].reset_index(drop=True)


def _step(base, changes):
    """分段常数序列:changes = {日期: 新值}。"""
    s = pd.Series(np.nan, index=DATES)
    s.iloc[0] = base
    for d, v in changes.items():
        s.loc[pd.Timestamp(d)] = v
    return s.ffill().to_numpy()


def _px(base, drops):
    """价格:drops = {除权日: 除以的比例}(之后整体下移)。"""
    s = pd.Series(float(base), index=DATES)
    for d, r in drops.items():
        s.loc[pd.Timestamp(d):] /= r
    return s.to_numpy()


@pytest.fixture
def synth(monkeypatch):
    S = Synth()
    state = {"upto": None}
    monkeypatch.setattr(ND, "_load_daily_basic", lambda: S.daily_basic(state["upto"]))
    monkeypatch.setattr(FU, "build_report_level", lambda: S.report(state["upto"]))
    monkeypatch.setattr(FU, "_check_report_coverage", lambda *a, **k: None)
    monkeypatch.setattr(ND, "listing_info", lambda codes: pd.DataFrame(
        {"list_date": pd.Series(S.lst).reindex(codes), "delist_date": pd.NaT}, index=codes))
    monkeypatch.setattr(ND, "TOTAL_SHARE_METHOD", "songzhuan")
    S.state = state
    return S


def _ts(S, codes=None, dates=DATES):
    codes = pd.Index(sorted(S.fs) if codes is None else codes)
    return ND.total_share(dates, codes)


REPORTS_FLAT = [("2022-09-30", "2022-10-28", 1e9), ("2022-12-31", "2023-03-30", 1e9),
                ("2023-03-31", "2023-04-28", 1e9), ("2023-06-30", "2023-08-30", 1e9),
                ("2023-09-30", "2023-10-30", 1e9)]


# ---------------------------------------------------------------- 送转识别
def test_songzhuan_steps_rules():
    """深市同日、沪市隔日、回购专户导致的非整比例 → 送转;解禁(价格不动)、小比例解禁碰上大跌 → 不是。"""
    T = 6
    fs = np.ones((T, 6)) * 100.0
    px = np.ones((T, 6)) * 10.0
    fs[3:, 0] = 140; px[3:, 0] = 10 / 1.4 * 1.03                  # 深市:同日跳升 1.4 + 除权(当日 +3%)
    fs[3:, 1] = 130; px[2:, 1] = 10 / 1.3                          # 沪市:t-1 除权,t 流通股本更新
    fs[3:, 2] = 139.67; px[3:, 2] = 10 / 1.3967                    # 10 转 4,回购专户不参与 -> 1.3967
    fs[3:, 3] = 250                                                # 解禁:流通股本 2.5 倍,价格不动
    fs[3:, 4] = 107.3; px[3:, 4] = 10 * 0.92                       # 小比例解禁 + 当日跌 8%
    fs[3:, 5] = 137; px[3:, 5] = 10 * 0.92                         # 1.37 倍解禁 + 跌 8%:隐含收益 +23% 不像除权
    st = ND._songzhuan_steps(fs, px)
    assert st[3, 0] == pytest.approx(1.4) and st[3, 1] == pytest.approx(1.3) and st[3, 2] == pytest.approx(1.3967)
    assert (st[:, 3:] == 1.0).all()
    assert (np.delete(st, 3, axis=0) == 1.0).all()                 # 其它日子都是 1


# ---------------------------------------------------------------- 口径
def test_unlock_is_not_extrapolated_but_legacy_accumulates(synth, monkeypatch):
    """总股本 10 亿不变,流通股本 2.5 亿 -> 5 亿 -> 10 亿(两次解禁,价格不动):新口径恒为 10 亿;旧口径累积到 40 亿。"""
    synth.stock("A.SZ", _step(2.5e8, {"2023-02-15": 5e8, "2023-07-17": 1e9}), 10.0, REPORTS_FLAT)
    new = _ts(synth)["A.SZ"]
    assert np.allclose(new.loc["2022-12-01":], 1e9)
    monkeypatch.setattr(ND, "TOTAL_SHARE_METHOD", "legacy")
    old = _ts(synth)["A.SZ"]
    assert old.max() == pytest.approx(4e9)                         # 原 bug:解禁倍数 4 累积在总股本上


def test_songzhuan_extrapolated_and_anchored_at_period_end(synth):
    """5-15 送转 1.4(H1 财报期末之前,已在 H1 股本里);7-10 送转 1.3(H1 期末之后、8-30 公告之前,H1 股本未含)。
    8-30 H1 财报生效后应仍是 1.4 × 1.3 = 18.2 亿,而不是被重置成 14 亿(旧口径按生效日锚定会丢掉第二次送转)。"""
    fs = _step(4e8, {"2023-05-15": 5.6e8, "2023-07-10": 7.28e8})
    px = _px(14.0, {"2023-05-15": 1.4, "2023-07-10": 1.3})
    synth.stock("B.SZ", fs, px, [("2022-12-31", "2023-03-30", 1e9), ("2023-03-31", "2023-04-28", 1e9),
                                 ("2023-06-30", "2023-08-30", 1.4e9), ("2023-09-30", "2023-10-30", 1.82e9)])
    s = _ts(synth)["B.SZ"]
    assert np.allclose(s.loc[:"2023-03-29"], 4e8)                 # 首份财报生效前:无 PIT 股本 -> 流通股本
    assert np.allclose(s.loc["2023-03-30":"2023-05-12"], 1e9)
    assert np.allclose(s.loc["2023-05-15":"2023-07-07"], 1.4e9)
    assert np.allclose(s.loc["2023-07-10":], 1.82e9)               # 跨过 8-30 生效日连续


def test_report_already_reflecting_songzhuan_is_not_double_counted(synth):
    """沪市:7-10 除权、7-11 流通股本 ×1.5。H1(期末 6-30)财报 8-25 公告,股本已是送转后的 30 亿(数据源的这种情况约 6%)。
    8-25 之后应是 30 亿,不是 45 亿。"""
    fs = _step(1e9, {"2023-07-11": 1.5e9})
    px = _px(9.0, {"2023-07-10": 1.5})
    synth.stock("C.SH", fs, px, [("2023-03-31", "2023-04-28", 2e9), ("2023-06-30", "2023-08-25", 3e9),
                                 ("2023-09-30", "2023-10-27", 3e9)])
    s = _ts(synth)["C.SH"]
    assert np.allclose(s.loc["2023-05-01":"2023-07-10"], 2e9)      # 沪市除权日当天流通股本还没更新(滞后一天)
    assert np.allclose(s.loc["2023-07-11":], 3e9)


def test_new_listing_adds_first_day_float(synth):
    """3-15 上市,发行前股本 3 亿(招股书期与年报都是上市前报告期),首日流通 0.8 亿:上市后到 Q1 财报生效前 = 3.8 亿;
    另一只的年报(期末早于上市、公告晚于上市)已含发行股份 4 亿 -> 不再加。"""
    fs = _step(np.nan, {"2023-03-15": 8e7, "2023-09-15": 2e8})     # 9-15 网下 / 战配解禁:不外推
    synth.stock("E.SH", fs, 20.0, [("2022-09-30", "2023-03-01", 3e8), ("2022-12-31", "2023-04-20", 3e8),
                                   ("2023-03-31", "2023-04-28", 4e8), ("2023-06-30", "2023-08-30", 4e8)],
                list_date="2023-03-15", start="2023-03-15")
    synth.stock("F.SH", fs, 20.0, [("2022-09-30", "2023-03-01", 3e8), ("2022-12-31", "2023-04-20", 4e8),
                                   ("2023-03-31", "2023-04-28", 4e8)], list_date="2023-03-15", start="2023-03-15")
    s = _ts(synth)
    assert np.allclose(s["E.SH"].loc["2023-03-15":"2023-04-27"], 3.8e8)
    assert np.allclose(s["E.SH"].loc["2023-04-28":], 4e8)
    assert np.allclose(s["F.SH"].loc["2023-03-15":"2023-04-19"], 3.8e8)
    assert np.allclose(s["F.SH"].loc["2023-04-20":], 4e8)          # 已含发行股份:不重复加


def test_float_floor_catches_full_float_additive_issuance(synth):
    """全流通股票 6-01 换股合并 / 重整转增,流通股本 10 亿 -> 12 亿、价格不除权:总股本取流通股本下界 12 亿。"""
    synth.stock("G.SZ", _step(1e9, {"2023-06-01": 1.2e9}), 10.0, REPORTS_FLAT[:3])
    s = _ts(synth)["G.SZ"]
    assert np.allclose(s.loc[:"2023-05-31"], 1e9) and np.allclose(s.loc["2023-06-01":], 1.2e9)


def test_total_share_is_point_in_time(synth):
    """截断不变性:把 t 之后的日线与公告日在 t 之后的财报全部删掉重算,t 及之前逐位不变。"""
    synth.stock("A.SZ", _step(2.5e8, {"2023-02-15": 5e8, "2023-07-17": 1e9}), 10.0, REPORTS_FLAT)
    synth.stock("B.SZ", _step(4e8, {"2023-05-15": 5.6e8, "2023-07-10": 7.28e8}), _px(14.0, {"2023-05-15": 1.4, "2023-07-10": 1.3}),
                [("2022-12-31", "2023-03-30", 1e9), ("2023-03-31", "2023-04-28", 1e9),
                 ("2023-06-30", "2023-08-30", 1.4e9), ("2023-09-30", "2023-10-30", 1.82e9)])
    synth.stock("C.SH", _step(1e9, {"2023-07-11": 1.5e9}), _px(9.0, {"2023-07-10": 1.5}),
                [("2023-03-31", "2023-04-28", 2e9), ("2023-06-30", "2023-08-25", 3e9), ("2023-09-30", "2023-10-27", 3e9)])
    synth.stock("E.SH", _step(np.nan, {"2023-03-15": 8e7, "2023-09-15": 2e8}), 20.0,
                [("2022-09-30", "2023-03-01", 3e8), ("2022-12-31", "2023-04-20", 3e8), ("2023-03-31", "2023-04-28", 4e8)],
                list_date="2023-03-15", start="2023-03-15")
    full = _ts(synth)
    for t in ["2023-03-15", "2023-05-15", "2023-07-10", "2023-07-11", "2023-08-29", "2023-08-30", "2023-10-30"]:
        synth.state["upto"] = t
        part = _ts(synth, dates=DATES[DATES <= t])
        synth.state["upto"] = None
        pd.testing.assert_frame_equal(part, full.loc[:t], check_exact=True)


# ---------------------------------------------------------------- 真实数据
@pytest.fixture(scope="module")
def real():
    if not (HAS_DATA and (ROOT / "cache" / "daily" / "close_px.parquet").exists()):
        pytest.skip("需要 data_8:21 与 cache/daily")
    from afactor.panel import Panel
    p = Panel()
    d, c = p.dates, p.codes
    rep = FU.build_report_level()
    return dict(dates=d, codes=c, new=ND.total_mv(d, c), rc=ND.raw_close(d, c), fs=ND.float_share(d, c),
                sh=FU._pit_daily(rep, "total_share", d, c).replace(0, np.nan))


def test_real_review_examples(real):
    """审查例:600339.SH(旧口径全程 3.60 倍)、600390.SH(7.30 倍)总股本 = 流通股本,新口径 = 未复权价 × PIT 总股本。"""
    for c in ["600339.SH", "600390.SH", "600519.SH"]:
        r = (real["new"][c] / (real["rc"][c] * real["sh"][c])).dropna()
        assert np.allclose(r.loc["2021-01-01":], 1.0, atol=1e-6), c


def test_real_byd_songzhuan_continuity(real):
    """比亚迪 2025-07-29 送转(10 送 8 转 12,3 倍):除权当日起总股本 = 3 × PIT;8-29 H1 财报(期末 6-30,送转前股本)
    生效后仍是 3 倍(按报告期末锚定),12-31(Q3 财报已含送转)与 PIT 相等。"""
    c = "002594.SZ"
    sh = real["new"][c] / real["rc"][c]
    pit = real["sh"][c]
    assert sh.loc["2025-07-28"] / pit.loc["2025-07-28"] == pytest.approx(1.0, abs=1e-6)
    for d in ["2025-07-29", "2025-08-28", "2025-08-29", "2025-09-30"]:
        assert sh.loc[d] / pit.loc[d] == pytest.approx(3.0, rel=1e-4), d
    assert sh.loc["2025-12-31"] / pit.loc["2025-12-31"] == pytest.approx(1.0, abs=1e-6)


def test_real_overstatement_bounded(real):
    """全面板(流通股本 <= PIT 总股本的格子):total_mv / (价 × PIT 总股本) > 1.5 的格子 < 0.5%,
    没有股票持续 250 天以上 > 1.5(旧口径 13.6%、1,318 只;mvfix_extrapolation.csv)。"""
    keep = real["new"].notna() & real["sh"].notna() & (real["fs"] <= real["sh"] * 1.01)
    ratio = (real["new"] / (real["rc"] * real["sh"])).where(keep)
    r = ratio.to_numpy()
    r = r[np.isfinite(r)]
    assert (r > 1.5).mean() < 0.005
    assert int(((ratio > 1.5).sum() >= 250).sum()) == 0
    assert (r < 0.999).mean() == 0.0                                # 外推只会向上(流通股本下界 / 送转 / 新股)


def test_real_daily_basic_truncation_invariance(real, monkeypatch):
    """把 2023-06-30 之后的 daily_basic 行删掉(经旧脚本的注入约定 _CACHE["db"])重算,截止日及之前逐位不变。"""
    t = pd.Timestamp("2023-06-30")
    codes = real["codes"][::7]
    dates = real["dates"][real["dates"] <= t]
    full = ND.total_share(real["dates"], codes).loc[:t]
    db = ND._load_daily_basic()
    monkeypatch.setitem(ND._CACHE, "db", db[db["trade_date"] <= t].copy())
    part = ND.total_share(dates, codes)
    pd.testing.assert_frame_equal(part, full, check_exact=True)
