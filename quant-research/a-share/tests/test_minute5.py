"""minute5(前缀 m5)的单元测试:聚合规则、价格路径、窗口超额分布、baostock 抓取的超时 / 错位防护。
不读任何真实数据、不联网(baostock 用本地假服务器)。"""
import time
import numpy as np
import pandas as pd
import pytest

from afactor import minute5 as M5
from afactor.build_daily_5m import _day_features_5m


def _synthetic(N=3, seed=0):
    rng = np.random.default_rng(seed)
    P = 10 * np.exp(np.cumsum(rng.normal(0, 1e-3, (241, N)), axis=0))
    O = np.vstack([P[:1] * 0.999, P[:-1]])                   # 每分钟 open = 上一分钟 close(idx0 略低)
    V = rng.uniform(1e5, 1e6, (241, N))
    return P, O, V


def test_bar_definition():
    assert len(M5.BAR5_LO) == 48 and M5.BAR5_LO[0] == 0 and M5.BAR5_HI[0] == 5
    assert M5.BAR5_LO[1] == 6 and M5.BAR5_HI[1] == 10                  # 09:40 = 09:36..09:40
    assert M5.BAR5_HI[23] == 120 and M5.BAR5_LO[24] == 121             # 11:30 / 13:05
    assert M5.BAR5_LO[47] == 236 and M5.BAR5_HI[47] == 240             # 15:00 含收盘集合竞价
    # 覆盖 0..240 且不重叠
    cover = np.concatenate([np.arange(lo, hi + 1) for lo, hi in zip(M5.BAR5_LO, M5.BAR5_HI)])
    assert np.array_equal(cover, np.arange(241))
    assert M5.BAR5_TIMES[1] == "09:40" and M5.BAR5_TIMES[5] == "10:00" and M5.BAR5_TIMES[41] == "14:30"


def test_aggregate_day_matches_definition():
    P, O, V = _synthetic()
    P5, O5, V5 = M5.aggregate_day(P, O, V)
    for k, (lo, hi) in enumerate(zip(M5.BAR5_LO, M5.BAR5_HI)):
        np.testing.assert_array_equal(P5[k], P[hi])
        np.testing.assert_array_equal(O5[k], O[lo])
        np.testing.assert_allclose(V5[k], V[lo:hi + 1].sum(axis=0))
    np.testing.assert_allclose(V5.sum(axis=0), V.sum(axis=0))


def test_halted_column_stays_nan():
    P, O, V = _synthetic()
    P[:, 1] = np.nan; O[:, 1] = np.nan; V[:, 1] = np.nan
    P5, O5, V5 = M5.aggregate_day(P, O, V)
    assert np.isnan(P5[:, 1]).all() and np.isnan(V5[:, 1]).all()
    assert np.isfinite(V5[:, 0]).all()


def test_daily_level_fields_exact():
    P, O, V = _synthetic()
    P5, O5, V5 = M5.aggregate_day(P, O, V)
    f = _day_features_5m(P5, O5, V5)
    np.testing.assert_array_equal(f["close_px"], P[240])
    np.testing.assert_allclose(f["amount"], V.sum(axis=0))
    np.testing.assert_array_equal(f["open_auction_px"], O[0])
    np.testing.assert_allclose(f["ret_intraday"], P[240] / O[0] - 1)


def test_window_excess_distribution():
    idx = pd.bdate_range("2024-01-01", periods=30)
    s = pd.Series(0.01, index=idx); b = pd.Series(0.0, index=idx)
    w = M5.window_excess_distribution(s - b, 5)
    assert len(w) == 26 and np.allclose(w, 0.05)
    wc = M5.window_excess_distribution(pd.DataFrame({"strategy": s, "benchmark": b}), 5)
    assert np.allclose(wc, 1.01 ** 5 - 1)
    with pytest.raises(ValueError):
        M5.window_excess_distribution(s, 5, how="compound")
    assert M5.percentile_in(0.05, pd.Series([0.0, 0.1])) == 50.0


# ---------------- 审查修复(2026-10-10)----------------
def test_window_skip_head_and_forward_excess():
    """P0:历史分布去掉开头 skip_head 行;前向窗口同一定义(跳过前 FWD_SKIP_HEAD 个建仓日)。"""
    assert M5.FWD_SKIP_HEAD == M5.FWD_REBALANCE - 1 == 4
    idx = pd.bdate_range("2024-01-01", periods=30)
    rs = pd.Series(np.arange(30) * 1e-3, index=idx)
    rb = pd.Series(0.0, index=idx)
    df = pd.DataFrame({"strategy": rs, "benchmark": rb})
    w0 = M5.window_excess_distribution(df, 5, how="sum")
    w4 = M5.window_excess_distribution(df, 5, how="sum", skip_head=4)
    assert len(w0) == 26 and len(w4) == 22
    assert w4.index[0] == idx[8] and np.isclose(w4.iloc[0], (4 + 5 + 6 + 7 + 8) * 1e-3)
    pd.testing.assert_series_equal(w4, w0.iloc[4:], check_names=False)
    # 前向:跳过前 4 天后的全部天数 / 指定 n
    v, n = M5.forward_window_excess(df.iloc[:12], how="sum")
    assert n == 8 and np.isclose(v, sum(range(4, 12)) * 1e-3)
    v5, n5 = M5.forward_window_excess(df.iloc[:12], n_days=5, how="compound")
    assert n5 == 5 and np.isclose(v5, np.prod(1 + np.arange(4, 9) * 1e-3) - 1)
    # 与历史分布同一定义:前向的第一个窗口 == 历史 skip_head=4 分布的第一个值
    assert np.isclose(M5.forward_window_excess(df, n_days=5, how="sum")[0], w4.iloc[0])
    with pytest.raises(ValueError):
        M5.forward_window_excess(df.iloc[:6], n_days=5)


def _fake_bs_server(delay_codes, delay):
    """本地假 baostock 服务器(127.0.0.1,不联网):按协议回包;delay_codes 里的代码延迟 delay 秒回包。
    每只股票的价格 = 代码数字,便于识别"数据属于谁"。"""
    import socket, threading, json, zlib
    SPLIT = "\x01"
    PX = {"sh.600000": "6.00", "sz.000001": "1.00", "sz.000002": "2.00"}

    def resp(code, fields, start, end):
        val = dict(date="2026-09-01", code=code, open=PX[code], high=PX[code], low=PX[code], close=PX[code],
                   volume="100", amount="1000.0")
        rows = [[(f"20260901{hm}00000" if f == "time" else val[f]) for f in fields.split(",")] for hm in ("0935", "0940")]
        body = SPLIT.join(["0", "success", "query_history_k_data_plus", "anonymous", "1", "2000",
                           json.dumps({"record": rows}), code, fields, start, end, "5", "3"]) + "\n"
        comp = zlib.compress(body.encode())
        return ("00.9.40" + SPLIT + "96" + SPLIT + str(len(comp)).zfill(10)).encode() + comp + b"<![CDATA[]]>\n"

    def handle(conn):
        buf = b""
        while True:
            try:
                d = conn.recv(65536)
            except OSError:
                return
            if not d:
                return
            buf += d
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                parts = line.decode().split(SPLIT)
                code, fields, start, end = parts[6], parts[7], parts[8], parts[9]

                def reply(c=code, f=fields, s=start, e=end, dl=(delay if code in delay_codes else 0.0)):
                    time.sleep(dl)
                    try:
                        conn.sendall(resp(c, f, s, e))
                    except OSError:
                        pass
                threading.Thread(target=reply, daemon=True).start() if code in delay_codes else reply()

    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0)); srv.listen(8)

    def accept_loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(c,), daemon=True).start()
    threading.Thread(target=accept_loop, daemon=True).start()
    return srv



def test_fetch_bs_timeout_relogin_and_desync_guard():
    """P1:超时不被 baostock 吞掉(BsTimeout 是 BaseException);失败后重登换新 socket,下一只拿到自己的数据;
    不重登时,错位的迟到响应被代码校验拦下(BsDesyncError),而不是静默返回上一只的数据。"""
    import socket
    import baostock.common.context as ctx
    srv = _fake_bs_server({"sh.600000"}, delay=0.8)

    def connect():
        old = getattr(ctx, "default_socket", None)
        if old is not None:
            old.close()
        s = socket.socket(); s.connect(srv.getsockname())
        setattr(ctx, "default_socket", s); setattr(ctx, "user_id", "anonymous")

    try:
        # 修复后的默认路径:超时 -> 重登 -> 下一只股票数据正确
        connect()
        with pytest.raises(M5.BsTimeout):
            M5.fetch_bs_5m("600000.SH", "2026-09-01", "2026-09-01", timeout=0.3, on_fail=connect)
        time.sleep(1.0)                                   # 迟到的响应落到已关闭的旧连接上
        df = M5.fetch_bs_5m("000001.SZ", "2026-09-01", "2026-09-01", timeout=2, on_fail=connect)
        assert set(df["close"]) == {1.0} and len(df) == 2 and list(df.columns) == \
            ["open", "high", "low", "close", "volume", "amount"]
        df = M5.fetch_bs_5m("000002.SZ", "2026-09-01", "2026-09-01", timeout=2, on_fail=connect)
        assert set(df["close"]) == {2.0}
        # 不重登(旧行为的处境):迟到响应留在同一 socket 里,下一次请求读到它 -> 必须报错而不是返回 6.0
        connect()
        with pytest.raises(M5.BsTimeout):
            M5.fetch_bs_5m("600000.SH", "2026-09-01", "2026-09-01", timeout=0.3, on_fail=None)
        time.sleep(1.0)
        with pytest.raises(M5.BsDesyncError):
            M5.fetch_bs_5m("000001.SZ", "2026-09-01", "2026-09-01", timeout=2, on_fail=None)
        # 日期越界同样拦下
        connect()
        with pytest.raises(M5.BsDesyncError):
            M5.fetch_bs_5m("000001.SZ", "2026-09-02", "2026-09-03", timeout=2, on_fail=None)
    finally:
        s = getattr(ctx, "default_socket", None)
        if s is not None:
            s.close()
        setattr(ctx, "default_socket", None)
        srv.close()
