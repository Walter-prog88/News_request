"""XSB2 取证与前向对账工具(标准库,无三方依赖)。

数据:research/xsb2_forensics/trades.csv(主播真实买入,持续追加新行即可)
     research/xsb2_forensics/bars_cache.json(腾讯 qfq 日线缓存,新代码自动补抓)
用法:python3 scripts/run_xsb2_forensics.py            # 全表:入场画像 + 前向收益 + 出场规则复盘
     python3 scripts/run_xsb2_forensics.py --refresh  # 强制刷新全部行情到最新

三张表:
  1. 入场画像:回撤/前期涨幅/年线/量比/当日涨幅 —— 对照 XSB2_PROTOCOL §0 的门
  2. 前向收益:买入日收盘 → 最新收盘,途中最大/最深
  3. 出场规则复盘:−20% 止损 / 前高不放量止盈 / 60 日强平(收盘近似,与买入持有对照)
背景与结论:RESEARCH_LOG §15;协议 research/XSB2_PROTOCOL.md。
"""
import json, csv, os, sys, time, urllib.request, statistics as st
from datetime import date

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FDIR = os.path.join(ROOT, "research", "xsb2_forensics")
CACHE_F = os.path.join(FDIR, "bars_cache.json")
REFRESH = "--refresh" in sys.argv


def load_trades():
    with open(os.path.join(FDIR, "trades.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch(sym, start="2022-01-01"):
    out = {}
    end = date.today().isoformat()
    mid = "2024-06-30"
    for a, b in [(start, mid), ("2024-07-01", end)]:
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
               f"param={sym},day,{a},{b},640,qfq")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        d = json.load(urllib.request.urlopen(req, timeout=15))["data"][sym]
        for bar in (d.get("qfqday") or d.get("day") or []):
            out[bar[0]] = bar
        time.sleep(0.15)
    return sorted(out.values(), key=lambda x: x[0])


def ensure_bars(trades):
    cache = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}
    today = date.today().isoformat()
    for t in trades:
        sym = t["exch"] + t["code"]
        stale = REFRESH or sym not in cache or (
            cache[sym] and cache[sym][-1][0] < today and t["buy_date"] >= "2026-08-01")
        if sym not in cache or stale:
            try:
                cache[sym] = fetch(sym)
                print(f"fetched {sym} {t['name']} -> {cache[sym][-1][0]}")
            except Exception as e:
                print(f"FAIL {sym} {t['name']}: {e}")
    json.dump(cache, open(CACHE_F, "w"))
    return cache


def entry_metrics(bars, bd):
    dates = [b[0] for b in bars]
    c = [float(b[2]) for b in bars]
    v = [float(b[5]) for b in bars]
    if bd not in dates:
        nxt = [x for x in dates if x >= bd]
        if not nxt:
            return None, bd
        bd = nxt[0]
    i = dates.index(bd)
    if i < 260:
        return None, bd
    hi252 = max(c[i - 251:i + 1])
    lo504 = min(c[max(0, i - 503):i + 1])
    ihi = max(range(i - 251, i + 1), key=lambda k: c[k])
    m250 = sum(c[i - 249:i + 1]) / 250
    m250p = sum(c[i - 269:i - 19]) / 250 if i >= 269 else m250
    adv20 = sum(v[i - 20:i]) / 20
    return dict(pb=c[i] / hi252 - 1, dsh=i - ihi, rally=hi252 / lo504 - 1,
                vs250=c[i] / m250 - 1, ma250_up=m250 > m250p,
                volr=v[i] / adv20 if adv20 else float("nan"),
                ret_d=c[i] / c[i - 1] - 1), bd


def replay(bars, bd):
    """出场规则复盘(收盘近似):sl20 / 前高不放量 / 60日强平,否则持有。"""
    dates = [b[0] for b in bars]
    c = [float(b[2]) for b in bars]
    v = [float(b[5]) for b in bars]
    i = dates.index(bd)
    e = c[i]
    for k in range(i + 1, len(c)):
        hi252 = max(c[max(0, k - 251):k + 1])
        adv20 = sum(v[k - 20:k]) / 20
        if c[k] <= e * 0.80:
            return c[k] / e - 1, "sl20", k - i
        if c[k] >= 0.95 * hi252 and v[k] <= adv20:
            return c[k] / e - 1, "tp_hi", k - i
        if k - i >= 60:
            return c[k] / e - 1, "time", k - i
    return c[-1] / e - 1, "持有", len(c) - 1 - i


def main():
    trades = load_trades()
    cache = ensure_bars(trades)
    print(f"\n{'名称':<6}{'买入日':<11}{'类':<2}{'回撤%':>7}{'前涨%':>7}{'vs年线%':>7}"
          f"{'年线↑':>4}{'量比':>6}{'当日%':>6} | {'持有%':>7}{'最大%':>7}{'最深%':>7}"
          f" | {'规则%':>7}{'原因':<5}")
    hold_all, rule_all = [], []
    for t in trades:
        sym = t["exch"] + t["code"]
        bars = cache.get(sym)
        if not bars:
            continue
        m, bd = entry_metrics(bars, t["buy_date"])
        if m is None:
            continue
        dates = [b[0] for b in bars]
        c = [float(b[2]) for b in bars]
        i = dates.index(bd)
        e = c[i]
        fwd = c[i + 1:]
        hold = c[-1] / e - 1
        mx = (max(fwd) / e - 1) if fwd else 0.0
        mn = (min(fwd) / e - 1) if fwd else 0.0
        rr, why, held = replay(bars, bd)
        hold_all.append(hold)
        rule_all.append(rr)
        print(f"{t['name']:<6}{bd:<11}{t['user_cluster']:<2}{m['pb']*100:7.1f}"
              f"{m['rally']*100:7.0f}{m['vs250']*100:7.1f}{str(m['ma250_up'])[0]:>4}"
              f"{m['volr']:6.2f}{m['ret_d']*100:6.1f} | {hold*100:7.1f}{mx*100:7.1f}"
              f"{mn*100:7.1f} | {rr*100:7.1f}{why:<5}")
    n = len(hold_all)
    print(f"\n{n} 笔  买入持有: 中位 {st.median(hold_all)*100:+.1f}% 平均 "
          f"{st.mean(hold_all)*100:+.1f}% 胜 {sum(1 for x in hold_all if x > 0)}/{n}")
    print(f"{'':6}套用出场: 中位 {st.median(rule_all)*100:+.1f}% 平均 "
          f"{st.mean(rule_all)*100:+.1f}% 胜 {sum(1 for x in rule_all if x > 0)}/{n}")


if __name__ == "__main__":
    main()
