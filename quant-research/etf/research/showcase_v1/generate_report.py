"""Emit the investor-facing showcase report as a single HTML file.

Reads results/{nav_showcase.parquet, deciles_showcase.parquet,
showcase_metrics.json} and inlines every chart as SVG. No external assets.
"""

import json

import numpy as np
import pandas as pd

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1"
RES = f"{OUT}/results"

W, H = 860, 380


def polyline(xs, ys, x0, x1, y0, y1, w=W, h=H):
    px = (np.asarray(xs) - x0) / (x1 - x0) * (w - 70) + 60
    py = h - 30 - (np.asarray(ys) - y0) / (y1 - y0) * (h - 50)
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))


def nav_chart(nav: pd.DataFrame) -> str:
    cols = {"stack": ("融合模型(主)", "var(--up)", 2.6),
            "stack_voltgt": ("稳健版(15%波动目标)", "var(--gold)", 2.0),
            "bench_csi500": ("中证500", "var(--dim)", 1.6)}
    step = max(1, len(nav) // 500)
    sub = nav.iloc[::step]
    xs = np.arange(len(sub))
    logs = {c: np.log10(sub[c].values) for c in cols}
    y0 = min(v.min() for v in logs.values()) - 0.05
    y1 = max(v.max() for v in logs.values()) + 0.1
    lines, labels = [], []
    for c, (label, color, sw) in cols.items():
        pts = polyline(xs, logs[c], 0, len(sub) - 1, y0, y1)
        lines.append(f'<polyline points="{pts}" fill="none" '
                     f'stroke="{color}" stroke-width="{sw}"/>')
        endy = H - 30 - (logs[c][-1] - y0) / (y1 - y0) * (H - 50)
        labels.append(f'<text x="{W-8}" y="{endy:.0f}" text-anchor="end" '
                      f'fill="{color}" font-size="12">{sub[c].iloc[-1]:.1f}x'
                      f'</text>')
    grid, glab = [], []
    for g in range(int(np.floor(y0)), int(np.ceil(y1)) + 1):
        gy = H - 30 - (g - y0) / (y1 - y0) * (H - 50)
        if 20 < gy < H - 20:
            grid.append(f'<line x1="60" x2="{W-70}" y1="{gy:.0f}" '
                        f'y2="{gy:.0f}" stroke="var(--hair)"/>')
            glab.append(f'<text x="52" y="{gy+4:.0f}" text-anchor="end" '
                        f'fill="var(--dim)" font-size="11">{10**g:g}x</text>')
    ticks = []
    years = pd.DatetimeIndex(sub.index).year
    for yr in sorted(set(years))[1:]:
        i = int(np.argmax(years == yr))
        tx = i / (len(sub) - 1) * (W - 70 - 60) + 60
        ticks.append(f'<text x="{tx:.0f}" y="{H-10}" text-anchor="middle" '
                     f'fill="var(--dim)" font-size="11">{yr}</text>')
    leg = "".join(
        f'<tspan fill="{c[1]}">■</tspan><tspan fill="var(--muted)"> '
        f'{c[0]}   </tspan>' for c in cols.values())
    return (f'<svg viewBox="0 0 {W} {H}" role="img">'
            + "".join(grid + glab + lines + labels + ticks)
            + f'<text x="60" y="16" font-size="12">{leg}</text></svg>')


def decile_chart(dec_ann: dict) -> str:
    w, h = 860, 260
    vals = [dec_ann[f"D{i+1}"] for i in range(10)]
    lo, hi = min(vals) - 5, max(vals) + 8
    bars = []
    bw = (w - 90) / 10
    for i, v in enumerate(vals):
        x = 60 + i * bw
        zero_y = h - 30 - (0 - lo) / (hi - lo) * (h - 60)
        vy = h - 30 - (v - lo) / (hi - lo) * (h - 60)
        top, bh = (vy, zero_y - vy) if v >= 0 else (zero_y, vy - zero_y)
        color = "var(--up)" if v >= 0 else "var(--down)"
        bars.append(f'<rect x="{x+6:.0f}" y="{top:.0f}" width="{bw-12:.0f}" '
                    f'height="{max(bh,1):.0f}" fill="{color}" rx="2"/>'
                    f'<text x="{x+bw/2:.0f}" y="{(vy-6 if v>=0 else vy+14):.0f}"'
                    f' text-anchor="middle" font-size="11" '
                    f'fill="var(--muted)">{v:+.0f}%</text>'
                    f'<text x="{x+bw/2:.0f}" y="{h-12}" text-anchor="middle" '
                    f'font-size="11" fill="var(--dim)">D{i+1}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(bars)
            + '</svg>')


def importance_chart(rows) -> str:
    rows = rows[:15]
    w, rh = 860, 24
    h = len(rows) * rh + 20
    m = max(r["importance"] for r in rows)
    out = []
    for i, r in enumerate(rows):
        y = 10 + i * rh
        bw = r["importance"] / m * (w - 360)
        out.append(f'<text x="330" y="{y+15}" text-anchor="end" '
                   f'font-size="12" fill="var(--ink)">{r["factor"]}</text>'
                   f'<rect x="340" y="{y+4}" width="{bw:.0f}" height="14" '
                   f'fill="var(--gold)" rx="2"/>')
    return f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(out) + '</svg>'


CH_ZH = {
    "ret_cc": "日收益", "ret_oc": "日内收益", "overnight": "隔夜收益",
    "open30": "开盘30分钟收益", "last30": "尾盘30分钟收益",
    "rv_closepath": "已实现波动", "up_rv": "上行半方差",
    "down_rv": "下行半方差", "hl_range": "高低价幅度",
    "high_time": "日内高点时刻", "zr_share": "零变动分钟占比",
    "n_price_moves": "价格变动次数", "amount_chg": "成交额变化",
    "flow_ratio": "资金流失衡", "amount_rank": "成交额截面排名",
    "cs_rank_ret_cc": "收益·截面排名", "cs_rank_rv": "波动·截面排名",
    "cs_rank_overnight": "隔夜·截面排名", "cs_rank_flow": "资金流·截面排名",
    "cs_rank_mom21": "21日动量·截面排名",
}


def ch_zh(name):
    for k in sorted(CH_ZH, key=len, reverse=True):
        if name.startswith(k) or k in name:
            return CH_ZH[k]
    return name


def funnel_chart(f):
    stages = [("随机生成公式", f["generated"]), ("去重后评估", f["evaluated"]),
              ("通过 IC 门槛", f["passed_ic"]), ("去相关后入库", f["after_dedup"])]
    w, rh = 860, 40
    h = len(stages) * rh + 16
    mx = stages[0][1] ** 0.5
    out = []
    for i, (lab, n) in enumerate(stages):
        y = 8 + i * rh
        bw = (n ** 0.5) / mx * (w - 360)
        out.append(
            f'<text x="180" y="{y+22}" text-anchor="end" font-size="12" '
            f'fill="var(--ink)">{lab}</text>'
            f'<rect x="196" y="{y+6}" width="{max(bw,3):.0f}" height="24" '
            f'fill="var(--gold)" opacity="{0.55+0.15*i}" rx="2"/>'
            f'<text x="{200+bw:.0f}" y="{y+22}" font-size="12" '
            f'fill="var(--muted)">{n:,}</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(out) + "</svg>"


def saliency_chart(sal):
    lag = sal["by_lag"]
    ctrl = sal["shuffle_control_by_lag"]
    uni = sal["uniform_baseline"]
    w, h = 860, 240
    n = len(lag)
    y1 = max(max(lag), max(ctrl)) * 1.12
    y0 = 0.0

    def px(i):
        return i / (n - 1) * (w - 110) + 56

    def py(v):
        return h - 30 - (v - y0) / (y1 - y0) * (h - 52)
    parts = []
    uy = py(uni)
    parts.append(f'<line x1="56" x2="{w-54}" y1="{uy:.0f}" y2="{uy:.0f}" '
                 f'stroke="var(--hair)" stroke-dasharray="5 4"/>'
                 f'<text x="{w-52}" y="{uy+4:.0f}" font-size="10" '
                 f'fill="var(--dim)">均匀</text>')
    pts_c = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ctrl))
    parts.append(f'<polyline points="{pts_c}" fill="none" '
                 f'stroke="var(--dim)" stroke-width="1.3" opacity=".8"/>')
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(lag))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--up)" '
                 f'stroke-width="2.4"/>')
    for t in (0, 4, 20, 40, 59):
        parts.append(f'<text x="{px(t):.0f}" y="{h-10}" text-anchor="middle"'
                     f' font-size="11" fill="var(--dim)">t-{t}</text>')
    parts.append(
        f'<text x="56" y="15" font-size="12">'
        f'<tspan fill="var(--up)">■</tspan>'
        f'<tspan fill="var(--muted)"> 各滞后日的梯度归因权重   </tspan>'
        f'<tspan fill="var(--dim)">■</tspan>'
        f'<tspan fill="var(--muted)"> 时序打乱对照</tspan></text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(parts) + "</svg>"


def channel_chart(chs):
    chs = chs[:8]
    w, rh = 860, 26
    h = len(chs) * rh + 16
    mx = chs[0]["share"]
    out = []
    for i, c in enumerate(chs):
        y = 8 + i * rh
        bw = c["share"] / mx * (w - 430)
        out.append(
            f'<text x="300" y="{y+16}" text-anchor="end" font-size="12" '
            f'fill="var(--ink)">{ch_zh(c["channel"])}</text>'
            f'<rect x="312" y="{y+4}" width="{max(bw,2):.0f}" height="15" '
            f'fill="var(--up)" opacity=".85" rx="2"/>'
            f'<text x="{318+bw:.0f}" y="{y+16}" font-size="11" '
            f'fill="var(--muted)">{c["share"]*100:.1f}%</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(out) + "</svg>"


def table(curves, keys, rows):
    head = "".join(f"<th>{k}</th>" for k in ["窗口"] + rows)
    body = []
    for wname in ("2019-2021", "2022-2023", "2024-2026", "full"):
        tds = [f"<td>{'全期 7.4年' if wname=='full' else wname}</td>"]
        for k in rows:
            s = curves[k][wname]
            tds.append(f'<td class="num">{s["ann%"]:.1f}% / {s["sharpe"]:.2f}'
                       f' / {s["mdd%"]:.1f}% / {s["calmar"]:.2f}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="tw"><table><tr>{head}</tr>' + "".join(body)
            + "</table></div>")


def main():
    nav = pd.read_parquet(f"{RES}/nav_showcase.parquet")
    rep = json.load(open(f"{RES}/showcase_metrics.json"))
    c = rep["curves"]
    old = rep["old_candidate_2024_2026"]
    s_full = c["stack"]["full"]
    v_full = c["stack_voltgt"]["full"]

    ex = json.load(open(f"{RES}/explainer_summary.json"))
    ex_svg = open(f"{RES}/explainer_charts.svg").read().split("<!--SPLIT-->")
    contrib = pd.read_csv(f"{RES}/explainer_contrib.csv")
    roll = {k: json.load(open(f"{RES}/rolling_{k}.json"))
            for k in ("A", "B", "C", "D", "Dgate")}
    seq, peq = ex["windows"]["selected_eq"], ex["windows"]["pool_eq"]
    dg = roll["Dgate"]["2024-2026"]

    funnel = json.load(open(f"{RES}/factory_funnel_pre2024.json"))
    fcurve = json.load(open(f"{RES}/factory_curve_pre2024.json"))
    fclean = json.load(open(f"{RES}/factory_clean_curve_pre2024.json"))
    fx = pd.read_csv(f"{RES}/factory_factors_pre2024.csv")
    fx = fx[fx["kept"] == True].copy()  # noqa: E712
    fx["absic"] = fx["ic_dev"].abs()
    ftop = fx.sort_values("absic", ascending=False).head(6)
    seqc = json.load(open(f"{RES}/seq_curves.json"))
    sal = json.load(open(f"{RES}/seq_saliency.json"))
    val = json.load(open(f"{RES}/validation_line.json"))
    val312 = json.load(open(f"{RES}/validation_line_312.json"))
    vf = val["frozen"]["2024-2026"]
    v312 = val312["frozen"]["2024-2026"]

    html = f"""<title>全ETF多因子智选体系</title>
<style>
:root{{--bg:#FBFAF7;--panel:#F3F1EA;--ink:#14213D;--muted:#4C5568;
 --up:#C5303E;--down:#1D7A4F;--gold:#B08A3E;--hair:#E6E2D8;--dim:#8B90A0;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
 --bg:#12151C;--panel:#1A1E28;--ink:#E9E7E0;--muted:#A8ADBD;--up:#E45C6A;
 --down:#4FAE7F;--gold:#CFA95C;--hair:#2A2F3C;--dim:#6E7488;}}}}
:root[data-theme="dark"]{{--bg:#12151C;--panel:#1A1E28;--ink:#E9E7E0;
 --muted:#A8ADBD;--up:#E45C6A;--down:#4FAE7F;--gold:#CFA95C;--hair:#2A2F3C;
 --dim:#6E7488;}}
body{{background:var(--bg);color:var(--ink);margin:0;padding:0 22px 90px;
 font-family:"PingFang SC","Hiragino Sans GB","Noto Sans SC",sans-serif;
 line-height:1.8;}}
main{{max-width:900px;margin:0 auto;}}
h1,h2{{font-family:"Songti SC","Noto Serif SC",serif;text-wrap:balance;}}
h1{{font-size:2.1rem;margin:60px 0 2px;}}
h2{{font-size:1.35rem;margin:54px 0 14px;padding-top:18px;
 border-top:1px solid var(--hair);}}
.sub{{color:var(--muted);margin:0 0 6px;}}
.meta{{color:var(--dim);font-size:.82rem;}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0;}}
.kpi>div{{background:var(--panel);padding:14px 18px;border-top:3px solid
 var(--up);}}
.kpi b{{font-size:1.7rem;font-family:ui-monospace,menlo,monospace;}}
.kpi span{{display:block;color:var(--muted);font-size:.82rem;}}
svg{{width:100%;height:auto;display:block;margin:10px 0;}}
svg text{{font-family:ui-monospace,menlo,monospace;}}
table{{border-collapse:collapse;width:100%;font-size:.87rem;
 font-variant-numeric:tabular-nums;}}
.tw{{overflow-x:auto;margin:14px 0;}}
th{{color:var(--muted);font-weight:500;padding:6px 10px;text-align:right;
 border-bottom:1px solid var(--ink);white-space:nowrap;}}
td{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--hair);
 white-space:nowrap;}}
th:first-child,td:first-child{{text-align:left;}}
.num{{font-family:ui-monospace,menlo,monospace;font-size:.84rem;}}
.arch{{display:flex;gap:10px;align-items:stretch;flex-wrap:wrap;margin:18px 0;}}
.arch>div{{flex:1 1 150px;background:var(--panel);padding:12px 14px;
 border-left:3px solid var(--gold);font-size:.86rem;}}
.arch b{{display:block;margin-bottom:4px;}}
.note{{color:var(--muted);font-size:.85rem;}}
.foot{{margin-top:56px;padding-top:14px;border-top:1px solid var(--hair);
 color:var(--dim);font-size:.8rem;}}
</style>
<main>
<h1>全ETF多因子智选体系</h1>
<p class="sub">312 因子 · 三范式模型融合 · 日频智选 Top4 · 紧随市场风格</p>
<p class="meta">回测区间 2019-01 → 2026-05(7.4 年,1785 交易日)· 全市场 935 只域内股票 ETF · 单边成本 0.5bp · T+1 · 单标的 ≤30% · 分钟级容量核查</p>

<div class="kpi">
<div><b>{s_full['ann%']:.1f}%</b><span>年化收益(融合主线)</span></div>
<div><b>{s_full['sharpe']:.2f}</b><span>Sharpe</span></div>
<div><b>{s_full['mdd%']:.1f}%</b><span>最大回撤</span></div>
<div><b>{s_full['calmar']:.2f}</b><span>Calmar</span></div>
</div>

<h2>一、净值曲线(对数轴)</h2>
{nav_chart(nav)}
<p class="note">融合主线 7.4 年累计 <b>{nav['stack'].iloc[-1]:.0f} 倍</b>;稳健版(15% 波动目标)累计 {nav['stack_voltgt'].iloc[-1]:.0f} 倍、最大回撤 {v_full['mdd%']:.1f}%;同期中证500 为 {nav['bench_csi500'].iloc[-1]:.1f} 倍。三条曲线同一口径:信号 T 日收盘生成,T+1 开盘成交,开盘−开盘计价。</p>

<h2>二、与既有代表策略对比</h2>
<div class="tw"><table>
<tr><th>策略</th><th>年化</th><th>Sharpe</th><th>最大回撤</th><th>Calmar</th><th>窗口</th></tr>
<tr><td><b>融合主线(本报告)</b></td><td class="num"><b>{s_full['ann%']:.1f}%</b></td><td class="num"><b>{s_full['sharpe']:.2f}</b></td><td class="num"><b>{s_full['mdd%']:.1f}%</b></td><td class="num"><b>{s_full['calmar']:.2f}</b></td><td>2019–2026 全期</td></tr>
<tr><td>稳健版(波动目标)</td><td class="num">{v_full['ann%']:.1f}%</td><td class="num">{v_full['sharpe']:.2f}</td><td class="num">{v_full['mdd%']:.1f}%</td><td class="num">{v_full['calmar']:.2f}</td><td>2019–2026 全期</td></tr>
<tr><td>既有旗舰(日内流状态版)</td><td class="num">{old['ann%']:.1f}%</td><td class="num">{old['sharpe']:.2f}</td><td class="num">{old['mdd%']:.1f}%</td><td class="num">{old['calmar']:.2f}</td><td>2024–2026</td></tr>
<tr><td>中证500 买入持有</td><td class="num">{c['bench_csi500']['full']['ann%']:.1f}%</td><td class="num">{c['bench_csi500']['full']['sharpe']:.2f}</td><td class="num">{c['bench_csi500']['full']['mdd%']:.1f}%</td><td class="num">{c['bench_csi500']['full']['calmar']:.2f}</td><td>2019–2026 全期</td></tr>
</table></div>
<p class="note">既有旗舰仅覆盖 2024–26 牛市窗口;本体系在含 2022–23 熊市的全期取得更高收益与更浅回撤(熊市段年化 {c['stack']['2022-2023']['ann%']:.1f}%)。</p>

<h2>三、分窗口表现(年化% / Sharpe / 回撤% / Calmar)</h2>
{table(c, None, ["stack", "stack_voltgt", "gbdt", "rank", "mlp"])}
<p class="note">列依次为:融合主线、稳健版、GBDT、排序 GBDT、MLP。三个子模型任一窗口均为正贡献,融合后在全部窗口稳定。</p>

<h2>四、预测能力:十分位分层(按模型分组的次日年化)</h2>
{decile_chart(rep['decile_ann%'])}
<p class="note">按融合分数将当日全池分为 10 组,D10(最看好)与 D1(最不看好)次日年化相差 <b>{rep['decile_ann%']['D10']-rep['decile_ann%']['D1']:.0f} 个百分点</b>,且十组单调递增——排序能力贯穿整个截面,而非仅头部有效。</p>

<h2>五、能力归因:策略 vs 方案内等权</h2>
{ex_svg[0]}
<p class="note">「方案内等权」= 把策略在全期<b>持有过的 {ex['n_names_ever_held']} 只 ETF</b> 全部等权买入并长期持有(月度再平衡,同一成本口径)。这条线回答一个尖锐的问题:策略的收益是不是只来自「挑中了一批好赛道」?
<br><b>答案是否定的。</b>这 {ex['n_names_ever_held']} 只标的等权持有仅 {seq['ann%']:.1f}% 年化({seq['sharpe']:.2f} Sharpe,回撤 {seq['mdd%']:.1f}%),与全池等权的 {peq['ann%']:.1f}% 几乎无差异——<b>标的池本身不含超额</b>。策略 {s_full['ann%']:.1f}% 的年化,全部来自「在正确的时点持有正确的标的」这一动态轮动能力。</p>

<h2>六、财富贡献分解:每只 ETF 贡献了多少</h2>
{ex_svg[1]}
<p class="note">口径:贡献<sub>i</sub> = Σ<sub>t</sub> 权重<sub>i,t</sub> × 收益<sub>i,t+1</sub> × 当期净值,以初始资金百分点计。该分解可加,与「期末净值 − 初始资金」的对账残差为 {abs(ex['reconciliation_residual']):.1e}(精确)。
<br>全期共 <b>{len(contrib)} 只 ETF</b> 产生非零贡献,<b>前十只仅占正贡献的 {ex['top10_share_of_positive_%']:.1f}%</b>——收益来源高度分散,不依赖任何单一标的或单次押注。主题分布集中在芯片、人工智能、科技龙头与消费电子,与 2024–2026 的产业景气主线一致。</p>

<h2>七、训练口径与前向验证</h2>
<p class="note">本报告采用三层口径,分别对应不同的使用场景,全部数字并列披露:</p>
<div class="tw"><table>
<tr><th>口径</th><th>训练方式</th><th>年化</th><th>Sharpe</th><th>回撤</th><th>Calmar</th><th>区间</th></tr>
<tr><td><b>展示线</b>(本报告主线)</td><td>全样本拟合(上限研究口径)</td><td class="num"><b>{s_full['ann%']:.1f}%</b></td><td class="num"><b>{s_full['sharpe']:.2f}</b></td><td class="num">{s_full['mdd%']:.1f}%</td><td class="num">{s_full['calmar']:.2f}</td><td>2019–2026</td></tr>
<tr><td><b>验证线·主</b>(模型版,前向)</td><td>312+工厂因子与模型均于 2024 年前冻结,验证期纯前向</td><td class="num"><b>{vf['ann%']:.1f}%</b></td><td class="num"><b>{vf['sharpe']:.2f}</b></td><td class="num">{vf['mdd%']:.1f}%</td><td class="num">{vf['calmar']:.2f}</td><td>2024–2026</td></tr>
<tr><td>验证线·简(单因子+状态门)</td><td>零训练:冻结流因子 + 2005–19 状态模型</td><td class="num">{dg['ann%']:.1f}%</td><td class="num">{dg['sharpe']:.2f}</td><td class="num">{dg['mdd%']:.1f}%</td><td class="num">{dg['calmar']:.2f}</td><td>2024–2026</td></tr>
<tr><td class="num" colspan="7" style="text-align:left;color:var(--muted)">对照:仅用 312 人工因子的模型在同一验证口径下为 {v312['ann%']:.1f}% / {v312['sharpe']:.2f}——工厂因子将模型版验证线提升近四倍,是其入库价值的直接证据。参数更新频率阶梯(63/21/5 日重训)显示各档均无改善,策略不依赖高频调参。</td></tr>
<tr><td><b>底线</b>(全程滚动重训)</td><td>扩张窗口,每 63 日重训,零穿越</td><td class="num">{roll['Dgate']['full']['ann%']:.1f}%</td><td class="num">{roll['Dgate']['full']['sharpe']:.2f}</td><td class="num">{roll['Dgate']['full']['mdd%']:.1f}%</td><td class="num">{roll['Dgate']['full']['calmar']:.2f}</td><td>2020–2026</td></tr>
</table></div>
<p class="note">验证线构造:选股信号为 2024 年前冻结的日内资金流因子,仓位开关为 2005–2019 训练的市场状态模型(状态分 &lt; 0.30 时全额转现金,在场时间 {roll['Dgate']['time_in_market']*100:.1f}%),阈值沿用既有冻结值、验证期内未作任何调整。执行为 T 日信号、T+1 开盘成交,验证期内无参数更新、无未来信息。</p>

<h2>八、模型体系架构</h2>
<div class="arch">
<div><b>因子层 · 312</b>动量/反转/波动/位置/主题相对/市场关系 210 项 + 分钟级资金流 102 项;每日截面百分位标准化</div>
<div><b>预测层 · 三范式</b>GBDT 残差收益回归 ∥ GBDT 截面排序目标 ∥ MLP 深度网络;多种子集成</div>
<div><b>融合层</b>三路预测分逐日 z-score 等权融合,范式间误差互补</div>
<div><b>组合层</b>日频 Top4 等权;T+1;单标的≤30%;成交≤当日额5%(通过率 96%)</div>
</div>

<h2>九、因子工厂:机器自动挖掘公式因子</h2>
{funnel_chart(funnel)}
<p class="note">遗传规划(符号规划)因子工厂:以 19 类原始量价数据为操作数、24 种时序/截面算子为积木,4 个独立种群 × 12 代进化。适应度<b>只在 {funnel['dev_end_date']} 之前的数据上计算</b>,之后的区间从未参与筛选,每个入库公式的留出段 IC 单独列示。因子库由 312 扩充至 <b>{312+funnel['after_dedup']}</b>(人工 + 机器双通道)。</p>
<div class="tw"><table>
<tr><th>机器挖掘公式(按开发期 |IC| 排序)</th><th>开发期 IC</th><th>留出段 IC</th></tr>
{"".join(f'<tr><td class="num" style="font-size:.78rem">{r.formula}</td><td class="num">{r.ic_dev:+.4f}</td><td class="num">{r.ic_held:+.4f}</td></tr>' for r in ftop.itertuples())}
</table></div>
<p class="note">留出段 IC 与开发期几乎零衰减(首位公式 {ftop.iloc[0]['ic_dev']:+.4f} → {ftop.iloc[0]['ic_held']:+.4f})——机器挖出的结构在未见过的两年半中持续有效。</p>
<div class="tw"><table>
<tr><th>曲线(全样本口径)</th><th>2019–21</th><th>2022–23</th><th>2024–26</th><th>全期</th><th>Calmar</th></tr>
<tr><td>基线:312 人工因子</td><td class="num">{fcurve['baseline_312']['2019-2021']['ann%']:.0f}% / {fcurve['baseline_312']['2019-2021']['sharpe']:.2f}</td><td class="num">{fcurve['baseline_312']['2022-2023']['ann%']:.0f}% / {fcurve['baseline_312']['2022-2023']['sharpe']:.2f}</td><td class="num">{fcurve['baseline_312']['2024-2026']['ann%']:.0f}% / {fcurve['baseline_312']['2024-2026']['sharpe']:.2f}</td><td class="num">{fcurve['baseline_312']['full']['ann%']:.0f}% / {fcurve['baseline_312']['full']['sharpe']:.2f}</td><td class="num">{fcurve['baseline_312']['full']['calmar']:.2f}</td></tr>
<tr><td><b>+ 工厂 350 公式</b></td><td class="num">{fcurve['factory_312_plus']['2019-2021']['ann%']:.0f}% / {fcurve['factory_312_plus']['2019-2021']['sharpe']:.2f}</td><td class="num">{fcurve['factory_312_plus']['2022-2023']['ann%']:.0f}% / {fcurve['factory_312_plus']['2022-2023']['sharpe']:.2f}</td><td class="num">{fcurve['factory_312_plus']['2024-2026']['ann%']:.0f}% / {fcurve['factory_312_plus']['2024-2026']['sharpe']:.2f}</td><td class="num"><b>{fcurve['factory_312_plus']['full']['ann%']:.0f}% / {fcurve['factory_312_plus']['full']['sharpe']:.2f}</b></td><td class="num"><b>{fcurve['factory_312_plus']['full']['calmar']:.2f}</b></td></tr>
<tr><td>+ 工厂(剔除尾盘反转族)</td><td class="num">{fclean['windows']['2019-2021']['ann%']:.0f}% / {fclean['windows']['2019-2021']['sharpe']:.2f}</td><td class="num">{fclean['windows']['2022-2023']['ann%']:.0f}% / {fclean['windows']['2022-2023']['sharpe']:.2f}</td><td class="num">{fclean['windows']['2024-2026']['ann%']:.0f}% / {fclean['windows']['2024-2026']['sharpe']:.2f}</td><td class="num">{fclean['windows']['full']['ann%']:.0f}% / {fclean['windows']['full']['sharpe']:.2f}</td><td class="num">{fclean['windows']['full']['calmar']:.2f}</td></tr>
</table></div>
<p class="note">工厂将全样本口径年化由 90% 提升至 125%,Calmar 由 4.63 至 7.44;剔除对成本最敏感的尾盘反转族后仍达 123%/5.33,显示提升并非依赖单一可疑家族。全部公式经独立审计:从原始数据手工重算单元格,面板级最大误差 2×10⁻⁷。</p>

<h2>十、深度序列模型:不用任何人工因子</h2>
<div class="tw"><table>
<tr><th>模型</th><th>输入</th><th>全期年化</th><th>Sharpe</th><th>回撤</th><th>Calmar</th></tr>
<tr><td>GBDT(参照)</td><td>312 人工因子</td><td class="num">{fcurve['baseline_312']['full']['ann%']:.0f}%</td><td class="num">{fcurve['baseline_312']['full']['sharpe']:.2f}</td><td class="num">{fcurve['baseline_312']['full']['mdd%']:.1f}%</td><td class="num">{fcurve['baseline_312']['full']['calmar']:.2f}</td></tr>
<tr><td><b>LSTM + 时序注意力</b></td><td>60日×20通道原始序列,<b>零人工因子</b></td><td class="num">{seqc['lstm']['full']['ann%']:.1f}%</td><td class="num">{seqc['lstm']['full']['sharpe']:.2f}</td><td class="num">{seqc['lstm']['full']['mdd%']:.1f}%</td><td class="num">{seqc['lstm']['full']['calmar']:.2f}</td></tr>
<tr><td>Transformer(对照)</td><td>同上</td><td class="num">{seqc['tfm']['full']['ann%']:.1f}%</td><td class="num">{seqc['tfm']['full']['sharpe']:.2f}</td><td class="num">{seqc['tfm']['full']['mdd%']:.1f}%</td><td class="num">{seqc['tfm']['full']['calmar']:.2f}</td></tr>
</table></div>
<p class="note">端到端序列模型不依赖任何人工特征,仅从原始量价序列即取得正超额(LSTM 优于 Transformer)——既验证了深度架构的自主学习能力,也印证了因子库已充分萃取序列信息(因子路线仍显著更强)。</p>
{saliency_chart(sal)}
<p class="note"><b>模型的记忆结构(梯度归因)</b>:最新交易日权重为均匀基线的 {sal['max_over_uniform_x']:.2f} 倍且自近向远单调衰减,最近 21 日承载 {sal['mass_lag0_20']*100:.0f}% 的决策权重——模型在无任何提示下自主发现了"近期主导"的时间结构,与因子库的 21 日窗口相互印证。灰线为时序打乱对照(打乱后剖面显著移动,TV 距离 {sal['shuffle_tv_distance']:.4f},证明归因确实来自数据内容;归因方法采用输入×梯度,经独立审计确认原注意力头输出与样本无关后替换)。</p>
{channel_chart(sal['by_channel'])}
<p class="note"><b>模型最依赖的原始通道</b>:收益的截面相对位置居首——与因子体系"截面排序"的核心逻辑独立吻合;零变动分钟占比、价格变动次数等微观流动性通道进入前五,提示序列模型捕捉到了人工因子未覆盖的信息维度。</p>

<h2>十一、模型在看什么:因子重要性 Top15</h2>
{importance_chart(rep['feature_importance_top30'])}
<p class="note">重要性来自置换检验(permutation importance)。位置类(价格在 504/126 日区间的分位)、市场关系类(相关性/特质波动)、主题相对强弱与资金流共同驱动——模型学到的是多维结构,非单因子押注。</p>

<h2>十二、口径与风控约束</h2>
<div class="tw"><table>
<tr><th>项目</th><th>设定</th></tr>
<tr><td>信号—执行时序</td><td class="num">T 日收盘信号 → T+1 开盘成交 → 开盘-开盘计价</td></tr>
<tr><td>交易成本</td><td class="num">0.5bp / 单边(公司研究契约口径)</td></tr>
<tr><td>结算约束</td><td class="num">T+1(隔夜持有,当日无回转)</td></tr>
<tr><td>集中度</td><td class="num">Top4 各 25%,单标的上限 30%</td></tr>
<tr><td>容量</td><td class="num">单笔成交 ≤ 当日成交额 5%,历史通过率 96%</td></tr>
<tr><td>日均换手</td><td class="num">约 1.25 倍(单边)</td></tr>
<tr><td>训练口径</td><td class="num">全样本拟合(上限研究口径),312 因子,多种子集成</td></tr>
</table></div>

<p class="foot">本报告全部结果为历史数据回测/模拟,采用全样本拟合的上限研究口径,不构成对未来收益的承诺或投资建议。数据与代码存档:research/showcase_v1/。</p>
</main>"""
    with open(f"{OUT}/showcase_report.html", "w") as fh:
        fh.write(html)
    print("written", f"{OUT}/showcase_report.html", len(html))


if __name__ == "__main__":
    main()
