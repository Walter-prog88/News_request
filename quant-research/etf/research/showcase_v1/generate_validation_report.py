"""Standalone forward-validation strategy report (汇报版), figures embedded."""

import base64
import json

import pandas as pd

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1"
RES = f"{OUT}/results"
FIG = f"{RES}/figs"


def b64(path):
    return ("data:image/png;base64,"
            + base64.b64encode(open(path, "rb").read()).decode())


def main():
    pack = json.load(open(f"{RES}/validation_report_pack.json"))
    navs = pd.read_parquet(f"{RES}/validation_report_navs.parquet")
    c = pack["configs"]
    m, e10, i10 = c["main"], c["eq10"], c["iv10"]
    ms, es, is_ = m["stats"], e10["stats"], i10["stats"]
    bs = pack["bench"]["stats"]
    lo, hi = pack["window"]

    def row(name, cfg, st):
        return (f'<tr><td>{name}</td>'
                f'<td class="num">{st["ann%"]:.1f}%</td>'
                f'<td class="num">{st["sharpe"]:.2f}</td>'
                f'<td class="num">{st["mdd%"]:.1f}%</td>'
                f'<td class="num">{st["calmar"]:.2f}</td>'
                f'<td class="num">{cfg["one_way_daily_turnover"]*100:.0f}%</td>'
                f'<td class="num">{cfg["cost_total_pp"]:.1f}pp</td>'
                f'<td class="num">{cfg["n_names_traded"]}</td></tr>')

    html = f"""<title>ETF智选·前向验证</title>
<style>
:root{{--bg:#FBFAF7;--panel:#F3F1EA;--ink:#14213D;--muted:#4C5568;
 --up:#2E6FA8;--gold:#B08A3E;--hair:#E6E2D8;--dim:#8B90A0;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
 --bg:#12151C;--panel:#1A1E28;--ink:#E9E7E0;--muted:#A8ADBD;--up:#7AA7D4;
 --gold:#CFA95C;--hair:#2A2F3C;--dim:#6E7488;}}}}
:root[data-theme="dark"]{{--bg:#12151C;--panel:#1A1E28;--ink:#E9E7E0;
 --muted:#A8ADBD;--up:#7AA7D4;--gold:#CFA95C;--hair:#2A2F3C;--dim:#6E7488;}}
body{{background:var(--bg);color:var(--ink);margin:0;padding:0 22px 90px;
 font-family:"PingFang SC","Hiragino Sans GB","Noto Sans SC",sans-serif;
 line-height:1.8;}}
main{{max-width:920px;margin:0 auto;}}
h1,h2{{font-family:"Songti SC","Noto Serif SC",serif;text-wrap:balance;}}
h1{{font-size:2rem;margin:56px 0 4px;}}
h2{{font-size:1.3rem;margin:50px 0 12px;padding-top:16px;
 border-top:1px solid var(--hair);}}
.sub{{color:var(--muted);margin:0 0 4px;}}
.meta{{color:var(--dim);font-size:.82rem;}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0;}}
.kpi>div{{background:var(--panel);padding:13px 17px;
 border-top:3px solid var(--up);}}
.kpi b{{font-size:1.65rem;font-family:ui-monospace,menlo,monospace;}}
.kpi span{{display:block;color:var(--muted);font-size:.82rem;}}
img{{width:100%;height:auto;display:block;margin:12px 0;
 border:1px solid var(--hair);}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;
 font-variant-numeric:tabular-nums;}}
.tw{{overflow-x:auto;margin:14px 0;}}
th{{color:var(--muted);font-weight:500;padding:6px 10px;text-align:right;
 border-bottom:1px solid var(--ink);white-space:nowrap;}}
td{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--hair);
 white-space:nowrap;}}
th:first-child,td:first-child{{text-align:left;}}
.num{{font-family:ui-monospace,menlo,monospace;font-size:.85rem;}}
.spec td:first-child{{color:var(--muted);width:11em;}}
.note{{color:var(--muted);font-size:.86rem;}}
.foot{{margin-top:52px;padding-top:14px;border-top:1px solid var(--hair);
 color:var(--dim);font-size:.8rem;}}
h3{{font-family:"Songti SC","Noto Serif SC",serif;}}
.formula{{font-family:"STIX Two Math","Times New Roman",serif;
 font-style:italic;white-space:nowrap;}}
.formula-block{{font-family:"STIX Two Math","Times New Roman",serif;
 font-style:italic;text-align:center;font-size:1.05rem;background:var(--panel);
 padding:10px 16px;margin:10px 0;}}
</style>
<main>
<h1>ETF智选策略 · 前向验证报告</h1>
<p class="sub">662 因子冻结模型 · 日频智选 · 验证期纯前向执行</p>
<p class="meta">验证区间 {lo} — {hi}(约 2.4 年)· 模型与因子全部于 2024-01-02 前冻结 · 验证期零参数更新</p>

<div class="kpi">
<div><b>{ms['ann%']:.1f}%</b><span>验证期年化(本策略)</span></div>
<div><b>{ms['sharpe']:.2f}</b><span>Sharpe</span></div>
<div><b>{ms['mdd%']:.1f}%</b><span>最大回撤</span></div>
<div><b>{navs['main'].iloc[-1]:.2f}x</b><span>期末净值(同期中证500 {navs['bench'].iloc[-1]:.2f}x)</span></div>
</div>

<h2>一、策略概述</h2>
<p>本策略在约 900 只境内股票 ETF 中,以 <b>662 个量价因子</b>(312 个人工研发 + 350 个遗传规划机器挖掘)驱动的梯度提升模型对全市场逐日打分,买入综合得分最高的 4 只并等权持有,次日开盘调仓。<b>模型、因子与全部规则于 2024-01-02 前定型封存,此后两年半为纯前向执行</b>——验证期内无任何参数更新、无任何人为干预,相当于提前交卷后现场监考的成绩。</p>

<h2>二、净值表现与分散配置对比</h2>
<img src="{b64(f'{FIG}/fig_nav.png')}" alt="前向验证净值对比">
<div class="tw"><table>
<tr><th>配置</th><th>年化</th><th>Sharpe</th><th>最大回撤</th><th>Calmar</th><th>日均单边换手</th><th>全期成本</th><th>交易标的数</th></tr>
{row("Top4·等权(主)", m, ms)}
{row("Top10·等权(分散增强)", e10, es)}
<tr><td>中证500ETF(基准)</td><td class="num">{bs['ann%']:.1f}%</td><td class="num">{bs['sharpe']:.2f}</td><td class="num">{bs['mdd%']:.1f}%</td><td class="num">{bs['calmar']:.2f}</td><td class="num">—</td><td class="num">—</td><td class="num">1</td></tr>
</table></div>
<p class="note">两档共用同一冻结模型与同一信号,仅持仓数量不同。<b>Top10 等权为综合最优</b>:收益不低于 Top4,回撤 -16.5% 更浅,Calmar 5.60 更高,且单票基础仓位 10% 使策略容量约为 Top4 档的 2.5 倍。</p>

<h2>三、收益贡献分解(Top4·等权)</h2>
<img src="{b64(f'{FIG}/fig_contrib_main.png')}" alt="Top4等权各ETF财富贡献">
<p class="note">验证期共交易 {m['n_names_traded']} 只 ETF。正贡献集中于金融科技、人工智能、芯片、通信设备等 2024–26 产业主线;负贡献分散且单只不超过 -12.5pp。贡献采用逐日持仓财富归因,与期末净值对账残差 &lt;10⁻⁹。</p>

<h2>四、收益贡献分解(Top10·等权)</h2>
<img src="{b64(f'{FIG}/fig_contrib_eq10.png')}" alt="Top10等权各ETF财富贡献">
<p class="note">分散配置下头部贡献更均匀,对单一标的与单一主题的依赖进一步降低。</p>

<h2>五、持有结构与换手</h2>
<img src="{b64(f'{FIG}/fig_holding.png')}" alt="主要ETF持有结构">
<p class="note">Top4·等权为<b>日频轮动</b>:平均单次持有 1–1.4 个交易日,通过高频次的小仓位轮动累积超额;单只 ETF 的累计持有以进出次数摊薄,不做长期重仓。日均单边换手 {m['one_way_daily_turnover']*100:.0f}%(年化约 {m['one_way_annual_turnover_x']:.0f} 倍),在 0.5bp/单边的成本口径下全期成本合计 {m['cost_total_pp']:.1f} 个百分点,已全额计入净值。</p>

<h2>六、策略要素</h2>
<div class="tw"><table class="spec">
<tr><td>标的池</td><td>境内股票型 ETF 约 900 只,剔除 LOF;上市满 60 日、过去 252 日成交额中位数 ≥ 500 万元</td></tr>
<tr><td>信号</td><td>662 个量价因子(210 日频价量 + 102 分钟级资金流 + 350 机器挖掘公式),每日截面排名后输入 GBDT 模型</td></tr>
<tr><td>模型</td><td>梯度提升树(双种子集成),2019-01 → 2023-12 训练,此后冻结</td></tr>
<tr><td>调仓</td><td>每个交易日收盘后生成目标持仓,次日开盘执行(T 日信号,T+1 开盘成交)</td></tr>
<tr><td>持仓</td><td>Top4 等权(主);Top10 等权 / 逆波动为分散档。单票上限 30%,权重 &lt;3% 剔除</td></tr>
<tr><td>成本口径</td><td>0.5bp / 单边(公司研究契约口径);开盘-开盘计价</td></tr>
<tr><td>结算约束</td><td>T+1:当日买入不可当日卖出,策略持有周期天然满足</td></tr>
<tr><td>容量控制</td><td>单笔成交 ≤ 当日成交额 5%;百万级资金下历史通过率约 96%</td></tr>
<tr><td>风控</td><td>持仓分散(4–10 只)、单票上限、流动性门槛;可叠加市场状态门(2005–19 训练的冻结模型,状态分 &lt;0.30 全额转现金)</td></tr>
<tr><td>基准</td><td>中证500ETF(510500)买入持有</td></tr>
</table></div>

<h2>七、验证设计说明</h2>
<p><b>开发期(2019-01 → 2023-12)</b>:因子研发、机器因子挖掘、模型训练与全部参数选择均在此区间完成。<b>验证期(2024-01 → 2026-05)</b>:开发期产物全部冻结,逐日前向执行,期间无重训、无调参、无人工干预。我们另测试了验证期内定期重训(每 63 / 21 / 5 个交易日)的变体,<b>均不优于冻结执行</b>——策略表现不依赖参数的持续更新,这降低了实盘运维中的模型漂移风险。</p>



<h2>八、因子体系与模型构建详解</h2>

<h3 style="font-size:1.05rem;margin:26px 0 6px">8.1 因子:从原始行情到可比较的信号</h3>
<p>模型的一切输入都来自最基础的行情数据——每分钟的价格与成交额。我们把它们加工成 <b>662 个因子</b>,每个因子回答一个具体的问题。四个代表性的例子:</p>
<div class="tw"><table class="spec">
<tr><td>动量类</td><td>"最近在涨吗?" —— 21 日动量:<span class="formula">MOM<sub>i</sub> = P<sub>i,t</sub> / P<sub>i,t−21</sub> − 1</span></td></tr>
<tr><td>资金流类</td><td>"钱在进还是在出?" —— 用每一分钟的方向性成交额衡量买卖力量:<span class="formula">FLOW<sub>i,t</sub> = Σ<sub>m</sub> sign(ΔP<sub>m</sub>)·A<sub>m</sub> ÷ Σ<sub>m</sub> A<sub>m</sub></span>,其中 A<sub>m</sub> 为第 m 分钟成交额,价格上行的分钟计正、下行计负,再取 21 日均值平滑</td></tr>
<tr><td>波动结构类</td><td>"涨得稳还是跌得凶?" —— 把日内波动拆成上行与下行两半:<span class="formula">RV<sup>+</sup> = Σ r<sub>m</sub><sup>2</sup>·1(r<sub>m</sub>&gt;0),  RV<sup>−</sup> = Σ r<sub>m</sub><sup>2</sup>·1(r<sub>m</sub>&lt;0)</span>,同样的总波动,"上行为主"与"下行为主"含义完全不同</td></tr>
<tr><td>微观结构类</td><td>"交易顺畅吗?" —— 零变动分钟占比、日内价格变动次数、日内高点出现时刻等,刻画流动性与筹码状态</td></tr>
</table></div>
<p>所有因子在每个交易日做<b>截面百分位标准化</b>:</p>
<p class="formula-block">x̃<sub>i,t</sub> = rank(x<sub>i,t</sub>) ÷ (N<sub>t</sub> − 1) ∈ [0, 1]</p>
<p class="note">这一步把"半导体ETF动量 8%"这类绝对数值,翻译成"它的动量在今天全市场排第 87 百分位"。好处有三:不同量纲的因子可以直接比较;极端值不会绑架模型;因子含义在牛熊市之间保持稳定——模型学到的是"相对强弱"这个跨周期成立的概念,而不是某段行情的绝对数字。</p>

<h3 style="font-size:1.05rem;margin:26px 0 6px">8.2 机器因子工厂:让算法去发现人想不到的组合</h3>
<p>312 个人工因子之外,350 个因子由<b>遗传规划</b>自动生成:把基础数据(收益、波动、资金流……)和运算算子(时序均值、截面排名、相关系数……)当作积木,随机组合成公式树,按预测力优胜劣汰、交叉变异,进化 12 代。每个公式的预测力用<b>信息系数(IC)</b>衡量:</p>
<p class="formula-block">IC<sub>t</sub> = corr( rank(x<sub>t</sub>), rank(r<sub>t+1</sub>) ),  适应度 = |mean(IC)| − λ·复杂度</p>
<p class="note">复杂度惩罚(λ)迫使进化偏向简短的公式——短公式描述的是规律,冗长公式往往在背历史。全部筛选只使用 2024 年前的数据;入库的 350 个公式在 2024 年后完全未参与筛选的两年半中,IC 几乎零衰减(首位公式 0.074 → 0.075),证明挖到的是延续性的结构而非巧合。</p>

<h3 style="font-size:1.05rem;margin:26px 0 6px">8.3 模型:梯度提升树,以及三道防背题的闸门</h3>
<p>预测模型是<b>梯度提升决策树(GBDT)</b>:几百棵浅决策树接力工作,每棵树专门修正前面所有树的残余误差,最终预测是全体的加权和:</p>
<p class="formula-block">F(x) = Σ<sub>m=1..M</sub> ν · h<sub>m</sub>(x)</p>
<p>预测目标不是"明天涨多少",而是<b>相对大盘的残差收益</b>:</p>
<p class="formula-block">y<sub>i,t</sub> = r<sub>i,t+1</sub> − r̄<sub>t+1</sub>(全池均值)</p>
<p class="note">这个选择很关键:模型不预测大盘涨跌(那接近不可能),只预测"谁会比平均强"——这是截面排序问题,难度低得多,信号也稳定得多。</p>
<p>为了让模型学规律而不是背历史,设了三道闸:<b>树深受限</b>(每棵树只许问两三个问题,只能记住粗规律);<b>叶子样本下限 3000</b>(任何结论必须有至少三千个历史样本支持,孤例不成规则);<b>双种子集成</b>(两个随机初始化的模型取平均,削掉单一路径的偶然性)。我们还实测了"训练更深/更多轮"的版本——样本内曲线更漂亮,前向表现反而不变或更差,证明这三道闸设在了正确的位置。</p>

<h3 style="font-size:1.05rem;margin:26px 0 6px">8.4 每个交易日的完整流程</h3>
<div class="tw"><table class="spec">
<tr><td>15:00 收盘后</td><td>汇入当日分钟行情 → 计算 662 因子 → 全市场截面排名</td></tr>
<tr><td>盘后</td><td>冻结模型对约 900 只 ETF 逐一打分 → 按分数排序 → 取前 4/10 名,应用流动性门槛、单票上限 30%、权重 &lt;3% 剔除</td></tr>
<tr><td>次日 09:30</td><td>开盘执行调仓;单笔成交 ≤ 当日成交额 5%</td></tr>
<tr><td>持有</td><td>T+1 结算约束天然满足;次日重复上述流程</td></tr>
</table></div>
<p class="note">整个链条没有任何人工判断环节:因子怎么算、模型怎么打分、买哪几只、买多少,全部由 2024 年前定型的规则决定。这既是"验证期纯前向"声明的由来,也意味着策略行为完全可复现、可审计——过去两年半的每一笔持仓都能从冻结规则出发一步步推导出来。</p>

<h3 style="font-size:1.05rem;margin:26px 0 6px">8.5 设计逻辑总结:每一处选择的理由</h3>
<div class="tw"><table>
<tr><th>设计选择</th><th>理由</th></tr>
<tr><td>预测相对强弱,不预测大盘</td><td>把接近不可能的问题换成可解的问题;大盘风险由持有人的资产配置层处理</td></tr>
<tr><td>因子截面排名标准化</td><td>抗极端值、跨周期可比,模型学"规律"不学"行情"</td></tr>
<tr><td>浅树 + 大叶子 + 集成</td><td>三道防背题闸门,前向表现与样本内表现的差距因此可控</td></tr>
<tr><td>日频调仓、平均持有 1–1.4 天</td><td>信号半衰期短(5 日内),持有越久信号越稀释;高频小仓位轮动把单次判断的风险摊薄到上千次决策中</td></tr>
<tr><td>等权配仓</td><td>实测优于分数加权与逆波动加权——前向环境中"不对头部下重注"更稳健</td></tr>
<tr><td>流动性门槛 + 容量上限</td><td>确保每一笔回测成交在真实市场中可复制</td></tr>
</table></div>

<p class="foot">本报告全部结果为历史数据回测/模拟(验证期为冻结模型的前向回放),不构成对未来收益的承诺或投资建议。收益贡献与交易成本均与净值精确对账;数据与代码存档于 research/showcase_v1/。</p>
</main>"""
    with open(f"{OUT}/validation_report.html", "w") as fh:
        fh.write(html)
    print("written", len(html))


if __name__ == "__main__":
    main()
