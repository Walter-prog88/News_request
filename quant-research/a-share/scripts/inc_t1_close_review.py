"""Close T1 documentation from existing ledgers; never loads prices or reruns other tracks."""
from pathlib import Path
import hashlib
import math
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / 'research/ledger'

def json_finite(value):
    """Map nonfinite numbers to JSON null throughout nested artifact payloads."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    return value


def table(d):
    def fmt(v):
        if isinstance(v, float):
            return f'{v:.3f}' if pd.notna(v) else '—'
        return str(v).replace('|', '\\|')
    return '| ' + ' | '.join(d.columns) + ' |\n|' + '|'.join(['---'] * len(d.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(map(fmt, r)) + ' |' for r in d.itertuples(index=False, name=None))


def main():
    S = pd.read_csv(L / 'inc_t1_summary.csv')
    G = pd.read_csv(L / 'inc_t1_layer1_grid.csv')
    R = pd.read_csv(L / 'inc_t1_seal_multiroll.csv')
    old = pd.read_csv(L / 'inc_t1_prereview_summary.csv')
    review = json.loads((L / 'inc_t1_original_review.json').read_text())
    dispositions = [
        ('R1','P0','采纳；收益伪影已修正，成交模型仍受限','两层既有封板修复复用；W7三档、W8三档均证据不足。单次顺延仍可能再次封板；rollN只是价格诊断，不能声称完整可成交模拟。'),
        ('R2','P1','采纳；已修复并回归验证','第二层每行按ref_mode同价格口径扣减；VWAP对W1 VWAP，延迟TWAP对W1 TWAP；旧TWAP差保留为对照。'),
        ('R3','P2','采纳；撤销整体覆盖逐格判据','Top20通过格逐一保留；策略净超额为负和池子漂移风险只影响部署判断，不改执行增量机械判定。'),
        ('R4','P2','采纳；已修正文案','第一层main相对W1 VWAP；不称为TWAP切换到VWAP的执行alpha。'),
        ('R5','P2','采纳；明确未解决的测量局限','成交bar价含买卖方向组成、可能与半价差重复；同年s_W含前视；s_one是消融，不是实际报价/冲击校准。')]
    D = pd.DataFrame(dispositions, columns=['id','severity','disposition','evidence_and_limit'])
    D['original_title'] = [x['title'] for x in review['findings']]
    D.to_csv(L / 'inc_t1_review.csv',index=False)
    e = S[(S.strategy == 'COMP_v5_exclude') & S.scheme.isin(['W5','W7','W8','W9'])].copy()
    e['资金'] = e.X.map({1e7:'1000万',1e8:'1亿',1e9:'10亿'})
    cols=['资金','scheme','dC_full','dC_dev','dC_test','t_full','d_realized_A','t_realized_A','d_model_B','l2_total_diff_full','l2_d_excess_full','verdict_mechanical','verdict']
    comparison=e.merge(old[['strategy','X','scheme','dC_full','verdict']],on=['strategy','X','scheme'],suffixes=('','_old'))
    comparison.to_csv(L / 'inc_t1_review_changed_results.csv',index=False)
    robust=G[(G.strategy=='COMP_v5_exclude') & (G.period=='2021-2025') & G.scheme.isin(['W5','W9']) & G.variant.isin(['s_one','Y0.5','Y1.5','twap'])][['X','scheme','variant','d_credible_pp']]
    passed=S[(S.strategy!='COMP_v5_exclude') & (S.verdict=='有增量')][['strategy','X','scheme','dC_full','t_full','d_realized_A','t_realized_A','d_px_mkt_excluded','verdict']]
    roll=R[(R.role=='full') & R.window.isin(['W5','W7','W8','W9'])][['X','window','mode','px_rel_saving_pp','t_nw']]
    report='''# INC-T1 执行优化：审查后修订

本报告替代旧主报告。预注册 `INC_T1_PREREG.md` 原文不变；旧报告保留在 `INC_T1_REPORT_prereview.md`，旧结论不可再引用为当前结论。只用已落盘的2021–2025计算（B×GBT为2023–2025）；未读取2026数据，未联网、未提交、未改其他轨道。

排雷的 W5（13:01–14:00）和 W9（全日VWAP）在修正后仍满足预注册机械判据。W7降为三档资金均证据不足，W8三档均证据不足。T1是样本内执行增量证据，尚未完成真实可成交验证；不能直接修改现行冻结执行方案或部署。Top20逐格通过结果保留，不能用“策略本身不赚钱”覆盖它们。

## 修订结果与基线

所有数值为年化百分点。第一层 ΔC 是恒定AUM算术节省，main相对同口径W1 VWAP（09:31–09:40），不是相对W1 TWAP。第二层是收盘终点计价后的几何年化收益差，与第一层不能相加；Δ同窗超额再扣各自同窗同频池基准。VWAP方案对W1 VWAP；单价W0/W8亦对W1 VWAP；D1 TWAP对W1 TWAP。

'''+table(e[cols])+'''

W5可供综合使用的两层点估计区间为1000万 +0.375～+0.381、1亿 +0.481～+0.490、10亿 +0.707～+0.778 pp/年；这是不同估计对象的区间，不是置信区间或可承诺收益。W9相应为 +0.410～+0.525、+0.485～+0.611、+0.621～+0.675。W9更少受日内局部封板影响，但仍依赖成本与基准假设。

W5原第一层 +0.465/+0.565/+0.850 已下降到 +0.381/+0.481/+0.778；原第二层 +0.45/+0.56/+0.78 混入价格口径差，现为 +0.375/+0.490/+0.707。第二层总差现 +0.490/+0.605/+0.821，仅作组合总收益参考。旧“+5.28→+5.73”是TWAP与VWAP混比，不再使用。旧口径列 `*_vs_w1twap` 与当前 `*_vs_ref` 并存，二者差才是同族W1的口径差，不能加入任何增量。

## 封板修复与未关闭的模型限制

原审查的例子002222.SZ、2021-12-23在W5整窗涨停，买入不能按10% POV视为成交。修复第一层main采用方向性dir：封涨停挡买、封跌停挡卖，整腿顺延次日同窗；tight和any并列敏感性。第二层 `_seal` 采用any，双向禁成交，继承引擎约定，不能称为方向性可成交验证。旧无后缀族保留用于前后对照。

10亿W7的单次顺延机械结果仍是“有增量”（A=+0.184、t=2.074），不是原审查声称的所有规模都自动失败。多次顺延的已实现价格诊断降到+0.060、t=0.681，故修订判断降为证据不足；`verdict_mechanical`保留机械结果，`verdict`记录审查后判断。1000万W7也只有+0.059、t=0.662；W8检验窗多次顺延为负。该诊断不含价差、冲击、POV截量、持仓反馈，只有1000万和10亿，不能冒充第二次完整回测或把数值插入原ΔC。

'''+table(roll)+'''

第一层只顺延一次，次日仍封板也可能按其价格计成本；缺价及无量顺延也不是无限重试。rollN最多检查5次延迟，仍封板或无法定价的残余按零项处理，并非真实订单取消现金账。因此P0的错误收益主张已处置，但持续封板订单的完整执行模型未完成。W5的rollN已实现节省约+0.315/+0.316（1000万/10亿），t=3.88/3.95，方向未消失；W9约+0.375/+0.373、t=6.56/6.62。不能把修复后的模型点估计称为完全可实现收入。

## 冲击假设与报价测量

s_W用同年池均值波动/成交额比，含全年未来信息，只是事后成本分解；逐股票日缩放也用当日实现量。Y未校准，POV用当天实现成交额。要形成可部署方案，需冻结过去窗口估计的系数、真实订单执行规则，并以报价与订单回放校准。这里保留预注册消融，未事后优化参数。

'''+table(robust)+'''

W5去窗口冲击缩放（s_one）仅 +0.335/+0.337/+0.381，10亿main约一半来自模型项，不能把+0.778全称为模型无关收益。成交价来自分钟bar成交价近似，不是买卖盘中间价；开盘调出股可能多在买价成交，已实现路径含交易方向组成，另加半价差可能重复计量。没有报价数据，无法分离数bp的组成效应；外部5分钟成交量核对只验证VWAP近似，不验证买入排队或半价差。

## Top20逐格结果与部署判断

'''+table(passed)+'''

以上是当前修复账本的通过格，保留其执行增量判定。买卖分离没有第二层表达，池子级漂移另列且不纳入判据；真实净收益必须计入漂移与现金暴露。Top20策略自身扣成本超额为负只影响投资/部署判断。撤回原报告未经登记的“总体证据不足”覆盖逐格结论，其他格以summary为准；不能将本轨道增量当成Top20策略转正证明。

## 原审查逐条处置

原文从review代理ad3fa5ac884a784c8.jsonl的StructuredOutput提取，保存为 `inc_t1_original_review.json`；原输出JSON按 `d['result']` 解码后T1的fix为null，说明中断而非完成。

'''+table(D[['id','severity','disposition','evidence_and_limit']])+'''

## 复用、偏离与验证

复用a9e5bc3c7316ffd08已跑完的封板腿、引擎模式、summarize、seal诊断及三档资金账本，没有重跑大规模收益计算。此次只刷新判定、图和报告，并补同基线逐行代数、roll1诊断对账、审查降级回归测试。旧窗口聚合37项、旧/新28模式引擎断言、成交清单门槛仍从原账本验证。

偏离登记：原报告§7记录全部保留于prereview档案；新增D08封板规则修复（dir第一层/any第二层），D09第二层同口径基线修正，D10撤销Top20整体覆盖逐格判据，D11基线名称修正，D12披露同年冲击及成交价组成，D13连续封板价格诊断对10亿W7增加事后保守降级。D13不回写预注册、不冒称新预注册研究；诊断未进入原机械判据。

完整逐格：`inc_t1_summary.csv`；旧新主窗口对照：`inc_t1_review_changed_results.csv`；复查意见：`inc_t1_review.csv`；机器可读综合交接：`inc_t1_completion.json`。图 `reports/inc_t1_layer1_single.png`、`inc_t1_layer2.png`、`inc_t1_paths.png` 已从修复账本重新生成，只展示模型结果。

本轨道文档审查处置完成；真实方向性订单执行、持续封板现金/库存反馈、前视冲击系数替换和报价校准仍未完成。所有结果在历史样本内，五个年份并非独立前向验证；长历史换手率版本无分钟数据，不能直接外推。现行冻结方案保持不变。
'''
    (ROOT/'research/INC_T1_REPORT.md').write_text(report)
    result=dict(track='T1',review_documentation_closed=True,implementation_ready=False,
                verdict='排雷W5/W9有历史模型增量；W7/W8证据不足；Top20保留逐格通过判定',
                W5_layer1_pp=[.3806,.4810,.7777],W5_layer2_same_convention_excess_pp=[.3748,.4901,.7073],
                W9_layer1_pp=[.4096,.4846,.6747],W9_layer2_same_convention_excess_pp=[.5254,.6110,.6207],
                aum=[1e7,1e8,1e9],pending_design=['持续封板的完整订单/现金/库存回放','第二层方向性封板','事前冲击缩放与报价校准'],
                prereg_sha256=hashlib.sha256((ROOT/'research/INC_T1_PREREG.md').read_bytes()).hexdigest())
    (L/'inc_t1_completion.json').write_text(json.dumps(json_finite(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if (L/'inc_t1_causal_journal_checks.csv').exists():
        from inc_t1_causal_report import append_results
        append_results()

if __name__=='__main__':main()
