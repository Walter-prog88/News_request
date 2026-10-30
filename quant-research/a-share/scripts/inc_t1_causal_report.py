"""Append bounded replay evidence to T1 report and machine-readable handoff."""
from pathlib import Path
import json
import pandas as pd
from inc_t1_close_review import table, json_finite
ROOT=Path(__file__).resolve().parents[1]
L=ROOT/'research/ledger'
MARK='\n\n## Bounded account replay: engineering follow-up (D14)'


def append_results():
    CS=pd.read_csv(L/'inc_t1_causal_summary.csv')
    audit=pd.read_csv(L/'inc_t1_causal_journal_checks.csv')
    assert audit.passed.all()
    full=CS[CS.period=='2021-2025'][['X','window','ann_pp','pool_ann_pp','total_diff_vs_W1_pp','d_excess_vs_W1_pp','paired_excess_arith_pp','t_nw','cost_ann_pp','mean_exposure']]
    split=CS[CS.window.isin(['W5','W9'])][['X','window','period','d_excess_vs_W1_pp','paired_excess_arith_pp','t_nw']]
    section=MARK+'''

This follow-up completes the feasible bounded engineering requested after the earlier documentation closure. It supersedes any inference that the old model increment is a deployable gain. It does not relabel preregistered cells or rerun the entire grid. Scope: exclusion W1/W5/W9, three initial AUMs, five independent phases, plus identical-rule full-pool controls.

Orders are fixed at previous-close NAV and prices, generated every five days, active only in the selected window, and explicitly CANCELLED at that window's end. Remaining inventory stays invested; residual orders do not persist overnight and are only regenerated on the next five-day rebalance. There are no invented next-day fills, including repeated sealed limits. Buy orders at upper-limit minute closes and sell orders at lower-limit minute closes are blocked; opposite directions are permitted. The stock/order participation rate is locked at previous close: requested notional divided by the preceding 20-day median selected-window turnover, capped at 10%. Each completed bar uses this fixed rate times its observed turnover; no day's final turnover is used to set its rate. This approximates causal volume-weighted pacing rather than front-loading every order at 10%. Each stock's minute-side notional is <=10% of that minute's observed turnover. Buys use start-of-minute cash, with commission/spread/impact included; contemporaneous sells cannot finance them. Sells cannot exceed opening inventory (T+1). Cash cannot be borrowed (roundoff tolerance 1e-6 yuan).

The word causal refers to state/order recurrence: a bar consumes no subsequent bar. A completed minute CLOSE and its full observed turnover are still retrospective execution proxies, not proof of executable intrabar order placement, quote access or queue priority. Empirical quote/orderbook calibration is unavailable and remains an evidence gap.

Window impact scaling is FIXED1, the existing preregistered sensitivity; no same-year s_W is consumed. Sigma, ADV and half-spread use the production cost model's shift(1), at most trade_date-1 data. Future perturbation and truncated-prefix tests validate the actual cost model. Impact is charged as the incremental 1.5 power of each stock's daily cumulative filled notional, preventing artificial cost reduction by slicing into minutes. Commission and date-correct sell stamp tax debit cash.

Each of the five phases uses the FULL initial AUM independently; daily returns are averaged. This is conservative for impact versus a real product splitting capital five ways. Capital subsequently grows with NAV, unlike the old constant-AUM model. Inventory uses fractional adjusted-price units, without 100-share rounding, minimum commissions or actual corporate-action settlement. These are disclosed research/product approximations, not hidden sealed-limit fills.

Full-pool controls execute EXACTLY the same five-day phases, prices, directional limits, cash financing, POV capacity, lagged estimates, fixed1 costs and cancellation rule. `total_diff_vs_W1_pp` is the raw exclusion annual-return difference. `d_excess_vs_W1_pp` subtracts each window's own same-rule pool excess before comparing against W1; they are different objects and never added. `paired_excess_arith_pp` and `t_nw` use paired daily differences of these excess returns. All pp columns are annual percentage points.

'''+table(full)+'''

Development/test split:

'''+table(split)+'''

W5 paired same-rule excess is +0.602/+0.597/+0.554 pp/year arithmetically, with full NW t=6.94/6.88/6.25; both subperiods are positive. Its geometric excess difference is +0.689/+0.681/+0.627. W9 is +0.241/+0.232/+0.114 geometrically; 1B development excess is -0.031 and full t=1.205, so it is not a robust three-AUM recommendation. The larger W5 result than the old fixed-trade ledger is model/rule sensitivity, not an additional increment to stack: previous-close sizing, growing NAV, volume pacing and finite cancellation change inventory/cash paths. W5 modeled annual transaction cost is actually slightly higher than W1 at all AUMs; its gain is not a reduction in that modeled cost.

These are POST-REVIEW diagnostics, not new preregistered discoveries. The older W5/W9 mechanical pass labels remain historical evidence; bounded implementation conclusions must use the new table, especially same-rule excess and development/test consistency. No investment adoption, freeze change or new forward-test claim follows.

Artifacts: `inc_t1_causal_daily.csv` (phase cash/NAV/cost/cancellation), `inc_t1_causal_summary.csv`, `inc_t1_causal_checks.csv` (runtime capacity/cash/cancellation/cutoff), `inc_t1_causal_sources.csv` (actually read historical months), `inc_t1_causal_rules.json` (rule frozen before replay outcomes). `cache/_tmp/inc_t1_causal/orders_YYYYMM.parquet` contains per-stock requested/filled/cancelled units, notional, costs, blocked-minute counts and maximum actual minute participation. `inc_t1_causal_journal_checks.csv` independently reconstructs cash, inventory and close-valued NAV from these saved fills, rather than rerunning prices to produce another backtest. Terminal units are in `cache/_tmp/inc_t1_causal/terminal_inventory.parquet`.

The initial diagnostic was manually interrupted after completed month 202207: filling as fast as 10% POV allowed small W9 orders to finish near the open, rather than providing long-window volume-weighted pacing. The corrected replay locks the rate from lagged selected-window volume. The superseded account journals remain in `cache/_tmp/inc_t1_causal_superseded_frontload/`; the preserved log `logs/inc_t1_causal_superseded_frontload.log` ends with the manual KeyboardInterrupt. This was a rule correction, not a runtime assertion failure. `inc_t1_causal_run_manifest.json` records the change and earlier corrected startup/report-edit errors. Superseded outcomes are not used.

Directional execution, finite order lifecycle, cash/inventory feedback and same-year impact look-ahead are now implemented for this bounded replay. Quote/queue calibration and actual-share product constraints remain evidence gaps. The original five-year, sample-internal signal and survivor/adjustment limitations also remain. No 2026 inputs, commits or changes to other tracks.
'''
    p=ROOT/'research/INC_T1_REPORT.md';s=p.read_text().split(MARK)[0]
    s=s.replace('本轨道文档审查处置完成；真实方向性订单执行、持续封板现金/库存反馈、前视冲击系数替换和报价校准仍未完成。','旧全网格账本仅完成文档审查处置；方向性订单、有限生命周期现金/库存和事前系数随后已在本报告D14的三个窗口账户回放中实现，报价校准仍是证据缺口。')
    lead='\n> Latest bounded replay: W5 same-rule pool excess delta +0.689/+0.681/+0.627 pp/year; W9 +0.241/+0.232/+0.114. W9 at 1B fails development-sign consistency (dev -0.031; full paired t=1.205). These post-review account diagnostics supersede deployability inferences from the original mechanical tables. No investment adoption.\n'
    heading,body=s.split('\n',1)
    if '> Latest bounded replay:' not in s: s=heading+'\n'+lead+body
    p.write_text(s+section)
    f=L/'inc_t1_completion.json';result=json.loads(f.read_text())
    result.update(bounded_replay_complete=True,implementation_ready=False,
        verdict_mechanical_original=result['verdict'],
        verdict='Post-review account diagnostic: W5 positive at 3 AUMs; W9 weaker, 1B development negative and full t=1.205; no investment adoption',
        engineering_scope='W1/W5/W9; 3 initial AUMs; 5 full-AUM phases; same-rule pool; day-order cancellation',
        causal_full=CS[CS.period=='2021-2025'].to_dict('records'),pending_design=[],
        evidence_gaps=['minute-close/observed-turnover proxy lacks empirical quote and queue calibration',
                       'fractional adjusted units, no actual share/corporate-action/lots/minimum-commission replay'],
        adoption='none; post-review diagnostic only')
    f.write_text(json.dumps(json_finite(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':append_results()
