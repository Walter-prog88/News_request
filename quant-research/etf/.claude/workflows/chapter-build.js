// Reusable template for build+audit chapter pairs (factor factory /
// sequence models). Builder and auditor prompts are deliberately
// isolated; auditors must not import builder code.
export const meta = {
  name: 'chapter-build',
  description: 'Build the factor-factory and LSTM/Transformer chapters for the showcase report',
  phases: [
    { title: 'Build', detail: 'genetic factor factory and sequence models, in parallel' },
    { title: 'Audit', detail: 'independent recomputation of one cell / one metric each' },
  ],
}

const SHARED = `
CONTEXT — you are extending an existing, working research package. Read these
files BEFORE writing code; follow their conventions exactly.

Python interpreter (ONLY this one, it has torch+sklearn+pandas):
  /path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound/.venv/bin/python
Note: lightgbm/xgboost are INSTALLED BUT BROKEN (missing libomp). Use
sklearn.ensemble.HistGradientBoosting* and torch only.

Working dir: /path/to/user/Desktop/ETF-Research/index project/research/showcase_v1
Existing files to read first:
  build_matrix.py     how the 312-factor training matrix was built
  sweep_dial.py       load_env(), run_topk(), stats() — REUSE THESE, do not
                      reimplement the backtest
  finalize.py         how the showcase curves were assembled
  explainer_charts.py SVG chart conventions used by the report

Caches (session scratch, already populated):
  CACHE = /private/tmp/claude-501/-Users-walter-Desktop-index-project/0c89f974-a14e-4130-8a6f-3d35374216a4/scratchpad/showcase_cache
    X.npy         (495169, 312) float32, per-day cross-sectional rank in [0,1], NaN kept
    y_res.npy     residual next-period return (open t+1 -> open t+2, demeaned per day)
    y_rank.npy    within-day percentile rank of that return
    row_day.npy   int32 day index into manifest["dates"]
    row_sym.npy   int32 symbol index into manifest["symbols"]
    manifest.json {n_rows, n_features, feature_names, dates, symbols}
    dial_iter10.npy         (T,N) the chosen GBDT score panel
    insample/score_mlp.npy  (T,N) the MLP score panel
  ZOO = /private/tmp/claude-501/-Users-walter-Desktop-index-project/0c89f974-a14e-4130-8a6f-3d35374216a4/scratchpad/zoo_cache
    meta.json  {dates, symbols}   (2005 symbols, superset of the showcase pool)
    per-field (T,N) float32 arrays aligned to meta.json:
      ret_cc.npy  close.npy  continuous_open.npy  high_continuous.npy
      low_continuous.npy  overnight_ret_continuous.npy  ret_oc_continuous.npy
      open30_ret_continuous.npy  last30_ret.npy  rv_closepath_continuous.npy
      up_rv_continuous.npy  down_rv_continuous.npy
      high_time_ordinal_continuous.npy  zr_share_continuous.npy
      n_price_moves.npy  path_abs_return_continuous.npy  flat_bar_ratio.npy
      daily_ohlc_valid.npy  liq_gate_v2.npy
    daily_amount_all.parquet  flow_ratio_all.parquet  (index=dates, cols=symbols)

HOUSE RULES (non-negotiable):
- Signal uses information up to day t only; fills at t+1 continuous open;
  marking is open-to-open. sweep_dial.load_env() already returns O2O and the
  eligibility mask G with this convention — use them.
- Cost 0.5bp per executed side. Top-4 equal weight. Never change these.
- Full-sample fit is the accepted convention here (documented in the report as
  "upper-bound research口径"). Do NOT add cross-validation to the headline
  path; it is a separate disclosed layer already built.
- Write ONLY inside research/showcase_v1/. Never modify existing .py files that
  you were told to read — add new files.
- Comments in English, sparse, explaining WHY not WHAT. No emoji.
- Save results as JSON + parquet under research/showcase_v1/results/.
`

phase('Build')

const FACTORY = `${SHARED}

YOUR TASK — build the genetic-programming factor factory. New file:
research/showcase_v1/factor_factory.py

Design:
1. Operand set, read from ZOO arrays, restricted to the showcase pool
   (manifest["symbols"]) and aligned to manifest["dates"]:
   ret_cc, overnight_ret_continuous, ret_oc_continuous, open30_ret_continuous,
   last30_ret, rv_closepath_continuous, up_rv_continuous, down_rv_continuous,
   high_time_ordinal_continuous, zr_share_continuous, n_price_moves,
   path_abs_return_continuous, flat_bar_ratio, plus daily amount and flow ratio
   from the two parquets, plus close/high/low derived ratios.
2. Operator set on (T,N) panels, all strictly causal (no forward peeking):
   unary  : neg, abs, sign, log1p_abs, rank_cs (cross-sectional rank),
             zscore_cs, ts_mean_w, ts_std_w, ts_max_w, ts_min_w, ts_delta_w,
             ts_sum_w, delay_w, ts_rank_w  for w in {3,5,10,21,63}
   binary : add, sub, mul, div_safe, ts_corr_w, ts_cov_w  (w in {5,10,21,63})
3. Genetic loop: random formula trees (depth <= 3), population 400, 12
   generations, tournament selection, crossover + mutation, elitism 10%.
   Fitness = |mean daily cross-sectional rank IC| against y_res, computed on
   the SAME (day,sym) rows as X.npy, minus a complexity penalty
   (0.0002 * node_count) to prefer simple formulas.
   IMPORTANT: evaluate fitness on days BEFORE index 1400 only, and report each
   survivor's IC on the remaining days as a held-out column in the output
   table. This costs nothing and makes the chapter defensible.
4. Deduplicate survivors: drop any whose |Spearman corr| with an already-kept
   survivor's signal exceeds 0.7.
5. Target output: 250-400 kept formulas. Save:
   results/factory_factors.csv   (formula string, node_count, ic_dev, ic_held,
                                  ic_ir, kept flag)
   results/factory_funnel.json   {generated, evaluated, passed_ic, after_dedup}
   And save the surviving factor VALUES as a (n_kept, T, N) float32 npz at
   CACHE/factory_panel.npz with a names array — the report's retrain step needs
   it. Keep memory sane: evaluate in batches, store only survivors.
6. Then extend the training matrix: build X2 = concat(X, factory ranks) for the
   same rows, train the SAME HistGradientBoostingRegressor as finalize.py
   (learning_rate=0.05, max_leaf_nodes=31, max_iter=10, min_samples_leaf=3000,
   l2_regularization=1.0, seeds 0 and 1, full sample, y=y_res), panelize, and
   backtest with sweep_dial.run_topk. Save results/factory_curve.json with the
   window stats from sweep_dial.stats for 2019-2021 / 2022-2023 / 2024-2026 /
   full, plus the same for the 312-only baseline so the chapter can show the
   delta.

Print a compact JSON summary at the end. Runtime budget ~60-90 min; if the
genetic loop is too slow, reduce population before reducing generations.
RETURN: a JSON object with the funnel counts, the top 12 formulas with their
dev and held-out IC, and the factory_curve window stats vs baseline.`

const LSTM = `${SHARED}

YOUR TASK — build the sequence-model chapter. New file:
research/showcase_v1/seq_models.py

Design:
1. Input tensor per (day t, symbol i): the past 60 sessions x 20 channels.
   Channels (all causal, ending at day t):
     a) 5 return-structure : ret_cc, overnight_ret_continuous,
        ret_oc_continuous, open30_ret_continuous, last30_ret
     b) 4 volatility       : rv_closepath_continuous, up_rv_continuous,
        down_rv_continuous, log(high_continuous/low_continuous)
     c) 3 microstructure   : high_time_ordinal_continuous, zr_share_continuous,
        n_price_moves
     d) 3 liquidity/flow   : log-change of daily amount, flow ratio,
        cross-sectional rank of daily amount
     e) 5 cross-sectional  : same-day cross-sectional rank of ret_cc,
        rv_closepath_continuous, overnight_ret_continuous, flow ratio, and
        21-day momentum. These let the encoder see relative position over
        time, which a per-symbol sequence cannot.
   Per-symbol channels: standardize with a causal rolling 252-day mean/std
   (min_periods 60). Cross-sectional channels are already in [0,1].
   Fill remaining NaN with 0 after standardization (0.5 for rank channels).
   Build with numpy stride tricks / batched gather to avoid materializing
   495169 x 60 x 20 in RAM at once — generate batches on the fly from the
   (T,N,C) panel.
2. Models, both on the same tensor:
   LSTM : 2-layer LSTM hidden 64 -> additive temporal attention over the 60
          steps -> Linear head. MUST expose the attention weights.
   TFM  : 2-layer Transformer encoder, d_model 64, 4 heads, learned positional
          embedding, mean-pool -> Linear head. Same head size, same budget.
   Train on y_res, MSE loss, AdamW lr 1e-3 wd 1e-4, batch 4096, 8 epochs,
   2 seeds each, device "mps" if torch.backends.mps.is_available() else "cpu".
   Full-sample fit (house convention).
3. Panelize each model's score to (T,N), backtest with sweep_dial.run_topk,
   and report sweep_dial.stats for 2019-2021 / 2022-2023 / 2024-2026 / full.
4. Attention analysis (this is the chapter's centrepiece):
   - average attention weight by lag (0..59) over all samples -> a 60-vector
   - the same split by year, as a (n_years, 60) matrix for a heatmap
   - report which lags carry local maxima
   Save results/seq_attention.json with both, and
   results/seq_curves.json with the backtest stats for LSTM and TFM.
5. Render two inline SVG fragments into results/seq_charts.svg separated by a
   line containing exactly <!--SPLIT--> :
   chart A = attention-by-lag profile (line, x=lag 0..59, mark local maxima)
   chart B = attention heatmap by year (rows=years, cols=lag, colour ramp)
   Match the SVG conventions in explainer_charts.py: viewBox, CSS variables
   var(--up)/var(--gold)/var(--dim)/var(--ink)/var(--muted)/var(--hair), no
   external assets, no fixed pixel width on the svg element.

Print a compact JSON summary. Runtime budget ~90 min. If training is too slow,
cut seeds to 1 before cutting epochs.
RETURN: a JSON object with LSTM and Transformer window stats, the attention
local-maxima lags, and the attention-by-lag vector rounded to 4 decimals.`

const built = await parallel([
  () => agent(FACTORY, { label: 'build:factor-factory', phase: 'Build' }),
  () => agent(LSTM, { label: 'build:seq-models', phase: 'Build' }),
])

phase('Audit')

const AUDIT_SHARED = `${SHARED}

You are an INDEPENDENT AUDITOR. Do not trust the builder's code. Recompute the
target quantity yourself, from the raw sources, with your OWN implementation.
Do not import the builder's module. Do not modify anything under
research/showcase_v1 — read only, and write scratch files only under
/private/tmp/claude-501/-Users-walter-Desktop-index-project/0c89f974-a14e-4130-8a6f-3d35374216a4/scratchpad/.
Report the two numbers and their absolute difference. State AGREEMENT or
DISAGREEMENT explicitly. If you cannot locate an artefact, say so plainly
rather than inventing a number.`

const audits = await parallel([
  () => agent(`${AUDIT_SHARED}

AUDIT TARGET — the factor factory.
1. Read results/factory_factors.csv. Pick the top-ranked surviving formula.
2. Parse its formula string by hand and recompute the factor's value at one
   specific (date, symbol) cell of your choosing from the ZOO raw arrays,
   using your own operator implementations. Compare against the stored panel
   at CACHE/factory_panel.npz for the same cell.
3. Separately, recompute that formula's mean daily cross-sectional rank IC on
   the development rows (day index < 1400) and compare with the ic_dev column.
4. CRITICAL look-ahead check: verify the formula contains no operator that
   reads the future. Confirm by shifting the whole factor panel forward one day
   and checking the IC drops toward zero rather than rising.
RETURN a JSON object: {cell_recomputed, cell_stored, abs_diff, ic_recomputed,
ic_stored, lookahead_check_passed, verdict}.`,
    { label: 'audit:factor-factory', phase: 'Audit' }),
  () => agent(`${AUDIT_SHARED}

AUDIT TARGET — the sequence models.
1. Read results/seq_curves.json and results/seq_attention.json.
2. Independently rebuild the input tensor for ONE (date, symbol) sample: the
   60x20 window. Verify with your own code that every channel ends at day t and
   contains no value from day t+1 or later — this is the look-ahead check that
   matters most for a sequence model. Report the exact array indices you
   checked.
3. Verify the attention vector sums to approximately 1 per sample and that the
   reported local maxima are actually local maxima of the stored vector.
4. Re-run the backtest of the stored LSTM score panel yourself using
   sweep_dial.load_env + your own Top-4 loop (you may reuse load_env for the
   price data, but write your own selection and P&L loop), and compare the
   full-period Sharpe with the stored value.
RETURN a JSON object: {tensor_lookahead_ok, indices_checked, attention_sums_ok,
maxima_confirmed, sharpe_recomputed, sharpe_stored, abs_diff, verdict}.`,
    { label: 'audit:seq-models', phase: 'Audit' }),
])

return { built, audits }
