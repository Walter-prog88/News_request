// Reusable: axis list may be passed via `args` (array of axis names);
// falls back to the full round-A list. Paths assume this repo layout;
// set INDEX_CACHE_ROOT guidance in prompts if caches moved.
export const meta = {
  name: 'zoo-axis-sweep',
  description: 'Mechanical zoo sweep: 9 axes -> verify top modules -> walk-forward composite',
  phases: [
    { title: 'Sweep', detail: 'one agent per information-source axis' },
    { title: 'Verify', detail: 'extra-lag + permutation on top modules' },
    { title: 'Compose', detail: 'causal walk-forward selection over all modules' },
  ],
}

const DIR = '/path/to/user/Desktop/ETF-Research/index project/research/zoo_v1'
const PY = '/path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound/.venv/bin/python'
const AXES = ['mom', 'vol', 'onid', 'sess', 'micro', 'shape', 'flow_h1', 'flow_h3', 'flow_q10']

const SWEEP_SCHEMA = {
  type: 'object', required: ['axis', 'n_runs', 'top'],
  properties: {
    axis: { type: 'string' }, n_runs: { type: 'number' },
    error: { type: 'string' },
    audit_note: { type: 'string' },
    top: { type: 'array', items: { type: 'object',
      required: ['id', 'worst_clean', 'full'],
      properties: { id: { type: 'string' }, worst_clean: { type: ['number', 'null'] },
        full: { type: ['number', 'null'] }, spec: { type: 'string' } } } },
  },
}
const VERIFY_SCHEMA = {
  type: 'object', required: ['id', 'full', 'lag1_full'],
  properties: { id: { type: 'string' }, full: { type: ['number', 'null'] },
    lag1_full: { type: ['number', 'null'] }, perm_p: { type: ['number', 'null'] },
    verdict: { type: 'string' }, error: { type: 'string' } },
}
const COMPOSE_SCHEMA = {
  type: 'object', required: ['summary'],
  properties: { summary: { type: 'string' }, n_modules: { type: 'number' },
    primary_stats: { type: 'string' }, error: { type: 'string' } },
}

phase('Sweep')
const sweep = await parallel(AXES.map(a => () => agent(
  `You are running one axis of a FROZEN mechanical factor sweep. Rules:
- Do NOT edit, add, or delete any file under ${DIR}. The grids are frozen.
- Run exactly: cd "${DIR}" && "${PY}" run_axis.py ${a}
  (long-running: up to 40 minutes; use a 3600000ms timeout on the Bash call)
- If it errors, return {axis:"${a}", n_runs:0, top:[], error:"<message>"} and stop.
- On success read ${DIR}/results/${a}.json. Return the 12 modules with highest
  worst_clean (ignore null worst_clean), each as {id, worst_clean, full, spec}
  with spec JSON-stringified.
- AUDIT: pick one module from your output, recompute its signal value for one
  (date, symbol) cell independently from the raw parquet files under
  /path/to/user/Desktop/ETF-Research/full-etf project2 /full-etf-universe-store/daily/versions/daily-base-v2-*/
  using the spec definition, and compare with the harness. One cell is enough.
  Report agreement or mismatch in audit_note. Do not fix anything yourself.`,
  { label: `sweep:${a}`, phase: 'Sweep', schema: SWEEP_SCHEMA },
)))

// barrier justified: verify targets are ranked across ALL axes
const ok = sweep.filter(Boolean).filter(r => !r.error)
const failed = sweep.filter(Boolean).filter(r => r.error).map(r => r.axis)
if (failed.length) log(`axes failed: ${failed.join(', ')}`)
const cands = ok.flatMap(r => r.top.map(t => ({ ...t, axis: r.axis })))
  .filter(t => t.worst_clean !== null && t.worst_clean > -0.1 && (t.full ?? -9) > 0.3)
  .sort((x, y) => y.worst_clean - x.worst_clean)
const seen = new Set()
const targets = []
for (const t of cands) {
  if (!seen.has(t.id)) { seen.add(t.id); targets.push(t) }
  if (targets.length >= 10) break
}
log(`sweep done: ${ok.length}/${AXES.length} axes, ${cands.length} candidates, verifying top ${targets.length}`)

phase('Verify')
const verified = await parallel(targets.map(t => () => agent(
  `Verify one module of a factor study. Do NOT edit files. Run exactly:
cd "${DIR}" && "${PY}" run_axis.py verify --id ${t.id} --perms 50
(up to 30 minutes; use a 3600000ms Bash timeout). Parse the JSON line it
prints. Return it as {id, full, lag1_full, perm_p, verdict} where verdict is:
"pass" if perm_p is null or <0.1 AND lag1_full > 0.3*full (signal decays
gracefully, not a look-ahead artifact); otherwise "fail" with a short reason
appended. If the command errors return {id:"${t.id}", full:null, lag1_full:null,
error:"<message>"}.`,
  { label: `verify:${t.id}`, phase: 'Verify', schema: VERIFY_SCHEMA },
)))

phase('Compose')
const comp = await agent(
  `Run the walk-forward composite of a factor study. Do NOT edit files. Run:
cd "${DIR}" && "${PY}" walkforward.py
(up to 30 minutes; use a 3600000ms Bash timeout). Then read
${DIR}/composite/summary.json. Return {n_modules, primary_stats, summary}:
primary_stats = JSON string of the trail252_thr0.3 entry's stats;
summary = 2-3 sentences: which variant is best on worst_clean, what the primary
variant's full-window and per-window Sharpes are, and the top-5 modules by
average weight in the primary variant.`,
  { label: 'compose', phase: 'Compose', schema: COMPOSE_SCHEMA },
)

return {
  axes_ok: ok.map(r => `${r.axis}:${r.n_runs}`),
  axes_failed: failed,
  audit_notes: ok.map(r => `${r.axis}: ${r.audit_note || 'n/a'}`),
  verified,
  composite: comp,
}