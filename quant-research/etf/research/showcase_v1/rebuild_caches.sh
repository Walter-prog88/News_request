#!/bin/zsh
# Rebuild every cache the execution experiments need, into the persistent
# INDEX_CACHE_ROOT. Deterministic end to end (fixed seeds), so the rebuilt
# panels are bit-identical to the purged session caches.
set -e
export INDEX_CACHE_ROOT=~/index-cache
PY=/path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound/.venv/bin/python
ZOO="/path/to/user/Desktop/ETF-Research/index project/research/zoo_v1"
SHOW="/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1"
mkdir -p ~/index-cache

echo "== [1/6] zoo daily bootstrap =="
cd "$ZOO"
$PY -c "import harness as H; H.build_cache(); P = H.Panels(); P.col('overnight_ret_continuous'); P.col('ret_oc_continuous'); print('zoo dailies ok')"

echo "== [2/6] minute aggregation (all-asset amount/flow) =="
$PY build_minute_allasset.py

echo "== [3/6] liq gate v2 =="
$PY -c "import harness as H; P = H.Panels(); P.liq2(); print('liq_gate_v2 ok')"

echo "== [4/6] showcase training matrix =="
cd "$SHOW"
$PY build_matrix.py

echo "== [5/6] clean factory -> X2_pre2024 =="
FACTORY_DEV_END=1213 FACTORY_TAG=_pre2024 $PY factor_factory.py

echo "== [6/6] frozen panel + twap windows + exec bars =="
$PY -c "
import numpy as np
from sweep_dial import CACHE, load_env
from validation_report_data import frozen_scores
man, dates, G, O2O = load_env()
P = frozen_scores((len(dates), len(man['symbols'])))
np.save(f'{CACHE}/score_frozen_pre2024_panel.npy', P)
print('frozen panel ok', P.shape)
"
$PY twap_decay.py
$PY exec_grid.py
echo "== all caches rebuilt =="
