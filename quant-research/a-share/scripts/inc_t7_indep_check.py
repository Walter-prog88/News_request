"""INC-T7 独立复核(断点续做时对"沿用的网格结果"做的核对):用官方入口 ST.run_weights / LE.run(不经 inc_t7_grid.run_cell 的相位循环)
重跑几个 k=1 / k=10 格子,与网格落盘的相位平均日收益逐日比较。结果写 cache/_tmp/inc_t7_indep.json,由 inc_t7_report.py 并入 inc_t7_checks.csv。
用法:PYTHONPATH=src python3 scripts/inc_t7_indep_check.py
"""
import sys, warnings, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from afactor import strategy as ST, lh_engine as LE, impact as IM
from afactor.panel import Panel
from afactor.config import TMP
import cap_capacity_curve as CAP
out={}
# minute: 独立路径 run_weights(官方入口) vs 缓存的相位平均日序列
p=Panel(); d=p.dates
m=np.asarray((d>=pd.Timestamp('2020-07-01'))&(d<=pd.Timestamp('2025-12-31')))
score=pd.read_parquet(TMP/'comp_v5_score_open10.parquet')
D=pd.read_parquet(TMP/'inc_t7_grid_daily_minute.parquet')
for k,band in ((1,(0.2,0.35)),(10,(0.2,0.35)),(1,(0.3,0.45))):
    W=ST.build_weights(score,p,'V1000',scheme='exclude',buffer=band,date_mask=m)
    cm=CAP.ConstAUMCost(IM.CostModel(p,impact_y=1.0,spread_mult=1.0))
    res=ST.run_weights(W,p,'open10',1e7,rebalance=k,date_mask=m,cost_model=cm,record=False)
    name=f"k{k}|{int(band[0]*100)}/{int(band[1]*100)}|Y1"
    c=D[D.cell==name].set_index('date')['ret']
    dd=(res['rets'].reindex(c.index)-c).abs().max()
    out[f'minute_{name}']=float(dd)
    print('minute',name,'max |diff| daily ret =',dd,flush=True)
# 长历史
X=LE.load_inputs(); ad=LE.adapter(X)
D=pd.read_parquet(TMP/'inc_t7_grid_daily_lh.parquet')
for k,band in ((1,(0.2,0.35)),(10,(0.2,0.35))):
    W=LE.weights(X,'S1',buffer=band)
    res=LE.run(W,ad,rebalance=k,cost='Y1',record=False)
    name=f"k{k}|{int(band[0]*100)}/{int(band[1]*100)}|Y1"
    c=D[D.cell==name].set_index('date')['ret']
    r=res['rets'] if isinstance(res,dict) else res
    dd=(r.reindex(c.index)-c).abs().max()
    out[f'lh_{name}']=float(dd)
    print('lh',name,'max |diff| daily ret =',dd,flush=True)

(TMP/'inc_t7_indep.json').write_text(json.dumps(out,indent=1))
print(out)
