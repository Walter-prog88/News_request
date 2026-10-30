"""硬门槛 A 检验:守门员放行的子集里做 Top20,是否跑赢该子集的等权(用户定的及格线)。
守门 = 个股/板块接近 252 日高点(条件 IC 最高的状态)。2023-2025 OOS 打分,相位平均,含成本。"""
import os, sys, warnings; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST
from afactor.config import LEDGER
pd.set_option("display.width",250)
p=Panel(); u=ST.universe(p,'V1000'); c=p.raw('close_px'); years=np.asarray(p.dates.year)
lo=c.rolling(252,min_periods=120).min(); hi=c.rolling(252,min_periods=120).max()
rpq=((c-lo)/(hi-lo)).rank(axis=1,pct=True)                      # T 日收盘可得
raw=pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code']=raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$',r'\2.\1',regex=True)
lab=pd.Series(dict(zip(raw.code,raw.industry))).reindex(p.codes).replace('',np.nan).dropna()
cnt=lab[lab.index.isin(u.columns[u.mean()>0.2])].value_counts(); lab=lab[lab.map(cnt)>=15]
ir=c.pct_change().where(u)[lab.index].T.groupby(lab).mean().T; idx=(1+ir.fillna(0)).cumprod()
srp=(idx-idx.rolling(252,min_periods=120).min())/(idx.rolling(252,min_periods=120).max()-idx.rolling(252,min_periods=120).min())
stock_srp=pd.DataFrame({cc:srp[lab[cc]] for cc in lab.index}).reindex(columns=p.codes)
sc=pd.DataFrame(np.nan,index=p.dates,columns=p.codes,dtype=np.float32)
for y in [2023,2024,2025]:
    s=pd.read_parquet(LEDGER/f'scores/B_gbt_open_h5_{y}.parquet'); sc.loc[s.index,s.columns]=s.to_numpy()
gates={'无门(全池)':None,
       '个股接近高点(rp>0.8)':rpq>0.8,
       '板块接近高点(srp>0.8)':stock_srp>0.8,
       '板块高∩个股高':(stock_srp>0.8)&(rpq>0.8),
       '对照:个股回撤深(rp<0.2)':rpq<0.2}
rows=[]
for gname,g in gates.items():
    s_g=sc if g is None else sc.where(g)
    ew=pd.DataFrame(0.0,index=p.dates,columns=p.codes) if g is None else pd.DataFrame(0.0,index=p.dates,columns=p.codes).where(g)
    for y in [2023,2024,2025]:
        dm=years==y
        pool=ST.benchmark(p,'V1000','open10',date_mask=dm)['ann_return']
        r_ew=ST.run_strategy(ew,p,'V1000','open10',scheme='top',n=100000,rebalance=5,date_mask=dm)   # 门内等权(含成本)
        r_top=ST.run_strategy(s_g,p,'V1000','open10',scheme='top',n=20,rebalance=5,date_mask=dm)
        pa=r_top.get('phase_ann') or [r_top['ann_return']]
        nsel=(g.reindex(p.dates)[dm].where(u[dm]).sum(axis=1).mean() if g is not None else u[dm].sum(axis=1).mean())
        rows.append(dict(守门=gname,年=y,门内股票数=nsel,全池基准=pool,门内等权=r_ew['ann_return'],
                         门内Top20=r_top['ann_return'],A超额=r_top['ann_return']-r_ew['ann_return'],
                         对全池超额=r_top['ann_return']-pool,夏普=r_top['sharpe'],回撤=r_top['max_dd'],
                         相位极差=max(pa)-min(pa)))
df=pd.DataFrame(rows); df.to_csv(LEDGER/'gate_test_open_h5.csv',index=False)
print('=== 逐年(V1000,主口径,相位平均,含成本)===')
print(df.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\n=== 三年平均 ===')
print(df.groupby('守门',sort=False).agg(门内股票数=('门内股票数','mean'),门内等权=('门内等权','mean'),门内Top20=('门内Top20','mean'),
    A超额=('A超额','mean'),A最差年=('A超额','min'),对全池超额=('对全池超额','mean'),夏普=('夏普','mean'),回撤=('回撤','mean'),相位极差=('相位极差','mean')).round(4).to_string())
