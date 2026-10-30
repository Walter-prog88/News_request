"""条件 IC:同一个信号在不同"状态"子集里的 |Rank IC|(ETF 项目 style_log Q2 的板块级/个股级复现)。
ETF 项目(市场级,n=3120):接近高点 0.267 vs 回撤深 0.069;低波 0.243 vs 高波 0.010;趋势上/下无分辨。
本脚本只用 2020-2025,信号=考试同款 B×GBT OOS 打分 + 三个核心因子。"""
import os, sys, warnings; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import factors as FA, strategy as ST
from afactor.config import LEDGER
from afactor.fast_eval import _nw_t
p=Panel(); u=ST.universe(p,'V1000'); c=p.raw('close_px'); tw=p.raw('twap_open10'); years=np.asarray(p.dates.year)
fwd=(tw.shift(-6)/tw.shift(-1)-1).where(u)
lo=c.rolling(252,min_periods=120).min(); hi=c.rolling(252,min_periods=120).max()
rpq=((c-lo)/(hi-lo)).shift(1).rank(axis=1,pct=True)
volq=c.pct_change().rolling(20).std().shift(1).rank(axis=1,pct=True)
FA.build_all()
sc=pd.DataFrame(np.nan,index=p.dates,columns=p.codes,dtype=np.float32)
for y in [2023,2024,2025]:
    s=pd.read_parquet(LEDGER/f'scores/B_gbt_open_h5_{y}.parquet'); sc.loc[s.index,s.columns]=s.to_numpy()
sig={'B×GBT(OOS 23-25)':sc}
for n in ['ivol_20','rev_20','id_v_ac1_m5']:
    sig[n]=FA.REGISTRY[n][1](p).reindex(index=p.dates,columns=p.codes)
raw=pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code']=raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$',r'\2.\1',regex=True)
lab=pd.Series(dict(zip(raw.code,raw.industry))).reindex(p.codes).replace('',np.nan).dropna()
cnt=lab[lab.index.isin(u.columns[u.mean()>0.2])].value_counts(); lab=lab[lab.map(cnt)>=15]
ir=c.pct_change().where(u)[lab.index].T.groupby(lab).mean().T; idx=(1+ir.fillna(0)).cumprod()
slo=idx.rolling(252,min_periods=120).min(); shi=idx.rolling(252,min_periods=120).max()
srp=((idx-slo)/(shi-slo)).shift(1); svol=ir.rolling(20).std().shift(1).rank(axis=1,pct=True)
m2s=lambda M: pd.DataFrame({cc:M[lab[cc]] for cc in lab.index}).reindex(columns=p.codes)
stock_srp, stock_svol = m2s(srp), m2s(svol)
# 预转 numpy,加速
U=u.to_numpy(bool); F=fwd.to_numpy(np.float32)
from scipy.stats import rankdata
def cond_ic(S,mask,yrs,step=2):
    A=S.to_numpy(np.float32); M=None if mask is None else mask.to_numpy()
    out=[]
    for i in range(0,len(p.dates)-7,step):
        if years[i] not in yrs: continue
        ok=U[i]&np.isfinite(A[i])&np.isfinite(F[i])
        if M is not None: ok&=(M[i]==True)
        if ok.sum()<150: continue
        out.append(np.corrcoef(rankdata(A[i,ok]),rankdata(F[i,ok]))[0,1])
    v=np.array(out); return (abs(v.mean()), _nw_t(v,5)) if len(v)>10 else (np.nan,np.nan)
def table(title, masks, labels):
    print(f'\n=== {title} ===')
    print(f'{"信号":20s}' + ''.join(f'{l:>18s}' for l in labels))
    for name,S in sig.items():
        yrs={2023,2024,2025} if 'GBT' in name else set(range(2020,2026))
        print(f'{name:20s}' + ''.join(f'    {a:.3f} (t{abs(t):4.1f})' for a,t in (cond_ic(S,m,yrs) for m in masks)))
table('个股级:信号在不同状态子集里的 |Rank IC|(5 日,可交易口径)',
      [None, rpq>0.8, rpq<0.2, volq<0.3, volq>0.7], ['无条件','接近高点','回撤深','低波','高波'])
table('板块级:所属板块处于不同状态时,成分股上的 |Rank IC|',
      [stock_srp>0.8, stock_srp<0.2, stock_svol<0.3, stock_svol>0.7], ['板块接近高点','板块回撤深','板块低波','板块高波'])
table('交叉:板块接近高点 ∩ 个股状态',
      [(stock_srp>0.8)&(rpq>0.8), (stock_srp>0.8)&(rpq<0.2), (stock_srp>0.8)&(volq<0.3)], ['板块高∩个股高','板块高∩个股回撤','板块高∩个股低波'])
