import sys, gc, numpy as np, pandas as pd
sys.path.insert(0,"/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
from afactor.strategy import universe
from afactor import newdata
import psutil, os
p=Panel(); dates,codes=p.dates,p.codes; T,N=len(dates),len(codes)
U=universe(p,'V1000').values.copy()
p._cache.clear(); gc.collect()
turn=newdata.turnover_rate(dates,codes).reindex(index=dates,columns=codes).values.astype(np.float32)
mv=newdata.total_mv(dates,codes).reindex(index=dates,columns=codes).values.astype(np.float32)
t20=pd.DataFrame(turn).rolling(20,min_periods=15).mean().values.astype(np.float32)
def pr(a):
    out=np.full(a.shape,255,np.uint8)
    for s in range(0,T,300):
        e=min(s+300,T)
        x=np.where(U[s:e],a[s:e],np.nan).astype(np.float32)
        r=pd.DataFrame(x).rank(axis=1,pct=True).values
        ok=np.isfinite(r); blk=np.full(x.shape,255,np.uint8)
        blk[ok]=np.clip(r[ok]*200,0,200).astype(np.uint8); out[s:e]=blk
    return out
np.save('/tmp/rk_turn20.npy', pr(t20))
np.save('/tmp/rk_logmv.npy', pr(np.log(np.where(mv>0,mv,np.nan)).astype(np.float32)))
np.save('/tmp/U.npy', U)
print("peak rss GB", psutil.Process(os.getpid()).memory_info().rss/1e9)
