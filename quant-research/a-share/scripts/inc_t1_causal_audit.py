"""Independent journal reconstruction of T1 replay cash/inventory/NAV, not a second backtest."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from afactor.panel import Panel
from afactor.config import TMP,LEDGER


def audit():
    daily=pd.read_csv(LEDGER/'inc_t1_causal_daily.csv',parse_dates=['date'])
    panel=Panel();assert panel.dates.max()<pd.Timestamp('2026-01-01')
    mark=panel.raw('close_px').ffill().fillna(0).astype(float)
    cp={c:i for i,c in enumerate(panel.codes)}
    books={};balances={};rows=[];seen=0
    for f in sorted((TMP/'inc_t1_causal').glob('orders_*.parquet')):
        month=f.stem[-6:];assert month<='202512'
        q=pd.read_parquet(f)
        assert pd.to_datetime(q.date).max()<pd.Timestamp('2026-01-01')
        assert (pd.to_datetime(q.signal_date)<pd.to_datetime(q.date)).all()
        maxres=float(np.max(np.abs(q.requested_units-q.filled_units-q.cancelled_units)))
        maxpart=float(q.max_minute_participation.max())
        max_pacing_excess=float((q.max_minute_participation-q.locked_participation).max())
        assert max_pacing_excess<=1e-9
        max_casherr=max_naverr=0.;min_units=0.
        # Build the cash and units journal ONLY from the saved actual fills/fees.
        groups={key:g for key,g in q.groupby(['date','window','X','strategy','phase'])}
        d=daily[daily.date.dt.strftime('%Y%m')==month]
        for r in d.sort_values('date').itertuples():
            key=(r.window,r.X,r.strategy,r.phase)
            if key not in books:books[key]=np.zeros(len(cp));balances[key]=r.X
            g=groups.get((r.date,r.window,r.X,r.strategy,r.phase))
            if g is not None:
                ix=np.array([cp[c] for c in g.code]);fill=g.filled_units.to_numpy()
                assert np.all(-np.minimum(fill,0)<=books[key][ix]+1e-8), 'T+1/short inventory violation'
                books[key][ix]+=fill
                balances[key]-=float((np.sign(g.requested_units)*g.notional+g.cost).sum())
            min_units=min(min_units,float(books[key].min()))
            casherr=abs(balances[key]-r.cash);max_casherr=max(max_casherr,casherr)
            nav=balances[key]+books[key]@mark.loc[r.date].to_numpy()
            max_naverr=max(max_naverr,abs(nav-r.nav))
            assert casherr<=1e-5*r.X/1e7 and abs(nav-r.nav)<=1e-5*r.X/1e7
        seen+=len(q)
        rows.append(dict(month=month,orders=len(q),max_order_units_residual=maxres,max_participation=maxpart,max_pacing_excess=max_pacing_excess,
                         max_cash_journal_error_yuan=max_casherr,max_nav_journal_error_yuan=max_naverr,
                         minimum_units=min_units,passed=maxres<1e-8 and maxpart<=.100000001 and min_units>=-1e-8))
        print(month,'cash/NAV reconciled',len(q),flush=True)
    df=pd.DataFrame(rows);assert df.passed.all()
    df.to_csv(LEDGER/'inc_t1_causal_journal_checks.csv',index=False)
    held=[]
    for key,v in books.items():
        w,X,s,phase=key
        for i in np.flatnonzero(v>1e-12):held.append(dict(window=w,X=X,strategy=s,phase=phase,code=panel.codes[i],terminal_adjusted_units=v[i]))
    pd.DataFrame(held).to_parquet(TMP/'inc_t1_causal'/'terminal_inventory.parquet',index=False)
    print('AUDIT PASS',len(df),'months',seen,'orders',flush=True)

if __name__=='__main__':audit()
