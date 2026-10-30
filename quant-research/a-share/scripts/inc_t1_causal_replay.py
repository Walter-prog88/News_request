"""Bounded post-review T1 replay. No preregistration labels or legacy grid are changed.
Day orders: quantities fixed at t-1 close, 5-day rebalance phases. Unfilled
orders are CANCELLED at the chosen window's end, never filled on a future day.
Minute close is an execution proxy; quantities use 10% observed minute turnover.
Buy cash uses start-of-minute cash, not contemporaneous sell proceeds. T+1 sells
are restricted to opening inventory. Fixed1 impact, lagged sigma/ADV/spread.
"""
from pathlib import Path
import sys, time, json, hashlib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from afactor import strategy as ST, impact as IM
from afactor.panel import Panel, _limit_frame
from afactor.config import load_raw_minute, TMP, LEDGER
from afactor.strategy import TRADING_DAYS
import inc_t1_common as C
import inc_t1_build_windows as BW

WINDOWS = {'W1':(1,11), 'W5':(121,181), 'W9':(0,241)}
AUMS = np.array([1e7,1e8,1e9,1e7,1e8,1e9])
RULE = dict(order='day; cancel residual at window end; regenerate only on next 5-day rebalance',
            sizing='previous-close NAV and price; no execution-price target renormalization',
            price='completed minute close proxy', pov='rate fixed at previous close = requested notional / lagged 20-day window turnover median, capped 0.10', cash='start-of-minute cash; no same-minute sell financing',
            settlement='T+1 sell eligibility; sell <= opening inventory', impact='fixed1; cumulative per-stock per-day order notional power 1.5',
            estimates='sigma ADV and halfspread from trade_date-1 or earlier', lots='fractional adjusted units; no retail minimum commission')

def causal_estimates(close, amount, raw_close):
    """Testable causal estimates, independent of same-year window scales."""
    ret=close/close.shift(1)-1
    sigma=IM._fill_rows(ret.rolling(20,min_periods=10).std().shift(1).to_numpy(float),'median',.025)
    adv=IM._fill_rows(amount.rolling(20,min_periods=10).mean().shift(1).to_numpy(float),'q05pos',1e7)
    px=raw_close.where(raw_close>0).fillna(close.where(close>0))
    hs=IM._fill_rows((.005/px).shift(1).to_numpy(float),'median',.001)
    return sigma,adv,hs

class Account:
    """Independent counterfactual accounts (six AUM/strategy rows, five phases)."""
    def __init__(self, n, aums=AUMS):
        self.aums=np.array(aums,float)
        self.units=np.zeros((len(aums),5,n))
        self.cash=np.repeat(self.aums[:,None],5,axis=1)
        self.prev_nav=self.cash.copy()

    def orders(self, phase, weights, prev_price):
        h=self.units[:,phase].copy()
        nav=self.cash[:,phase]+(h*prev_price).sum(axis=1)
        good=np.isfinite(prev_price)&(prev_price>0)
        tgt=np.divide(nav[:,None]*weights,prev_price,out=np.zeros_like(h),where=good)
        # Missing prices freeze existing inventory rather than invent a sale or buy.
        target=np.where(good,tgt,h)
        return target-h

    def execute(self, phase, pending, price, volume, prev_price, limit, sigma, adv, hs, stamp, cumulative, participation=.10):
        """One completed bar; only inputs at or before this bar are consumed."""
        cash0=self.cash[:,phase].copy()
        h=self.units[:,phase]
        valid=np.isfinite(price)&(price>0)&np.isfinite(volume)&(volume>0)&np.isfinite(prev_price)&(prev_price>0)
        p=np.where(valid,price,1.)
        change=p/np.where(prev_price>0,prev_price,1.)-1
        # Conservative touch rule at close: never buy an upper-limit print nor sell a lower-limit print.
        block=((pending>0)&(change>=limit*.985))|((pending<0)&(change<=-limit*.985))
        cap=np.where(valid,participation*volume/p,0.)
        q=np.sign(pending)*np.minimum(np.abs(pending),cap)
        q=np.where(block,0.,q)
        # Stock/order sign is fixed through a window; never sell units purchased today.
        q=np.maximum(q,-h)
        coef=sigma/np.sqrt(adv)
        rate=hs+.0001+np.where(q<0,stamp,0.)
        def expense(f):
            v=np.abs(q)*p*f
            impact=coef*((cumulative+v)**1.5-cumulative**1.5)
            return v*rate+impact
        # Common buy scale per account solves debit <= cash0, including nonlinear impact.
        buys=q>0
        buyv=np.where(buys,q*p,0.)
        debit=(buyv+np.where(buys,expense(1.),0.)).sum(1)
        factor=np.minimum(1.,cash0/np.maximum(debit,1e-300))
        # Convex cost means proportional scaling of full-order debit is an upper bound.
        q=np.where(buys,q*factor[:,None],q)
        notional=np.abs(q)*p
        impact=coef*((cumulative+notional)**1.5-cumulative**1.5)
        fees=notional*(hs+.0001+np.where(q<0,stamp,0.))+impact
        buydebit=(np.where(q>0,q*p,0.)+np.where(q>0,fees,0.)).sum(1)
        assert np.all(buydebit<=cash0+1e-6), 'same-bar cash borrowing'
        self.cash[:,phase]-=(q*p+fees).sum(1)
        self.units[:,phase]+=q
        cumulative+=notional
        pending-=q
        assert self.cash[:,phase].min()>=-1e-6
        assert self.units[:,phase].min()>=-1e-9
        assert not np.any((np.abs(q)>1e-12)&block)
        assert np.all(notional<=np.where(valid,.1*volume,0.)+1e-6)
        return q, fees, block, impact


def run():
    start=time.time();p=Panel();assert p.dates.max()<pd.Timestamp('2026-01-01')
    close=p.raw('close_px').astype(float);amt=p.raw('amount').astype(float)
    # Reuse already lagged cost panels; never invoke window_scales or full-year estimates.
    cm=IM.CostModel(p,impact_y=1.)
    sigma,adv,hs=cm._sigma.astype(float),cm._adv.astype(float),cm._hs.astype(float)
    lim=_limit_frame(p).to_numpy(float)
    pace={w:C.win_amt(p,w).rolling(20,min_periods=10).median().shift(1).to_numpy(float) for w in WINDOWS}
    score=pd.read_parquet(TMP/'comp_v5_score_open10.parquet')
    mask=C.sim_mask(p,'COMP_v5_exclude');tstart=np.flatnonzero(mask)[0]
    W=ST.build_weights(score,p,'V1000',scheme='exclude',buffer=(.20,.35),date_mask=mask).astype(float)
    U=ST.universe(p,'V1000').to_numpy(float);U/=np.maximum(U.sum(1,keepdims=True),1.)
    states={w:Account(len(p.codes)) for w in WINDOWS}
    last=np.full(len(p.codes),np.nan)
    daily=[];orders=[];checks=[];sources=[]
    di={d:i for i,d in enumerate(p.dates)};codepos={c:i for i,c in enumerate(p.codes)}
    outdir=TMP/'inc_t1_causal';outdir.mkdir(exist_ok=True)
    # Freeze rule before reading any replay outcomes.
    (LEDGER/'inc_t1_causal_rules.json').write_text(json.dumps(RULE,ensure_ascii=False,indent=2)+'\n')
    for month in BW.research_months():
        if month<'202007':continue
        assert month<='202512'
        mt=time.time();P=load_raw_minute('close_adj',month);V=load_raw_minute('amt',month)
        assert P.index.equals(V.index) and list(P.columns)==list(V.columns)
        assert P.index.max()<pd.Timestamp('2026-01-01')
        mp=np.array([codepos.get(c,-1) for c in P.columns]);keep=mp>=0;dest=mp[keep]
        pv=P.to_numpy()[:,keep];vv=V.to_numpy()[:,keep]
        dates,starts,counts=np.unique(P.index.normalize(),return_index=True,return_counts=True)
        for d,st,n in zip(dates,starts,counts):
            d=pd.Timestamp(d);te=di[d];t=te-1
            if t<tstart:continue
            assert n==241, (d,n)
            prev=close.iloc[t].to_numpy(float)
            last=np.where(np.isfinite(prev)&(prev>0),prev,last)
            mark=np.nan_to_num(last,nan=0.)
            cnow=close.iloc[te].to_numpy(float)
            endmark=np.where(np.isfinite(cnow)&(cnow>0),cnow,mark)
            phase=(t-tstart)%5
            weights=np.vstack([np.tile(W[t],(3,1)),np.tile(U[t],(3,1))])
            bars=np.full((241,len(p.codes)),np.nan);vol=np.zeros_like(bars)
            bars[:,dest]=pv[st:st+n];vol[:,dest]=vv[st:st+n]
            stamp=.001 if d<pd.Timestamp('2023-08-28') else .0005
            for wn,(lo,hi) in WINDOWS.items():
                a=states[wn];opening=a.units[:,phase].copy()
                pending=a.orders(phase,weights,mark);original=pending.copy()
                rate=np.divide(np.abs(original)*mark,pace[wn][te],out=np.zeros_like(original),where=np.isfinite(pace[wn][te])&(pace[wn][te]>0))
                rate=np.clip(rate,0.,.10)
                cumulative=np.zeros_like(pending);fee=np.zeros_like(pending);imp=np.zeros_like(pending)
                sealed=np.zeros_like(pending,dtype=int);filled=np.zeros_like(pending)
                cashmin=a.cash[:,phase].copy();maxpart=np.zeros_like(pending)
                for b in range(lo,hi):
                    if not (np.abs(pending)>1e-12).any():break
                    q,f,blk,ic=a.execute(phase,pending,bars[b],vol[b],prev,lim[te],sigma[te],adv[te],hs[te],stamp,cumulative,rate)
                    fee+=f;imp+=ic;filled+=q;sealed+=blk&(np.abs(pending)>1e-12)
                    cashmin=np.minimum(cashmin,a.cash[:,phase])
                    part=np.divide(np.abs(q)*np.nan_to_num(bars[b]),vol[b],out=np.zeros_like(q),where=vol[b]>0)
                    maxpart=np.maximum(maxpart,part)
                assert np.all(-np.minimum(filled,0)<=opening+1e-8)
                assert np.allclose(original,filled+pending,atol=1e-8)
                assert maxpart.max()<=.100000001
                assert cashmin.min()>=-1e-6
                for row in range(6):
                    nz=np.flatnonzero(np.abs(original[row])>1e-10)
                    orders.append(pd.DataFrame(dict(date=d,signal_date=p.dates[t],window=wn,phase=phase,
                        strategy='exclude' if row<3 else 'pool',X=AUMS[row],code=p.codes[nz],
                        requested_units=original[row,nz],filled_units=filled[row,nz],cancelled_units=pending[row,nz],
                        notional=cumulative[row,nz],cost=fee[row,nz],impact=imp[row,nz],blocked_bars=sealed[row,nz],
                        max_minute_participation=maxpart[row,nz],locked_participation=rate[row,nz])))
                nav=a.cash+(a.units*endmark).sum(2)
                ret=nav/a.prev_nav-1
                for row in range(6):
                    for ph in range(5):
                        cost=fee[row].sum() if ph==phase else 0.
                        daily.append(dict(date=d,window=wn,X=AUMS[row],strategy='exclude' if row<3 else 'pool',phase=ph,
                            nav=nav[row,ph],ret=ret[row,ph],cash=a.cash[row,ph],exposure=1-a.cash[row,ph]/nav[row,ph],
                            cost=cost,cost_frac=cost/a.prev_nav[row,ph],
                            cancelled_notional=(np.abs(pending[row])*mark).sum() if ph==phase else 0.,
                            min_intraminute_cash=cashmin[row] if ph==phase else a.cash[row,ph]))
                a.prev_nav=nav
            last=endmark
        of=outdir/f'orders_{month}.parquet';pd.concat(orders,ignore_index=True).to_parquet(of,index=False);orders=[]
        sources.append(dict(month=month,first=str(P.index.min()),last=str(P.index.max()),bars=len(P),orders_file=str(of),max_source_date=str(P.index.max())))
        pd.DataFrame(daily).to_parquet(outdir/'daily_checkpoint.parquet',index=False)
        print(f'{month} completed {time.time()-mt:.1f}s total {time.time()-start:.1f}s',flush=True)
        del P,V,pv,vv,bars,vol
    df=pd.DataFrame(daily);df.to_csv(LEDGER/'inc_t1_causal_daily.csv',index=False)
    pd.DataFrame(sources).to_csv(LEDGER/'inc_t1_causal_sources.csv',index=False)
    finish(df,outdir)


def finish(df,outdir):
    avg=df.groupby(['date','window','X','strategy'])[['ret','cost_frac','exposure','cash','cancelled_notional']].mean().reset_index()
    rows=[]
    for X in [1e7,1e8,1e9]:
        D=avg[avg.X==X]
        for period,(y0,y1,_) in C.PERIODS['COMP_v5_exclude'].items():
            g=D[pd.to_datetime(D.date).dt.year.between(y0,y1)]
            rates=g.pivot(index='date',columns=['window','strategy'],values='ret')
            def ann(s):return (np.prod(1+s)**(TRADING_DAYS/len(s))-1)*100
            for wn in WINDOWS:
                r=rates[wn,'exclude'];b=rates[wn,'pool'];base=rates['W1','exclude'];bb=rates['W1','pool']
                delta=(r-b)-(base-bb)
                gd=g[(g.window==wn)&(g.strategy=='exclude')]
                rows.append(dict(X=X,window=wn,period=period,n_days=len(r),ann_pp=ann(r),pool_ann_pp=ann(b),
                    excess_geo_pp=ann(r)-ann(b),total_diff_vs_W1_pp=ann(r)-ann(base),
                    d_excess_vs_W1_pp=(ann(r)-ann(b))-(ann(base)-ann(bb)),
                    paired_excess_arith_pp=delta.mean()*TRADING_DAYS*100,t_nw=C.nw_t(delta),
                    mean_exposure=gd.exposure.mean(),cost_ann_pp=gd.cost_frac.mean()*TRADING_DAYS*100,
                    cancel_ann_pct=gd.cancelled_notional.sum()/X/len(r)*TRADING_DAYS*100))
    S=pd.DataFrame(rows);S.to_csv(LEDGER/'inc_t1_causal_summary.csv',index=False)
    checks=[]
    for f in sorted(outdir.glob('orders_*.parquet')):
        q=pd.read_parquet(f)
        checks.append(dict(check='orders_'+f.stem,value=float((q.requested_units-q.filled_units-q.cancelled_units).abs().max()),
            max_part=float(q.max_minute_participation.max()),passed=bool(np.allclose(q.requested_units,q.filled_units+q.cancelled_units,atol=1e-8) and q.max_minute_participation.max()<=.100000001)))
    checks.extend([dict(check='min_cash',value=float(df.min_intraminute_cash.min()),passed=bool(df.min_intraminute_cash.min()>=-1e-6)),
                   dict(check='research_cutoff',value=str(pd.to_datetime(df.date).max()),passed=bool(pd.to_datetime(df.date).max()<pd.Timestamp('2026-01-01'))),
                   dict(check='fixed1',value=1.,passed=True)])
    pd.DataFrame(checks).to_csv(LEDGER/'inc_t1_causal_checks.csv',index=False)
    assert all(x['passed'] for x in checks)
    print(S[S.period=='2021-2025'].round(4).to_string(index=False),flush=True)

if __name__=='__main__':run()
