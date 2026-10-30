"""Economic invariants and future perturbation tests for bounded T1 cash replay."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import inc_t1_causal_replay as R


def step(a,pending,p=10.,volume=1e8,prev=10.,lim=.1,hs=0.,sigma=0.,cum=None,participation=.10):
    n=pending.shape[1]
    ar=lambda v:np.broadcast_to(np.array(v,float),(n,)).copy()
    c=np.zeros_like(pending) if cum is None else cum
    return a.execute(0,pending,ar(p),ar(volume),ar(prev),ar(lim),ar(sigma),ar(1e8),ar(hs),0.,c,participation)


def test_directional_limit_and_t1_inventory():
    a=R.Account(2,[1000]);a.units[0,0]=[10,10]
    pending=np.array([[2.,-2.]])
    q,*_=step(a,pending,p=[11,9])
    assert np.array_equal(q,[[0.,0.]])
    q,*_=step(a,pending,p=[9,11])
    assert q[0,0]>0 and q[0,1]<0


def test_minute_side_capacity_and_cash_including_costs():
    a=R.Account(2,[100.]);pending=np.array([[100.,100.]])
    q,fees,*_=step(a,pending,volume=[500,500],hs=.01,sigma=.03)
    assert (q*10<=50+1e-10).all()
    assert (q*10+fees).sum()<=100+1e-8
    assert a.cash.min()>=0
    assert np.allclose(pending+q,[[100,100]])


def test_no_same_bar_sell_financing():
    a=R.Account(2,[0.]);a.units[0,0]=[10,0]
    pending=np.array([[-10.,10.]])
    q,*_=step(a,pending)
    assert q[0,0]==-10 and q[0,1]==0
    q,*_=step(a,pending)
    assert q[0,1]>0 and a.cash.min()>=0


def test_consecutive_seals_never_fictionally_fill_and_explicit_cancel():
    a=R.Account(1,[1000]);pending=np.array([[10.]])
    for _ in range(8):
        q,*_=step(a,pending,p=11.)
        assert q.sum()==0 and pending.sum()==10
    cancelled=pending.copy();pending[:]=0
    assert a.units.sum()==0 and a.cash[0,0]==1000 and cancelled.sum()==10


def test_missing_price_or_volume_never_fill():
    for px,vol in [(np.nan,1000),(10,0),(10,np.nan)]:
        a=R.Account(1,[100]);pending=np.array([[10.]])
        q,*_=step(a,pending,p=px,volume=vol)
        assert q.sum()==0 and a.units.sum()==0


def test_inventory_conservation_and_cumulative_impact():
    a=R.Account(1,[10000]);pending=np.array([[10.]])
    cum=np.zeros_like(pending);impact=0.
    for _ in range(2):
        q,f,b,ic=step(a,pending,volume=500,sigma=.02,cum=cum);impact+=ic.sum()
    assert np.isclose(a.units.sum(),10.)
    assert np.isclose(impact,.02/np.sqrt(1e8)*100**1.5)


def test_future_bar_perturbation_leaves_prefix_inventory_cash_orders_unchanged():
    def run(prices):
        a=R.Account(1,[100]);pending=np.array([[10.]]);snap=[]
        for price in prices:
            step(a,pending,p=price,volume=100)
            snap.append((a.units.copy(),a.cash.copy(),pending.copy()))
        return snap
    x=run([10,10,10,10]);y=run([10,10,11,9])
    for i in [0,1]:
        for j in range(3):assert np.array_equal(x[i][j],y[i][j])


def test_cost_estimates_ignore_today_and_future_prices_volumes():
    dates=pd.date_range('2020-01-01',periods=80)
    rng=np.random.default_rng(3)
    close=pd.DataFrame(10*np.exp(np.cumsum(rng.normal(0,.02,(80,25)),axis=0)),index=dates)
    amount=pd.DataFrame(rng.uniform(1e7,1e8,(80,25)),index=dates)
    raw=close.copy();a=R.causal_estimates(close,amount,raw)
    close.iloc[45:]*=1.7;amount.iloc[45:]*=20;raw.iloc[45:]*=3
    b=R.causal_estimates(close,amount,raw)
    for x,y in zip(a,b):assert np.array_equal(x[:46],y[:46])


def test_actual_costmodel_future_prefix_invariance(monkeypatch):
    """Test the IM.CostModel used in production, not just a helper formula."""
    import afactor.impact as IM
    dates=pd.date_range('2021-01-01',periods=80)
    rng=np.random.default_rng(7)
    c=pd.DataFrame(10*np.exp(np.cumsum(rng.normal(0,.02,(80,25)),axis=0)),index=dates)
    amt=pd.DataFrame(rng.uniform(1e7,1e8,(80,25)),index=dates)
    class P:
        sealed=False
        def __init__(self,cl,am):self.cl=cl;self.am=am;self._derived={};self.dates=cl.index;self.codes=cl.columns
        def raw(self,k):return self.cl if k=='close_px' else self.am
    monkeypatch.setattr(IM,'tick_floor',lambda p,tick=.01:tick/p.raw('close_px'))
    p=P(c,amt);a=IM.CostModel(p)
    c2=c.copy();v2=amt.copy();c2.iloc[45:]*=2.;v2.iloc[45:]*=17
    b=IM.CostModel(P(c2,v2))
    for k in ['_sigma','_adv','_hs']:
        assert np.array_equal(getattr(a,k)[:46],getattr(b,k)[:46])
    trunc=IM.CostModel(P(c.iloc[:46],amt.iloc[:46]))
    for k in ['_sigma','_adv','_hs']:
        assert np.array_equal(getattr(a,k)[:46],getattr(trunc,k))


def test_locked_orders_do_not_use_trade_price():
    a=R.Account(2,[100.]);w=np.array([[.5,.5]])
    locked=a.orders(0,w,np.array([10.,20.]))
    assert np.allclose(locked,[[5.,2.5]])
    # A future price gap changes cost/filled amount, never the original requested units.
    request=locked.copy();step(a,locked,p=[10.5,19.])
    assert np.allclose(request,[[5.,2.5]])


def test_locked_pov_paces_long_window_instead_of_front_loading():
    a=R.Account(1,[1000]);pending=np.array([[10.]])
    rate=100./10000. # prior-close order / historical window-volume estimate
    q,*_=step(a,pending,volume=1000,participation=rate)
    assert np.isclose(q[0,0],1.) and np.isclose(pending[0,0],9.)
    for _ in range(9):step(a,pending,volume=1000,participation=rate)
    assert np.isclose(a.units.sum(),10.)


def test_lagged_window_volume_rate_is_prefix_invariant():
    volume=pd.DataFrame({'a':np.arange(80,dtype=float)+10000})
    before=volume.rolling(20,min_periods=10).median().shift(1)
    volume.iloc[45:]*=1000
    after=volume.rolling(20,min_periods=10).median().shift(1)
    assert before.iloc[:46].equals(after.iloc[:46])


def test_buffered_signal_target_prefix_never_uses_future_scores_or_pool():
    from afactor.backtest import weights_exclude_buffer
    rng=np.random.default_rng(71)
    scores=rng.normal(size=(80,25)).astype(np.float32)
    pool=rng.uniform(size=(80,25))>.1
    first=weights_exclude_buffer(scores,pool,sign=1,enter=.20,exit=.35)
    scores[45:]*=-100;pool[45:]=~pool[45:]
    changed=weights_exclude_buffer(scores,pool,sign=1,enter=.20,exit=.35)
    assert np.array_equal(first[:45],changed[:45])
    prefix=weights_exclude_buffer(scores[:45],pool[:45],sign=1,enter=.20,exit=.35)
    assert np.array_equal(first[:45],prefix)
