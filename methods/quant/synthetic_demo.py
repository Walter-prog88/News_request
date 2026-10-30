"""Synthetic factor pipeline only: no external market data or backtest."""
import json
import numpy as np
import pandas as pd
from factor_ops import cs_rank, cs_neutralize_1d, ts_delay, ts_mean

def main():
    rng = np.random.default_rng(19)
    dates = pd.date_range('2020-01-01', periods=40, freq='B')
    columns = [f'SYNTHETIC_{i:03d}' for i in range(50)]
    signal = pd.DataFrame(rng.normal(size=(40, 50)), index=dates, columns=columns)
    exposure = pd.DataFrame(rng.normal(size=(40, 50)), index=dates, columns=columns)
    factor = cs_neutralize_1d(cs_rank(ts_mean(ts_delay(signal, 1), 5)), exposure)
    residual_inner_product = factor.mul(exposure.sub(exposure.mean(axis=1), axis=0)).sum(axis=1)
    print(json.dumps({'data':'synthetic seeded random values only', 'shape':list(factor.shape),
                      'valid_values':int(factor.notna().sum().sum()),
                      'max_abs_residual_exposure_inner_product':float(residual_inner_product.abs().max()),
                      'return_or_performance_estimate':False}, indent=2))

if __name__ == '__main__':
    main()
