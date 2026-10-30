"""Mathematical checks on synthetic inputs; no strategy evaluation."""
import unittest
import numpy as np
import pandas as pd
from factor_ops import cs_neutralize_1d, cs_zscore, ts_delay, ts_mean

class FactorOpsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.x = pd.DataFrame(rng.normal(size=(30, 50)))
        self.s = pd.DataFrame(rng.normal(size=(30, 50)))

    def test_neutralized_residual_has_zero_mean_and_exposure(self):
        y = cs_neutralize_1d(self.x, self.s)
        self.assertLess(float(y.mean(axis=1).abs().max()), 1e-12)
        centered = self.s.sub(self.s.mean(axis=1), axis=0)
        self.assertLess(float((y * centered).sum(axis=1).abs().max()), 1e-10)

    def test_future_changes_do_not_change_prefix(self):
        changed = self.x.copy()
        changed.iloc[20:] += 1000
        expected = ts_mean(ts_delay(self.x, 1), 5)
        actual = ts_mean(ts_delay(changed, 1), 5)
        pd.testing.assert_frame_equal(expected.iloc[:21], actual.iloc[:21])

    def test_constant_cross_section_remains_unavailable(self):
        actual = cs_zscore(pd.DataFrame(np.ones((3, 50))))
        self.assertTrue(actual.isna().all().all())

if __name__ == '__main__':
    unittest.main()
