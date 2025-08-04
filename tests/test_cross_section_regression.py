"""This module contains unit tests for cross-sectional regression utilities."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal

from utils.cross_section_regression import (
    calculate_factor_returns,
    long_to_wide,
    run_cross_sectional_regression,
    wide_to_long,
)

# Unit tests for cross-sectional regression utilities.


class TestRunCrossSectionalRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n_obs = 100
        cls.assets = [f"A{i}" for i in range(5)]
        cls.factors = ["f1", "f2", "f3"]

        cls.true_params = np.array([0.5, -1.2, 2.0])

        # random exposures
        X_raw = np.random.randn(cls.n_obs, len(cls.factors))
        cls.X = pd.DataFrame(
            X_raw,
            index=cls.assets * (cls.n_obs // len(cls.assets)),
            columns=cls.factors,
        )
        cls.X.index = pd.Index(cls.X.index[: cls.n_obs], name="asset")

        # generate returns
        noise = np.random.normal(0, 0.01, cls.n_obs)
        y_vals = X_raw.dot(cls.true_params) + noise
        cls.y = pd.Series(y_vals, index=cls.X.index, name="ret")

        # equal weights
        cls.w = pd.Series(1.0, index=cls.X.index, name="w")

    # normal case
    def test_estimation_accuracy(self):
        beta_hat, resid = run_cross_sectional_regression(self.y, self.X, self.w)
        # factor returns should be close to true parameters
        for i, f in enumerate(self.factors):
            self.assertAlmostEqual(beta_hat[f], self.true_params[i], delta=1e-2)
        # residuals should have correct dimension and mean close to 0
        self.assertEqual(len(resid), len(self.y))
        self.assertAlmostEqual(resid.mean(), 0.0, delta=1e-2)

    # outlier robustness
    def test_extreme_outliers(self):
        y_out = self.y.copy()
        y_out.iloc[::10] += 100
        beta_ref, _ = run_cross_sectional_regression(self.y, self.X, self.w)
        beta_out, _ = run_cross_sectional_regression(y_out, self.X, self.w)
        for f in self.factors:
            self.assertAlmostEqual(beta_ref[f], beta_out[f], delta=0.5)


class TestCalculateFactorReturns(unittest.TestCase):
    """Multi-period factor return and residual calculation function robustness test"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.periods = pd.date_range("2025-01-01", periods=12, freq="M")
        cls.assets = [f"A{i}" for i in range(4)]
        cls.factors = ["f1", "f2"]
        cls.true_params = np.array([1.0, -0.7])

        # ---------- exposures_df : (period, factor·asset) ----------
        cols = pd.MultiIndex.from_product([cls.factors, cls.assets], names=["factor", "asset"])
        cls.exposures_df = pd.DataFrame(
            np.random.randn(len(cls.periods), len(cols)),
            index=cls.periods,
            columns=cols,
        )

        # ---------- returns_df ----------
        ret_mat = []
        for t in cls.periods:
            X_t = (
                cls.exposures_df.loc[t]
                .unstack("factor")  # (asset, factor)
                .loc[cls.assets, cls.factors]
                .values
            )
            noise = np.random.normal(0, 0.02, X_t.shape[0])
            ret_mat.append(X_t.dot(cls.true_params) + noise)

        cls.returns_df = pd.DataFrame(ret_mat, index=cls.periods, columns=cls.assets)

        # ---------- weights_df ----------
        cls.weights_df = pd.DataFrame(1.0, index=cls.periods, columns=cls.assets)

    # ---------- Normal case ----------
    def test_factor_return_shape_and_accuracy(self):
        beta_df, resid_df = calculate_factor_returns(
            self.returns_df, self.exposures_df, self.weights_df
        )
        # Shape should be correct
        self.assertEqual(beta_df.shape, (len(self.periods), len(self.factors)))
        self.assertEqual(resid_df.shape, self.returns_df.shape)

        # Each period's factor returns should be close to the true values
        for f, true_v in zip(self.factors, self.true_params):
            self.assertTrue(np.allclose(beta_df[f], true_v, atol=0.1))

        # Residuals should have mean close to 0
        self.assertAlmostEqual(resid_df.stack().mean(), 0.0, delta=0.05)

    # ---------- Outlier robustness ----------
    def test_outlier_stability(self):
        returns_out = self.returns_df.copy()
        returns_out.iloc[0, 0] += 5  # Insert outlier in first period, first asset

        beta_ref, _ = calculate_factor_returns(self.returns_df, self.exposures_df, self.weights_df)
        beta_out, _ = calculate_factor_returns(returns_out, self.exposures_df, self.weights_df)
        # Outliers should not cause large shifts
        diff = (beta_ref - beta_out).abs().max().max()
        self.assertLess(diff, 2.0)


class TestExposureFormatConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(21)
        cls.dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        cls.symbols = ["AAPL", "MSFT"]
        cls.factors = ["beta", "value"]

        # get wide
        arrays = [
            np.repeat(cls.factors, len(cls.symbols)),
            cls.symbols * len(cls.factors),
        ]
        mcols = pd.MultiIndex.from_arrays(arrays, names=["factor", "symbol"])
        data = np.random.randn(len(cls.dates), len(mcols))
        cls.wide = pd.DataFrame(data, index=cls.dates, columns=mcols)

        # get long
        cls.long = wide_to_long(cls.wide)

    # test wide → long
    def test_wide_to_long_columns_order(self):
        expected_cols = ["date", "symbol", "factor", "loading"]
        self.assertListEqual(list(self.long.columns), expected_cols)

    # Test that the long DataFrame has the correct columns after conversion.
    def test_wide_to_long_row_count(self):
        exp_rows = len(self.dates) * len(self.symbols) * len(self.factors)
        self.assertEqual(len(self.long), exp_rows)

    # test long → wide
    def test_long_to_wide_roundtrip(self):
        wide2 = long_to_wide(self.long)

        assert_index_equal(wide2.columns, self.wide.columns)

        assert_frame_equal(wide2, self.wide, check_exact=True)

    # Test that the wide DataFrame has the correct columns after conversion.
    def test_long_to_wide_columns_hierarchy(self):
        wide2 = long_to_wide(self.long)
        self.assertEqual(wide2.columns.nlevels, 2)
        self.assertListEqual(list(wide2.columns.names), ["factor", "symbol"])


if __name__ == "__main__":
    unittest.main()
