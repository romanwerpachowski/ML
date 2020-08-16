"""Unit tests for linear_regression module."""

import numpy as np
from scipy import stats
from sklearn import linear_model
import unittest

from PyML import linear_regression

class LinearRegressionTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(4098549032)

    def test_univariate_with_intercept(self):
        n = 25
        slope = 0.1
        intercept = -0.9
        noise = 0.1
        x = np.random.randn(n)
        y = x * slope + intercept + noise * np.random.randn(n)
        result = linear_regression.univariate(x, y)
        slope, intercept, rvalue, _, stderr = stats.linregress(x, y)
        self.assertEqual(n, result.n)
        self.assertEqual(n - 2, result.dof)
        self.assertAlmostEqual(slope, result.slope, delta=1e-15)
        self.assertAlmostEqual(intercept, result.intercept, delta=1e-15)
        self.assertAlmostEqual(rvalue**2, result.r2, delta=1e-15)
        self.assertAlmostEqual(stderr**2, result.var_slope, delta=1e-15)

    def test_univariate_with_intercept_regular(self):
        n = 25
        slope = 0.1
        intercept = -0.9
        noise = 0.1
        x0 = - 0.4
        dx = 0.04
        x = np.linspace(x0, x0 + dx * (n - 1), n)
        y = x * slope + intercept + noise * np.random.randn(n)
        result = linear_regression.univariate_regular(x0, dx, y)
        slope, intercept, rvalue, _, stderr = stats.linregress(x, y)
        self.assertEqual(n, result.n)
        self.assertEqual(n - 2, result.dof)
        self.assertAlmostEqual(slope, result.slope, delta=1e-15)
        self.assertAlmostEqual(intercept, result.intercept, delta=1e-15)
        self.assertAlmostEqual(rvalue**2, result.r2, delta=1e-15)
        self.assertAlmostEqual(stderr**2, result.var_slope, delta=1e-15)

    def test_univariate_without_intercept(self):
        n = 25
        slope = 0.1
        noise = 0.1
        x = np.random.randn(n)
        y = x * slope + noise * np.random.randn(n)
        result = linear_regression.univariate_without_intercept(x, y)
        X = np.empty((n, 1), dtype=float)
        X[:, 0] = x
        lr = linear_model.LinearRegression(fit_intercept=False)
        lr.fit(X, y)
        self.assertEqual(n, result.n)
        self.assertEqual(n - 1, result.dof)
        self.assertAlmostEqual(lr.coef_[0], result.slope, delta=1e-15)
        self.assertEqual(0, result.intercept)
        self.assertEqual(0, result.var_intercept)
        self.assertEqual(0, result.cov_slope_intercept)

    def test_multivariate_add_ones(self):
        n = 25
        d = 4
        X = np.random.randn(n, d)
        y = -0.5 * X[:, 0] + 0.1 * X[:, 1] - X[:, 3] + 4 + 0.2 * np.random.randn(n)        
        result = linear_regression.multivariate(X, y, True)
        self.assertEqual(n, result.n)
        self.assertEqual(n - d - 1, result.dof)
        lr = linear_model.LinearRegression()
        lr.fit(X, y)
        np.testing.assert_array_almost_equal(result.beta[:-1], lr.coef_, decimal=15)
        self.assertAlmostEqual(result.beta[-1], lr.intercept_, delta=1e-15)

    def test_multivariate(self):
        n = 25
        d = 5
        X = np.random.randn(n, d)
        X[:, d - 1] = 1
        y = -0.5 * X[:, 0] + 0.1 * X[:, 1] - X[:, 3] + 4 + 0.2 * np.random.randn(n)
        result = linear_regression.multivariate(X, y, False)
        self.assertEqual(n, result.n)
        self.assertEqual(n - d, result.dof)
        lr = linear_model.LinearRegression(fit_intercept=False)  # Force lr not to add another column with 1's.
        lr.fit(X, y)
        np.testing.assert_array_almost_equal(result.beta, lr.coef_, decimal=14)
        

if __name__ == "__main__": 
    unittest.main()
    print("Done!")