"""Unit tests for linear_regression module."""

import unittest

import numpy as np
from scipy import stats
from sklearn import linear_model

from cppyml import linear_regression


class LinearRegressionTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(4098549032)

    def test_univariate_ols_result(self):
        n = 100
        dof = 98
        var_y = 0.3
        r2 = 0.8
        slope = -0.1
        intercept = 10
        var_slope = 0.001
        var_intercept = 0.002
        cov_slope_intercept = -0.0005
        result = linear_regression.UnivariateOLSResult(
            n, dof, var_y, r2, slope, intercept, var_slope, var_intercept, cov_slope_intercept)
        self.assertEqual(n, result.n)
        self.assertEqual(dof, result.dof)
        self.assertEqual(var_y, result.var_y)
        self.assertEqual(r2, result.r2)
        self.assertEqual(slope, result.slope)
        self.assertEqual(intercept, result.intercept)
        self.assertEqual(var_slope, result.var_slope)
        self.assertEqual(var_intercept, result.var_intercept)
        self.assertEqual(cov_slope_intercept, result.cov_slope_intercept)

    def test_multivariate_ols_result(self):
        n = 100
        dof = 98
        var_y = 0.3
        r2 = 0.8
        beta = np.array([-0.4, 0.2])
        cov = np.array([[0.004, -0.0001],
                        [-0.00010001, 0.03]])
        result = linear_regression.MultivariateOLSResult(
            n, dof, var_y, r2, beta, cov)
        self.assertTrue(result.cov.flags["C_CONTIGUOUS"])
        self.assertEqual(n, result.n)
        self.assertEqual(dof, result.dof)
        self.assertEqual(var_y, result.var_y)
        self.assertEqual(r2, result.r2)
        np.testing.assert_array_equal(beta, result.beta)
        np.testing.assert_array_equal(cov, result.cov)

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
        self.assertTrue(result.cov.flags["C_CONTIGUOUS"])
        self.assertEqual(n, result.n)
        self.assertEqual(n - d, result.dof)
        lr = linear_model.LinearRegression(fit_intercept=False)  # Force lr not to add another column with 1's.
        lr.fit(X, y)
        np.testing.assert_array_almost_equal(result.beta, lr.coef_, decimal=14)
        

if __name__ == "__main__": 
    unittest.main()    