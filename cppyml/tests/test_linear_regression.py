"""Unit tests for linear_regression module.

(C) 2020 Roman Werpachowski.
"""

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
        rss = 0.01
        tss = 0.1
        slope = -0.1
        intercept = 10
        var_slope = 0.001
        var_intercept = 0.002
        cov_slope_intercept = -0.0005
        var_y = rss / dof
        r2 = 1 - rss / tss
        adjusted_r2 = 1 - (rss / dof) / (tss / (n - 1))
        result = linear_regression.UnivariateOLSResult(
            n, dof, rss, tss, slope, intercept, var_slope, var_intercept, cov_slope_intercept)
        self.assertEqual(n, result.n)
        self.assertEqual(dof, result.dof)
        self.assertEqual(var_y, result.var_y)
        self.assertEqual(r2, result.r2)
        self.assertEqual(adjusted_r2, result.adjusted_r2)
        self.assertEqual(slope, result.slope)
        self.assertEqual(intercept, result.intercept)
        self.assertEqual(var_slope, result.var_slope)
        self.assertEqual(var_intercept, result.var_intercept)
        self.assertEqual(cov_slope_intercept, result.cov_slope_intercept)

    def test_multivariate_ols_result(self):
        n = 100
        dof = 98        
        rss = 0.01
        tss = 0.1
        var_y = rss / dof
        r2 = 1 - rss / tss
        adjusted_r2 = 1 - (rss / dof) / (tss / (n - 1))
        beta = np.array([-0.4, 0.2])
        cov = np.array([[0.004, -0.0001],
                        [-0.00010001, 0.03]])
        result = linear_regression.MultivariateOLSResult(
            n, dof, rss, tss, beta, cov)
        self.assertTrue(result.cov.flags["C_CONTIGUOUS"])
        self.assertEqual(n, result.n)
        self.assertEqual(dof, result.dof)
        self.assertEqual(var_y, result.var_y)
        self.assertEqual(r2, result.r2)
        self.assertEqual(adjusted_r2, result.adjusted_r2)
        np.testing.assert_array_equal(beta, result.beta)
        np.testing.assert_array_equal(cov, result.cov)

    def test_ridge_regression_result(self):
        n = 100
        dof = 98
        rss = 0.01
        tss = 0.1
        var_y = rss / dof
        r2 = 1 - rss / tss
        adjusted_r2 = 1 - (rss / dof) / (tss / (n - 1))
        beta = np.array([-0.4, 0.2])
        cov = np.array([[0.004, -0.0001],
                        [-0.00010001, 0.03]])
        effective_dof = 99
        result = linear_regression.RidgeRegressionResult(
            n, dof, rss, tss, beta, cov, effective_dof)
        self.assertTrue(result.cov.flags["C_CONTIGUOUS"])
        self.assertEqual(n, result.n)
        self.assertEqual(dof, result.dof)
        self.assertEqual(var_y, result.var_y)
        self.assertEqual(r2, result.r2)
        self.assertEqual(adjusted_r2, result.adjusted_r2)
        np.testing.assert_array_equal(beta, result.beta)
        np.testing.assert_array_equal(cov, result.cov)
        self.assertEqual(effective_dof, result.effective_dof)

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
        adjusted_r2 = 1 - (n - 1) * (1 - rvalue**2) / (n - 2)
        self.assertAlmostEqual(adjusted_r2, result.adjusted_r2, delta=1e-15)
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
        adjusted_r2 = 1 - (n - 1) * (1 - rvalue**2) / (n - 2)
        self.assertAlmostEqual(adjusted_r2, result.adjusted_r2, delta=1e-15)
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
        self.assertAlmostEqual(result.r2, result.adjusted_r2, delta=1e-15)

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
        adjusted_r2 = 1 - (n - 1) * (1 - result.r2) / (n - d - 1)
        self.assertAlmostEqual(adjusted_r2, result.adjusted_r2, delta=1e-15)

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
        adjusted_r2 = 1 - (n - 1) * (1 - result.r2) / (n - d)
        self.assertAlmostEqual(adjusted_r2, result.adjusted_r2, delta=1e-15)

    def test_recursive_multivariate_ols_no_data(self):
        rmols = linear_regression.RecursiveMultivariateOLS()
        self.assertEqual(0, rmols.n)
        self.assertEqual(0, rmols.d)
        self.assertEqual(0, len(rmols.beta))

    def test_recursive_multivariate_ols_errors_init(self):
        with self.assertRaises(ValueError):
            linear_regression.RecursiveMultivariateOLS(np.zeros((5, 10)), np.zeros(5))
        with self.assertRaises(ValueError):
            linear_regression.RecursiveMultivariateOLS(np.zeros((10, 10)), np.zeros(12))

    def test_recursive_multivariate_ols_errors_update(self):
        rmols = linear_regression.RecursiveMultivariateOLS()
        with self.assertRaises(ValueError):
            rmols.update(np.zeros((5, 10)), np.zeros(5))
        with self.assertRaises(ValueError):
            rmols.update(np.zeros((10, 10)), np.zeros(12))

    def test_recursive_multivariate_ols_errors_one_sample(self):
        n = 10
        d = 3
        X = np.random.randn(n, d)
        true_beta = 0.5 - np.random.rand(d)
        y = np.matmul(X, true_beta) + 0.1 * np.random.randn(n)
        rmols1 = linear_regression.RecursiveMultivariateOLS()
        rmols1.update(X, y)
        self.assertEqual(n, rmols1.n)
        self.assertEqual(d, rmols1.d)
        rmols2 = linear_regression.RecursiveMultivariateOLS(X, y)
        self.assertEqual(n, rmols2.n)
        self.assertEqual(d, rmols2.d)
        np.testing.assert_array_equal(rmols1.beta, rmols2.beta)
        expected_beta = linear_regression.multivariate(X, y).beta
        np.testing.assert_array_almost_equal(expected_beta, rmols1.beta, 16)

    def test_recursive_multivariate_ols_many_samples(self):
        d = 10
        true_beta = 0.5 - np.random.rand(d)
        sample_sizes = [d, 4, 20, 6, 20, 4, 1, 100]
        rmols = linear_regression.RecursiveMultivariateOLS()
        cumulative_n = 0
        total_n = sum(sample_sizes)
        all_X = np.random.randn(total_n, d)
        all_y = np.matmul(all_X, true_beta) + 0.1 * np.random.randn(total_n)
        for sample_idx, n in enumerate(sample_sizes):
            X = all_X[cumulative_n : (cumulative_n + n)]
            y = all_y[cumulative_n : (cumulative_n + n)]
            rmols.update(X, y)
            cumulative_n += n
            cumulative_X = all_X[:cumulative_n]
            cumulative_y = all_y[:cumulative_n]
            ols_beta = linear_regression.multivariate(cumulative_X, cumulative_y).beta
            np.testing.assert_array_almost_equal(ols_beta, rmols.beta, 13)

    def test_ridge(self):
        n =25
        d = 4
        X = np.random.randn(n, d)
        true_beta = np.random.rand(d)
        intercept = 0.2
        y = np.matmul(X, true_beta) + 0.4 * np.random.randn(n) + intercept
        means = np.mean(X, axis=0)
        Xstd = X - means
        stds = np.std(Xstd, axis=0, ddof=0)
        Xstd = Xstd / stds
        lam = 0.01
        ridge = linear_model.Ridge(alpha=lam, fit_intercept=True, normalize=False)
        result1 = linear_regression.ridge(Xstd, y, lam, do_standardise=False)
        ridge.fit(Xstd, y)
        result2 = linear_regression.ridge(Xstd, y, lam, do_standardise=True)
        result3 = linear_regression.ridge(X, y, lam, do_standardise=True)
        for result in (result1, result2, result3):
            self.assertEqual(n, result.n)
            self.assertEqual(n - d - 1, result.dof)            
        for idx, result in enumerate((result1, result2)):
            self.assertEqual(d + 1, len(result.beta), msg=str(idx))
            np.testing.assert_array_almost_equal(ridge.coef_, result.beta[:d], 14, err_msg=str(idx))            
        np.testing.assert_array_almost_equal(ridge.coef_ / stds, result3.beta[:d], 14)
        self.assertAlmostEqual(ridge.intercept_, result1.beta[d], delta=1e-14)
        self.assertAlmostEqual(ridge.intercept_, result2.beta[d], delta=1e-14)
        self.assertAlmostEqual(ridge.intercept_ - np.dot(ridge.coef_, means / stds), result3.beta[d], delta=1e-14)
        r2 = ridge.score(Xstd, y)
        self.assertAlmostEqual(r2, result1.r2, delta=1e-14)
        self.assertAlmostEqual(r2, result2.r2, delta=1e-14)
        self.assertAlmostEqual(r2, result3.r2, delta=1e-14)
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - d - 1)
        self.assertAlmostEqual(adjusted_r2, result1.adjusted_r2, delta=1e-14)
        self.assertAlmostEqual(adjusted_r2, result2.adjusted_r2, delta=1e-14)
        self.assertAlmostEqual(adjusted_r2, result3.adjusted_r2, delta=1e-14)

    def test_press_univariate_with_intercept(self):
        x = np.array([-1, 0, 1])
        y = np.array([1, 0, 1])
        actual1 = linear_regression.press_univariate(x, y, True)
        actual2 = linear_regression.press_univariate(x, y)
        self.assertEqual(actual1, actual2)
        self.assertAlmostEqual(9, actual1, delta=1e-15)

    def test_press_univariate_without_intercept(self):
        x = np.array([-1, 0, 1])
        y = np.array([1, 0, 1])
        actual = linear_regression.press_univariate(x, y, False)
        self.assertAlmostEqual(8, actual, delta=1e-15)

    def test_press_no_regularisation(self):
        X = np.array([[-1, 1], [0, 1], [1, 1]])
        y = np.array([1, 0, 1])
        actual1 = linear_regression.press(X, y)
        actual2 = linear_regression.press(X, y, "none")        
        self.assertEqual(actual1, actual2)
        self.assertAlmostEqual(9, actual1, delta=1e-15)

    def test_press_ridge_zero_strength(self):
        X = np.array([[-1], [0], [1]])
        y = np.array([1, 0, 1])
        actual1 = linear_regression.press(X, y, "ridge")
        actual2 = linear_regression.press(X, y, "ridge", 0)
        self.assertEqual(actual1, actual2)
        self.assertAlmostEqual(9, actual1, delta=1e-15)

    def test_press_ridge(self):
        X = np.array([[-1], [0], [1]])
        y = np.array([1, 0, 1])
        actual = linear_regression.press(X, y, "ridge", 0.1)
        self.assertGreater(9, actual)


if __name__ == "__main__": 
    unittest.main()    