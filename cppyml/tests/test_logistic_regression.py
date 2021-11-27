"""Unit tests for logistic_regression module.

(C) 2021 Roman Werpachowski.
"""

import unittest

import numpy as np
from sklearn import linear_model

from cppyml import logistic_regression


class TestConjugateGradientLogisticRegression(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(33452345)

    def test_against_sklearn(self):
        n = 100
        d = 5

        X = np.random.randn(n, d)
        w = np.random.randn(d)
        z = np.matmul(X, w) + np.random.randn(n) * 0.1
        y = np.sign(z)
        y[y == 0] = 1
        tested = logistic_regression.ConjugateGradientLogisticRegression()
        lam = 0.2
        tol = 1e-15
        tested.set_lam(lam)
        tested.set_weight_absolute_tolerance(tol)
        tested.set_weight_relative_tolerance(tol)
        actual = tested.fit(X, y)
        self.assertTrue(actual.converged)
        benchmark = linear_model.LogisticRegression(tol=tol, C=1/lam)
        expected = benchmark.fit(X, (y + 1) / 2)
        np.testing.assert_allclose(expected.coef_[0], actual.w, rtol=2e-2)

    def test_setters(self):
        lr = logistic_regression.ConjugateGradientLogisticRegression()
        lam = 0.1
        rtol = 1e-8
        atol = 1e-14
        max_steps = 100
        lr.set_lam(lam)
        lr.set_weight_absolute_tolerance(atol)
        lr.set_weight_relative_tolerance(rtol)
        lr.set_maximum_steps(max_steps)
        self.assertEqual(lam, lr.lam)
        self.assertEqual(rtol, lr.weight_relative_tolerance)
        self.assertEqual(atol, lr.weight_absolute_tolerance)
        self.assertEqual(max_steps, lr.maximum_steps)


if __name__ == "__main__":
    unittest.main()
