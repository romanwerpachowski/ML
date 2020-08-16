"""Unit tests for linear_regression module."""

import numpy as np
from scipy import stats
import unittest

from PyML import linear_regression

class LinearRegressionTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(4098549032)

    def univariate_with_intercept(self):
        n = 25
        slope = 0.1
        intercept = -0.9
        noise = 0.1
        x = np.random.randn(n)
        y = x * slope + intercept + noise * np.random.randn(n)
        result = linear_regression.univariate(x, y)
        slope, intercept, rvalue, _, stderr = stats.linregress(x, y)
        self.assertAlmostEqual(slope, result.slope, delta=1e-15)


if __name__ == "__main__": 
    unittest.main()