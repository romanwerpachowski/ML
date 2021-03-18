"""Unit tests for utils module.

(C) 2020 Roman Werpachowski.
"""
import numpy as np
import unittest

from cppyml import utils


class TestStandardiseFeatures(unittest.TestCase):

    def test_empty(self):
        X = np.array([[]])
        result = utils.standardise_features(X)
        self.assertNotEqual(id(X), id(result))
        np.testing.assert_equal(result, X)

    def test_wrong_shape(self):
        X = np.array([0, 1])
        with self.assertRaises(ValueError):
            utils.standardise_features(X)

    def test_one_row(self):
        X = np.array([[-0.2, 0, 2]])
        expected = np.array([[0, 0, 0]])
        actual = utils.standardise_features(X)
        np.testing.assert_equal(actual, expected)

    def test_two_rows(self):
        X = np.array([
            [-0.2, 0, 2],
            [0.2, 1, 3],
        ])
        mean_row = (X[0] + X[1]) / 2
        stds = np.array([
            np.std(X[:, 0], ddof=0),
            np.std(X[:, 1], ddof=0),
            np.std(X[:, 2], ddof=0),
            ])
        expected = np.array([
            (X[0] - mean_row) / stds,
            (X[1] - mean_row) / stds,
        ])
        actual = utils.standardise_features(X)
        np.testing.assert_almost_equal(actual, expected, decimal=12)


if __name__ == "__main__": 
    unittest.main()
