"""Unit tests for decision_trees module.

(C) 2020 Roman Werpachowski.
"""

import unittest

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cppyml import decision_trees


class DecisionTreesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.diabetes = load_diabetes(return_X_y=True)

    def setUp(self):
        np.random.seed(1066)
        self.min_split_size = 5        

    def test_classification(self):
        iris = load_iris()

        clf = DecisionTreeClassifier(random_state=0, min_samples_split=self.min_split_size)
        clf.fit(iris.data, iris.target)
        sklearn_accuracy = clf.score(iris.data, iris.target)

        tree, _, _ = decision_trees.classification_tree(iris.data, iris.target, alphas=[], min_split_size=self.min_split_size)
        pyml_accuracy = decision_trees.classification_tree_accuracy(tree, iris.data, iris.target)        
        
        self.assertAlmostEqual(sklearn_accuracy, pyml_accuracy, delta=1e-12)

    def test_regression(self):
        X, y = self.diabetes

        regressor = DecisionTreeRegressor(random_state=0, min_samples_split=self.min_split_size)
        regressor.fit(X, y)
        y_hat = regressor.predict(X)
        sklearn_mse = ((y - y_hat)**2).mean()

        tree, _, _ = decision_trees.regression_tree(X, y, alphas=[], min_split_size=self.min_split_size)
        pyml_mse = decision_trees.regression_tree_mean_squared_error(tree, X, y)
        self.assertGreater(pyml_mse, 0)        
        self.assertAlmostEqual(sklearn_mse, pyml_mse, delta=1e-12)

    def test_regression_with_pruning(self):
        X, y = self.diabetes
        tree, _, _ = decision_trees.regression_tree(X, y, alphas=[], min_split_size=self.min_split_size)
        mse = decision_trees.regression_tree_mean_squared_error(tree, X, y)
        pruned_tree, _, _ = decision_trees.regression_tree(X, y, min_split_size=self.min_split_size)
        pruned_mse = decision_trees.regression_tree_mean_squared_error(pruned_tree, X, y)        
        self.assertGreater(pruned_mse, mse)


if __name__ == "__main__": 
    unittest.main()