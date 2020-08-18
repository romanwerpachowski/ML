"""Unit tests for decision_trees module."""

import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from PyML import decision_trees


class DecisionTreesTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(1066)

    def test_classification(self):
        min_split_size = 5        
        iris = load_iris()

        clf = DecisionTreeClassifier(random_state=0, min_samples_split=min_split_size)
        clf.fit(iris.data, iris.target)
        sklearn_accuracy = clf.score(iris.data, iris.target)

        tree, _, _ = decision_trees.classification_tree(iris.data.T, iris.target, alphas=[], min_split_size=min_split_size)
        pyml_accuracy = decision_trees.classification_tree_accuracy(tree, iris.data.T, iris.target)
        
        self.assertAlmostEqual(sklearn_accuracy, pyml_accuracy, 1e-6)


if __name__ == "__main__": 
    unittest.main()