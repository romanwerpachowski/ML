"""Demo program for decision_trees module.

(C) 2020 Roman Werpachowski.
"""
import time
import warnings

from cppyml import decision_trees
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("default")
import seaborn as sns


def main():
    print("""

*** CLASSIFICATION TREE DEMO ***

Compares cppyml with sklearn.

""")
    np.random.seed(1066)
    warnings.filterwarnings("error")

    min_split_size = 5
    clf = DecisionTreeClassifier(random_state=0, min_samples_split=min_split_size)
    iris = load_iris()

    n_timing_iters = 100    
    
    t0 = time.perf_counter()
    for _ in range(n_timing_iters):        
        clf.fit(iris.data, iris.target)
    t1 = time.perf_counter()
    print("sklearn time: %g" % (t1 - t0))
    print("sklearn accuracy: %g" % clf.score(iris.data, iris.target))

    t0 = time.perf_counter()
    for _ in range(n_timing_iters):        
        tree, _, _ = decision_trees.classification_tree(iris.data, iris.target, alphas=[], min_split_size=min_split_size)
    t1 = time.perf_counter()
    print("cppyml time: %g" % (t1 - t0))
    print("cppyml accuracy: %g" % decision_trees.classification_tree_accuracy(tree, iris.data, iris.target))

if __name__ == "__main__":
    main()