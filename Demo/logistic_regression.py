"""Demo program for logistic_regression module.

(C) 2021 Roman Werpachowski.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cppyml import logistic_regression

def main():
    print("""

*** LINEAR REGRESSION DEMO ***

Times logistic regression against sklearn.

""")
    np.random.seed(1066)
    n_timing_iters = 100
    n = 100
    d = 5

    X = np.random.randn(n, d)
    w = np.random.randn(d)
    z = np.matmul(X, w) + np.random.randn(n) * 0.6
    y = np.sign(z)
    y[y == 0] = 1
    y01 = (1 + y) / 2
    
    cglr = logistic_regression.ConjugateGradientLogisticRegression()
    lam = 0.2
    tol = 1e-7
    cglr .set_lam(lam)
    cglr .set_weight_absolute_tolerance(tol)
    cglr .set_weight_relative_tolerance(tol)    

    t0 = time.perf_counter()
    for _ in range(n_timing_iters):
        result = cglr.fit(X, y)
    t1 = time.perf_counter()

    print("cppyml time: %g" % (t1 - t0))
    print("cppyml result: %s" % result)

    lr = LogisticRegression(tol=tol, C=1/lam)

    t0 = time.perf_counter()    
    for _ in range(n_timing_iters):
        lr.fit(X, y)
    t1 = time.perf_counter()
    print("sklearn.linear_model.LogisticRegression time: %g" % (t1 - t0))
    print("sklearn.linear_model.LogisticRegression result: coef=%s, r2=%g" % (lr.coef_, lr.score(X, y)))


if __name__ == "__main__":
    main()
