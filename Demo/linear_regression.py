"""Demo program for linear_regression module.

(C) 2020 Roman Werpachowski.
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge

from cppyml import linear_regression

def main():
    print("""

*** LINEAR REGRESSION DEMO ***

Times different versions of linear regression against standard Python libraries (scipy and sklearn).

""")
    np.random.seed(1066)
    n_timing_iters = 100
    n = 25
    slope = 0.1
    intercept = -0.9
    noise = 0.1
    warnings.filterwarnings("error")

    print("*** Univariate with intercept ***")
    x = np.random.randn(n)
    y = x * slope + intercept + noise * np.random.randn(n)

    t0 = time.perf_counter()
    for _ in range(n_timing_iters):
        result = linear_regression.univariate(x, y)
    t1 = time.perf_counter()

    print("cppyml time: %g" % (t1 - t0))
    print("cppyml result: %s" % result)

    t0 = time.perf_counter()
    for _ in range(n_timing_iters):
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
    t1 = time.perf_counter()
    print("scipy.stats.linregress time: %g" % (t1 - t0))
    print("scipy.stats.linregress result: slope=%g, intercept=%g, r2=%g, var_slope=%g" % (slope, intercept, rvalue**2, stderr**2))

    print("\n*** Univariate without intercept ***")    
    y = x * slope + noise * np.random.randn(n)

    t0 = time.perf_counter()
    for _ in range(n_timing_iters):
        result = linear_regression.univariate_without_intercept(x, y)
    t1 = time.perf_counter()
    print("cppyml time: %g" % (t1 - t0))
    print("cppyml result: %s" % result)

    lr = LinearRegression(fit_intercept=False)
    X = np.reshape(x, (-1, 1))
    t0 = time.perf_counter()    
    for _ in range(n_timing_iters):
        lr.fit(X, y)
    t1 = time.perf_counter()
    print("sklearn.linear_model.LinearRegression time: %g" % (t1 - t0))
    # R2 will be different because LinearRegression uses a different base model.
    print("sklearn.linear_model.LinearRegression result: coef=%s, r2=%g" % (lr.coef_, lr.score(X, y)))

    print("\n*** Multivariate ***")
    d = 4
    X = np.empty((n, d + 1))
    X[:,:d] = np.random.randn(n, d)
    X[:, d] = 1
    beta = np.random.rand(d + 1)
    y = np.matmul(X, beta) + 0.4 * np.random.randn(n)

    t0 = time.perf_counter()
    for _ in range(n_timing_iters):
        result = linear_regression.multivariate(X, y)
    t1 = time.perf_counter()
    print("cppyml time: %g" % (t1 - t0))
    print("cppyml result: %s" % result)

    t0 = time.perf_counter()    
    for _ in range(n_timing_iters):
        lr.fit(X, y)
    t1 = time.perf_counter()
    print("sklearn.linear_model.LinearRegression time: %g" % (t1 - t0))
    print("sklearn.linear_model.LinearRegression result: coef=%s, r2=%g" % (lr.coef_, lr.score(X, y)))

    print("\n*** Multivariate w/ ridge regression - already standardised ***")
    
    lam = 0.01
    X = X[:, :d]
    X = X - np.mean(X, axis=0)
    X = X / np.std(X, axis=0, ddof=0)
    ridge = Ridge(alpha=lam, fit_intercept=True, normalize=False)

    t0 = time.perf_counter()
    for _ in range(n_timing_iters):
        result = linear_regression.ridge(X, y, lam, do_standardise=False)
    t1 = time.perf_counter()
    print("cppyml time: %g" % (t1 - t0))
    print("cppyml result: %s" % result)

    t0 = time.perf_counter()    
    for _ in range(n_timing_iters):
        ridge.fit(X, y)
    t1 = time.perf_counter()
    print("sklearn.linear_model.Ridge time: %g" % (t1 - t0))
    print("sklearn.linear_model.Ridge result: coef=%s, intercept=%g, r2=%g" % (ridge.coef_, ridge.intercept_, ridge.score(X, y)))

if __name__ == "__main__":
    main()