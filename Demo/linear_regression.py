"""Demo program for linear_regression module."""
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from PyML import linear_regression

def main():
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

    print("PyML time: %g" % (t1 - t0))
    print("PyML result: %s" % result)

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
    print("PyML time: %g" % (t1 - t0))
    print("PyML result: %s" % result)

    lr = LinearRegression(fit_intercept=False)
    t0 = time.perf_counter()
    X = np.reshape(x, (-1, 1))
    for _ in range(n_timing_iters):
        lr.fit(X, y)
    t1 = time.perf_counter()
    print("sklearn.linear_model.LinearRegression time: %g" % (t1 - t0))
    # R2 will be different because LinearRegression uses a different base model.
    print("sklearn.linear_model.LinearRegression result: coef=%s, r2=%g" % (lr.coef_, lr.score(X, y)))

if __name__ == "__main__":
    main()