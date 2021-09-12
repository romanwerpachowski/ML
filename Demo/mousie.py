"""Demo program for clustering module.

(C) 2020 Roman Werpachowski.
"""
import time
import warnings


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from cppyml import clustering
import sklearn.cluster
import sklearn.mixture
warnings.filterwarnings("default")
import seaborn as sns

def main():
    print("""

*** E-M & K-means CLUSTERING DEMO ***

Uses the classic "mouse" test set. The algorithm should correctly split the face and ears into three separate clusters.
k-means clustering fails on this set.

Compares cppyml with sklearn.

""")
    np.random.seed(1066)
    warnings.filterwarnings("error")

    dims = 2
    sample_size = 1000
    face_radius = 1
    ear_radius = 0.3
    ear_angle = np.pi / 4  # How far is ear from top of head.
    num_components = 3
    radii = (face_radius, ear_radius, ear_radius)
    ear_weight = 2
    weights = [face_radius**2, ear_weight * ear_radius**2, ear_weight * ear_radius**2]
    probabilities = weights / np.sum(weights)
    
    indices = np.random.choice(np.arange(num_components), sample_size, p=probabilities)

    center_xs = [0, (face_radius + ear_radius) * np.sin(-ear_angle), (face_radius + ear_radius) * np.sin(ear_angle)]
    center_ys = [0, (face_radius + ear_radius) * np.cos(-ear_angle), (face_radius + ear_radius) * np.cos(ear_angle)]

    data = np.empty((sample_size, dims))
    for i in range(sample_size):
        k = indices[i]
        phi = np.random.rand() * 2 * np.pi
        r = np.sqrt(np.random.rand()) * radii[k]
        data[i, 0] = center_xs[k] + r * np.cos(phi)
        data[i, 1] = center_ys[k] + r * np.sin(phi)

    abs_tol = 1e-10
    max_iter = 1000
    
    n_timing_iters = 100
    
    cppyml_report = pd.Series(index=["converged", "time", "log-likelihood"], dtype=float)
    em = clustering.EM(num_components)
    em.set_seed(42)
    em.set_absolute_tolerance(abs_tol)
    em.set_relative_tolerance(0)
    em.set_means_initialiser(clustering.KPP())
    em.set_maximum_steps(max_iter)
    t0 = time.perf_counter()
    for _ in range(n_timing_iters):        
        converged = int(em.fit(data))
    cppyml_report["converged"] = float(converged)
    t1 = time.perf_counter()
    cppyml_report["time"] = (t1 - t0) / n_timing_iters
    cppyml_report["log-likelihood"] = em.log_likelihood

    sklearn_report = pd.Series(index=cppyml_report.index, dtype=cppyml_report.dtype)    
    gmm = sklearn.mixture.GaussianMixture(num_components, tol=abs_tol, max_iter=max_iter, random_state=999, n_init=1, reg_covar=1e-15)
    t0 = time.perf_counter()
    for _ in range(n_timing_iters):        
        gmm.fit(data)
    t1 = time.perf_counter()
    sklearn_report["converged"] = 1  # gmm raises a warning if not converged.
    sklearn_report["time"] = (t1 - t0) / n_timing_iters
    sklearn_report["log-likelihood"] = gmm.score(data)
       
    em_report = pd.DataFrame()
    em_report["cppyml"] = cppyml_report
    em_report["sklearn"] = sklearn_report

    print("E-M algorithms")
    print(em_report)

    km = sklearn.cluster.KMeans(num_components, max_iter=max_iter, tol=abs_tol, random_state=1984)
    t0 = time.perf_counter()
    for _ in range(n_timing_iters):
        km.fit(data)
    t1 = time.perf_counter()
    km_class = km.labels_
    km_sklearn_report = pd.Series({"converged": 1, "inertia": km.inertia_, "time": (t1 - t0) / n_timing_iters})

    km_cppyml = clustering.KMeans(num_components)
    km_cppyml.set_maximum_steps(max_iter)
    km_cppyml.set_absolute_tolerance(abs_tol)
    km_cppyml.set_seed(42)
    km_cppyml.set_centroids_initialiser(clustering.KPP())

    t0 = time.perf_counter()
    for _ in range(n_timing_iters):
        converged = km_cppyml.fit(data)
    t1 = time.perf_counter()
    km_cppyml_report = pd.Series({"converged": float(converged), "inertia": km_cppyml.inertia, "time": (t1 - t0) / n_timing_iters})

    km_report = pd.DataFrame()
    km_report["cppyml"] = km_cppyml_report
    km_report["sklearn"] = km_sklearn_report

    print("K-means algorithms")
    print(km_report)
    
    cppyml_class = np.argmax(em.responsibilities, axis=1)
    sklearn_class = gmm.predict(data)

    palette = {
        0: "red",
        1: "green",
        2: "blue"
    }

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=cppyml_class, ax=ax[0], palette=palette)
    ax[0].set_title("cppyml")
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=sklearn_class, ax=ax[1], palette=palette)
    ax[1].set_title("sklearn")
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=km_class, ax=ax[2], palette=palette)
    ax[2].set_title("k-means")
    plt.show()
    fig.savefig("mousie.pdf")


if __name__ == "__main__":
    main()