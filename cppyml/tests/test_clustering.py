"""Unit tests for clustering module.

(C) 2020 Roman Werpachowski.
"""
import unittest
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.mixture

from cppyml import clustering


class ClusteringTest(unittest.TestCase):

    def test_mousie(self):
        seed = 999
        np.random.seed(seed)
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
    
        em = clustering.EM(num_components)
        em.set_seed(42)
        em.set_absolute_tolerance(abs_tol)
        em.set_relative_tolerance(0)
        em.set_means_initialiser(clustering.KPP())
        em.set_maximum_steps(max_iter)
        converged = int(em.fit(data))
        self.assertTrue(converged)
        pyml_ll = em.log_likelihood

        warnings.filterwarnings("error")
        gmm = sklearn.mixture.GaussianMixture(num_components, tol=abs_tol, max_iter=max_iter, random_state=seed, n_init=1, reg_covar=1e-15)
        gmm.fit(data)  # gmm raises a warning if not converged.
        sklearn_ll = gmm.score(data)

        self.assertAlmostEqual(sklearn_ll, pyml_ll, delta=1e-10)


if __name__ == "__main__": 
    unittest.main()