"""Unit tests for clustering module.

(C) 2020-21 Roman Werpachowski.
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

    @classmethod
    def setUpClass(cls):
        cls.seed = 999
        np.random.seed(cls.seed)
        cls.dims = 2
        sample_size = 1000
        face_radius = 1
        ear_radius = 0.3
        ear_angle = np.pi / 4  # How far is ear from top of head.
        cls.num_components = 3
        radii = (face_radius, ear_radius, ear_radius)
        ear_weight = 2
        weights = [face_radius ** 2, ear_weight * ear_radius ** 2, ear_weight * ear_radius ** 2]
        probabilities = weights / np.sum(weights)

        indices = np.random.choice(np.arange(cls.num_components), sample_size, p=probabilities)

        center_xs = [0, (face_radius + ear_radius) * np.sin(-ear_angle), (face_radius + ear_radius) * np.sin(ear_angle)]
        center_ys = [0, (face_radius + ear_radius) * np.cos(-ear_angle), (face_radius + ear_radius) * np.cos(ear_angle)]

        cls.data = np.empty((sample_size, cls.dims))
        for i in range(sample_size):
            k = indices[i]
            phi = np.random.rand() * 2 * np.pi
            r = np.sqrt(np.random.rand()) * radii[k]
            cls.data[i, 0] = center_xs[k] + r * np.cos(phi)
            cls.data[i, 1] = center_ys[k] + r * np.sin(phi)

    def test_em(self):
        abs_tol = 1e-10
        max_iter = 1000
    
        em = clustering.EM(self.num_components)
        em.set_seed(42)
        em.set_absolute_tolerance(abs_tol)
        em.set_relative_tolerance(0)
        em.set_means_initialiser(clustering.KPP())
        em.set_maximum_steps(max_iter)
        converged = int(em.fit(self.data))
        self.assertTrue(converged)
        pyml_ll = em.log_likelihood

        warnings.filterwarnings("error")
        gmm = sklearn.mixture.GaussianMixture(self.num_components, tol=abs_tol, max_iter=max_iter,
                                              random_state=self.seed, n_init=1, reg_covar=1e-15)
        gmm.fit(self.data)  # gmm raises a warning if not converged.
        sklearn_ll = gmm.score(self.data)

        self.assertAlmostEqual(sklearn_ll, pyml_ll, delta=1e-10)

        u = em.assign_responsibilities(np.array([0, 0]))
        self.assertEqual(3, len(u))
        self.assertAlmostEqual(1, sum(u), delta=1e-15)
        self.assertLessEqual(0, min(u))
        # (0, 0) is the middle of face.
        self.assertAlmostEqual(1, max(u), delta=1e-9)

    def test_k_means(self):
        abs_tol = 1e-10
        max_iter = 1000

        km = clustering.KMeans(self.num_components)
        km.set_seed(42)
        km.set_absolute_tolerance(abs_tol)
        km.set_centroids_initialiser(clustering.KPP())
        km.set_maximum_steps(max_iter)
        km.set_number_initialisations(10)
        converged = int(km.fit(self.data))
        self.assertTrue(converged)
        self.assertGreater(km.inertia, 0)
        self.assertEqual(0, min(km.labels))
        self.assertEqual(self.num_components - 1, max(km.labels))
        self.assertTupleEqual((self.num_components, self.dims), km.centroids.shape)
        for i, centroid in enumerate(km.centroids):
            label, distance = km.assign_label(centroid)
            self.assertEqual(i, label)
            self.assertEqual(0, distance)


if __name__ == "__main__": 
    unittest.main()
