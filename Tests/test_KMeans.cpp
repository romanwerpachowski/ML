/* (C) 2021 Roman Werpachowski. */
#include <random>
#include <gtest/gtest.h>
#include "ML/Clustering.hpp"
#include "ML/KMeans.hpp"


static void test_two_gaussians(std::shared_ptr<const ml::Clustering::CentroidsInitialiser> centroids_initialiser)
{
	std::default_random_engine rng;
	std::uniform_real_distribution<double> u01(0, 1);
	std::normal_distribution<double> standard_normal;
	const unsigned int num_clusters = 2;
	const unsigned int num_dimensions = 3;
	const unsigned int sample_size = 400;
	constexpr double p0 = 0.25;
	Eigen::MatrixXd centroids(num_dimensions, num_clusters);
	centroids << 0.4, -1.2,
		0.11, 2.2,
		0.5, 1.6;
	Eigen::MatrixXd sigmas(num_dimensions, num_clusters);
	sigmas << 0.05, 0.2,
		0.04, 0.1,
		0.01, 0.2;
	Eigen::MatrixXd data(num_dimensions, sample_size);

	std::vector<unsigned int> ground_truth_labels(sample_size);
	for (unsigned int i = 0; i < sample_size; ++i) {
		const unsigned int k = u01(rng) < p0 ? 0 : 1;
		ground_truth_labels[i] = k;
		const auto mean = centroids.col(k);
		const auto sigma_vec = sigmas.col(k);
		for (unsigned int l = 0; l < num_dimensions; ++l) {
			data(l, i) = standard_normal(rng) * sigma_vec[l] + mean[l];
		}
	}

	ml::Clustering::KMeans km(num_clusters);
	ASSERT_EQ(num_clusters, km.number_clusters());
	km.set_absolute_tolerance(1e-8);
	km.set_maximum_steps(100);
	if (centroids_initialiser) {
		km.set_centroids_initialiser(centroids_initialiser);
	}
	const unsigned int seed = 63413131;
	km.set_seed(seed);
	ASSERT_TRUE(km.fit(data)) << "KMeans::fit did not converge";
	ASSERT_EQ(num_clusters, static_cast<unsigned int>(km.centroids().cols()));
	ASSERT_EQ(num_dimensions, static_cast<unsigned int>(km.centroids().rows()));
	ASSERT_EQ(sample_size, static_cast<unsigned int>(km.labels().size()));	
	const Eigen::MatrixXd centroids_col_major(km.centroids());

	Eigen::VectorXd u(num_clusters);
	double inertia = 0;
	for (unsigned int i = 0; i < sample_size; ++i) {
		const auto label_and_distance = km.assign_label(data.col(i));
		ASSERT_EQ(label_and_distance.first, km.labels()[i]) << i;
		ASSERT_NEAR((km.centroids().col(label_and_distance.first) - data.col(i)).squaredNorm(), label_and_distance.second, 1e-15) << i;
		inertia += label_and_distance.second;
	}
	ASSERT_NEAR(inertia, km.inertia(), 1e-15);

	// KMeans could have discovered the clusters in either order.
	if (ground_truth_labels[0] != km.labels()[0]) {
		for (unsigned int i = 0; i < sample_size; ++i) {
			ground_truth_labels[i] = 1 - ground_truth_labels[i];
		}
		centroids.col(0).swap(centroids.col(1));
	}
	ASSERT_NEAR(0., (centroids - km.centroids()).norm(), 2e-2) << km.centroids();	
	ASSERT_EQ(ground_truth_labels, km.labels());

	// Test multi-init.
	km.set_seed(seed);
	km.set_number_initialisations(3);
	ASSERT_TRUE(km.fit(data)) << "KMeans::fit did not converge";
	ASSERT_LE(km.inertia(), inertia);

	ml::Clustering::KMeans km1(1);
	if (centroids_initialiser) {
		km1.set_centroids_initialiser(centroids_initialiser);
	}
	km1.fit(data);
	ASSERT_NEAR(0., (data.rowwise().mean() - km1.centroids().col(0)).norm(), 1e-14);

	for (unsigned int i = 0; i < sample_size; ++i) {
		ASSERT_EQ(0u, km1.assign_label(data.col(i)).first) << i;
	}
}

TEST(KMeansTest, two_gaussians_forgy)
{
	test_two_gaussians(std::make_shared<ml::Clustering::Forgy>());
}

TEST(KMeansTest, two_gaussians_random_partition)
{
	test_two_gaussians(std::make_shared<ml::Clustering::RandomPartition>());
}

TEST(KMeansTest, two_gaussians_kpp)
{
	test_two_gaussians(std::make_shared<ml::Clustering::KPP>());
}

TEST(KMeansTest, deterministic)
{
	const unsigned int num_clusters = 2;
	const unsigned int num_dimensions = 3;

	Eigen::MatrixXd data(num_dimensions, num_clusters);
	data << -1, 0,
		1, 0.5,
		0.5, 0.5;

	ml::Clustering::KMeans km(num_clusters);

	ASSERT_TRUE(km.fit(data));

	ASSERT_EQ(0., km.inertia());

	for (unsigned int i = 0; i < num_clusters; ++i) {
		ASSERT_EQ(i, km.labels()[i]) << i;
		ASSERT_EQ(0, (km.centroids().col(i) - data.col(i)).norm()) << i;
	}
}