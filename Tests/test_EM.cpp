/* (C) 2020 Roman Werpachowski. */
#include <random>
#include <gtest/gtest.h>
#include "ML/Clustering.hpp"
#include "ML/EM.hpp"


static void test_two_gaussians(std::shared_ptr<const ml::Clustering::CentroidsInitialiser> means_initialiser, const bool maximise_first)
{
	std::default_random_engine rng;
	std::uniform_real_distribution<double> u01(0, 1);
	std::normal_distribution<double> standard_normal;
	const unsigned int num_components = 2;
	const unsigned int num_dimensions = 3;
	const unsigned int sample_size = 400;
	constexpr double p0 = 0.25;
	Eigen::MatrixXd means(num_dimensions, num_components);
	means << 0.4, -1.2,
		0.11, 2.2,
		0.5, 1.6;
	Eigen::MatrixXd sigmas(num_dimensions, num_components);
	sigmas << 0.05, 0.2,
		0.04, 0.1,
		0.01, 0.2;
	Eigen::MatrixXd data(num_dimensions, sample_size);

	for (unsigned int i = 0; i < sample_size; ++i) {
		const unsigned int k = u01(rng) < p0 ? 0 : 1;
		const auto mean = means.col(k);
		const auto sigma_vec = sigmas.col(k);
		for (unsigned int l = 0; l < num_dimensions; ++l) {
			data(l, i) = standard_normal(rng) * sigma_vec[l] + mean[l];
		}
	}

	ml::EM em(num_components);
	ASSERT_EQ(num_components, em.number_components());
	em.set_absolute_tolerance(1e-8);
	em.set_relative_tolerance(1e-8);
	em.set_maximum_steps(100);
	if (means_initialiser) {
		em.set_means_initialiser(means_initialiser);
	}
	em.set_maximise_first(maximise_first);
	const unsigned int seed = 63413131;
	em.set_seed(seed);
	ASSERT_TRUE(em.fit(data)) << "EM::fit did not converge";
	ASSERT_EQ(num_components, static_cast<unsigned int>(em.mixing_probabilities().size()));
	ASSERT_EQ(num_components, static_cast<unsigned int>(em.means().cols()));
	ASSERT_EQ(num_dimensions, static_cast<unsigned int>(em.means().rows()));
	ASSERT_EQ(sample_size, static_cast<unsigned int>(em.responsibilities().rows()));
	ASSERT_EQ(num_components, static_cast<unsigned int>(em.responsibilities().cols()));
	const Eigen::MatrixXd means_col_major(em.means());

	Eigen::VectorXd u(num_components);
	for (unsigned int i = 0; i < sample_size; ++i) {		
		em.assign_responsibilities(data.col(i), u);
		ASSERT_NEAR(0, (u - em.responsibilities().row(i).transpose()).norm(), 1e-15) << i;
	}

	std::vector<Eigen::MatrixXd> covariances(num_components);
	for (unsigned int k = 0; k < num_components; ++k) {
		const auto sigma_vec = sigmas.col(k);
		covariances[k].setZero(num_dimensions, num_dimensions);
		for (unsigned int l = 0; l < num_dimensions; ++l) {
			covariances[k](l, l) = std::pow(sigma_vec[l], 2);
		}
	}

	Eigen::VectorXd mixing_probabilities(num_components);
	mixing_probabilities << p0, 1 - p0;

	// EM could have discovered the clusters in either order.
	constexpr bool first_p_lower = p0 < 1 - p0;
	if ((em.mixing_probabilities()[0] < em.mixing_probabilities()[1]) != first_p_lower) {
		std::swap(mixing_probabilities[0], mixing_probabilities[1]);
		means.col(0).swap(means.col(1));
		std::swap(covariances[0], covariances[1]);
	}
	ASSERT_NEAR(0., (mixing_probabilities - em.mixing_probabilities()).norm(), 2e-2) << em.mixing_probabilities();
	ASSERT_NEAR(0., (means - em.means()).norm(), 2e-2) << em.means();
	for (unsigned int k = 0; k < num_components; ++k) {
		ASSERT_NEAR(0., (covariances[k] - em.covariance(k)).norm(), 1e-2) << "Covariance[" << k << "]:\n" << em.covariance(k);
	}

	ml::EM em1(1);
	if (means_initialiser) {
		em1.set_means_initialiser(means_initialiser);
	}
	em1.set_maximise_first(maximise_first);
	em1.fit(data);
	ASSERT_LE(em1.log_likelihood(), em.log_likelihood()) << em1.log_likelihood();
	ASSERT_NEAR(0., (data.rowwise().mean() - em1.means().col(0)).norm(), 1e-14);

	u.resize(1);
	for (unsigned int i = 0; i < sample_size; ++i) {
		em1.assign_responsibilities(data.col(i), u);
		ASSERT_NEAR(0, (u - em1.responsibilities().row(i).transpose()).norm(), 1e-15) << i;
	}
}

TEST(EMTest, two_gaussians_forgy)
{
	test_two_gaussians(std::make_shared<ml::Clustering::Forgy>(), false);
}

TEST(EMTest, two_gaussians_random_partition)
{
	test_two_gaussians(std::make_shared<ml::Clustering::RandomPartition>(), false);
}

TEST(EMTest, two_gaussians_kpp)
{
	test_two_gaussians(std::make_shared<ml::Clustering::KPP>(), false);
}

TEST(EMTest, two_gaussians_closest_mean)
{
	test_two_gaussians(nullptr, true);
}