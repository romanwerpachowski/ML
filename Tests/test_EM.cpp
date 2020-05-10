#include "pch.h"
#include <random>
#include "ML/EM.hpp"

TEST(TestEM, two_gaussians)
{
	std::default_random_engine rng;
	std::uniform_real_distribution<double> u01(0, 1);
	std::normal_distribution<double> standard_normal;
	const unsigned int num_components = 2;
	const unsigned int num_dimensions = 3;
	const unsigned int sample_size = 400;
	const double p0 = 0.25;
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
	em.set_maximum_steps(10);
	ASSERT_TRUE(em.fit(data)) << "EM did not converge";
	ASSERT_EQ(num_components, em.mixing_probabilities().size());
	ASSERT_EQ(num_components, em.means().cols());
	ASSERT_EQ(num_dimensions, em.means().rows());

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
	const bool first_p_lower = p0 < 1 - p0;
	if ((em.mixing_probabilities()[0] < em.mixing_probabilities()[1]) != first_p_lower) {
		std::swap(mixing_probabilities[0], mixing_probabilities[1]);
		means.col(0).swap(means.col(1));
		std::swap(covariances[0], covariances[1]);
	}
	ASSERT_NEAR(0., (mixing_probabilities - em.mixing_probabilities()).norm(), 1e-2) << em.mixing_probabilities();		
	ASSERT_NEAR(0., (means - em.means()).norm(), 2e-2) << em.means();
	for (unsigned int k = 0; k < num_components; ++k) {
		ASSERT_NEAR(0., (covariances[k] - em.covariance(k)).norm(), 1e-2) << "Covariance[" << k << "]:\n" << em.covariance(k);
	}

	ml::EM em1(1);
	em1.fit(data);
	ASSERT_LE(em1.log_likelihood(), em.log_likelihood()) << em1.log_likelihood();
	ASSERT_NEAR(0., (data.rowwise().mean() - em1.means().col(0)).norm(), 1e-14);
}