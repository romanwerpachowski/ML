#include "pch.h"
#include <random>
#include "ML/EM.hpp"

TEST(TestEM, two_gaussians)
{
	std::default_random_engine rng;
	std::uniform_real_distribution<double> u01(0, 1);
	std::normal_distribution<double> standard_normal;
	const double abs_tol = 1e-8;
	const double rel_tol = 1e-8;
	const unsigned int num_components = 2;
	const unsigned int num_dimensions = 3;
	const unsigned int sample_size = 100;
	const double p0 = 0.4;
	Eigen::MatrixXd means(num_dimensions, num_components);
	means << 0.4, -1.2,
		0.11, 2.2,
		0.5, 1.6;
	Eigen::MatrixXd sigmas(num_dimensions, num_components);
	sigmas << 0.05, 0.8,
		0.04, 0.6,
		0.01, 0.4;
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
	em.set_absolute_tolerance(abs_tol);
	em.set_relative_tolerance(rel_tol);
	em.set_verbose(true);
	EXPECT_TRUE(em.fit(data)) << "EM did not converge";

	std::vector<Eigen::MatrixXd> covariances(num_components, Eigen::MatrixXd(num_dimensions, num_dimensions));
	for (unsigned int k = 0; k < num_components; ++k) {
		const auto sigma_vec = sigmas.col(k);
		for (unsigned int l = 0; l < num_dimensions; ++l) {
			covariances[k](l, l) = std::pow(sigma_vec[l], 2);
		}
	}

	Eigen::VectorXd mixing_probabilities(num_components);
	mixing_probabilities << p0, 1 - p0;
	EXPECT_EQ(num_components, em.mixing_probabilities().size());
	EXPECT_NEAR(0., (mixing_probabilities - em.mixing_probabilities()).norm(), 1e-14) << em.mixing_probabilities();	
	EXPECT_EQ(num_components, em.means().cols());
	EXPECT_EQ(num_dimensions, em.means().rows());
	EXPECT_NEAR(0., (means - em.means()).norm(), 1e-14) << em.means();
}