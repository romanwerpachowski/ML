#include <random>
#include <gtest/gtest.h>
#include "ML/LinearRegression.hpp"

using namespace ml::LinearRegression;


TEST(LinearRegression, univariate_errors)
{
	Eigen::VectorXd x(4);
	Eigen::VectorXd y(3);
	ASSERT_THROW(univariate(x, y), std::invalid_argument);
	x.resize(1);
	y.resize(1);
	ASSERT_THROW(univariate(x, y), std::invalid_argument);
}

TEST(LinearRegression, univariate_two_points)
{
	Eigen::Vector2d x(0.1, 0.2);
	Eigen::Vector2d y(0.5, 0.3);
	const UnivariateLinearRegressionResult result = univariate(x, y);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_NEAR(0.7, result.intercept, 1e-15);
	ASSERT_NEAR(-1, result.correlation, 1e-15);
	ASSERT_NEAR(1, result.r2, 1e-15);
	ASSERT_TRUE(std::isnan(result.observation_variance_estimate)) << result.observation_variance_estimate;
}

TEST(LinearRegression, univariate_two_points_regular)
{
	Eigen::Vector2d y(0.5, 0.3);
	const UnivariateLinearRegressionResult result = univariate(.1, .1, y);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_NEAR(.7, result.intercept, 1e-15);
	ASSERT_NEAR(-1, result.correlation, 1e-15);
	ASSERT_NEAR(1, result.r2, 1e-15);
	ASSERT_TRUE(std::isnan(result.observation_variance_estimate)) << result.observation_variance_estimate;
}

TEST(LinearRegression, univariate_high_noise)
{
	const int n = 10000;
	Eigen::VectorXd x(n);
	Eigen::VectorXd y(n);
	const double slope = 0;
	const double intercept = 1e-6;
	const double noise_strength = 1e4;
	std::default_random_engine rng(784957984);
	std::uniform_int_distribution uniform(0, 1);
	for (int i = 0; i < n; ++i) {
		const double x_i = static_cast<double>(i);
		x[i] = x_i;
		y[i] = noise_strength * (0.5 - static_cast<double>(uniform(rng))) + intercept + slope * x_i;
	}
	const auto result = univariate(x, y);
	EXPECT_NEAR(slope, result.slope, noise_strength * 1e-2);
	EXPECT_NEAR(intercept, result.intercept, 2e-2 * noise_strength);
	EXPECT_NEAR(0, result.correlation, 1e-2);
	EXPECT_NEAR(0, result.r2, 1e-4);
	const double expected_observation_variance = noise_strength * noise_strength / 4;
	EXPECT_NEAR(expected_observation_variance, result.observation_variance_estimate, expected_observation_variance * 1e-3);
}

TEST(LinearRegression, univariate_high_noise_regular)
{
	const int n = 10000;
	Eigen::VectorXd y(n);
	const double x0 = 0;
	const double dx = 1;
	const double slope = 0;
	const double intercept = 1e-6;
	const double noise_strength = 1e4;
	std::default_random_engine rng(784957984);
	std::uniform_int_distribution uniform(0, 1);
	for (int i = 0; i < n; ++i) {
		const double x_i = x0 + static_cast<double>(i) * dx;
		y[i] = noise_strength * (0.5 - static_cast<double>(uniform(rng))) + intercept + slope * x_i;
	}
	const auto result = univariate(x0, dx, y);
	EXPECT_NEAR(slope, result.slope, noise_strength * 1e-2);
	EXPECT_NEAR(intercept, result.intercept, 2e-2 * noise_strength);
	EXPECT_NEAR(0, result.correlation, 1e-2);
	EXPECT_NEAR(0, result.r2, 1e-4);
	const double expected_observation_variance = noise_strength * noise_strength / 4;
	EXPECT_NEAR(expected_observation_variance, result.observation_variance_estimate, expected_observation_variance * 1e-3);
}

TEST(LinearRegression, univariate_true_model)
{
	const double noise_std_dev = 0.96;
	const double x_range = 5; // width of the X range
	const double slope = -0.6;
	const double intercept = 1.2;
	const unsigned int n = 1000000;
	std::default_random_engine rng(784957984);
	std::normal_distribution<double> noise_dist(0, noise_std_dev);
	std::uniform_real_distribution<double> x_dist(0, x_range);
	Eigen::VectorXd x(n);
	Eigen::VectorXd y(n);
	for (unsigned int i = 0; i < n; ++i) {
		x[i] = x_dist(rng);
		y[i] = intercept + slope * x[i] + noise_dist(rng);
	}
	const auto result = univariate(x, y);
	const double tol = 2e-3;
	EXPECT_NEAR(intercept, result.intercept, tol);
	EXPECT_NEAR(slope, result.slope, tol);	
	const double x_var = x_range * x_range / 12;
	const double noise_var = noise_std_dev * noise_std_dev;
	const double y_var = x_var * slope * slope + noise_var;
	const double xy_cov = slope * x_var;
	const double xy_corr = xy_cov / std::sqrt(x_var * y_var);
	EXPECT_NEAR(xy_corr, result.correlation, tol);
	EXPECT_NEAR(xy_corr * xy_corr, result.r2, tol);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.observation_variance_estimate, tol);
}

TEST(LinearRegression, univariate_true_model_regular)
{
	const double noise_std_dev = 0.96;
	const double x_range = 5; // width of the X range
	const double slope = -0.6;
	const double intercept = 1.2;
	const unsigned int n = 1000000;
	const double dx = x_range / (n - 1);
	const double x0 = 0;
	std::default_random_engine rng(784957984);
	std::normal_distribution<double> noise_dist(0, noise_std_dev);
	Eigen::VectorXd y(n);
	for (unsigned int i = 0; i < n; ++i) {
		y[i] = intercept + slope * (x0 + i * dx) + noise_dist(rng);
	}
	const auto result = univariate(x0, dx, y);
	const double tol = 5e-4;
	EXPECT_NEAR(intercept, result.intercept, tol);
	EXPECT_NEAR(slope, result.slope, tol);
	const double x_var = x_range * x_range / 12;
	const double noise_var = noise_std_dev * noise_std_dev;
	const double y_var = x_var * slope * slope + noise_var;
	const double xy_cov = slope * x_var;
	const double xy_corr = xy_cov / std::sqrt(x_var * y_var);
	EXPECT_NEAR(xy_corr, result.correlation, tol);
	EXPECT_NEAR(xy_corr * xy_corr, result.r2, tol);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.observation_variance_estimate, 2e-3);
}