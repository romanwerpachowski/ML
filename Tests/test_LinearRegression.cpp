#include <random>
#include <gtest/gtest.h>
#include "ML/LinearRegression.hpp"
#include "ML/Statistics.hpp"

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
	ASSERT_TRUE(std::isnan(result.var_y)) << result.var_y;
	ASSERT_TRUE(std::isnan(result.var_slope)) << result.var_slope;
	ASSERT_TRUE(std::isnan(result.var_intercept)) << result.var_intercept;
}

TEST(LinearRegression, univariate_two_points_regular)
{
	Eigen::Vector2d y(0.5, 0.3);
	const UnivariateLinearRegressionResult result = univariate(.1, .1, y);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_NEAR(.7, result.intercept, 1e-15);
	ASSERT_NEAR(-1, result.correlation, 1e-15);
	ASSERT_NEAR(1, result.r2, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y)) << result.var_y;
	ASSERT_TRUE(std::isnan(result.var_slope)) << result.var_slope;
	ASSERT_TRUE(std::isnan(result.var_intercept)) << result.var_intercept;
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
	EXPECT_NEAR(expected_observation_variance, result.var_y, expected_observation_variance * 1e-3);
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
	EXPECT_NEAR(expected_observation_variance, result.var_y, expected_observation_variance * 1e-3);
}

TEST(LinearRegression, univariate_true_model)
{
	const double noise_std_dev = 0.1;
	const double x_range = 5; // width of the X range
	const double slope = -0.6;
	const double intercept = 1.2;
	const unsigned int n = 1000;
	std::default_random_engine rng(784957984);
	std::normal_distribution<double> noise_dist(0, noise_std_dev);
	std::uniform_real_distribution<double> x_dist(0, x_range);	
	Eigen::VectorXd x(n);
	Eigen::VectorXd y(n);
	for (unsigned int i = 0; i < n; ++i) {
		x[i] = x_dist(rng);
	}
	auto sample_noise_and_run_regression = [&]() -> ml::LinearRegression::UnivariateLinearRegressionResult {				
		for (unsigned int i = 0; i < n; ++i) {
			y[i] = intercept + slope * x[i] + noise_dist(rng);
		}
		return univariate(x, y);
	};
	const auto result = sample_noise_and_run_regression();
	EXPECT_NEAR(intercept, result.intercept, 2e-3);
	EXPECT_NEAR(slope, result.slope, 1e-3);	
	const double x_var = x_range * x_range / 12;
	const double noise_var = noise_std_dev * noise_std_dev;
	const double y_var = x_var * slope * slope + noise_var;
	const double xy_cov = slope * x_var;
	const double xy_corr = xy_cov / std::sqrt(x_var * y_var);
	EXPECT_NEAR(xy_corr, result.correlation, 3e-4);
	EXPECT_NEAR(xy_corr * xy_corr, result.r2, 5e-4);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y, 6e-4);

	// Calculate sample statistics of estimators.
	const unsigned int n_samples = 1000;
	std::vector<double> intercepts(n_samples);
	std::vector<double> slopes(n_samples);	
	for (unsigned int i = 0; i < n_samples; ++i) {
		const auto result_i = sample_noise_and_run_regression();
		intercepts[i] = result_i.intercept;
		slopes[i] = result_i.slope;		
	}
	const auto slope_sse_and_mean = ml::Statistics::sse_and_mean(slopes.begin(), slopes.end());
	EXPECT_NEAR(slope_sse_and_mean.first / (n_samples - 1), result.var_slope, 1e-7);
	EXPECT_NEAR(slope_sse_and_mean.second, result.slope, 2e-3);
	const auto intercept_sse_and_mean = ml::Statistics::sse_and_mean(intercepts.begin(), intercepts.end());
	EXPECT_NEAR(intercept_sse_and_mean.first / (n_samples - 1), result.var_intercept, 2e-6);
	EXPECT_NEAR(intercept_sse_and_mean.second, result.intercept, 2e-3);
	const auto covariance = ml::Statistics::covariance(slopes, intercepts);
	EXPECT_NEAR(covariance, result.cov_slope_intercept, 1e-7);
}

TEST(LinearRegression, univariate_true_model_regular)
{
	const double noise_std_dev = 0.1;
	const double x_range = 5; // width of the X range
	const double slope = -0.6;
	const double intercept = 1.2;
	const unsigned int n = 1000;
	const double dx = x_range / (n - 1);
	const double x0 = 0;
	std::default_random_engine rng(784957984);
	std::normal_distribution<double> noise_dist(0, noise_std_dev);
	Eigen::VectorXd y(n);
	auto sample_noise_and_run_regression = [&]() -> ml::LinearRegression::UnivariateLinearRegressionResult {
		for (unsigned int i = 0; i < n; ++i) {
			y[i] = intercept + slope * (x0 + i * dx) + noise_dist(rng);
		}
		return univariate(x0, dx, y);
	};
	const auto result = sample_noise_and_run_regression();
	EXPECT_NEAR(intercept, result.intercept, 1e-2);
	EXPECT_NEAR(slope, result.slope, 3e-3);
	const double x_var = x_range * x_range / 12;
	const double noise_var = noise_std_dev * noise_std_dev;
	const double y_var = x_var * slope * slope + noise_var;
	const double xy_cov = slope * x_var;
	const double xy_corr = xy_cov / std::sqrt(x_var * y_var);
	EXPECT_NEAR(xy_corr, result.correlation, 2e-4);
	EXPECT_NEAR(xy_corr * xy_corr, result.r2, 4e-4);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y, 3e-4);

	// Calculate sample statistics of estimators.
	const unsigned int n_samples = 1000;
	std::vector<double> intercepts(n_samples);
	std::vector<double> slopes(n_samples);
	for (unsigned int i = 0; i < n_samples; ++i) {
		const auto result_i = sample_noise_and_run_regression();
		intercepts[i] = result_i.intercept;
		slopes[i] = result_i.slope;
	}
	const auto slope_sse_and_mean = ml::Statistics::sse_and_mean(slopes.begin(), slopes.end());
	EXPECT_NEAR(slope_sse_and_mean.first / (n_samples - 1), result.var_slope, 2e-7);
	EXPECT_NEAR(slope_sse_and_mean.second, result.slope, 3e-3);
	const auto intercept_sse_and_mean = ml::Statistics::sse_and_mean(intercepts.begin(), intercepts.end());
	EXPECT_NEAR(intercept_sse_and_mean.first / (n_samples - 1), result.var_intercept, 3e-7);
	EXPECT_NEAR(intercept_sse_and_mean.second, result.intercept, 1e-2);
	const auto covariance = ml::Statistics::covariance(slopes, intercepts);
	EXPECT_NEAR(covariance, result.cov_slope_intercept, 2e-7);
}