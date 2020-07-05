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
}

TEST(LinearRegression, univariate_two_points_regular)
{
	Eigen::Vector2d y(0.5, 0.3);
	const UnivariateLinearRegressionResult result = univariate(.1, .1, y);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_NEAR(.7, result.intercept, 1e-15);
	ASSERT_NEAR(-1, result.correlation, 1e-15);
	ASSERT_NEAR(1, result.r2, 1e-15);
}

TEST(LinearRegression, univariate_high_noise)
{
	const int n = 10000;
	Eigen::VectorXd x(n);
	Eigen::VectorXd y(n);
	const double slope = 0;
	const double intercept = 1e-6;
	const double noise_strength = 1e4;
	for (int i = 0; i < n; ++i) {
		const double x_i = static_cast<double>(i);
		x[i] = x_i;
		y[i] = noise_strength * (0.5 - static_cast<double>(i % 2)) + intercept + slope * x_i;
	}
	const auto result = univariate(x, y);
	EXPECT_NEAR(slope, result.slope, 5e-4);
	EXPECT_NEAR(intercept, result.intercept, 2e-4 * noise_strength);
	EXPECT_NEAR(0, result.correlation, 2e-4);
	EXPECT_NEAR(0, result.r2, 5e-8);
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
	for (int i = 0; i < n; ++i) {
		const double x_i = x0 + static_cast<double>(i) * dx;
		y[i] = noise_strength * (0.5 - static_cast<double>(i % 2)) + intercept + slope * x_i;
	}
	const auto result = univariate(x0, dx, y);
	EXPECT_NEAR(slope, result.slope, 5e-4);
	EXPECT_NEAR(intercept, result.intercept, 2e-4 * noise_strength);
	EXPECT_NEAR(0, result.correlation, 2e-4);
	EXPECT_NEAR(0, result.r2, 5e-8);
}