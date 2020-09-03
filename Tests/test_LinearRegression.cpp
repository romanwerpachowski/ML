/* (C) 2020 Roman Werpachowski. */
#include <random>
#include <gtest/gtest.h>
#include "ML/LinearRegression.hpp"
#include "ML/RecursiveMultivariateOLS.hpp"
#include "ML/Statistics.hpp"

using namespace ml::LinearRegression;


class LinearRegressionTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		srand(42); // Eigen++ uses old RNG API.
	}
};

inline double calc_sse(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y, const double slope, const double intercept)
{
	return (y.array() - slope * x.array() - intercept).square().sum();
}

static double calc_sse(const double x0, const double dx, Eigen::Ref<const Eigen::VectorXd> y, const double slope, const double intercept)
{
	double sse = 0;
	for (Eigen::Index i = 0; i < y.size(); ++i) {
		sse += std::pow(y[i] - slope * (x0 + static_cast<double>(i) * dx) - intercept, 2);
	}
	return sse;
}

inline double calc_sse(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> beta)
{
	return (y - X.transpose() * beta).squaredNorm();
}

inline double calc_sse(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, const double lambda, Eigen::Ref<const Eigen::VectorXd> beta)
{
	const auto d = X.rows();
	return (y - X.transpose() * beta.head(d) - Eigen::VectorXd::Constant(y.size(), beta[d])).squaredNorm() + lambda * beta.head(d).squaredNorm();
}

static void test_sse_minimisation(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y, const double slope, const double intercept, const double slope_delta, const double intercept_delta)
{
	const double min_sse = calc_sse(x, y, slope, intercept);
	if (slope_delta > 0) {
		ASSERT_LE(min_sse, calc_sse(x, y, slope + slope_delta, intercept));
		ASSERT_LE(min_sse, calc_sse(x, y, slope - slope_delta, intercept));
	}
	if (intercept_delta > 0) {
		ASSERT_LE(min_sse, calc_sse(x, y, slope, intercept + intercept_delta));
		ASSERT_LE(min_sse, calc_sse(x, y, slope, intercept - intercept_delta));
	}
}

static void test_sse_minimisation(const double x0, const double dx, Eigen::Ref<const Eigen::VectorXd> y, const double slope, const double intercept, const double slope_delta, const double intercept_delta)
{
	const double min_sse = calc_sse(x0, dx, y, slope, intercept);
	if (slope_delta > 0) {
		ASSERT_LE(min_sse, calc_sse(x0, dx, y, slope + slope_delta, intercept));
		ASSERT_LE(min_sse, calc_sse(x0, dx, y, slope - slope_delta, intercept));
	}
	if (intercept_delta > 0) {
		ASSERT_LE(min_sse, calc_sse(x0, dx, y, slope, intercept + intercept_delta));
		ASSERT_LE(min_sse, calc_sse(x0, dx, y, slope, intercept - intercept_delta));
	}
}

static void test_sse_minimisation(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> beta, Eigen::Ref<const Eigen::VectorXd> beta_deltas)
{
	const double min_sse = calc_sse(X, y, beta);
	for (Eigen::Index i = 0; i < beta.size(); ++i) {
		const double delta = beta_deltas[i];
		if (delta > 0) {
			const auto v_delta = Eigen::VectorXd::Unit(beta.size(), i) * delta;
			ASSERT_LE(min_sse, calc_sse(X, y, beta + v_delta));
			ASSERT_LE(min_sse, calc_sse(X, y, beta - v_delta));
		}		
	}
}

static void test_sse_minimisation(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, const double lambda, Eigen::Ref<const Eigen::VectorXd> beta, Eigen::Ref<const Eigen::VectorXd> beta_deltas)
{
	const double min_sse = calc_sse(X, y, lambda, beta);
	for (Eigen::Index i = 0; i < beta.size(); ++i) {
		const double delta = beta_deltas[i];
		if (delta > 0) {
			const auto v_delta = Eigen::VectorXd::Unit(beta.size(), i) * delta;
			ASSERT_LE(min_sse, calc_sse(X, y, lambda, beta + v_delta));
			ASSERT_LE(min_sse, calc_sse(X, y, lambda, beta - v_delta));
		}
	}	
}

TEST_F(LinearRegressionTest, univariate_errors)
{
	Eigen::VectorXd x(4);
	Eigen::VectorXd y(3);
	ASSERT_THROW(univariate(x, y), std::invalid_argument);
	x.resize(1);
	y.resize(1);
	ASSERT_THROW(univariate(x, y), std::invalid_argument);
}

TEST_F(LinearRegressionTest, univariate_two_points)
{
	Eigen::Vector2d x(0.1, 0.2);
	Eigen::Vector2d y(0.5, 0.3);
	const UnivariateOLSResult result = univariate(x, y);
	ASSERT_EQ(2u, result.n);
	ASSERT_EQ(0u, result.dof);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_NEAR(0.7, result.intercept, 1e-15);
	ASSERT_NEAR(1, result.r2, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y)) << result.var_y;
	ASSERT_TRUE(std::isnan(result.var_slope)) << result.var_slope;
	ASSERT_TRUE(std::isnan(result.var_intercept)) << result.var_intercept;
	ASSERT_TRUE(std::isnan(result.cov_slope_intercept)) << result.cov_slope_intercept;
	test_sse_minimisation(x, y, result.slope, result.intercept, 1e-8, 1e-8);
}

TEST_F(LinearRegressionTest, univariate_two_points_regular)
{
	constexpr double x0 = 0.1;
	constexpr double dx = 0.1;
	Eigen::Vector2d y(0.5, 0.3);
	const UnivariateOLSResult result = univariate(x0, dx, y);
	ASSERT_EQ(2u, result.n);
	ASSERT_EQ(0u, result.dof);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_NEAR(.7, result.intercept, 1e-15);
	ASSERT_NEAR(1, result.r2, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y)) << result.var_y;
	ASSERT_TRUE(std::isnan(result.var_slope)) << result.var_slope;
	ASSERT_TRUE(std::isnan(result.var_intercept)) << result.var_intercept;
	ASSERT_TRUE(std::isnan(result.cov_slope_intercept)) << result.cov_slope_intercept;
	test_sse_minimisation(x0, dx, y, result.slope, result.intercept, 1e-8, 1e-8);
}

TEST_F(LinearRegressionTest, univariate_high_noise)
{
	const unsigned int n = 10000;
	Eigen::VectorXd x(n);
	Eigen::VectorXd y(n);
	const double intercept = 1e-6;
	const double noise_strength = 1e4;
	std::default_random_engine rng(784957984);
	std::uniform_int_distribution uniform(0, 1);
	for (unsigned int i = 0; i < n; ++i) {
		const double x_i = static_cast<double>(i);
		x[i] = x_i;
		y[i] = noise_strength * (0.5 - static_cast<double>(uniform(rng))) + intercept;
	}
	const auto result = univariate(x, y);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 2, result.dof);
	EXPECT_NEAR(0, result.slope, noise_strength * 1e-2);
	EXPECT_NEAR(intercept, result.intercept, 2e-2 * noise_strength);
	EXPECT_NEAR(0, result.r2, 3e-4);
	EXPECT_GE(result.r2, 0);
	const double expected_observation_variance = noise_strength * noise_strength / 4;
	EXPECT_NEAR(expected_observation_variance, result.var_y, expected_observation_variance * 1e-3);
	test_sse_minimisation(x, y, result.slope, result.intercept, 1e-6, 1e-6);
}

TEST_F(LinearRegressionTest, univariate_regular_vs_standard)
{
	constexpr unsigned int n = 100;
	constexpr double x0 = -1.4;
	constexpr double dx = 0.23;
	Eigen::VectorXd x(n);
	Eigen::VectorXd y(n);
	constexpr double intercept = 0.3;
	constexpr double noise_strength = 0.1;
	std::default_random_engine rng(784957984);
	std::uniform_int_distribution uniform(0, 1);
	for (unsigned int i = 0; i < n; ++i) {
		const double x_i = x0 + i * dx;
		x[i] = x_i;
		y[i] = noise_strength * (0.5 - static_cast<double>(uniform(rng))) + intercept;
	}
	const auto r1 = univariate(x, y);
	const auto r2 = univariate(x0, dx, y);
	constexpr double tol = 1e-15;
	ASSERT_NEAR(r1.slope, r2.slope, tol);
	ASSERT_NEAR(r1.intercept, r2.intercept, tol);
	ASSERT_NEAR(r1.var_y, r2.var_y, tol);
	ASSERT_NEAR(r1.r2, r2.r2, tol);
	ASSERT_NEAR(r1.var_intercept, r2.var_intercept, tol);
	ASSERT_NEAR(r1.var_slope, r2.var_slope, tol);
	ASSERT_NEAR(r1.cov_slope_intercept, r2.cov_slope_intercept, tol);
}

TEST_F(LinearRegressionTest, univariate_high_noise_regular)
{
	const unsigned int n = 10000;
	Eigen::VectorXd y(n);
	const double x0 = 0;
	const double dx = 1;
	const double intercept = 1e-6;
	const double noise_strength = 1e4;
	std::default_random_engine rng(784957984);
	std::uniform_int_distribution uniform(0, 1);
	for (unsigned int i = 0; i < n; ++i) {
		y[i] = noise_strength * (0.5 - static_cast<double>(uniform(rng))) + intercept;
	}
	const auto result = univariate(x0, dx, y);	
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 2, result.dof);
	EXPECT_NEAR(0, result.slope, noise_strength * 1e-2);
	EXPECT_NEAR(intercept, result.intercept, 2e-2 * noise_strength);
	EXPECT_NEAR(0, result.r2, 3e-4);
	EXPECT_GE(result.r2, 0);
	const double expected_observation_variance = noise_strength * noise_strength / 4;
	EXPECT_NEAR(expected_observation_variance, result.var_y, expected_observation_variance * 1e-3);
	test_sse_minimisation(x0, dx, y, result.slope, result.intercept, 1e-6, 1e-6);
}

TEST_F(LinearRegressionTest, univariate_true_model)
{
	const double noise_std_dev = 0.2;
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
	auto sample_noise_and_run_regression = [&]() -> UnivariateOLSResult {
		for (unsigned int i = 0; i < n; ++i) {
			y[i] = intercept + slope * x[i] + noise_dist(rng);
		}
		return univariate(x, y);
	};
	const auto result = sample_noise_and_run_regression();
	test_sse_minimisation(x, y, result.slope, result.intercept, 1e-8, 1e-8);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 2, result.dof);
	EXPECT_NEAR(intercept, result.intercept, 4e-3);
	EXPECT_NEAR(slope, result.slope, 4e-3);	
	const double x_var = x_range * x_range / 12;
	const double noise_var = noise_std_dev * noise_std_dev;
	const double y_var = x_var * slope * slope + noise_var;
	const double xy_cov = slope * x_var;
	const double xy_corr = xy_cov / std::sqrt(x_var * y_var);
	EXPECT_NEAR(xy_corr * xy_corr, result.r2, 2e-3);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y, 3e-3);

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
	EXPECT_NEAR(slope_sse_and_mean.first / (n_samples - 1), result.var_slope, 4e-7);
	EXPECT_NEAR(slope_sse_and_mean.second, result.slope, 4e-3);
	const auto intercept_sse_and_mean = ml::Statistics::sse_and_mean(intercepts.begin(), intercepts.end());
	EXPECT_NEAR(intercept_sse_and_mean.first / (n_samples - 1), result.var_intercept, 8e-6);
	EXPECT_NEAR(intercept_sse_and_mean.second, result.intercept, 3e-3);
	const auto covariance = ml::Statistics::covariance(slopes, intercepts);
	EXPECT_NEAR(covariance, result.cov_slope_intercept, 4e-7);
}

TEST_F(LinearRegressionTest, univariate_true_model_regular)
{
	const double noise_std_dev = 0.2;
	const double x_range = 5; // width of the X range
	const double slope = -0.6;
	const double intercept = 1.2;
	const unsigned int n = 1000;
	const double dx = x_range / (n - 1);
	const double x0 = 0;
	std::default_random_engine rng(784957984);
	std::normal_distribution<double> noise_dist(0, noise_std_dev);
	Eigen::VectorXd y(n);
	auto sample_noise_and_run_regression = [&]() -> UnivariateOLSResult {
		for (unsigned int i = 0; i < n; ++i) {
			y[i] = intercept + slope * (x0 + i * dx) + noise_dist(rng);
		}
		return univariate(x0, dx, y);
	};
	const auto result = sample_noise_and_run_regression();
	test_sse_minimisation(x0, dx, y, result.slope, result.intercept, 1e-8, 1e-8);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 2, result.dof);
	EXPECT_NEAR(intercept, result.intercept, 2e-2);
	EXPECT_NEAR(slope, result.slope, 5e-3);
	const double x_var = x_range * x_range / 12;
	const double noise_var = noise_std_dev * noise_std_dev;
	const double y_var = x_var * slope * slope + noise_var;
	const double xy_cov = slope * x_var;
	const double xy_corr = xy_cov / std::sqrt(x_var * y_var);
	EXPECT_NEAR(xy_corr * xy_corr, result.r2, 2e-3);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y, 1e-3);

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
	EXPECT_NEAR(slope_sse_and_mean.first / (n_samples - 1), result.var_slope, 5e-7);
	EXPECT_NEAR(slope_sse_and_mean.second, result.slope, 5e-3);
	const auto intercept_sse_and_mean = ml::Statistics::sse_and_mean(intercepts.begin(), intercepts.end());
	EXPECT_NEAR(intercept_sse_and_mean.first / (n_samples - 1), result.var_intercept, 8e-6);
	EXPECT_NEAR(intercept_sse_and_mean.second, result.intercept, 2e-2);
	const auto covariance = ml::Statistics::covariance(slopes, intercepts);
	EXPECT_NEAR(covariance, result.cov_slope_intercept, 2e-6);
}

TEST_F(LinearRegressionTest, univariate_without_intercept_errors)
{
	Eigen::VectorXd x(4);
	Eigen::VectorXd y(3);
	ASSERT_THROW(univariate_without_intercept(x, y), std::invalid_argument);
	x.resize(0);
	y.resize(0);
	ASSERT_THROW(univariate_without_intercept(x, y), std::invalid_argument);
}

TEST_F(LinearRegressionTest, univariate_without_intercept_one_point)
{
	Eigen::VectorXd x(1);
	x << 0.5;
	Eigen::VectorXd y(1);
	y << -1;
	const UnivariateOLSResult result = univariate_without_intercept(x, y);
	ASSERT_EQ(1u, result.n);
	ASSERT_EQ(0u, result.dof);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_EQ(0, result.intercept);
	ASSERT_NEAR(1, result.r2, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y)) << result.var_y;
	ASSERT_TRUE(std::isnan(result.var_slope)) << result.var_slope;
	ASSERT_EQ(0, result.var_intercept);
	ASSERT_EQ(0, result.cov_slope_intercept);
	test_sse_minimisation(x, y, result.slope, result.intercept, 1e-8, 0);
}

TEST_F(LinearRegressionTest, univariate_without_intercept_high_noise)
{
	const unsigned int n = 10000;
	Eigen::VectorXd x(n);
	Eigen::VectorXd y(n);
	const double noise_strength = 1e4;
	std::default_random_engine rng(784957984);
	std::uniform_int_distribution uniform(0, 1);
	for (unsigned int i = 0; i < n; ++i) {
		const double x_i = static_cast<double>(i);
		x[i] = x_i;
		y[i] = noise_strength * (0.5 - static_cast<double>(uniform(rng)));
	}
	const auto result = univariate_without_intercept(x, y);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 1, result.dof);
	EXPECT_NEAR(0, result.slope, noise_strength * 1e-6);
	EXPECT_EQ(0, result.intercept);
	EXPECT_NEAR(0, result.r2, 3e-5);
	EXPECT_GE(result.r2, 0);
	const double expected_observation_variance = noise_strength * noise_strength / 4;
	EXPECT_NEAR(expected_observation_variance, result.var_y, expected_observation_variance * 1e-2);
	test_sse_minimisation(x, y, result.slope, result.intercept, 1e-6, 0);
}

TEST_F(LinearRegressionTest, univariate_without_intercept_true_model)
{
	const double noise_std_dev = 0.2;
	const double x_range = 5; // width of the X range
	const double slope = -0.6;
	const unsigned int n = 1000;
	std::default_random_engine rng(784957984);
	std::normal_distribution<double> noise_dist(0, noise_std_dev);
	std::uniform_real_distribution<double> x_dist(0, x_range);
	Eigen::VectorXd x(n);
	Eigen::VectorXd y(n);
	for (unsigned int i = 0; i < n; ++i) {
		x[i] = x_dist(rng);
	}
	auto sample_noise_and_run_regression = [&]() -> UnivariateOLSResult {
		for (unsigned int i = 0; i < n; ++i) {
			y[i] = slope * x[i] + noise_dist(rng);
		}
		return univariate_without_intercept(x, y);
	};
	const auto result = sample_noise_and_run_regression();
	test_sse_minimisation(x, y, result.slope, result.intercept, 1e-8, 0);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 1, result.dof);
	EXPECT_NEAR(slope, result.slope, 3e-3);
	const double mean_x = x_range / 2;
	const double var_x = x_range * x_range / 12;
	const double expected_r2 = 1 - noise_std_dev * noise_std_dev / (slope * slope * (var_x + mean_x * mean_x)  + noise_std_dev * noise_std_dev);
	EXPECT_NEAR(expected_r2, result.r2, 2e-3);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y, 3e-3);

	// Calculate sample statistics of estimators.
	const unsigned int n_samples = 1000;
	std::vector<double> slopes(n_samples);
	for (unsigned int i = 0; i < n_samples; ++i) {
		const auto result_i = sample_noise_and_run_regression();
		slopes[i] = result_i.slope;
	}
	const auto slope_sse_and_mean = ml::Statistics::sse_and_mean(slopes.begin(), slopes.end());
	EXPECT_NEAR(slope_sse_and_mean.first / (n_samples - 1), result.var_slope, 2e-7);
	EXPECT_NEAR(slope_sse_and_mean.second, result.slope, 3e-3);
}

TEST_F(LinearRegressionTest, multivariate_error)
{
	Eigen::MatrixXd X(3, 10);
	Eigen::VectorXd y(9);
	ASSERT_THROW(multivariate(X, y), std::invalid_argument);
	X.resize(3, 2);
	y.resize(2);
	ASSERT_THROW(multivariate(X, y), std::invalid_argument);
}

TEST_F(LinearRegressionTest, multivariate_exact_fit)
{
	Eigen::MatrixXd X(2, 2);
	X << 0.1, 0.2,
		1, 1;
	Eigen::Vector2d y(0.5, 0.3);
	const MultivariateOLSResult result = multivariate(X, y);
	ASSERT_EQ(2u, result.n);
	ASSERT_EQ(0u, result.dof);
	ASSERT_NEAR(-2, result.beta[0], 1e-14);
	ASSERT_NEAR(0.7, result.beta[1], 1e-15);
	ASSERT_NEAR(1, result.r2, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y)) << result.var_y;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			ASSERT_TRUE(std::isnan(result.cov(i, j))) << "cov(" << i << ", " << j << ") == " << result.cov(i, j);
		}
	}
	test_sse_minimisation(X, y, result.beta, Eigen::Vector2d(1e-8, 1e-8));
}

TEST_F(LinearRegressionTest, multivariate_true_model)
{
	const double noise_std_dev = 0.2;
	const double x_range = 5; // width of the X range
	const unsigned int dim = 4;
	Eigen::VectorXd beta(dim);
	beta << -0.5, 0.3, 0, 4;
	const unsigned int n = 1000;
	std::default_random_engine rng(784957984);
	std::normal_distribution<double> noise_dist(0, noise_std_dev);
	std::uniform_real_distribution<double> x_dist(0, x_range);
	Eigen::MatrixXd X(dim, n);
	Eigen::VectorXd y(n);
	for (unsigned int c = 0; c < n; ++c) {
		for (unsigned int r = 0; r < dim; ++r) {
			X(r, c) = x_dist(rng);
		}		
	}
	auto sample_noise_and_run_regression = [&]() -> MultivariateOLSResult {
		for (unsigned int i = 0; i < n; ++i) {
			y[i] = beta.dot(X.col(i)) + noise_dist(rng);
		}
		return multivariate(X, y);
	};
	const auto result = sample_noise_and_run_regression();
	test_sse_minimisation(X, y, result.beta, Eigen::VectorXd::Constant(dim, 1e-8));
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - dim, result.dof);
	EXPECT_NEAR(0, (beta - result.beta).lpNorm<Eigen::Infinity>(), 1e-2) << (result.beta - beta);
	const double expected_r2 = 1 - noise_std_dev * noise_std_dev / (beta.squaredNorm() * x_range * x_range / 12 + noise_std_dev * noise_std_dev);
	EXPECT_NEAR(expected_r2, result.r2, 3e-5);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y, 3e-3);

	// Calculate sample statistics of estimators.
	const unsigned int n_samples = 1000;
	Eigen::MatrixXd betas(n_samples, dim);
	for (unsigned int i = 0; i < n_samples; ++i) {
		const auto result_i = sample_noise_and_run_regression();
		betas.row(i) = result_i.beta;
	}
	for (unsigned int i = 0; i < dim; ++i) {
		EXPECT_NEAR(result.beta[i], betas.col(i).mean(), 1e-2) << i;
		for (unsigned int j = 0; j < dim; ++j) {
			const double cov_ij = ml::Statistics::covariance(betas.col(i), betas.col(j));
			EXPECT_NEAR(cov_ij, result.cov(i, j), 2e-6) << i << " " << j;
			if (i != j) {
				EXPECT_NEAR(result.cov(i, j), result.cov(j, i), 1e-15) << i << " " << j;
			}
			else {
				EXPECT_GE(result.cov(i, i), 0) << i;
			}
		}
	}	
}

TEST_F(LinearRegressionTest, add_ones_error)
{
	Eigen::MatrixXd X(2, 0);
	ASSERT_THROW(add_ones(X), std::invalid_argument);
}

TEST_F(LinearRegressionTest, add_ones)
{
	Eigen::MatrixXd X(0, 2);
	Eigen::MatrixXd actual(add_ones(X));
	ASSERT_EQ(Eigen::MatrixXd::Ones(1, 2), actual);
	X.resize(1, 2);
	X << 0.5, 0.3;
	actual = add_ones(X);
	ASSERT_EQ(X, actual.topRows(1));
	ASSERT_EQ(Eigen::MatrixXd::Ones(1, 2), actual.bottomRows(1));
}

TEST_F(LinearRegressionTest, recursive_multivariate_ols_no_data)
{
	RecursiveMultivariateOLS rmols;
	ASSERT_EQ(0u, rmols.n());
	ASSERT_EQ(0u, rmols.d());
	ASSERT_EQ(0, rmols.beta().size());
}

TEST_F(LinearRegressionTest, recursive_multivariate_ols_one_sample)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	const Eigen::MatrixXd X(Eigen::MatrixXd::Random(d, n));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	RecursiveMultivariateOLS rmols1;
	rmols1.update(X, y);
	ASSERT_EQ(n, rmols1.n());
	ASSERT_EQ(d, rmols1.beta().size());
	RecursiveMultivariateOLS rmols2(X, y);
	ASSERT_EQ(n, rmols2.n());
	ASSERT_EQ(d, rmols2.beta().size());
	ASSERT_EQ(0, (rmols1.beta() - rmols2.beta()).norm()) << rmols1.beta() << " vs " << rmols2.beta();
	const auto expected_beta = multivariate(X, y).beta;
	constexpr double eps1 = 1e-16;
	ASSERT_NEAR(0, (expected_beta - rmols1.beta()).norm(), eps1) << rmols1.beta();
	ASSERT_NEAR(0, (expected_beta - rmols2.beta()).norm(), eps1) << rmols2.beta();	
}

TEST_F(LinearRegressionTest, recursive_multivariate_ols_many_samples)
{
	constexpr unsigned int d = 10;
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d));
	const std::vector<unsigned int> sample_sizes({ d, 4, 20, 6, 20, 4, 1, 100 });
	RecursiveMultivariateOLS rmols;
	const unsigned int total_n = std::accumulate(sample_sizes.begin(), sample_sizes.end(), 0u);
	unsigned int cumulative_n = 0;
	const Eigen::MatrixXd all_X = Eigen::MatrixXd::Random(d, total_n);
	const Eigen::VectorXd all_y = all_X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(total_n);
	unsigned int sample_idx = 0;
	for (const auto n : sample_sizes) {
		const auto X = all_X.block(0, cumulative_n, d, n);
		const auto y = all_y.segment(cumulative_n, n);
		rmols.update(X, y);
		cumulative_n += n;
		const auto cumulative_X = all_X.leftCols(cumulative_n);
		const auto cumulative_y = all_y.head(cumulative_n);		
		const auto ols_beta = multivariate(cumulative_X, cumulative_y).beta;
		const auto beta_error = rmols.beta() - ols_beta;
		ASSERT_NEAR(0, beta_error.norm(), 2e-14) << sample_idx << ":\n" << beta_error << "\nOLS beta:\n" << ols_beta << "\nRecursive OLS beta:\n" << rmols.beta();
		++sample_idx;
	}
}

TEST_F(LinearRegressionTest, recursive_multivariate_ols_one_by_one)
{
	constexpr unsigned int d = 10;
	constexpr unsigned int n_vectors = 200;
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d));	
	const unsigned int total_n = d + n_vectors;
	const Eigen::MatrixXd all_X = Eigen::MatrixXd::Random(d, total_n);
	const Eigen::VectorXd all_y = all_X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(total_n);
	RecursiveMultivariateOLS rmols(all_X.leftCols(d), all_y.head(d));
	for (unsigned int i = d; i < total_n; ++i) {
		rmols.update(all_X.block(0, i, d, 1), all_y.segment(i, 1));
		const auto cumulative_X = all_X.leftCols(i + 1);
		const auto cumulative_y = all_y.head(i + 1);
		const auto ols_beta = multivariate(cumulative_X, cumulative_y).beta;
		const auto beta_error = rmols.beta() - ols_beta;
		ASSERT_NEAR(0, beta_error.norm(), 1e-7) << i << ":\n" << beta_error << "\nOLS beta:\n" << ols_beta << "\nRecursive OLS beta:\n" << rmols.beta();
	}
}

TEST_F(LinearRegressionTest, recursive_multivariate_ols_errors)
{
	Eigen::MatrixXd X(Eigen::MatrixXd::Random(10, 5));
	Eigen::VectorXd y(Eigen::VectorXd::Random(5));
	ASSERT_THROW(RecursiveMultivariateOLS(X, y), std::invalid_argument);
	RecursiveMultivariateOLS rmols;
	ASSERT_THROW(rmols.update(X, y), std::invalid_argument);
	X.resize(10, 20);
	y.resize(21);
	ASSERT_THROW(RecursiveMultivariateOLS(X, y), std::invalid_argument);
	ASSERT_THROW(rmols.update(X, y), std::invalid_argument);
}

TEST_F(LinearRegressionTest, standardise_errors)
{
	Eigen::MatrixXd X;
	ASSERT_THROW(standardise(X), std::invalid_argument);
	X.resize(3, 2);
	X << 2, 2,
		0, 1,
		-1, 1;
	ASSERT_THROW(standardise(X), std::invalid_argument);
}

TEST_F(LinearRegressionTest, standardise_with_params_errors)
{
	Eigen::MatrixXd X;
	Eigen::VectorXd means;
	Eigen::VectorXd standard_deviations;
	ASSERT_THROW(standardise(X, means, standard_deviations), std::invalid_argument);
	X.resize(3, 2);
	X << 2, 2,
		0, 1,
		-1, 1;
	ASSERT_THROW(standardise(X, means, standard_deviations), std::invalid_argument);
}

TEST_F(LinearRegressionTest, standardise)
{
	Eigen::MatrixXd X(2, 3);
	X << 0, 1, 2,
		0, 0, 2;
	standardise(X);
	Eigen::MatrixXd expected(2, 3);
	const double a = 1 / std::sqrt(2 / 3.);
	const double b = 1 / std::sqrt(2);
	expected << -a, 0, a,
		-b, -b, 2 * b;
	ASSERT_NEAR(0, (X - expected).norm(), 1e-15);
}

TEST_F(LinearRegressionTest, standardise_with_params)
{
	Eigen::MatrixXd X(2, 3);
	X << 0, 1, 2,
		0, 0, 2;
	const Eigen::MatrixXd Xorig(X);
	Eigen::VectorXd means;
	Eigen::VectorXd standard_deviations;
	standardise(X, means, standard_deviations);
	Eigen::MatrixXd expected(2, 3);
	const double a = 1 / std::sqrt(2 / 3.);
	const double b = 1 / std::sqrt(2);
	expected << -a, 0, a,
		-b, -b, 2 * b;
	ASSERT_NEAR(0, (X - expected).norm(), 1e-15) << X;
	ASSERT_EQ(2u, means.size());
	ASSERT_EQ(2u, standard_deviations.size());
	Eigen::VectorXd w(2);
	w << 1, 2. / 3;
	ASSERT_NEAR(0, (w - means).norm(), 1e-15) << means;
	w << std::sqrt(2 / 3.), 2 * std::sqrt(2) / 3;
	ASSERT_NEAR(0, (w - standard_deviations).norm(), 1e-15) << standard_deviations;
	unstandardise(X, means, standard_deviations);
	ASSERT_NEAR(0, (X - Xorig).norm(), 1e-15) << X;
}

TEST_F(LinearRegressionTest, standardise_with_params_same_vector)
{
	Eigen::MatrixXd X(2, 3);
	X << 0, 1, 2,
		0, 0, 2;
	Eigen::VectorXd standard_deviations;
	standardise(X, standard_deviations, standard_deviations);
	Eigen::MatrixXd expected(2, 3);
	const double a = 1 / std::sqrt(2 / 3.);
	const double b = 1 / std::sqrt(2);
	expected << -a, 0, a,
		-b, -b, 2 * b;
	ASSERT_NEAR(0, (X - expected).norm(), 1e-15) << X;
	ASSERT_EQ(2u, standard_deviations.size());
	Eigen::VectorXd w(2);
	w << std::sqrt(2 / 3.), 2 * std::sqrt(2) / 3;
	ASSERT_NEAR(0, (w - standard_deviations).norm(), 1e-15) << standard_deviations;
}

TEST_F(LinearRegressionTest, unstandardise_errors)
{
	Eigen::MatrixXd X(2, 3);
	Eigen::VectorXd v1(X.rows() + 1);
	Eigen::VectorXd v2(X.rows());
	ASSERT_THROW(unstandardise(X, v1, v2), std::invalid_argument);
	ASSERT_THROW(unstandardise(X, v2, v1), std::invalid_argument);
	v1.resize(X.rows());
	v2 << 0.1, 0;
	ASSERT_THROW(unstandardise(X, v1, v2), std::domain_error);
	v2 << -0.1, 0.2;
	ASSERT_THROW(unstandardise(X, v1, v2), std::domain_error);
	v2 << 0.1, std::numeric_limits<double>::quiet_NaN();
	ASSERT_THROW(unstandardise(X, v1, v2), std::domain_error);
}

template <bool DoStandardise> void test_ridge_errors()
{
	Eigen::MatrixXd X(3, 10);
	Eigen::VectorXd y(9);
	ASSERT_THROW(ridge<DoStandardise>(X, y, 1), std::invalid_argument);
	X.resize(3, 2);
	y.resize(2);
	ASSERT_THROW(ridge<DoStandardise>(X, y, 1), std::invalid_argument);
	X = Eigen::MatrixXd::Random(3, 10);
	y.resize(10);
	ASSERT_THROW(ridge<DoStandardise>(X, y, -1), std::domain_error);
}

TEST_F(LinearRegressionTest, ridge_errors)
{
	test_ridge_errors<false>();
	test_ridge_errors<true>();
}

TEST_F(LinearRegressionTest, ridge_zero_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	standardise(X0);
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	const auto expected = multivariate(X, y);
	const auto actual = ridge<false>(X0, y, 0);
	ASSERT_EQ(expected.n, actual.n);
	ASSERT_EQ(expected.dof, actual.dof);
	constexpr double tol = 1e-15;
	ASSERT_NEAR(expected.var_y, actual.var_y, tol);
	ASSERT_EQ(expected.dof, actual.effective_dof);
	ASSERT_NEAR(expected.r2, actual.r2, tol);
	ASSERT_NEAR(0, (expected.beta - actual.beta).norm(), tol) << actual.beta;
	ASSERT_NEAR(0, (expected.cov - actual.cov).norm(), tol) << actual.cov;
}

TEST_F(LinearRegressionTest, ridge_nonzero_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	standardise(X0);
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	const auto unregularised = multivariate(X, y);
	const double lambda = 1e-4;
	const auto regularised = ridge<false>(X0, y, lambda);
	ASSERT_EQ(unregularised.n, regularised.n);
	ASSERT_EQ(unregularised.dof, regularised.dof);
	ASSERT_LT(unregularised.var_y, regularised.var_y);	
	ASSERT_GT(unregularised.r2, regularised.r2);
	ASSERT_LT(0, regularised.r2);
	const Eigen::VectorXd beta_diff(unregularised.beta - regularised.beta);
	const double tol = 1e-16;
	ASSERT_NEAR(unregularised.beta[d], regularised.beta[d], tol);
	ASSERT_NEAR(y.mean(), regularised.beta[d], tol);
	const double beta_diff_norm = beta_diff.norm();
	ASSERT_LT(0, beta_diff_norm) << beta_diff;
	ASSERT_GT(1e-4, beta_diff_norm) << beta_diff;
	ASSERT_GT(unregularised.beta.norm(), regularised.beta.norm());
	ASSERT_LT(unregularised.dof, regularised.effective_dof);
	ASSERT_LT(0, regularised.effective_dof);
	test_sse_minimisation(X0, y, lambda, regularised.beta, Eigen::VectorXd::Constant(d + 1, 1e-8));
	ASSERT_EQ(d + 1, regularised.cov.rows());
	ASSERT_EQ(d + 1, regularised.cov.cols());
	for (unsigned int i = 0; i < d; ++i) {
		ASSERT_LE(0, regularised.cov(i, i)) << i;
		ASSERT_GT(unregularised.cov(i, i), regularised.cov(i, i)) << i << ": " << unregularised.cov(i, i) - regularised.cov(i, i);
	}
	ASSERT_NEAR(regularised.var_y / n, regularised.cov(d, d), tol);
}

TEST_F(LinearRegressionTest, ridge_yuge_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	standardise(X0);
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	constexpr double lambda = 1e50;
	const auto result = ridge<false>(X0, y, 1e50);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - d - 1, result.dof);
	const double sst = (y.array() - y.mean()).square().sum();
	const double tol = 1e-16;
	ASSERT_NEAR(sst / result.dof, result.var_y, tol);
	ASSERT_NEAR(0, result.r2, tol);
	ASSERT_NEAR(y.mean(), result.beta[d], tol);
	ASSERT_NEAR(0, result.beta.head(d).norm(), tol);
	ASSERT_NEAR(n - 1, result.effective_dof, tol);
	test_sse_minimisation(X0, y, lambda, result.beta, Eigen::VectorXd::Constant(d + 1, 1e-8));
	for (unsigned int i = 0; i < d; ++i) {
		ASSERT_LE(0, result.cov(i, i)) << i;
		ASSERT_NEAR(0, result.cov(i, i), tol) << i;
	}
	ASSERT_NEAR(result.var_y / n, result.cov(d, d), tol);
}

TEST_F(LinearRegressionTest, ridge_very_small_slopes)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	standardise(X0);
	const Eigen::MatrixXd X(add_ones(X0));
	Eigen::VectorXd true_beta(d + 1);
	constexpr double b = 1e-6;
	true_beta << -b, b, 0, 0.5;
	const Eigen::VectorXd y(X.transpose() * true_beta);
	const auto expected = multivariate(X, y);
	constexpr double lambda = 1e-5;
	const auto actual = ridge<false>(X0, y, lambda);
	ASSERT_EQ(expected.n, actual.n);
	ASSERT_EQ(expected.dof, actual.dof);
	constexpr double tol = lambda * b;
	ASSERT_NEAR(expected.var_y, actual.var_y, tol);
	ASSERT_NEAR(expected.dof, actual.effective_dof, lambda);
	ASSERT_NEAR(expected.r2, actual.r2, tol);
	ASSERT_NEAR(0, (expected.beta - actual.beta).norm(), tol) << actual.beta;
	test_sse_minimisation(X0, y, lambda, actual.beta, Eigen::VectorXd::Constant(d + 1, 1e-8));
	for (unsigned int i = 0; i < d; ++i) {
		ASSERT_LE(0, actual.cov(i, i)) << i;
		ASSERT_GE(expected.cov(i, i) + 1e-20, actual.cov(i, i)) << i << ": " << expected.cov(i, i) - actual.cov(i, i);
	}
	ASSERT_NEAR(actual.var_y / n, actual.cov(d, d), tol);	
}
