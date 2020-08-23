#include <random>
#include <gtest/gtest.h>
#include "ML/LinearRegression.hpp"
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
}

TEST_F(LinearRegressionTest, univariate_two_points_regular)
{
	Eigen::Vector2d y(0.5, 0.3);
	const UnivariateOLSResult result = univariate(.1, .1, y);
	ASSERT_EQ(2u, result.n);
	ASSERT_EQ(0u, result.dof);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_NEAR(.7, result.intercept, 1e-15);
	ASSERT_NEAR(1, result.r2, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y)) << result.var_y;
	ASSERT_TRUE(std::isnan(result.var_slope)) << result.var_slope;
	ASSERT_TRUE(std::isnan(result.var_intercept)) << result.var_intercept;
	ASSERT_TRUE(std::isnan(result.cov_slope_intercept)) << result.cov_slope_intercept;
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