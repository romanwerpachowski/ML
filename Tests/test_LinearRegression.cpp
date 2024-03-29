/* (C) 2020 Roman Werpachowski. */
#include <cmath>
#include <random>
#include <Eigen/Eigenvalues>
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

static void test_sse_minimisation(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, const double lambda, Eigen::Ref<const Eigen::VectorXd> beta, Eigen::Ref<const Eigen::VectorXd> beta_deltas, const double tol)
{
	const double min_sse = calc_sse(X, y, lambda, beta);
	for (Eigen::Index i = 0; i < beta.size(); ++i) {
		const double delta = beta_deltas[i];
		if (delta > 0) {
			const auto v_delta = Eigen::VectorXd::Unit(beta.size(), i) * delta;
			double sse = calc_sse(X, y, lambda, beta + v_delta);
			ASSERT_LE(min_sse, sse + tol) << "diff: " << sse - min_sse;
			sse = calc_sse(X, y, lambda, beta - v_delta);
			ASSERT_LE(min_sse, sse + tol) << "diff: " << sse - min_sse;
		}
	}	
}

static void test_result(const Result& result, double tol = 1e-15)
{
	ASSERT_LT(result.dof, result.n);
	const unsigned int num_params = result.n - result.dof;
	ASSERT_LE(0, result.rss);
	if (num_params > 1) {
		ASSERT_LE(result.rss, result.tss) << (result.rss - result.tss) / result.tss;
	}
	if (result.tss > 0) {
		ASSERT_GE(1, result.r2());
		ASSERT_NEAR(1 - result.rss / result.tss, result.r2(), tol);
	} else {
		ASSERT_EQ(1u, num_params);
		ASSERT_TRUE(std::isnan(result.r2()));
	}	
	if (result.dof) {
		ASSERT_NEAR(result.rss / result.dof, result.var_y(), tol);
		ASSERT_NEAR(1 - (result.n - 1) * result.rss / (result.tss * result.dof), result.adjusted_r2(), tol);
		if (num_params > 1) {
			ASSERT_LT(result.adjusted_r2(), result.r2());
		}
		ASSERT_GE(1, result.adjusted_r2());
	} else {
		ASSERT_TRUE(std::isnan(result.adjusted_r2())) << result.adjusted_r2();
		ASSERT_TRUE(std::isnan(result.var_y())) << result.var_y();
	}
}

static void test_result(const UnivariateOLSResult& result, double tol = 1e-15)
{
	test_result(static_cast<const Result&>(result), tol);
	ASSERT_TRUE(std::isfinite(result.slope)) << result.slope;
	ASSERT_TRUE(std::isfinite(result.intercept)) << result.slope;
	const unsigned int num_params = result.n - result.dof;
	if (num_params == 1) {
		ASSERT_EQ(0, result.intercept);		
		ASSERT_EQ(0, result.var_intercept);
		ASSERT_EQ(0, result.cov_slope_intercept);
	} else {
		ASSERT_EQ(2u, num_params);
	}
	if (result.dof) {
		if (num_params == 2) {
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es;
			Eigen::Matrix2d cov;
			cov(0, 0) = result.var_slope;
			cov(1, 1) = result.var_intercept;
			cov(0, 1) = cov(1, 0) = result.cov_slope_intercept;
			es.compute(cov);
			ASSERT_LE(0, es.eigenvalues().minCoeff()) << cov;
		} else {			
			ASSERT_LE(0, result.var_slope);
		}
	} else {
		ASSERT_TRUE(std::isnan(result.var_slope)) << result.var_slope;
		if (num_params == 2) {
			ASSERT_TRUE(std::isnan(result.var_intercept)) << result.var_intercept;
			ASSERT_TRUE(std::isnan(result.cov_slope_intercept)) << result.cov_slope_intercept;
		}
	}
	const double x = 0.5;
	const double expected_y = x * result.slope + result.intercept;
	ASSERT_NEAR(expected_y, result.predict(x), 1e-16);
	Eigen::VectorXd vec_x(1);
	vec_x << x;
	const auto vec_y = result.predict(vec_x);
	ASSERT_EQ(1, vec_y.size());
	ASSERT_NEAR(expected_y, vec_y[0], 1e-16);
}

template <class R> static void test_beta_and_cov(const R& result, double tol)
{
	ASSERT_EQ(result.beta.size(), result.cov.rows());
	for (Eigen::Index i = 0; i < result.beta.size(); ++i) {
		ASSERT_TRUE(std::isfinite(result.beta[i])) << result.beta;
	}
	if (result.dof) {
		ASSERT_NEAR(0, (result.cov - result.cov.transpose()).norm(), tol) << result.cov;
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(result.cov);
		ASSERT_LE(0, es.eigenvalues().minCoeff());
	} else {
		for (Eigen::Index r = 0; r < result.cov.rows(); ++r) {
			for (Eigen::Index c = 0; c < result.cov.cols(); ++c) {
				ASSERT_TRUE(std::isnan(result.cov(r, c))) << result.cov;
			}
		}
	}
}

static void test_result(const MultivariateOLSResult& result, double tol = 1e-15)
{
	test_result(static_cast<const Result&>(result), tol);
	test_beta_and_cov(result, tol);	
}

static void test_result(const RidgeRegressionResult& result, double tol = 1e-15)
{
	test_result(static_cast<const Result&>(result), tol);
	test_beta_and_cov(result, tol);
	ASSERT_LE(result.dof, result.effective_dof);
	
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
	const Eigen::Vector2d x(0.1, 0.2);
	const Eigen::Vector2d y(0.5, 0.3);
	const UnivariateOLSResult result = univariate(x, y);
	test_result(result);
	ASSERT_NEAR(0, (y - result.predict(x)).norm(), 1e-15);
	ASSERT_EQ(2u, result.n);
	ASSERT_EQ(0u, result.dof);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_NEAR(0.7, result.intercept, 1e-15);
	ASSERT_NEAR(1, result.r2(), 1e-15);
	ASSERT_NEAR(0, result.rss, 1e-15);
	ASSERT_NEAR(2e-2, result.tss, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y())) << result.var_y();
	ASSERT_TRUE(std::isnan(result.var_slope)) << result.var_slope;
	ASSERT_TRUE(std::isnan(result.var_intercept)) << result.var_intercept;
	ASSERT_TRUE(std::isnan(result.cov_slope_intercept)) << result.cov_slope_intercept;
	test_sse_minimisation(x, y, result.slope, result.intercept, 1e-8, 1e-8);
}

TEST_F(LinearRegressionTest, univariate_two_points_regular)
{
	constexpr double x0 = 0.1;
	constexpr double dx = 0.1;
	const Eigen::Vector2d y(0.5, 0.3);
	const UnivariateOLSResult result = univariate(x0, dx, y);
	test_result(result);
	const Eigen::Vector2d x(0.1, 0.2);
	ASSERT_NEAR(0, (y - result.predict(x)).squaredNorm(), 1e-15);
	ASSERT_EQ(2u, result.n);
	ASSERT_EQ(0u, result.dof);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_NEAR(.7, result.intercept, 1e-15);
	ASSERT_NEAR(1, result.r2(), 1e-15);
	ASSERT_NEAR(0, result.rss, 1e-15);
	ASSERT_NEAR(2e-2, result.tss, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y())) << result.var_y();
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
	test_result(result);
	const double rss = ((y - x * result.slope).array() - result.intercept).square().sum();
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(rss, result.rss, 1e-14 * result.rss);
	ASSERT_NEAR(tss, result.tss, 1e-14 * result.tss);
	ASSERT_NEAR(result.rss, (y - result.predict(x)).squaredNorm(), 1e-15 * result.rss);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 2, result.dof);
	EXPECT_NEAR(0, result.slope, noise_strength * 1e-2);
	EXPECT_NEAR(intercept, result.intercept, 2e-2 * noise_strength);
	EXPECT_NEAR(0, result.r2(), 3e-4);
	EXPECT_GE(result.r2(), 0);
	const double expected_observation_variance = noise_strength * noise_strength / 4;
	EXPECT_NEAR(expected_observation_variance, result.var_y(), expected_observation_variance * 1e-3);
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
	test_result(r1);
	test_result(r2);
	ASSERT_NEAR(r1.rss, (y - r1.predict(x)).squaredNorm(), 1e-15 * r1.rss);
	ASSERT_NEAR(r2.rss, (y - r2.predict(x)).squaredNorm(), 1e-15 * r2.rss);
	constexpr double tol = 1e-15;
	ASSERT_NEAR(r1.slope, r2.slope, tol);
	ASSERT_NEAR(r1.intercept, r2.intercept, tol);
	ASSERT_NEAR(r1.var_y(), r2.var_y(), tol);
	ASSERT_NEAR(r1.r2(), r2.r2(), tol);
	ASSERT_NEAR(r1.rss, r2.rss, tol);
	ASSERT_NEAR(r1.tss, r2.tss, tol);
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
	test_result(result);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 2, result.dof);
	EXPECT_NEAR(0, result.slope, noise_strength * 1e-2);
	EXPECT_NEAR(intercept, result.intercept, 2e-2 * noise_strength);
	EXPECT_NEAR(0, result.r2(), 3e-4);
	EXPECT_GE(result.r2(), 0);
	const double expected_observation_variance = noise_strength * noise_strength / 4;
	EXPECT_NEAR(expected_observation_variance, result.var_y(), expected_observation_variance * 1e-3);
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
	test_result(result);
	test_sse_minimisation(x, y, result.slope, result.intercept, 1e-8, 1e-8);
	const double rss = ((y - x * result.slope).array() - result.intercept).square().sum();
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(rss, result.rss, 1e-14 * result.rss);
	ASSERT_NEAR(tss, result.tss, 1e-14 * result.tss);
	ASSERT_NEAR(result.rss, (y - result.predict(x)).squaredNorm(), 1e-14 * result.rss);	
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 2, result.dof);
	EXPECT_NEAR(intercept, result.intercept, 4e-3);
	EXPECT_NEAR(slope, result.slope, 4e-3);	
	const double x_var = x_range * x_range / 12;
	const double noise_var = noise_std_dev * noise_std_dev;
	const double y_var = x_var * slope * slope + noise_var;
	const double xy_cov = slope * x_var;
	const double xy_corr = xy_cov / std::sqrt(x_var * y_var);
	EXPECT_NEAR(xy_corr * xy_corr, result.r2(), 2e-3);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y(), 3e-3);

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
	test_result(result);
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
	EXPECT_NEAR(xy_corr * xy_corr, result.r2(), 2e-3);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y(), 1e-3);

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
	test_result(result);
	ASSERT_NEAR(result.rss, (y - result.predict(x)).squaredNorm(), 1e-15 * result.rss);
	ASSERT_EQ(1u, result.n);
	ASSERT_EQ(0u, result.dof);
	ASSERT_NEAR(-2, result.slope, 1e-15);
	ASSERT_EQ(0, result.intercept);
	ASSERT_TRUE(std::isnan(result.r2()));
	ASSERT_TRUE(std::isnan(result.adjusted_r2()));
	ASSERT_NEAR(0, result.rss, 1e-15);
	ASSERT_NEAR(0, result.tss, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y())) << result.var_y();
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
	test_result(result);
	const double rss = (y - x * result.slope).squaredNorm();
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(rss, result.rss, 1e-13 * rss);
	ASSERT_NEAR(tss, result.tss, 1e-15 * tss);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 1, result.dof);
	EXPECT_NEAR(0, result.slope, noise_strength * 1e-6);
	EXPECT_EQ(0, result.intercept);
	ASSERT_NEAR(result.r2(), result.adjusted_r2(), 1e-15);
	EXPECT_LT(result.r2(), 0);
	ASSERT_NEAR(rss / result.dof, result.var_y(), 1e-15 * rss / result.dof);
	const double expected_observation_variance = noise_strength * noise_strength / 4;
	EXPECT_NEAR(expected_observation_variance, result.var_y(), expected_observation_variance * 1e-2);
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
	test_result(result);
	test_sse_minimisation(x, y, result.slope, result.intercept, 1e-8, 0);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - 1, result.dof);
	EXPECT_NEAR(slope, result.slope, 3e-3);
	const double rss = (y - x * result.slope).squaredNorm();
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(rss, result.rss, 1e-13 * rss);
	ASSERT_NEAR(tss, result.tss, 1e-15 * tss);
	ASSERT_NEAR(result.r2(), result.adjusted_r2(), 1e-15);
	ASSERT_NEAR(rss / result.dof, result.var_y(), 1e-14);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y(), 3e-3);		

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
	test_result(result);
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(tss, result.tss, 1e-15);
	const double rss = (y - X.transpose() * result.beta).squaredNorm();
	ASSERT_NEAR(rss, result.rss, 1e-15);
	ASSERT_NEAR(0, (y - result.predict(X)).squaredNorm(), 1e-15);
	ASSERT_EQ(2u, result.n);
	ASSERT_EQ(0u, result.dof);
	ASSERT_NEAR(-2, result.beta[0], 1e-14);
	ASSERT_NEAR(0.7, result.beta[1], 1e-15);
	ASSERT_NEAR(1, result.r2(), 1e-15);
	ASSERT_TRUE(std::isnan(result.adjusted_r2()));
	ASSERT_NEAR(0, result.rss, 1e-15);
	ASSERT_NEAR(2e-2, result.tss, 1e-15);
	ASSERT_TRUE(std::isnan(result.var_y())) << result.var_y();
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
	test_result(result);
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(tss, result.tss, 1e-15);
	const double rss = (y - X.transpose() * result.beta).squaredNorm();
	ASSERT_NEAR(rss, result.rss, 1e-15);
	ASSERT_NEAR(result.rss, (y - result.predict(X)).squaredNorm(), 1e-15);
	test_sse_minimisation(X, y, result.beta, Eigen::VectorXd::Constant(dim, 1e-8));
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - dim, result.dof);	
	EXPECT_NEAR(0, (beta - result.beta).lpNorm<Eigen::Infinity>(), 1e-2) << (result.beta - beta);
	const double expected_r2 = 1 - noise_std_dev * noise_std_dev / (beta.squaredNorm() * x_range * x_range / 12 + noise_std_dev * noise_std_dev);
	EXPECT_NEAR(expected_r2, result.r2(), 3e-5);
	EXPECT_NEAR(noise_std_dev * noise_std_dev, result.var_y(), 3e-3);

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
	ASSERT_EQ(d, static_cast<unsigned int>(rmols1.beta().size()));
	RecursiveMultivariateOLS rmols2(X, y);
	ASSERT_EQ(n, rmols2.n());
	ASSERT_EQ(d, static_cast<unsigned int>(rmols2.beta().size()));
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
	ASSERT_EQ(2, means.size());
	ASSERT_EQ(2, standard_deviations.size());
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
	ASSERT_EQ(2, standard_deviations.size());
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
	constexpr double tol = 1e-15;
	test_result(actual);
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(tss, actual.tss, 1e-15);
	ASSERT_NEAR(actual.rss, (y - actual.predict(X0)).squaredNorm(), tol);
	ASSERT_EQ(expected.n, actual.n);
	ASSERT_EQ(expected.dof, actual.dof);
	ASSERT_NEAR(expected.var_y(), actual.var_y(), tol);
	ASSERT_EQ(expected.dof, actual.effective_dof);
	ASSERT_NEAR(expected.r2(), actual.r2(), tol);
	ASSERT_NEAR(expected.rss, actual.rss, tol);
	ASSERT_NEAR(expected.tss, actual.tss, tol);
	ASSERT_NEAR(0, (expected.beta - actual.beta).norm(), tol) << actual.beta;
	ASSERT_NEAR(0, (expected.cov - actual.cov).norm(), tol) << actual.cov;
}

TEST_F(LinearRegressionTest, ridge_nonzero_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	constexpr double tol = 1e-16;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	standardise(X0);
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	const auto unregularised = multivariate(X, y);
	const double lambda = 1e-4;
	const auto regularised = ridge<false>(X0, y, lambda);
	test_result(regularised);
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(tss, regularised.tss, tol);
	ASSERT_EQ(unregularised.n, regularised.n);
	ASSERT_EQ(unregularised.dof, regularised.dof);
	ASSERT_LT(unregularised.var_y(), regularised.var_y());	
	ASSERT_GT(unregularised.r2(), regularised.r2());	
	ASSERT_LT(0, regularised.r2());
	const Eigen::VectorXd beta_diff(unregularised.beta - regularised.beta);	
	ASSERT_NEAR(regularised.rss, (y - regularised.predict(X0)).squaredNorm(), tol);
	ASSERT_LE(unregularised.rss, regularised.rss);
	ASSERT_NEAR(unregularised.tss, regularised.tss, tol);
	ASSERT_NEAR(unregularised.beta[d], regularised.beta[d], tol);
	ASSERT_NEAR(y.mean(), regularised.beta[d], tol);
	const double beta_diff_norm = beta_diff.norm();
	ASSERT_LT(0, beta_diff_norm) << beta_diff;
	ASSERT_GT(1e-4, beta_diff_norm) << beta_diff;
	ASSERT_GT(unregularised.beta.norm(), regularised.beta.norm());
	ASSERT_LT(unregularised.dof, regularised.effective_dof);
	ASSERT_LT(0, regularised.effective_dof);
	test_sse_minimisation(X0, y, lambda, regularised.beta, Eigen::VectorXd::Constant(d + 1, 1e-8), 0);
	ASSERT_EQ(d + 1, static_cast<unsigned int>(regularised.cov.rows()));
	ASSERT_EQ(d + 1, static_cast<unsigned int>(regularised.cov.cols()));
	for (unsigned int i = 0; i < d; ++i) {
		ASSERT_LE(0, regularised.cov(i, i)) << i;
		ASSERT_GT(unregularised.cov(i, i), regularised.cov(i, i)) << i << ": " << unregularised.cov(i, i) - regularised.cov(i, i);
	}
	ASSERT_NEAR(regularised.var_y() / n, regularised.cov(d, d), tol);
	const auto regularised2 = ridge(X0, y, lambda, false);
	ASSERT_EQ(0., (regularised.beta - regularised2.beta).norm());
}

TEST_F(LinearRegressionTest, ridge_yuge_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	constexpr double tol = 1e-16;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	standardise(X0);
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	constexpr double lambda = 1e50;
	const auto result = ridge<false>(X0, y, 1e50);
	test_result(result);
	const double tss = (y.array() - y.mean()).square().sum();	
	ASSERT_NEAR(tss, result.tss, tol);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - d - 1, result.dof);
	const double sst = (y.array() - y.mean()).square().sum();	
	ASSERT_NEAR(result.rss, (y - result.predict(X0)).squaredNorm(), tol);
	ASSERT_NEAR(sst / result.dof, result.var_y(), tol);
	ASSERT_NEAR(0, result.r2(), tol);
	ASSERT_NEAR(result.tss, result.rss, tol);
	ASSERT_NEAR(y.mean(), result.beta[d], tol);
	ASSERT_NEAR(0, result.beta.head(d).norm(), tol);
	ASSERT_NEAR(n - 1, result.effective_dof, tol);
	test_sse_minimisation(X0, y, lambda, result.beta, Eigen::VectorXd::Constant(d + 1, 1e-8), 0);
	for (unsigned int i = 0; i < d; ++i) {
		ASSERT_LE(0, result.cov(i, i)) << i;
		ASSERT_NEAR(0, result.cov(i, i), tol) << i;
	}
	ASSERT_NEAR(result.var_y() / n, result.cov(d, d), tol);
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
	test_result(actual);
	ASSERT_EQ(expected.n, actual.n);
	ASSERT_EQ(expected.dof, actual.dof);
	constexpr double tol = lambda * b;
	ASSERT_NEAR(actual.rss, (y - actual.predict(X0)).squaredNorm(), tol);
	ASSERT_NEAR(expected.var_y(), actual.var_y(), tol);
	ASSERT_NEAR(expected.dof, actual.effective_dof, lambda);
	ASSERT_NEAR(expected.r2(), actual.r2(), tol);
	ASSERT_NEAR(expected.rss, actual.rss, tol);
	ASSERT_NEAR(expected.tss, actual.tss, tol);
	ASSERT_NEAR(0, (expected.beta - actual.beta).norm(), tol) << actual.beta;
	test_sse_minimisation(X0, y, lambda, actual.beta, Eigen::VectorXd::Constant(d + 1, 1e-8), 0);
	for (unsigned int i = 0; i < d; ++i) {
		ASSERT_LE(0, actual.cov(i, i)) << i;
		ASSERT_GE(expected.cov(i, i) + 1e-20, actual.cov(i, i)) << i << ": " << expected.cov(i, i) - actual.cov(i, i);
	}
	ASSERT_NEAR(actual.var_y() / n, actual.cov(d, d), tol);	
}

TEST_F(LinearRegressionTest, ridge_do_standardise_zero_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	X0.row(0) *= 2;
	X0.row(1) /= 2;
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	const auto expected = multivariate(X, y);
	const auto actual = ridge<true>(X0, y, 0);
	test_result(actual);
	ASSERT_EQ(expected.n, actual.n);
	ASSERT_EQ(expected.dof, actual.dof);
	constexpr double tol = 1e-15;
	ASSERT_NEAR(actual.rss, (y - actual.predict(X0)).squaredNorm(), tol);
	ASSERT_NEAR(expected.var_y(), actual.var_y(), tol);
	ASSERT_EQ(expected.dof, actual.effective_dof);
	ASSERT_NEAR(expected.r2(), actual.r2(), tol);
	ASSERT_NEAR(expected.rss, actual.rss, tol);
	ASSERT_NEAR(expected.tss, actual.tss, tol);
	ASSERT_NEAR(0, (expected.beta - actual.beta).norm(), tol) << actual.beta;
	ASSERT_NEAR(0, (expected.cov - actual.cov).norm(), tol) << expected.cov << "\n\n" << actual.cov;
}


TEST_F(LinearRegressionTest, ridge_do_standardise_nonzero_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	X0.row(0) *= 2;
	X0.row(1) /= 2;
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	const auto unregularised = multivariate(X, y);
	const double lambda = 1e-4;
	const auto regularised = ridge<true>(X0, y, lambda);
	test_result(regularised);
	ASSERT_NEAR(regularised.rss, (y - regularised.predict(X0)).squaredNorm(), 1e-15);
	ASSERT_EQ(unregularised.n, regularised.n);
	ASSERT_EQ(unregularised.dof, regularised.dof);
	ASSERT_LT(unregularised.var_y(), regularised.var_y());
	ASSERT_GT(unregularised.r2(), regularised.r2());
	ASSERT_LT(0, regularised.r2());
	ASSERT_LE(unregularised.rss, regularised.rss);
	ASSERT_NEAR(unregularised.tss, regularised.tss, 1e-16);
	const Eigen::VectorXd beta_diff(unregularised.beta - regularised.beta);
	const double beta_diff_norm = beta_diff.norm();
	ASSERT_LT(0, beta_diff_norm) << beta_diff;
	ASSERT_GT(1e-4, beta_diff_norm) << beta_diff;
	ASSERT_GT(unregularised.beta.norm(), regularised.beta.norm());
	ASSERT_LT(unregularised.dof, regularised.effective_dof);
	ASSERT_LT(0, regularised.effective_dof);
	test_sse_minimisation(X0, y, lambda, regularised.beta, Eigen::VectorXd::Constant(d + 1, 1e-8), 1e-11);
	ASSERT_EQ(d + 1, static_cast<unsigned int>(regularised.cov.rows()));
	ASSERT_EQ(d + 1, static_cast<unsigned int>(regularised.cov.cols()));
	for (unsigned int i = 0; i < d; ++i) {
		ASSERT_LE(0, regularised.cov(i, i)) << i;
		ASSERT_GT(unregularised.cov(i, i), regularised.cov(i, i)) << i << ": " << unregularised.cov(i, i) - regularised.cov(i, i);
	}
	const auto regularised2 = ridge(X0, y, lambda, true);
	ASSERT_EQ(0., (regularised.beta - regularised2.beta).norm());
}

TEST_F(LinearRegressionTest, ridge_covariance)
{
	constexpr unsigned int n = 1000;
	constexpr unsigned int d = 3;
	Eigen::MatrixXd X(Eigen::MatrixXd::Random(d, n));
	standardise(X);
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d));
	const Eigen::VectorXd y_hat(X.transpose() * true_beta + Eigen::VectorXd::Constant(n, -0.24));
	Eigen::VectorXd y(n);
	std::default_random_engine rng(784957984);
	std::normal_distribution<double> noise_dist(0, 0.1);
	auto sample_noise_and_run_regression = [&]() -> RidgeRegressionResult {
		for (unsigned int i = 0; i < n; ++i) {
			y[i] = y_hat[i] + noise_dist(rng);
		}
		return ridge<false>(X, y, 0.1);
	};
	const auto result = sample_noise_and_run_regression();
	test_result(result);
	// Calculate sample statistics of estimators.
	const unsigned int n_samples = 1000;
	Eigen::MatrixXd betas(n_samples, d + 1);
	for (unsigned int i = 0; i < n_samples; ++i) {
		const auto result_i = sample_noise_and_run_regression();
		betas.row(i) = result_i.beta;
	}
	Eigen::MatrixXd sample_cov(d + 1, d + 1);
	for (unsigned int i = 0; i <= d; ++i) {
		for (unsigned int j = 0; j <= i; ++j) {
			const double cov_ij = ml::Statistics::covariance(betas.col(i), betas.col(j));
			sample_cov(i, j) = sample_cov(j, i) = cov_ij;			
		}
	}
	ASSERT_NEAR(0, (sample_cov - result.cov).norm(), 2e-6) << "estimate:\n" << result.cov << "\n\nsample:\n" << sample_cov << "\n\ndifference:\n" << (sample_cov - result.cov);
}

TEST_F(LinearRegressionTest, ridge_do_standardise_covariance)
{
	constexpr unsigned int n = 1000;
	constexpr unsigned int d = 3;
	const Eigen::MatrixXd X(Eigen::MatrixXd::Random(d, n));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d));
	const Eigen::VectorXd y_hat(X.transpose() * true_beta + Eigen::VectorXd::Constant(n, -0.24));
	Eigen::VectorXd y(n);
	std::default_random_engine rng(784957984);
	std::normal_distribution<double> noise_dist(0, 0.1);
	auto sample_noise_and_run_regression = [&]() -> RidgeRegressionResult {
		for (unsigned int i = 0; i < n; ++i) {
			y[i] = y_hat[i] + noise_dist(rng);
		}
		return ridge<true>(X, y, 0.1);
	};
	const auto result = sample_noise_and_run_regression();
	test_result(result);
	// Calculate sample statistics of estimators.
	const unsigned int n_samples = 1000;
	Eigen::MatrixXd betas(n_samples, d + 1);
	for (unsigned int i = 0; i < n_samples; ++i) {
		const auto result_i = sample_noise_and_run_regression();
		betas.row(i) = result_i.beta;
	}
	Eigen::MatrixXd sample_cov(d + 1, d + 1);
	for (unsigned int i = 0; i <= d; ++i) {
		for (unsigned int j = 0; j <= i; ++j) {
			const double cov_ij = ml::Statistics::covariance(betas.col(i), betas.col(j));
			sample_cov(i, j) = sample_cov(j, i) = cov_ij;
		}
	}
	ASSERT_NEAR(0, (sample_cov - result.cov).norm(), 5e-6) << "estimate:\n" << result.cov << "\n\nsample:\n" << sample_cov << "\n\ndifference:\n" << (sample_cov - result.cov);
}

TEST_F(LinearRegressionTest, press_multivariate)
{
	Eigen::MatrixXd X(2, 3);
	X << -1, 0, 1,
		1, 1, 1;
	Eigen::VectorXd y(3);
	y << 1, 0, 1;
	const double press_statistic = press(X, y, multivariate);
	ASSERT_NEAR(4 + 1 + 4, press_statistic, 1e-15);
}

TEST_F(LinearRegressionTest, press_ridge_zero_lambda)
{
	Eigen::MatrixXd X(1, 3);
	X << -1, 0, 1;
	Eigen::VectorXd y(3);
	y << 1, 0, 1;
	const double press_statistic = press(X, y, [](Eigen::Ref<const Eigen::MatrixXd> train_X, Eigen::Ref<const Eigen::VectorXd> train_y) {
		return ridge<true>(train_X, train_y, 0);
		});
	ASSERT_NEAR(4 + 1 + 4, press_statistic, 1e-15);
}

TEST_F(LinearRegressionTest, press_ridge)
{
	Eigen::MatrixXd X(1, 3);
	X << -1, 0, 1;
	Eigen::VectorXd y(3);
	y << 1, 0, 1;
	const double press_statistic = press(X, y, [](Eigen::Ref<const Eigen::MatrixXd> train_X, Eigen::Ref<const Eigen::VectorXd> train_y) {
		return ridge<true>(train_X, train_y, 0);
		});
	ASSERT_NEAR(4 + 1 + 4, press_statistic, 1e-15);
}

TEST_F(LinearRegressionTest, press_univariate_with_intercept)
{
	Eigen::VectorXd x(3);
	x << -1, 0, 1;
	Eigen::VectorXd y(3);
	y << 1, 0, 1;
	const double press_statistic = press_univariate<true>(x, y);
	ASSERT_NEAR(4 + 1 + 4, press_statistic, 1e-15);
}

TEST_F(LinearRegressionTest, press_univariate_without_intercept)
{
	Eigen::VectorXd x(3);
	x << -1, 0, 1;
	Eigen::VectorXd y(3);
	y << 1, 0, 1;
	const double press_statistic = press_univariate<false>(x, y);
	ASSERT_NEAR(4 + 0 + 4, press_statistic, 1e-15);
}

TEST_F(LinearRegressionTest, lasso_zero_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	standardise(X0);
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	const auto expected = multivariate(X, y);
	const auto actual = lasso<false>(X0, y, 0);
	constexpr double tol = 1e-15;
	test_result(actual);
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(tss, actual.tss, 1e-15);
	ASSERT_NEAR(actual.rss, (y - actual.predict(X0)).squaredNorm(), tol);
	ASSERT_EQ(expected.n, actual.n);
	ASSERT_EQ(expected.dof, actual.dof);
	ASSERT_NEAR(expected.var_y(), actual.var_y(), tol);
	ASSERT_EQ(expected.dof, actual.effective_dof);
	ASSERT_NEAR(expected.r2(), actual.r2(), tol);
	ASSERT_NEAR(expected.rss, actual.rss, tol);
	ASSERT_NEAR(expected.tss, actual.tss, tol);
	ASSERT_NEAR(0, (expected.beta - actual.beta).norm(), tol) << actual.beta;	
}

TEST_F(LinearRegressionTest, lasso_nonzero_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	constexpr double tol = 5e-16;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	standardise(X0);
	const Eigen::MatrixXd X(add_ones(X0));
	Eigen::VectorXd true_beta(d + 1);
	true_beta << 0.4, 1e-8, -0.6, 0.1;
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	const auto unregularised = multivariate(X, y);
	const double lambda = 1.;
	const auto regularised = lasso<false>(X0, y, lambda);
	ASSERT_NEAR(0, regularised.beta[1], 1e-14) << regularised.beta;
	test_result(regularised);
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(tss, regularised.tss, tol);
	ASSERT_EQ(unregularised.n, regularised.n);
	ASSERT_EQ(unregularised.dof, regularised.dof);
	ASSERT_LT(unregularised.var_y(), regularised.var_y());
	ASSERT_GT(unregularised.r2(), regularised.r2());
	ASSERT_LT(0, regularised.r2());
	const Eigen::VectorXd beta_diff(unregularised.beta - regularised.beta);
	ASSERT_NEAR(regularised.rss, (y - regularised.predict(X0)).squaredNorm(), tol);
	ASSERT_LE(unregularised.rss, regularised.rss);
	ASSERT_NEAR(unregularised.tss, regularised.tss, tol);
	ASSERT_NEAR(unregularised.beta[d], regularised.beta[d], tol);
	ASSERT_NEAR(y.mean(), regularised.beta[d], tol);
	const double beta_diff_norm = beta_diff.norm();
	ASSERT_LT(0, beta_diff_norm) << beta_diff;
	ASSERT_GT(unregularised.beta.lpNorm<1>(), regularised.beta.lpNorm<1>());
	ASSERT_EQ(unregularised.dof + 1, regularised.effective_dof);
	test_sse_minimisation(X0, y, lambda, regularised.beta, Eigen::VectorXd::Constant(d + 1, 1e-8), 1e-8);
	const auto regularised2 = lasso(X0, y, lambda, false);
	ASSERT_EQ(0., (regularised.beta - regularised2.beta).norm());
}

TEST_F(LinearRegressionTest, lasso_yuge_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	constexpr double tol = 1e-16;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	standardise(X0);
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	constexpr double lambda = 1e50;
	const auto result = lasso<false>(X0, y, 1e50);
	test_result(result);
	const double tss = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(tss, result.tss, tol);
	ASSERT_EQ(n, result.n);
	ASSERT_EQ(n - d - 1, result.dof);
	const double sst = (y.array() - y.mean()).square().sum();
	ASSERT_NEAR(result.rss, (y - result.predict(X0)).squaredNorm(), tol);
	ASSERT_NEAR(sst / result.dof, result.var_y(), tol);
	ASSERT_NEAR(0, result.r2(), tol);
	ASSERT_NEAR(result.tss, result.rss, tol);
	ASSERT_NEAR(y.mean(), result.beta[d], tol);
	ASSERT_NEAR(0, result.beta.head(d).norm(), tol);
	ASSERT_NEAR(n - 1, result.effective_dof, tol);
	test_sse_minimisation(X0, y, lambda, result.beta, Eigen::VectorXd::Constant(d + 1, 1e-8), 0);
}

TEST_F(LinearRegressionTest, lasso_do_standardise_zero_lambda)
{
	constexpr unsigned int n = 10;
	constexpr unsigned int d = 3;
	Eigen::MatrixXd X0(Eigen::MatrixXd::Random(d, n));
	X0.row(0) *= 2;
	X0.row(1) /= 2;
	const Eigen::MatrixXd X(add_ones(X0));
	const Eigen::VectorXd true_beta(Eigen::VectorXd::Random(d + 1));
	const Eigen::VectorXd y(X.transpose() * true_beta + 0.1 * Eigen::VectorXd::Random(n));
	const auto expected = multivariate(X, y);
	const auto actual = lasso<true>(X0, y, 0);
	test_result(actual);
	ASSERT_EQ(expected.n, actual.n);
	ASSERT_EQ(expected.dof, actual.dof);
	constexpr double tol = 1e-15;
	ASSERT_NEAR(actual.rss, (y - actual.predict(X0)).squaredNorm(), tol);
	ASSERT_NEAR(expected.var_y(), actual.var_y(), tol);
	ASSERT_EQ(expected.dof, actual.effective_dof);
	ASSERT_NEAR(expected.r2(), actual.r2(), tol);
	ASSERT_NEAR(expected.rss, actual.rss, tol);
	ASSERT_NEAR(expected.tss, actual.tss, tol);
	ASSERT_NEAR(0, (expected.beta - actual.beta).norm(), tol) << actual.beta;	
}

TEST_F(LinearRegressionTest, multivariate_polynomial)
{
	constexpr unsigned int n = 101;
	Eigen::MatrixXd X(4, n);
	Eigen::VectorXd y(n);
	for (unsigned int i = 0; i < n; ++i) {
		const double x = -1 + 0.02 * static_cast<double>(i);
		X(0, i) = x;
		X(1, i) = x * x;
		X(2, i) = x * x * x;
		X(3, i) = 1;
		y[i] = std::abs(x) + 1 + 0.2 * std::sin(x);
	}
	const auto result = multivariate(X, y);
	// Test against: sklearn.linear_model.Ridge
	ASSERT_NEAR(0.944624786854704, result.r2(), 1e-15);
	Eigen::VectorXd expected_beta(4);
	expected_beta << 0.199600230558526, 0.928490907343161, -0.0314887196669726, 1.18926358655283;
	ASSERT_NEAR(0, (result.beta - expected_beta).norm(), 1e-14) << result.beta;
}

TEST_F(LinearRegressionTest, ridge_without_standardisation_polynomial)
{
	constexpr unsigned int n = 101;
	Eigen::MatrixXd X(3, n);
	Eigen::VectorXd y(n);
	for (unsigned int i = 0; i < n; ++i) {
		const double x = -1 + 0.02 * static_cast<double>(i);
		X(0, i) = x;
		X(1, i) = x * x;
		X(2, i) = x * x * x;
		y[i] = std::abs(x) + 1 + 0.2 * std::sin(x);
	}
	standardise(X);
	const auto result = ridge<false>(X, y, 0.2);
	// Test against: sklearn.linear_model.Ridge
	ASSERT_NEAR(0.944617571172089, result.r2(), 1e-15);
	Eigen::VectorXd expected_beta(4);
	expected_beta << 0.114840832263246, 0.28175948879102, -0.0108204127651298, 1.5049504950495;
	ASSERT_NEAR(0, (result.beta - expected_beta).norm(), 1e-14) << result.beta;
}

TEST_F(LinearRegressionTest, lasso_without_standardisation_polynomial)
{
	constexpr unsigned int n = 101;
	Eigen::MatrixXd X(3, n);
	Eigen::VectorXd y(n);
	for (unsigned int i = 0; i < n; ++i) {
		const double x = -1 + 0.02 * static_cast<double>(i);
		X(0, i) = x;
		X(1, i) = x * x;
		X(2, i) = x * x * x;
		y[i] = std::abs(x) + 1 + 0.2 * std::sin(x);
	}
	standardise(X);
	const auto result = lasso<false>(X, y, 0.1 * static_cast<double>(n));
	// Test against: sklearn.linear_model.Lasso
	ASSERT_NEAR(0.892348728286198, result.r2(), 5e-16);
	Eigen::VectorXd expected_beta(4);
	expected_beta << 0.0551505195043211, 0.232317428372784, 0, 1.5049504950495;
	ASSERT_NEAR(0, (result.beta - expected_beta).norm(), 2e-14) << result.beta;
}

TEST_F(LinearRegressionTest, multivariate_predict)
{
	MultivariateOLSResult result;
	result.beta.resize(3);
	result.beta << 1, 0, -1;
	Eigen::MatrixXd X(3, 2);
	X << 0.5, 0.5,
		0.5, 0.5,
		-0.5, 0.5;
	const Eigen::VectorXd y(result.predict(X));
	ASSERT_EQ(2, y.size());
	ASSERT_EQ(1, y[0]);
	ASSERT_EQ(0, y[1]);
	ASSERT_EQ(1, result.predict_single(X.col(0)));
	ASSERT_EQ(0, result.predict_single(X.col(1)));
}

TEST_F(LinearRegressionTest, regularised_predict)
{
	RegularisedRegressionResult result;
	result.beta.resize(3);
	result.beta << 1, -1, 0.5;
	Eigen::MatrixXd X(2, 2);
	X << 0.5, 0.5,
		-0.5, 0.5;
	const Eigen::VectorXd y(result.predict(X));
	ASSERT_EQ(2, y.size());
	ASSERT_EQ(1.5, y[0]);
	ASSERT_EQ(0.5, y[1]);
	ASSERT_EQ(1.5, result.predict_single(X.col(0)));
	ASSERT_EQ(0.5, result.predict_single(X.col(1)));
}