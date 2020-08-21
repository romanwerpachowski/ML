#include <benchmark/benchmark.h>
#include "ML/LinearRegression.hpp"


static void univariate_linear_regression(benchmark::State& state)
{
	const auto sample_size = state.range(0);
	const Eigen::VectorXd x(Eigen::VectorXd::Random(sample_size));
	const Eigen::VectorXd y(0.1 * x.array().sin() + x.array());
	for (auto _ : state) {
		ml::LinearRegression::univariate(x, y);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(univariate_linear_regression)->RangeMultiplier(10)->Range(10, 10000)->Complexity();


static void univariate_linear_regression_regular(benchmark::State& state)
{
	const auto sample_size = state.range(0);
	const Eigen::VectorXd y(Eigen::VectorXd::Random(sample_size));
	for (auto _ : state) {
		ml::LinearRegression::univariate(0.05, 0.1, y);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(univariate_linear_regression_regular)->RangeMultiplier(10)->Range(10, 10000)->Complexity();


static void univariate_linear_regression_without_intercept(benchmark::State& state)
{
	const auto sample_size = state.range(0);
	const Eigen::VectorXd x(Eigen::VectorXd::Random(sample_size));
	const Eigen::VectorXd y(0.1 * x.array().sin() + x.array());
	for (auto _ : state) {
		ml::LinearRegression::univariate_without_intercept(x, y);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(univariate_linear_regression_without_intercept)->RangeMultiplier(10)->Range(10, 10000)->Complexity();


template <unsigned int D> static void multivariate_linear_regression(benchmark::State& state)
{
	const auto sample_size = state.range(0);
	const Eigen::MatrixXd X(Eigen::MatrixXd::Random(D, sample_size));
	const Eigen::VectorXd beta(Eigen::VectorXd::Random(D));
	const Eigen::VectorXd y(X.transpose() * beta + 0.02 * Eigen::VectorXd::Random(sample_size));
	for (auto _ : state) {
		ml::LinearRegression::multivariate(X, y);
	}
	state.SetComplexityN(state.range(0));
}

constexpr auto multivariate_linear_regression_1d = multivariate_linear_regression<1>;
constexpr auto multivariate_linear_regression_2d = multivariate_linear_regression<2>;
constexpr auto multivariate_linear_regression_5d = multivariate_linear_regression<5>;
constexpr auto multivariate_linear_regression_10d = multivariate_linear_regression<10>;
constexpr auto multivariate_linear_regression_50d = multivariate_linear_regression<50>;

BENCHMARK(multivariate_linear_regression_1d)->RangeMultiplier(10)->Range(10, 10000)->Complexity();
BENCHMARK(multivariate_linear_regression_2d)->RangeMultiplier(10)->Range(10, 10000)->Complexity();
BENCHMARK(multivariate_linear_regression_5d)->RangeMultiplier(10)->Range(10, 10000)->Complexity();
BENCHMARK(multivariate_linear_regression_10d)->RangeMultiplier(10)->Range(100, 10000)->Complexity();
BENCHMARK(multivariate_linear_regression_50d)->RangeMultiplier(10)->Range(100, 10000)->Complexity();