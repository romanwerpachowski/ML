#include <random>
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


template <unsigned int D> static void recursive_multivariate_linear_regression_constant_sample_size(benchmark::State& state)
{
	const auto sample_size = state.range(0);
	const Eigen::VectorXd beta(Eigen::VectorXd::Random(D));
	Eigen::MatrixXd X(D, sample_size);
	Eigen::VectorXd y(sample_size);
	
	
	X = Eigen::MatrixXd::Random(D, D);
	y = X.transpose() * beta + 0.02 * Eigen::VectorXd::Random(D);
	ml::LinearRegression::RecursiveMultivariateOLS rmols(X, y);
	for (auto _ : state) {
		state.PauseTiming();
		X = Eigen::MatrixXd::Random(D, sample_size);
		y = X.transpose() * beta + 0.02 * Eigen::VectorXd::Random(sample_size);
		state.ResumeTiming();
		rmols.update(X, y);
	}
	state.SetComplexityN(state.range(0));
}

constexpr auto recursive_multivariate_linear_regression_constant_sample_size_5d = recursive_multivariate_linear_regression_constant_sample_size<5>;
constexpr auto recursive_multivariate_linear_regression_constant_sample_size_500d = recursive_multivariate_linear_regression_constant_sample_size<500>;

BENCHMARK(recursive_multivariate_linear_regression_constant_sample_size_5d)->RangeMultiplier(2)->Range(1, 128)->Complexity();
BENCHMARK(recursive_multivariate_linear_regression_constant_sample_size_500d)->RangeMultiplier(2)->Range(1, 32)->Complexity();


template <unsigned int D> static void recursive_multivariate_linear_regression_random_sample_size(benchmark::State& state)
{
	const unsigned int avg_sample_size = static_cast<unsigned int>(state.range(0));
	const Eigen::VectorXd beta(Eigen::VectorXd::Random(D));
	auto rng = std::default_random_engine(555);
	auto sample_size_distr = std::uniform_int_distribution<unsigned int>(1u, 2 * avg_sample_size + 1);
	const auto X = Eigen::MatrixXd::Random(D, D);
	ml::LinearRegression::RecursiveMultivariateOLS rmols(X, X.transpose() * beta + 0.02 * Eigen::VectorXd::Random(D));
	for (auto _ : state) {
		state.PauseTiming();
		const auto sample_size = sample_size_distr(rng);
		const auto X = Eigen::MatrixXd::Random(D, sample_size);
		const auto y = X.transpose() * beta + 0.02 * Eigen::VectorXd::Random(sample_size);
		state.ResumeTiming();
		rmols.update(X, y);
	}
	state.SetComplexityN(state.range(0));
}

constexpr auto recursive_multivariate_linear_regression_random_sample_size_5d = recursive_multivariate_linear_regression_random_sample_size<5>;
constexpr auto recursive_multivariate_linear_regression_random_sample_size_500d = recursive_multivariate_linear_regression_random_sample_size<500>;

BENCHMARK(recursive_multivariate_linear_regression_random_sample_size_5d)->RangeMultiplier(2)->Range(1, 128)->Complexity();
BENCHMARK(recursive_multivariate_linear_regression_random_sample_size_500d)->RangeMultiplier(2)->Range(1, 32)->Complexity();