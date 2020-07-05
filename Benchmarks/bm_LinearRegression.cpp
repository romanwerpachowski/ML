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