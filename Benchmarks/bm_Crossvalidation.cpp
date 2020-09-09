/* (C) 2020 Roman Werpachowski. */
#include <benchmark/benchmark.h>
#include "ML/Crossvalidation.hpp"


static void k_fold(benchmark::State& state)
{
	const auto sample_size = state.range(0);
	const int dim = 4;
	const Eigen::MatrixXd X(Eigen::MatrixXd::Random(dim, sample_size));
	const Eigen::VectorXd y(Eigen::VectorXd::Random(sample_size));
	const int num_folds = 10;
	const auto train_func = [](const Eigen::MatrixXd& /*train_X*/, const Eigen::VectorXd& /*train_y*/) -> double {
		return 0;
	};
	const auto test_func = [](double model, const Eigen::MatrixXd& /*train_X*/, const Eigen::VectorXd& test_y) -> double {
		return std::pow(test_y[0] - model, 2);
	};
	for (auto _ : state) {
		benchmark::DoNotOptimize(ml::Crossvalidation::k_fold(X, y, train_func, test_func, num_folds));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(k_fold)->RangeMultiplier(10)->Range(100, 1000000)->Complexity();


static void leave_one_out(benchmark::State& state)
{
	const auto sample_size = state.range(0);
	const int dim = 4;
	const Eigen::MatrixXd X(Eigen::MatrixXd::Random(dim, sample_size));
	const Eigen::VectorXd y(Eigen::VectorXd::Random(sample_size));
	const auto train_func = [](const Eigen::MatrixXd& /*train_X*/, const Eigen::VectorXd& /*train_y*/) -> double {
		return 0;
	};
	const auto test_func = [](double model, const Eigen::MatrixXd& /*train_X*/, const Eigen::VectorXd& test_y) -> double {
		return std::pow(test_y[0] - model, 2);
	};
	for (auto _ : state) {
		benchmark::DoNotOptimize(ml::Crossvalidation::leave_one_out(X, y, train_func, test_func));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(leave_one_out)->RangeMultiplier(10)->Range(10, 10000)->Complexity();