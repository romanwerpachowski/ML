#include <random>
#include <benchmark/benchmark.h>
#include "ML/DecisionTree.hpp"

static void BM_univariate_regression_tree(benchmark::State& state)
{
	std::default_random_engine rng;
	std::normal_distribution normal;
	const auto m = static_cast<int>(state.range(0));
	const int n = m * m;
	const double sigma = 0.01;

	Eigen::MatrixXd X(2, n);
	Eigen::VectorXd y(n);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < m; ++j) {
			const int k = i * m + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			if (i < 4) {
				if (j < 2) {
					y[k] = 0.2;
				} else {
					y[k] = 0.9;
				}
			} else {
				if (j < 6) {
					y[k] = 0.5;
				} else {
					y[k] = 0.25;
				}
			}
			// Add noise.
			y[k] += sigma * normal(rng);
		}
	}
	// Benchmarked code.
	for (auto _ : state) {
		ml::UnivariateRegressionTree tree(ml::DecisionTrees::univariate_regression_tree(X, y, 100, 2));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_univariate_regression_tree)->RangeMultiplier(2)->Range(2, 128)->UseRealTime()->Complexity();


static void BM_cost_complexity_prune(benchmark::State& state)
{
	std::default_random_engine rng;
	std::normal_distribution normal;
	const auto m = static_cast<int>(state.range(0));
	const int n = m * m;
	const double sigma = 0.01;

	Eigen::MatrixXd X(2, n);
	Eigen::VectorXd y(n);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < m; ++j) {
			const int k = i * m + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			if (i < 4) {
				if (j < 2) {
					y[k] = 0.2;
				} else {
					y[k] = 0.9;
				}
			} else {
				if (j < 6) {
					y[k] = 0.5;
				} else {
					y[k] = 0.25;
				}
			}
			// Add noise.
			y[k] += sigma * normal(rng);
		}
	}
	const ml::UnivariateRegressionTree tree(ml::DecisionTrees::univariate_regression_tree(X, y, 100, 2));
	// Benchmarked code.
	for (auto _ : state) {
		state.PauseTiming();
		// Do not measure the cost of copying the tree.
		ml::UnivariateRegressionTree pruned_tree(tree);
		state.ResumeTiming();
		ml::DecisionTrees::cost_complexity_prune(pruned_tree, 0.01);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_cost_complexity_prune)->RangeMultiplier(2)->Range(2, 64)->Complexity();

static void BM_tree_copy(benchmark::State& state)
{
	std::default_random_engine rng;
	std::normal_distribution normal;
	const auto m = static_cast<int>(state.range(0));
	const int n = m * m;
	const double sigma = 0.01;

	Eigen::MatrixXd X(2, n);
	Eigen::VectorXd y(n);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < m; ++j) {
			const int k = i * m + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			if (i < 4) {
				if (j < 2) {
					y[k] = 0.2;
				} else {
					y[k] = 0.9;
				}
			} else {
				if (j < 6) {
					y[k] = 0.5;
				} else {
					y[k] = 0.25;
				}
			}
			// Add noise.
			y[k] += sigma * normal(rng);
		}
	}
	const ml::UnivariateRegressionTree tree(ml::DecisionTrees::univariate_regression_tree(X, y, 100, 2));
	// Benchmarked code.
	for (auto _ : state) {
		ml::UnivariateRegressionTree copy(tree);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_tree_copy)->RangeMultiplier(2)->Range(2, 64)->Complexity();