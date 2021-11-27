/* (C) 2021 Roman Werpachowski. */
#include <random>
#include <benchmark/benchmark.h>
#include "ML/LogisticRegression.hpp"


template <unsigned int D> static void conjugate_gradient_logistic_regression(benchmark::State& state)
{
	const auto sample_size = static_cast<Eigen::Index>(state.range(0));
	const Eigen::MatrixXd X(Eigen::MatrixXd::Random(D, sample_size));
	const Eigen::VectorXd beta(Eigen::VectorXd::Random(D));
	Eigen::VectorXd y(X.transpose() * beta + 0.02 * Eigen::VectorXd::Random(sample_size));
	for (Eigen::Index i = 0; i < sample_size; ++i) {
		y[i] = y[i] > 0 ? 1 : -1;
	}
	ml::ConjugateGradientLogisticRegression lr;
	lr.set_maximum_steps(1000); // Most people will give up at that point.
	for (auto _ : state) {
		lr.fit(X, y);
	}
	state.SetComplexityN(state.range(0));
}

constexpr auto conjugate_gradient_logistic_regression_5d = conjugate_gradient_logistic_regression<5>;
constexpr auto conjugate_gradient_logistic_regression_50d = conjugate_gradient_logistic_regression<50>;
constexpr auto conjugate_gradient_logistic_regression_500d = conjugate_gradient_logistic_regression<500>;

BENCHMARK(conjugate_gradient_logistic_regression_5d)->RangeMultiplier(10)->Range(10, 1000)->Complexity();
BENCHMARK(conjugate_gradient_logistic_regression_50d)->RangeMultiplier(10)->Range(100, 10000)->Complexity();