/* (C) 2020 Roman Werpachowski. */
#include <benchmark/benchmark.h>
#include "ML/LinearAlgebra.hpp"


static void xAx_symmetric(benchmark::State& state)
{
	const auto n = static_cast<Eigen::Index>(state.range(0));
	const Eigen::MatrixXd A0 = Eigen::MatrixXd::Random(n, n);
	const Eigen::MatrixXd A = (A0 + A0.transpose()) / 2;
	const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
	for (auto _ : state) {
		ml::LinearAlgebra::xAx_symmetric(A, x);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(xAx_symmetric)->RangeMultiplier(4)->Range(4, 1024)->Complexity(benchmark::oNSquared);
//BENCHMARK(xAx_symmetric)->DenseRange(4, 25, 1)->Complexity(benchmark::oNSquared);


static void xxT(benchmark::State& state)
{
	const auto n = static_cast<Eigen::Index>(state.range(0));
	Eigen::MatrixXd A(n, n);
	const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
	for (auto _ : state) {
		ml::LinearAlgebra::xxT(x, A);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(xxT)->RangeMultiplier(4)->Range(4, 1024)->Complexity(benchmark::oNSquared);
//BENCHMARK(xxT)->DenseRange(4, 25, 1)->Complexity(benchmark::oNSquared);


static void add_a_xxT(benchmark::State& state)
{
	const auto n = static_cast<Eigen::Index>(state.range(0));
	Eigen::MatrixXd A(n, n);
	const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
	for (auto _ : state) {
		ml::LinearAlgebra::add_a_xxT(x, A, 0.5);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(add_a_xxT)->RangeMultiplier(4)->Range(4, 1024)->Complexity(benchmark::oNSquared);
//BENCHMARK(add_a_xxT)->DenseRange(4, 25, 1)->Complexity(benchmark::oNSquared);