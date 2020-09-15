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