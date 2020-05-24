#include <random>
#include <benchmark/benchmark.h>
#include "ML/Statistics.hpp"

static void BM_calc_sse(benchmark::State& state)
{
	std::vector<double> data(state.range(0));
	std::default_random_engine rng;
	rng.seed(340934091);
	std::normal_distribution normal;
	for (auto& x : data) {
		x = normal(rng);
	}
	for (auto _ : state) {
		benchmark::DoNotOptimize(ml::Statistics::calc_sse(data.begin(), data.end()));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_calc_sse)->RangeMultiplier(10)->Range(10, 1000000)->Complexity();