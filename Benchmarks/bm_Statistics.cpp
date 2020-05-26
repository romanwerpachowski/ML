#include <random>
#include <benchmark/benchmark.h>
#include "ML/Statistics.hpp"

static void BM_sse(benchmark::State& state)
{
	std::vector<double> data(state.range(0));
	std::default_random_engine rng;
	rng.seed(340934091);
	std::normal_distribution normal;
	for (auto& x : data) {
		x = normal(rng);
	}
	for (auto _ : state) {
		benchmark::DoNotOptimize(ml::Statistics::sse(data.begin(), data.end()));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_sse)->RangeMultiplier(10)->Range(10, 100000)->Complexity();

static void BM_sse_and_mean(benchmark::State& state)
{
	std::vector<double> data(state.range(0));
	std::default_random_engine rng;
	rng.seed(340934091);
	std::normal_distribution normal;
	for (auto& x : data) {
		x = normal(rng);
	}
	for (auto _ : state) {
		benchmark::DoNotOptimize(ml::Statistics::sse_and_mean(data.begin(), data.end()));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_sse_and_mean)->RangeMultiplier(10)->Range(10, 100000)->Complexity();

template <typename T> static void BM_gini_index(benchmark::State& state)
{
	std::vector<T> data(state.range(0));
	std::default_random_engine rng;
	rng.seed(340934091);
	std::uniform_int_distribution d(0, 9);
	for (auto& x : data) {
		x = d(rng);
	}
	for (auto _ : state) {
		benchmark::DoNotOptimize(ml::Statistics::gini_index(data.begin(), data.end()));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_gini_index, double)->RangeMultiplier(10)->Range(10, 100000)->Complexity();
BENCHMARK_TEMPLATE(BM_gini_index, int)->RangeMultiplier(10)->Range(10, 100000)->Complexity();

template <typename T> static void BM_gini_index_and_mode(benchmark::State& state)
{
	std::vector<T> data(state.range(0));
	std::default_random_engine rng;
	rng.seed(340934091);
	std::uniform_int_distribution d(0, 9);
	for (auto& x : data) {
		x = d(rng);
	}
	for (auto _ : state) {
		benchmark::DoNotOptimize(ml::Statistics::gini_index_and_mode(data.begin(), data.end()));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_gini_index_and_mode, double)->RangeMultiplier(10)->Range(10, 100000)->Complexity();
BENCHMARK_TEMPLATE(BM_gini_index_and_mode, int)->RangeMultiplier(10)->Range(10, 100000)->Complexity();