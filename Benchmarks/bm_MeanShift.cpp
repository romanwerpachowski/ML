/* (C) 2021 Roman Werpachowski. */
#include <memory>
#include <random>
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include "ML/Kernels.hpp"
#include "ML/MeanShift.hpp"


using namespace ml::Clustering;
using namespace ml::Kernels;


template <Eigen::Index D> static void single_gaussian_cluster(benchmark::State& state)
{
    const auto n = static_cast<Eigen::Index>(state.range(0));

    Eigen::MatrixXd data(D, n);
    std::default_random_engine rng(342394823);
    std::normal_distribution n01;
    for (Eigen::Index c = 0; c < n; ++c) {
        for (Eigen::Index r = 0; r < D; ++r) {
            data(r, c) = n01(rng);
        }
    }
    for (auto _ : state) {
        MeanShift ms(std::shared_ptr<const DifferentiableRadialBasisFunction>(new GaussianRBF), 1);
        ms.fit(data);
    }
    state.SetComplexityN(state.range(0));
}


constexpr auto single_gaussian_cluster_2d = single_gaussian_cluster<2>;


BENCHMARK(single_gaussian_cluster_2d)->RangeMultiplier(10)->Range(10, 10000)->Complexity();
