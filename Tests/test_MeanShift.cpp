/* (C) 2021 Roman Werpachowski. */
#include <random>
#include <gtest/gtest.h>
#include "ML/Kernels.hpp"
#include "ML/MeanShift.hpp"


using namespace ml::Clustering;
using namespace ml::Kernels;


TEST(MeanShiftTest, single_cluster)
{
    const Eigen::Index n = 100;
    const Eigen::Index d = 2;
    Eigen::MatrixXd data(d, n);
    std::default_random_engine rng(342394823);
    std::normal_distribution n01;
    for (Eigen::Index c = 0; c < n; ++c) {
        for (Eigen::Index r = 0; r < d; ++r) {
            data(r, c) = n01(rng);
        }
    }
    MeanShift ms(std::shared_ptr<const DifferentiableRadialBasisFunction>(new GaussianRBF), 1);
    ms.fit(data);
    std::cout << ms.centroids().transpose();
    ASSERT_EQ(1u, ms.number_clusters());
    ASSERT_EQ(static_cast<size_t>(n), ms.labels().size());
    for (const auto label : ms.labels()) {
        ASSERT_EQ(0u, label);
    }
}
