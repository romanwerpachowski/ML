/* (C) 2021 Roman Werpachowski. */
#include <gtest/gtest.h>
#include "ML/Kernels.hpp"
#include "ML/MeanShift.hpp"


using namespace ml::Clustering;
using namespace ml::Kernels;


TEST(MeanShift, single_cluster)
{
    const Eigen::Index n = 1000;
    const Eigen::Index d = 5;
    const Eigen::MatrixXd data(Eigen::MatrixXd::Random(d, n));
    MeanShift ms(std::shared_ptr<const DifferentiableRadialBasisFunction>(new GaussianRBF), 0.1);
    ms.fit(data);
    ASSERT_EQ(1u, ms.number_clusters());
    ASSERT_EQ(static_cast<size_t>(n), ms.labels().size());
    for (const auto label : ms.labels()) {
        ASSERT_EQ(0u, label);
    }
}
