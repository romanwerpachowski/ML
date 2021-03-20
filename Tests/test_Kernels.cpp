/* (C) 2021 Roman Werpachowski. */
#include <gtest/gtest.h>
#include "ML/Kernels.hpp"

using namespace ml::Kernels;


class ParabolicRBF : public DoubleDifferentiableRadialBasisFunction
{
public:
    double value(double x) const override
    {
        return x * x;
    }

    double gradient(double x) const override
    {
        return 2 * x;
    }

    double second_derivative(double) const override
    {
        return 2;
    }
};


TEST(Kernels, rbf_value)
{
    const RBFKernel<> K(std::make_unique<ParabolicRBF>(), 2);
    const Eigen::Vector2d x1(-1, 1);
    const Eigen::Vector2d x2(1, 1);
    ASSERT_NEAR(16, K.value(x1, x2), 1e-15);
}

TEST(Kernels, rbf_gradient)
{
    const DifferentiableRBFKernel<> K(std::make_unique<ParabolicRBF>(), 2);
    const Eigen::Vector2d x1(-1, 1);
    const Eigen::Vector2d x2(1, 1);
    Eigen::VectorXd grad(2);
    K.gradient(x1, x2, grad);
    ASSERT_NEAR(-32, grad[0], 1e-15);
    ASSERT_NEAR(0, grad[1], 1e-15);
}
