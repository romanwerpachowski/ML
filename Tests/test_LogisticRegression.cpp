/* (C) 2021 Roman Werpachowski. */
#include <gtest/gtest.h>
#include "ML/LogisticRegression.hpp"

using namespace ml;


TEST(LogisticRegression, probability)
{
    Eigen::VectorXd w(2);
    w << 1, 1;
    Eigen::VectorXd x(2);

    x << 0, 0;
    ASSERT_NEAR(0.5, LogisticRegression::probability(x, -1, w), 1e-16);
    ASSERT_NEAR(0.5, LogisticRegression::probability(x, 1, w), 1e-16);

    x << -1e10, 1e10;
    ASSERT_NEAR(0.5, LogisticRegression::probability(x, -1, w), 1e-16);
    ASSERT_NEAR(0.5, LogisticRegression::probability(x, 1, w), 1e-16);

    x << 1e10, 1e10;
    ASSERT_NEAR(0, LogisticRegression::probability(x, -1, w), 1e-16);
    ASSERT_NEAR(1, LogisticRegression::probability(x, 1, w), 1e-16);

    x << -1e10, -1e10;
    ASSERT_NEAR(1, LogisticRegression::probability(x, -1, w), 1e-16);
    ASSERT_NEAR(0, LogisticRegression::probability(x, 1, w), 1e-16);
}