/* (C) 2021 Roman Werpachowski. */
#include <cmath>
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

TEST(LogisticRegression, log_likelihood)
{
    Eigen::VectorXd w(2);
    w << 1, 1;

    Eigen::MatrixXd X(2, 5);
    X << 0.5, -0.2, 0.3, 0.3, 0.9,
        -0.5, 0.7, -0.9, 0.9, 0.3;

    Eigen::VectorXd y(5);
    y << -1, -1, -1, 1, 1;

    const double lam = 0.01;

    const auto actual = LogisticRegression::log_likelihood(X, y, w, lam);

    double expected = -lam * w.squaredNorm() / 2;
    for (Eigen::Index i = 0; i < X.cols(); ++i) {
        expected += log(LogisticRegression::probability(X.col(i), y[i], w));
    }
    ASSERT_NEAR(expected, actual, 1e-15);
}