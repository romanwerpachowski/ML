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
    Eigen::VectorXd y(5);
    y << -1, -1, -1, 1, 1;
    Eigen::MatrixXd X(w.size(), y.size());
    X << 0.5, -0.2, 0.3, 0.3, 0.9,
        -0.5, 0.7, -0.9, 0.9, 0.3;
    const double lam = 0.01;

    const auto actual = LogisticRegression::log_likelihood(X, y, w, lam);
    double expected = -lam * w.squaredNorm() / 2;
    for (Eigen::Index i = 0; i < X.cols(); ++i) {
        expected += log(LogisticRegression::probability(X.col(i), y[i], w));
    }
    ASSERT_NEAR(expected, actual, 1e-15);
}

TEST(LogisticRegression, grad_log_likelihood)
{
    Eigen::VectorXd w(2);
    w << 1, 1;
    Eigen::VectorXd y(5);
    y << -1, -1, -1, 1, 1;
    Eigen::MatrixXd X(w.size(), y.size());
    X << 0.5, -0.2, 0.3, 0.3, 0.9,
        -0.5, 0.7, -0.9, 0.9, 0.3;
    const double lam = 0.01;

    Eigen::VectorXd actual_grad(w.size());
    LogisticRegression::grad_log_likelihood(X, y, w, lam, actual_grad);
    const double eps = 1e-8;
    for (Eigen::Index i = 0; i < w.size(); ++i) {
        const double w_i_orig = w[i];
        w[i] = w_i_orig + eps;
        const double ll_up = LogisticRegression::log_likelihood(X, y, w, lam);
        w[i] = w_i_orig - eps;
        const double ll_down = LogisticRegression::log_likelihood(X, y, w, lam);
        w[i] = w_i_orig;
        const double expected_grad = (ll_up - ll_down) / (2 * eps);
        ASSERT_NEAR(expected_grad, actual_grad[i], 1E-7) << i;
    }
}