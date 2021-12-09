/* (C) 2021 Roman Werpachowski. */
#include <cmath>
#include <random>
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

TEST(LogisticRegression, hessian_log_likelihood)
{
    Eigen::VectorXd w(2);
    w << 1, 1;
    Eigen::VectorXd y(5);
    y << -1, -1, -1, 1, 1;
    Eigen::MatrixXd X(w.size(), y.size());
    X << 0.5, -0.2, 0.3, 0.3, 0.9,
        -0.5, 0.7, -0.9, 0.9, 0.3;
    const double lam = 0.01;

    Eigen::MatrixXd actual_H(w.size(), w.size());
    LogisticRegression::hessian_log_likelihood(X, y, w, lam, actual_H);
    Eigen::VectorXd grad_up(w.size());
    Eigen::VectorXd grad_down(w.size());
    Eigen::VectorXd expected_H_i(w.size());
    const double eps = 1e-8;
    for (Eigen::Index i = 0; i < w.size(); ++i) {
        const double w_i_orig = w[i];
        w[i] = w_i_orig + eps;
        LogisticRegression::grad_log_likelihood(X, y, w, lam, grad_up);
        w[i] = w_i_orig - eps;
        LogisticRegression::grad_log_likelihood(X, y, w, lam, grad_down);
        w[i] = w_i_orig;
        expected_H_i = (grad_up - grad_down) / (2 * eps);
        for (Eigen::Index j = 0; j < w.size(); ++j) {
            ASSERT_NEAR(expected_H_i[j], actual_H(i, j), 1E-7) << i << " " << j;
        }        
    }
}

TEST(LogisticRegression, predict)
{
    LogisticRegression::Result result;
    result.w.resize(2);    
    result.w << 1, 1;
    Eigen::VectorXd y(5);
    Eigen::MatrixXd X(result.w.size(), y.size());
    X << 0.5, -0.2, 0.3, 0.3, 0.9,
        -0.5, 0.7, -0.9, 0.9, 0.3;        
    result.predict(X, y);
    for (Eigen::Index i = 0; i < X.cols(); ++i) {
        const double p1 = LogisticRegression::probability(X.col(i), 1, result.w);
        const double expected = p1 > 0.5 ? 1 : -1;
        ASSERT_EQ(expected, y[i]) << i;
        ASSERT_EQ(expected, result.predict_single(X.col(i))) << i;
    }
    const Eigen::VectorXd y2 = result.predict(X);
    ASSERT_EQ(0, (y - y2).norm());
}

TEST(ConjugateGradientLogisticRegression, separable)
{
    std::default_random_engine rng(784957984);
    std::normal_distribution n01;
    const unsigned int n = 100;
    const unsigned int d = 10;
    Eigen::VectorXd w(d);
    Eigen::MatrixXd X(d, n);
    for (unsigned int k = 0; k < d; ++k) {
        w[k] = n01(rng);
        for (unsigned int i = 0; i < n; ++i) {
            X(k, i) = n01(rng);
        }
    }
    Eigen::VectorXd y(n);
    for (unsigned int i = 0; i < n; ++i) {
        const double score = X.col(i).dot(w);
        y[i] = score >= 0 ? 1 : -1;
    }
    ConjugateGradientLogisticRegression cglr;
    cglr.set_relative_tolerance(1e-6);
    cglr.set_maximum_steps(100);
    const auto result = cglr.fit(X, y);
    ASSERT_TRUE(result.converged);
    Eigen::VectorXd pred_y = result.predict(X);
    ASSERT_EQ(0, (y - pred_y).norm());
    ASSERT_LT(result.steps_taken, 100u);
}

TEST(ConjugateGradientLogisticRegression, non_separable)
{
    std::default_random_engine rng(784957984);
    std::normal_distribution n01;
    const unsigned int n = 100;
    const unsigned int d = 10;
    Eigen::VectorXd w(d);
    Eigen::MatrixXd X(d, n);
    for (unsigned int k = 0; k < d; ++k) {
        w[k] = n01(rng);
        for (unsigned int i = 0; i < n; ++i) {
            X(k, i) = n01(rng);
        }
    }
    Eigen::VectorXd y(n);
    for (unsigned int i = 0; i < n; ++i) {
        const double score = X.col(i).dot(w);
        y[i] = score >= 0 ? 1 : -1;
    }
    // Flip 10% of labels.
    for (unsigned int i = 0; i < n / 10; ++i) {
        y[i] *= -1;
    }
    ConjugateGradientLogisticRegression cglr;
    cglr.set_lam(0);
    cglr.set_relative_tolerance(1e-15);
    cglr.set_maximum_steps(100);
    const auto result = cglr.fit(X, y);
    ASSERT_TRUE(result.converged);
    ASSERT_LT(result.steps_taken, 100u);
    Eigen::VectorXd pred_y = result.predict(X);
    const double expected_mse = sqrt(4 * n / 10);
    ASSERT_NEAR(expected_mse, (y - pred_y).norm(), expected_mse * 0.4);
}