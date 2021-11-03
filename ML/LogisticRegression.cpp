/* (C) 2021 Roman Werpachowski. */
#include <cassert>
#include <cmath>
#include <stdexcept>
#include "LogisticRegression.hpp"

namespace ml
{
    double LogisticRegression::probability(Eigen::Ref<const Eigen::VectorXd> x, double y, Eigen::Ref<const Eigen::VectorXd> w)
    {
        assert(y == -1 || y == 1);
        return 1 / (1 + exp(-y * w.dot(x)));
    }

    double LogisticRegression::log_likelihood(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> w, const double lam)
    {
        if (lam < 0) {
            throw std::domain_error("Lambda cannot be negative");
        }
        if (y.size() != X.cols()) {
            throw std::invalid_argument("Size mismatch: y.size() != X.cols()");
        }
        if (w.size() != X.rows()) {
            throw std::invalid_argument("Size mismatch: w.size() != X.rows()");
        }
        double l = 0;
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            l -= log1p(exp(-y[i] * w.dot(X.col(i))));
        }
        l -= lam * w.squaredNorm() / 2;
        return l;
    }

    void LogisticRegression::grad_log_likelihood(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> w, double lam, Eigen::Ref<Eigen::VectorXd> g)
    {
        if (lam < 0) {
            throw std::domain_error("Lambda cannot be negative");
        }
        if (y.size() != X.cols()) {
            throw std::invalid_argument("Size mismatch: y.size() != X.cols()");
        }
        if (w.size() != X.rows()) {
            throw std::invalid_argument("Size mismatch: w.size() != X.rows()");
        }
        if (w.size() != g.size()) {
            throw std::invalid_argument("Size mismatch: w.size() != g.size()");
        }
        g = - lam * w;
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            g += probability(X.col(i), -y[i], w) * y[i] * X.col(i);
        }
    }
}