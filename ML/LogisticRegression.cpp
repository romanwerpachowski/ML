/* (C) 2021 Roman Werpachowski. */
#include <cassert>
#include <cmath>
#include <stdexcept>
#include "LogisticRegression.hpp"

namespace ml
{
    LogisticRegression::~LogisticRegression()
    {}

    double LogisticRegression::probability(Eigen::Ref<const Eigen::VectorXd> x, double y, Eigen::Ref<const Eigen::VectorXd> w)
    {
        assert(y == -1 || y == 1);
        return 1 / (1 + exp(-y * w.dot(x)));
    }

    double LogisticRegression::log_likelihood(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> w, const double lam)
    {
        if (!(lam >= 0)) {
            throw std::domain_error("Lambda must be non-negative");
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
        if (!(lam >= 0)) {
            throw std::domain_error("Lambda must be non-negative");
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

    void LogisticRegression::hessian_log_likelihood(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> w, double lam, Eigen::Ref<Eigen::MatrixXd> H)
    {
        if (!(lam >= 0)) {
            throw std::domain_error("Lambda must be non-negative");
        }
        const auto n = y.size();
        if (n != X.cols()) {
            throw std::invalid_argument("Size mismatch: y.size() != X.cols()");
        }
        const auto dim = w.size();
        if (dim != X.rows()) {
            throw std::invalid_argument("Size mismatch: w.size() != X.rows()");
        }
        if (dim != H.rows()) {
            throw std::invalid_argument("Size mismatch: w.size() != H.rows()");
        }
        if (dim != H.cols()) {
            throw std::invalid_argument("Size mismatch: w.size() != H.cols()");
        }        
        H = -lam * Eigen::MatrixXd::Identity(dim, dim);
        for (Eigen::Index i = 0; i < n; ++i) {
            const auto x_i = X.col(i);
            const double p = probability(x_i, 1, w);
            H -= p * (1 - p) * x_i * x_i.transpose();
        }
    }

    AbstractLogisticRegression::AbstractLogisticRegression(double lam, double weight_relative_tolerance, double weight_absolute_tolerance)
        : lam_(lam), weight_relative_tolerance_(weight_relative_tolerance), weight_absolute_tolerance_(weight_absolute_tolerance)
    {
        if (!(lam >= 0)) {
            throw std::domain_error("Lambda must be non-negative");
        }
        if (!(weight_relative_tolerance >= 0)) {
            throw std::domain_error("Relative weight tolerance must be non-negative");
        }
        if (!(weight_absolute_tolerance >= 0)) {
            throw std::domain_error("Absolute weight tolerance must be non-negative");
        }
    }

    bool AbstractLogisticRegression::converged(Eigen::Ref<const Eigen::VectorXd> old_weights, Eigen::Ref<const Eigen::VectorXd> new_weights)
    {
        const double old_weights_norm = old_weights.norm();
        const double new_weights_norm = new_weights.norm();
        const double weights_diff_norm = (old_weights - new_weights).norm();
        return weights_diff_norm <= weight_absolute_tolerance_ + std::max(old_weights_norm, new_weights_norm) * weight_relative_tolerance_;
    }
}