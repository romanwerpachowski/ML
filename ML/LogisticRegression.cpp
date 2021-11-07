/* (C) 2021 Roman Werpachowski. */
#include <cassert>
#include <cmath>
#include <stdexcept>
#include "LinearAlgebra.hpp"
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

    void LogisticRegression::predict(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> w, Eigen::Ref<Eigen::VectorXd> y)
    {
        const auto dim = w.size();
        if (dim != X.rows()) {
            throw std::invalid_argument("Size mismatch: w.size() != X.rows()");
        }
        const auto n = X.cols();
        if (n != y.size()) {
            throw std::invalid_argument("Size mismatch: X.cols() != y.size()");
        }
        for (Eigen::Index i = 0; i < n; ++i) {
            if (w.dot(X.col(i)) > 0) {
                y[i] = 1;
            } else {
                y[i] = -1;
            }
        }
    }

    AbstractLogisticRegression::AbstractLogisticRegression()
    {
        lam_ = 1e-3;
        weight_absolute_tolerance_ = 0;
        weight_relative_tolerance_ = 1e-8;
        maximum_steps_ = 100;
    }

    void AbstractLogisticRegression::set_lam(double lam)
    {
        if (!(lam >= 0)) {
            throw std::domain_error("Lambda must be non-negative");
        }
        lam_ = lam;
    }

    void AbstractLogisticRegression::set_weight_absolute_tolerance(double weight_absolute_tolerance)
    {
        if (!(weight_absolute_tolerance >= 0)) {
            throw std::domain_error("Absolute weight tolerance must be non-negative");
        }
        weight_absolute_tolerance_ = weight_absolute_tolerance;
    }

    void AbstractLogisticRegression::set_weight_relative_tolerance(double weight_relative_tolerance)
    {
        if (!(weight_relative_tolerance >= 0)) {
            throw std::domain_error("Relative weight tolerance must be non-negative");
        }
        weight_relative_tolerance_ = weight_relative_tolerance;
    }

    void AbstractLogisticRegression::set_maximum_steps(unsigned int maximum_steps)
    {
        maximum_steps_ = maximum_steps;
    }

    bool AbstractLogisticRegression::weights_converged(Eigen::Ref<const Eigen::VectorXd> old_weights, Eigen::Ref<const Eigen::VectorXd> new_weights) const
    {
        const double old_weights_norm = old_weights.norm();
        const double new_weights_norm = new_weights.norm();
        const double weights_diff_norm = (old_weights - new_weights).norm();
        return weights_diff_norm <= weight_absolute_tolerance_ + std::max(old_weights_norm, new_weights_norm) * weight_relative_tolerance_;
    }

    LogisticRegression::Result ConjugateGradientLogisticRegression::fit(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y) const
    {
        const auto n = y.size();
        const auto d = X.rows();
        if (!n) {
            throw std::invalid_argument("Need at least 1 example");
        }
        if (!d) {
            throw std::invalid_argument("Need at least 1 feature");
        }
        if (X.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        Eigen::VectorXd prev_w;
        Result result;
        result.converged = false;
        result.w = Eigen::VectorXd::Zero(d);
        Eigen::VectorXd g(d);
        Eigen::VectorXd prev_g;
        Eigen::MatrixXd H(d, d);
        Eigen::VectorXd update_direction(d);
        Eigen::VectorXd prev_update_direction;
        unsigned int iter = 0;
        while (iter < maximum_steps() && !result.converged) {
            prev_w = result.w;
            prev_g = g;
            prev_update_direction = update_direction;
            grad_log_likelihood(X, y, prev_w, lam(), g);
            hessian_log_likelihood(X, y, prev_w, lam(), H);
            update_direction = g;
            if (iter) {
                assert(prev_g.size() == d);
                assert(prev_update_direction.size() == d);
                assert(prev_w.size() == d);
                const auto diff_g = g - prev_g;
                update_direction = g;
                const double denom = prev_update_direction.dot(diff_g);
                if (denom != 0) {
                    const double beta = g.dot(diff_g) / denom;
                    update_direction -= beta * prev_update_direction;
                }                                
            }
            result.w -= update_direction * g.dot(update_direction) / LinearAlgebra::xAx_symmetric(H, update_direction);
            result.converged = weights_converged(prev_w, result.w);
            ++iter;
        }
        result.steps_taken = iter;
        return result;
    }
}