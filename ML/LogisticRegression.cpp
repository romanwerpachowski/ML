/* (C) 2021 Roman Werpachowski. */
#include <cmath>
#include <stdexcept>
#include "LogisticRegression.hpp"

namespace ml
{
    double LogisticRegression::probability(Eigen::Ref<const Eigen::VectorXd> x, double y, Eigen::Ref<const Eigen::VectorXd> w)
    {
        return 1 / (1 + exp(-y * w.dot(x)));
    }

    double LogisticRegression::likelihood(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> w, const double lam)
    {
        if (lam < 0) {
            throw std::domain_error("Lambda cannot be negative");
        }
        // NFY
    }
}