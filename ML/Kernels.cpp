/* (C) 2021 Roman Werpachowski. */
#include <cmath>
#include "Kernels.hpp"

namespace ml
{
    namespace Kernels
    {
        Kernel::~Kernel()
        {}

        void Kernel::validate_arguments(const Eigen::Ref<const Eigen::VectorXd> x1, const Eigen::Ref<const Eigen::VectorXd> x2) const
        {
            if (x1.size() != dim()) {
                throw std::invalid_argument("Wrong dimension of x1");
            }
            if (x2.size() != dim()) {
                throw std::invalid_argument("Wrong dimension of x2");
            }
        }

        RadialBasisFunction::~RadialBasisFunction()
        {}

        double GaussianRBF::value(double r2) const
        {
            if (r2 < 0) {
                throw std::domain_error("GaussianRBF: negative argument");
            }
            return std::exp(-r2);
        }

        double GaussianRBF::gradient(double r2) const
        {
            if (r2 < 0) {
                throw std::domain_error("GaussianRBF: negative argument");
            }
            return -std::exp(-r2);
        }

        double GaussianRBF::second_derivative(double r2) const
        {
            if (r2 < 0) {
                throw std::domain_error("GaussianRBF: negative argument");
            }
            return std::exp(-r2);
        }
    }    
}
