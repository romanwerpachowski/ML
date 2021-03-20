/* (C) 2021 Roman Werpachowski. */
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
    }    
}