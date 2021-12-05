#pragma once
/* (C) 2021 Roman Werpachowski. */
#include "Clustering.hpp"
#include <Eigen/Core>

namespace ml
{
    namespace Clustering
    {
        /**
         * @brief Mean-shift clustering model.
        */
        class MeanShift : public Model
        {
        public:
        private:
            virtual void calculate_point_shift(Eigen::Ref<const Eigen::MatrixXd> data, unsigned int i, Eigen::VectorXd& shift) const = 0;
        };
    }
}
