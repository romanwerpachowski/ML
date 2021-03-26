#pragma once
/* (C) 2021 Roman Werpachowski. */
#include <memory>
#include <Eigen/Core>
#include "Clustering.hpp"

namespace ml
{
    class BallTree;

    namespace Kernels
    {
        class DifferentiableRadialBasisFunction;
    }

    namespace Clustering
    {
        /**
         * @brief Mean shift clustering model.
         * 
         * Based on Algorithm 3 in "Modified Subspace Constrained Mean Shift Algorithm", Ghassabeh, Y. A. and Rudzicz, F., Journal of Classification (2020), https://www.cs.toronto.edu/~frank/Download/GhassabehRudzicz-2020-ModifiedSubspaceConstrainedMeanShiftAlgorithm.pdf
        */
        class MeanShift : public Model
        {
        public:
            /**
             * @brief Constructor.
             * 
             * @param rbf Radial basis function defining the kernel. Assumed to be monotonically decreasing with distance.
             * @param h Window radius.
             * 
             * @throw std::invalid_argument If `rbf` is null.
             * @throw std::domain_error If `h <= 0`.
            */
            DLL_DECLSPEC MeanShift(std::shared_ptr<const Kernels::DifferentiableRadialBasisFunction> rbf, double h);

            /**
             * @brief Fits the model to data.
             * @param data Matrix of feature vectors, with data points in columns.
             * @return True if the fit converged (it always does).
            */
            DLL_DECLSPEC bool fit(Eigen::Ref<const Eigen::MatrixXd> data) override;

            unsigned int number_clusters() const override
            {
                return number_clusters_;
            }

            const std::vector<unsigned int>& labels() const override
            {
                return labels_;
            }

            const Eigen::MatrixXd& centroids() const override
            {
                return centroids_;
            }
        private:
            Eigen::MatrixXd centroids_;
            std::vector<unsigned int> labels_;
            std::shared_ptr<const Kernels::DifferentiableRadialBasisFunction> rbf_;
            double h_; /**< Window radius. */
            double h2_; /**< Window radius (squared). */
            unsigned int number_clusters_;

            void calc_new_position(const BallTree& data_tree, Eigen::Ref<const Eigen::VectorXd> old_pos, Eigen::Ref<Eigen::VectorXd> new_pos) const;

            void shift_until_stationary(const BallTree& data_tree, Eigen::Ref<Eigen::VectorXd> pos, Eigen::Ref<Eigen::VectorXd> work) const;
        };
    }
}
