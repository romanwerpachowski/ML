/* (C) 2021 Roman Werpachowski. */
#include <cassert>
#include <iostream>
#include "BallTree.hpp"
#include "Kernels.hpp"
#include "MeanShift.hpp"

namespace ml
{
    namespace Clustering
    {
        MeanShift::MeanShift(std::shared_ptr<const Kernels::DifferentiableRadialBasisFunction> rbf, double h)
            : rbf_(rbf), h_(h), h2_(h* h)
        {
            if (rbf == nullptr) {
                throw std::invalid_argument("MeanShift: RBF is null");
            }
            if (h <= 0) {
                throw std::domain_error("MeanShift: Window radius must be positive");
            }            
        }

        bool MeanShift::fit(Eigen::Ref<const Eigen::MatrixXd> data)
        {
            BallTree data_tree(data, 20);
            fit(data_tree);
            return true;
        }

        void MeanShift::fit(BallTree& data_tree)
        {
            const auto& data = data_tree.data();
            Eigen::MatrixXd work(data);
            const auto n = data.cols();
            const auto d = data.rows();
            Eigen::VectorXd work_v(d);
            for (Eigen::Index i = 0; i < n; ++i) {                
                shift_until_stationary(data_tree, work.col(i), work_v);
            }            
            labels_.resize(n);
            number_clusters_ = 0;
            data_tree.labels().setConstant(-1);
            for (Eigen::Index i = 0; i < n; ++i) {
                // Find the cluster to which this point belongs, or create a new one.
                const auto j = data_tree.find_nearest_neighbour(work.col(i));
                if (data_tree.labels()[j] == -1) {
                    data_tree.labels()[j] = static_cast<double>(number_clusters_);                    
                    ++number_clusters_;
                }
                labels_[i] = static_cast<unsigned int>(data_tree.labels()[j]);
            }
        }

        void MeanShift::calc_new_position(const BallTree& data_tree, const Eigen::Ref<const Eigen::VectorXd> old_pos, Eigen::Ref<Eigen::VectorXd> new_pos) const
        {
            assert(old_pos.size() == new_pos.size());
            new_pos.setZero();
            double sum_g = 0;
            const auto& data = data_tree.data();
            for (Eigen::Index j = 0; j < data.cols(); ++j) {
                const double r2 = (old_pos - data.col(j)).squaredNorm();
                const double g = -rbf_->gradient(r2 / h2_);
                new_pos += data.col(j) * g;
                sum_g += g;
            }
            if (sum_g) { // No better idea what to do when sum_g == 0.
                new_pos /= sum_g;
            }
            // Find closest data point.
            new_pos = data.col(data_tree.find_nearest_neighbour(new_pos));
        }

        void MeanShift::shift_until_stationary(const BallTree& data_tree, Eigen::Ref<Eigen::VectorXd> pos, Eigen::Ref<Eigen::VectorXd> work) const
        {
            bool converged = false;
            unsigned int iter = 0;
            while (!converged) {
                calc_new_position(data_tree, pos, work);
                converged = (pos - work).norm() == 0;
                pos = work;
                ++iter;
            }            
        }
    }
}
