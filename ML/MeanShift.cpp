/* (C) 2021 Roman Werpachowski. */
#include <cassert>
#include <iostream>
#include <map>
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

        struct CentroidInfo
        {
            unsigned int cluster_label;
            unsigned int number_points;
        };

        bool MeanShift::fit(Eigen::Ref<const Eigen::MatrixXd> data)
        {
            BallTree tree(data, 20);
            Eigen::MatrixXd work(data);
            const auto n = data.cols();
            const auto d = data.rows();
            Eigen::VectorXd work_v(d);
            for (Eigen::Index data_idx = 0; data_idx < n; ++data_idx) {
                shift_until_stationary(tree, work.col(data_idx), work_v);
            }
            labels_.resize(n);
            number_clusters_ = 0;
            std::map<unsigned int, CentroidInfo> centroids; // Maps tree indices of centroids to info about them.
            for (Eigen::Index data_idx = 0; data_idx < n; ++data_idx) {
                // Find the cluster to which this point belongs, or create a new one.
                const auto tree_idx = tree.find_nearest_neighbour(work.col(data_idx));
                auto centroids_iter = centroids.find(tree_idx);
                if (centroids_iter == centroids.end()) {
                    centroids[tree_idx] = { number_clusters_, 1 };
                    ++number_clusters_;
                } else {
                    ++centroids_iter->second.number_points;
                }
                labels_[data_idx] = static_cast<unsigned int>(tree.labels()[tree_idx]);
            }
            centroids_.resize(data.rows(), centroids.size());
            for (const auto& key_value : centroids) {
                const auto& ci = key_value.second;
                centroids_.col(ci.cluster_label) = tree.data().col(key_value.first);
            }
            return true;
        }

        void MeanShift::calc_new_position(const BallTree& tree, const Eigen::Ref<const Eigen::VectorXd> old_pos, Eigen::Ref<Eigen::VectorXd> new_pos) const
        {
            assert(old_pos.size() == new_pos.size());
            new_pos.setZero();
            double sum_g = 0;
            const auto& data = tree.data();
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
            new_pos = data.col(tree.find_nearest_neighbour(new_pos));
        }

        void MeanShift::shift_until_stationary(const BallTree& tree, Eigen::Ref<Eigen::VectorXd> pos, Eigen::Ref<Eigen::VectorXd> work) const
        {
            bool converged = false;
            unsigned int iter = 0;
            while (!converged) {
                calc_new_position(tree, pos, work);
                converged = (pos - work).norm() == 0;
                pos = work;
                ++iter;
            }            
        }
    }
}
