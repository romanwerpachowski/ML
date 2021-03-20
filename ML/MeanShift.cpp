/* (C) 2021 Roman Werpachowski. */
#include <cassert>
#include <iostream>
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
            absolute_tolerance_ = 1e-12;
            relative_tolerance_ = 1e-12;
            perturbation_strength_ = h / 1000.0;
        }

        void MeanShift::set_absolute_tolerance(double absolute_tolerance)
        {
            if (absolute_tolerance < 0) {
                throw std::domain_error("MeanShift: Negative absolute tolerance");
            }
            absolute_tolerance_ = absolute_tolerance;
        }

        void MeanShift::set_relative_tolerance(double relative_tolerance)
        {
            if (relative_tolerance < 0) {
                throw std::domain_error("MeanShift: Negative relative tolerance");
            }
            relative_tolerance_ = relative_tolerance;
        }

        bool MeanShift::fit(Eigen::Ref<const Eigen::MatrixXd> data)
        {
            Eigen::MatrixXd work(data);
            const auto n = data.cols();
            const auto d = data.rows();
            Eigen::VectorXd work_v(d);
            labels_.resize(n);
            number_clusters_ = 0;
            Eigen::VectorXd tentative_mode(d);
            for (Eigen::Index i = 0; i < n; ++i) {                
                shift_until_stationary(data, work.col(i), work_v);
                std::cout << i << ": " << work.col(i) << "\n";
                tentative_mode = work.col(i);
                bool found_mode_maximum = false;
                while (!found_mode_maximum) {
                    work.col(i) += Eigen::VectorXd::Random(d) * perturbation_strength_;
                    shift_until_stationary(data, work.col(i), work_v);
                    std::cout << i << ": " << work.col(i) << "\n";
                    found_mode_maximum = close_within_tolerance(tentative_mode, work.col(i));
                    tentative_mode = work.col(i);
                }
                // Find the cluster to which this point belongs, or create a new one.
                unsigned int cluster_label = number_clusters_;
                for (Eigen::Index j = 0; j < i; ++j) {
                    if (close_within_tolerance(work.col(j), work.col(i))) {
                        cluster_label = labels_[j];
                        break;
                    }
                }
                labels_[i] = cluster_label;
                if (cluster_label == number_clusters_) {
                    ++number_clusters_;
                }
            }     
            return true;
        }

        bool MeanShift::close_within_tolerance(const Eigen::Ref<const Eigen::VectorXd> x1, const Eigen::Ref<const Eigen::VectorXd> x2) const
        {
            assert(x1.size() == x2.size());
            /*for (Eigen::Index k = 0; k < x1.size(); ++k) {
                const double dx = std::abs(x1[k] - x2[k]);
                if (dx <= absolute_tolerance_) {
                    continue;
                }
                if (dx <= relative_tolerance_ * std::max(std::abs(x1[k]), std::abs(x2[k]))) {
                    continue;
                }
                return false;
            }
            return true;*/
            const double dr = (x1 - x2).norm();
            return (dr <= absolute_tolerance_) || (dr <= relative_tolerance_ * std::max(x1.norm(), x2.norm()));
        }

        void MeanShift::calc_new_position(const Eigen::Ref<const Eigen::MatrixXd> data, const Eigen::Ref<const Eigen::VectorXd> old_pos, Eigen::Ref<Eigen::VectorXd> new_pos) const
        {
            assert(old_pos.size() == new_pos.size());
            new_pos.setZero();
            double sum_g = 0;
            for (Eigen::Index j = 0; j < data.cols(); ++j) {
                const double r2 = (old_pos - data.col(j)).squaredNorm();
                const double g = -rbf_->gradient(r2 / h2_);
                new_pos += data.col(j) * g;
                sum_g += g;
            }
            if (sum_g) { // No better idea what to do when sum_g == 0.
                new_pos /= sum_g;
            }
        }

        void MeanShift::shift_until_stationary(const Eigen::Ref<const Eigen::MatrixXd> data, Eigen::Ref<Eigen::VectorXd> pos, Eigen::Ref<Eigen::VectorXd> work) const
        {
            bool converged = false;
            while (!converged) {
                calc_new_position(data, pos, work);
                converged = close_within_tolerance(pos, work);
                pos = work;
            }
        }
    }
}
