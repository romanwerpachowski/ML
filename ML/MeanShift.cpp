/* (C) 2021 Roman Werpachowski. */
#include "Kernels.hpp"
#include "MeanShift.hpp"

namespace ml
{
    namespace Clustering
    {
        MeanShift::MeanShift(std::shared_ptr<Kernels::DoubleDifferentiableRadialBasisFunction> rbf, double h)
            : rbf_(rbf), h2_(h * h)
        {
            if (rbf == nullptr) {
                throw std::invalid_argument("MeanShift: RBF is null");
            }
            if (h <= 0) {
                throw std::domain_error("MeanShift: Window radius must be positive");
            }
            absolute_tolerance_ = 1e-12;
            relative_tolerance_ = 1e-14;
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
            Eigen::VectorXd new_pos(d);
            std::vector<bool> converged(n, false);
            bool all_converged = false;
            while (!all_converged) {
                all_converged = true;
                for (Eigen::Index i = 0; i < n; ++i) {
                    if (!converged[i]) {
                        auto curr_i = work.col(i);
                        new_pos.setZero();
                        double sum_g = 0;
                        for (Eigen::Index j = 0; j < n; ++j) {
                            const double r2 = (curr_i - data.col(j)).squaredNorm();
                            const double g = - rbf_->gradient(r2 / h2_);
                            new_pos += data.col(j) * g;
                            sum_g += g;
                        }
                        if (sum_g) { // No better idea what to do when sum_g == 0.
                            new_pos /= sum_g;
                        }
                        bool converged_i = true;
                        for (Eigen::Index k = 0; k < d; ++k) {
                            const double x_ik = curr_i[k];
                            const double new_x_ik = new_pos[k];
                            const double dx = std::abs(x_ik - new_x_ik);
                            if (dx <= absolute_tolerance_) {
                                continue;
                            }
                            if (dx <= relative_tolerance_ * std::max(std::abs(x_ik), std::abs(new_x_ik))) {
                                continue;
                            }
                            converged_i = false;
                            break;
                        }
                        converged[i] = converged_i;
                        curr_i = new_pos;
                        all_converged &= converged_i;
                    }
                }
            }
            // Find cluster centroids by finding local maxima.
            number_clusters_ = 0;
            labels_.resize(n);
            Eigen::MatrixXd half_hessian(d, d); // Hessian divided by 2.
            for (Eigen::Index i = 0; i < n; ++i) {
                half_hessian.setZero();
                const auto curr_i = work.col(i);
                for (Eigen::Index j = 0; j < n; ++j) {
                    const auto orig_j = data.col(j);
                    const double r2 = (curr_i - data.col(j)).squaredNorm();
                    const double s = r2 / h2_;
                    const double rbf1der = rbf_->gradient(s);
                    half_hessian += rbf1der * Eigen::MatrixXd::Identity(d, d) / h2_;
                    const double rbf2der = rbf_->second_derivative(s);
                    const auto dx_ij = curr_i - orig_j;
                    half_hessian += 2 * (dx_ij).transpose() * dx_ij * rbf2der;
                }
                // Test if hessian is < 0. If yes, this point is a mode.
            }
            return all_converged;
        }
    }
}
