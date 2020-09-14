/* (C) 2020 Roman Werpachowski. */
#include "RecursiveMultivariateOLS.hpp"

namespace ml
{
	namespace LinearRegression
	{
		RecursiveMultivariateOLS::RecursiveMultivariateOLS()
			: n_(0), d_(0)
		{}

		RecursiveMultivariateOLS::RecursiveMultivariateOLS(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y)
		{
			initialise(X, y);
		}

		void RecursiveMultivariateOLS::update(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y)
		{
			if (!n_) {
				initialise(X, y);
			}
			else {
				const unsigned int n_i = static_cast<unsigned int>(X.cols());
				if (!n_i) {
					throw std::invalid_argument("No new data points");
				}
				if (d_ != static_cast<unsigned int>(X.rows())) {
					throw std::invalid_argument("Data dimension mismatch");
				}
				if (X.cols() != y.size()) {
					throw std::invalid_argument("X matrix has different number of data points than Y has values");
				}
				// Update P.
				K_.noalias() = P_ * X;
				assert(static_cast<unsigned int>(K_.rows()) == d_);
				assert(static_cast<unsigned int>(K_.cols()) == n_i);
				W_.noalias() = X.transpose() * K_;
				assert(static_cast<unsigned int>(W_.rows()) == n_i);
				assert(static_cast<unsigned int>(W_.cols()) == n_i);
				W_ += Eigen::MatrixXd::Identity(n_i, n_i);
				helper_decomp_.compute(W_);
				V_ = helper_decomp_.solve(K_.transpose());
				assert(static_cast<unsigned int>(V_.rows()) == n_i);
				assert(static_cast<unsigned int>(V_.cols()) == d_);
				P_.noalias() -= K_ * V_;
				// Update beta.
				K_.noalias() = P_ * X;
				assert(static_cast<unsigned int>(K_.rows()) == d_);
				assert(static_cast<unsigned int>(K_.cols()) == n_i);
				residuals_ = y - X.transpose() * beta_;
				beta_ += K_ * residuals_;
				n_ += n_i;
			}
		}

		void RecursiveMultivariateOLS::initialise(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y)
		{
			d_ = static_cast<unsigned int>(X.rows());
			n_ = static_cast<unsigned int>(X.cols());
			P_.resize(d_, d_);
			beta_ = calculate_XXt_beta(X, y, P_, helper_decomp_, 0);			
			P_ = helper_decomp_.solve(Eigen::MatrixXd::Identity(d_, d_));
		}
	}
}