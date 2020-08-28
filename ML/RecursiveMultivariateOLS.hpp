#pragma once
#include <Eigen/Cholesky>
#include "LinearRegression.hpp"

namespace ml
{
	namespace LinearRegression
	{
		/** @brief Recursive multivariate Ordinary Least Squares.

		Given a stream of pairs \f$(X_i, \vec{y}_i)\f$, updates the least-squares estimate for \f$\vec{\beta}\f$ using the model

		\f$ \vec{y}_0 = X_0^T \vec{\beta} + \vec{e}_0 \f$

		\f$ \vec{y}_1 = X_1^T \vec{\beta} + \vec{e}_1 \f$

		...

		where \f$\vec{e}_i\f$ are i.i.d. Gaussian.

		Based on https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/2/436/files/2017/07/22-notes-6250-f16.pdf
		*/
		class RecursiveMultivariateOLS
		{
		public:
			/** @brief Initialises without data. */
			DLL_DECLSPEC RecursiveMultivariateOLS();

			/** @brief Initialises with the first sample and calculates the first beta estimate.

			@param[in] X D x N matrix of X values, with data points in columns.
			@param[in] y Y vector with length N.
			@throw std::invalid_argument If `y.size() != X.cols()` or `X.cols() < X.rows()`.
			*/
			DLL_DECLSPEC RecursiveMultivariateOLS(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

			/** @brief Updates the beta estimate with a new sample.
			@param[in] X D x N matrix of X values, with data points in columns.
			@param[in] y Y vector with length N.
			@throw std::invalid_argument If `(X, y)` is the first sample (i.e., `n() == 0`) and `X.cols() < X.rows()`, or `y.size() != X.cols()`.
			*/
			DLL_DECLSPEC void update(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

			/** @brief Returns the number of data points seen so far. */
			unsigned int n() const
			{
				return n_;
			}

			/** @brief Returns the dimension of data points. If n() == 0, returs 0. */
			unsigned int d() const
			{
				return d_;
			}

			/** @brief Returns the current estimate of beta. If n() == 0, returns an empty vector. */
			const Eigen::VectorXd& beta() const
			{
				return beta_;
			}
		private:
			Eigen::LDLT<Eigen::MatrixXd> helper_decomp_; /**< N_i x N_i decomposition. */
			Eigen::MatrixXd P_; /**< D x D information matrix, equal to (X_1 * X_1^T + X_2 * X_2 + ...)^-1. */
			Eigen::MatrixXd K_; /**< D x N_i helper matrix. */
			Eigen::MatrixXd W_; /**< N_i x N_i helper matrix. */
			Eigen::MatrixXd V_; /**< N_i x D helper matrix. */
			Eigen::VectorXd beta_; /**< Current estimate of beta. */
			Eigen::VectorXd residuals_; /**< Helper vector w/ size N_i. */
			unsigned int n_; /**< Number of data points seen so far. */
			unsigned int d_; /**< Dimension of each x data point. */

			/// Initialise recursive OLS.
			void initialise(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);
		};
	}
}
