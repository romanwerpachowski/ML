#pragma once
#include <string>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	/** Linear regression algorithms. */
	namespace LinearRegression {

		/** Result of linear regression. */
		struct Result
		{
			unsigned int n; /**< Number of data points. */
			unsigned int dof; /**< Number of degrees of freedom. */
			double var_y; /**< Estimated variance of observations Y. */
			double r2; /**< R2 = 1 - fraction of variance unexplained relative to a "base model". */
		};

		/** Result of 1D Ordinary Least Squares regression (with or without intercept).*/
		struct UnivariateOLSResult: public Result
		{
			double slope; /**< Coefficient multiplying X values when predicting Y. */
			double intercept; /**< Constant added to slope * X when predicting Y. */			
			/** The following assume independent Gaussian error terms. */
			double var_slope; /**< Estimated variance of the slope. */			
			double var_intercept; /**< Estimated variance of the intercept. */
			double cov_slope_intercept; /**< Estimated covariance of the slope and the intercept. */

			/** Represent result as string. */
			DLL_DECLSPEC std::string to_string() const;
		};

		/** Result of multivariate Ordinary Least Squares regression.		
		*/
		struct MultivariateOLSResult : public Result
		{
			Eigen::VectorXd beta; /**< Fitted coefficients of the model y_i = beta^T X_i. */
			/** The following assume independent Gaussian error terms. */
			Eigen::MatrixXd cov; /**< Covariance matrix of beta coefficients. */

			/** Represent result as string. */
			DLL_DECLSPEC std::string to_string() const;

			DLL_DECLSPEC ~MultivariateOLSResult();
		};

		/** Carries out univariate (aka simple) linear regression with intercept.

		R2 coefficient is calculated w/r to a model returning average Y, and is equal to Corr(X, Y)^2:
			R2 = 1 - \sum_{i=1}^n (y_i - hat{y}_i)^2 / \sum_{i=1}^n (y_i - avg(Y))^2.

		@param x X vector.
		@param y Y vector.
		@throw std::invalid_argument If x and y have different sizes, or if their size is less than 2.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** Carries out univariate (aka simple) linear regression with intercept on regularly spaced points.

		R2 coefficient is calculated w/r to a model returning average Y, and is equal to Corr(X, Y)^2:
			R2 = 1 - \sum_{i=1}^n (y_i - hat{y}_i)^2 / \sum_{i=1}^n (y_i - avg(Y))^2.

		@param x0 First X value.
		@param dx Positive X increment.
		@param y Y vector.
		@throw std::invalid_argument If y.size() < 2.
		@throw std::domain_error If dx <= 0.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(double x0, double dx, Eigen::Ref<const Eigen::VectorXd> y);

		/** Carries out univariate (aka simple) linear regression without intercept.

		R2 coefficient is calculated w/r to a model returning 0 and is therefore not equal to Corr(X, Y)^2:
			R2 = 1 - \sum_{i=1}^n (y_i - hat{y}_i)^2 / \sum_{i=1}^n (y_i)^2.
		
		Intercept , var_intercept and cov_slope_intercept are set to 0.

		@param x X vector.
		@param y Y vector.
		@throw std::invalid_argument If x and y have different sizes, or if their size is less than 1.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate_without_intercept(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** Carries out multivariate linear regression.

		Given X and y, finds beta minimising || y - X^T * beta ||^2.

		R2 is always calculated w/r to model returning average y.

		If fitting with intercept is desired, include a row of 1's in the X values.

		@param X D x N matrix of X values, with data points in columns.
		@param y Y vector with length N.
		@throw std::invalid_argument If y.size() != X.cols() or X.cols() < X.rows().
		*/
		DLL_DECLSPEC MultivariateOLSResult multivariate(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** Add another row with 1s in every column to X.
		@throw std::invalid_argument If X.cols() == 0.
		@return New matrix with a row filled with 1's added at the end.
		*/
		DLL_DECLSPEC Eigen::MatrixXd add_ones(Eigen::Ref<const Eigen::MatrixXd> X);

		/** Given a stream of pairs (X_i, y_i), updates the least-squares estimate for beta solving the equations

		y_0 = X_0^T * beta + e_0
		y_1 = X_1^T * beta + e_1
		...

		Based on https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/2/436/files/2017/07/22-notes-6250-f16.pdf
		*/
		class RecursiveMultivariateOLS
		{
		public:
			/** Initialises without data. */
			DLL_DECLSPEC RecursiveMultivariateOLS();

			/** Initialises with the first sample and calculates the first beta estimate.

			@param X D x N matrix of X values, with data points in columns.
			@param y Y vector with length N.
			@throw std::invalid_argument If y.size() != X.cols() or X.cols() < X.rows().
			*/
			DLL_DECLSPEC RecursiveMultivariateOLS(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

			/** Update the beta estimate with a new sample.
			@param X D x N matrix of X values, with data points in columns.
			@param y Y vector with length N.
			@throw std::invalid_argument If (X, y) is the first sample (i.e., n() == 0) and X.cols() < X.rows(), or y.size() != X.cols().
			*/
			DLL_DECLSPEC void update(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

			/** Returns the number of data points seen so far. */
			unsigned int n() const
			{
				return n_;
			}

			/** Returns the dimension of data points. If n() == 0, returs 0. */
			unsigned int d() const
			{
				return d_;
			}

			/** Returns the current estimate of beta. If n() == 0, returns an empty vector. */
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

			void initialise(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);
		};
	}
}