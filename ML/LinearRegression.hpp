#pragma once
#include <string>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	/** @brief Linear regression algorithms. */
	namespace LinearRegression {

		/** @brief Result of linear regression. */
		struct Result
		{
			unsigned int n; /**< Number of data points. */
			unsigned int dof; /**< Number of degrees of freedom. */
			double var_y; /**< Estimated variance of observations Y. */
			double r2; /**< R2 coefficient: 1 - fraction of variance unexplained relative to a "base model". */
		};

		/** @brief Result of 1D Ordinary Least Squares regression (with or without intercept).
		
		The following members are calculated assuming independent Gaussian error terms:
		- #var_slope
		- #var_intercept
		- #cov_slope_intercept
		*/
		struct UnivariateOLSResult: public Result
		{
			double slope; /**< Coefficient multiplying X values when predicting Y. */
			double intercept; /**< Constant added to slope * X when predicting Y. */			
			double var_slope; /**< Estimated variance of the slope. */			
			double var_intercept; /**< Estimated variance of the intercept. */
			double cov_slope_intercept; /**< Estimated covariance of the slope and the intercept. */

			/** @brief Formats the result as string. */
			DLL_DECLSPEC std::string to_string() const;
		};

		/** @brief Result of multivariate Ordinary Least Squares regression.		

		The `cov` matrix is calculated asuming independent Gaussian error terms.
		*/
		struct MultivariateOLSResult : public Result
		{
			Eigen::VectorXd beta; /**< Fitted coefficients of the model \f$y_i = \beta^T X_i + \epsilon_i\f$. */
			Eigen::MatrixXd cov; /**< Covariance matrix of beta coefficients. */

			/** @brief Formats the result as string. */
			DLL_DECLSPEC std::string to_string() const;

			/** @brief Destructor. */
			DLL_DECLSPEC ~MultivariateOLSResult();
		};

		/** @brief Carries out univariate (aka simple) linear regression with intercept.

		R2 coefficient is calculated w/r to a model returning average Y, and is equal to correlation of X and Y squared:

			\f$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2} {\sum_{i=1}^n (y_i - \bar{y})^2}\f$

		where \f$ \bar{y} = n^{-1} \sum_{i=1}^n y_i \f$.

		@param x X vector.
		@param y Y vector.
		@throw std::invalid_argument If `x` and `y` have different sizes, or if their size is less than 2.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out univariate (aka simple) linear regression with intercept on regularly spaced points.

		R2 coefficient is calculated w/r to a model returning average Y, and is equal to correlation of X and Y squared:

			\f$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2} {\sum_{i=1}^n (y_i - \bar{y})^2}\f$

		where \f$ \bar{y} = n^{-1} \sum_{i=1}^n y_i \f$.

		@param x0 First X value.
		@param dx Positive X increment.
		@param y Y vector.
		@throw std::invalid_argument If `y.size() < 2`.
		@throw std::domain_error If `dx <= 0`.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(double x0, double dx, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out univariate (aka simple) linear regression without intercept.

		R2 coefficient is calculated w/r to a model returning average Y, and is therefore _not_ equal to correlation of X and Y squared:

			\f$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2} {\sum_{i=1}^n y_i^2}.\f$		
		
		In returned UnivariateOLSResult struct, `intercept`, `var_intercept` and `cov_slope_intercept` are set to 0.

		@param x X vector.
		@param y Y vector.
		@throw std::invalid_argument If `x` and `y` have different sizes, or if their size is less than 1.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate_without_intercept(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out multivariate linear regression.

		Given X and y, finds beta minimising \f$ \lVert \vec{y} - X^T \vec{\beta} \rVert^2 \f$.

		R2 is always calculated w/r to model returning average y.

		If fitting with intercept is desired, include a row of 1's in the X values.

		@param X D x N matrix of X values, with data points in columns.
		@param y Y vector with length N.
		@throw std::invalid_argument If y.size() != X.cols() or X.cols() < X.rows().
		*/
		DLL_DECLSPEC MultivariateOLSResult multivariate(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Adds another row with 1s in every column to X.
		@throw std::invalid_argument If X.cols() == 0.
		@return New matrix with a row filled with 1's added at the end.
		*/
		DLL_DECLSPEC Eigen::MatrixXd add_ones(Eigen::Ref<const Eigen::MatrixXd> X);

		/** @brief Recursive multivariate Ordinary Least Squares.
		
		Given a stream of pairs (X_i, y_i), updates the least-squares estimate for beta solving the equations

		y_0 = X_0^T * beta + e_0
		y_1 = X_1^T * beta + e_1
		...

		Based on https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/2/436/files/2017/07/22-notes-6250-f16.pdf
		*/
		class RecursiveMultivariateOLS
		{
		public:
			/** @brief Initialises without data. */
			DLL_DECLSPEC RecursiveMultivariateOLS();

			/** @brief Initialises with the first sample and calculates the first beta estimate.

			@param X D x N matrix of X values, with data points in columns.
			@param y Y vector with length N.
			@throw std::invalid_argument If y.size() != X.cols() or X.cols() < X.rows().
			*/
			DLL_DECLSPEC RecursiveMultivariateOLS(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

			/** @brief Updates the beta estimate with a new sample.
			@param X D x N matrix of X values, with data points in columns.
			@param y Y vector with length N.
			@throw std::invalid_argument If (X, y) is the first sample (i.e., n() == 0) and X.cols() < X.rows(), or y.size() != X.cols().
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

			void initialise(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);
		};
	}
}