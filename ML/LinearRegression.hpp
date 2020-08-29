/* (C) 2020 Roman Werpachowski. */
#pragma once
#include <string>
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
			unsigned int dof; /**< Number of residual degrees of freedom (e.g. `n - 2` for univariate regression with intercept). */
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
			Eigen::VectorXd beta; /**< Fitted coefficients of the model \f$\hat{y} = \vec{\beta} \cdot \vec{x}\f$. */
			Eigen::MatrixXd cov; /**< Covariance matrix of beta coefficients. */

			/** @brief Formats the result as string. */
			DLL_DECLSPEC std::string to_string() const;
		};

		/** @brief Result of a (multivariate) ridge regression with intercept.

		Does not contain error estimates because they are not easy to estimate reliably for regularised regression.

		Intercept is reported separately because it has a special status: it's not regularised.
		*/
		struct RidgeRegressionResult : public Result
		{
			Eigen::VectorXd slopes; /**< Regularised coefficients multiplying independent variables. */
			double intercept; /**< Unregularised intercept. */
			double effective_dof; /**< Effective number of residual degrees of freedom \f$ N - \mathrm{tr} [ X^T (X X^T + \lambda I)^{-1} X ] - 1 \f$. */

			/** @brief Formats the result as string. */
			DLL_DECLSPEC std::string to_string() const;
		};

		/** @brief Carries out univariate (aka simple) linear regression with intercept.

		R2 coefficient is calculated w/r to a model returning average Y, and is equal to correlation of X and Y squared:

			\f$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2} {\sum_{i=1}^n (y_i - \bar{y})^2}\f$

		where \f$ \bar{y} = n^{-1} \sum_{i=1}^n y_i \f$.

		@param[in] x X vector.
		@param[in] y Y vector.
		@return UnivariateOLSResult object.
		@throw std::invalid_argument If `x` and `y` have different sizes, or if their size is less than 2.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out univariate (aka simple) linear regression with intercept on regularly spaced points.

		R2 coefficient is calculated w/r to a model returning average Y, and is equal to correlation of X and Y squared:

			\f$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2} {\sum_{i=1}^n (y_i - \bar{y})^2}\f$

		where \f$ \bar{y} = n^{-1} \sum_{i=1}^n y_i \f$.

		@param[in] x0 First X value.
		@param[in] dx Positive X increment.
		@param[in] y Y vector.
		@return UnivariateOLSResult object.
		@throw std::invalid_argument If `y.size() < 2`.
		@throw std::domain_error If `dx <= 0`.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(double x0, double dx, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out univariate (aka simple) linear regression without intercept.

		R2 coefficient is calculated w/r to a model returning average Y, and is therefore _not_ equal to correlation of X and Y squared:

			\f$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2} {\sum_{i=1}^n y_i^2}.\f$		
		
		@param[in] x X vector.
		@param[in] y Y vector.
		@return UnivariateOLSResult object with `intercept`, `var_intercept` and `cov_slope_intercept` set to 0.
		@throw std::invalid_argument If `x` and `y` have different sizes, or if their size is less than 1.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate_without_intercept(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out multivariate linear regression.

		Given X and y, finds \f$\vec{beta}\f$ minimising \f$ \lVert \vec{y} - X^T \vec{\beta} \rVert^2 \f$.

		R2 is always calculated w/r to model returning average y.

		If fitting with intercept is desired, include a row of 1's in the X values.

		@param[in] X D x N matrix of X values, with data points in columns.
		@param[in] y Y vector with length N.
		@return MultivariateOLSResult object.
		@throw std::invalid_argument If `y.size() != X.cols()` or `X.cols() < X.rows()`.
		@see add_ones()
		*/
		DLL_DECLSPEC MultivariateOLSResult multivariate(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out multivariate ridge regression with intercept.

		Given X and y, finds beta and beta0 minimising \f$ \lVert \vec{y} - X^T \vec{\beta} \rVert^2 + \lambda \lVert \vec{\beta} \rVert^2 \f$.

		R2 is always calculated w/r to model returning average y. The matrix `X` is assumed to be standardised.

		@param[in] X D x N matrix of X values, with data points in columns.
		@param[in] y Y vector with length N.
		@param[in] lambda Regularisation strength.
		@return BiasedMultivariateOLSResult object with `beta.size() == X.rows() + 1`.
		@throw std::invalid_argument If `y.size() != X.cols()` or `X.cols() < X.rows()`.
		@throw std::domain_error If `lambda < 0`.
		@see standardise()
		*/
		DLL_DECLSPEC RidgeRegressionResult ridge(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, double lambda);

		/** @brief Adds another row with 1s in every column to X.
		@param[in] X Matrix of independent variables with data points in columns.
		@return New matrix with a row filled with 1's added at the end.
		@throw std::invalid_argument If `X.cols() == 0`.		
		*/
		DLL_DECLSPEC Eigen::MatrixXd add_ones(Eigen::Ref<const Eigen::MatrixXd> X);

		/** @brief Standardises independent variables.

		From each row, `standardise` subtracts its mean and divides it by its standard deviation.
		The standard deviation is calculated using the biased estimator (with denominator equal to
		the number of columns in `X`), so that standardisation works also for N x 1 data.

		@param[in, out] X Matrix of independent variables with data points in columns.
		@throw std::invalid_argument If any row of `X` has all values the same, or `X` is empty.
		*/
		DLL_DECLSPEC void standardise(Eigen::Ref<Eigen::MatrixXd> X);

		/** @brief Standardises independent variables.

		From each row, `standardise` subtracts its mean and divides it by its standard deviation.
		The standard deviation is calculated using the biased estimator (with denominator equal to
		the number of columns in `X`), so that standardisation works also for N x 1 data.

		This version of `standardise` saves original mean and standard deviation for
		every row in provided vectors.

		@param[in, out] X D x N matrix of independent variables with data points in columns.
		@param[out] means At exit has length D and contains means of rows of `X`.
		@param[out] standard_deviations At exit has length D and contains standard deviations of rows of `X`. 
			If `means` and `standard_deviations` refer to the same vector, at exit this vector will contain the standard deviations.
		@throw std::invalid_argument If any row of `X` has all values the same, or `X` is empty.
		*/
		DLL_DECLSPEC void standardise(Eigen::Ref<Eigen::MatrixXd> X, Eigen::VectorXd& means, Eigen::VectorXd& standard_deviations);

		/** @brief Calculates X*X^T, inverts it, and calculates beta.

		Calculates the decomposition of \f$ X X^T + \lambda I \f$ and returns the result of

		\f$ (X X^T + \lambda I)^{-1} X \vec{y} \f$.

		@param lambda Regularisation constant for ridge regression.
		@private Shared between multiple linear regression algorithms.
		*/
		Eigen::VectorXd calculate_XXt_beta(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, Eigen::LDLT<Eigen::MatrixXd>& xxt_decomp, double lambda);
	}
}