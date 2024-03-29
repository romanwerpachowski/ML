#pragma once
/* (C) 2020 Roman Werpachowski. */
#include <string>
#include <Eigen/Core>
#include "dll.hpp"
#include "Crossvalidation.hpp"

namespace ml 
{
	/** @brief Linear regression algorithms. 
	
	For multivariate regression we depart from the textbook convention and assume 
	that independent variables X are laid out columnwise, i.e., data points are
	in columns.
	*/
	namespace LinearRegression {

		/** @brief Result of linear regression. 
		
		Supports R2 calculated w/r to a "base model" returning average Y. 
		R2 is defined as 1 - RSS / TSS, where RSS is the residual sum of squares for the fitted model:

		\f$ \sum_{i=1}^N (\hat{y}_i - y_i)^2 \f$
		
		and TSS is the RSS for the "base model":

		\f$ \mathrm{TSS} = \sum_{i=1}^N (y_i - N^{-1} \sum_{j=1}^N y_j)^2 \f$.
		*/
		struct Result
		{
			unsigned int n; /**< Number of data points. */
			unsigned int dof; /**< Number of residual degrees of freedom (e.g. `n - 2` or `n - 1` for univariate regression with or without intercept). */
			double rss; /**< Residual sum of squares (RSS). */
			double tss; /**< Total sum of squares (TSS, equal to the RSS for the "base model" always returning average Y).*/

			/** @brief Estimated variance of observations Y, equal to `rss / dof`. */
			double var_y() const {
				if (dof) {
					return rss / static_cast<double>(dof);
				}
				else {
					return std::numeric_limits<double>::quiet_NaN();
				}
			}

			/** @brief R2 coefficient.
			
			1 - fraction of variance unexplained relative to a "base model" (returning average Y), estimated as population variance. Equal to `1 - rss / tss`.
			
			*/
			double r2() const {
				return 1 - rss / tss;
			}
			
			/** @brief Adjusted R2 coefficient.
			
			1 - fraction of variance unexplained relative to a "base model" (returning average Y), estimated as sample variance. Equal to `1 - (rss / dof) / (tss / (n - 1))`.
			*/
			double adjusted_r2() const {
				if (dof) {
					return 1 - (rss / static_cast<double>(dof)) / (tss / static_cast<double>(n - 1));
				}
				else {
					return std::numeric_limits<double>::quiet_NaN();
				}
			}
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

			/** @brief Predicts Y given X.
			 @param x Vector of independent variable values.
			 @return Vector of predicted Y(X) with size `X.cols()`.
			*/
			DLL_DECLSPEC Eigen::VectorXd predict(Eigen::Ref<const Eigen::VectorXd> x) const;

			/** @brief Predicts Y given X.
			 @param x Independent variable value.
			 @return Predicted Y(X).
			*/
			double predict(double x) const
			{
				return x * slope + intercept;
			}
		};

		/** @brief Result of multivariate Ordinary Least Squares regression.		

		The #cov matrix is calculated asuming independent Gaussian error terms.
		*/
		struct MultivariateOLSResult : public Result
		{
			Eigen::VectorXd beta; /**< Fitted coefficients of the model \f$\hat{y} = \vec{\beta} \cdot \vec{x}\f$. */
			Eigen::MatrixXd cov; /**< Covariance matrix of beta coefficients. */

			/** @brief Formats the result as string. */
			DLL_DECLSPEC std::string to_string() const;

			/** @brief Predicts Y given X.
			 @param X Matrix of independent variables with data points in columns.
			 @return Vector of predicted Y(X) with size `X.cols()`.
			 @throw std::invalid_argument If `X.rows() != beta.size()`.
			*/
			DLL_DECLSPEC Eigen::VectorXd predict(Eigen::Ref<const Eigen::MatrixXd> X) const;

			/** @brief Predicts Y given X.
			 @param x Vector of independent variables.
			 @return Predicted Y(X).
			 @throw std::invalid_argument If `X.size() != beta.size()`.
			*/
			DLL_DECLSPEC double predict_single(Eigen::Ref<const Eigen::VectorXd> x) const;
		};

		/** @brief Result of a multivariate regularised regression with intercept.
		 
		Regularisation is applied to everything except the intercept, which is the last coefficient in `beta`.

		Contrary to MultivariateOLSResult, it does not assume that inputs `X` contain a row of 1s.

		#var_y is calculated using #dof as the denominator.
		*/
		struct RegularisedRegressionResult : public Result
		{
			Eigen::VectorXd beta; /**< Fitted coefficients of the model \f$\hat{y} = \vec{\beta'} \cdot \vec{x} + \beta_0 \f$, concatenated as \f$ (\vec{\beta'}, \beta_0) \f$. */
			double effective_dof; /**< Effective number of residual degrees of freedom \f$ N - \mathrm{tr} [ X^T (X X^T + \lambda I)^{-1} X ] - 1 \f$. */

			/** @brief Predicts Y given X.
			 @param X Matrix of independent variables with data points in columns.
			 @return Vector of predicted Y(X) with size `X.cols()`.
			 @throw std::invalid_argument If `X.rows() + 1 != beta.size()`.
			*/
			DLL_DECLSPEC Eigen::VectorXd predict(Eigen::Ref<const Eigen::MatrixXd> X) const;

			/** @brief Predicts Y given X.
			 @param x Vector of independent variables.
			 @return Predicted Y(X).
			 @throw std::invalid_argument If `X.size() + 1 != beta.size()`.
			*/
			DLL_DECLSPEC double predict_single(Eigen::Ref<const Eigen::VectorXd> x) const;
		};

		/** @brief Result of a multivariate ridge regression with intercept.		*/
		struct RidgeRegressionResult : public RegularisedRegressionResult
		{
			Eigen::MatrixXd cov;  /**< Covariance matrix of beta coefficients. */
			
			/** @brief Formats the result as string. */
			DLL_DECLSPEC std::string to_string() const;

			using RegularisedRegressionResult::predict;			
		};

		/** @brief Result of a multivariate Lasso regression with intercept.		*/
		struct LassoRegressionResult : public RegularisedRegressionResult
		{
			/** @brief Formats the result as string. */
			DLL_DECLSPEC std::string to_string() const;

			using RegularisedRegressionResult::predict;			
		};

		/** @brief Carries out univariate (aka simple) linear regression with intercept.

		@param[in] x X vector.
		@param[in] y Y vector.
		@return UnivariateOLSResult object.
		@throw std::invalid_argument If `x` and `y` have different sizes, or if their size is less than 2.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out univariate (aka simple) linear regression with intercept on regularly spaced points.

		@param[in] x0 First X value.
		@param[in] dx Positive X increment.
		@param[in] y Y vector.
		@return UnivariateOLSResult object.
		@throw std::invalid_argument If `y.size() < 2`.
		@throw std::domain_error If `dx <= 0`.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(double x0, double dx, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out univariate (aka simple) linear regression without intercept.

		@param[in] x X vector.
		@param[in] y Y vector.
		@return UnivariateOLSResult object with `intercept`, `var_intercept` and `cov_slope_intercept` set to 0.
		@throw std::invalid_argument If `x` and `y` have different sizes, or if their size is less than 1.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate_without_intercept(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out multivariate linear regression.

		Given X and y, finds \f$ \vec{\beta} \f$ minimising \f$ \lVert \vec{y} - X^T \vec{\beta} \rVert^2 \f$.

		If fitting with intercept is desired, include a row of 1's in the X values.

		@param[in] X D x N matrix of X values, with data points in columns.
		@param[in] y Y vector with length N.
		@return MultivariateOLSResult object.
		@throw std::invalid_argument If `y.size() != X.cols()` or `X.cols() < X.rows()`.
		@see add_ones()
		*/
		DLL_DECLSPEC MultivariateOLSResult multivariate(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out multivariate ridge regression with intercept.

		Given X and y, finds \f$ \vec{\beta'} \f$ and \f$ \beta_0 \f$ minimising \f$ \lVert \vec{y} - X^T \vec{\beta'} - \beta_0 \rVert^2 + \lambda \lVert \vec{\beta'} \rVert^2 \f$,
		where \f$ \vec{\beta'} \f$ and \f$ \beta_0 \f$ are concatenated as RidgeRegressionResult#beta in the returned RidgeRegressionResult object.

		The matrix `X` is either assumed to be standardised (`DoStandardise == false`)
		or is standardised internally (`DoStandardise == true`; requires a matrix copy).


		@param[in] X D x N matrix of X values, with data points in columns. Should NOT contain a row with all 1's.
		@param[in] y Y vector with length N.
		@param[in] lambda Regularisation strength.
		@tparam DoStandardise Whether to standardise `X` internally.
		@return RidgeRegressionResult object with `beta.size() == X.rows() + 1`. If `DoStandardise == true`, `beta`
		will be rescaled and shifted to original `X` units and origins, and `cov` will be transformed accordingly.
		@throw std::invalid_argument If `y.size() != X.cols()` or `X.cols() < X.rows()`.
		@throw std::domain_error If `lambda < 0`.
		@see standardise()
		*/
		template <bool DoStandardise> RidgeRegressionResult ridge(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, double lambda);

		/** @brief Carries out multivariate ridge regression with intercept, standardising `X` inputs internally.
		@see ridge().
		*/
		template <> DLL_DECLSPEC RidgeRegressionResult ridge<true>(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, double lambda);

		/** @brief Carries out multivariate ridge regression with intercept, assuming standardised `X` inputs.
		@see ridge().
		*/
		template <> DLL_DECLSPEC RidgeRegressionResult ridge<false>(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, double lambda);

		/** @brief Carries out multivariate ridge regression with intercept, allowing the user switch internal standardisation of `X` data on or off. 
		@see ridge().
		*/
		inline RidgeRegressionResult ridge(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, double lambda, bool do_standardise)
		{
			if (do_standardise) {
				return ridge<true>(X, y, lambda);
			} else {
				return ridge<false>(X, y, lambda);
			}
		}

		/** @brief Carries out multivariate Lasso regression with intercept.

		Given X and y, finds \f$ \vec{\beta'} \f$ and \f$ \beta_0 \f$ minimising \f$ \lVert \vec{y} - X^T \vec{\beta'} - \beta_0 \rVert^2 + \lambda \lVert \vec{\beta'} \rVert^1 \f$,
		where \f$ \vec{\beta'} \f$ and \f$ \beta_0 \f$ are concatenated as LassoRegressionResult#beta in the returned LassoRegressionResult object.

		The matrix `X` is either assumed to be standardised (`DoStandardise == false`)
		or is standardised internally (`DoStandardise == true`; requires a matrix copy).

		Uses the iterated ridge regression method of Fan and Li (2001).


		@param[in] X D x N matrix of X values, with data points in columns. Should NOT contain a row with all 1's.
		@param[in] y Y vector with length N.
		@param[in] lambda Regularisation strength.
		@tparam DoStandardise Whether to standardise `X` internally.
		@return LassoRegressionResult object with `beta.size() == X.rows() + 1`. If `DoStandardise == true`, `beta`
		will be rescaled and shifted to original `X` units and origins, and `cov` will be transformed accordingly.
		@throw std::invalid_argument If `y.size() != X.cols()` or `X.cols() < X.rows()`.
		@throw std::domain_error If `lambda < 0`.
		@see standardise()
		*/
		template <bool DoStandardise> LassoRegressionResult lasso(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, double lambda);

		/** @brief Carries out multivariate Lasso regression with intercept, standardising `X` inputs internally.
		@see lasso().
		*/
		template <> DLL_DECLSPEC LassoRegressionResult lasso<true>(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, double lambda);

		/** @brief Carries out multivariate Lasso regression with intercept, assuming standardised `X` inputs.
		@see lasso().
		*/
		template <> DLL_DECLSPEC LassoRegressionResult lasso<false>(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, double lambda);

		/** @brief Carries out multivariate Lasso regression with intercept, allowing the user switch internal standardisation of `X` data on or off.
		@see lasso().
		*/
		inline LassoRegressionResult lasso(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, double lambda, bool do_standardise)
		{
			if (do_standardise) {
				return lasso<true>(X, y, lambda);
			} else {
				return lasso<false>(X, y, lambda);
			}
		}

		/** @brief Calculates the PRESS statistic (Predicted Residual Error Sum of Squares). 

		See https://en.wikipedia.org/wiki/PRESS_statistic for details.

		@warning When calculating PRESS for regularised OLS, `regression` must standardise the data internally (call ridge() with `DoStandardise == true`).

		@tparam Regression Functor type implementing particular regression.
		@param[in] X D x N matrix of X values, with data points in columns.
		@param[in] y Y vector with length N.
		@param[in] regression Regression functor. `regression(X, y)` should return a result object supporting a `predict(X)` call (e.g. MultivariateOLSResult). Must standardise the data internally if necessary.
		@return Value of the PRESS statistic.
		@throw std::invalid_argument If `X.cols() != y.size()`.
		*/
		template <class Regression> double press(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Regression regression)
		{
			const auto trainer = [&regression](const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y) -> auto {
				return regression(X, y);
			};
			const auto tester = [](const decltype(regression(X, y))& result, const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y) -> double {
				return (y - result.predict(X)).squaredNorm() / static_cast<double>(y.size());
			};
			return Crossvalidation::leave_one_out(X, y, trainer, tester) * static_cast<double>(y.size());
		}

		/** @brief Calculates the PRESS statistic (Predicted Residual Error Sum of Squares) for univariate regression.

		See https://en.wikipedia.org/wiki/PRESS_statistic for details.

		@param[in] x X vector with length N.
		@param[in] y Y vector with same length as `x`.
		@tparam WithIntercept Whether the regression is with intercept or not.
		@return Value of the PRESS statistic.
		@throw std::invalid_argument If `x.size() != y.size()`.
		*/
		template <bool WithIntercept> double press_univariate(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y)
		{
			const auto trainer = [](const Eigen::Ref<const Eigen::VectorXd> x, const Eigen::Ref<const Eigen::VectorXd> y) -> UnivariateOLSResult {
				if (WithIntercept) {
					return univariate(x, y);
				} else {
					return univariate_without_intercept(x, y);
				}
			};
			const auto tester = [](const UnivariateOLSResult& result, const Eigen::Ref<const Eigen::VectorXd> x, const Eigen::Ref<const Eigen::VectorXd> y) -> double {
				return (y - result.predict(x)).squaredNorm() / static_cast<double>(y.size());
			};
			return Crossvalidation::leave_one_out_scalar(x, y, trainer, tester) * static_cast<double>(y.size());
		}



		/** @brief Adds another row with 1s in every column to X.
		@param[in] X Matrix of independent variables with data points in columns.
		@return New matrix with a row filled with 1's added at the end.
		@throw std::invalid_argument If `X.cols() == 0`.		
		*/
		DLL_DECLSPEC Eigen::MatrixXd add_ones(Eigen::Ref<const Eigen::MatrixXd> X);

		/** @brief Standardises independent variables.

		From each row, `standardise` subtracts its mean and divides it by its standard deviation.

		@param[in, out] X Matrix of independent variables with data points in columns.
		@throw std::invalid_argument If any row of `X` has all values the same, or `X` is empty.
		*/
		DLL_DECLSPEC void standardise(Eigen::Ref<Eigen::MatrixXd> X);

		/** @brief Standardises independent variables.

		From each row, `standardise` subtracts its mean and divides it by its standard deviation.		

		This version of `standardise` saves original mean and standard deviation for
		every row in provided vectors.

		@param[in, out] X D x N matrix of independent variables with data points in columns.
		@param[out] means At exit has length D and contains means of rows of `X`.
		@param[out] standard_deviations At exit has length D and contains standard deviations of rows of `X`. 
			If `means` and `standard_deviations` refer to the same vector, at exit this vector will contain the standard deviations.
		@throw std::invalid_argument If any row of `X` has all values the same, or `X` is empty.
		*/
		DLL_DECLSPEC void standardise(Eigen::Ref<Eigen::MatrixXd> X, Eigen::VectorXd& means, Eigen::VectorXd& standard_deviations);

		/** @brief Reverses the outcome of standardise().

		From each row, `standardise` multiplies it by its standard deviation and adds its mean.

		@param[in, out] X D x N matrix of standardised independent variables with data points in columns.
		@param[in] means Means of rows of `X`.
		@param[in] standard_deviations Standard deviations of rows of `X`.		
		@throw std::invalid_argument If `X.rows() != means.size()` or `X.rows() != standard_deviations.size()`.
		@throw std::domain_error If any element of `standard_deviations` is not positive.
		*/
		DLL_DECLSPEC void unstandardise(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> means, Eigen::Ref<const Eigen::VectorXd> standard_deviations);

		/** @brief Calculates X*X^T, inverts it, and calculates beta.

		Calculates the decomposition of \f$ X X^T + \lambda I \f$ and returns the result of

		\f$ (X X^T + \lambda I)^{-1} X \vec{y} \f$.

		@param lambda Ridge regularisation constant for every feature.
		@private Shared between multiple linear regression algorithms.
		*/
		Eigen::VectorXd calculate_XXt_beta(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<Eigen::MatrixXd> XXt, Eigen::LDLT<Eigen::MatrixXd>& xxt_decomp, const Eigen::Ref<const Eigen::VectorXd> lambda);
	}
}
