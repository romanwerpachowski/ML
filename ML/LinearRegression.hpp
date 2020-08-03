#pragma once
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
		};

		/** Result of 1D Ordinary Least Squares regression (with or without intercept).

		When fitting with the intercept, R2 = 1 - \sum_{i=1}^n (y_i - hat{y}_i)^2 / \sum_{i=1}^n (y_i - avg(Y))^2 = Corr(X, Y)^2.

		When fitting without the intercept:
			- the R2 coefficient is calculated using a different method:
				R2 = 1 - \sum_{i=1}^n (y_i - hat{y}_i)^2 / \sum_{i=1}^n (y_i)^2 != Corr(X, Y)^2.
			- intercept , var_intercept and cov_slope_intercept are set to 0.
		*/
		struct UnivariateOLSResult: public Result
		{
			double slope; /**< Coefficient multiplying X values when predicting Y. */
			double intercept; /**< Constant added to slope * X when predicting Y. */
			double r2; /**< 1 - fraction of variance unexplained */
			double var_y; /**< Estimated variance of observations y_i. */
			/** The following assume independent Gaussian error terms. */
			double var_slope; /**< Estimated variance of the slope. */			
			double var_intercept; /**< Estimated variance of the intercept. */
			double cov_slope_intercept; /**< Estimated covariance of the slope and the intercept. */
		};

		/** Carry out univariate (aka simple) linear regression.
		@param x X vector.
		@param y Y vector.
		@throw std::invalid_argument If x and y have different sizes, or if their size is less than 2.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** Carry out univariate (aka simple) linear regression on regularly spaced points.
		@param x0 First X value.
		@param dx X increment.
		@param y Y vector.
		@throw std::invalid_argument If y.size is less than 2.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate(double x0, double dx, Eigen::Ref<const Eigen::VectorXd> y);

		/** Carry out univariate (aka simple) linear regression without intercept.
		@param x X vector.
		@param y Y vector.
		@throw std::invalid_argument If x and y have different sizes, or if their size is less than 1.
		*/
		DLL_DECLSPEC UnivariateOLSResult univariate_without_intercept(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);
	}
}