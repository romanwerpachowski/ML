#pragma once
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	/** Linear regression algorithms. */
	namespace LinearRegression {

		/** Result of 1D linear regression. 

		hat{y} = x * slope + intercept.
		*/
		struct UnivariateLinearRegressionResult
		{
			double slope;
			double intercept;
			double correlation; /**< Estimated linear correlation between Y and X. */
			double r2; /**< R^2 coefficient = correlation^2. */
			/** The following assume independent Gaussian error terms (OLS). */
			double var_y; /**< Estimated variance of observations y_i. */
			double var_slope; /**< Estimated variance of the slope. */
			double var_intercept; /**< Estimated variance of the intercept. */
			double cov_slope_intercept; /**< Estimated covariance of the slope and the intercept. */
		};

		/** Carry out univariate (aka simple) linear regression.
		@param x X vector.
		@param y Y vector.
		@throw std::invalid_argument If x and y have different sizes, or if their size is less than 2.
		*/
		DLL_DECLSPEC UnivariateLinearRegressionResult univariate(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y);

		/** Carry out univariate (aka simple) linear regression on regularly spaced points.
		@param x0 First X value.
		@param dx X increment.
		@param y Y vector.
		@throw std::invalid_argument If x and y have different sizes, or if their size is less than 2.
		*/
		DLL_DECLSPEC UnivariateLinearRegressionResult univariate(double x0, double dx, Eigen::Ref<const Eigen::VectorXd> y);
	}
}