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
			double var_y; /**< Estimated variance of observations y_i. */
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
		};

		/** Result of multivariate Ordinary Least Squares regression.		
		*/
		struct MultivariateOLSResult : public Result
		{
			Eigen::VectorXd beta; /**< Fitted coefficients of the model y_i = beta^T X_i. */
			/** The following assume independent Gaussian error terms. */
			Eigen::MatrixXd cov; /**< Covariance matrix of beta coefficients. */
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
		@param dx X increment.
		@param y Y vector.
		@throw std::invalid_argument If y.size() is less than 2.
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

		R2 is always calculated w/r to model returning average Y.

		If fitting with intercept is desired, include a row of 1's in the X values.

		@param X X matrix, with data points in columns.
		@param y Y vector.
		@throw std::invalid_argument If y.size() != X.cols() or y.size() < X.rows().
		*/
		DLL_DECLSPEC MultivariateOLSResult multivariate(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** Add another row with 1s in every column to X.
		@throw std::invalid_argument If X.cols() == 0.
		*/
		DLL_DECLSPEC Eigen::MatrixXd add_intercept(Eigen::Ref<const Eigen::MatrixXd> X);
	}
}