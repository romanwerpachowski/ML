#pragma once
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	/** Linear regression algorithms. */
	namespace LinearRegression {

		/** Result of 1D linear regression. */
		struct UnivariateLinearRegressionResult
		{
			double slope;
			double intercept;
			double correlation;
			double r2;
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