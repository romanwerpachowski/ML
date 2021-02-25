/* (C) 2020 Roman Werpachowski. */
#pragma once
#include "LinearRegression.hpp"

namespace ml
{
    namespace LinearRegression
    {
		/** @brief LASSO solution path calculated using the Least Angle Regression algorithm. */
		struct LeastAngleRegressionResult
		{
			std::vector<LassoRegressionResult> lasso_path; /**<  Solutions along the LASSO path. */
			unsigned int n; /**< Number of data points. */
			unsigned int dof; /**< Number of residual degrees of freedom (e.g. `n - 2` or `n - 1` for univariate regression with or without intercept). */
			double tss; /**< Total sum of squares (TSS, equal to the RSS for the "base model" always returning average Y).*/

			/** @brief Formats the result as string. */
			DLL_DECLSPEC std::string to_string() const;
		};

		/** @brief Carries out Least Angle Regression.

		Implements the Least Angle Regression version of Lasso regression.

		The matrix `X` is either assumed to be standardised (`DoStandardise == false`)
		or is standardised internally (`DoStandardise == true`; requires a matrix copy).

		@param[in] X D x N matrix of X values, with data points in columns. Should not contain a row with all 1's.
		@param[in] y Y vector with length N.
		@tparam DoStandardise Whether to standardise `X` internally.
		*/
		template <bool DoStandardise> LeastAngleRegressionResult least_angle_regression(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out Least Angle Regression, standardising `X` inputs internally.
		*/
		template <> LeastAngleRegressionResult least_angle_regression<true>(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Carries out Least Angle Regression, assuming `X` is already standardized.
		*/
		template <> LeastAngleRegressionResult least_angle_regression<false>(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);
    }
}