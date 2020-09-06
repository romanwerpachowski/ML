/* (C) 2020 Roman Werpachowski. */
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <Eigen/Cholesky>
#include "LinearAlgebra.hpp"
#include "LinearRegression.hpp"

namespace ml
{
	namespace LinearRegression
	{	
		static void members_to_string(const Result& result, std::stringstream& s)
		{
			s << "n=" << result.n << ", dof=" << result.dof << ", base_dof=" << result.base_dof << ", rss=" << result.rss << ", tss=" << result.tss;
		}

		std::string UnivariateOLSResult::to_string() const
		{
			std::stringstream s;
			s << "UnivariateOLSResult(";
			members_to_string(static_cast<const Result&>(*this), s);
			s << ", slope=" << slope << ", intercept=" << intercept;
			s << ", var_slope=" << var_slope << ", var_intercept=" << var_intercept << ", cov_slope_intercept=" << cov_slope_intercept;
			s << ")";
			return s.str();
		}

		std::string MultivariateOLSResult::to_string() const
		{
			std::stringstream s;
			s << "MultivariateOLSResult(";
			s << "n=" << n << ", dof=" << dof << ", r2=" << r2() << ", var_y=" << var_y();
			s << ", beta=[" << beta.transpose() << "]";
			s << ", cov=[" << cov << "]";
			s << ")";
			return s.str();
		}

		std::string RidgeRegressionResult::to_string() const
		{
			std::stringstream s;
			s << "RidgeRegressionResult(";
			members_to_string(static_cast<const Result&>(*this), s);
			s << ", beta=[" << beta.transpose() << "]";
			s << ", cov=[" << cov << "]";
			s << ", effective_dof=" << effective_dof;
			s << ")";
			return s.str();
		}

		static UnivariateOLSResult calc_univariate_linear_regression_result(
			const double sxx, const double sxy, const double tss, const double mx,
			const double my, const unsigned int n, const bool with_intercept)
		{
			UnivariateOLSResult result;
			result.n = n;
			const unsigned int num_params = with_intercept ? 2 : 1;
			result.dof = n - num_params;
			result.slope = sxy / sxx;
			result.intercept = my - result.slope * mx;
			// Residual sum of squares.			
			result.rss = std::max(0., (tss + result.slope * result.slope * sxx - 2 * result.slope * sxy));
			result.tss = tss;
			result.var_slope = result.var_y() / sxx;
			// sum_{i=1}^n x_i^2 = sxx + n * mx * mx;
			// result.var_intercept = (sxx + n * mx * mx) * result.var_y / sxx / n;
			if (with_intercept) {
				result.base_dof = n - 1;
				result.var_intercept = result.var_y() * (1. / n + mx * mx / sxx);
				result.cov_slope_intercept = -mx * result.var_y() / sxx;
			}
			else {
				result.base_dof = n;
				result.var_intercept = result.cov_slope_intercept = 0;
			}			
			return result;
		}

		UnivariateOLSResult univariate(const Eigen::Ref<const Eigen::VectorXd> x, const Eigen::Ref<const Eigen::VectorXd> y)
		{
			const auto n = static_cast<unsigned int>(x.size());
			if (n != y.size()) {
				throw std::invalid_argument("X and Y vectors have different sizes");
			}
			if (n < 2) {
				throw std::invalid_argument("Need at least 2 points for regresssion");
			}
			const auto mx = x.mean();
			const auto my = y.mean();
			const auto x_centred = x.array() - mx;
			const auto y_centred = y.array() - my;			
			const auto sxy = (x_centred * y_centred).sum();
			const auto sxx = (x_centred * x_centred).sum();
			// Total sum of squares:
			const auto tss = (y_centred * y_centred).sum();
			return calc_univariate_linear_regression_result(sxx, sxy, tss, mx, my, n, true);
		}

		UnivariateOLSResult univariate(const double x0, const double dx, const Eigen::Ref<const Eigen::VectorXd> y)
		{
			if (dx <= 0) {
				throw std::domain_error("dx must be positive");
			}
			const auto n = static_cast<unsigned int>(y.size());
			if (n < 2) {
				throw std::invalid_argument("Need at least 2 points for regresssion");
			}
			const auto half_width_x = (n - 1) * dx / 2;
			const auto mx = x0 + half_width_x;
			const auto my = y.mean();
			const auto y_centred = y.array() - my;
			const auto indices = Eigen::VectorXd::LinSpaced(y.size(), 0, n - 1);
			const auto sxy = dx * (y_centred * indices.array()).sum();
			const auto sxx = dx * dx * n * (n * n - 1) / 12.;
			// Total sum of squares:
			const auto tss = (y_centred * y_centred).sum();
			return calc_univariate_linear_regression_result(sxx, sxy, tss, mx, my, n, true);
		}

		UnivariateOLSResult univariate_without_intercept(Eigen::Ref<const Eigen::VectorXd> x, const Eigen::Ref<const Eigen::VectorXd> y)
		{
			const auto n = static_cast<unsigned int>(x.size());
			if (n != y.size()) {
				throw std::invalid_argument("X and Y vectors have different sizes");
			}
			if (n < 1) {
				throw std::invalid_argument("Need at least 1 point for regresssion without intercept");
			}
			const auto sxy = (x.array() * y.array()).sum();
			const auto sxx = (x.array() * x.array()).sum();
			// Total sum of squares:
			const auto tss = (y.array() * y.array()).sum();
			return calc_univariate_linear_regression_result(sxx, sxy, tss, 0, 0, n, false);
		}

		Eigen::VectorXd calculate_XXt_beta(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<Eigen::MatrixXd> XXt, Eigen::LDLT<Eigen::MatrixXd>& xxt_decomp, const double lambda)
		{
			if (lambda < 0) {
				throw std::domain_error("Ridge regularisation constant cannot be negative");
			}
			// X is an q x N matrix and y is a N-size vector.
			const auto n = static_cast<unsigned int>(X.cols());
			if (n != y.size()) {
				throw std::invalid_argument("X matrix has different number of data points than Y has values");
			}
			const auto q = static_cast<unsigned int>(X.rows());
			if (n < q) {
				throw std::invalid_argument("Not enough data points for regression");
			}
			const Eigen::VectorXd b(X * y);
			assert(b.size() == X.rows());
			XXt.noalias() = X * X.transpose();
			if (lambda) {
				XXt += lambda * Eigen::MatrixXd::Identity(q, q);
			}
			assert(XXt.rows() == XXt.cols());
			assert(XXt.rows() == X.rows());
			xxt_decomp.compute(XXt);
			return xxt_decomp.solve(b);
		}

		MultivariateOLSResult multivariate(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y)
		{
			// X is an q x N matrix and y is a N-size vector.
			const auto q = static_cast<unsigned int>(X.rows());
			const auto n = static_cast<unsigned int>(X.cols());
			Eigen::LDLT<Eigen::MatrixXd> xxt_decomp;
			MultivariateOLSResult result;
			Eigen::MatrixXd XXt(q, q);
			result.beta = calculate_XXt_beta(X, y, XXt, xxt_decomp, 0);
			result.n = n;
			result.dof = n - q;
			result.base_dof = n - 1;
			assert(result.beta.size() == X.rows());
			// Residual sum of squares:
			result.rss = (y - X.transpose() * result.beta).squaredNorm();
			if (result.dof) {
				result.cov = xxt_decomp.solve(Eigen::MatrixXd::Identity(q, q));
				result.cov *= result.var_y();
			} else {
				result.cov = Eigen::MatrixXd::Constant(q, q, std::numeric_limits<double>::quiet_NaN());
			}
			
			const auto my = y.mean();
			const auto y_centred = y.array() - my;
			// Total sum of squares:
			result.tss = y_centred.square().sum();
			return result;
		}

		template <> RidgeRegressionResult ridge<false>(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const double lambda)
		{
			// X is an q x N matrix and y is a N-size vector.
			const auto q = X.rows();
			const auto n = X.cols();
			RidgeRegressionResult result;
			result.n = static_cast<unsigned int>(n);
			result.dof = static_cast<unsigned int>(n - q - 1); // -1 for the intercept.
			result.base_dof = result.n - 1;
			result.beta.resize(q + 1);
			const double intercept = y.mean();
			result.beta[q] = intercept;
			Eigen::MatrixXd XXt(q, q);
			Eigen::LDLT<Eigen::MatrixXd> xxt_decomp;
			result.beta.head(q) = calculate_XXt_beta(X, y, XXt, xxt_decomp, lambda);			
			// Use the fact that intercept == mean(y).
			const Eigen::VectorXd y_centred(y.array() - intercept);
			// Residual sum of squares:
			result.rss = (y_centred - X.transpose() * result.beta.head(q)).squaredNorm();
			// Total sum of squares:
			result.tss = y_centred.squaredNorm();
			if (lambda > 0) {
				result.effective_dof = std::max(static_cast<double>(n) - (X.transpose() * xxt_decomp.solve(X)).trace() - 1, static_cast<double>(result.dof));
			}
			else {
				result.effective_dof = result.dof;
			}
			result.cov.resize(q + 1, q + 1);
			// Before multiplying by Var(Y).
			// Var(intercept):
			result.cov(q, q) = 1. / static_cast<double>(n);
			// Cov(slopes):
			auto cov_slopes = result.cov.block(0, 0, q, q);
			if (lambda > 0) {				
				const Eigen::MatrixXd inv_xxt_lambda(xxt_decomp.solve(Eigen::MatrixXd::Identity(q, q)));
				cov_slopes.noalias() = XXt * inv_xxt_lambda;
				XXt.noalias() = inv_xxt_lambda * cov_slopes;
				cov_slopes = XXt;
			}
			else {
				cov_slopes = xxt_decomp.solve(Eigen::MatrixXd::Identity(q, q));
			}
			// Cov(intercept, slopes) is zero by assumption of standardisation.
			result.cov.col(q).head(q).setZero();
			result.cov.row(q).head(q).setZero();
			// Scale by Var(Y):
			result.cov *= result.var_y();
			return result;
		}

		template <> RidgeRegressionResult ridge<true>(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const double lambda)
		{
			Eigen::MatrixXd workX(X);
			Eigen::VectorXd means;
			Eigen::VectorXd standard_deviations;
			standardise(workX, means, standard_deviations);
			auto result = ridge<false>(workX, y, lambda);
			const auto q = X.rows();
			auto slopes = result.beta.head(q);
			// Using Matlab notation: ./ and .* are elementwise / and *.
			// new_slopes = slopes ./ standard_deviations
			slopes.array() /= standard_deviations.array();
			auto cov_slopes = result.cov.block(0, 0, q, q);
			// We only need to rescale this part, because Cov(slopes, intercept) == 0.
			// Cov(new_slopes, new_slopes) = Cov(slopes, slopes) ./ (standard_deviations * standard_deviations^T)
			// Cov(new_slopes, intercept) = Cov(slopes, intercept) = 0
			cov_slopes.array() /= (standard_deviations * standard_deviations.transpose()).array();
			// new_intercept = intercept - new_slopes^T * means
			// Cov(new_slopes, new_intercept) = - Cov(new_slopes) * means
			// Var(new_intercept) = Cov(intercept - new_slopes^T * means, intercept - new_slopes^T * means) = Var(intercept) + means^T * Cov(new_slopes, new_slopes) * means
			result.beta[q] -= slopes.dot(means);
			result.cov.col(q).head(q) = - cov_slopes * means;
			result.cov.row(q).head(q) = result.cov.col(q).head(q);
			result.cov(q, q) += LinearAlgebra::xAx_symmetric(cov_slopes, means);
			return result;
		}

		Eigen::MatrixXd add_ones(const Eigen::Ref<const Eigen::MatrixXd> X)
		{
			if (!X.cols()) {
				throw std::invalid_argument("No data points in X");
			}
			Eigen::MatrixXd X_with_intercept(X.rows() + 1, X.cols());
			X_with_intercept.topRows(X.rows()) = X;
			X_with_intercept.bottomRows(1) = Eigen::RowVectorXd::Ones(X.cols());
			return X_with_intercept;
		}

		void standardise(Eigen::Ref<Eigen::MatrixXd> X)
		{
			Eigen::VectorXd w;
			return standardise(X, w, w);
		}

		void standardise(Eigen::Ref<Eigen::MatrixXd> X, Eigen::VectorXd& means, Eigen::VectorXd& standard_deviations)
		{
			if (!X.size()) {
				throw std::invalid_argument("Standardising an empty matrix");
			}
			const auto d = X.rows();
			means.resize(d);
			means = X.rowwise().mean();
			X.colwise() -= means;
			standard_deviations.resize(d);
			standard_deviations = X.rowwise().squaredNorm();
			standard_deviations /= static_cast<double>(X.cols());
			standard_deviations = standard_deviations.array().sqrt();
			for (Eigen::Index i = 0; i < d; ++i) {
				const auto sigma = standard_deviations[i];
				if (!sigma) {
					throw std::invalid_argument("At least one row has constant values");
				}
				X.row(i) /= sigma;
			}
		}

		void unstandardise(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> means, Eigen::Ref<const Eigen::VectorXd> standard_deviations)
		{
			const auto d = X.rows();
			if (means.size() != d) {
				throw std::invalid_argument("Incorrect size of means vector");
			}
			if (standard_deviations.size() != d) {
				throw std::invalid_argument("Incorrect size of standard deviations vector");
			}
			for (Eigen::Index i = 0; i < X.rows(); ++i) {
				const auto sigma = standard_deviations[i];
				if (!(sigma > 0)) {
					throw std::domain_error("Standard deviation is not positive");
				}
				X.row(i) *= sigma;
			}
			X.colwise() += means;
		}
	}
}
