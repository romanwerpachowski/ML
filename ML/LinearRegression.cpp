#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <Eigen/Cholesky>
#include "LinearRegression.hpp"

namespace ml
{
	namespace LinearRegression
	{
		std::string UnivariateOLSResult::to_string() const
		{
			std::stringstream s;
			s << "UnivariateOLSResult(";
			s << "n=" << n << ", dof=" << dof << ", r2=" << r2 << ", var_y=" << var_y;
			s << ", slope=" << slope << ", intercept=" << intercept;
			s << ", var_slope=" << var_slope << ", var_intercept=" << var_intercept << ", cov_slope_intercept=" << cov_slope_intercept << ")";
			return s.str();
		}

		std::string MultivariateOLSResult::to_string() const
		{
			std::stringstream s;
			s << "MultivariateOLSResult(";
			s << "n=" << n << ", dof=" << dof << ", r2=" << r2 << ", var_y=" << var_y;
			s << ", beta=[" << beta.transpose() << "]";
			s << ", cov=[" << cov << "])";
			return s.str();
		}

		MultivariateOLSResult::~MultivariateOLSResult()
		{}

		static UnivariateOLSResult calc_univariate_linear_regression_result(
			const double sxx, const double sxy, const double syy, const double mx,
			const double my, const unsigned int n, const bool with_intercept)
		{
			UnivariateOLSResult result;
			result.n = n;
			const unsigned int num_params = with_intercept ? 2 : 1;
			result.dof = n - num_params;
			result.slope = sxy / sxx;
			result.intercept = my - result.slope * mx;
			const double sse = std::max(0., (syy + result.slope * result.slope * sxx - 2 * result.slope * sxy));
			result.r2 = 1 - sse / syy;
			if (n > num_params) {
				result.var_y = sse / result.dof;
			}
			else {
				result.var_y = std::numeric_limits<double>::quiet_NaN();
			}
			result.var_slope = result.var_y / sxx;
			// sum_{i=1}^n x_i^2 = sxx + n * mx * mx;
			// result.var_intercept = (sxx + n * mx * mx) * result.var_y / sxx / n;
			if (with_intercept) {
				result.var_intercept = result.var_y * (1. / n + mx * mx / sxx);
				result.cov_slope_intercept = -mx * result.var_y / sxx;
			}
			else {
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
			const auto syy = (y_centred * y_centred).sum();
			return calc_univariate_linear_regression_result(sxx, sxy, syy, mx, my, n, true);
		}

		UnivariateOLSResult univariate(const double x0, const double dx, const Eigen::Ref<const Eigen::VectorXd> y)
		{
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
			const auto syy = (y_centred * y_centred).sum();
			return calc_univariate_linear_regression_result(sxx, sxy, syy, mx, my, n, true);
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
			const auto syy = (y.array() * y.array()).sum();
			return calc_univariate_linear_regression_result(sxx, sxy, syy, 0, 0, n, false);
		}

		MultivariateOLSResult multivariate(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y)
		{
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
			const Eigen::MatrixXd XXt(X * X.transpose());
			assert(XXt.rows() == XXt.cols());
			assert(XXt.rows() == X.rows());
			MultivariateOLSResult result;
			result.n = n;
			result.dof = n - q;
			Eigen::LLT<Eigen::MatrixXd> llt;
			llt.compute(XXt);
			result.beta = llt.solve(b);
			assert(result.beta.size() == X.rows());
			const double sse = (y - X.transpose() * result.beta).squaredNorm();
			if (result.dof) {
				result.var_y = sse / result.dof;
			}
			else {
				result.var_y = std::numeric_limits<double>::quiet_NaN();
			}
			result.cov = llt.solve(Eigen::MatrixXd::Identity(q, q)) * result.var_y;
			const auto my = y.mean();
			const auto y_centred = y.array() - my;
			const auto syy = (y_centred * y_centred).sum();
			result.r2 = 1 - sse / syy;
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
	}
}