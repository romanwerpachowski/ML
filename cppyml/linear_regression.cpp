#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "ML/LinearRegression.hpp"
#include "types.hpp"


namespace py = pybind11;


namespace ml
{
	namespace LinearRegression
	{
		/** Version of MultivariateOLSResult accepting and returning covariance matrix in row-major order. */
		struct MultivariateOLSResultRowMajor : public MultivariateOLSResult
		{
			/** Wrap C++ result. */
			MultivariateOLSResultRowMajor(MultivariateOLSResult&& wrapped)
				: MultivariateOLSResult(std::move(wrapped))
			{}

			/** Wrap const C++ result. */
			MultivariateOLSResultRowMajor(const MultivariateOLSResult& wrapped)
				: MultivariateOLSResult(wrapped)
			{}

			/** Construct the result from user data. */
			MultivariateOLSResultRowMajor(unsigned int n, unsigned int dof, double var_y, double r2, const Eigen::Ref<const Eigen::VectorXd> beta, const Eigen::Ref<const MatrixXdR> cov)
				: MultivariateOLSResult{ n, dof, var_y, r2, beta, cov.transpose() }
			{}

			Eigen::Ref<const MatrixXdR> cov_row_major() const
			{
				return cov.transpose();
			}
		};

		/** Version of "multivariate" taking an X with row-major order. */
		static MultivariateOLSResultRowMajor multivariate_row_major(const Eigen::Ref<const MatrixXdR> X, const Eigen::Ref<const Eigen::VectorXd> y, bool add_ones)
		{
			Eigen::Ref<const Eigen::MatrixXd> XT = X.transpose();
			assert(XT.data() == X.data()); // No copying.
			if (!add_ones) {
				return MultivariateOLSResultRowMajor(multivariate(XT, y));
			}
			else {
				const auto XT_with_ones = LinearRegression::add_ones(XT);
				return MultivariateOLSResultRowMajor(multivariate(XT_with_ones, y));
			}			
		}
	}	
}


void init_linear_regression(py::module& m)
{
	auto m_lin_reg = m.def_submodule("linear_regression", "Linear regression algorithms.");

	py::class_<ml::LinearRegression::UnivariateOLSResult>(m_lin_reg, "UnivariateOLSResult")
		.def(py::init<unsigned int, unsigned int, double, double, double, double, double, double, double>(),
			py::arg("n"), py::arg("dof"), py::arg("var_y"), py::arg("r2"), py::arg("slope"), py::arg("intercept"),
			py::arg("var_slope"), py::arg("var_intercept"), py::arg("cov_slope_intercept"),
			R"(Constructs a new instance of UnivariateOLSResult.

Args:
	n: Number of data points.
	dof: Number of degrees of freedom.
	var_y: Estimated variance of observations Y.
	r2: R2 = 1 - fraction of variance unexplained relative to a "base model" (method-dependent).
	slope: Coefficient multiplying X values when predicting Y.
	intercept: Constant added to slope * X when predicting Y.
	var_slope: Estimated variance of the slope.
	var_intercept: Estimated variance of the intercept.
	cov_slope_intercept: Estimated covariance of the slope and the intercept.
)")
		.def("__repr__", &ml::LinearRegression::UnivariateOLSResult::to_string)
		.def_readonly("n", &ml::LinearRegression::UnivariateOLSResult::n, "Number of data points.")
		.def_readonly("dof", &ml::LinearRegression::UnivariateOLSResult::dof, "Number of degrees of freedom.")
		.def_readonly("var_y", &ml::LinearRegression::UnivariateOLSResult::var_y, "Estimated variance of observations Y.")
		.def_readonly("r2", &ml::LinearRegression::UnivariateOLSResult::r2, "R2 = 1 - fraction of variance unexplained relative to a \"base model\".")
		.def_readonly("slope", &ml::LinearRegression::UnivariateOLSResult::slope, "Coefficient multiplying X values when predicting Y.")
		.def_readonly("intercept", &ml::LinearRegression::UnivariateOLSResult::intercept, "Constant added to slope * X when predicting Y.")
		.def_readonly("var_slope", &ml::LinearRegression::UnivariateOLSResult::var_slope, "Estimated variance of the slope.")
		.def_readonly("var_intercept", &ml::LinearRegression::UnivariateOLSResult::var_intercept, "Estimated variance of the intercept.")
		.def_readonly("cov_slope_intercept", &ml::LinearRegression::UnivariateOLSResult::cov_slope_intercept, "Estimated covariance of the slope and the intercept.")
		.doc() = R"(Result of univariate Ordinary Least Squares regression (with or without intercept).

The following properties assume independent Gaussian error terms: `var_slope`, `var_intercept` and `cov_slope_intercept`.)";

	py::class_<ml::LinearRegression::MultivariateOLSResultRowMajor>(m_lin_reg, "MultivariateOLSResult")
		.def(py::init<unsigned int, unsigned int, double, double, const Eigen::Ref<const Eigen::VectorXd>, const Eigen::Ref<const MatrixXdR>>(),
			py::arg("n"), py::arg("dof"), py::arg("var_y"), py::arg("r2"), py::arg("beta"), py::arg("cov"),
			R"(Constructs a new instance of MultivariateOLSResult.

Args:
	n: Number of data points.
	dof: Number of degrees of freedom.
	var_y: Estimated variance of observations Y.
	r2: R2 = 1 - fraction of variance unexplained relative to a "base model" (method-dependent).
	beta: Fitted coefficients of the model y_i = beta^T X_i.
	cov: Covariance matrix of beta coefficients.
)")
		.def("__repr__", &ml::LinearRegression::MultivariateOLSResultRowMajor::to_string)
		.def_readonly("n", &ml::LinearRegression::MultivariateOLSResultRowMajor::n, "Number of data points.")
		.def_readonly("dof", &ml::LinearRegression::MultivariateOLSResultRowMajor::dof, "Number of degrees of freedom.")
		.def_readonly("var_y", &ml::LinearRegression::MultivariateOLSResultRowMajor::var_y, "Estimated variance of observations Y.")
		.def_readonly("r2", &ml::LinearRegression::MultivariateOLSResultRowMajor::r2, "R2 = 1 - fraction of variance unexplained relative to a \"base model\".")
		.def_readonly("beta", &ml::LinearRegression::MultivariateOLSResultRowMajor::beta, "Fitted coefficients of the model y_i = beta^T X_i.")
		.def_property_readonly("cov", &ml::LinearRegression::MultivariateOLSResultRowMajor::cov_row_major, "Covariance matrix of beta coefficients.")
		.doc() = R"(Result of multivariate Ordinary Least Squares regression.

The `cov` property assumes independent Gaussian error terms.)";

	m_lin_reg.def("univariate", static_cast<ml::LinearRegression::UnivariateOLSResult(*)(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<const Eigen::VectorXd>)>(ml::LinearRegression::univariate),
		py::arg("x"), py::arg("y"), R"(Carries out univariate (aka simple) linear regression with intercept.

R2 coefficient is calculated w/r to a model returning average Y, and is equal to Corr(X, Y)^2:
	R2 = 1 - \sum_{i=1}^n (y_i - hat{y}_i)^2 / \sum_{i=1}^n (y_i - avg(Y))^2.

Args:
	x: X vector.
	y: Y vector. `x` and `y` must have same length not less than 2.

Returns:
	Instance of `UnivariateOLSResult`.
)");

	m_lin_reg.def("univariate_regular", static_cast<ml::LinearRegression::UnivariateOLSResult(*)(double, double, Eigen::Ref<const Eigen::VectorXd>)>(ml::LinearRegression::univariate),
		py::arg("x0"), py::arg("dx"), py::arg("y"), R"(Carries out univariate (aka simple) linear regression with intercept on regularly spaced points.

R2 coefficient is calculated w/r to a model returning average Y, and is equal to Corr(X, Y)^2:
	R2 = 1 - \sum_{i=1}^n (y_i - hat{y}_i)^2 / \sum_{i=1}^n (y_i - avg(Y))^2.

Args:
	x0: First X value.
	dx: X increment.
	y: Y vector with length not less than 2.

Returns:
	Instance of `UnivariateOLSResult`.
)");

	m_lin_reg.def("univariate_without_intercept", static_cast<ml::LinearRegression::UnivariateOLSResult(*)(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<const Eigen::VectorXd>)>(ml::LinearRegression::univariate_without_intercept),
		py::arg("x"), py::arg("y"), R"(Carries out univariate (aka simple) linear regression without intercept.

R2 coefficient is calculated w/r to a model returning 0 and is therefore not equal to Corr(X, Y)^2:
	R2 = 1 - \sum_{i=1}^n (y_i - hat{y}_i)^2 / \sum_{i=1}^n (y_i)^2.

Args:
	x: X vector.
	y: Y vector. `x` and `y` must have same length not less than 1.

Returns:
	Instance of `UnivariateOLSResult` with `intercept`, `var_intercept` and `cov_slope_intercept` set to 0.
)");

	m_lin_reg.def("multivariate", &ml::LinearRegression::multivariate_row_major,
		py::arg("X"), py::arg("y"), py::arg("add_ones") = false,
		R"(Carries out multivariate linear regression.

R2 is always calculated w/r to model returning average Y.
If fitting with intercept is desired, include a row of 1's in the X values.

Args:
	X: X matrix (shape N x D, with D <= N), with data points in rows.
	y: Y vector with length N.

Returns:
	Instance of `MultivariateOLSResult`.
)");
}