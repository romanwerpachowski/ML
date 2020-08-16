#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "ML/LinearRegression.hpp"


namespace py = pybind11;


void init_linear_regression(py::module& m)
{
	auto m_lin_reg = m.def_submodule("linear_regression", "Linear regression algorithms.");

	py::class_<ml::LinearRegression::UnivariateOLSResult>(m_lin_reg, "UnivariateOLSResult")
		.def_readonly("slope", &ml::LinearRegression::UnivariateOLSResult::slope, "Coefficient multiplying X values when predicting Y.")
		.def_readonly("intercept", &ml::LinearRegression::UnivariateOLSResult::intercept, "Constant added to slope * X when predicting Y.")
		.def_readonly("var_slope", &ml::LinearRegression::UnivariateOLSResult::var_slope, "Estimated variance of the slope.")
		.def_readonly("var_intercept", &ml::LinearRegression::UnivariateOLSResult::var_intercept, "Estimated variance of the intercept.")
		.def_readonly("cov_slope_intercept", &ml::LinearRegression::UnivariateOLSResult::cov_slope_intercept, "Estimated covariance of the slope and the intercept.")
		.doc() = R"(Result of univariate Ordinary Least Squares regression (with or without intercept).

The following properties assume independent Gaussian error terms: `var_slope`, `var_intercept` and `cov_slope_intercept`.)";

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
}