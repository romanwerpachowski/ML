#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "ML/LinearRegression.hpp"
#include "types.hpp"


namespace py = pybind11;


namespace ml
{
	namespace LinearRegression
	{
		static MultivariateOLSResult multivariate_row_major(const Eigen::Ref<const MatrixXdR> X, const Eigen::Ref<const Eigen::VectorXd> y/*, bool add_ones*/)
		{
			Eigen::Ref<const Eigen::MatrixXd> XT = X.transpose();
			assert(XT.data() == X.data()); // No copying.
			//if (!add_ones) {
				return multivariate(XT, y);
			/*}
			else {
				// TODO: doesn't work now. Fix it.
				const auto XT_with_ones = LinearRegression::add_ones(XT);
				return multivariate(XT_with_ones, y);
			}*/
		}
	}	
}


void init_linear_regression(py::module& m)
{
	auto m_lin_reg = m.def_submodule("linear_regression", "Linear regression algorithms.");

	py::class_<ml::LinearRegression::UnivariateOLSResult>(m_lin_reg, "UnivariateOLSResult")
		.def("__repr__", &ml::LinearRegression)
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

	py::class_<ml::LinearRegression::MultivariateOLSResult>(m_lin_reg, "MultivariateOLSResult")
		.def_readonly("n", &ml::LinearRegression::MultivariateOLSResult::n, "Number of data points.")
		.def_readonly("dof", &ml::LinearRegression::MultivariateOLSResult::dof, "Number of degrees of freedom.")
		.def_readonly("var_y", &ml::LinearRegression::MultivariateOLSResult::var_y, "Estimated variance of observations Y.")
		.def_readonly("r2", &ml::LinearRegression::MultivariateOLSResult::r2, "R2 = 1 - fraction of variance unexplained relative to a \"base model\".")
		.def_readonly("beta", &ml::LinearRegression::MultivariateOLSResult::beta, "Fitted coefficients of the model y_i = beta^T X_i.")
		.def_readonly("cov", &ml::LinearRegression::MultivariateOLSResult::cov, "Covariance matrix of beta coefficients.")
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

	m_lin_reg.def("multivariate", &ml::LinearRegression::multivariate_row_major, py::arg("X"), py::arg("y"),/* py::arg("add_ones") = false,*/
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