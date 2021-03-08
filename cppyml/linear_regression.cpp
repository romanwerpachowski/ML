/* (C)2020 Roman Werpachowski. */
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "ML/LinearRegression.hpp"
#include "ML/RecursiveMultivariateOLS.hpp"
#include "types.hpp"


namespace py = pybind11;


namespace ml
{
    namespace LinearRegression
    {
        /** @brief Linear regression result accepting and returning covariance matrix in row-major order.

        @tparam R Wrapped result struct.
        */
        template <class R> struct ResultWithCovariance : public R
        {
            /** Construct the result from user data. */
            template <class ... Types> ResultWithCovariance(Types ... args)
                : R{ args... }
            {}

            /** Wrap C++ result. */
            ResultWithCovariance(R&& wrapped)
                : R(std::move(wrapped))
            {}

            /** Wrap const C++ result. */
            ResultWithCovariance(const R& wrapped)
                : R(wrapped)
            {}

            Eigen::Ref<const MatrixXdR> cov_row_major() const
            {
                return this->cov.transpose();
            }
        };

        /** @brief Version of MultivariateOLSResult accepting and returning covariance matrix in row-major order. 
        */
        struct MultivariateOLSResultRowMajor : public ResultWithCovariance<MultivariateOLSResult>
        {
            using ResultWithCovariance<MultivariateOLSResult>::ResultWithCovariance;

            MultivariateOLSResultRowMajor(unsigned int n, unsigned int dof, double rss, double tss, Eigen::Ref<const Eigen::VectorXd> beta, Eigen::Ref<const MatrixXdR> cov)
                : ResultWithCovariance<MultivariateOLSResult>(n, dof, rss, tss, beta, cov.transpose())
            {}
        };

        struct RidgeRegressionResultRowMajor : public ResultWithCovariance<RidgeRegressionResult>
        {
            using ResultWithCovariance<RidgeRegressionResult>::ResultWithCovariance;

            RidgeRegressionResultRowMajor(unsigned int n, unsigned int dof, double rss, double tss, Eigen::Ref<const Eigen::VectorXd> beta, double effective_dof, Eigen::Ref<const MatrixXdR> cov)
                : ResultWithCovariance<RidgeRegressionResult>(n, dof, rss, tss, beta, effective_dof, cov.transpose())
            {}
        };

        /** Version of "multivariate" taking an X with row-major order. */
        static MultivariateOLSResultRowMajor multivariate_row_major(const Eigen::Ref<const MatrixXdR> X, const Eigen::Ref<const Eigen::VectorXd> y, const bool add_ones)
        {
            Eigen::Ref<const Eigen::MatrixXd> XT = X.transpose();
            if (!add_ones) {
                return MultivariateOLSResultRowMajor(multivariate(XT, y));
            }
            else {
                const auto XT_with_ones = LinearRegression::add_ones(XT);
                return MultivariateOLSResultRowMajor(multivariate(XT_with_ones, y));
            }			
        }

        /** Version of RecursiveMultivariateOLS taking X_i with row-major order. */
        class RecursiveMultivariateOLSRowMajor : public RecursiveMultivariateOLS
        {
        public:
            RecursiveMultivariateOLSRowMajor()
                : RecursiveMultivariateOLS()
            {}

            RecursiveMultivariateOLSRowMajor(Eigen::Ref<const MatrixXdR> X, Eigen::Ref<const Eigen::VectorXd> y)
                : RecursiveMultivariateOLS(X.transpose(), y)
            {}

            void update(Eigen::Ref<const MatrixXdR> X, Eigen::Ref<const Eigen::VectorXd> y)
            {
                RecursiveMultivariateOLS::update(X.transpose(), y);
            }
        };

        static RidgeRegressionResultRowMajor ridge_row_major(Eigen::Ref<const MatrixXdR> X, Eigen::Ref<const Eigen::VectorXd> y, const double lambda, const bool do_standardise)
        {
            return RidgeRegressionResultRowMajor(ridge(X.transpose(), y, lambda, do_standardise));
        }

        static double press_cppyml(Eigen::Ref<const MatrixXdR> X, Eigen::Ref<const Eigen::VectorXd> y, const char* regularisation, const double reg_lambda)
        {
            if (!strcmp(regularisation, "ridge")) {
                return press(X.transpose(), y, [reg_lambda](Eigen::Ref<const Eigen::MatrixXd> train_X, Eigen::Ref<const Eigen::VectorXd> train_y) {
                    return ridge<true>(train_X, train_y, reg_lambda);
                    });				
            } else if (!strcmp(regularisation, "none")) {
                return press(X.transpose(), y, [](Eigen::Ref<const Eigen::MatrixXd> train_X, Eigen::Ref<const Eigen::VectorXd> train_y) {
                    return multivariate(train_X, train_y);
                    });
            } else {
                throw std::invalid_argument("Unknown regression type. Valid types: \"none\" or \"ridge\".");
            }
        }

        static double press_univariate_cppyml(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y, const bool with_intercept)
        {
            if (with_intercept) {
                return press_univariate<true>(x, y);
            } else {
                return press_univariate<false>(x, y);
            }
        }
    }	
}


void init_linear_regression(py::module& m)
{
    auto m_lin_reg = m.def_submodule("linear_regression", "Linear regression algorithms.");

    constexpr bool default_do_standardise = false;

    py::class_<ml::LinearRegression::Result> result(m_lin_reg, "Result");
    result.def_readonly("n", &ml::LinearRegression::Result::n, "Number of data points.")
        .def_readonly("dof", &ml::LinearRegression::Result::dof, "Number of residual degrees of freedom (e.g. `n - 2` or `n - 1` for univariate regression with or without intercept).")
        .def_readonly("rss", &ml::LinearRegression::Result::rss, "Residual sum of squares: sum_{i=1}^N (hat{y}_i - y_i)^2.")
        .def_readonly("tss", &ml::LinearRegression::Result::tss, "Total sum of squares: sum_{i=1}^N (y_i - N^{-1} sum_{j=1}^N y_j)^2.")
        .def_property_readonly("var_y", &ml::LinearRegression::Result::var_y, "Estimated variance of observations Y, equal to `rss / dof`.")
        .def_property_readonly("r2", &ml::LinearRegression::Result::r2, "1 - fraction of variance unexplained relative to the base model. Equal to `1 - rss / tss`.")
        .def_property_readonly("adjusted_r2", &ml::LinearRegression::Result::adjusted_r2, "1 - fraction of variance unexplained relative to the base model. Uses sample variances. Equal to `1 - rss * (n - 1) / tss / dof`.");

    py::class_<ml::LinearRegression::UnivariateOLSResult>(m_lin_reg, "UnivariateOLSResult", result)
        .def(py::init<unsigned int, unsigned int, double, double, double, double, double, double, double>(),
            py::arg("n"), py::arg("dof"), py::arg("rss"), py::arg("tss"), py::arg("slope"), py::arg("intercept"),
            py::arg("var_slope"), py::arg("var_intercept"), py::arg("cov_slope_intercept"),
            R"(Constructs a new instance of UnivariateOLSResult.

Args:
    n: Number of data points.
    dof: Number of residual degrees of freedom.
    rss: Residual sum of squares.
    tss: Total sum of squares.
    slope: Coefficient multiplying X values when predicting Y.
    intercept: Constant added to slope * X when predicting Y.
    var_slope: Estimated variance of the slope.
    var_intercept: Estimated variance of the intercept.
    cov_slope_intercept: Estimated covariance of the slope and the intercept.
)")
        .def("__repr__", &ml::LinearRegression::UnivariateOLSResult::to_string)
        .def_readonly("slope", &ml::LinearRegression::UnivariateOLSResult::slope, "Coefficient multiplying X values when predicting Y.")
        .def_readonly("intercept", &ml::LinearRegression::UnivariateOLSResult::intercept, "Constant added to slope * X when predicting Y.")
        .def_readonly("var_slope", &ml::LinearRegression::UnivariateOLSResult::var_slope, "Estimated variance of the slope.")
        .def_readonly("var_intercept", &ml::LinearRegression::UnivariateOLSResult::var_intercept, "Estimated variance of the intercept.")
        .def_readonly("cov_slope_intercept", &ml::LinearRegression::UnivariateOLSResult::cov_slope_intercept, "Estimated covariance of the slope and the intercept.")
        .doc() = R"(Result of univariate Ordinary Least Squares regression (with or without intercept).

The following properties assume independent Gaussian error terms: `var_slope`, `var_intercept` and `cov_slope_intercept`.)";

    py::class_<ml::LinearRegression::MultivariateOLSResultRowMajor>(m_lin_reg, "MultivariateOLSResult", result)
        .def(py::init<unsigned int, unsigned int, double, double, const Eigen::Ref<const Eigen::VectorXd>, const Eigen::Ref<const MatrixXdR>>(),
            py::arg("n"), py::arg("dof"), py::arg("rss"), py::arg("tss"), py::arg("beta"), py::arg("cov"),
            R"(Constructs a new instance of MultivariateOLSResult.

Args:
    n: Number of data points.
    dof: Number of residual degrees of freedom.
    rss: Residual sum of squares.
    tss: Total sum of squares.
    beta: Fitted coefficients of the model y_i = beta^T X_i.
    cov: Covariance matrix of beta coefficients.
)")
        .def("__repr__", &ml::LinearRegression::MultivariateOLSResultRowMajor::to_string)
        .def_readonly("beta", &ml::LinearRegression::MultivariateOLSResultRowMajor::beta, "Fitted coefficients of the model y_i = beta^T X_i.")
        .def_property_readonly("cov", &ml::LinearRegression::MultivariateOLSResultRowMajor::cov_row_major, "Covariance matrix of beta coefficients.")
        .doc() = R"(Result of multivariate Ordinary Least Squares regression.

The `cov` property assumes independent Gaussian error terms.
)";

    py::class_<ml::LinearRegression::RidgeRegressionResultRowMajor>(m_lin_reg, "RidgeRegressionResult", result)
        .def(py::init<unsigned int, unsigned int, double, double, const Eigen::Ref<const Eigen::VectorXd>, double, const Eigen::Ref<const MatrixXdR>>(),
            py::arg("n"), py::arg("dof"), py::arg("rss"), py::arg("tss"), py::arg("beta"), py::arg("effective_dof"), py::arg("cov"),
            R"(Constructs a new instance of RidgeRegressionResult.

Args:
    n: Number of data points.
    dof: Number of residual degrees of freedom.
    rss: Residual sum of squares.
    tss: Total sum of squares.
    beta: Fitted coefficients of the model y_i = beta'^T X_i + beta0, in which beta' is regularised and beta0 is not.
    cov: Covariance matrix of beta coefficients.
    effective_dof: Effective number of residual degrees of freedom: N - tr [ X^T (X * X^T + lambda * I)^{-1} X ] - 1.
)")
        .def("__repr__", &ml::LinearRegression::RidgeRegressionResultRowMajor::to_string)
        .def_readonly("beta", &ml::LinearRegression::RidgeRegressionResultRowMajor::beta, "Fitted coefficients of the model y_i = beta'^T X_i, followed by beta0.")
        .def_property_readonly("cov", &ml::LinearRegression::RidgeRegressionResultRowMajor::cov_row_major, "Covariance matrix of (beta', beta0) coefficients.")
        .def_readonly("effective_dof", &ml::LinearRegression::RidgeRegressionResultRowMajor::effective_dof, "Effective number of residual degrees of freedom: N - tr [ X^T (X * X^T + lambda * I)^{-1} X ] - 1.")
        .doc() = R"(Result of a (multivariate) ridge regression with intercept.

Does not contain error estimates because they are not easy to estimate reliably for regularised regression.

Intercept is the last coefficient in `beta`.

`var_y` is calculated using `dof` as the denominator.
)";

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
    dx: Positive X increment.
    y: Y vector with length not less than 2.

Returns:
    Instance of `UnivariateOLSResult`.
)");

    m_lin_reg.def("univariate_without_intercept", static_cast<ml::LinearRegression::UnivariateOLSResult(*)(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<const Eigen::VectorXd>)>(ml::LinearRegression::univariate_without_intercept),
        py::arg("x"), py::arg("y"),
        R"(Carries out univariate (aka simple) linear regression without intercept.

The R2 coefficient is calculated w/r to a model returning average Y:
    R2 = 1 - \sum_{i=1}^n (y_i - hat{y}_i)^2 / \sum_{i=1}^n (y_i - avg(Y))^2.

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
If fitting with intercept is desired, include a row of 1's in the X values
or set the parameter `add_ones` to `True`.


Args:
    X: X matrix (shape N x D, with D <= N), with data points in rows.
    y: Y vector with length N.
    add_ones: Whether to automatically add a column of 1's at the end of `X` (optional, defaults to `False`).

Returns:
    Instance of `MultivariateOLSResult`.
)");

    py::class_ <ml::LinearRegression::RecursiveMultivariateOLSRowMajor>(m_lin_reg, "RecursiveMultivariateOLS")
        .def(py::init<>(), "Initialises without data.")
        .def(py::init<Eigen::Ref<const MatrixXdR>, Eigen::Ref<const Eigen::VectorXd>>(),
            py::arg("X"), py::arg("y"),
            R"(Initialises with the first sample and calculates the first beta estimate.

Args:
    X: N x D matrix of X values, with data points in rows and N >= D.
    y: Y vector with length N.
)")
        .def("update", &ml::LinearRegression::RecursiveMultivariateOLSRowMajor::update,
            py::arg("X"), py::arg("y"),
        R"(Updates the beta estimate with a new sample.

Args:
    X: N x D matrix of X values, with data points in rows.
    y: Y vector with length N.

Throws:
    ValueError: If n == 0 (i.e., (X, y) is the first sample) and N < D.
)")
        .def_property_readonly("n", &ml::LinearRegression::RecursiveMultivariateOLSRowMajor::n, "Number of data points seen so far.")
        .def_property_readonly("d", &ml::LinearRegression::RecursiveMultivariateOLSRowMajor::d, "Dimension of data points. If n == 0, returs 0.")
        .def_property_readonly("beta", &ml::LinearRegression::RecursiveMultivariateOLSRowMajor::beta, "Current beta estimate. If n == 0, returns an empty array.")
        .doc() = R"(Given a stream of pairs (X_i, y_i), updates the least-squares estimate for beta solving the equations

        y_0 = X_0^T * beta + e_0
        y_1 = X_1^T * beta + e_1
        ...

        Based on https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/2/436/files/2017/07/22-notes-6250-f16.pdf
)";

    m_lin_reg.def("ridge", &ml::LinearRegression::ridge_row_major,
        py::arg("X"), py::arg("y"), py::arg("lambda"), py::arg("do_standardise") = default_do_standardise,
        R"(Carries out multivariate ridge regression with intercept.

Given X and y, finds beta' and beta0 minimising || y - \beta'^T X - beta0 ||^2 + lambda * || beta' ||^2.

R2 is always calculated w/r to model returning average y. 
The matrix `X` is assumed to be standardised unless `do_standardise` is set to `True`.

Args:
    X: X matrix (shape N x D, with D <= N), with data points in rows.
    y: Y vector with length N.
    do_standardise: Whether to automatically subtract the mean from each row in `X` and divide it by its standard deviation (defaults to False).

Returns:
    Instance of `RidgeRegressionResult`. If `do_standardise` was `True`, the `beta` vector will be rescaled and shifted
    to original `X` units and origins, and the `cov` matrix will be transformed accordingly.
)");

    m_lin_reg.def("press", &ml::LinearRegression::press_cppyml,
        py::arg("X"), py::arg("y"), py::arg("regularisation") = "none", py::arg("reg_lambda") = 0.,
        R"(Calculates the PRESS statistic (Predicted Residual Error Sum of Squares).

See https://en.wikipedia.org/wiki/PRESS_statistic for details.

NOTE: Training data will be standardised internally if using regularisation.

Args:
    X: X matrix (shape N x D, with D <= N), with data points in rows. Unstandardised.
    y: Y vector with length N.
    regularisation: Type of regularisation: "none" or "ridge". Defaults to "none".
    reg_lambda: Non-negative regularisation strength. Defaults to 0. Ignored if `regularisation == "none"`.

Returns:
    Value of the PRESS statistic.
)");

    m_lin_reg.def("press_univariate", &ml::LinearRegression::press_univariate_cppyml,
        py::arg("x"), py::arg("y"), py::arg("with_intercept") = true,
        R"(Calculates the PRESS statistic (Predicted Residual Error Sum of Squares) for univariate regression.

See https://en.wikipedia.org/wiki/PRESS_statistic for details.

Args:
    x: X vector with length N.
    y: Y vector with same length as `x`.
    with_intercept: Whether the regression is with intercept or not (defaults to True).

Returns:
    Value of the PRESS statistic.
)");
}