/* (C) 2021 Roman Werpachowski. */
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "ML/LogisticRegression.hpp"
#include "types.hpp"


namespace py = pybind11;


namespace ml
{
    namespace LogisticRegressionPython
    {
        /**
         * @brief Wraps Logistic Regression result to accept the feature matrix in row-major order.
         * @tparam R Wrapped result class.
        */
        template <class R> struct Result: public R
        {
            /** Construct the result from user data. */
            template <class ... Types> Result(Types ... args)
                : R{ args... }
            {}

            /** Wrap C++ result. */
            Result(R&& wrapped)
                : R(std::move(wrapped))
            {}

            /** Wrap const C++ result. */
            Result(const R& wrapped)
                : R(wrapped)
            {}

            /**
             * @brief Predicts labels for features X given w. Version which returns a new vector.
             * @param X N x D matrix of X values, with data points in rows.
             * @return Vector with -1 or 1 values.
             * @throw std::invalid_argument If matrix or vector dimensions do not match.
            */
            Eigen::VectorXd predict_row_major(Eigen::Ref<const MatrixXdR> X) const
            {
                return this->predict(X.transpose());
            }
        };

        /**
         * @brief Wraps a Logistic Regression model to accept the feature matrix in row-major order.
         * @tparam M Wrapped model class.
         * @tparam R Wrapped result class.
        */
        template <class M, class R> class Model : public M
        {
        public:
            /** Construct the result from user data. */
            template <class ... Types> Model(Types ... args)
                : M{ args... }
            {}

            /** Wrap C++ result. */
            Model(M&& wrapped)
                : M(std::move(wrapped))
            {}

            /** Wrap const C++ result. */
            Model(const M& wrapped)
                : M(wrapped)
            {}

            /**
             * @brief Fits the model and returns the result.
             *
             * If fitting with intercept is desired, include a column of 1's in the X values.
             *
             * @param X N x D matrix of X values, with data points in rows.
             * @param y Y vector with length N. Values should be -1 or 1.
             * @throw std::invalid_argument if N or D are zero, or if dimensions of `X` and `y` do not match.
             * @return Result object.
            */
            Result<R> fit_row_major(Eigen::Ref<const MatrixXdR> X, Eigen::Ref<const Eigen::VectorXd> y) const
            {
                return Result<R>(this->fit(X.transpose(), y));
            }
        };

        class WrappedAbstractLogisticRegression : public Model<AbstractLogisticRegression, LogisticRegression::Result>
        {
        public:
            using AbstractLogisticRegression::lam;
        };
    }
}

void init_logistic_regression(py::module& m)
{
    auto m_log_reg = m.def_submodule("logistic_regression", "Logistic regression algorithms.");

    typedef ml::LogisticRegressionPython::Result<ml::LogisticRegression::Result> WrappedResult;
    py::class_<WrappedResult> result(m_log_reg, "Result");
    result.def("__repr__", &WrappedResult::to_string)
        .def_readonly("w", &WrappedResult::w, "Fitted coefficients of the LR model.")
        .def_readonly("steps_taken", &WrappedResult::steps_taken, "Number of steps taken to converge.")
        .def_readonly("converged", &WrappedResult::converged, "Did it converge?")
        .def("predict", &WrappedResult::predict_row_major, py::arg("X"), "Predicts labels for features X given w. Returns the predicted label vector.");

    typedef ml::LogisticRegressionPython::Model<ml::AbstractLogisticRegression, ml::LogisticRegression::Result> WrappedAbstractLogisticRegression;
    py::class_<WrappedAbstractLogisticRegression> abstract_logistic_regression(m_log_reg, "LogisticRegression");
    abstract_logistic_regression.def("fit", &WrappedAbstractLogisticRegression::fit_row_major, py::arg("X"), py::arg("y"), R"(Fits the model and returns the result.

If fitting with intercept is desired, include a column of 1's in the X values.

Args:
    X: N x D matrix of X values, with data points in rows.
    y: Y vector with length N. Values should be -1 or 1.

Returns:
    Instance of `Result`.
)")
        .def_property_readonly("lam", &WrappedAbstractLogisticRegression::lam, "Regularisation parameter: inverse variance of the Gaussian prior for `w`")
        .def_property_readonly("weight_absolute_tolerance", &WrappedAbstractLogisticRegression::weight_absolute_tolerance, "Absolute tolerance for fitted weights")
        .def_property_readonly("weight_relative_tolerance", &WrappedAbstractLogisticRegression::weight_relative_tolerance, "Relative tolerance for fitted weights")
        .def_property_readonly("maximum_steps", &WrappedAbstractLogisticRegression::maximum_steps, "Maximum number of steps allowed")
        .def("set_lam", &WrappedAbstractLogisticRegression::set_lam, py::arg("lam"), "Sets the regularisation parameter.")
        .def("set_weight_absolute_tolerance", &WrappedAbstractLogisticRegression::set_weight_absolute_tolerance, py::arg("weight_absolute_tolerance"), "Sets absolute tolerance for weight convergence.")
        .def("set_weight_relative_tolerance", &WrappedAbstractLogisticRegression::set_weight_relative_tolerance, py::arg("weight_relative_tolerance"), "Sets relative tolerance for weight convergence.")
        .def("set_maximum_steps", &WrappedAbstractLogisticRegression::set_maximum_steps, py::arg("maximum_steps"), "Sets maximum number of steps.");

    typedef ml::LogisticRegressionPython::Model<ml::ConjugateGradientLogisticRegression, ml::LogisticRegression::Result> WrappedConjugateGradientLogisticRegression;
    py::class_<WrappedConjugateGradientLogisticRegression> conjugate_gradient_logistic_regression(m_log_reg, "ConjugateGradientLogisticRegression", abstract_logistic_regression);
    conjugate_gradient_logistic_regression.def(py::init()).doc() = R"(Binomial logistic regression algorithm.

Implemented the conjugate gradient algorithm described in Thomas P. Minka, "A comparison of numerical optimizers for logistic regression".
)";
}