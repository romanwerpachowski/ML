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
            /** Construct the result from user data. */
            template <class ... Types> Model(Types ... args)
                : R{ args... }
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
             * @brief Fit the model and return the result.
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
    }
}

void init_logistic_regression(py::module& m)
{
    auto m_log_reg = m.def_submodule("logistic_regression", "Logistic regression algorithms.");

    py::class_<ml::LogisticRegressionPython::Result<ml::LogisticRegression::Result>> result(m_log_reg, "Result");
    result.def_readonly("w", &ml::LogisticRegressionPython::Result<ml::LogisticRegression::Result>::w, "Fitted coefficients of the LR model.")
        .def_readonly("steps_taken", &ml::LogisticRegressionPython::Result<ml::LogisticRegression::Result>::steps_taken, "Number of steps taken to converge.")
        .def_readonly("converged", &ml::LogisticRegressionPython::Result<ml::LogisticRegression::Result>::converged, "Did it converge?")
        .def("predict", &ml::LogisticRegressionPython::Result<ml::LogisticRegression::Result>::predict_row_major, py::arg("X"), "Predicts labels for features X given w. Returns the predicted label vector.");

    py::class_<ml::LogisticRegressionPython::Model<ml::LogisticRegression, ml::LogisticRegression::Result>> logistic_regression(m_log_reg, "LogisticRegression");
}