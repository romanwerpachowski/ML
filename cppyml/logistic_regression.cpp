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
         * @brief Wraps Logistic Regression result to accept Python data structures.
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
}