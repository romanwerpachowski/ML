/* (C) 2020 Roman Werpachowski. */
#pragma once
#include <vector>
#include <cassert>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
	/** @brief Methods used for cross-validation. */
	namespace Crossvalidation
	{		
		/** @brief Calculates indices delimiting a fold.

		Calculates i0 and i1 such that the k-th fold consists of data points with indices in the [i0, i1) range.

		@param[in] total_len Total number of data points.
		@param[in] k Fold index with `0 <= k < num_folds`.
		@param[in] num_folds Total number of folds with `num_folds <= total_len`.
		@param[out] i0 Lower fold bound (inclusive).
		@param[out] i1 Upper fold bound (exclusive).

		@throw std::invalid_argument If `num_folds > total_len` or `k >= num_folds`.
		*/
		DLL_DECLSPEC void calc_fold_indices(size_t total_len, unsigned int k, unsigned int num_folds, size_t& i0, size_t& i1);

		/** @brief Returns k-th fold contents for vector data.
		
		@param[in] data Data matrix with data points in columns.
		@param[in] k Fold index with `0 <= k < num_folds`.
		@param[in] num_folds Total number of folds with `num_folds <= data.cols()`.

		@return Reference to data from the `k`-th fold.

		@throw std::invalid_argument If `num_folds > data.cols()` or `k >= num_folds`.
		*/
		DLL_DECLSPEC Eigen::Ref<const Eigen::MatrixXd> only_kth_fold_2d(Eigen::Ref<const Eigen::MatrixXd> data, unsigned int k, unsigned int num_folds);

		/** @brief Returns k-th fold contents for scalar data.

		@param[in] data Data vector.
		@param[in] k Fold index with `0 <= k < num_folds`.
		@param[in] num_folds Total number of folds with `num_folds <= data.size()`.

		@return Reference to data from the `k`-th fold.

		@throw std::invalid_argument If `num_folds > data.size()` or `k >= num_folds`.
		*/
		DLL_DECLSPEC Eigen::Ref<const Eigen::VectorXd> only_kth_fold_1d(Eigen::Ref<const Eigen::VectorXd> data, unsigned int k, unsigned int num_folds);

		/** @brief Returns k-th fold contents for scalar data.

		@param[in] data Data vector.
		@param[in] k Fold index with `0 <= k < num_folds`.
		@param[in] num_folds Total number of folds with `num_folds <= data.size()`.
		@tparam T Scalar data type.

		@return Data from the `k`-th fold.

		@throw std::invalid_argument If `num_folds > data.size()` or `k >= num_folds`.
		*/
		template <class T> std::vector<T> only_kth_fold_1d(const std::vector<T>& data, const unsigned int k, const unsigned int num_folds) {
			
			size_t i0, i1;
			calc_fold_indices(data.size(), k, num_folds, i0, i1);
			return std::vector<T>(data.begin() + i0, data.begin() + i1);
		}

		/** @brief Returns the contents of all except the k-th fold for vector data.

		@param[in] data Data matrix with data points in columns.
		@param[in] k Fold index with `0 <= k < num_folds`.
		@param[in] num_folds Total number of folds with `num_folds <= data.size()`.

		@return Data from all folds except the `k`-th one.

		@throw std::invalid_argument If `num_folds > data.cols()` or `k >= num_folds`.
		*/
		DLL_DECLSPEC Eigen::MatrixXd without_kth_fold_2d(Eigen::Ref<const Eigen::MatrixXd> data, unsigned int k, unsigned int num_folds);

		/** @brief Returns the contents of all except the k-th fold for scalar data.

		@param[in] data Data vector.
		@param[in] k Fold index with `0 <= k < num_folds`.
		@param[in] num_folds Total number of folds with `num_folds <= data.size()`.

		@return Data from all folds except the `k`-th one.

		@throw std::invalid_argument If `num_folds > data.size()` or `k >= num_folds`.
		*/
		DLL_DECLSPEC Eigen::VectorXd without_kth_fold_1d(Eigen::Ref<const Eigen::VectorXd> data, unsigned int k, unsigned int num_folds);

		/** @brief Returns the contents of all except the k-th fold for scalar data.

		@param[in] data Data vector.
		@param[in] k Fold index with `0 <= k < num_folds`.
		@param[in] num_folds Total number of folds with `num_folds <= data.size()`.
		@tparam T Scalar data type.

		@return Data from all folds except the `k`-th one.

		@throw std::invalid_argument If `num_folds > data.size()` or `k >= num_folds`.
		*/
		template <class T> std::vector<T> without_kth_fold_1d(const std::vector<T>& data, unsigned int k, unsigned int num_folds) {
			size_t i0, i1;
			const size_t total_len = data.size();
			calc_fold_indices(total_len, k, num_folds, i0, i1);
			std::vector<T> remaining(total_len - (i1 - i0));
			std::copy(data.begin(), data.begin() + i0, remaining.begin());
			std::copy(data.begin() + i1, data.end(), remaining.begin() + i0);
			return remaining;
		}

		/** @brief Calculates model test error using k-fold cross-validation.

		See https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation

		`train_func` and `test_func` functors should accept X data as `Eigen::Ref<const Eigen::MatrixXd>` reference,
		and y data as `Eigen::Ref<const Eigen::VectorXd>`.

		@param[in] X Matrix with all features (data points in columns).
		@param[in] y Vector with all responses (scalars).
		@param[in] train_func Functor returning a trained model given training features and responses as arguments.
		@param[in] test_func Functor calculating test error per data point given the model, test features and test responses as arguments.
		@param[in] num_folds Number of folds.

		@tparam Trainer Functor type for model training.
		@tparam Tester Functor type for calculating test error.

		@return Error value per data point.

		@throw std::invalid_argument If `num_folds > X.cols()` or `y.size() != X.cols()`.
		*/
		template <class Trainer, class Tester> double k_fold(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Trainer train_func, Tester test_func, const unsigned int num_folds)
		{
			if (X.cols() != y.size()) {
				throw std::invalid_argument("Data size mismatch");
			}
			double sum_weighted_errors = 0;
			for (unsigned int k = 0; k < num_folds; ++k) {
				const Eigen::MatrixXd train_X(without_kth_fold_2d(X, k, num_folds));
				const Eigen::VectorXd train_y(without_kth_fold_1d(y, k, num_folds));
				const auto test_X = only_kth_fold_2d(X, k, num_folds);
				const auto test_y= only_kth_fold_1d(y, k, num_folds);
				auto trained_model = train_func(train_X, train_y);
				const double test_error = test_func(trained_model, test_X, test_y);
				sum_weighted_errors += test_error * static_cast<double>(test_y.size());
			}
			return sum_weighted_errors / static_cast<double>(y.size());
		}

		/** @brief Calculates model test error using k-fold cross-validation (scalar X version).

		See https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation

		`train_func` and `test_func` functors should accept X and y data as `Eigen::Ref<const Eigen::VectorXd>`
		references.

		@param[in] x Vector with all features (scalars).
		@param[in] y Vector with all responses (scalars).
		@param[in] train_func Functor returning a trained model given training features and responses as arguments.
		@param[in] test_func Functor calculating test error per data point given the model, test features and test responses as arguments.
		@param[in] num_folds Number of folds.

		@tparam Trainer Functor type for model training.
		@tparam Tester Functor type for calculating test error.

		@return Error value per data point.

		@throw std::invalid_argument If `num_folds > X.cols()` or `y.size() != y.size()`.
		*/
		template <class Trainer, class Tester> double k_fold_scalar(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y, Trainer train_func, Tester test_func, const unsigned int num_folds)
		{
			if (x.size() != y.size()) {
				throw std::invalid_argument("Data size mismatch");
			}
			double sum_weighted_errors = 0;
			for (unsigned int k = 0; k < num_folds; ++k) {
				const Eigen::VectorXd train_X(without_kth_fold_1d(x, k, num_folds));
				const Eigen::VectorXd train_y(without_kth_fold_1d(y, k, num_folds));
				const auto test_X = only_kth_fold_1d(x, k, num_folds);
				const auto test_y = only_kth_fold_1d(y, k, num_folds);
				auto trained_model = train_func(train_X, train_y);
				const double test_error = test_func(trained_model, test_X, test_y);
				sum_weighted_errors += test_error * static_cast<double>(test_y.size());
			}
			return sum_weighted_errors / static_cast<double>(y.size());
		}

		/** @brief Calculates model test error using leave-one-out cross-validation.

		See https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation

		`train_func` and `test_func` functors should accept X data as `Eigen::Ref<const Eigen::MatrixXd>` reference,
		and y data as `Eigen::Ref<const Eigen::VectorXd>`.

		@param[in] X Matrix with all features (data points in columns).
		@param[in] y Vector with all responses (scalars).
		@param[in] train_func Functor returning a trained model given training features and responses as arguments.
		@param[in] test_func Functor calculating test error per data point given the model, test features and test responses as arguments.

		@tparam Trainer Functor type for model training.
		@tparam Tester Functor type for calculating test error.

		@return Error value per data point.

		@throw std::invalid_argument If `y.size() < 2` or `y.size() != X.cols()`.
		*/
		template <class Trainer, class Tester> double leave_one_out(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Trainer train_func, Tester test_func)
		{
			const Eigen::Index n = y.size();
			if (n < 2) {
				throw std::invalid_argument("Too few data points");
			}
			if (X.cols() != n) {
				throw std::invalid_argument("Data size mismatch");
			}
			double sum_weighted_errors = 0;			
			Eigen::MatrixXd train_X(X.rows(), n - 1);
			Eigen::VectorXd train_y(n - 1);
			for (Eigen::Index k = 0; k < n; ++k) {
				if (k) {
					train_X.leftCols(k) = X.leftCols(k);
					train_y.head(k) = y.head(k);
				}
				if (k + 1 < n) {
					const auto l = n - k - 1;
					train_X.rightCols(l) = X.rightCols(l);
					train_y.tail(l) = y.tail(l);
				}
				auto trained_model = train_func(train_X, train_y);
				const double test_error = test_func(trained_model, X.block(0, k, X.rows(), 1), y.segment(k, 1));
				sum_weighted_errors += test_error;
			}
			return sum_weighted_errors / static_cast<double>(n);
		}

		/** @brief Calculates model test error using leave-one-out cross-validation (scalar X version).

		See https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation

		`train_func` and `test_func` functors should accept X and y data as `Eigen::Ref<const Eigen::VectorXd>`
		references.

		@param[in] x Vector with all features (scalars).
		@param[in] y Vector with all responses (scalars).
		@param[in] train_func Functor returning a trained model given training features and responses as arguments.
		@param[in] test_func Functor calculating test error per data point given the model, test features and test responses as arguments.

		@tparam Trainer Functor type for model training.
		@tparam Tester Functor type for calculating test error.

		@return Error value per data point.

		@throw std::invalid_argument If `y.size() < 2` or `y.size() != x.size()`.
		*/
		template <class Trainer, class Tester> double leave_one_out_scalar(const Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y, Trainer train_func, Tester test_func)
		{
			const Eigen::Index n = y.size();
			if (n < 2) {
				throw std::invalid_argument("Too few data points");
			}
			if (x.size() != n) {
				throw std::invalid_argument("Data size mismatch");
			}
			double sum_weighted_errors = 0;
			Eigen::VectorXd train_x(n - 1);
			Eigen::VectorXd train_y(n - 1);
			for (Eigen::Index k = 0; k < n; ++k) {
				if (k) {
					train_x.head(k) = x.head(k);
					train_y.head(k) = y.head(k);
				}
				if (k + 1 < n) {
					const auto l = n - k - 1;
					train_x.tail(l) = x.tail(l);
					train_y.tail(l) = y.tail(l);
				}
				auto trained_model = train_func(train_x, train_y);
				const double test_error = test_func(trained_model, x.segment(k, 1), y.segment(k, 1));
				sum_weighted_errors += test_error;
			}
			return sum_weighted_errors / static_cast<double>(n);
		}		
	}
}