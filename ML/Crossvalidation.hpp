#pragma once
#include <vector>
#include <cassert>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
	/** Methods used for crossvalidation. */
	namespace Crossvalidation
	{
		DLL_DECLSPEC void calc_fold_indices(size_t total_len, unsigned int k, unsigned int num_folds, size_t& i0, size_t& i1);

		/** Return k-th fold column-wise (each column is a data point). */
		DLL_DECLSPEC Eigen::MatrixXd only_kth_fold_2d(Eigen::Ref<const Eigen::MatrixXd> data, unsigned int k, unsigned int num_folds);
		DLL_DECLSPEC Eigen::VectorXd only_kth_fold_1d(Eigen::Ref<const Eigen::VectorXd> data, unsigned int k, unsigned int num_folds);

		template <class T> std::vector<T> only_kth_fold_1d(const std::vector<T>& data, const unsigned int k, const unsigned int num_folds) {
			
			size_t i0, i1;
			calc_fold_indices(data.size(), k, num_folds, i0, i1);
			return std::vector<T>(data.begin() + i0, data.begin() + i1);
		}

		/** Return all except the k-th fold column-wise (each column is a data point). */
		DLL_DECLSPEC Eigen::MatrixXd without_kth_fold_2d(Eigen::Ref<const Eigen::MatrixXd> data, unsigned int k, unsigned int num_folds);
		DLL_DECLSPEC Eigen::VectorXd without_kth_fold_1d(Eigen::Ref<const Eigen::VectorXd> data, unsigned int k, unsigned int num_folds);

		template <class T> std::vector<T> without_kth_fold_1d(const std::vector<T>& data, unsigned int k, unsigned int num_folds) {
			size_t i0, i1;
			const size_t total_len = data.size();
			calc_fold_indices(total_len, k, num_folds, i0, i1);
			std::vector<T> remaining(total_len - (i1 - i0));
			std::copy(data.begin(), data.begin() + i0, remaining.begin());
			std::copy(data.begin() + i1, data.end(), remaining.begin() + i0);
			return remaining;
		}

		template <class Train, class Test> double calc_test_error(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Train train_func, Test test_func, const unsigned int num_folds)
		{
			double sum_weighted_errors = 0;
			for (unsigned int k = 0; k < num_folds; ++k) {
				const Eigen::MatrixXd train_X(without_kth_fold_2d(X, k, num_folds));
				const Eigen::VectorXd train_y(without_kth_fold_1d(y, k, num_folds));
				const Eigen::MatrixXd test_X(only_kth_fold_2d(X, k, num_folds));
				const Eigen::VectorXd test_y(only_kth_fold_1d(y, k, num_folds));
				auto trained_model = train_func(train_X, train_y);
				const double test_error = test_func(trained_model, test_X, test_y);
				sum_weighted_errors += test_error * static_cast<double>(test_y.size());
			}
			return sum_weighted_errors / static_cast<double>(y.size());
		}
	}
}