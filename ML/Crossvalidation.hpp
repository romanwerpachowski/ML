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
		Eigen::MatrixXd only_kth_fold_2d(Eigen::Ref<const Eigen::MatrixXd> data, unsigned int k, unsigned int num_folds);
		DLL_DECLSPEC Eigen::VectorXd only_kth_fold_1d(Eigen::Ref<const Eigen::VectorXd> data, unsigned int k, unsigned int num_folds);

		template <class T> std::vector<T> only_kth_fold_1d(const std::vector<T>& data, const unsigned int k, const unsigned int num_folds) {
			
			size_t i0, i1;
			calc_fold_indices(data.size(), k, num_folds, i0, i1);
			return std::vector<T>(data.begin() + i0, data.begin() + i1);
		}

		/** Return all except the k-th fold column-wise (each column is a data point). */
		Eigen::MatrixXd without_kth_fold_2d(Eigen::Ref<const Eigen::MatrixXd> data, unsigned int k, unsigned int num_folds);
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
	}
}