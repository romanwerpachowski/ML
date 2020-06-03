#include <stdexcept>
#include "Crossvalidation.hpp"

namespace ml
{
	namespace Crossvalidation
	{
		void calc_fold_indices(const size_t total_len, const unsigned int k, const unsigned int num_folds, size_t& i0, size_t& i1)
		{
			if (k >= num_folds) {
				throw std::invalid_argument("Fold index too large");
			}
			if (num_folds > total_len) {
				throw std::invalid_argument("Too many folds requested");
			}
			const size_t fold_len = total_len / num_folds;
			const size_t remainder = total_len % num_folds;
			i0 = k * fold_len;
			i1 = i0 + fold_len;
			if (k + 1 == num_folds) {
				i1 += remainder;
			}
			assert(i1 <= total_len);
		}

		Eigen::MatrixXd only_kth_fold_2d(Eigen::Ref<const Eigen::MatrixXd, 0> data, const unsigned int k, const unsigned int num_folds)
		{
			size_t i0, i1;
			const auto total_size = data.cols();
			calc_fold_indices(total_size, k, num_folds, i0, i1);
			const size_t taken_len = i1 - i0;
			assert(taken_len);
			Eigen::MatrixXd taken(data.rows(), taken_len);
			taken = data.block(0, i0, data.rows(), taken_len);
			return taken;
		}

		Eigen::VectorXd only_kth_fold_1d(Eigen::Ref<const Eigen::VectorXd, 0> data, const unsigned int k, const unsigned int num_folds)
		{
			size_t i0, i1;
			const auto total_size = data.size();
			calc_fold_indices(total_size, k, num_folds, i0, i1);
			const size_t taken_len = i1 - i0;
			assert(taken_len);
			Eigen::VectorXd taken(taken_len);
			taken = data.segment(i0, taken_len);
			return taken;
		}

		Eigen::MatrixXd without_kth_fold_2d(Eigen::Ref<const Eigen::MatrixXd> data, const unsigned int k, const unsigned int num_folds)
		{
			size_t i0, i1;
			const auto total_size = data.cols();
			calc_fold_indices(total_size, k, num_folds, i0, i1);
			const size_t remaining_len = total_size - (i1 - i0);
			Eigen::MatrixXd remaining(data.rows(), remaining_len);
			remaining.leftCols(i0) = data.leftCols(i0);
			remaining.rightCols(remaining_len - i0) = data.rightCols(remaining_len - i0);
			return remaining;
		}

		Eigen::VectorXd without_kth_fold_1d(Eigen::Ref<const Eigen::VectorXd> data, const unsigned int k, const unsigned int num_folds)
		{
			size_t i0, i1;
			const auto total_size = data.size();
			calc_fold_indices(total_size, k, num_folds, i0, i1);
			const size_t remaining_len = total_size - (i1 - i0);
			Eigen::VectorXd remaining(remaining_len);
			remaining.head(i0) = data.head(i0);
			remaining.tail(remaining_len - i0) = data.tail(remaining_len - i0);
			return remaining;
		}
	}
}