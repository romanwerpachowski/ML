#include <cmath>
#include <iostream>
#include <limits>
#include <utility>
#include "DecisionTree.hpp"
#include "Statistics.hpp"

namespace ml
{	
	namespace DecisionTrees
	{

		static const auto SORTED_FEATURE_COMPARATOR = [](const std::pair<Eigen::Index, double>& a, const std::pair<Eigen::Index, double>& b) { return a.second < b.second; };

		std::pair<unsigned int, double> find_best_split_reg_1d(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, IteratorRange<std::pair<Eigen::Index, double>> features)
		{
			const auto number_dimensions = X.rows();
			const auto sample_size = y.size();
			if (X.cols() != sample_size) {
				throw std::invalid_argument("Data size mismatch");
			}
			if (sample_size < 2) {
				throw std::invalid_argument("Sample size must be at least 2 for splitting");
			}
			std::vector<double> sorted_y(sample_size);
			std::vector<double> sum_sse_for_feature_index(sample_size);
			std::vector<double> sum_sse(number_dimensions);
			std::vector<double> thresholds(number_dimensions);

			// Find best threshold for each feature.
			for (Eigen::Index feature_index = 0; feature_index < number_dimensions; ++feature_index) {
				const auto X_f = X.row(feature_index);
				if (X_f.minCoeff() == X_f.maxCoeff()) {
					// Block splitting on this feature - we have no data to find a threshold.
					thresholds[feature_index] = -std::numeric_limits<double>::infinity();
					sum_sse[feature_index] = std::numeric_limits<double>::infinity();
				} else {
					auto features_it = features.first;
					for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
						*features_it = std::make_pair(i, X_f[i]);
					}
					assert(features_it == features.second);
					std::sort(features.first, features.second, SORTED_FEATURE_COMPARATOR);
					features_it = features.first;
					for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
						sorted_y[i] = y(features_it->first);
					}
					assert(features_it == features.second);
					// i is the number of features below the threshold.
					sum_sse_for_feature_index.front() = Statistics::calc_sse(sorted_y.begin(), sorted_y.end());
					features_it = features.first;
					auto prev_feature = *features_it;
					auto sum_sse_for_feature_index_it = sum_sse_for_feature_index.begin() + 1;
					for (Eigen::Index i = 1; i < sample_size; ++i, ++sum_sse_for_feature_index_it) {
						++features_it;
						const auto next_feature = *features_it;
						// Only consider splits between different values of X_f.						
						if (prev_feature.second < next_feature.second) {
							*sum_sse_for_feature_index_it = Statistics::calc_sse(sorted_y.begin(), sorted_y.begin() + i) + Statistics::calc_sse(sorted_y.begin() + i, sorted_y.end());
						} else {
							*sum_sse_for_feature_index_it = std::numeric_limits<double>::infinity();
						}
						prev_feature = next_feature;
					}
					assert(++features_it == features.second);
					const auto min_sse_it = std::min_element(sum_sse_for_feature_index.begin(), sum_sse_for_feature_index.end());
					sum_sse[feature_index] = *min_sse_it;
					const auto num_samples_below_threshold = static_cast<Eigen::Index>(min_sse_it - sum_sse_for_feature_index.begin());
					if (!num_samples_below_threshold) {
						thresholds[feature_index] = -std::numeric_limits<double>::infinity();
					} else {
						features_it = features.first + (num_samples_below_threshold - 1);
						const auto lower_value = features_it->second;
						++features_it;
						thresholds[feature_index] = lower_value + 0.5 * (features_it->second - lower_value);
					}
				}
			}
			const auto lowest_sse_it = std::min_element(sum_sse.begin(), sum_sse.end());
			const auto best_feature_index = static_cast<unsigned int>(lowest_sse_it - sum_sse.begin());
			return std::make_pair(best_feature_index, thresholds[best_feature_index]);
		}

		std::unique_ptr<RegressionTree1D::Node> tree_regression_1d_without_pruning(const Eigen::Ref<Eigen::MatrixXd> X, const Eigen::Ref<Eigen::VectorXd> y, const unsigned int allowed_split_levels, const unsigned int min_sample_size, IteratorRange<std::pair<Eigen::Index, double>> features)
		{
			const double mean = y.mean();
			const auto sse = (y.array() - mean).square().sum();
			const auto sample_size = y.size();
			if (!sse || !allowed_split_levels || sample_size < min_sample_size) {
				return std::make_unique<RegressionTree1D::LeafNode>(sse, mean);
			} else {
				const auto split = find_best_split_reg_1d(X, y, features);
				if (split.second == -std::numeric_limits<double>::infinity()) {
					return std::make_unique<RegressionTree1D::LeafNode>(sse, mean);
				} else {
					std::unique_ptr<RegressionTree1D::SplitNode> split_node(new RegressionTree1D::SplitNode(sse, mean, split.second, split.first));
					const auto X_f = X.row(split.first);

					auto features_it = features.first;
					for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
						*features_it = std::make_pair(i, X_f[i]);
					}
					assert(features_it == features.second);
					std::sort(features.first, features.second, SORTED_FEATURE_COMPARATOR);

					Eigen::MatrixXd sorted_X(X.rows(), X.cols());
					Eigen::VectorXd sorted_y(y.size());

					features_it = features.first;
					for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
						const auto src_idx = features_it->first;
						sorted_y[i] = y(src_idx);
						sorted_X.col(i) = X.col(src_idx);
					}
					assert(features_it == features.second);

					for (features_it = features.first; features_it != features.second; ++features_it) {
						if (features_it->second >= split.second) {
							break;
						}
					}
					const Eigen::Index num_samples_below_threshold = std::distance(features.first, features_it);
					assert(num_samples_below_threshold);
					split_node->lower = tree_regression_1d_without_pruning(sorted_X.leftCols(num_samples_below_threshold), sorted_y.head(num_samples_below_threshold), allowed_split_levels - 1, min_sample_size, std::make_pair(features.first, features_it));
					split_node->higher = tree_regression_1d_without_pruning(sorted_X.rightCols(sample_size - num_samples_below_threshold), sorted_y.tail(sample_size - num_samples_below_threshold), allowed_split_levels - 1, min_sample_size, std::make_pair(features_it, features.second));
					return split_node;
				}
			}
		}

		RegressionTree1D tree_regression_1d(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const unsigned int max_split_levels, const unsigned int min_sample_size)
		{
			if (min_sample_size < 2) {
				throw std::invalid_argument("Minimum sample size for splitting must be >= 2");
			}
			Eigen::MatrixXd work_X(X);
			Eigen::VectorXd work_y(y);
			std::vector<std::pair<Eigen::Index, double>> features(y.size());
			return RegressionTree1D(tree_regression_1d_without_pruning(work_X, work_y, max_split_levels, min_sample_size, std::make_pair(features.begin(), features.end())));
		}
	}
}