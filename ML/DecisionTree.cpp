#include <cmath>
#include <iostream>
#include <limits>
#include "DecisionTree.hpp"
#include "Statistics.hpp"

namespace ml
{	
	static const auto SORTED_FEATURE_COMPARATOR = [](const std::pair<Eigen::Index, double>& a, const std::pair<Eigen::Index, double>& b) { return a.second < b.second; };	

	std::pair<unsigned int, double> find_best_split_reg_1d(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y)
	{
		const auto number_dimensions = X.rows();
		const auto sample_size = y.size();
		if (X.cols() != sample_size) {
			throw std::invalid_argument("Data size mismatch");
		}
		if (sample_size < 2) {
			throw std::invalid_argument("Sample size must be at least 2 for splitting");
		}
		std::vector<std::pair<Eigen::Index, double>> features(sample_size);
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
				for (Eigen::Index i = 0; i < sample_size; ++i) {
					features[i] = std::make_pair(i, X_f[i]);
				}
				std::sort(features.begin(), features.end(), SORTED_FEATURE_COMPARATOR);
				for (Eigen::Index i = 0; i < sample_size; ++i) {
					sorted_y[i] = y(features[i].first);
				}
				// i is the number of features below the threshold.
				for (Eigen::Index i = 0; i < sample_size; ++i) {
					// Only consider splits between different values of X_f.
					if (!i || features[i - 1].second < features[i].second) {
						sum_sse_for_feature_index[i] = Statistics::calc_sse(sorted_y.begin(), sorted_y.begin() + i) + Statistics::calc_sse(sorted_y.begin() + i, sorted_y.end());
					} else {
						sum_sse_for_feature_index[i] = std::numeric_limits<double>::infinity();
					}
				}
				const auto min_sse_it = std::min_element(sum_sse_for_feature_index.begin(), sum_sse_for_feature_index.end());
				sum_sse[feature_index] = *min_sse_it;
				const auto num_samples_below_threshold = static_cast<Eigen::Index>(min_sse_it - sum_sse_for_feature_index.begin());
				if (!num_samples_below_threshold) {
					thresholds[feature_index] = -std::numeric_limits<double>::infinity();
				} else {
					const auto lower_value = features[num_samples_below_threshold - 1].second;
					thresholds[feature_index] = lower_value + 0.5 * (features[num_samples_below_threshold].second - lower_value);
				}
			}
		}
		const auto lowest_sse_it = std::min_element(sum_sse.begin(), sum_sse.end());
		const auto best_feature_index = static_cast<unsigned int>(lowest_sse_it - sum_sse.begin());		
		return std::make_pair(best_feature_index, thresholds[best_feature_index]);
	}

	std::unique_ptr<RegressionTree1D::Node> tree_regression_1d_without_pruning(const Eigen::Ref<Eigen::MatrixXd> X, const Eigen::Ref<Eigen::VectorXd> y, const unsigned int allowed_split_levels, const unsigned int min_sample_size, std::vector<std::pair<Eigen::Index, double>>::iterator features_begin, std::vector<std::pair<Eigen::Index, double>>::iterator features_end)
	{
		const double mean = y.mean();
		const auto sse = (y.array() - mean).square().sum();
		const auto sample_size = y.size();
		if (!sse || !allowed_split_levels || sample_size < min_sample_size) {
			return std::make_unique<RegressionTree1D::LeafNode>(sse, mean);
		} else {
			const auto split = find_best_split_reg_1d(X, y);
			if (split.second == -std::numeric_limits<double>::infinity()) {
				return std::make_unique<RegressionTree1D::LeafNode>(sse, mean);
			} else {
				std::unique_ptr<RegressionTree1D::SplitNode> split_node(new RegressionTree1D::SplitNode(sse, mean, split.second, split.first));				
				const auto X_f = X.row(split.first);

				auto features_it = features_begin;
				for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
					*features_it = std::make_pair(i, X_f[i]);
				}
				assert(features_it == features_end);
				std::sort(features_begin, features_end, SORTED_FEATURE_COMPARATOR);

				Eigen::MatrixXd sorted_X(X.rows(), X.cols());
				Eigen::VectorXd sorted_y(y.size());

				features_it = features_begin;
				for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
					const auto src_idx = features_it->first;
					sorted_y[i] = y(src_idx);
					sorted_X.col(i) = X.col(src_idx);
				}
				assert(features_it == features_end);
				
				for (features_it = features_begin; features_it != features_end; ++features_it) {
					if (features_it->second >= split.second) {
						break;
					}
				}
				const Eigen::Index num_samples_below_threshold = std::distance(features_begin, features_it);
				assert(num_samples_below_threshold);				
				split_node->lower = tree_regression_1d_without_pruning(sorted_X.leftCols(num_samples_below_threshold), sorted_y.head(num_samples_below_threshold), allowed_split_levels - 1, min_sample_size, features_begin, features_it);
				split_node->higher = tree_regression_1d_without_pruning(sorted_X.rightCols(sample_size - num_samples_below_threshold), sorted_y.tail(sample_size - num_samples_below_threshold), allowed_split_levels - 1, min_sample_size, features_it, features_end);
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
		return RegressionTree1D(tree_regression_1d_without_pruning(work_X, work_y, max_split_levels, min_sample_size, features.begin(), features.end()));
	}
}