#include <cmath>
#include <future>
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
		constexpr bool USE_THREADS = false;
		constexpr Eigen::Index MIN_SAMPLE_SIZE_FOR_NEW_THREADS = 256;
		constexpr unsigned int DEFAULT_MAX_NUM_THREADS = 2;

		struct SSE
		{
			template <typename Iter> static double calc(Iter begin, Iter end)
			{
				return Statistics::sse(begin, end);
			}
		};

		struct GiniIndex
		{
			template <typename Iter> static double calc(Iter begin, Iter end)
			{
				return static_cast<double>(std::distance(begin, end)) * Statistics::gini_index(begin, end);
			}
		};

		struct Mean
		{
			static double calc(Eigen::Ref<const Eigen::VectorXd> y)
			{
				return y.mean();
			}
		};

		template <typename Error> static std::pair<unsigned int, double> find_best_split_1d(
			const Eigen::Ref<const Eigen::MatrixXd> X,
			const Eigen::Ref<const Eigen::VectorXd> y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			const VectorRange<std::pair<Eigen::Index, double>> features)
		{
			const auto number_dimensions = X.rows();
			const auto sample_size = y.size();
			assert(X.cols() == sample_size);
			assert(sample_size >= 2);
			assert(y.size() == sorted_y.size());
			assert(static_cast<ptrdiff_t>(sample_size) == std::distance(features.first, features.second));			
			const double error_whole_sample = Error::calc(y.data(), y.data() + sample_size);
			double lowest_sum_errors = error_whole_sample;
			double best_threshold = -std::numeric_limits<double>::infinity();
			unsigned int best_feature_index = 0;

			// Find best threshold for each feature.
			for (Eigen::Index feature_index = 0; feature_index < number_dimensions; ++feature_index) {
				const auto X_f = X.row(feature_index);
				if (X_f.minCoeff() != X_f.maxCoeff()) {
					auto features_it = features.first;
					for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
						*features_it = std::make_pair(i, X_f[i]);
					}
					assert(features_it == features.second);

					std::sort(features.first, features.second, SORTED_FEATURE_COMPARATOR);
					auto sorted_y_it = sorted_y.data();
					features_it = features.first;
					for (Eigen::Index i = 0; i < sample_size; ++i, ++sorted_y_it, ++features_it) {
						*sorted_y_it = y(features_it->first);
					}
					assert(features_it == features.second);
					assert(sorted_y_it == sorted_y.data() + sample_size);
					
					features_it = features.first;
					const auto sorted_y_end = sorted_y.data() + sample_size;
					double lowest_sum_errors_for_feature = error_whole_sample;
					auto prev_feature = *features_it;
					sorted_y_it = sorted_y.data() + 1;					
					Eigen::Index best_num_samples_below_threshold = 0;
					for (Eigen::Index num_samples_below_threshold = 1; num_samples_below_threshold < sample_size; ++num_samples_below_threshold, ++sorted_y_it) {
						++features_it;
						const auto next_feature = *features_it;
						// Only consider splits between different values of X_f.						
						if (prev_feature.second < next_feature.second) {
							const double sum_errors = Error::calc(sorted_y.data(), sorted_y_it) + Error::calc(sorted_y_it, sorted_y_end);
							if (sum_errors < lowest_sum_errors_for_feature) {
								lowest_sum_errors_for_feature = sum_errors;
								best_num_samples_below_threshold = num_samples_below_threshold;
							}
						}
						prev_feature = next_feature;
					}
					assert(sorted_y_it == sorted_y.data() + sample_size);
					assert(++features_it == features.second);					
					if (lowest_sum_errors_for_feature < lowest_sum_errors) {
						lowest_sum_errors = lowest_sum_errors_for_feature;
						best_feature_index = static_cast<unsigned int>(feature_index);
						if (!best_num_samples_below_threshold) {
							best_threshold = -std::numeric_limits<double>::infinity();
						} else {
							features_it = features.first + (best_num_samples_below_threshold - 1);
							const auto lower_value = features_it->second;
							++features_it;
							best_threshold = lower_value + 0.5 * (features_it->second - lower_value);
						}
					}
				}
			}
			return std::make_pair(best_feature_index, best_threshold);
		}

		std::pair<unsigned int, double> find_best_split_regression_1d(
			const Eigen::Ref<const Eigen::MatrixXd> X,
			const Eigen::Ref<const Eigen::VectorXd> y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			const VectorRange<std::pair<Eigen::Index, double>> features)
		{
			return find_best_split_1d<SSE>(X, y, sorted_y, features);
		}

		std::pair<unsigned int, double> find_best_split_classification_1d(
			const Eigen::Ref<const Eigen::MatrixXd> X,
			const Eigen::Ref<const Eigen::VectorXd> y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			const VectorRange<std::pair<Eigen::Index, double>> features)
		{
			return find_best_split_1d<GiniIndex>(X, y, sorted_y, features);
		}

		static std::unique_ptr<RegressionTree1D::Node> regression_tree_1d_without_pruning(
			RegressionTree1D::SplitNode* const parent,
			Eigen::Ref<Eigen::MatrixXd> unsorted_X,
			Eigen::Ref<Eigen::MatrixXd> sorted_X,
			Eigen::Ref<Eigen::VectorXd> unsorted_y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			const unsigned int allowed_split_levels,
			const unsigned int min_sample_size,
			const VectorRange<std::pair<Eigen::Index, double>> features,
			const unsigned int max_num_threads)
		{
			const double mean = unsorted_y.mean();
			const auto sse = (unsorted_y.array() - mean).square().sum();
			const auto sample_size = unsorted_y.size();
			assert(unsorted_X.cols() == sample_size);
			assert(unsorted_X.rows() == sorted_X.rows());
			assert(unsorted_X.cols() == sorted_X.cols());
			assert(sorted_y.size() == sample_size);
			if (!sse || !allowed_split_levels || sample_size < min_sample_size) {
				return std::make_unique<RegressionTree1D::LeafNode>(sse, mean, parent);
			} else {
				const auto split = find_best_split_regression_1d(unsorted_X, unsorted_y, sorted_y, features);
				if (split.second == -std::numeric_limits<double>::infinity()) {
					return std::make_unique<RegressionTree1D::LeafNode>(sse, mean, parent);
				} else {
					std::unique_ptr<RegressionTree1D::SplitNode> split_node(new RegressionTree1D::SplitNode(sse, mean, parent, split.second, split.first));
					const auto X_f = unsorted_X.row(split.first);

					auto features_it = features.first;
					for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
						*features_it = std::make_pair(i, X_f[i]);
					}
					assert(features_it == features.second);
					std::sort(features.first, features.second, SORTED_FEATURE_COMPARATOR);					

					features_it = features.first;
					for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
						const auto src_idx = features_it->first;
						sorted_y[i] = unsorted_y[src_idx];
						sorted_X.col(i) = unsorted_X.col(src_idx);
					}
					assert(features_it == features.second);

					for (features_it = features.first; features_it != features.second; ++features_it) {
						if (features_it->second >= split.second) {
							break;
						}
					}
					const Eigen::Index num_samples_below_threshold = std::distance(features.first, features_it);
					assert(num_samples_below_threshold);
					if (USE_THREADS && sample_size >= MIN_SAMPLE_SIZE_FOR_NEW_THREADS && max_num_threads > 1) {
						auto future_lower = std::async(std::launch::async, [&split_node, &sorted_X, &unsorted_X, &sorted_y, &unsorted_y, allowed_split_levels, min_sample_size, features, features_it, num_samples_below_threshold, max_num_threads]() {
							return regression_tree_1d_without_pruning(
								split_node.get(),
								sorted_X.leftCols(num_samples_below_threshold),
								unsorted_X.leftCols(num_samples_below_threshold),
								sorted_y.head(num_samples_below_threshold),
								unsorted_y.head(num_samples_below_threshold),
								allowed_split_levels - 1,
								min_sample_size,
								std::make_pair(features.first, features_it),
								max_num_threads / 2);
							});
						auto future_higher = std::async(std::launch::async, [&split_node, &sorted_X, &unsorted_X, &sorted_y, &unsorted_y, allowed_split_levels, min_sample_size, features, features_it, num_samples_below_threshold, sample_size, max_num_threads]() {
							return regression_tree_1d_without_pruning(
								split_node.get(),
								sorted_X.rightCols(sample_size - num_samples_below_threshold),
								unsorted_X.rightCols(sample_size - num_samples_below_threshold),
								sorted_y.tail(sample_size - num_samples_below_threshold),
								unsorted_y.tail(sample_size - num_samples_below_threshold),
								allowed_split_levels - 1,
								min_sample_size,
								std::make_pair(features_it, features.second),
								max_num_threads / 2);
							});
						split_node->lower = std::move(future_lower.get());
						split_node->higher = std::move(future_higher.get());
					} else {
						// sorted <-> unsorted
						split_node->lower = regression_tree_1d_without_pruning(
							split_node.get(),
							sorted_X.leftCols(num_samples_below_threshold),
							unsorted_X.leftCols(num_samples_below_threshold),
							sorted_y.head(num_samples_below_threshold),
							unsorted_y.head(num_samples_below_threshold),
							allowed_split_levels - 1,
							min_sample_size,
							std::make_pair(features.first, features_it), 0);
						split_node->higher = regression_tree_1d_without_pruning(
							split_node.get(),
							sorted_X.rightCols(sample_size - num_samples_below_threshold),
							unsorted_X.rightCols(sample_size - num_samples_below_threshold),
							sorted_y.tail(sample_size - num_samples_below_threshold),
							unsorted_y.tail(sample_size - num_samples_below_threshold),
							allowed_split_levels - 1,
							min_sample_size,
							std::make_pair(features_it, features.second), 0);
					}
					return split_node;
				}
			}
		}

		RegressionTree1D regression_tree_1d(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const unsigned int max_split_levels, const unsigned int min_sample_size)
		{
			if (min_sample_size < 2) {
				throw std::invalid_argument("Minimum sample size for splitting must be >= 2");
			}
			const auto number_dimensions = X.rows();
			const auto sample_size = y.size();
			if (X.cols() != sample_size) {
				throw std::invalid_argument("Data size mismatch");
			}
			if (sample_size < 2) {
				throw std::invalid_argument("Sample size must be at least 2 for splitting");
			}
			Eigen::MatrixXd unsorted_X(X);
			Eigen::VectorXd unsorted_y(y);
			Eigen::MatrixXd sorted_X(number_dimensions, sample_size);
			Eigen::VectorXd sorted_y(sample_size);
			std::vector<std::pair<Eigen::Index, double>> features(sample_size);			
			const auto max_num_threads = std::min(std::thread::hardware_concurrency(), DEFAULT_MAX_NUM_THREADS);
			return RegressionTree1D(regression_tree_1d_without_pruning(
				nullptr, unsorted_X, sorted_X, unsorted_y, sorted_y, max_split_levels, min_sample_size, from_vector(features), max_num_threads ? max_num_threads : DEFAULT_MAX_NUM_THREADS));
		}
	}
}