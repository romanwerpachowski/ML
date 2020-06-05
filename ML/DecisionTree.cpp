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

		template <typename Metrics> static std::pair<unsigned int, double> find_best_split_1d(
			const Metrics metrics,
			const Eigen::Ref<const Eigen::MatrixXd> X,
			const Eigen::Ref<const Eigen::VectorXd> y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			const double error_whole_sample,
			const VectorRange<std::pair<Eigen::Index, double>> features)
		{
			const auto number_dimensions = X.rows();
			const auto sample_size = y.size();
			assert(X.cols() == sample_size);
			assert(sample_size >= 2);
			assert(y.size() == sorted_y.size());
			assert(static_cast<ptrdiff_t>(sample_size) == std::distance(features.first, features.second));
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
							const double sum_errors = metrics.error_for_splitting(sorted_y.data(), sorted_y_it) + metrics.error_for_splitting(sorted_y_it, sorted_y_end);
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

		template <class Y, class Metrics> static std::unique_ptr<typename DecisionTree<Y>::Node> tree_1d_without_pruning(
			const Metrics metrics,
			typename DecisionTree<Y>::SplitNode* const parent,
			Eigen::Ref<Eigen::MatrixXd> unsorted_X,
			Eigen::Ref<Eigen::MatrixXd> sorted_X,
			Eigen::Ref<Eigen::VectorXd> unsorted_y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			const unsigned int allowed_split_levels,
			const unsigned int min_sample_size,
			const VectorRange<std::pair<Eigen::Index, double>> features,
			const unsigned int max_num_threads)
		{
			const auto sample_size = unsorted_y.size();
			assert(unsorted_X.cols() == sample_size);
			assert(unsorted_X.rows() == sorted_X.rows());
			assert(unsorted_X.cols() == sorted_X.cols());
			assert(sorted_y.size() == sample_size);
			const auto unsorted_y_end = unsorted_y.data() + sample_size;
			const auto error_and_value = metrics.error_and_value(unsorted_y.data(), unsorted_y_end);
			const double error = error_and_value.first;
			const double error_for_splitting = metrics.error_for_splitting(unsorted_y.data(), unsorted_y_end, error);
			const Y value = error_and_value.second;
			if (!error || !allowed_split_levels || sample_size < min_sample_size) {
				return std::make_unique<typename DecisionTree<Y>::LeafNode>(error, value, parent);
			} else {
				const auto split = find_best_split_1d(metrics, unsorted_X, unsorted_y, sorted_y, error_for_splitting, features);
				if (split.second == -std::numeric_limits<double>::infinity()) {
					return std::make_unique<typename DecisionTree<Y>::LeafNode>(error, value, parent);
				} else {
					std::unique_ptr<typename DecisionTree<Y>::SplitNode> split_node(new typename DecisionTree<Y>::SplitNode(error, value, parent, split.second, split.first));
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
						auto future_lower = std::async(std::launch::async, [&split_node, &sorted_X, &unsorted_X, &sorted_y, &unsorted_y, allowed_split_levels, min_sample_size, features, features_it, num_samples_below_threshold, max_num_threads, metrics]() {
							return tree_1d_without_pruning<Y>(
								metrics,
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
						auto future_higher = std::async(std::launch::async, [&split_node, &sorted_X, &unsorted_X, &sorted_y, &unsorted_y, allowed_split_levels, min_sample_size, features, features_it, num_samples_below_threshold, sample_size, max_num_threads, metrics]() {
							return tree_1d_without_pruning<Y>(
								metrics,
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
						split_node->lower = tree_1d_without_pruning<Y>(
							metrics,
							split_node.get(),
							sorted_X.leftCols(num_samples_below_threshold),
							unsorted_X.leftCols(num_samples_below_threshold),
							sorted_y.head(num_samples_below_threshold),
							unsorted_y.head(num_samples_below_threshold),
							allowed_split_levels - 1,
							min_sample_size,
							std::make_pair(features.first, features_it), 0);
						split_node->higher = tree_1d_without_pruning<Y>(
							metrics,
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

		template <typename Y, typename Metrics> static DecisionTree<Y> tree_1d(const Metrics metrics, const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const unsigned int max_split_levels, const unsigned int min_sample_size)
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
			return DecisionTree<Y>(tree_1d_without_pruning<Y>(
				metrics, nullptr, unsorted_X, sorted_X, unsorted_y, sorted_y, max_split_levels, min_sample_size, from_vector(features), max_num_threads ? max_num_threads : DEFAULT_MAX_NUM_THREADS));
		}

		struct ClassificationMetrics
		{
			unsigned int num_classes;

			ClassificationMetrics(unsigned int K)
				: num_classes(K)
			{}

			template <typename Iter> std::pair<double, unsigned int> error_and_value(const Iter begin, const Iter end) const
			{
				const unsigned int mode = Statistics::mode(begin, end, num_classes);
				size_t num_misclassified = 0;
				for (auto it = begin; it != end; ++it) {
					if (*it != mode) {
						++num_misclassified;
					}
				}
				return std::make_pair(static_cast<double>(num_misclassified), mode);
			}

			template <typename Iter> double error_for_splitting(Iter begin, Iter end) const
			{
				return static_cast<double>(std::distance(begin, end))* Statistics::gini_index(begin, end, num_classes);
			}

			template <typename Iter> double error_for_splitting(Iter begin, Iter end, double /*error*/) const
			{
				return error_for_splitting(begin, end);
			}
		};

		struct UnivariateRegressionMetrics
		{
			template <typename Iter> std::pair<double, double> error_and_value(Iter begin, Iter end) const
			{
				return Statistics::sse_and_mean(begin, end);
			}

			template <typename Iter> double error_for_splitting(Iter begin, Iter end) const
			{
				return error_and_value(begin, end).first;
			}

			template <typename Iter> double error_for_splitting(Iter, Iter, double error) const
			{
				return error;
			}
		};

		std::pair<unsigned int, double> find_best_split_univariate_regression(
			const Eigen::Ref<const Eigen::MatrixXd> X,
			const Eigen::Ref<const Eigen::VectorXd> y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			const VectorRange<std::pair<Eigen::Index, double>> features)
		{
			const UnivariateRegressionMetrics metrics;
			const double error_whole_sample = metrics.error_for_splitting(y.data(), y.data() + y.size());
			return find_best_split_1d(metrics, X, y, sorted_y, error_whole_sample, features);
		}

		UnivariateRegressionTree univariate_regression_tree(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const unsigned int max_split_levels, const unsigned int min_sample_size)
		{
			return tree_1d<double>(UnivariateRegressionMetrics(), X, y, max_split_levels, min_sample_size);
		}

		ClassificationTree classification_tree(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const unsigned int max_split_levels, const unsigned int min_sample_size)
		{
			return tree_1d<unsigned int>(ClassificationMetrics(static_cast<unsigned int>(y.maxCoeff()) + 1), X, y, max_split_levels, min_sample_size);
		}

		double univariate_regression_tree_mean_squared_error(const UnivariateRegressionTree& tree, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y)
		{
			const auto sample_size = y.size();
			if (!sample_size) {
				return std::numeric_limits<double>::quiet_NaN();
			}
			if (X.cols() != sample_size) {
				throw std::invalid_argument("Data size mismatch");
			}
			double mse = 0;
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				mse += (std::pow(y[i] - tree(X.col(i)), 2) - mse) / static_cast<double>(i + 1);
			}
			return mse;
		}

		double classification_tree_accuracy(const ClassificationTree& tree, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y)
		{
			const auto sample_size = y.size();
			if (!sample_size) {
				return std::numeric_limits<double>::quiet_NaN();
			}
			if (X.cols() != sample_size) {
				throw std::invalid_argument("Data size mismatch");
			}
			int num_correctly_classified = 0;
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				if (y[i] == static_cast<double>(tree(X.col(i)))) {
					++num_correctly_classified;
				}
			}
			return static_cast<double>(num_correctly_classified) / static_cast<double>(sample_size);
		}

		template <class Y, class F> std::pair<double, double> find_best_alpha(const std::vector<double>& alphas, const DecisionTree<Y>& unpruned_tree, F test_error_function, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y)
		{
			double min_test_error = std::numeric_limits<double>::infinity();
			for (double alpha : alphas) {
				DecisionTree<Y> tree(unpruned_tree);
				cost_complexity_prune(tree, alpha);				
			}
		}
	}
}