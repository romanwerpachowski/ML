/* (C) 2020 Roman Werpachowski. */
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include "Crossvalidation.hpp"
#include "DecisionTrees.hpp"
#include "Features.hpp"
#include "Statistics.hpp"

namespace ml
{	
	namespace DecisionTrees
	{		
		constexpr bool USE_THREADS = false; /**< @brief Whether to use threads when fitting a tree .*/
		constexpr Eigen::Index MIN_SAMPLE_SIZE_FOR_NEW_THREADS = 256; /**< @brief Minimum sample size for which it's worth launching a new thread.*/
		constexpr unsigned int DEFAULT_MAX_NUM_THREADS = 2; /**< @brief Default maximum number of threads.*/

		template <typename Metrics> static std::pair<unsigned int, double> find_best_split_1d(
			const Metrics metrics,
			const Eigen::Ref<const Eigen::MatrixXd> X,
			const Eigen::Ref<const Eigen::VectorXd> y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			const double error_whole_sample,
			const Features::VectorRange<Features::IndexedFeatureValue> features)
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
				if (X.row(feature_index).minCoeff() != X.row(feature_index).maxCoeff()) {
					Features::set_to_nth(X, feature_index, features);

					std::sort(features.first, features.second, Features::INDEXED_FEATURE_COMPARATOR_ASCENDING);
					auto sorted_y_it = sorted_y.data();
					auto features_it = features.first;
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
						// Only consider splits between different values of features.						
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
			const Features::VectorRange<Features::IndexedFeatureValue> features,
			const unsigned int max_num_threads)
		{
			const auto sample_size = static_cast<unsigned int>(unsorted_y.size());
			assert(static_cast<unsigned int>(unsorted_X.cols()) == sample_size);
			assert(unsorted_X.rows() == sorted_X.rows());
			assert(unsorted_X.cols() == sorted_X.cols());
			assert(static_cast<unsigned int>(sorted_y.size()) == sample_size);
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
					Features::set_to_nth(unsorted_X, split.first, features);
					std::sort(features.first, features.second, Features::INDEXED_FEATURE_COMPARATOR_ASCENDING);

					auto features_it = features.first;
					for (unsigned int i = 0; i < sample_size; ++i, ++features_it) {
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
					if constexpr (USE_THREADS && sample_size >= MIN_SAMPLE_SIZE_FOR_NEW_THREADS && max_num_threads > 1) {
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
			std::vector<Features::IndexedFeatureValue> features(sample_size);
			const auto max_num_threads = std::min(std::thread::hardware_concurrency(), DEFAULT_MAX_NUM_THREADS);
			return DecisionTree<Y>(tree_1d_without_pruning<Y>(
				metrics, nullptr, unsorted_X, sorted_X, unsorted_y, sorted_y, max_split_levels, min_sample_size, Features::from_vector(features), max_num_threads ? max_num_threads : DEFAULT_MAX_NUM_THREADS));
		}

		/** @brief Metrics for classification trees. */
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

		/** @brief Metrics for regression trees. */
		struct RegressionMetrics
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

		std::pair<unsigned int, double> find_best_split_regression(
			const Eigen::Ref<const Eigen::MatrixXd> X,
			const Eigen::Ref<const Eigen::VectorXd> y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			const Features::VectorRange<Features::IndexedFeatureValue> features)
		{
			const RegressionMetrics metrics;
			const double error_whole_sample = metrics.error_for_splitting(y.data(), y.data() + y.size());
			return find_best_split_1d(metrics, X, y, sorted_y, error_whole_sample, features);
		}

		RegressionTree regression_tree(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const unsigned int max_split_levels, const unsigned int min_sample_size)
		{
			return tree_1d<double>(RegressionMetrics(), X, y, max_split_levels, min_sample_size);
		}

		ClassificationTree classification_tree(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const unsigned int max_split_levels, const unsigned int min_sample_size)
		{
			return tree_1d<unsigned int>(ClassificationMetrics(static_cast<unsigned int>(y.maxCoeff()) + 1), X, y, max_split_levels, min_sample_size);
		}

		double regression_tree_mean_squared_error(const RegressionTree& tree, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y)
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
				const double err_i = std::pow(y[i] - tree(X.col(i)), 2);
				mse += (err_i - mse) / static_cast<double>(i + 1);
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

		template <class Trainer, class Tester> std::pair<double, double> find_best_alpha(const std::vector<double>& alphas, Trainer grow_function, Tester test_error_function, const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const unsigned int num_folds)
		{
			double min_cv_test_error = std::numeric_limits<double>::infinity();
			double best_alpha = -1;
			for (double alpha : alphas) {
				auto train_func = [alpha, grow_function](const Eigen::Ref<const Eigen::MatrixXd> train_X, const Eigen::Ref<const Eigen::VectorXd> train_y) {
					auto tree = grow_function(train_X, train_y);
					cost_complexity_prune(tree, alpha);
					return tree;
				};
				const double cv_test_error = Crossvalidation::k_fold(X, y, train_func, test_error_function, num_folds);
				if (cv_test_error < min_cv_test_error) {
					min_cv_test_error = cv_test_error;
					best_alpha = alpha;
				}
			}
			return std::make_pair(best_alpha, min_cv_test_error);
		}

		template <class Y, class Metrics, class Tester> std::tuple<DecisionTree<Y>, double, double> tree_1d_auto_prune(const Metrics metrics, Tester test_error_function, const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y, const unsigned int max_split_levels, const unsigned int min_sample_size, const std::vector<double>& alphas, const unsigned int num_folds)
		{
			double alpha = std::numeric_limits<double>::quiet_NaN();
			double min_cv_test_error = std::numeric_limits<double>::quiet_NaN();
			if (alphas.size() > 1) {
				auto grow_function = [max_split_levels, min_sample_size, metrics](const Eigen::Ref<const Eigen::MatrixXd> train_X, const Eigen::Ref<const Eigen::VectorXd> train_y) {
					return tree_1d<Y, Metrics>(metrics, train_X, train_y, max_split_levels, min_sample_size);
				};
				const auto best_alpha_and_min_cv_test_error = find_best_alpha(alphas, grow_function, test_error_function, X, y, num_folds);
				alpha = best_alpha_and_min_cv_test_error.first;
				min_cv_test_error = best_alpha_and_min_cv_test_error.second;
			} else if (alphas.size() == 1) {
				alpha = alphas.front();				
			}
			DecisionTree<Y> tree(tree_1d<Y, Metrics>(metrics, X, y, max_split_levels, min_sample_size));
			if (!std::isnan(alpha)) {
				cost_complexity_prune(tree, alpha);
			}
			return std::make_tuple(std::move(tree), alpha, min_cv_test_error);
		}

		std::tuple<RegressionTree, double, double> regression_tree_auto_prune(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size, const std::vector<double>& alphas, const unsigned int num_folds)
		{
			return tree_1d_auto_prune<double>(RegressionMetrics(), regression_tree_mean_squared_error, X, y, max_split_levels, min_sample_size, alphas, num_folds);
		}

		std::tuple<ClassificationTree, double, double> classification_tree_auto_prune(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size, const std::vector<double>& alphas, const unsigned int num_folds)
		{
			return tree_1d_auto_prune<unsigned int>(ClassificationMetrics(static_cast<unsigned int>(y.maxCoeff()) + 1), classification_tree_misclassification_rate, X, y, max_split_levels, min_sample_size, alphas, num_folds);
		}
	}
}