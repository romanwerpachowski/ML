#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include "ML/DecisionTrees.hpp"

typedef ml::UnivariateRegressionTree RegTree;

typedef std::pair<Eigen::Index, double> IndexedFeatureValue;

/** Creates an iterator pair containing begin() and end(). */
template <class T> static ml::DecisionTrees::VectorRange<T> from_vector(std::vector<T>& v)
{
	return std::make_pair(v.begin(), v.end());
}

TEST(DecisionTreeTest, tree)
{
	auto root = std::make_unique<RegTree::SplitNode>(2.4, -0.1, nullptr, 0.5, 0);
	root->lower = std::make_unique<RegTree::LeafNode>(1, -1, root.get());

	auto next_split = std::make_unique<RegTree::SplitNode>(1.2, 0.4, root.get(), 0.5, 1);
	next_split->lower = std::make_unique<RegTree::LeafNode>(0.5, 0, next_split.get());
	next_split->higher = std::make_unique<RegTree::LeafNode>(0.5, 1, next_split.get());
	root->higher = std::move(next_split);

	std::unordered_set<RegTree::SplitNode*> lowest_split_nodes;
	root->collect_lowest_split_nodes(lowest_split_nodes);
	lowest_split_nodes.clear();
	root->higher->collect_lowest_split_nodes(lowest_split_nodes);
	lowest_split_nodes.clear();
	root->lower->collect_lowest_split_nodes(lowest_split_nodes);

	Eigen::Vector2d x(0, 0);
	ASSERT_EQ(-1, root->operator()(x));
	x << 0, 0.5;
	ASSERT_EQ(-1, root->operator()(x));
	x << 0, 1;
	ASSERT_EQ(-1, root->operator()(x));

	x << 0.5, 0;
	ASSERT_EQ(0, root->operator()(x));
	x << 0.5, 0.5;
	ASSERT_EQ(1, root->operator()(x));
	x << 0.5, 1;
	ASSERT_EQ(1, root->operator()(x));

	x << 1, 0;
	ASSERT_EQ(0, root->operator()(x));
	x << 1, 0.5;
	ASSERT_EQ(1, root->operator()(x));
	x << 1, 1;
	ASSERT_EQ(1, root->operator()(x));

	RegTree tree(std::move(root));
	ASSERT_EQ(5u, tree.count_nodes());
	ASSERT_EQ(1u, tree.number_lowest_split_nodes());
	ASSERT_EQ(3u, tree.count_leaf_nodes());
	ASSERT_EQ(2.4, tree.original_error());
	ASSERT_NEAR(2, tree.total_leaf_error(), 1e-15);
	ASSERT_NEAR(2 + 0.3 * 3, tree.cost_complexity(0.3), 1e-15);	

	x << 0, 0;
	ASSERT_EQ(-1, tree.operator()(x));
	x << 0, 0.5;
	ASSERT_EQ(-1, tree.operator()(x));
	x << 0, 1;
	ASSERT_EQ(-1, tree.operator()(x));

	x << 0.5, 0;
	ASSERT_EQ(0, tree.operator()(x));
	x << 0.5, 0.5;
	ASSERT_EQ(1, tree.operator()(x));
	x << 0.5, 1;
	ASSERT_EQ(1, tree.operator()(x));

	x << 1, 0;
	ASSERT_EQ(0, tree.operator()(x));
	x << 1, 0.5;
	ASSERT_EQ(1, tree.operator()(x));
	x << 1, 1;
	ASSERT_EQ(1, tree.operator()(x));
}

TEST(DecisionTreeTest, find_best_split_univariate_regression_constant_y)
{
	const int sample_size = 100;
	Eigen::MatrixXd X(2, sample_size);
	Eigen::VectorXd y(sample_size);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			const int k = i * 10 + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			y[k] = 0;
		}
	}
	std::vector<IndexedFeatureValue> features(sample_size);
	Eigen::VectorXd sorted_y(sample_size);
	const auto split = ml::DecisionTrees::find_best_split_univariate_regression(X, y, sorted_y, from_vector(features));
	ASSERT_EQ(-std::numeric_limits<double>::infinity(), split.second);	
}

TEST(DecisionTreeTest, find_best_split_univariate_regression_linear_in_x0)
{
	const int sample_size = 100;
	Eigen::MatrixXd X(2, sample_size);
	Eigen::VectorXd y(sample_size);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			const int k = i * 10 + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			y[k] = 0.2 * X(0, k) + 1.5;
		}
	}
	std::vector<IndexedFeatureValue> features(sample_size);
	Eigen::VectorXd sorted_y(sample_size);
	const auto split = ml::DecisionTrees::find_best_split_univariate_regression(X, y, sorted_y, from_vector(features));
	ASSERT_EQ(0u, split.first);
	ASSERT_NEAR(4.5, split.second, 1e-15);
}

TEST(DecisionTreeTest, find_best_split_univariate_regression_linear)
{
	const int sample_size = 100;
	Eigen::MatrixXd X(2, sample_size);
	Eigen::VectorXd y(sample_size);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			const int k = i * 10 + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			y[k] = 0.2 * X(0, k) + 0.01 * X(1, k) + 1.5;
		}
	}
	std::vector<IndexedFeatureValue> features(sample_size);
	Eigen::VectorXd sorted_y(sample_size);
	const auto split = ml::DecisionTrees::find_best_split_univariate_regression(X, y, sorted_y, from_vector(features));
	ASSERT_EQ(0u, split.first);
	ASSERT_NEAR(4.5, split.second, 1e-15);	
}

TEST(DecisionTreeTest, find_best_split_univariate_regression_const)
{
	const int sample_size = 100;
	Eigen::MatrixXd X(2, sample_size);
	Eigen::VectorXd y(sample_size);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			const int k = i * 10 + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			y[k] = 0;
		}
	}
	std::vector<IndexedFeatureValue> features(sample_size);
	Eigen::VectorXd sorted_y(sample_size);
	const auto split = ml::DecisionTrees::find_best_split_univariate_regression(X, y, sorted_y, from_vector(features));
	ASSERT_EQ(0u, split.first);
	ASSERT_EQ(-std::numeric_limits<double>::infinity(), split.second);
}

TEST(DecisionTreeTest, stepwise)
{
	Eigen::MatrixXd X(2, 100);
	Eigen::VectorXd y(100);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			const int k = i * 10 + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			if (i < 4) {
				if (j < 2) {
					y[k] = 0.2;
				} else {
					y[k] = 0.9;
				}
			} else {
				if (j < 6) {
					y[k] = 0.5;
				} else {
					y[k] = 0.25;
				}
			}
		}
	}

	const RegTree tree(ml::DecisionTrees::univariate_regression_tree(X, y, 2, 10));
	const RegTree tree1(ml::DecisionTrees::univariate_regression_tree(X, y, 1, 10));
	ASSERT_EQ(tree1.original_error(), tree1.original_error());
	ASSERT_EQ(7u, tree.count_nodes());
	ASSERT_EQ(3u, tree1.count_nodes());
	double sse = 0;
	double sse1 = 0;
	for (int i = 0; i < 100; ++i) {
		sse += std::pow(tree(X.col(i)) - y[i], 2);
		sse1 += std::pow(tree1(X.col(i)) - y[i], 2);
	}	
	ASSERT_NEAR(0, sse, 1e-15);
	ASSERT_GE(sse1, sse);
	ASSERT_LE(sse1, tree.original_error());
	ASSERT_NEAR(sse, tree.total_leaf_error(), 1e-15);
	ASSERT_NEAR(sse1, tree1.total_leaf_error(), 1e-14);

	ASSERT_EQ(2u, tree.number_lowest_split_nodes());

	const ml::UnivariateRegressionTree tree_copy(tree);
	ASSERT_EQ(2u, tree_copy.number_lowest_split_nodes());
}

TEST(DecisionTreeTest, univariate_regression_with_pruning)
{
	const int train_sample_size = 1000;
	const int test_sample_size = 100;
	const int num_dimensions = 2;
	
	Eigen::MatrixXd train_X(num_dimensions, train_sample_size);
	Eigen::VectorXd train_y(train_sample_size);
	const auto f = [](double x, double y) {
		return std::abs(std::cos(x + y) * std::sin(2 * (x - y))) - 1;
	};
	std::default_random_engine rng;
	rng.seed(3523423);
	std::normal_distribution normal;
	for (int i = 0; i < train_sample_size; ++i) {
		train_X(0, i) = normal(rng);
		train_X(1, i) = normal(rng);
		train_y[i] = f(train_X(0, i), train_X(1, i));
	}	
	Eigen::MatrixXd test_X(num_dimensions, test_sample_size);
	Eigen::VectorXd test_y(test_sample_size);
	for (int i = 0; i < test_sample_size; ++i) {
		test_X(0, i) = normal(rng);
		test_X(1, i) = normal(rng);
		test_y[i] = f(test_X(0, i), test_X(1, i));
	}
	// Grow a big tree.
	RegTree tree(ml::DecisionTrees::univariate_regression_tree(train_X, train_y, 100, 2));
	const double train_sse = tree.total_leaf_error();
	const auto orig_num_modes = tree.count_nodes();
	const double alpha = 0.001;
	const auto orig_cost_complexity = tree.cost_complexity(alpha);
	ASSERT_EQ(0, train_sse);	
	const double train_mse = train_sse / static_cast<double>(train_sample_size);
	
	const double test_mse = ml::DecisionTrees::univariate_regression_tree_mean_squared_error(tree, test_X, test_y);
	ASSERT_GT(test_mse, train_mse + 0.01);
	ASSERT_LE(test_mse, train_mse + 0.035);

	ml::DecisionTrees::cost_complexity_prune(tree, alpha);	
	ASSERT_GT(tree.total_leaf_error(), train_sse);
	ASSERT_LT(tree.count_nodes(), orig_num_modes);
	ASSERT_LT(tree.cost_complexity(alpha), orig_cost_complexity);
	const double pruned_test_mse = ml::DecisionTrees::univariate_regression_tree_mean_squared_error(tree, test_X, test_y);
	ASSERT_GT(pruned_test_mse, train_sse + 0.01);
}

TEST(DecisionTreeTest, univariate_regression_with_crossvalidation)
{
	const int train_sample_size = 1000;
	const int test_sample_size = 100;
	const int num_dimensions = 2;
	Eigen::MatrixXd train_X(num_dimensions, train_sample_size);
	Eigen::VectorXd train_y(train_sample_size);
	const auto f = [](double x, double y) {
		return (x > 0 ? 7 : 0) - (y > 0 ? 4 : -1);
	};
	std::default_random_engine rng;
	rng.seed(3523423);	
	const double noise_strength = 0.01;
	std::normal_distribution normal;
	const auto noise = [&normal, &rng]() {
		return normal(rng);
	};
	for (int i = 0; i < train_sample_size; ++i) {
		train_X(0, i) = normal(rng);
		train_X(1, i) = normal(rng);
		train_y[i] = f(train_X(0, i), train_X(1, i)) + noise_strength * noise();
	}
	Eigen::MatrixXd test_X(num_dimensions, test_sample_size);
	Eigen::VectorXd test_y(test_sample_size);
	for (int i = 0; i < test_sample_size; ++i) {
		test_X(0, i) = normal(rng);
		test_X(1, i) = normal(rng);
		test_y[i] = f(test_X(0, i), test_X(1, i)) + noise_strength * noise();
	}
	const std::vector<double> alphas({ 1e-3, 3e-3, 1e-2 });
	auto result = ml::DecisionTrees::univariate_regression_tree_auto_prune(train_X, train_y, 100, 2, alphas, 10);
	const RegTree& tree = std::get<0>(result);
	const auto alpha = std::get<1>(result);
	ASSERT_GT(alpha, alphas.front());
	ASSERT_LT(alpha, alphas.back());
	const auto cv_test_error = std::get<2>(result);
	ASSERT_GT(cv_test_error, 0);
	ASSERT_NE(alphas.end(), std::find(alphas.begin(), alphas.end(), alpha));
	const double test_error = ml::DecisionTrees::univariate_regression_tree_mean_squared_error(tree, test_X, test_y);
	ASSERT_NEAR(test_error, noise_strength * noise_strength, 2e-5);	
	ASSERT_GT(cv_test_error, test_error);
}

TEST(DecisionTreeTest, pruning)
{
	std::default_random_engine rng;
	rng.seed(3523423);
	std::normal_distribution normal;
	const int n_i = 10;
	const int n_j = 10;
	const int n = n_i * n_j;
	const double sigma = 0.01;

	Eigen::MatrixXd X(2, n);
	Eigen::VectorXd y(n);
	for (int i = 0; i < n_i; ++i) {
		for (int j = 0; j < n_j; ++j) {
			const int k = i * n_j + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			if (i < 4) {
				if (j < 2) {
					y[k] = 0.2;
				} else {
					y[k] = 0.9;
				}
			} else {
				if (j < 6) {
					y[k] = 0.5;
				} else {
					y[k] = 0.25;
				}
			}
			// Add noise.
			y[k] += sigma * normal(rng);
		}
	}
	// Grow a big tree.
	RegTree tree(ml::DecisionTrees::univariate_regression_tree(X, y, 100, 2));

	ASSERT_EQ(n, static_cast<int>(tree.count_leaf_nodes()));
	ASSERT_NEAR(0, tree.total_leaf_error(), 1e-15);
	const double alpha = 0.01;
	const double orig_cost_complexity = tree.cost_complexity(alpha);
	ml::DecisionTrees::cost_complexity_prune(tree, alpha);
	ASSERT_EQ(4u, tree.count_leaf_nodes());
	ASSERT_LT(tree.cost_complexity(alpha), orig_cost_complexity);
	const auto tle_sse = tree.total_leaf_error();
	ASSERT_GE(tle_sse, 0);
	ASSERT_NEAR(n * sigma * sigma, tle_sse, 1e-2);
	double pruned_sse = 0;
	for (int i = 0; i < n; ++i) {
		const auto y_hat = tree(X.col(i));
		pruned_sse += std::pow(y[i] - y_hat, 2);
	}
	ASSERT_NEAR(pruned_sse, tle_sse, 1e-15);
}

TEST(DecisionTreeTest, from_vector)
{
	std::vector<double> v(4);
	const auto iter_range = from_vector(v);
	ASSERT_EQ(v.begin(), iter_range.first);
	ASSERT_EQ(v.end(), iter_range.second);
}

TEST(DecisionTreeTest, classification_with_pruning)
{
	const int train_sample_size = 1000;
	const int test_sample_size = 100;
	const int num_dimensions = 2;
	const double alpha = 1;
	Eigen::MatrixXd train_X(num_dimensions, train_sample_size);
	Eigen::VectorXd train_y(train_sample_size);
	const auto f = [](double x, double y) -> unsigned int {
		const auto score = std::cos(x + y) * std::sin(2 * (x - y));
		if (score < 0) {
			return 0;
		} else {
			return 1;
		}
	};
	std::default_random_engine rng;
	rng.seed(3523423);
	std::normal_distribution normal;	
	for (int i = 0; i < train_sample_size; ++i) {
		train_X(0, i) = normal(rng);
		train_X(1, i) = normal(rng);
		train_y[i] = f(train_X(0, i), train_X(1, i));
	}
	Eigen::MatrixXd test_X(num_dimensions, test_sample_size);
	Eigen::VectorXd test_y(test_sample_size);
	for (int i = 0; i < test_sample_size; ++i) {
		test_X(0, i) = normal(rng);
		test_X(1, i) = normal(rng);
		test_y[i] = f(test_X(0, i), test_X(1, i));
	}
	// Grow a big tree.
	auto tree(ml::DecisionTrees::classification_tree(train_X, train_y, 100, 2));
	const double train_error = tree.total_leaf_error();
	const auto orig_num_modes = tree.count_nodes();
	const auto orig_cost_complexity = tree.cost_complexity(alpha);
	ASSERT_EQ(0, train_error);
	double train_accuracy = 0;
	for (int i = 0; i < train_sample_size; ++i) {
		const unsigned int y_hat = tree(train_X.col(i));
		if (static_cast<double>(y_hat) == train_y[i]) {
			++train_accuracy;
		}
	}
	train_accuracy /= static_cast<double>(train_sample_size);
	ASSERT_EQ(1, train_accuracy);
	ASSERT_NEAR(1 - train_error, train_accuracy, 1e-15);
	const double test_accuracy = ml::DecisionTrees::classification_tree_accuracy(tree, test_X, test_y);
	ASSERT_LT(test_accuracy, train_accuracy);
	ASSERT_GE(test_accuracy, 0.87);

	ml::DecisionTrees::cost_complexity_prune(tree, alpha);
	ASSERT_LT(tree.count_nodes(), orig_num_modes);
	ASSERT_LT(tree.cost_complexity(alpha), orig_cost_complexity);
	ASSERT_GT(tree.total_leaf_error(), train_error);	
	const double pruned_test_accuracy = ml::DecisionTrees::classification_tree_accuracy(tree, test_X, test_y);
	// Pruning lowered accuracy.
	ASSERT_LT(pruned_test_accuracy, train_accuracy);
	ASSERT_GE(pruned_test_accuracy, 0.855);
}

TEST(DecisionTreeTest, classification_with_crossvalidation)
{
	const int train_sample_size = 1000;
	const int test_sample_size = 100;
	const int num_dimensions = 2;
	Eigen::MatrixXd train_X(num_dimensions, train_sample_size);
	Eigen::VectorXd train_y(train_sample_size);
	std::default_random_engine rng;
	rng.seed(3523423);
	std::uniform_real_distribution<double> u01(0, 1);
	const auto f = [&rng, &u01](double x, double /*y*/) -> unsigned int {
		const double score = 2 * x;
		const double p = 1 / (1 + std::exp(-score));
		if (u01(rng) < p) {
			return 0;
		} else {
			return 1;
		}
	};
	for (int i = 0; i < train_sample_size; ++i) {
		train_X(0, i) = 5 - 10 * u01(rng);
		train_X(1, i) = 5 - 10 * u01(rng);
		train_y[i] = f(train_X(0, i), train_X(1, i));
	}
	Eigen::MatrixXd test_X(num_dimensions, test_sample_size);
	Eigen::VectorXd test_y(test_sample_size);
	for (int i = 0; i < test_sample_size; ++i) {
		test_X(0, i) = 5 - 10 * u01(rng);
		test_X(1, i) = 5 - 10 * u01(rng);
		test_y[i] = f(test_X(0, i), test_X(1, i));
	}
	const std::vector<double> alphas({ 1, 10, 100 });
	auto result = ml::DecisionTrees::classification_tree_auto_prune(train_X, train_y, 100, 2, alphas, 10);
	const auto& tree = std::get<0>(result);
	ASSERT_EQ(3u, tree.count_nodes());
	const auto alpha = std::get<1>(result);
	ASSERT_GT(alpha, alphas.front());
	ASSERT_LT(alpha, alphas.back());
	const auto cv_test_error = std::get<2>(result);
	ASSERT_GT(cv_test_error, 0);
	ASSERT_NE(alphas.end(), std::find(alphas.begin(), alphas.end(), alpha));
	const double test_error = ml::DecisionTrees::classification_tree_misclassification_rate(tree, test_X, test_y);
	ASSERT_GE(0.071, test_error);
	ASSERT_GE(cv_test_error + 0.01, test_error);
}