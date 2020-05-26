#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include "ML/DecisionTree.hpp"

typedef ml::UnivariateRegressionTree RegTree;

TEST(DecisionTreeTest, nodes)
{
	auto root = std::make_unique<RegTree::SplitNode>(2.4, -0.1, nullptr, 0.5, 0);
	ASSERT_EQ(2.4, root->error);
	ASSERT_EQ(-0.1, root->value);
	ASSERT_EQ(nullptr, root->parent);
	ASSERT_EQ(0.5, root->threshold);
	ASSERT_FALSE(root->is_leaf());
	ASSERT_EQ(0, root->feature_index);
	root->lower = std::make_unique<RegTree::LeafNode>(1, -1, root.get());
	ASSERT_EQ(1, root->lower->error);
	ASSERT_EQ(-1, root->lower->value);
	ASSERT_EQ(root.get(), root->lower->parent);
	ASSERT_EQ(1, root->lower->total_leaf_error());
	ASSERT_EQ(0, root->lower->count_lower_nodes());
	ASSERT_TRUE(root->lower->is_leaf());

	auto next_split = std::make_unique<RegTree::SplitNode>(1.2, 0.4, root.get(), 0.5, 1);
	ASSERT_EQ(1, next_split->feature_index);
	const auto next_split_ptr = next_split.get();
	next_split->lower = std::make_unique<RegTree::LeafNode>(0.5, 0, next_split_ptr);
	next_split->higher = std::make_unique<RegTree::LeafNode>(0.5, 1, next_split_ptr);
	ASSERT_EQ(2, next_split->count_lower_nodes());
	root->higher = std::move(next_split);
	ASSERT_EQ(4, root->count_lower_nodes());
	ASSERT_EQ(3, root->count_leaf_nodes());
	ASSERT_NEAR(2, root->total_leaf_error(), 1e-15);

	std::unordered_set<RegTree::SplitNode*> lowest_split_nodes;
	root->collect_lowest_split_nodes(lowest_split_nodes);
	ASSERT_EQ(1, lowest_split_nodes.size());
	ASSERT_EQ(1, lowest_split_nodes.count(next_split_ptr));
	lowest_split_nodes.clear();
	root->higher->collect_lowest_split_nodes(lowest_split_nodes);
	ASSERT_EQ(1, lowest_split_nodes.size());
	ASSERT_EQ(1, lowest_split_nodes.count(next_split_ptr));
	lowest_split_nodes.clear();
	root->lower->collect_lowest_split_nodes(lowest_split_nodes);
	ASSERT_TRUE(lowest_split_nodes.empty());	

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
	ASSERT_EQ(5, tree.count_nodes());
	ASSERT_EQ(1, tree.number_lowest_split_nodes());
	ASSERT_EQ(3, tree.count_leaf_nodes());
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

TEST(DecisionTreeTest, leaf_node_cloning)
{
	auto leaf_orig = std::make_unique<RegTree::LeafNode>(0.5, 0.25, nullptr);
	auto leaf_copy = std::unique_ptr<RegTree::LeafNode>(leaf_orig->clone(nullptr));
	ASSERT_EQ(leaf_orig->error, leaf_copy->error);
	ASSERT_EQ(leaf_orig->value, leaf_copy->value);
	ASSERT_NE(leaf_orig.get(), leaf_copy.get());
	ASSERT_EQ(nullptr, leaf_copy->parent);	
}

TEST(DecisionTreeTest, split_node_cloning)
{
	auto leaf_orig = std::make_unique<RegTree::LeafNode>(0.5, 0.25, nullptr);
	const auto split_orig = std::make_unique<RegTree::SplitNode>(1.2, 0.21, nullptr, 0.7, 2);
	split_orig->lower = std::move(leaf_orig);
	split_orig->lower->parent = split_orig.get();
	split_orig->higher = std::make_unique<RegTree::LeafNode>(0.6, 0.4, split_orig.get());
	const auto split_copy = std::unique_ptr<RegTree::SplitNode>(split_orig->clone(nullptr));
	ASSERT_NE(split_orig.get(), split_copy.get());
	ASSERT_EQ(split_orig->error, split_copy->error);
	ASSERT_EQ(split_orig->value, split_copy->value);
	ASSERT_EQ(nullptr, split_copy->parent);
	ASSERT_EQ(split_orig->threshold, split_copy->threshold);
	ASSERT_EQ(split_orig->feature_index, split_copy->feature_index);
	ASSERT_NE(split_orig->lower.get(), split_copy->lower.get());
	ASSERT_NE(split_orig->higher.get(), split_copy->higher.get());
	ASSERT_EQ(split_orig->lower->error, split_copy->lower->error);
	ASSERT_EQ(split_orig->higher->error, split_copy->higher->error);
	ASSERT_EQ(split_copy.get(), split_copy->lower->parent);
	ASSERT_EQ(split_copy.get(), split_copy->higher->parent);
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
	std::vector<ml::DecisionTrees::IndexedFeatureValue> features(sample_size);
	Eigen::VectorXd sorted_y(sample_size);
	const auto split = ml::DecisionTrees::find_best_split_univariate_regression(X, y, sorted_y, ml::DecisionTrees::from_vector(features));
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
	std::vector<ml::DecisionTrees::IndexedFeatureValue> features(sample_size);
	Eigen::VectorXd sorted_y(sample_size);
	const auto split = ml::DecisionTrees::find_best_split_univariate_regression(X, y, sorted_y, ml::DecisionTrees::from_vector(features));
	ASSERT_EQ(0, split.first);
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
	std::vector<ml::DecisionTrees::IndexedFeatureValue> features(sample_size);
	Eigen::VectorXd sorted_y(sample_size);
	const auto split = ml::DecisionTrees::find_best_split_univariate_regression(X, y, sorted_y, ml::DecisionTrees::from_vector(features));
	ASSERT_EQ(0, split.first);
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
	std::vector<ml::DecisionTrees::IndexedFeatureValue> features(sample_size);
	Eigen::VectorXd sorted_y(sample_size);
	const auto split = ml::DecisionTrees::find_best_split_univariate_regression(X, y, sorted_y, ml::DecisionTrees::from_vector(features));
	ASSERT_EQ(0, split.first);
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
	ASSERT_EQ(7, tree.count_nodes());
	ASSERT_EQ(3, tree1.count_nodes());
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

	ASSERT_EQ(2, tree.number_lowest_split_nodes());

	const ml::UnivariateRegressionTree tree_copy(tree);
	ASSERT_EQ(2, tree_copy.number_lowest_split_nodes());
}

TEST(DecisionTreeTest, univariate_regression_with_pruning)
{
	const int train_sample_size = 1000;
	const int test_sample_size = 100;
	const int num_dimensions = 2;
	const double alpha = 0.001;
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
	// Grow a big tree.
	RegTree tree(ml::DecisionTrees::univariate_regression_tree(train_X, train_y, 100, 2));
	const double train_sse = tree.total_leaf_error();
	const auto orig_num_modes = tree.count_nodes();
	const auto orig_cost_complexity = tree.cost_complexity(alpha);
	ASSERT_EQ(0, train_sse);
	double test_sse = 0;
	Eigen::MatrixXd test_X(num_dimensions, test_sample_size);
	Eigen::VectorXd test_y(test_sample_size);	
	for (int i = 0; i < test_sample_size; ++i) {
		test_X(0, i) = normal(rng);
		test_X(1, i) = normal(rng);
		test_y[i] = f(test_X(0, i), test_X(1, i));
		const double y_hat = tree(test_X.col(i));
		test_sse += std::pow(test_y[i] - y_hat, 2);
	}
	ASSERT_GT(test_sse, train_sse + 0.01 * test_sample_size);
	ASSERT_LE(test_sse, train_sse + 0.04 * test_sample_size);

	ml::DecisionTrees::cost_complexity_prune(tree, alpha);	
	ASSERT_GT(tree.total_leaf_error(), train_sse);
	ASSERT_LT(tree.count_nodes(), orig_num_modes);
	ASSERT_LT(tree.cost_complexity(alpha), orig_cost_complexity);
	double pruned_test_sse = 0;
	for (int i = 0; i < test_sample_size; ++i) {
		const double y_hat = tree(test_X.col(i));
		pruned_test_sse += std::pow(test_y[i] - y_hat, 2);		
	}
	ASSERT_GT(test_sse, pruned_test_sse);
	ASSERT_GT(pruned_test_sse, train_sse);	
	ASSERT_GT(pruned_test_sse, train_sse + 0.01 * test_sample_size);
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

	ASSERT_EQ(n, tree.count_leaf_nodes());
	ASSERT_NEAR(0, tree.total_leaf_error(), 1e-15);
	const double alpha = 0.01;
	const double orig_cost_complexity = tree.cost_complexity(alpha);
	ml::DecisionTrees::cost_complexity_prune(tree, alpha);
	ASSERT_EQ(4, tree.count_leaf_nodes());
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
	const auto iter_range = ml::DecisionTrees::from_vector(v);
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
	// Grow a big tree.
	auto tree(ml::DecisionTrees::classification_tree(train_X, train_y, 100, 2));
	const double train_gini_index = tree.total_leaf_error();	
	const auto orig_num_modes = tree.count_nodes();
	const auto orig_cost_complexity = tree.cost_complexity(alpha);
	ASSERT_EQ(0, train_gini_index);
	double train_accuracy = 0;
	for (int i = 0; i < train_sample_size; ++i) {
		const unsigned int y_hat = tree(train_X.col(i));
		if (static_cast<double>(y_hat) == train_y[i]) {
			++train_accuracy;
		}
	}
	train_accuracy /= static_cast<double>(train_sample_size);
	ASSERT_EQ(1, train_accuracy);	
	Eigen::MatrixXd test_X(num_dimensions, test_sample_size);
	Eigen::VectorXd test_y(test_sample_size);
	double test_accuracy = 0;
	for (int i = 0; i < test_sample_size; ++i) {
		test_X(0, i) = normal(rng);
		test_X(1, i) = normal(rng);
		test_y[i] = f(test_X(0, i), test_X(1, i));
		const unsigned int y_hat = tree(test_X.col(i));
		if (static_cast<double>(y_hat) == test_y[i]) {
			++test_accuracy;
		}
	}
	test_accuracy /= static_cast<double>(test_sample_size);
	ASSERT_LT(test_accuracy, train_accuracy);
	ASSERT_GE(test_accuracy, 0.85);

	ml::DecisionTrees::cost_complexity_prune(tree, alpha);
	ASSERT_LT(tree.count_nodes(), orig_num_modes);
	ASSERT_LT(tree.cost_complexity(alpha), orig_cost_complexity);
	ASSERT_GT(tree.total_leaf_error(), train_gini_index);	
	double pruned_test_accuracy = 0;
	for (int i = 0; i < test_sample_size; ++i) {
		const unsigned int y_hat = tree(test_X.col(i));
		if (static_cast<double>(y_hat) == test_y[i]) {
			++pruned_test_accuracy;
		}
	}
	pruned_test_accuracy /= static_cast<double>(test_sample_size);
	ASSERT_LT(test_accuracy, pruned_test_accuracy);
	ASSERT_LT(pruned_test_accuracy, train_accuracy);
	ASSERT_GE(pruned_test_accuracy, 0.82);
}