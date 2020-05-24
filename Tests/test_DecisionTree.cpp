#include "pch.h"
#include <random>
#include "ML/DecisionTree.hpp"

TEST(DecisionTreeTest, nodes)
{
	typedef ml::DecisionTree<double> RegTree;

	auto root = std::make_unique<RegTree::SplitNode>(2.4, -0.1, 0.5, 0);
	ASSERT_EQ(2.4, root->error);
	ASSERT_EQ(-0.1, root->value);
	ASSERT_EQ(0.5, root->threshold);
	ASSERT_EQ(0, root->feature_index);
	root->lower = std::make_unique<RegTree::LeafNode>(1, -1);
	ASSERT_EQ(1, root->lower->error);
	ASSERT_EQ(1, root->lower->total_leaf_error());
	ASSERT_EQ(0, root->lower->count_lower_nodes());

	auto next_split = std::make_unique<RegTree::SplitNode>(1.2, 0.4, 0.5, 1);
	ASSERT_EQ(1.2, next_split->error);
	ASSERT_EQ(0.4, next_split->value);
	ASSERT_EQ(0.5, next_split->threshold);
	ASSERT_EQ(1, next_split->feature_index);
	next_split->lower = std::make_unique<RegTree::LeafNode>(0.5, 0);
	ASSERT_EQ(0.5, next_split->lower->error);
	ASSERT_EQ(0, next_split->lower->value);
	next_split->higher = std::make_unique<RegTree::LeafNode>(0.5, 1);
	ASSERT_EQ(0.5, next_split->higher->error);
	ASSERT_EQ(1, next_split->higher->value);
	ASSERT_EQ(2, next_split->count_lower_nodes());
	root->higher = std::move(next_split);
	ASSERT_EQ(4, root->count_lower_nodes());
	ASSERT_EQ(3, root->count_leaf_nodes());
	ASSERT_NEAR(2, root->total_leaf_error(), 1e-15);
	auto weakest_link = root->find_weakest_link(nullptr);
	ASSERT_EQ(root->higher.get(), std::get<0>(weakest_link));
	ASSERT_EQ(root.get(), std::get<1>(weakest_link));
	ASSERT_NEAR(0.2, std::get<2>(weakest_link), 1e-15);
	ASSERT_NEAR(root->total_leaf_error(), std::get<3>(weakest_link), 1e-15);
	weakest_link = root->higher->find_weakest_link(root.get());
	ASSERT_EQ(root->higher.get(), std::get<0>(weakest_link));
	ASSERT_EQ(root.get(), std::get<1>(weakest_link));
	ASSERT_NEAR(0.2, std::get<2>(weakest_link), 1e-15);
	ASSERT_NEAR(root->higher->total_leaf_error(), std::get<3>(weakest_link), 1e-15);
	weakest_link = root->lower->find_weakest_link(root.get());
	ASSERT_EQ(nullptr, std::get<0>(weakest_link));
	ASSERT_EQ(root.get(), std::get<1>(weakest_link));
	ASSERT_EQ(std::numeric_limits<double>::infinity(), std::get<2>(weakest_link));
	ASSERT_EQ(1, std::get<3>(weakest_link));

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
	ASSERT_EQ(3, tree.count_leaf_nodes());
	ASSERT_EQ(2.4, tree.original_error());
	ASSERT_NEAR(2, tree.total_leaf_error(), 1e-15);
	ASSERT_NEAR(2 + 0.3 * 3, tree.cost_complexity(0.3), 1e-15);
	const auto weakest_link2 = tree.find_weakest_link();
	ASSERT_NE(nullptr, std::get<0>(weakest_link2));
	ASSERT_NE(nullptr, dynamic_cast<RegTree::SplitNode*>(std::get<0>(weakest_link2)));
	ASSERT_NE(nullptr, std::get<1>(weakest_link2));	
	ASSERT_NE(nullptr, dynamic_cast<RegTree::SplitNode*>(std::get<1>(weakest_link2)));
	ASSERT_NEAR(0.2, std::get<2>(weakest_link2), 1e-15);

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

TEST(DecisionTreeTest, node_cloning)
{
	typedef ml::DecisionTree<double> RegTree;
	auto leaf_orig = std::make_unique<RegTree::LeafNode>(0.5, 0.25);
	auto leaf_copy = std::unique_ptr<RegTree::LeafNode>(leaf_orig->clone());
	ASSERT_EQ(leaf_orig->error, leaf_copy->error);
	ASSERT_EQ(leaf_orig->value, leaf_copy->value);
	ASSERT_NE(leaf_orig.get(), leaf_copy.get());
	const auto split_orig = std::make_unique<RegTree::SplitNode>(1.2, 0.21, 0.7, 2);
	split_orig->lower = std::move(leaf_orig);
	split_orig->higher = std::make_unique<RegTree::LeafNode>(0.6, 0.4);
	const auto split_copy = std::unique_ptr<RegTree::SplitNode>(split_orig->clone());
	ASSERT_NE(split_orig.get(), split_copy.get());
	ASSERT_EQ(split_orig->error, split_copy->error);
	ASSERT_EQ(split_orig->value, split_copy->value);
	ASSERT_EQ(split_orig->threshold, split_copy->threshold);
	ASSERT_EQ(split_orig->feature_index, split_copy->feature_index);
	ASSERT_NE(split_orig->lower.get(), split_copy->lower.get());
	ASSERT_NE(split_orig->higher.get(), split_copy->higher.get());
	ASSERT_EQ(split_orig->lower->error, split_copy->lower->error);
	ASSERT_EQ(split_orig->higher->error, split_copy->higher->error);
}

TEST(DecisionTreeTest, find_best_split_reg_1d_constant_y)
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
	const auto split = ml::DecisionTrees::find_best_split_reg_1d(X, y, sorted_y, ml::DecisionTrees::from_vector(features));
	ASSERT_EQ(-std::numeric_limits<double>::infinity(), split.second);	
}

TEST(DecisionTreeTest, find_best_split_reg_1d_linear_in_x0)
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
	const auto split = ml::DecisionTrees::find_best_split_reg_1d(X, y, sorted_y, ml::DecisionTrees::from_vector(features));
	ASSERT_EQ(0, split.first);
	ASSERT_NEAR(4.5, split.second, 1e-15);
}

TEST(DecisionTreeTest, find_best_split_reg_1d_linear)
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
	const auto split = ml::DecisionTrees::find_best_split_reg_1d(X, y, sorted_y, ml::DecisionTrees::from_vector(features));
	ASSERT_EQ(0, split.first);
	ASSERT_NEAR(4.5, split.second, 1e-15);	
}

TEST(DecisionTreeTest, find_best_split_reg_1d_const)
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
	const auto split = ml::DecisionTrees::find_best_split_reg_1d(X, y, sorted_y, ml::DecisionTrees::from_vector(features));
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

	ml::RegressionTree1D tree(ml::DecisionTrees::tree_regression_1d(X, y, 2, 10));
	ml::RegressionTree1D tree1(ml::DecisionTrees::tree_regression_1d(X, y, 1, 10));
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
}

TEST(DecisionTreeTest, pruning)
{
	std::default_random_engine rng;
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
	ml::RegressionTree1D tree(ml::DecisionTrees::tree_regression_1d(X, y, 100, 2));

	const auto weakest = tree.find_weakest_link();
	ASSERT_NE(nullptr, std::get<0>(weakest));
	ASSERT_NE(nullptr, std::get<1>(weakest));
	ASSERT_GT(std::get<2>(weakest), 0);
	ASSERT_EQ(0, std::get<0>(weakest)->lower->count_lower_nodes());
	ASSERT_EQ(0, std::get<0>(weakest)->higher->count_lower_nodes());

	ASSERT_EQ(n, tree.count_leaf_nodes());
	ASSERT_NEAR(0, tree.total_leaf_error(), 1e-15);
	const auto pruned_tree = ml::DecisionTrees::cost_complexity_prune(tree, 0.01);
	ASSERT_EQ(4, pruned_tree.count_leaf_nodes());
	const auto pruned_sse = pruned_tree.total_leaf_error();
	ASSERT_GE(pruned_sse, 0);
	ASSERT_NEAR(n * sigma * sigma, pruned_sse, 1e-2);
}

TEST(DecisionTreeTest, from_vector)
{
	std::vector<double> v(4);
	const auto iter_range = ml::DecisionTrees::from_vector(v);
	ASSERT_EQ(v.begin(), iter_range.first);
	ASSERT_EQ(v.end(), iter_range.second);
}