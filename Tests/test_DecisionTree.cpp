#include "pch.h"
#include "ML/DecisionTree.hpp"

TEST(DecisionTreeTest, nodes)
{
	typedef ml::DecisionTree<double> RegTree;

	auto root = std::make_unique<RegTree::SplitNode>(2.4, 0.5, 0);
	ASSERT_EQ(2.4, root->error);
	root->lower = std::make_unique<RegTree::LeafNode>(1, -1);
	ASSERT_EQ(1, root->lower->error);
	ASSERT_EQ(1, root->lower->total_leaf_error());
	ASSERT_EQ(0, root->lower->count_lower_nodes());

	auto next_split = std::make_unique<RegTree::SplitNode>(1.2, 0.5, 1);
	next_split->lower = std::make_unique<RegTree::LeafNode>(0.5, 0);
	next_split->higher = std::make_unique<RegTree::LeafNode>(0.5, 1);
	ASSERT_EQ(2, next_split->count_lower_nodes());
	root->higher = std::move(next_split);
	ASSERT_EQ(4, root->count_lower_nodes());
	ASSERT_NEAR(2, root->total_leaf_error(), 1e-15);

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
	ASSERT_EQ(2.4, tree.original_error());
	ASSERT_NEAR(2, tree.total_leaf_error(), 1e-15);

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

TEST(DecisionTreeTest, find_best_split_reg_1d_constant_y)
{
	Eigen::MatrixXd X(2, 100);
	Eigen::VectorXd y(100);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			const int k = i * 10 + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			y[k] = 0;
		}
	}
	const auto split = ml::find_best_split_reg_1d(X, y);
	ASSERT_EQ(-std::numeric_limits<double>::infinity(), split.second);	
}

TEST(DecisionTreeTest, find_best_split_reg_1d_linear_in_x0)
{
	Eigen::MatrixXd X(2, 100);
	Eigen::VectorXd y(100);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			const int k = i * 10 + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			y[k] = 0.2 * X(0, k) + 1.5;
		}
	}
	const auto split = ml::find_best_split_reg_1d(X, y);
	ASSERT_EQ(0, split.first);
	ASSERT_NEAR(4.5, split.second, 1e-15);
}

TEST(DecisionTreeTest, find_best_split_reg_1d_linear)
{
	Eigen::MatrixXd X(2, 100);
	Eigen::VectorXd y(100);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			const int k = i * 10 + j;
			X(0, k) = static_cast<double>(i);
			X(1, k) = static_cast<double>(j);
			y[k] = 0.2 * X(0, k) + 0.01 * X(1, k) + 1.5;
		}
	}
	const auto split = ml::find_best_split_reg_1d(X, y);
	ASSERT_EQ(0, split.first);
	ASSERT_NEAR(4.5, split.second, 1e-15);	
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

	ml::RegressionTree1D tree(ml::tree_regression_1d(X, y, 2, 10));
	ml::RegressionTree1D tree1(ml::tree_regression_1d(X, y, 1, 10));
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