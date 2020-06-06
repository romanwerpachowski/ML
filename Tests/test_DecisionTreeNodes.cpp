#include <unordered_set>
#include <gtest/gtest.h>
#include "ML/DecisionTreeNodes.hpp"

using namespace ml::DecisionTrees;

TEST(DecisionTreeNodesTest, nodes)
{
	auto root = std::make_unique<SplitNode<double>>(2.4, -0.1, nullptr, 0.5, 0);
	ASSERT_EQ(2.4, root->error);
	ASSERT_EQ(-0.1, root->value);
	ASSERT_EQ(nullptr, root->parent);
	ASSERT_EQ(0.5, root->threshold);
	ASSERT_FALSE(root->is_leaf());
	ASSERT_EQ(0, root->feature_index);
	root->lower = std::make_unique<LeafNode<double>>(1, -1, root.get());
	ASSERT_EQ(1, root->lower->error);
	ASSERT_EQ(-1, root->lower->value);
	ASSERT_EQ(root.get(), root->lower->parent);
	ASSERT_EQ(1, root->lower->total_leaf_error());
	ASSERT_EQ(0, root->lower->count_lower_nodes());
	ASSERT_TRUE(root->lower->is_leaf());

	auto next_split = std::make_unique<SplitNode<double>>(1.2, 0.4, root.get(), 0.5, 1);
	ASSERT_EQ(1, next_split->feature_index);
	const auto next_split_ptr = next_split.get();
	next_split->lower = std::make_unique<LeafNode<double>>(0.5, 0, next_split_ptr);
	next_split->higher = std::make_unique<LeafNode<double>>(0.5, 1, next_split_ptr);
	ASSERT_EQ(2, next_split->count_lower_nodes());
	root->higher = std::move(next_split);
	ASSERT_EQ(4, root->count_lower_nodes());
	ASSERT_EQ(3, root->count_leaf_nodes());
	ASSERT_NEAR(2, root->total_leaf_error(), 1e-15);

	std::unordered_set<SplitNode<double>*> lowest_split_nodes;
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
}

TEST(DecisionTreeNodesTest, leaf_node_cloning)
{
	auto leaf_orig = std::make_unique<LeafNode<double>>(0.5, 0.25, nullptr);
	auto leaf_copy = std::unique_ptr<LeafNode<double>>(leaf_orig->clone(nullptr));
	ASSERT_EQ(leaf_orig->error, leaf_copy->error);
	ASSERT_EQ(leaf_orig->value, leaf_copy->value);
	ASSERT_NE(leaf_orig.get(), leaf_copy.get());
	ASSERT_EQ(nullptr, leaf_copy->parent);
}

TEST(DecisionTreeNodesTest, split_node_cloning)
{
	auto leaf_orig = std::make_unique<LeafNode<double>>(0.5, 0.25, nullptr);
	const auto split_orig = std::make_unique<SplitNode<double>>(1.2, 0.21, nullptr, 0.7, 2);
	split_orig->lower = std::move(leaf_orig);
	split_orig->lower->parent = split_orig.get();
	split_orig->higher = std::make_unique<LeafNode<double>>(0.6, 0.4, split_orig.get());
	const auto split_copy = std::unique_ptr<SplitNode<double>>(split_orig->clone(nullptr));
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