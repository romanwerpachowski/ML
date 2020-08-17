#pragma once
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <Eigen/Core>
#include "DecisionTreeNodes.hpp"

namespace ml
{
	/** Decision tree.

	Data points are in columns.
	*/
	template <class Y> class DecisionTree
	{
	public:
		typedef Eigen::Ref<const Eigen::VectorXd> arg_type /**< Type for feature vector. */;
		typedef Y value_type; /**< Type of predicted value (integer for classification, real for regression). */
		typedef DecisionTrees::Node<Y> Node; /**< Tree node. Nodes are split (non-terminal) or leaf (terminal). */
		typedef DecisionTrees::SplitNode<Y> SplitNode; /**< Non-terminal node, which splits data depending on a threshold value of some feature. */
		typedef DecisionTrees::LeafNode<Y> LeafNode; /**< Terminal node, which returns a constant prediction value for features which ended up on it. */

		/** Constructs a decision tree by taking ownership of a root node. */
		DecisionTree(std::unique_ptr<Node>&& root)
			: root_(std::move(root))
		{
			if (!root_) {
				throw std::invalid_argument("Null root");
			}
			if (root_->parent) {
				throw std::invalid_argument("Root has no parent");
			}
			root_->collect_lowest_split_nodes(lowest_split_nodes_);
		}

		/** Move constructor. */
		DecisionTree(DecisionTree<Y>&& other) noexcept
			: root_(std::move(other.root_)), lowest_split_nodes_(std::move(other.lowest_split_nodes_))
		{}

		/** Copy constructor. */
		DecisionTree(const DecisionTree<Y>& other)
			: root_(other.root_->clone(nullptr))
		{
			root_->collect_lowest_split_nodes(lowest_split_nodes_);
		}

		/** Move assignment operator. */
		DecisionTree<Y>& operator=(DecisionTree<Y>&& other) noexcept
		{
			if (this != &other) {
				root_ = std::move(other.root_);
				lowest_split_nodes_ = std::move(other.lowest_split_nodes_);
			}
			return *this;
		}

		/** Copy assignment operator. */
		DecisionTree<Y>& operator=(const DecisionTree<Y>& other)
		{
			if (this != &other) {
				root_.reset(other.root_->clone(nullptr));
				lowest_split_nodes_.clear();
				root_->collect_lowest_split_nodes(lowest_split_nodes_);
			}
			return *this;
		}

		/** Returns a prediction given a feature vector. */
		Y operator()(arg_type x) const
		{
			return (*root_)(x);
		}

		/** Counts nodes in the tree. */
		unsigned int count_nodes() const
		{
			return 1 + root_->count_lower_nodes();
		}

		/** Counts leaf nodes in the tree. */
		unsigned int count_leaf_nodes() const
		{
			return root_->count_leaf_nodes();
		}

		/** Returns the prediction error for training data before any splits are made. */
		double original_error() const
		{
			return root_->error;
		}

		/** Returns the total prediction error for training data after all splits. */
		double total_leaf_error() const
		{
			return root_->total_leaf_error();
		}

		/** Calculates cost-complexity for given alpha. */
		double cost_complexity(double alpha) const
		{
			return total_leaf_error() + alpha * static_cast<double>(count_leaf_nodes());
		}

		/** Finds the weakest link and remove it, if the error does not increase too much.

		A "weakest link" is a split node which can be collapsed with the minimum increase of total_leaf_error().
		Only the lowest split node can be a weakest link.
		@param max_allowed_error_increase Maximum allowed increase in total leaf error.
		@return Whether a node was removed.
		@throw std::domain_error If max_allowed_error_increase < 0.
		*/
		bool remove_weakest_link(const double max_allowed_error_increase)
		{
			if (max_allowed_error_increase < 0) {
				throw std::domain_error("Maximum allowed error increase cannot be negative");
			}
			if (lowest_split_nodes_.empty()) {
				return false;
			}
			double lowest_error_increase = std::numeric_limits<double>::infinity();
			SplitNode* removed = nullptr;			
			for (auto split_node_ptr : lowest_split_nodes_) {
				assert(split_node_ptr);
				const SplitNode& split_node = *split_node_ptr;
				assert(split_node.lower);
				assert(split_node.higher);
				assert(split_node.lower->is_leaf());
				assert(split_node.higher->is_leaf());
				const double error_increase = split_node.error - (split_node.lower->error + split_node.higher->error);
				if (error_increase <= lowest_error_increase) {
					removed = split_node_ptr;
					lowest_error_increase = error_increase;
				}
			}
			assert(removed);
			assert(lowest_error_increase >= 0);
			if (lowest_error_increase > max_allowed_error_increase) {
				return false;
			}
			SplitNode* const parent_of_removed = removed->parent;
			auto new_leaf = std::make_unique<LeafNode>(removed->error, removed->value, parent_of_removed);
			if (parent_of_removed) {
				// Removing a non-root node from pruned tree.
				bool other_is_leaf;
				if (removed == parent_of_removed->lower.get()) {
					parent_of_removed->lower = std::move(new_leaf);
					other_is_leaf = parent_of_removed->higher->is_leaf();
				} else {
					assert(removed == parent_of_removed->higher.get());
					parent_of_removed->higher = std::move(new_leaf);
					other_is_leaf = parent_of_removed->lower->is_leaf();
				}
				// Update the set of lowest split nodes.
				lowest_split_nodes_.erase(removed);
				assert(!lowest_split_nodes_.count(removed));
				if (other_is_leaf) {
					lowest_split_nodes_.insert(parent_of_removed);
					assert(lowest_split_nodes_.count(parent_of_removed));
				}
			} else {
				// We removed the last split. Replace the pruned tree with the new leaf.
				root_ = std::move(new_leaf);
				lowest_split_nodes_.clear();
			}
			return true;
		}

		/** Counts lowest split nodes. */
		unsigned int number_lowest_split_nodes() const
		{
			return static_cast<unsigned int>(lowest_split_nodes_.size());
		}
	private:
		std::unique_ptr<Node> root_;
		std::unordered_set<SplitNode*> lowest_split_nodes_; /** Set of lowest split nodes. */
	};
}