#pragma once
#include <cassert>
#include <memory>
#include <stdexcept>
#include <utility>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
	/** Decision tree.

	Data points are in columns.
	*/
	template <class Y> class DecisionTree
	{
	public:
		typedef Eigen::Ref<const Eigen::VectorXd> arg_type;
		typedef Y value_type;

		struct SplitNode;

		struct Node
		{
			double error; /**< Error of the training sample seen by this node. */
			Y value; /**< Value which should be returned if we stop splitting at this node. */

			Node(double n_error, Y n_value)
				: error(n_error), value(n_value)
			{
				if (error < 0) {
					throw std::domain_error("Node error cannot be negative");
				}
			}

			virtual ~Node() {}

			virtual Y operator()(arg_type x) const = 0;

			/** Total number of nodes reachable from this one. */
			virtual unsigned int count_lower_nodes() const = 0;

			/** Total number of leaf nodes reachable from this one, including itself. */
			virtual unsigned int count_leaf_nodes() const = 0;

			/** Total error of the training samples seen by the leaf nodes reachable from this node (including its own if leaf).

			Has the invariant total_leaf_error() <= error.
			*/
			virtual double total_leaf_error() const = 0;

			/** Make a perfect copy of the node. */
			virtual Node* clone() const = 0;

			/** Find the pointers to the weakest link and its parent in the part of tree beginning with this node (including itself).

			A "weakest link" is a split node which can be collapsed with the minimum increase of total_leaf_error().

			@return Tuple of (pointer to weakest link, pointer to its parent, increase in total_leaf_error(), this->total_leaf_error()). Null pointer to weakest link means no such node could be found. Root node has a null parent. The 4th value is returned to avoid a double recursion.
			*/
			virtual std::tuple<SplitNode*, SplitNode*, double, double> find_weakest_link(SplitNode* parent) = 0;
		};

		DecisionTree(std::unique_ptr<Node>&& root)
			: root_(std::move(root))
		{
			if (!root_) {
				throw std::invalid_argument("Null root");
			}
		}

		DecisionTree(DecisionTree<Y>&& other)
			: root_(std::move(other.root_))
		{}

		DecisionTree(const DecisionTree<Y>& other)
			: root_(other.root_->clone())
		{}

		DecisionTree<Y>& operator=(DecisionTree<Y>&& other)
		{
			if (this != &other) {
				root_ = std::move(other.root_);
			}
			return *this;
		}

		DecisionTree<Y>& operator=(const DecisionTree<Y>& other)
		{
			if (this != &other) {
				root_.reset(other.root_->clone());
			}
			return *this;
		}

		struct SplitNode : public Node
		{
			std::unique_ptr<Node> lower; /**< Followed if x[feature_index] < threshold. */
			std::unique_ptr<Node> higher; /**< Followed if x[feature_index] >= threshold. */
			double threshold;
			unsigned int feature_index;

			using Node::error;
			using Node::value;

			SplitNode(double n_error, Y n_value, double n_threshold, unsigned int n_feature_index)
				: Node(n_error, n_value), threshold(n_threshold), feature_index(n_feature_index)
			{}

			Y operator()(arg_type x) const override
			{
				assert(lower);
				assert(higher);
				if (x[feature_index] < threshold) {
					return (*lower)(x);
				} else {
					return (*higher)(x);
				}
			}

			unsigned int count_lower_nodes() const override
			{
				assert(lower);
				assert(higher);
				return 2 + lower->count_lower_nodes() + higher->count_lower_nodes();
			}

			unsigned int count_leaf_nodes() const override
			{
				assert(lower);
				assert(higher);
				return lower->count_leaf_nodes() + higher->count_leaf_nodes();
			}

			double total_leaf_error() const override
			{
				assert(lower);
				assert(higher);
				return lower->total_leaf_error() + higher->total_leaf_error();
			}

			SplitNode* clone() const override
			{
				assert(lower);
				assert(higher);
				auto copy = std::make_unique<SplitNode>(error, value, threshold, feature_index);
				copy->lower = std::unique_ptr<Node>(lower->clone());
				copy->higher = std::unique_ptr<Node>(higher->clone());
				return copy.release();
			}

			std::tuple<SplitNode*, SplitNode*, double, double> find_weakest_link(SplitNode* parent) override
			{
				assert(lower);
				assert(higher);
				const auto weakest_link_lower = lower->find_weakest_link(this);
				const auto weakest_link_higher = higher->find_weakest_link(this);
				const auto weakest_link_child = std::get<2>(weakest_link_lower) < std::get<2>(weakest_link_higher) ? weakest_link_lower : weakest_link_higher;
				const double total_leaf_error = std::get<3>(weakest_link_lower) + std::get<3>(weakest_link_higher);
				const double increase = error - total_leaf_error;
				// Prefer collapsing higher nodes in case of a tie.
				if (increase <= std::get<2>(weakest_link_child)) {
					return std::make_tuple(this, parent, increase, total_leaf_error);
				} else {
					return std::make_tuple(std::get<0>(weakest_link_child), std::get<1>(weakest_link_child), std::get<2>(weakest_link_child), total_leaf_error);
				}
			}
		};

		struct LeafNode : public Node
		{
			LeafNode(double n_error, Y n_value)
				: Node(n_error, n_value)
			{}

			using Node::error;
			using Node::value;

			double operator()(arg_type x) const override
			{
				return value;
			}

			unsigned int count_lower_nodes() const override
			{
				return 0;
			}

			unsigned int count_leaf_nodes() const override
			{
				return 1;
			}

			double total_leaf_error() const override
			{
				return error;
			}

			LeafNode* clone() const override
			{
				return new LeafNode(error, value);
			}

			std::tuple<SplitNode*, SplitNode*, double, double> find_weakest_link(SplitNode* parent) override
			{
				// A leaf node cannot be collapsed.
				return std::tuple<SplitNode*, SplitNode*, double, double>(nullptr, parent, std::numeric_limits<double>::infinity(), error);
			}
		};

		Y operator()(Eigen::Ref<Eigen::VectorXd> x) const
		{
			return (*root_)(x);
		}

		unsigned int count_nodes() const
		{
			return 1 + root_->count_lower_nodes();
		}

		unsigned int count_leaf_nodes() const
		{
			return root_->count_leaf_nodes();
		}

		double original_error() const
		{
			return root_->error;
		}

		double total_leaf_error() const
		{
			return root_->total_leaf_error();
		}

		double cost_complexity(double alpha) const
		{
			return total_leaf_error() + alpha * static_cast<double>(count_leaf_nodes());
		}

		/** Find the pointers to the weakest link and its parent in the tree.

		A "weakest link" is a split node which can be collapsed with the minimum increase of total_leaf_error().

		@return Triple of (pointer to weakest link, pointer to its parent, increase in total_leaf_error()). Null pointer to weakest link means no such node could be found. Root node has a null parent.
		*/
		std::tuple<SplitNode*, SplitNode*, double> find_weakest_link()
		{
			const auto weakest_link = root_->find_weakest_link(nullptr);
			return std::make_tuple(std::get<0>(weakest_link), std::get<1>(weakest_link), std::get<2>(weakest_link));
		}
	private:
		std::unique_ptr<Node> root_;
	};

	typedef DecisionTree<double> RegressionTree1D;


	/** Helper functions for decision trees. */
	namespace DecisionTrees
	{
		template <typename T> using Range = std::pair<typename std::vector<T>::iterator, typename std::vector<T>::iterator>;

		DLL_DECLSPEC std::pair<unsigned int, double> find_best_split_reg_1d(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y);

		/**
		@param max_split_levels Maximum number of split nodes on the way to any leaf node.
		@param min_sample_size Minimum sample size which can be split (at least 2).
		*/
		DLL_DECLSPEC RegressionTree1D tree_regression_1d(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size);

		template <typename Y> DecisionTree<Y> cost_complexity_prune(const DecisionTree<Y>& full_tree, const double alpha)
		{
			if (alpha < 0) {
				throw std::domain_error("Alpha cannot be negative");
			}
			const auto number_split_nodes = full_tree.count_nodes() - full_tree.count_leaf_nodes();
			if (!number_split_nodes) {
				// No pruning possible.
				return full_tree;
			}
			std::vector<DecisionTree<Y>> trees;
			std::vector<double> cost_complexities;
			trees.reserve(static_cast<size_t>(number_split_nodes) + 1);
			cost_complexities.reserve(trees.capacity());
			trees.push_back(full_tree);
			cost_complexities.push_back(full_tree.cost_complexity(alpha));
			for (unsigned int i = 0; i < number_split_nodes; ++i) {
				// Copy the last tree.
				trees.push_back(trees.back());
				// Prune the copy.
				auto& pruned_tree = trees.back();
				// Find the split node to remove.
				const auto weakest_link = pruned_tree.find_weakest_link();
				typename DecisionTree<Y>::SplitNode* removed_node = std::get<0>(weakest_link);
				typename DecisionTree<Y>::SplitNode* removed_nodes_parent = std::get<1>(weakest_link);
				assert(removed_node);
				auto new_leaf = std::make_unique<typename DecisionTree<Y>::LeafNode>(removed_node->error, removed_node->value);
				if (removed_nodes_parent) {
					// Removing a non-root node from pruned tree.
					if (removed_node == removed_nodes_parent->lower.get()) {
						removed_nodes_parent->lower = std::move(new_leaf);
					} else {
						removed_nodes_parent->higher = std::move(new_leaf);
					}
				} else {
					assert(trees.size() == trees.capacity());
					// We removed the last split. Replace the pruned tree with the new leaf.
					pruned_tree = DecisionTree<Y>(std::move(new_leaf));
				}
				cost_complexities.push_back(pruned_tree.cost_complexity(alpha));
			}
			assert(!cost_complexities.empty());
			// Find the lowest cost complexity.
			const auto best_it = std::min_element(cost_complexities.begin(), cost_complexities.end());
			return trees[static_cast<size_t>(best_it - cost_complexities.begin())];
		}
	}
}