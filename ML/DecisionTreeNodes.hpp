#pragma once
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <Eigen/Core>

namespace ml
{
	/** Helper functions and classes for decision trees. */
	namespace DecisionTrees 
	{
		template <class Y> struct SplitNode;

		template <class Y> struct Node
		{
			typedef Eigen::Ref<const Eigen::VectorXd> arg_type;

			double error; /**< Error of the training sample seen by this node, used for pruning. */
			Y value; /**< Value which should be returned if we stop splitting at this node. */
			SplitNode<Y>* parent; /**< Link to parent node. */

			Node(double n_error, Y n_value, SplitNode<Y>* n_parent)
				: error(n_error), value(n_value), parent(n_parent)
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

			/** Make a perfect copy of the node.
			Function works recursively from root to leafs.
			@param cloned_parent Pointer to already cloned parent.
			*/
			virtual Node* clone(SplitNode<Y>* cloned_parent) const = 0;

			/** Return true if node is a leaf. */
			virtual bool is_leaf() const = 0;

			/** Walk over this node and all below it, adding to s every split node which has only leaf nodes as children. */
			virtual void collect_lowest_split_nodes(std::unordered_set<SplitNode<Y>*>& s) = 0;
		};

		template <class Y> struct SplitNode : public Node<Y>
		{
			std::unique_ptr<Node<Y>> lower; /**< Followed if x[feature_index] < threshold. */
			std::unique_ptr<Node<Y>> higher; /**< Followed if x[feature_index] >= threshold. */
			double threshold;
			unsigned int feature_index;

			using Node<Y>::arg_type;
			using Node<Y>::error;
			using Node<Y>::value;
			using Node<Y>::parent;

			SplitNode(double n_error, Y n_value, SplitNode<Y>* n_parent, double n_threshold, unsigned int n_feature_index)
				: Node<Y>(n_error, n_value, n_parent), threshold(n_threshold), feature_index(n_feature_index)
			{}

			Y operator()(arg_type x) const override
			{
				assert(lower);
				assert(higher);
				assert(this == lower->parent);
				assert(this == higher->parent);
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
				assert(this == lower->parent);
				assert(this == higher->parent);
				return 2 + lower->count_lower_nodes() + higher->count_lower_nodes();
			}

			unsigned int count_leaf_nodes() const override
			{
				assert(lower);
				assert(higher);
				assert(this == lower->parent);
				assert(this == higher->parent);
				return lower->count_leaf_nodes() + higher->count_leaf_nodes();
			}

			double total_leaf_error() const override
			{
				assert(lower);
				assert(higher);
				assert(this == lower->parent);
				assert(this == higher->parent);
				return lower->total_leaf_error() + higher->total_leaf_error();
			}

			SplitNode<Y>* clone(SplitNode<Y>* cloned_parent) const override
			{
				assert(lower);
				assert(higher);
				assert(this == lower->parent);
				assert(this == higher->parent);
				auto copy = std::make_unique<SplitNode<Y>>(error, value, cloned_parent, threshold, feature_index);
				copy->lower = std::unique_ptr<Node<Y>>(lower->clone(copy.get()));
				copy->higher = std::unique_ptr<Node<Y>>(higher->clone(copy.get()));
				return copy.release();
			}

			bool is_leaf() const override
			{
				return false;
			}

			void collect_lowest_split_nodes(std::unordered_set<SplitNode<Y>*>& s) override
			{
				assert(lower);
				assert(higher);
				assert(this == lower->parent);
				assert(this == higher->parent);
				int number_leaves = 0;
				if (!lower->is_leaf()) {
					lower->collect_lowest_split_nodes(s);
				} else {
					++number_leaves;
				}
				if (!higher->is_leaf()) {
					higher->collect_lowest_split_nodes(s);
				} else {
					++number_leaves;
				}
				if (number_leaves == 2) {
					assert(lower->is_leaf());
					assert(higher->is_leaf());
					s.insert(this);
				}
			}
		};

		template <class Y> struct LeafNode : public Node<Y>
		{
			LeafNode(double n_error, Y n_value, SplitNode<Y>* n_parent)
				: Node<Y>(n_error, n_value, n_parent)
			{}

			using Node<Y>::arg_type;
			using Node<Y>::error;
			using Node<Y>::value;
			using Node<Y>::parent;

			Y operator()(arg_type x) const override
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

			LeafNode* clone(SplitNode<Y>* cloned_parent) const override
			{
				return new LeafNode<Y>(error, value, cloned_parent);
			}

			bool is_leaf() const override
			{
				return true;
			}

			void collect_lowest_split_nodes(std::unordered_set<SplitNode<Y>*>&) override
			{}
		};
	}
}