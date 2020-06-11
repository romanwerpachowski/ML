#pragma once
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <Eigen/Core>
#include "DecisionTreeNodes.hpp"
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
		typedef DecisionTrees::Node<Y> Node;
		typedef DecisionTrees::SplitNode<Y> SplitNode;
		typedef DecisionTrees::LeafNode<Y> LeafNode;

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

		DecisionTree(DecisionTree<Y>&& other)
			: root_(std::move(other.root_)), lowest_split_nodes_(std::move(lowest_split_nodes_))
		{}

		DecisionTree(const DecisionTree<Y>& other)
			: root_(other.root_->clone(nullptr))
		{
			root_->collect_lowest_split_nodes(lowest_split_nodes_);
		}

		DecisionTree<Y>& operator=(DecisionTree<Y>&& other) noexcept
		{
			if (this != &other) {
				root_ = std::move(other.root_);
				lowest_split_nodes_ = std::move(other.lowest_split_nodes_);
			}
			return *this;
		}

		DecisionTree<Y>& operator=(const DecisionTree<Y>& other)
		{
			if (this != &other) {
				root_.reset(other.root_->clone(nullptr));
				lowest_split_nodes_.clear();
				root_->collect_lowest_split_nodes(lowest_split_nodes_);
			}
			return *this;
		}

		Y operator()(arg_type x) const
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

		/** Find the weakest link and remove it, if the error does not increase too much.

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

		/** Returns number of lowest split nodes. */
		unsigned int number_lowest_split_nodes() const
		{
			return static_cast<unsigned int>(lowest_split_nodes_.size());
		}
	private:
		std::unique_ptr<Node> root_;
		std::unordered_set<SplitNode*> lowest_split_nodes_; /** Set of lowest split nodes. */
	};

	typedef DecisionTree<double> UnivariateRegressionTree;
	typedef DecisionTree<unsigned int> ClassificationTree;


	/** Helper functions and classes for decision trees. */
	namespace DecisionTrees
	{
		/** Pair of vector iterators. */
		template <typename T> using VectorRange = std::pair<typename std::vector<T>::iterator, typename std::vector<T>::iterator>;

		/** Used to sort features by value. */
		typedef std::pair<Eigen::Index, double> IndexedFeatureValue;

		/** Creates an iterator pair containing begin() and end(). */
		template <typename T> VectorRange<T> from_vector(std::vector<T>& v)
		{
			return std::make_pair(v.begin(), v.end());
		}

		/** Finds the split on a single feature which minimises the sum of SSEs of split samples.

		This function is not meant to be used directly. It's exposed for testing.
		*/
		DLL_DECLSPEC std::pair<unsigned int, double> find_best_split_univariate_regression(
			const Eigen::Ref<const Eigen::MatrixXd> X,
			const Eigen::Ref<const Eigen::VectorXd> y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			VectorRange<std::pair<Eigen::Index, double>> features);

		/** Grows a univariate regression tree without pruning.
		@param X Independent variables (column-wise).
		@param y Dependent variable.
		@param max_split_levels Maximum number of split nodes on the way to any leaf node.
		@param min_sample_size Minimum sample size which can be split (at least 2).
		*/
		DLL_DECLSPEC UnivariateRegressionTree univariate_regression_tree(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size);

		/** Grows a classification tree without pruning.
		@param X Classification features (column-wise).
		@param y Class indices.
		@param max_split_levels Maximum number of split nodes on the way to any leaf node.
		@param min_sample_size Minimum sample size which can be split (at least 2).
		*/
		DLL_DECLSPEC ClassificationTree classification_tree(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size);

		/** Performs cost-complexity pruning in-place.

		@param alpha Cost of complexity per node.
		@throw std::domain_error If alpha < 0.
		*/
		template <typename Y> void cost_complexity_prune(DecisionTree<Y>& tree, const double alpha)
		{
			if (alpha < 0) {
				throw std::domain_error("Alpha cannot be negative");
			}
			// There can be only one minimum of cost complexity.
			while (tree.remove_weakest_link(alpha)) {}
		}

		/** Calculates tree mean squared error on (X, y) data. */
		DLL_DECLSPEC double univariate_regression_tree_mean_squared_error(const UnivariateRegressionTree& tree, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** Calculates tree accuracy on (X, y) data. */
		DLL_DECLSPEC double classification_tree_accuracy(const ClassificationTree& tree, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** 1 - tree accuracy. */
		inline double classification_tree_misclassification_rate(const ClassificationTree& tree, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y)
		{
			return 1 - classification_tree_accuracy(tree, X, y);
		}

		/** Grows a univariate regression tree with pruning.
		@param X Independent variables (column-wise).
		@param y Dependent variable.
		@param max_split_levels Maximum number of split nodes on the way to any leaf node.
		@param min_sample_size Minimum sample size which can be split (at least 2).
		@param alphas Candidate alphas for pruning to be selected by cross-validation. If this vector is empty, no pruning is done. If it has just one element, this value is used for pruning. If it has more than one, the one with smallest k-fold cross-validation test error is used.
		@param num_folds Number of folds for cross-validation. Ignored if cross-validation is not done.
		@return Tuple of: trained decision tree, chosen alpha (NaN if no pruning was done) and minimum cross-validation test error (NaN if no cross-validation was done).
		*/
		DLL_DECLSPEC std::tuple<UnivariateRegressionTree, double, double> univariate_regression_tree_auto_prune(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size, const std::vector<double>& alphas, const unsigned int num_folds);

		/** Grows a classification tree with pruning.
		@param X Features (column-wise).
		@param y Class indices.
		@param max_split_levels Maximum number of split nodes on the way to any leaf node.
		@param min_sample_size Minimum sample size which can be split (at least 2).
		@param alphas Candidate alphas for pruning to be selected by cross-validation. If this vector is empty, no pruning is done. If it has just one element, this value is used for pruning. If it has more than one, the one with smallest k-fold cross-validation test error is used.
		@param num_folds Number of folds for cross-validation. Ignored if cross-validation is not done.
		@return Tuple of: trained decision tree, chosen alpha (NaN if no pruning was done) and minimum cross-validation test error (NaN if no cross-validation was done).
		*/
		DLL_DECLSPEC std::tuple<ClassificationTree, double, double> classification_tree_auto_prune(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size, const std::vector<double>& alphas, const unsigned int num_folds);
	}
}