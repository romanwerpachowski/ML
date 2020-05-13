#pragma once
#include <cassert>
#include <memory>
#include <stdexcept>
#include <utility>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	template <class Y> class DecisionTree
	{
	public:
		typedef Eigen::Ref<const Eigen::VectorXd> arg_type;
		typedef Y value_type;

		struct Node
		{
			double error; /**< Error of the training sample seen by this node. */

			Node(double n_error)
				: error(n_error)
			{
				if (error < 0) {
					throw std::domain_error("Node error cannot be negative");
				}
			}

			virtual ~Node() {}

			virtual Y operator()(arg_type x) const = 0;

			virtual unsigned int count_lower_nodes() const = 0;

			/** Total error of the training samples seen by the leaf nodes reachable from this node (including its own if leaf). 
			
			Has the invariant total_leaf_error() <= error.
			*/
			virtual double total_leaf_error() const = 0;
		};

		DecisionTree(std::unique_ptr<Node>&& root)
			: root_(std::move(root))
		{
			if (!root_) {
				throw std::invalid_argument("Null root");
			}
		}

		struct SplitNode : public Node
		{
			std::unique_ptr<Node> lower; /**< Followed if x[feature_index] < threshold. */
			std::unique_ptr<Node> higher; /**< Followed if x[feature_index] >= threshold. */
			double threshold;
			unsigned int feature_index;

			SplitNode(double n_error, double n_threshold, unsigned int n_feature_index)
				: Node(n_error), threshold(n_threshold), feature_index(n_feature_index)
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

			double total_leaf_error() const override
			{
				assert(lower);
				assert(higher);
				return lower->total_leaf_error() + higher->total_leaf_error();
			}
		};

		struct LeafNode : public Node
		{
			LeafNode(double n_error, Y n_value)
				: Node(n_error), value(n_value)
			{}

			using Node::error;

			Y value;

			double operator()(arg_type x) const override
			{
				return value;
			}

			unsigned int count_lower_nodes() const override
			{
				return 0;
			}

			double total_leaf_error() const override
			{
				return error;
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

		double original_error() const 
		{
			return root_->error;
		}

		double total_leaf_error() const
		{
			return root_->total_leaf_error();
		}
	private:
		std::unique_ptr<Node> root_;		
	};

	typedef DecisionTree<double> RegressionTree1D;

	/// Data points are in columns ///

	CLASS_DECLSPEC std::pair<unsigned int, double> find_best_split_reg_1d(const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y);

	/**
	@param max_split_levels Maximum number of split nodes on the way to any leaf node.
	@param min_sample_size Minimum sample size which can be split.
	*/
	CLASS_DECLSPEC RegressionTree1D tree_regression_1d(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size);
}