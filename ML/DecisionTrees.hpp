#pragma once
#include <utility>
#include <Eigen/Core>
#include "DecisionTree.hpp"
#include "dll.hpp"

namespace ml
{
	/** @brief Decision tree for linear regression. */
	typedef DecisionTree<double> RegressionTree;

	/** @brief Decision tree for multinomial classification. */
	typedef DecisionTree<unsigned int> ClassificationTree;

	/** @brief Functions for manipulating decision trees. */
	namespace DecisionTrees
	{
		/** @brief Pair of vector iterators. */
		template <typename T> using VectorRange = std::pair<typename std::vector<T>::iterator, typename std::vector<T>::iterator>;

		/** @brief Grows a regression tree with pruning.
		@param[in] X Independent variables (column-wise).
		@param[in] y Dependent variable.
		@param[in] max_split_levels Maximum number of split nodes on the way to any leaf node.
		@param[in] min_sample_size Minimum sample size which can be split (at least 2).
		@param[in] alphas Candidate alphas for pruning to be selected by cross-validation. If this vector is empty, no pruning is done. If it has just one element, this value is used for pruning. If it has more than one, the one with smallest k-fold cross-validation test error is used.
		@param[in] num_folds Number of folds for cross-validation. Ignored if cross-validation is not done.
		@return Tuple of: trained regression tree, chosen alpha (NaN if no pruning was done) and minimum cross-validation test error (NaN if no cross-validation was done).
		@throw std::invalid_argument If `min_sample_size < 2`, `y.size() < 2` or `X.cols() != y.size()`.
		*/
		DLL_DECLSPEC std::tuple<RegressionTree, double, double> regression_tree_auto_prune(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size, const std::vector<double>& alphas, const unsigned int num_folds);

		/** @brief Grows a classification tree with pruning.
		@param[in] X Features (column-wise).
		@param[in] y Class indices.
		@param[in] max_split_levels Maximum number of split nodes on the way to any leaf node.
		@param[in] min_sample_size Minimum sample size which can be split (at least 2).
		@param[in] alphas Candidate alphas for pruning to be selected by cross-validation. If this vector is empty, no pruning is done. If it has just one element, this value is used for pruning. If it has more than one, the one with smallest k-fold cross-validation test error is used.
		@param[in] num_folds Number of folds for cross-validation. Ignored if cross-validation is not done.
		@return Tuple of: trained classification tree, chosen alpha (NaN if no pruning was done) and minimum cross-validation test error (NaN if no cross-validation was done).
		@throw std::invalid_argument If `min_sample_size < 2`, `y.size() < 2` or `X.cols() != y.size()`.
		*/
		DLL_DECLSPEC std::tuple<ClassificationTree, double, double> classification_tree_auto_prune(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size, const std::vector<double>& alphas, const unsigned int num_folds);		

		/** @brief Grows a regression tree without pruning.
		@param[in] X Independent variables (column-wise).
		@param[in] y Dependent variable.
		@param[in] max_split_levels Maximum number of split nodes on the way to any leaf node.
		@param[in] min_sample_size Minimum sample size which can be split (at least 2).
		@return Trained regression tree.
		@throw std::invalid_argument If `min_sample_size < 2`, `y.size() < 2` or `X.cols() != y.size()`.
		*/
		DLL_DECLSPEC RegressionTree regression_tree(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size);

		/** @brief Grows a classification tree without pruning.
		@param[in] X Classification features (column-wise).
		@param[in] y Class indices.
		@param[in] max_split_levels Maximum number of split nodes on the way to any leaf node.
		@param[in] min_sample_size Minimum sample size which can be split (at least 2).
		@return Trained classification tree.
		@throw std::invalid_argument If `min_sample_size < 2`, `y.size() < 2` or `X.cols() != y.size()`.
		*/
		DLL_DECLSPEC ClassificationTree classification_tree(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int max_split_levels, unsigned int min_sample_size);

		/** @brief  Performs cost-complexity pruning in-place.

		@param[in] alpha Cost of complexity per node.
		@tparam Y Decision tree output value type.
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

		/** @brief Calculates tree mean squared error (MSE) for a sample.

		MSE for tree \f$ f \f$ is defined as
		\f$ N^{-1} \sum_{i=1}^N (y_i - f(\vec{X}_i))^2 \f$.
		
		@param[in] tree Regression tree.
		@param[in] X Independent variables (column-wise).
		@param[in] y Dependent variable.

		@return MSE or NaN if `y.size() == 0`.

		@throw std::invalid_argument If `X.cols() != y.size()`.
		*/
		DLL_DECLSPEC double regression_tree_mean_squared_error(const RegressionTree& tree, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Calculates tree accuracy for a sample.

		Accuracy for classification tree \f$ f \f$ is defined as
		\f$ N^{-1} \sum_{i=1}^N \mathbb{1}_{y_i = f(\vec{X}_i)} \f$.
		
		@param[in] tree tree Classification tree.
		@param[in] X Classification features (column-wise).
		@param[in] y Class indices.

		@throw std::invalid_argument If `X.cols() != y.size()`.
		*/
		DLL_DECLSPEC double classification_tree_accuracy(const ClassificationTree& tree, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y);

		/** @brief Calculates tree misclassification rate for a sample.

		Misclassification rate for classification tree \f$ f \f$ is defined as
		\f$ N^{-1} \sum_{i=1}^N \mathbb{1}_{y_i \neq f(\vec{X}_i)} \f$.
		
		@param[in] tree Classification tree.
		@param[in] X Classification features (column-wise).
		@param[in] y Class indices.

		@throw std::invalid_argument If `X.cols() != y.size()`.
		*/
		inline double classification_tree_misclassification_rate(const ClassificationTree& tree, Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y)
		{
			return 1 - classification_tree_accuracy(tree, X, y);
		}

		/** @private Finds the split on a single feature which minimises the sum of SSEs of split samples.
		This function is not meant to be used directly. It's exposed for testing.
		*/
		DLL_DECLSPEC std::pair<unsigned int, double> find_best_split_regression(
			const Eigen::Ref<const Eigen::MatrixXd> X,
			const Eigen::Ref<const Eigen::VectorXd> y,
			Eigen::Ref<Eigen::VectorXd> sorted_y,
			VectorRange<std::pair<Eigen::Index, double>> features);		
	}
}