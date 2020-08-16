#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "ML/DecisionTrees.hpp"


namespace py = pybind11;


template <class Y> static void init_decision_tree_class(py::module& m_dec_trees, const char* class_name, const char* docstring)
{
	py::class_<ml::DecisionTree<Y>, std::unique_ptr<ml::DecisionTree<Y>>>(m_dec_trees, class_name)
		.def("__call__", [](const ml::UnivariateRegressionTree& tree, Eigen::Ref<const Eigen::VectorXd> x) -> double {
			return tree(x); }, py::is_operator())
		.def("cost_complexity", &ml::DecisionTree<Y>::cost_complexity, py::arg("alpha"), "Calculates cost-complexity for given alpha.")
		.def_property_readonly("number_nodes", &ml::DecisionTree<Y>::count_nodes, "Number of nodes.")
		.def_property_readonly("number_leaf_nodes", &ml::DecisionTree<Y>::count_leaf_nodes, "Number of leaf nodes.")
		.def_property_readonly("number_lowest_split_nodes", &ml::DecisionTree<Y>::number_lowest_split_nodes, "Number of lowest split nodes.")
		.def_property_readonly("original_error", &ml::DecisionTree<Y>::original_error, "Original error.")
		.def_property_readonly("total_leaf_error", &ml::DecisionTree<Y>::total_leaf_error, "Total leaf error.")
		.doc() = docstring;
}

constexpr unsigned int DEFAULT_MAX_SPLIT_LEVELS = 100;
constexpr unsigned int DEFAULT_MIN_SPLIT_SIZE = 10;
static const std::vector<double> DEFAULT_ALPHAS = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100 };
constexpr unsigned int DEFAULT_NUM_FOLDS = 10;


void init_decision_trees(py::module& m)
{
	auto m_dec_trees = m.def_submodule("decision_trees", "Decision tree algorithms.");

	init_decision_tree_class<double>(m_dec_trees, "UnivariateRegressionTree", "Univariate regression tree");
	init_decision_tree_class<unsigned int>(m_dec_trees, "ClassificationTree", "Classification tree");

	m_dec_trees.def("univariate_regression_tree", &ml::DecisionTrees::univariate_regression_tree_auto_prune, py::arg("X"),
		py::arg("y"), py::arg("max_split_levels") = DEFAULT_MAX_SPLIT_LEVELS, py::arg("min_split_size") = DEFAULT_MIN_SPLIT_SIZE, py::arg("alphas") = DEFAULT_ALPHAS, py::arg("num_folds") = DEFAULT_NUM_FOLDS,
		py::return_value_policy::move, R"(Grows a univariate regression tree with pruning.

Args:
	X: Independent variables (column-wise).
	y: Dependent variable (vector).
	max_split_levels: Maximum number of split nodes on the way to any leaf node.
	min_split_size: Minimum sample size which can be split (at least 2).
	alphas: Candidate alphas for pruning to be selected by cross-validation. If this vector is empty, no pruning is done. If it has just one element, this value is used for pruning. If it has more than one, the one with smallest k-fold cross-validation test error is used. Defaults to [1E-6, 1E-5, ..., 10, 100].
	num_folds: Number of folds for cross-validation. Ignored if cross-validation is not done.

Returns:
	Tuple of: trained decision tree, chosen alpha (NaN if no pruning was done) and minimum cross-validation test error (NaN if no cross-validation was done).)");

	m_dec_trees.def("classification_tree", &ml::DecisionTrees::classification_tree_auto_prune, py::arg("X"),
		py::arg("y"), py::arg("max_split_levels") = DEFAULT_MAX_SPLIT_LEVELS, py::arg("min_split_size") = DEFAULT_MIN_SPLIT_SIZE, py::arg("alphas") = DEFAULT_ALPHAS, py::arg("num_folds") = DEFAULT_NUM_FOLDS,
		py::return_value_policy::move, R"(Grows a classification tree with pruning.

Args:
	X: Independent variables (column-wise).
	y: Dependent variable (vector).
	max_split_levels: Maximum number of split nodes on the way to any leaf node.
	min_split_size: Minimum sample size which can be split (at least 2).
	alphas: Candidate alphas for pruning to be selected by cross-validation. If this vector is empty, no pruning is done. If it has just one element, this value is used for pruning. If it has more than one, the one with smallest k-fold cross-validation test error is used. Defaults to [1E-6, 1E-5, ..., 10, 100].
	num_folds: Number of folds for cross-validation. Ignored if cross-validation is not done.

Returns:
	Tuple of: trained decision tree, chosen alpha (NaN if no pruning was done) and minimum cross-validation test error (NaN if no cross-validation was done).)");

	m_dec_trees.def("univariate_regression_tree_mean_squared_error", &ml::DecisionTrees::univariate_regression_tree_mean_squared_error, py::arg("tree"), py::arg("X"), py::arg("y"),
		R"(Calculates univariate regression tree mean squared error on (X, y) data.

Args:
	tree: Univariate regression tree instance.
	X: Independent variables (column-wise).
	y: Dependent variable (vector).

Returns:
	Mean squared error.)");

	m_dec_trees.def("classification_tree_accuracy", &ml::DecisionTrees::classification_tree_accuracy, py::arg("tree"), py::arg("X"), py::arg("y"),
		R"(Calculates classification tree accuracy on (X, y) data.

Args:
	tree: Classification tree instance.
	X: Features (column-wise).
	y: Classes (vector).

Returns:
	Classification accuracy.)");
}