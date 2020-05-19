package dectree

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type base_node struct {
	error float64
	value mat.Vector
}

func (n base_node) get_error() float64 {
	return n.error
}

func (n base_node) get_value() mat.Vector {
	return n.value
}

// Internal node of the decision tree.
type split_node struct {
	base_node
	lower         node
	higher        node
	threshold     float64
	feature_index int
}

func (n split_node) evaluate(v mat.Vector) mat.Vector {
	if v.AtVec(n.feature_index) < n.threshold {
		return n.lower.evaluate(v)
	} else {
		return n.higher.evaluate(v)
	}
}

func (n split_node) count_lower_nodes() int {
	return 2 + n.lower.count_lower_nodes() + n.higher.count_lower_nodes()
}

func (n split_node) count_leaf_nodes() int {
	return n.lower.count_leaf_nodes() + n.higher.count_leaf_nodes()
}

func (n split_node) total_leaf_error() float64 {
	return n.lower.total_leaf_error() + n.higher.total_leaf_error()
}

func (n split_node) find_weakest_link(parent *split_node) weakest_link_info {
	lower_weakest := n.lower.find_weakest_link(&n)
	higher_weakest := n.higher.find_weakest_link(&n)

	var child_weakest weakest_link_info

	if lower_weakest.total_leaf_error_increase < higher_weakest.total_leaf_error_increase {
		child_weakest = lower_weakest
	} else {
		child_weakest = higher_weakest
	}

	total_leaf_error := lower_weakest.top_total_leaf_error + higher_weakest.top_total_leaf_error

	increase := n.error - total_leaf_error

	// Prefer collapsing higher nodes in case of a tie.
	if increase < child_weakest.total_leaf_error_increase {
		return weakest_link_info{&n, parent, increase, total_leaf_error}
	} else {
		child_weakest.top_total_leaf_error = total_leaf_error
		return child_weakest
	}
}

func (n split_node) clone() node {
	return split_node{base_node{n.error, n.value}, n.lower, n.higher, n.threshold, n.feature_index}
}

// Leaf node of the decision tree.
type leaf_node struct {
	base_node
}

func (n leaf_node) evaluate(mat.Vector) mat.Vector {
	return n.value
}

func (n leaf_node) count_lower_nodes() int {
	return 0
}

func (n leaf_node) count_leaf_nodes() int {
	return 1
}

func (n leaf_node) total_leaf_error() float64 {
	return n.error
}

func (n leaf_node) find_weakest_link(parent *split_node) weakest_link_info {
	// A leaf node cannot be collapsed.
	return weakest_link_info{nil, parent, math.Inf(1), n.error}
}

func (n leaf_node) clone() node {
	return leaf_node{base_node{n.error, n.value}}
}

type weakest_link_info struct {
	// pointer to weakest link
	node *split_node

	// pointer to its parent. Root node has a nil parent.
	parent *split_node

	// increase in total_leaf_error() if "node" is collapsed
	total_leaf_error_increase float64

	// Total leaf error of the node returning this struct (return to avoid a double recursion)
	top_total_leaf_error float64
}

type node interface {
	get_error() float64

	get_value() mat.Vector

	evaluate(mat.Vector) mat.Vector

	count_lower_nodes() int

	count_leaf_nodes() int

	total_leaf_error() float64

	find_weakest_link(parent *split_node) weakest_link_info

	clone() node
}
