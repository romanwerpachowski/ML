#pragma once
/** (C) 2021 Roman Werpachowski */
#include <queue>
#include <utility>
#include <Eigen/Core>
#include "Features.hpp"
#include "dll.hpp"

namespace ml
{
    /**
     * @brief Ball tree: an efficient tree structure for nearest-neighbour search in R^D space.
     * 
     * See https://en.wikipedia.org/wiki/Ball_tree
     * 
    */
    class BallTree
    {
    public:
        /**
         * @brief Constructor.
         * @param X Feature matrix, with data points in columns. Copied internally (twice).
        */
        DLL_DECLSPEC BallTree(Eigen::Ref<const Eigen::MatrixXd> X);

        /**
         * @brief Returns const reference to feature vectors ordered as binary tree. The root is `#tree().col(0)`.
        */
        const Eigen::MatrixXd& tree() const
        {
            return tree_;
        }

        /**
         * @brief Returns the index of the left child of the i-th element in the tree (#tree().col(i)), or `#size()` if this node has no left children.
         *
         * @throw std::out_of_range If `i >= #size()`.
        */
        DLL_DECLSPEC unsigned int left_child_index(unsigned int i) const;

        /**
         * @brief Returns the index of the right child of the i-th element in the tree (#tree().col(i)), or `#size()` if this node no right children.
         *
         * @throw std::out_of_range If `i >= #size()`.
        */
        DLL_DECLSPEC unsigned int right_child_index(unsigned int i) const;

        /**
         * @brief Finds up to k nearest neighbours for given target vector.
         * @param[in] x Target vector.
         * @param[in] k Number of nearest neighbours.
         * @param[out] nn Upon return, has size `min(k, size())` and contains the indices of up to `k` nearest neighbours of `x`.
         * @throw std::invalid_argument If `x.size() != #tree().rows()`.
        */
        DLL_DECLSPEC void find_k_nearest_neighbours(Eigen::Ref<const Eigen::VectorXd> x, unsigned int k, std::vector<unsigned int>& nn) const;

        /**
         * @brief Finds up to k nearest neighbours for given target vector.
         * @param[in] x Target vector.
         * @param[in] k Number of nearest neighbours.
         * @param[out] nn Upon return, contains up to k nearest neighbours of `x`.
         * @return Number of nearest neighbours found.
         * @throw std::invalid_argument If `x.size() != #tree().rows()`, `nn.rows() != #tree().rows()` or `nn.cols()` is lower than the number of nearest neighbours found.
        */
        DLL_DECLSPEC unsigned int find_k_nearest_neighbours(Eigen::Ref<const Eigen::VectorXd> x, unsigned int k, Eigen::Ref<Eigen::MatrixXd> nn) const;        

        /**
         * @brief Size of the tree.
        */
        auto size() const
        {
            return static_cast<unsigned int>(tree_.cols());
        }
    private:
        Eigen::MatrixXd tree_; /**< Ball tree data. */
        Eigen::VectorXd radii_; /**< radii_[i] is the radius of the subtree with root in tree_[i]. */
        std::vector<unsigned int> num_left_children_; /**< Number of left children for every node. */
        std::vector<unsigned int> num_right_children_; /**< Number of right children for every node. */

        /**
         * @brief Recursively constructs a ball tree.
         * @param work Work matrix, containing features being processed.
         * @param i Index of the current node in the tree.
         * @param features Range of iterators to a work vector used for sorting feature values.
        */
        void construct(Eigen::Ref<Eigen::MatrixXd> work, unsigned int i, Features::VectorRange<Features::IndexedFeatureValue> features);

        /**
         * @brief Pair of (index of the feature vector in the tree, distance of the tree feature vector from the target vector).
        */
        typedef std::pair<unsigned int, double> IndexedDistanceFromTarget;

        /**
         * @brief Compares `IndexedDistanceFromTarget` objects so that furthest vectors come first in the queue.
        */
        struct IndexedDistanceFromTargetComparator
        {
            bool operator()(const IndexedDistanceFromTarget& a, const IndexedDistanceFromTarget& b) const
            {
                return a.second < b.second;
            }
        };

        /**
         * @brief Max-first priority queue used to store the current candidates for k nearest neighbours.
        */
        typedef std::priority_queue<IndexedDistanceFromTarget, std::vector<IndexedDistanceFromTarget>, IndexedDistanceFromTargetComparator> MaxDistancePriorityQueue;

        /**
         * @brief Recursively finds k nearest neighbours (or less if there's not enough data) of the target vector.
         * @param x Target feature vector.
         * @param k Max. number of neighbours to find.
         * @param i Index of the current node in the tree where we search.
         * @param q Max-first priority queue storing current NN candidates. Ordered by distance from `x`.
        */
        void knn_search(Eigen::Ref<const Eigen::VectorXd> x, unsigned int k, unsigned int i, MaxDistancePriorityQueue& q) const;

        /**
         * @brief Gets the max. distance of `x` from the candidates in the queue.
         * @param x Target feature vector. 
         * @param k Max. number of neighbours to find.
         * @param q Max-first priority queue storing current NN candidates. Ordered by distance from `x`.
         * @return Max. distance if `q.size() == k`, +Inf otherwise.
        */
        double distance_from_queue(Eigen::Ref<const Eigen::VectorXd> x, unsigned int k, const MaxDistancePriorityQueue& q) const;
    };
}
