#pragma once
/** (C) 2021 Roman Werpachowski */
#include <queue>
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
         * @brief Returns const reference to feature vectors ordered as binary tree.
        */
        const Eigen::MatrixXd& tree() const
        {
            return tree_;
        }

        /**
         * @brief Finds up to k nearest neighbours for given target vector.
         * @param[in] x Target vector.
         * @param[in] k Number of nearest neighbours.
         * @param[out] nn Upon return, has size `min(k, tree().cols())` and contains the indices of up to `k` nearest neighbours of `x`.
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
    private:
        Eigen::MatrixXd tree_; /**< Ball tree data. */
        Eigen::VectorXd radii_; /**< radii_[i] is the radius of the subtree with root in tree_[i]. */
        std::vector<unsigned int> num_left_children_; /**< Number of left children for every node. */
        std::vector<unsigned int> num_right_children_; /**< Number of right children for every node. */

        void construct(Eigen::Ref<Eigen::MatrixXd> work, Eigen::Index i, Features::VectorRange<Features::IndexedFeatureValue> features);

        void knn_search(Eigen::Ref<const Eigen::VectorXd> x, unsigned int k, Eigen::Index i, std::queue<unsigned int>& q) const;

        double distance_from_queue(Eigen::Ref<const Eigen::VectorXd> x, const std::queue<unsigned int>& q) const;
    };
}