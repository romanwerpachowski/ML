#pragma once
/** (C) 2021 Roman Werpachowski */
#include <Eigen/Core>

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
        BallTree(Eigen::Ref<const Eigen::MatrixXd> X);
    private:
        Eigen::MatrixXd tree_; /**< Ball tree stored as an array. */        
    };
}