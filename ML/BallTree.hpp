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
     * See https://en.wikipedia.org/wiki/Ball_tree and Omohundro, Stephen M. (1989) "Five Balltree Construction Algorithms".
     * 
    */
    class BallTree
    {
    public:
        /**
         * @brief Constructor taking only features.
         * @param X Feature matrix, with data points in columns. Copied internally.
         * @param min_split_size Minimum number of points in the ball to consider splitting it into child balls.
         * @throw std::invalid_argument If `min_split_size < 3`.
        */
        DLL_DECLSPEC BallTree(Eigen::Ref<const Eigen::MatrixXd> X, unsigned int min_split_size);

        /**
         * @brief Constructor taking features and labels.
         * @param X Feature matrix, with data points in columns. Copied internally.
         * @param y Label vectors, with size equal to number of data points or empty. Copied internally.
         * @param min_split_size Minimum number of points in the ball to consider splitting it into child balls.
         * @throw std::invalid_argument If `min_split_size < 3` or `y.size() != X.cols()`.
        */
        DLL_DECLSPEC BallTree(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, unsigned int min_split_size);

        /**
         * @brief Returns const reference to feature vectors (reordered).
        */
        const Eigen::MatrixXd& data() const
        {
            return data_;
        }        

        /**
         * @brief Returns const reference to labels (reordered).
        */
        const Eigen::VectorXd& labels() const
        {
            return labels_;
        }

        /**
         * @brief Finds up to k nearest neighbours for given target vector.
         * Uses the KNS1 algorithm from http://people.ee.duke.edu/~lcarin/liu06a.pdf
         * @param[in] x Target vector.
         * @param[in] k Number of nearest neighbours.
         * @param[out] nn Upon return, has size `min(k, size())` and contains the indices (in #data()) of up to `k` nearest neighbours of `x`.
         * @throw std::invalid_argument If `x.size() != #dim()`.
        */
        DLL_DECLSPEC void find_k_nearest_neighbours(Eigen::Ref<const Eigen::VectorXd> x, unsigned int k, std::vector<unsigned int>& nn) const;        

        /**
         * @brief Finds up to k nearest neighbours for given target vector.
         * Uses the KNS1 algorithm from http://people.ee.duke.edu/~lcarin/liu06a.pdf
         * @param[in] x Target vector.
         * @param[in] k Number of nearest neighbours.
         * @param[out] nn Upon return, contains up to k nearest neighbours of `x`.
         * @return Number of nearest neighbours found.
         * @throw std::invalid_argument If `x.size() != #dim()`, `nn.rows() != #dim()` or `nn.cols()` is lower than the number of nearest neighbours found.
        */
        DLL_DECLSPEC unsigned int find_k_nearest_neighbours(Eigen::Ref<const Eigen::VectorXd> x, unsigned int k, Eigen::Ref<Eigen::MatrixXd> nnX, Eigen::Ref<Eigen::VectorXd> nny) const;

        /**
         * @brief Size of the tree (number of vectors).
        */
        auto size() const
        {
            return static_cast<unsigned int>(data_.cols());
        }

        /**
         * @brief Dimension of the feature vectors.
        */
        auto dim() const
        {
            return static_cast<unsigned int>(data_.rows());
        }
    private:
        struct Node
        {
            double radius; /**< Radius of the ball. */
            unsigned int pivot_index; /**< Index of the pivot vector in data_. */
            unsigned int start_index; /**< Index of the first ball vector in data_. */
            unsigned int end_index; /**< One-past-end index of the ball vectors in data_. */
            std::unique_ptr<Node> left_child;
            std::unique_ptr<Node> right_child;
        };

        Eigen::MatrixXd data_; /**< Ball tree feature vectors. */
        Eigen::VectorXd labels_; /**< Ball tree feature labels. */
        std::unique_ptr<Node> root_; /**< Ball tree root. */
        unsigned int min_split_size_;

        /**
         * @brief Recursively constructs a ball tree.
         * @param work Matrix block (a view on data_), containing the feature vectors being processed.
         * @param labels Vector segment (a view on labels_), containing the labels being processed.
         * @param offset Offset of the work block columns relative to the beginning of the data_ matrix.
         * @param node Current tree node.
         * @param features Range of iterators to a work vector used for sorting feature values.         
        */
        void construct(Eigen::Ref<Eigen::MatrixXd> work, Eigen::Ref<Eigen::VectorXd> labels, unsigned int offset, std::unique_ptr<Node>& node, Features::VectorRange<Features::IndexedFeatureValue> features);

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
         * @param node Pointer to current node being searched.
         * @param q Max-first priority queue storing current NN candidates. Ordered by distance from `x`.
        */
        void knn_search(Eigen::Ref<const Eigen::VectorXd> x, unsigned int k, const Node* node, MaxDistancePriorityQueue& q) const;

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
