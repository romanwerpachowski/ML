/** (C) 2021 Roman Werpachowski */
#include <cassert>
#include <iostream>
#include <limits>
#include "BallTree.hpp"

namespace ml
{
    BallTree::BallTree(Eigen::Ref<const Eigen::MatrixXd> X, const unsigned int min_split_size)
        : data_(X), min_split_size_(min_split_size)
    {
        if (min_split_size < 3) {
            throw std::invalid_argument("BallTree: min_split_size must be at least 3");
        }
        std::vector<Features::IndexedFeatureValue> features(size());
        construct(data_, 0, root_, Features::from_vector(features));
    }

    void BallTree::find_k_nearest_neighbours(Eigen::Ref<const Eigen::VectorXd> x, const unsigned int k, std::vector<unsigned int>& nn) const
    {
        if (x.size() != dim()) {
            throw std::invalid_argument("BallTree: wrong feature vector size");
        }
        MaxDistancePriorityQueue q;
        knn_search(x, k, root_.get(), q);
        nn.reserve(q.size());
        nn.clear();
        while (!q.empty()) {
            nn.push_back(q.top().first);
            q.pop();
        }
    }

    unsigned int BallTree::find_k_nearest_neighbours(Eigen::Ref<const Eigen::VectorXd> x, const unsigned int k, Eigen::Ref<Eigen::MatrixXd> nn) const
    {
        if (x.size() != dim()) {
            throw std::invalid_argument("BallTree: wrong feature vector size");
        }
        const auto num_neighbours = std::min(k, size());
        if (static_cast<unsigned int>(nn.cols()) < num_neighbours) {
            throw std::invalid_argument("BallTree: not enough room for all neighbours");
        }
        MaxDistancePriorityQueue q;
        knn_search(x, k, root_.get(), q);
        assert(q.size() == static_cast<size_t>(num_neighbours));
        Eigen::Index i = 0;
        while (!q.empty()) {
            nn.col(i) = data_.col(q.top().first);
            q.pop();
            ++i;
        }
        return num_neighbours;
    }

    static double calc_radius(Eigen::Ref<Eigen::MatrixXd> work, Eigen::Index pivot_idx)
    {
        double max_dist = 0;
        const auto pivot_vec = work.col(pivot_idx);
        for (Eigen::Index j = 0; j < work.cols(); ++j) {
            const double dist = (pivot_vec - work.col(j)).norm();
            if (dist > max_dist) {
                max_dist = dist;
            }
        }
        return max_dist;
    }

    void BallTree::construct(Eigen::Ref<Eigen::MatrixXd> work, const unsigned int offset, std::unique_ptr<Node>& node, Features::VectorRange<Features::IndexedFeatureValue> features)
    {
        assert(node == nullptr);
        if (work.cols() == 0) {
            return;
        } else if (work.cols() == 1) {
            node.reset(new Node{0, offset, offset, offset + 1, nullptr, nullptr});
        } else {
            // Find the dimension of largest spread.
            const auto spreads = work.rowwise().maxCoeff() - work.rowwise().minCoeff();
            assert(spreads.size() == work.rows());
            double max_spread = 0;
            Eigen::Index r = 0;
            for (Eigen::Index k = 0; k < work.rows(); ++k) {
                if (spreads[k] > max_spread) {
                    r = k;
                }
            }
            Features::set_to_nth(work, r, features);
            std::sort(features.first, features.second, Features::INDEXED_FEATURE_COMPARATOR_ASCENDING);
            const auto pivot_iter = features.first + (work.cols() / 2);
            const auto pivot_idx = Features::partition(work, pivot_iter->first, r);
            node.reset(new Node{ calc_radius(work, pivot_idx), offset + static_cast<unsigned int>(pivot_idx), offset, offset + static_cast<unsigned int>(work.cols()), nullptr, nullptr });
            if (work.cols() >= min_split_size_) {
                // Partition work space into Left and Right child features.                
                assert(pivot_idx < work.cols());
                // Make sure both child balls are non-empty.
                auto num_left = std::max(static_cast<unsigned int>(pivot_idx), 1u);
                auto num_right = static_cast<unsigned int>(work.cols() - num_left);
                assert(num_left);
                assert(num_right);
                construct(work.block(0, 0, work.rows(), num_left), offset, node->left_child, Features::VectorRange<Features::IndexedFeatureValue>(features.first, features.first + num_left));
                construct(work.block(0, num_left, work.rows(), num_right), offset + num_left, node->right_child, Features::VectorRange<Features::IndexedFeatureValue>(features.first + num_left, features.second));
            }
        }
    }

    void BallTree::knn_search(Eigen::Ref<const Eigen::VectorXd> x, const unsigned int k, const Node* node, MaxDistancePriorityQueue& q) const
    {
        assert(node);

        const double dist_from_pivot = (x - data_.col(node->pivot_index)).norm();
        const double dist_from_queue = distance_from_queue(x, k, q);

        if (dist_from_pivot - node->radius >= dist_from_queue) {
            return;
        }

        if (!node->left_child && !node->right_child) {
            for (unsigned int i = node->start_index; i < node->end_index; ++i) {
                const double dist_from_i = i == node->pivot_index ? dist_from_pivot : (x - data_.col(i)).norm();
                if (dist_from_i < dist_from_queue) {
                    q.push(IndexedDistanceFromTarget(i, dist_from_i));
                    if (q.size() > k) {
                        q.pop();
                    }
                }
            }
        } else {
            Node* n1 = node->left_child.get();            
            Node* n2 = node->right_child.get();
            assert(n1);
            assert(n2);
            const double d1 = (x - data_.col(n1->pivot_index)).norm();
            const double d2 = (x - data_.col(n2->pivot_index)).norm();
            if (d2 < d1) {
                std::swap(n1, n2);
            }
            knn_search(x, k, n1, q);
            knn_search(x, k, n2, q);
        }
    }

    double BallTree::distance_from_queue(Eigen::Ref<const Eigen::VectorXd> x, const unsigned int k, const MaxDistancePriorityQueue& q) const
    {
        assert(q.size() <= k);
        if (k && q.size() == k) {
            return q.top().second;
        } else {
            return std::numeric_limits<double>::infinity();
        }
    }
}
