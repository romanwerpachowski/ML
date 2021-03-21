/** (C) 2021 Roman Werpachowski */
#include <cassert>
#include <iostream>
#include <limits>
#include "BallTree.hpp"

namespace ml
{
    BallTree::BallTree(Eigen::Ref<const Eigen::MatrixXd> X)
        : tree_(X.rows(), X.cols()), radii_(X.cols()), num_left_children_(X.cols()), num_right_children_(X.cols())
    {
        Eigen::MatrixXd work(X);
        std::vector<Features::IndexedFeatureValue> features(X.cols());
        if (X.cols()) {
            construct(work, 0, Features::from_vector(features));
        }
    }

    unsigned int BallTree::left_child_index(unsigned int i) const
    {
        if (i >= size()) {
            throw std::out_of_range("BallTree: node index out of range");
        }
        if (num_left_children_[i]) {
            return i + 1;
        } else {
            return size();
        }
    }

    unsigned int BallTree::right_child_index(unsigned int i) const
    {
        if (i >= size()) {
            throw std::out_of_range("BallTree: node index out of range");
        }
        if (num_right_children_[i]) {
            return i + 1 + num_left_children_[i];
        } else {
            return size();
        }
    }

    void BallTree::find_k_nearest_neighbours(Eigen::Ref<const Eigen::VectorXd> x, const unsigned int k, std::vector<unsigned int>& nn) const
    {
        if (x.size() != tree_.rows()) {
            throw std::invalid_argument("BallTree: wrong feature vector size");
        }
        MaxDistancePriorityQueue q;
        knn_search(x, k, 0, q);
        nn.reserve(q.size());
        nn.clear();
        while (!q.empty()) {
            nn.push_back(q.top().first);
            q.pop();
        }
    }

    unsigned int BallTree::find_k_nearest_neighbours(Eigen::Ref<const Eigen::VectorXd> x, const unsigned int k, Eigen::Ref<Eigen::MatrixXd> nn) const
    {
        if (x.size() != tree_.rows()) {
            throw std::invalid_argument("BallTree: wrong feature vector size");
        }
        const auto num_neighbours = std::min(k, static_cast<unsigned int>(tree_.cols()));
        if (static_cast<unsigned int>(nn.cols()) < num_neighbours) {
            throw std::invalid_argument("BallTree: not enough room for all neighbours");
        }
        MaxDistancePriorityQueue q;
        knn_search(x, k, 0, q);
        assert(q.size() == static_cast<size_t>(num_neighbours));
        Eigen::Index i = 0;
        while (!q.empty()) {
            nn.col(i) = tree_.col(q.top().first);
            q.pop();
            ++i;
        }
        return num_neighbours;
    }

    void BallTree::construct(Eigen::Ref<Eigen::MatrixXd> work, const unsigned int i, Features::VectorRange<Features::IndexedFeatureValue> features)
    {
        assert(i < tree_.cols());
        if (work.cols() == 0) {
            return;
        } else if (work.cols() == 1) {            
            tree_.col(i) = work.col(0);
            radii_[i] = 0;
            num_left_children_[i] = 0;
            num_right_children_[i] = 0;
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
            Eigen::Index pivot_idx = pivot_iter->first;
            const auto root = work.col(pivot_idx);
            tree_.col(i) = root;
            double max_dist = 0;
            for (Eigen::Index j = 0; j < work.cols(); ++j) {
                const double dist = (root - work.col(j)).norm();
                if (dist > max_dist) {
                    max_dist = dist;
                }
            }
            radii_[i] = max_dist;
            // Partition work space into Left and Right child features.
            pivot_idx = Features::partition(work, pivot_idx, r);
            assert(pivot_idx < work.cols());
            assert((tree_.col(i) - work.col(pivot_idx)).norm() == 0);
            const auto num_left = static_cast<unsigned int>(pivot_idx);
            const auto num_right = static_cast<unsigned int>(work.cols() - (pivot_idx + 1));
            num_left_children_[i] = num_left;
            num_right_children_[i] = num_right;
            if (num_left) {
                construct(work.block(0, 0, work.rows(), pivot_idx), left_child_index(i), Features::VectorRange<Features::IndexedFeatureValue>(features.first, features.first + pivot_idx));
            }
            if (num_right) {
                construct(work.block(0, pivot_idx + 1, work.rows(), num_right), right_child_index(i), Features::VectorRange<Features::IndexedFeatureValue>(features.first + pivot_idx + 1, features.second));
            }
        }
    }

    void BallTree::knn_search(Eigen::Ref<const Eigen::VectorXd> x, const unsigned int k, const unsigned int i, MaxDistancePriorityQueue& q) const
    {
        assert(i < size());
        const double dist_from_i = (x - tree_.col(i)).norm();
        const double dist_from_q = distance_from_queue(x, k, q);
        if (dist_from_i - radii_[i] >= dist_from_q) {
            return;
        }
        if (dist_from_i < dist_from_q) {
            q.push(IndexedDistanceFromTarget(i, dist_from_i));
            if (q.size() > k) {
                q.pop();
            }
        }        
        unsigned int i1 = left_child_index(i);
        unsigned int i2 = right_child_index(i);
        double d1 = std::numeric_limits<double>::infinity();
        double d2 = std::numeric_limits<double>::infinity();
        if (i1 < size()) {
            d1 = (x - tree_.col(i1)).norm();
        }
        if (i2 < size()) {
            d2 = (x - tree_.col(i2)).norm();
        }
        if (d2 < d1) {
            std::swap(i1, i2);
        }
        if (i1 < size()) {
            knn_search(x, k, i1, q);
        }
        if (i2 < size()) {
            knn_search(x, k, i2, q);
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
