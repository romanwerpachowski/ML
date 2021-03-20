/** (C) 2021 Roman Werpachowski */
#include <cassert>
#include "BallTree.hpp"
#include "Features.hpp"

namespace ml
{
    static void construct(Eigen::Ref<Eigen::MatrixXd> work, unsigned int i, Eigen::Ref<Eigen::MatrixXd> tree, Features::VectorRange<Features::IndexedFeatureValue> features);

    BallTree::BallTree(Eigen::Ref<const Eigen::MatrixXd> X)
        : tree_(X.rows(), X.cols())
    {
        Eigen::MatrixXd work(X);
        std::vector<Features::IndexedFeatureValue> features(X.cols());
        construct(work, 0, tree_, Features::from_vector(features));
    }    

    void construct(Eigen::Ref<Eigen::MatrixXd> work, const unsigned int i, Eigen::Ref<Eigen::MatrixXd> tree, Features::VectorRange<Features::IndexedFeatureValue> features)
    {
        if (work.cols() == 1) {
            tree.col(i) = work.col(0);
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
            const Eigen::Index sorted_pivot_idx = work.cols() / 2;
            const auto pivot_iter = features.first + sorted_pivot_idx;
            const Eigen::Index pivot_feature_idx = pivot_iter->first;
            tree.col(i) = work.col(pivot_feature_idx);
            // TODO: finish.
        }
    }
}