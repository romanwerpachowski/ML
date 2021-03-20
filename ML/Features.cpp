/* (C) 2021 Roman Werpachowski. */
#include <algorithm>
#include <ostream>
#include "Features.hpp"


namespace ml
{
    namespace Features
    {
        void set_to_nth(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Index n, VectorRange<IndexedFeatureValue> features)
        {
            const auto sample_size = X.cols();
            if (n > X.rows()) {
                throw std::invalid_argument("Features: coordinate index too large");
            }
            if (static_cast<ptrdiff_t>(sample_size) != std::distance(features.first, features.second))
            {
                throw std::invalid_argument("Features: wrong iterator range size");
            }
            const auto X_f = X.row(n);
            auto features_it = features.first;
            for (Eigen::Index i = 0; i < sample_size; ++i, ++features_it) {
                *features_it = std::make_pair(i, X_f[i]);
            }
            assert(features_it == features.second);
        }

        void swap_columns(Eigen::Ref<Eigen::MatrixXd> X, const Eigen::Index i1, const Eigen::Index i2)
        {
            if (i1 >= X.cols()) {
                throw std::out_of_range("Features: index of the 1st swapped column out of range");
            }
            if (i2 >= X.cols()) {
                throw std::out_of_range("Features: index of the 2nd swapped column out of range");
            }
            auto col1 = X.col(i1);
            auto col2 = X.col(i2);
            for (Eigen::Index r = 0; r < X.rows(); ++r) {
                std::swap(col1[r], col2[r]);
            }
        }

        Eigen::Index partition(Eigen::Ref<Eigen::MatrixXd> X, const Eigen::Index pivot_idx, const Eigen::Index k)
        {
            if (pivot_idx >= X.cols()) {
                throw std::out_of_range("Features: pivot column index out of range");
            }
            if (k >= X.rows()) {
                throw std::out_of_range("Features: pivoting dimension index out of range");
            }            
            // Use https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
            auto p = (X.cols() - 1) / 2; // Should round down automatically.
            const auto A = X.row(k);
            const double pivot = A[pivot_idx];
            if (p != pivot_idx) {
                swap_columns(X, p, pivot_idx);
            }
            assert(pivot == A[p]);
            Eigen::Index i = -1;
            Eigen::Index j = X.cols();
            while (true) {
                do {
                    ++i;
                } while (A[i] < pivot);
                do {
                    --j;
                } while (A[j] > pivot);
                if (i >= j) {
                    return p;
                }
                // Track pivot location.
                if (p == i) {
                    p = j;
                } else if (p == j) {
                    p = i;
                }
                swap_columns(X, i, j);
            }
        }
    }    
}

namespace std
{
    std::ostream& operator<<(std::ostream& os, const ml::Features::IndexedFeatureValue& fv)
    {
        os << "(" << fv.first << ": " << fv.second << ")";
        return os;
    }
}