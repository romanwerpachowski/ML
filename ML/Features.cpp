/* (C) 2021 Roman Werpachowski. */
#include "Features.hpp"
#include <ostream>


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