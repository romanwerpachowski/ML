#pragma once
/* (C) 2021 Roman Werpachowski. */
#include <iosfwd>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
    /**
     * @brief Utilities and types for working with features.
    */
    namespace Features
    {
        typedef std::pair<Eigen::Index, double> IndexedFeatureValue; /**< @brief Used to sort feature vectors. */

        /**
         * @brief Compares indexed features for sorting in ascending order.
        */
        static const auto INDEXED_FEATURE_COMPARATOR_ASCENDING = [](const IndexedFeatureValue& a, const IndexedFeatureValue& b) { return a.second < b.second; };

        /**
         * @brief Compares indexed features for sorting in descending order.
        */
        static const auto INDEXED_FEATURE_COMPARATOR_DESCENDING = [](const IndexedFeatureValue& a, const IndexedFeatureValue& b) { return a.second > b.second; };

        /** @brief Pair of vector iterators. */
        template <typename T> using VectorRange = std::pair<typename std::vector<T>::iterator, typename std::vector<T>::iterator>;

        /** Creates an iterator pair containing begin() and end(). */
        template <typename T> VectorRange<T> from_vector(std::vector<T>& v)
        {
            return std::make_pair(v.begin(), v.end());
        }

        /**
         * @brief Copies the n-th coordinate to `features`.
         * @param[in] X Features matrix, with data points in columns.
         * @param[in] n Coordinate index.
         * @param[out] features Iterator range for indexed feature values.
         * @throw std::invalid_argument If size of the iterator range is different from the number of data points, or `n` is too large.
        */
        DLL_DECLSPEC void set_to_nth(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Index n, VectorRange<IndexedFeatureValue> features);

        
    }
}

namespace std
{
    /**
    * @brief Stream output operator.
    * 
    * Overloaded in std namespace, because VS 2019 doesn't pick it up when it's overloaded in ml::Features.
    */
    DLL_DECLSPEC std::ostream& operator<<(std::ostream& os, const ml::Features::IndexedFeatureValue& fv);
}