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

        /**
         * @brief Swaps two columns in feature matrix.
         * 
         * @param X Feature matrix.
         * @param i1 Index of the 1st column.
         * @param i2 Index of the 2nd column.
         * @throw std::out_of_range If `i1 >= X.cols()` or `i2 >= X.cols()`.
        */
        DLL_DECLSPEC void swap_columns(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Index i1, Eigen::Index i2);

        /**
         * @brief Partitions features (in columns) so that those with x[k] < pivot[k] are before the pivot, and those with x[k] > pivot[k] are after it.
         * 
         * pivot = X.col(pivot_idx).
         * 
         * @param X Features with data points in columns.
         * @param pivot_idx Pivot index.
         * @param k Dimension used for comparison.
         * 
         * @return Position of the pivot feature after partitioning.
         * 
         * @throw std::out_of_range If `pivot_idx >= X.cols()` or `k >= X.rows()`.
        */
        DLL_DECLSPEC Eigen::Index partition(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Index pivot_idx, Eigen::Index k);
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