/* (C) 2020 Roman Werpachowski. */
#pragma once
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	/** @brief Statistical functions. */
	namespace Statistics
	{
		/** @brief Calculates the average and sum of squared error for a sample.

		Given a range `[begin, end)` with N values, calculates

		\f$ \mathrm{SSE} = \sum_{i=1}^{N} (x_i - \bar{x})^2 \f$

		and

		\f$ \bar{x} = N^{-1} \sum_{i=1}^{N} x_i \f$.
	
		@param[in] begin Iterator pointing to the beginning of the range of sample values.
		@param[in] end Iterator pointing one past to the end of the range of sample values.
		@tparam Iter Iterator type.
		@return (\f$\mathrm{SSE}\f$, \f$\bar{x}\f$) pair.
		*/
		template <class Iter> std::pair<double, double> sse_and_mean(const Iter begin, const Iter end)
		{
			double sse = 0;
			double mean;
			if (begin == end) {
				mean = std::numeric_limits<double>::quiet_NaN();
			} else {
				const auto sum = std::accumulate(begin, end, 0.);
				const auto n = static_cast<double>(end - begin);
				mean = sum / n;
				for (auto it = begin; it != end; ++it) {
					sse += std::pow(*it - mean, 2);
				}
			}
			return std::make_pair(sse, mean);
		}

		/** @brief Calculates the average and sum of squared error for a sample.

		Given a range `[begin, end)` with N values, calculates

		\f$ \mathrm{SSE} = \sum_{i=1}^{N} (x_i - \bar{x})^2 \f$,

		where

		\f$ \bar{x} = N^{-1} \sum_{i=1}^{N} x_i \f$.

		@param[in] begin Iterator pointing to the beginning of the range of sample values.
		@param[in] end Iterator pointing one past to the end of the range of sample values.
		@tparam Iter Iterator type.
		@return SSE.		
		*/
		template <class Iter> double sse(const Iter begin, const Iter end)
		{
			return sse_and_mean(begin, end).first;
		}		

		/** @brief Calculates the Gini index of the sample.
		
		Gini index is defined as 
		
		\f$ \sum_{k=1}^K \hat{p}_k (1 - \hat{p}_k) \f$

		where \f$\hat{p}_k\f$ is the frequency of occurrence of class `k` in data.

		Takes as argument a range `[begin, end)` of class values from 0 to `K - 1`.

		@param[in] begin Iterator pointing to the beginning of the range of sample values.
		@param[in] end Iterator pointing one past to the end of the range of sample values.
		@param[in] K Number of classes, positive.
		@tparam Iter Iterator type.
		@return Gini index and the most frequent class. If `begin == end`, `mode == K`.
		*/
		template <class Iter> std::pair<double, unsigned int> gini_index_and_mode(const Iter begin, const Iter end, const unsigned int K)
		{			
			std::vector<unsigned int> counts(K, 0);
			const auto N = static_cast<double>(std::distance(begin, end));
			for (auto it = begin; it != end; ++it) {
				++counts[static_cast<size_t>(*it)];
			}
			double gi = 0;
			unsigned int mode = K;
			unsigned int k = 0;
			unsigned max_count = 0;
			for (auto c : counts) {
				if (c > max_count) {
					mode = k;
					max_count = c;
				}
				const double p = static_cast<double>(c) / N;
				gi += p * (1 - p);
				++k;
			}
			assert(mode < K);
			return std::make_pair(gi, mode);
		}

		/** @brief Calculates the Gini index of the sample.
		
		Gini index is defined as 
		
		\f$ \sum_{k=1}^K \hat{p}_k (1 - \hat{p}_k) \f$

		where \f$\hat{p}_k\f$ is the frequency of occurrence of class `k` in data.

		Takes as argument a range `[begin, end)` of class values from 0 to `K - 1`.

		@param[in] begin Iterator pointing to the beginning of the range of sample values.
		@param[in] end Iterator pointing one past to the end of the range of sample values.
		@param[in] K Number of classes, positive.
		@tparam Iter Iterator type.
		@return Gini index.
		*/
		template <class Iter> double gini_index(const Iter begin, const Iter end, const unsigned int K)
		{
			std::vector<unsigned int> counts(K, 0);
			const auto N = static_cast<double>(std::distance(begin, end));
			for (auto it = begin; it != end; ++it) {
				++counts[static_cast<size_t>(*it)];
			}
			double gi = 0;
			for (auto c : counts) {
				const double p = static_cast<double>(c) / N;
				gi += p * (1 - p);
			}
			return gi;
		}

		/** @brief Calculates the mode (most frequent value) of a sample.

		The sample is assumed to contain values in the `[0, K - 1]` range.
		
		@param[in] begin Iterator pointing to the beginning of the range of sample values.
		@param[in] end Iterator pointing one past to the end of the range of sample values.
		@param[in] K Positive number of distinct values.
		@tparam Iter Iterator type.
		@return Mode of the sample.
		*/
		template <class Iter> unsigned int mode(const Iter begin, const Iter end, const unsigned int K)
		{
			std::vector<unsigned int> counts(K, 0);
			for (auto it = begin; it != end; ++it) {
				++counts[static_cast<size_t>(*it)];
			}
			unsigned int mode = K;
			unsigned int k = 0;
			unsigned max_count = 0;
			for (auto c : counts) {
				if (c > max_count) {
					mode = k;
					max_count = c;
				}
				++k;
			}
			assert(mode < K);
			return mode;
		}

		/** @brief Calculates sample covariance of two vectors. 
		
		@param[in] xs X values.
		@param[in] ys Y values.
		@tparam R Scalar value type.

		@return Sample covariance (unbiased estimate of population covariance) or NaN if `xs.size() < 2`.

		@throw std::invalid_argument If `xs.size() != ys.size()`.		
		*/
		template <class R> R covariance(const std::vector<R>& xs, const std::vector<R>& ys)
		{			
			if (xs.size() != ys.size()) {
				throw std::invalid_argument("Length mismatch");
			}			
			const auto n = static_cast<R>(xs.size());
			if (n < 2) {
				return std::numeric_limits<double>::quiet_NaN();
			}
			const R sum_x = std::accumulate(xs.begin(), xs.end(), R(0));
			const R sum_y = std::accumulate(ys.begin(), ys.end(), R(0));
			const auto mean_x = sum_x / n;
			const auto mean_y = sum_y / n;
			auto it_x = xs.begin();			
			R sum_xy(0);
			for (auto it_y = ys.begin(); it_y != ys.end(); ++it_x, ++it_y) {
				sum_xy += (*it_x - mean_x) * (*it_y - mean_y);
			}
			return sum_xy / (n - 1);
		}

		/** @brief Calculates sample covariance of two vectors. 

		@param xs X values.
		@param ys Y values.

		@return Sample covariance (unbiased estimate of population covariance) or NaN if `xs.size() < 2`.

		@throw std::invalid_argument If `xs.size() != ys.size()`.		
		*/
		DLL_DECLSPEC double covariance(Eigen::Ref<const Eigen::VectorXd> xs, Eigen::Ref<const Eigen::VectorXd> ys);
	}	
}
