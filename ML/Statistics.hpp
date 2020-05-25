#pragma once
#include <limits>
#include <numeric>
#include <utility>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	/** All sorts of statistical functions. */
	namespace Statistics
	{
		/** Calculates mean(x) and sum_i (x_i - mean(x))^2 for given range.
		@return Pair of (SSE, mean).
		@tparam Iter Iterator type.
		*/
		template <typename Iter> std::pair<double, double> sse_and_mean(const Iter begin, const Iter end)
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

		/** Calculates sum_i (x_i - mean(x))^2 for given range.
		@tparam Iter Iterator type.
		*/
		template <typename Iter> double sse(const Iter begin, const Iter end)
		{
			return sse_and_mean(begin, end).first;
		}		

		/** Calculates the Gini index sum_{k=1}^K \hat{p}_k (1 - \hat{p}_k)
		for \hat{p}_k being the frequency of occurrence of class k in data.
		Takes as argument a range [begin, end) of class values from 0 to K - 1.
		@param K Number of classes, positive.
		@return Gini index and the most frequent class. If begin == end, mode == K.
		*/
		template <typename Iter> std::pair<double, unsigned int> gini_index_and_mode(const Iter begin, const Iter end, const unsigned int K)
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

		/** Calculates the Gini index sum_{k=1}^K \hat{p}_k (1 - \hat{p}_k)
		for \hat{p}_k being the frequency of occurrence of class k in data.
		Takes as argument a range [begin, end) of class values from 0 to K - 1.
		K value is detected from the data.
		@return Gini index and the most frequent class. If begin == end, mode == K.
		*/
		template <typename Iter> std::pair<double, unsigned int> gini_index_and_mode(const Iter begin, const Iter end)
		{
			const auto K = static_cast<unsigned int>(*std::max_element(begin, end)) + 1;
			return gini_index_and_mode(begin, end, K);
		}

		/** Calculates the Gini index sum_{k=1}^K \hat{p}_k (1 - \hat{p}_k)
		for \hat{p}_k being the frequency of occurrence of class k in data.
		Takes as argument a range [begin, end) of class values from 0 to K - 1.
		@param K Number of classes, positive.
		@return Gini index.
		*/
		template <typename Iter> double gini_index(const Iter begin, const Iter end, const unsigned int K)
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

		/** Calculates the Gini index sum_{k=1}^K \hat{p}_k (1 - \hat{p}_k)
		for \hat{p}_k being the frequency of occurrence of class k in data.
		Takes as argument a range [begin, end) of class values from 0 to K - 1.
		K value is detected from the data.
		*/
		template <typename Iter> double gini_index(const Iter begin, const Iter end)
		{
			const auto K = static_cast<unsigned int>(*std::max_element(begin, end)) + 1;
			return gini_index(begin, end, K);
		}
	}	
}
