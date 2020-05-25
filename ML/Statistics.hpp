#pragma once
#include <numeric>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	/** All sorts of statistical functions. */
	namespace Statistics
	{
		/** Calculates sum_i (x_i - average(x))^2 for given range.
		@tparam Iter Iterator type.
		*/
		template <typename Iter> double sse(const Iter begin, const Iter end)
		{
			if (begin == end) {
				return 0;
			} else {
				const auto sum = std::accumulate(begin, end, 0.);
				const auto n = static_cast<double>(end - begin);
				const auto mean = sum / n;
				double sse = 0;
				for (auto it = begin; it != end; ++it) {
					sse += std::pow(*it - mean, 2);
				}
				return sse;
			}
		}

		/** Calculates the Gini index sum_{k=1}^K \hat{p}_k (1 - \hat{p}_k)
		for \hat{p}_k being the frequency of occurrence of class k in data.
		Takes as argument a range [begin, end) of class values from 0 to K - 1.
		*/
		template <typename Iter> double gini_index(const Iter begin, const Iter end, const unsigned int K)
		{
			std::vector<double> counts(K, 0);
			const auto N = static_cast<double>(std::distance(begin, end));
			for (auto it = begin; it != end; ++it) {
				++counts[static_cast<size_t>(*it)];
			}
			double gi = 0;
			for (auto x : counts) {
				const double p = x / N;
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
