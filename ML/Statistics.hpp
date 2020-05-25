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
		template <typename Iter> double gini_index(const Iter begin, const Iter end)
		{
			const auto K = static_cast<unsigned int>(*std::max_element(begin, end));
			return gini_index(begin, end, K);
		}
	}	
}
