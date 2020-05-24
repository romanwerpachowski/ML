#pragma once
#include <numeric>
#include "dll.hpp"

namespace ml 
{
	/** All sorts of statistical functions. */
	namespace Statistics
	{
		/** Calculates sum_i (x_i - average(x))^2 for given range.
		@tparam Iter Iterator type.
		*/
		template <typename Iter> double calc_sse(const Iter begin, const Iter end)
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
	}	
}
