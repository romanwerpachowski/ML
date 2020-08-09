#pragma once
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <Eigen/Core>

namespace ml 
{
	/** Statistical functions. */
	namespace Statistics
	{
		/** Calculates mean(x) and sum_i (x_i - mean(x))^2 for given range.
		@return Pair of (SSE, mean).
		@tparam Iter Iterator type.
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

		/** Calculates sum_i (x_i - mean(x))^2 for given range.
		@tparam Iter Iterator type.
		*/
		template <class Iter> double sse(const Iter begin, const Iter end)
		{
			return sse_and_mean(begin, end).first;
		}		

		/** Calculates the Gini index sum_{k=1}^K \hat{p}_k (1 - \hat{p}_k)
		for \hat{p}_k being the frequency of occurrence of class k in data.
		Takes as argument a range [begin, end) of class values from 0 to K - 1.
		@param K Number of classes, positive.
		@return Gini index and the most frequent class. If begin == end, mode == K.
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

		/** Calculates the Gini index sum_{k=1}^K \hat{p}_k (1 - \hat{p}_k)
		for \hat{p}_k being the frequency of occurrence of class k in data.
		Takes as argument a range [begin, end) of class values from 0 to K - 1.
		@param K Number of classes, positive.
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

		/** Calculates the mode (most frequent value) of a sample containing values from 0 to K - 1. */
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

		/**! Calculates sample covariance of two vectors. */
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
	}	
}
