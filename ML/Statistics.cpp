#include "Statistics.hpp"

namespace ml
{
	namespace Statistics
	{
		double covariance(const Eigen::Ref<const Eigen::VectorXd> xs, const Eigen::Ref<const Eigen::VectorXd> ys)
		{
			if (xs.size() != ys.size()) {
				throw std::invalid_argument("Length mismatch");
			}
			const auto n = static_cast<double>(xs.size());
			if (n < 2) {
				return std::numeric_limits<double>::quiet_NaN();
			}
			const auto mean_x = xs.mean();
			const auto mean_y = ys.mean();
			const auto sum_xy = ((xs.array() - mean_x) * (ys.array() - mean_y)).sum();
			return sum_xy / (n - 1);
		}
	}
}