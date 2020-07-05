#include <cmath>
#include <stdexcept>
#include "LinearRegression.hpp"

namespace ml
{
	namespace LinearRegression
	{
		UnivariateLinearRegressionResult univariate(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> y)
		{
			const auto n = x.size();
			if (n != y.size()) {
				throw std::invalid_argument("X and Y vectors have different sizes");
			}
			if (n < 2) {
				throw std::invalid_argument("Need at least 2 points for regresssion");
			}
			const auto mx = x.mean();
			const auto my = y.mean();
			const auto x_centred = x.array() - mx;
			const auto y_centred = y.array() - my;
			const auto sxy = (x_centred * y_centred).sum();
			const auto sxx = (x_centred * x_centred).sum();
			const auto syy = (y_centred * y_centred).sum();
			UnivariateLinearRegressionResult result;
			result.slope = sxy / sxx;
			result.intercept = my - result.slope * mx;
			double shatyy = 0;
			for (Eigen::Index i = 0; i < n; ++i) {
				shatyy += std::pow(result.intercept + x[i] * result.slope - my, 2);
			}
			result.r2 = shatyy / syy;
			return result;
		}
	}
}