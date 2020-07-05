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
			result.correlation = sxy / std::sqrt(sxx * syy);
			result.r2 = sxy * sxy / (sxx * syy);
			return result;
		}

		UnivariateLinearRegressionResult univariate(const double x0, const double dx, Eigen::Ref<const Eigen::VectorXd> y)
		{
			const auto n = static_cast<double>(y.size());
			if (n < 2) {
				throw std::invalid_argument("Need at least 2 points for regresssion");
			}
			const auto mx = x0 + (n - 1) * dx / 2;
			const auto my = y.mean();
			const auto y_centred = y.array() - my;
			const auto indices = Eigen::VectorXd::LinSpaced(y.size(), 0, n - 1);
			const auto sxy = dx * (y_centred * indices.array()).sum();
			const auto sxx = dx * dx * n * (n * n - 1) / 12;
			const auto syy = (y_centred * y_centred).sum();
			UnivariateLinearRegressionResult result;
			result.slope = sxy / sxx;
			result.intercept = my - result.slope * mx;
			result.correlation = sxy / std::sqrt(sxx * syy);
			result.r2 = result.correlation * result.correlation;
			return result;
		}
	}
}