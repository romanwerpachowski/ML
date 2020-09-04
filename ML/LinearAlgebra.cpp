#include <cassert>
#include "LinearAlgebra.hpp"

namespace ml
{
	namespace LinearAlgebra
	{
		double xAx_symmetric(const Eigen::Ref<const Eigen::MatrixXd> A, const Eigen::Ref<const Eigen::VectorXd> x)
		{
			if (A.rows() != A.cols()) {
				throw std::invalid_argument("A matrix is not square");
			}
			if (x.size() != A.rows()) {
				throw std::invalid_argument("x has wrong size");
			}
			double sum = 0;
			const auto dim = A.rows();
			for (Eigen::Index i = 0; i < dim; ++i) {
				const auto x_i = x[i];
				sum += A(i, i) * x_i * x_i;
				for (Eigen::Index j = 0; j < i; ++j) {
					sum += 2 * A(j, i) * x_i * x[j];
				}
			}
			return sum;
		}
	}
}