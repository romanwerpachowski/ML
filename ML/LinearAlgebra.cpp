#include <cassert>
#include "LinearAlgebra.hpp"

namespace ml
{
	namespace LinearAlgebra
	{
		double xAx_symmetric(const Eigen::MatrixXd& A, const Eigen::VectorXd& x)
		{
			if (A.rows() != A.cols()) {
				throw std::invalid_argument("A matrix is not square");
			}
			if (x.size() != A.rows()) {
				throw std::invalid_argument("x has wrong size");
			}
			const auto dim = A.rows();
			if (dim < 15) {
				double sum = 0;
				for (Eigen::Index i = 0; i < dim; ++i) {
					const auto x_i = x[i];
					sum += A(i, i) * x_i * x_i;
					for (Eigen::Index j = 0; j < i; ++j) {
						sum += 2 * A(j, i) * x_i * x[j];
					}
				}
				return sum;
			}
			else {
				return x.transpose() * A.selfadjointView<Eigen::Upper>() * x;
			}
		}

		void xxT(const Eigen::VectorXd& x, Eigen::MatrixXd& dest)
		{
			const auto dim = x.size();
			if (dim < 16) {
				if (dest.rows() != dim || dest.cols() != dim) {
					dest.resize(dim, dim);
				}
				for (Eigen::Index i = 0; i < x.size(); ++i) {
					const auto x_i = x[i];
					dest(i, i) = x_i * x_i;
					for (Eigen::Index j = 0; j < i; ++j) {
						const auto x_i_x_j = x_i * x[j];
						dest(i, j) = x_i_x_j;
						dest(j, i) = x_i_x_j;
					}
				}
			} else {
				dest = x * x.transpose();
			}
		}
	}
}