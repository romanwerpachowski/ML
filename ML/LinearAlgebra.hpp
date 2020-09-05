/* (C) 2020 Roman Werpachowski. */
#pragma once
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
	/** @brief Linear algebra helper functions.
	*/
	namespace LinearAlgebra
	{
		/** #brief Calculates x^T A x for a symmetric matrix A.
		@param A Symmetric square matrix.
		@param x Vector.
		@return Value of \f$ \vec{x}^T A \vec{x} \f$.
		@throw std::invalid_argument If `A` is not square or `x.size() != A.rows()`.
		*/
		DLL_DECLSPEC double xAx_symmetric(const Eigen::MatrixXd& A, const Eigen::VectorXd& x);
	}
}
