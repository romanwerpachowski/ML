/* (C) 2020 Roman Werpachowski. */
#include <gtest/gtest.h>
#include "ML/LinearAlgebra.hpp"


using namespace ml::LinearAlgebra;

TEST(LinearAlgebraTest, xAx_symmetric_errors)
{
	Eigen::MatrixXd A(2, 3);
	Eigen::VectorXd x(2);
	ASSERT_THROW(xAx_symmetric(A, x), std::invalid_argument);
	A.resize(3, 3);
	ASSERT_THROW(xAx_symmetric(A, x), std::invalid_argument);
}

TEST(LinearAlgebraTest, xAx_symmetric)
{
	constexpr auto n = 120;
	const Eigen::MatrixXd A0 = Eigen::MatrixXd::Random(n, n);
	const Eigen::MatrixXd A = (A0 + A0.transpose()) / 2;
	const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
	const double actual = xAx_symmetric(A, x);
	const double expected = x.transpose() * A * x;
	ASSERT_NEAR(expected, actual, 1e-13);
}