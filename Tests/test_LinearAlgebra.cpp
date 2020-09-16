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

static void test_xAx_symmetric(const unsigned int n)
{
	const Eigen::MatrixXd A0 = Eigen::MatrixXd::Random(n, n);
	const Eigen::MatrixXd A = (A0 + A0.transpose()) / 2;
	const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
	const double actual = xAx_symmetric(A, x);
	const double expected = x.transpose() * A * x;
	ASSERT_NEAR(expected, actual, std::abs(expected) * 1e-14);
}

TEST(LinearAlgebraTest, xAx_symmetric_1024)
{
	test_xAx_symmetric(1024);
}

TEST(LinearAlgebraTest, xAx_symmetric_4)
{
	test_xAx_symmetric(4);
}

static void test_xxT(const unsigned int n)
{
	Eigen::MatrixXd actual;
	const Eigen::VectorXd x = Eigen::VectorXd::Random(n);
	xxT(x, actual);
	const Eigen::MatrixXd expected = x * x.transpose();
	ASSERT_NEAR(0, (actual - expected).norm(), expected.norm() * 1e-15);
}

TEST(LinearAlgebraTest, xxT_1024)
{
	test_xxT(1024);
}

TEST(LinearAlgebraTest, xxT_4)
{
	test_xxT(4);
}