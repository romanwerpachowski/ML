/* (C) 2020 Roman Werpachowski. */
#include <gtest/gtest.h>
#include <Eigen/Core>


TEST(EigenTest, matrix_ref_from_vector)
{
	Eigen::VectorXd x(10);
	Eigen::Ref<const Eigen::MatrixXd> map(x.transpose());
	ASSERT_EQ(x.data(), map.data());
	ASSERT_EQ(1, map.rows());
	ASSERT_EQ(x.size(), map.cols());
	Eigen::Vector3d x3(0, 1, 2);
	Eigen::Ref<const Eigen::MatrixXd> map3(x3.transpose());
	ASSERT_EQ(x3.data(), map3.data());
	ASSERT_EQ(1, map3.rows());
	ASSERT_EQ(x3.size(), map3.cols());
}