#include <stdexcept>
#include <gtest/gtest.h>
#include "ML/Crossvalidation.hpp"

using namespace ml;

TEST(CrossvalidationTest, calc_fold_indices)
{
	size_t i0, i1;
	Crossvalidation::calc_fold_indices(10, 0, 5, i0, i1);
	ASSERT_EQ(0, i0);
	ASSERT_EQ(2, i1);
	Crossvalidation::calc_fold_indices(10, 1, 3, i0, i1);
	ASSERT_EQ(3, i0);
	ASSERT_EQ(6, i1);
	Crossvalidation::calc_fold_indices(10, 2, 3, i0, i1);
	ASSERT_EQ(6, i0);
	ASSERT_EQ(10, i1);
}

TEST(CrossvalidationTest, calc_fold_indices_throws)
{
	size_t i0, i1;
	ASSERT_THROW(Crossvalidation::calc_fold_indices(100, 5, 5, i0, i1), std::invalid_argument);
	ASSERT_THROW(Crossvalidation::calc_fold_indices(8, 0, 10, i0, i1), std::invalid_argument);
}

TEST(CrossvalidationTest, only_kth_fold_1d_stl)
{
	const std::vector<int> data({ 0, 1, 2, 3, 4, 5, 6 });
	ASSERT_EQ(std::vector<int>({ 0, 1 }), Crossvalidation::only_kth_fold_1d(data, 0, 3));
	ASSERT_EQ(std::vector<int>({ 2, 3 }), Crossvalidation::only_kth_fold_1d(data, 1, 3));
	ASSERT_EQ(std::vector<int>({ 4, 5, 6 }), Crossvalidation::only_kth_fold_1d(data, 2, 3));
}

TEST(CrossvalidationTest, only_kth_fold_1d_eigen)
{
	Eigen::VectorXd data(7);
	data << 0, 1, 2, 3, 4, 5, 6;
	ASSERT_EQ(Eigen::Vector2d(0, 1), Crossvalidation::only_kth_fold_1d(data, 0, 3));
	ASSERT_EQ(Eigen::Vector2d(2, 3 ), Crossvalidation::only_kth_fold_1d(data, 1, 3));
	ASSERT_EQ(Eigen::Vector3d(4, 5, 6 ), Crossvalidation::only_kth_fold_1d(data, 2, 3));
}

TEST(CrossvalidationTest, without_kth_fold_1d_stl)
{
	const std::vector<int> data({ 0, 1, 2, 3, 4, 5, 6 });
	ASSERT_EQ(std::vector<int>({ 2, 3, 4, 5, 6 }), Crossvalidation::without_kth_fold_1d(data, 0, 3));
	ASSERT_EQ(std::vector<int>({ 0, 1, 4, 5, 6 }), Crossvalidation::without_kth_fold_1d(data, 1, 3));
	ASSERT_EQ(std::vector<int>({ 0, 1, 2, 3 }), Crossvalidation::without_kth_fold_1d(data, 2, 3));
}

static void assert_vectors_equal(const std::vector<double>& left, Eigen::Ref<const Eigen::VectorXd> right)
{
	const Eigen::Map<const Eigen::VectorXd> l(&left[0], left.size());
	ASSERT_EQ(l, right);
}

TEST(CrossvalidationTest, without_kth_fold_1d_eigen)
{
	Eigen::VectorXd data(7);
	data << 0, 1, 2, 3, 4, 5, 6;
	assert_vectors_equal(std::vector<double>({ 2, 3, 4, 5, 6 }), Crossvalidation::without_kth_fold_1d(data, 0, 3));
	assert_vectors_equal(std::vector<double>({ 0, 1, 4, 5, 6 }), Crossvalidation::without_kth_fold_1d(data, 1, 3));
	assert_vectors_equal(std::vector<double>({ 0, 1, 2, 3 }), Crossvalidation::without_kth_fold_1d(data, 2, 3));
}