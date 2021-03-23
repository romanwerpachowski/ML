/* (C) 2021 Roman Werpachowski. */
#include <gtest/gtest.h>
#include <Eigen/Core>
#include "ML/Features.hpp"


TEST(FeaturesTest, from_vector)
{
	std::vector<double> v(4);
	const auto iter_range = ml::Features::from_vector(v);
	ASSERT_EQ(v.begin(), iter_range.first);
	ASSERT_EQ(v.end(), iter_range.second);
}

TEST(FeaturesTest, set_to_nth)
{
	std::vector<ml::Features::IndexedFeatureValue> features(4);
	Eigen::MatrixXd X(2, 4);
	X << -0.1, 0.2, 0.2, 0.3,
		1, 2, 2, -1;
	ml::Features::set_to_nth(X, 1, ml::Features::from_vector(features));
	for (Eigen::Index i = 0; i < 4; ++i) {
		ASSERT_EQ(X(1, i), features[i].second) << features[i];
	}
}

TEST(FeaturesTest, swap_columns)
{
	Eigen::MatrixXd X(2, 4);
	X << -0.1, 0.2, 0.2, 0.3,
		1, 2, 2, -1;
	ml::Features::swap_columns(X, 0, 1);
	Eigen::MatrixXd expected(2, 4);
	expected << 0.2, -0.1, 0.2, 0.3,
		2, 1, 2, -1;
	ASSERT_NEAR(0, (expected - X).norm(), 1e-15) << X;
	ml::Features::swap_columns(X.block(0, 2, 2, 2), 0, 1);
	expected << 0.2, -0.1, 0.3, 0.2,
		2, 1, -1, 2;
	ASSERT_NEAR(0, (expected - X).norm(), 1e-15) << X;
}

TEST(FeaturesTest, partition)
{
	Eigen::MatrixXd X(2, 4);
	X << 0.3, 0.21, -0.3, 0.2,
		1, 2, -1, 3;
	auto pivot_idx = ml::Features::partition(X, 1, 0);
	Eigen::MatrixXd expected(2, 4);
	expected << 0.2, -0.3, 0.21, 0.3,
		3, -1, 2, 1;
	ASSERT_NEAR(0, (expected - X).norm(), 1e-15) << X;
	ASSERT_EQ(2, pivot_idx);
	pivot_idx = ml::Features::partition(X.block(0, 0, 2, 1), 0, 1);
	ASSERT_NEAR(0, (expected - X).norm(), 1e-15) << X;
	ASSERT_EQ(0, pivot_idx);
	pivot_idx = ml::Features::partition(X.block(0, 1, 2, 3), 1, 0);	
	ASSERT_NEAR(0, (expected - X).norm(), 1e-15) << X;
	ASSERT_EQ(1, pivot_idx);
	pivot_idx = ml::Features::partition(X.block(0, 1, 2, 3), 1, 1);
	expected << 0.2, -0.3, 0.3, 0.21,
		3, -1, 1, 2;
	ASSERT_NEAR(0, (expected - X).norm(), 1e-15) << X;
	ASSERT_EQ(2, pivot_idx);
}

TEST(FeaturesTest, partition_with_labels)
{
	Eigen::MatrixXd X(2, 4);
	X << 0.3, 0.21, -0.3, 0.2,
		1, 2, -1, 3;
	Eigen::VectorXd y(4);
	y << 10, 20, 30, 40;
	auto pivot_idx = ml::Features::partition(X, y, 1, 0);
	Eigen::MatrixXd expected_X(2, 4);
	expected_X << 0.2, -0.3, 0.21, 0.3,
		3, -1, 2, 1;
	Eigen::VectorXd expected_y(4);
	expected_y << 40, 30, 20, 10;
	ASSERT_NEAR(0, (expected_X - X).norm(), 1e-15) << X;
	ASSERT_NEAR(0, (expected_y - y).norm(), 1e-15) << y;
	ASSERT_EQ(2, pivot_idx);
	pivot_idx = ml::Features::partition(X.block(0, 0, 2, 1), y.segment(0, 1), 0, 1);
	ASSERT_NEAR(0, (expected_X - X).norm(), 1e-15) << X;
	ASSERT_NEAR(0, (expected_y - y).norm(), 1e-15) << y;
	ASSERT_EQ(0, pivot_idx);
	pivot_idx = ml::Features::partition(X.block(0, 1, 2, 3), y.segment(1, 3), 1, 0);
	ASSERT_NEAR(0, (expected_X - X).norm(), 1e-15) << X;
	ASSERT_NEAR(0, (expected_y - y).norm(), 1e-15) << y;
	ASSERT_EQ(1, pivot_idx);
	pivot_idx = ml::Features::partition(X.block(0, 1, 2, 3), y.segment(1, 3), 1, 1);
	expected_X << 0.2, -0.3, 0.3, 0.21,
		3, -1, 1, 2;
	expected_y << 40, 30, 10, 20;
	ASSERT_NEAR(0, (expected_X - X).norm(), 1e-15) << X;
	ASSERT_NEAR(0, (expected_y - y).norm(), 1e-15) << y;
	ASSERT_EQ(2, pivot_idx);
}

TEST(FeaturesTest, partition_with_labels_move_pivot)
{
	Eigen::MatrixXd X(2, 2);
	X << -1.5, 0,
		0.01, 0.01;
	Eigen::VectorXd y(2);
	y << 10, 30;
	const Eigen::MatrixXd orig_X(X);
	const Eigen::VectorXd orig_y(y);
	ml::Features::partition(X, y, 1, 0);
	ASSERT_EQ(0, (orig_X - X).norm());
	ASSERT_EQ(0, (orig_y - y).norm());
}
