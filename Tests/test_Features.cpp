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
