#include "pch.h" // must be on the top
#include <random>
#include "ML/Statistics.hpp"

TEST(StatisticsTest, sse)
{
	std::vector<double> data({ -0.4, 0.6, 1.3 });
	ASSERT_NEAR(0.9 * 0.9 + 0.1 * 0.1 + 0.8 * 0.8, ml::Statistics::sse(data.begin(), data.end()), 1e-15);
	ASSERT_EQ(0, ml::Statistics::sse(data.begin(), data.begin() + 1));
	ASSERT_EQ(0, ml::Statistics::sse(data.begin(), data.begin()));
	ASSERT_EQ(0, ml::Statistics::sse(data.end(), data.end()));
}

TEST(StatisticsTest, calc_sse_big_data)
{
	int k = 100000;
	const int n = 2 * k + 1;
	std::vector<double> data(n);
	k = -k;
	const double scale = 0.2;
	for (auto& x : data) {
		x = scale * static_cast<double>(k);
		++k;
	}
	std::default_random_engine rng;
	rng.seed(54523242);
	std::shuffle(data.begin(), data.end(), rng);
	const double expected = 26667066668000;
	ASSERT_NEAR(expected, ml::Statistics::sse(data.begin(), data.end()), 2e-14 * expected);
}

TEST(StatisticsTest, gini_index_constant_class)
{
	std::vector<double> data(100, 1);
	ASSERT_EQ(0, ml::Statistics::gini_index(data.begin(), data.end()));
	ASSERT_EQ(0, ml::Statistics::gini_index(data.begin(), data.end(), 2));
}

TEST(StatisticsTest, gini_index_two_equal_classes)
{
	std::vector<double> data(100);
	for (size_t i = 0; i < 50; ++i) {
		data[i] = 0;
		data[i + 50] = 2;
	}
	ASSERT_NEAR(0.5, ml::Statistics::gini_index(data.begin(), data.end()), 1e-15);
	ASSERT_NEAR(0.5, ml::Statistics::gini_index(data.begin(), data.end(), 3), 1e-15);
}

TEST(StatisticsTest, gini_index_two_unequal_classes)
{
	std::vector<double> data(100);
	for (size_t i = 0; i < 75; ++i) {
		data[i] = 0;
	}
	for (size_t i = 75; i < 100; ++i) {
		data[i] = 1;
	}
	ASSERT_NEAR(3./8, ml::Statistics::gini_index(data.begin(), data.end()), 1e-15);
	ASSERT_NEAR(3./8, ml::Statistics::gini_index(data.begin(), data.end(), 2), 1e-15);
}

TEST(StatisticsTest, gini_index_three_equal_classes)
{
	std::vector<double> data(99);
	for (size_t i = 0; i < 33; ++i) {
		data[i] = 0;
		data[i + 33] = 2;
		data[i + 66] = 1;
	}
	ASSERT_NEAR(2./3, ml::Statistics::gini_index(data.begin(), data.end()), 1e-15);
	ASSERT_NEAR(2./3, ml::Statistics::gini_index(data.begin(), data.end(), 3), 1e-15);
}