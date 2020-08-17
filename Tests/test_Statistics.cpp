#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include "ML/Statistics.hpp"

using namespace ml::Statistics;

TEST(StatisticsTest, sse)
{
	const std::vector<double> data({ -0.4, 0.6, 1.3 });
	ASSERT_NEAR(0.9 * 0.9 + 0.1 * 0.1 + 0.8 * 0.8, sse(data.begin(), data.end()), 1e-15);
	ASSERT_EQ(0, sse(data.begin(), data.begin() + 1));
	ASSERT_EQ(0, sse(data.begin(), data.begin()));
	ASSERT_EQ(0, sse(data.end(), data.end()));
}

TEST(StatisticsTest, sse_and_mean)
{
	const std::vector<double> data({ -0.4, 0.6, 1.3 });
	auto actual = sse_and_mean(data.begin(), data.end());
	ASSERT_NEAR(0.9 * 0.9 + 0.1 * 0.1 + 0.8 * 0.8, actual.first, 1e-15);
	ASSERT_NEAR(0.5, actual.second, 1e-15);
	actual = sse_and_mean(data.begin(), data.begin() + 1);
	ASSERT_EQ(0, actual.first);
	ASSERT_EQ(-0.4, actual.second);
	actual = sse_and_mean(data.begin(), data.begin());
	ASSERT_EQ(0, actual.first);
	ASSERT_TRUE(std::isnan(actual.second));
	actual = sse_and_mean(data.end(), data.end());
	ASSERT_EQ(0, actual.first);
	ASSERT_TRUE(std::isnan(actual.second));
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
	ASSERT_NEAR(expected, sse(data.begin(), data.end()), 2e-14 * expected);
	const auto actual = sse_and_mean(data.begin(), data.end());
	ASSERT_NEAR(expected, actual.first, 2e-14 * expected);
	ASSERT_NEAR(0, actual.second, 2e-13);
}

TEST(StatisticsTest, gini_index_constant_class)
{
	std::vector<int> data(100, 1);
	ASSERT_EQ(0, gini_index(data.begin(), data.end(), 2));
}

TEST(StatisticsTest, gini_index_two_equal_classes)
{
	std::vector<int> data(100);
	for (size_t i = 0; i < 50; ++i) {
		data[i] = 0;
		data[i + 50] = 2;
	}
	ASSERT_NEAR(0.5, gini_index(data.begin(), data.end(), 3), 1e-15);
}

TEST(StatisticsTest, gini_index_two_unequal_classes)
{
	std::vector<unsigned int> data(100);
	for (size_t i = 0; i < 75; ++i) {
		data[i] = 0;
	}
	for (size_t i = 75; i < 100; ++i) {
		data[i] = 1;
	}
	ASSERT_NEAR(3./8, gini_index(data.begin(), data.end(), 2), 1e-15);
}

TEST(StatisticsTest, gini_index_three_equal_classes)
{
	std::vector<unsigned int> data(99);
	for (size_t i = 0; i < 33; ++i) {
		data[i] = 0;
		data[i + 33] = 2;
		data[i + 66] = 1;
	}
	ASSERT_NEAR(2./3, gini_index(data.begin(), data.end(), 3), 1e-15);
}

TEST(StatisticsTest, gini_index_and_mode_constant_class)
{
	std::vector<unsigned int> data(100, 1);
	auto actual = gini_index_and_mode(data.begin(), data.end(), 2);
	ASSERT_EQ(0, actual.first);
	ASSERT_EQ(1u, actual.second);
}

TEST(StatisticsTest, gini_index_and_mode_two_equal_classes)
{
	std::vector<int> data(100);
	for (size_t i = 0; i < 50; ++i) {
		data[i] = 0;
		data[i + 50] = 2;
	}
	auto actual = gini_index_and_mode(data.begin(), data.end(), 3);
	ASSERT_NEAR(0.5, actual.first, 1e-15);
	ASSERT_EQ(0u, actual.second);
}

TEST(StatisticsTest, gini_index_and_mode_two_unequal_classes)
{
	std::vector<unsigned int> data(100);
	for (size_t i = 0; i < 75; ++i) {
		data[i] = 1;
	}
	for (size_t i = 75; i < 100; ++i) {
		data[i] = 0;
	}
	auto actual = gini_index_and_mode(data.begin(), data.end(), 2);
	ASSERT_NEAR(3. / 8, actual.first, 1e-15);
	ASSERT_EQ(1u, actual.second);	
}

TEST(StatisticsTest, gini_index_and_mode_three_equal_classes)
{
	std::vector<double> data(99);
	for (size_t i = 0; i < 33; ++i) {
		data[i] = 0;
		data[i + 33] = 2;
		data[i + 66] = 1;
	}
	auto actual = gini_index_and_mode(data.begin(), data.end(), 3);
	ASSERT_NEAR(2. / 3, actual.first, 1e-15);
	ASSERT_EQ(0u, actual.second);
}

TEST(StatisticsTest, mode)
{
	std::vector<double> data({ 1, 0, 1, 1, 0, 0, 2, 2, 2, 1, 0, 1, 2, 1, 1, 1, 0, 1 });
	const auto m = mode(data.begin(), data.end(), 3);
	ASSERT_EQ(1u, m);
}