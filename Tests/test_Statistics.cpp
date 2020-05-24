#include "pch.h" // must be on the top
#include <random>
#include "ML/Statistics.hpp"

TEST(StatisticsTest, calc_sse)
{
	std::vector<double> data({ -0.4, 0.6, 1.3 });
	ASSERT_NEAR(0.9 * 0.9 + 0.1 * 0.1 + 0.8 * 0.8, ml::Statistics::calc_sse(data.begin(), data.end()), 1e-15);
	ASSERT_EQ(0, ml::Statistics::calc_sse(data.begin(), data.begin() + 1));
	ASSERT_EQ(0, ml::Statistics::calc_sse(data.begin(), data.begin()));
	ASSERT_EQ(0, ml::Statistics::calc_sse(data.end(), data.end()));
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
	ASSERT_NEAR(expected, ml::Statistics::calc_sse(data.begin(), data.end()), 2e-14 * expected);
}