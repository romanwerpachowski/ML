/* (C) 2020 Roman Werpachowski. */
#include <stdexcept>
#include <gtest/gtest.h>
#include "ML/RootFinding.hpp"

TEST(SolveQuadratic, errors)
{
    double x1, x2;
    ASSERT_THROW(ml::RootFinding::solve_quadratic(0, 1, 1, x1, x2), std::domain_error);
}

TEST(SolveQuadratic, no_roots)
{
    double x1, x2;
    x1 = 0.9999;
    x2 = 0.1233;
    ASSERT_EQ(0u, ml::RootFinding::solve_quadratic(1, 0, 1, x1, x2));
    ASSERT_EQ(0.9999, x1);
    ASSERT_EQ(0.1233, x2);
}

TEST(SolveQuadratic, one_root)
{
    double x1, x2;
    x2 = 0.1233;
    ASSERT_EQ(1u, ml::RootFinding::solve_quadratic(1, -2, 1, x1, x2));
    ASSERT_NEAR(1, x1, 1e-15);
    ASSERT_EQ(0.1233, x2);
}

TEST(SolveQuadratic, two_roots)
{
    double x1, x2;
    ASSERT_EQ(2u, ml::RootFinding::solve_quadratic(1, -1, -2, x1, x2));
    if (x1 > x2) {
        std::swap(x1, x2);
    }
    ASSERT_NEAR(x1, -1, 1e-15);
    ASSERT_NEAR(x2, 2, 1e-15);
}

TEST(SolveQuadratic, very_large_and_very_small_root)
{
    double x1, x2;
    ASSERT_EQ(2u, ml::RootFinding::solve_quadratic(1, - 1 - 1e-15, 1e-15, x1, x2));
    if (x1 > x2) {
        std::swap(x1, x2);
    }
    ASSERT_NEAR(1e-15, x1, 1e-40);
    ASSERT_NEAR(1, x2, 1e-15);

}