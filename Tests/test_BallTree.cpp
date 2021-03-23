/* (C) 2021 Roman Werpachowski. */
#include <gtest/gtest.h>
#include <Eigen/Core>
#include "ML/BallTree.hpp"


TEST(BallTreeTest, construction_empty)
{
    Eigen::MatrixXd X(2, 0);
    ml::BallTree bt(X, 3);
    ASSERT_EQ(X.rows(), bt.data().rows());
    ASSERT_EQ(X.cols(), bt.data().cols());
}


TEST(BallTreeTest, construction_one_vec)
{
    Eigen::MatrixXd X(2, 1);
    X << 0.5, 0.5;
    ml::BallTree bt(X, 3);
    ASSERT_EQ(X.rows(), bt.data().rows());
    ASSERT_EQ(X.cols(), bt.data().cols());
    ASSERT_NEAR(0, (X - bt.data()).norm(), 1e-15) << bt.data();
}


TEST(BallTreeTest, construction_two_vec)
{
    Eigen::MatrixXd X(3, 2);
    X << 0.5, 0.5,
        -1.5, 1.5,
        0, 0;
    ml::BallTree bt(X, 3);
    ASSERT_EQ(X.rows(), bt.data().rows());
    ASSERT_EQ(X.cols(), bt.data().cols());
    ASSERT_NEAR(0, (X - bt.data()).norm(), 1e-15) << bt.data();
}


TEST(BallTreeTest, construction_three_vec)
{
    Eigen::MatrixXd X(2, 3);
    X << 0.5, 0.6, 0.4,
        -1.5, 1.5, 0;
    Eigen::VectorXd y(3);
    y << 10, 20, 30;
    ml::BallTree bt(X, y, 3);
    ASSERT_EQ(X.rows(), bt.data().rows());
    ASSERT_EQ(X.cols(), bt.data().cols());
    ASSERT_EQ(y.size(), bt.labels().size());
    Eigen::MatrixXd expected(2, 3);
    expected << 0.5, 0.4, 0.6,
        -1.5, 0, 1.5;    
    Eigen::VectorXd expected_y(3);
    expected_y << 10, 30, 20;
    ASSERT_NEAR(0, (expected - bt.data()).norm(), 1e-15) << bt.data();
    ASSERT_EQ(0, (expected_y - bt.labels()).norm()) << bt.labels();
}


TEST(BallTreeTest, construction_large)
{
    const unsigned int n = 100;
    const unsigned int d = 5;
    Eigen::MatrixXd X(Eigen::MatrixXd::Random(d, n));
    Eigen::VectorXd y(Eigen::VectorXd::Random(n));
    ml::BallTree bt(X, y, 20);
    ASSERT_EQ(X.rows(), bt.data().rows());
    ASSERT_EQ(X.cols(), bt.data().cols());
    ASSERT_NEAR(X.sum(), bt.data().sum(), 1e-14);
    ASSERT_NEAR(y.sum(), bt.labels().sum(), 1e-14);
}


TEST(BallTreeTest, find_zero_nearest_neighbours)
{
    Eigen::MatrixXd X(2, 4);
    X << 0.5, 0.6, 0.4, 2,
        -1.5, 1.5, 0, 4;
    std::vector<unsigned int> nn;
    ml::BallTree bt(X, 3);
    Eigen::VectorXd x(2);
    x << 0.49, -1.51;
    bt.find_k_nearest_neighbours(x, 0, nn);
    ASSERT_EQ(0u, nn.size());    
}


TEST(BallTreeTest, find_one_nearest_neighbour)
{
    Eigen::MatrixXd X(2, 4);
    X << 0.5, 0.6, 0.4, 2,
        -1.5, 1.5, 0, 4;
    std::vector<unsigned int> nn;
    ml::BallTree bt(X, 3);
    Eigen::VectorXd x(2);
    x << 0.49, -1.51;
    bt.find_k_nearest_neighbours(x, 1, nn);
    ASSERT_EQ(1u, nn.size());
    const auto nn0 = bt.data().col(nn[0]);
    ASSERT_EQ(0, (X.col(0) - nn0).norm()) << nn0;
}


TEST(BallTreeTest, find_two_nearest_neighbour)
{
    Eigen::MatrixXd X(2, 4);
    X << 0.5, 0.6, 0.4, 2,
        -1.5, 1.5, 0, 4;
    Eigen::VectorXd y(4);
    y << 10, 20, 30, 40;
    std::vector<unsigned int> nn;
    ml::BallTree bt(X, y, 3);
    Eigen::VectorXd x(2);
    x << 0.49, -1.51;
    bt.find_k_nearest_neighbours(x, 2, nn);
    ASSERT_EQ(2u, nn.size());
    ASSERT_EQ(0, (X.col(2) - bt.data().col(nn[0])).norm()) << bt.data().col(nn[0]);
    ASSERT_EQ(0, (X.col(0) - bt.data().col(nn[1])).norm()) << bt.data().col(nn[1]);
    ASSERT_EQ(y[2], bt.labels()[nn[0]]);
    ASSERT_EQ(y[0], bt.labels()[nn[1]]);
}


TEST(BallTreeTest, find_many_nearest_neighbour)
{
    Eigen::MatrixXd X(2, 4);
    X << 0.5, 0.6, 0.4, 2,
        -1.5, 1.5, 0, 4;
    std::vector<unsigned int> nn;
    ml::BallTree bt(X, 3);
    Eigen::VectorXd x(2);
    x << 0.49, -1.51;
    bt.find_k_nearest_neighbours(x, 100, nn);
    ASSERT_EQ(4u, nn.size());
    std::sort(nn.begin(), nn.end());
    ASSERT_EQ(std::vector<unsigned int>({ 0, 1, 2, 3 }), nn);
}
