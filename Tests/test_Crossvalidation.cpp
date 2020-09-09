/* (C) 2020 Roman Werpachowski. */
#include <stdexcept>
#include <gtest/gtest.h>
#include "ML/Crossvalidation.hpp"
#include "ML/LinearRegression.hpp"

using namespace ml;

TEST(CrossvalidationTest, calc_fold_indices)
{
	size_t i0, i1;
	Crossvalidation::calc_fold_indices(10, 0, 5, i0, i1);
	ASSERT_EQ(0, i0);
	ASSERT_EQ(2, i1);
	// Rounding down.
	Crossvalidation::calc_fold_indices(10, 0, 3, i0, i1);
	ASSERT_EQ(0, i0);
	ASSERT_EQ(3, i1);
	Crossvalidation::calc_fold_indices(10, 1, 3, i0, i1);
	ASSERT_EQ(3, i0);
	ASSERT_EQ(6, i1);
	Crossvalidation::calc_fold_indices(10, 2, 3, i0, i1);
	ASSERT_EQ(6, i0);
	ASSERT_EQ(10, i1);
	// Rounding up.
	Crossvalidation::calc_fold_indices(5, 0, 3, i0, i1);
	ASSERT_EQ(0, i0);
	ASSERT_EQ(2, i1);
	Crossvalidation::calc_fold_indices(5, 1, 3, i0, i1);
	ASSERT_EQ(2, i0);
	ASSERT_EQ(4, i1);
	Crossvalidation::calc_fold_indices(5, 2, 3, i0, i1);
	ASSERT_EQ(4, i0);
	ASSERT_EQ(5, i1);
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

TEST(CrossvalidationTest, only_kth_fold_2d)
{
	Eigen::MatrixXd data(2, 5);
	data << 0, 1, 2, 3, 4,
		5, 6, 7, 8, 9;
	Eigen::MatrixXd expected(2, 2);
	expected << 3, 4,
		8, 9;
	ASSERT_EQ(expected, Crossvalidation::only_kth_fold_2d(data, 1, 2));
	expected.resize(2, 3);
	expected << 0, 1, 2,
		5, 6, 7;
	ASSERT_EQ(expected, Crossvalidation::only_kth_fold_2d(data, 0, 2));
	expected.resize(2, 2);
	expected << 0, 1,
		5, 6;
	ASSERT_EQ(expected, Crossvalidation::only_kth_fold_2d(data, 0, 3));
	expected << 2, 3,
		7, 8;
	ASSERT_EQ(expected, Crossvalidation::only_kth_fold_2d(data, 1, 3));
	expected.resize(2, 1);
	expected << 4,
		9;
	ASSERT_EQ(expected, Crossvalidation::only_kth_fold_2d(data, 2, 3));
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

TEST(CrossvalidationTest, without_kth_fold_2d)
{
	Eigen::MatrixXd data(2, 5);
	data << 0, 1, 2, 3, 4,
		5, 6, 7, 8, 9;
	Eigen::MatrixXd expected(2, 2);
	expected << 3, 4,
		8, 9;
	ASSERT_EQ(expected, Crossvalidation::without_kth_fold_2d(data, 0, 2));
	expected.resize(2, 3);
	expected << 0, 1, 2,
		5, 6, 7;
	ASSERT_EQ(expected, Crossvalidation::without_kth_fold_2d(data, 1, 2));
	expected.resize(2, 3);
	expected << 0, 1, 4,
		5, 6, 9;
	ASSERT_EQ(expected, Crossvalidation::without_kth_fold_2d(data, 1, 3));
}

TEST(CrossvalidationTest, k_fold)
{
	const int sample_size = 100;
	const int dim = 4;
	const Eigen::MatrixXd X(Eigen::MatrixXd::Random(dim, sample_size));
	const Eigen::VectorXd y(Eigen::VectorXd::Random(sample_size));
	const int num_folds = 9;
	const double error = Crossvalidation::k_fold(X, y,
		[](const Eigen::MatrixXd& /*train_X*/, const Eigen::VectorXd& train_y) -> double {
			return train_y.mean();
		},
		[](double model, const Eigen::MatrixXd& /*train_X*/, const Eigen::Ref<const Eigen::VectorXd> test_y) -> double {
			return std::pow(test_y.mean() - model, 2) / static_cast<double>(test_y.size());
		}, num_folds);
	ASSERT_GE(error, 0);
	ASSERT_LE(error, 0.01);
}

TEST(CrossvalidationTest, leave_one_out)
{
	Eigen::MatrixXd X(2, 3);
	X << -1, 0, 1,
		1, 1, 1;
	Eigen::VectorXd y(3);
	y << 1, 0, 1;
	const auto trainer = [](const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y) -> LinearRegression::MultivariateOLSResult {
		return LinearRegression::multivariate(X, y);
	};
	const auto tester = [](const LinearRegression::MultivariateOLSResult& model, const Eigen::Ref<const Eigen::MatrixXd> X, const Eigen::Ref<const Eigen::VectorXd> y) -> double {
		return (y - model.predict(X)).squaredNorm() / static_cast<double>(y.size());
	};
	const double loocv_error = Crossvalidation::leave_one_out(X, y, trainer, tester);
	ASSERT_NEAR((4 + 1 + 4) / 3, loocv_error, 1e-15);
	const double kfold_error = Crossvalidation::k_fold(X, y, trainer, tester, 3);
	ASSERT_NEAR(kfold_error, loocv_error, 1e-15);
}

TEST(CrossvalidationTest, cv_scalar)
{
	Eigen::VectorXd x(3);
	x << -1, 0, 1;
	Eigen::VectorXd y(3);
	y << 1, 0, 1;
	const auto trainer = [](const Eigen::Ref<const Eigen::VectorXd> x, const Eigen::Ref<const Eigen::VectorXd> y) {
		return LinearRegression::univariate(x, y);
	};
	const auto tester = [](const LinearRegression::UnivariateOLSResult& model, const Eigen::Ref<const Eigen::VectorXd> x, const Eigen::Ref<const Eigen::VectorXd> y) -> double {
		return (y - model.predict(x)).squaredNorm() / static_cast<double>(y.size());
	};
	const double loocv_error = Crossvalidation::leave_one_out_scalar(x, y, trainer, tester);
	ASSERT_NEAR((4 + 1 + 4) / 3, loocv_error, 1e-15);
	const double kfold_error = Crossvalidation::k_fold_scalar(x, y, trainer, tester, 3);
	ASSERT_NEAR(kfold_error, loocv_error, 1e-15);
}