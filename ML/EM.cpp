#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include "EM.hpp"

#define PI 3.14159265358979323846

namespace ml
{
	EM::EM(const unsigned int number_components)
		: mixing_probabilities_(number_components)
		, means_(1, number_components)
		, covariances_(number_components)
		, absolute_tolerance_(1e-8)
		, relative_tolerance_(1e-8)
		, number_components_(number_components)
		, maximum_steps_(1000)
		, verbose_(false)
	{
		if (!number_components) {
			throw std::invalid_argument("At least one component required");
		}
	}	

	void EM::set_seed(unsigned int seed)
	{
		prng_.seed(seed);
	}

	void EM::set_absolute_tolerance(double absolute_tolerance)
	{
		if (absolute_tolerance < 0) {
			throw std::domain_error("Negative absolute tolerance");
		}
		absolute_tolerance_ = absolute_tolerance;
	}

	void EM::set_relative_tolerance(double relative_tolerance)
	{
		if (relative_tolerance < 0) {
			throw std::domain_error("Negative relative tolerance");
		}
		relative_tolerance_ = relative_tolerance;
	}

	const Eigen::MatrixXd& EM::covariance(unsigned int k) const {
		if (k >= number_components_) {
			throw std::invalid_argument("Bad component index");
		}
		return covariances_[k];
	}

	bool EM::fit(const Eigen::MatrixXd& data)
	{
		const auto number_dimensions = data.rows();
		const auto sample_size = data.cols();
		if (!number_dimensions) {
			throw std::invalid_argument("At least one dimension required");
		}
		if (sample_size < number_components_) {
			throw std::invalid_argument("Not enough data ");
		}

		means_.resize(number_dimensions, number_components_);
		mixing_probabilities_.fill(1. / static_cast<double>(number_components_));
		
		if (sample_size == number_components_) {
			// An exact deterministic fit is possible.
			// Center each Gaussian on a different sample point, and set variance to 0.
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				means_.col(i) = data.col(i);
				covariances_[i].setZero(number_dimensions, number_dimensions);
			}
			return true;
		}

		// Initialise means and covariances to sensible guesses.
		std::vector<Eigen::Index> initial_means_indices(sample_datapoints_without_replacement(sample_size));
		const Eigen::MatrixXd sample_covariance(calculate_sample_covariance(data));
		assert(sample_covariance.rows() == number_dimensions);
		assert(sample_covariance.cols() == number_dimensions);
		for (unsigned int k = 0; k < number_components_; ++k) {
			means_.col(k) = data.col(initial_means_indices[k]);
			covariances_[k] = sample_covariance;
		}

		// Work variables.
		Eigen::MatrixXd responsibilities(sample_size, number_components_);
		Eigen::MatrixXd tmp_matrix(number_dimensions, number_dimensions);
		Eigen::VectorXd centred_datapoint(number_dimensions);		
		const Eigen::MatrixXd epsilon(1e-15 * Eigen::MatrixXd::Identity(number_dimensions, number_dimensions));
		const auto log_likelihood_normalisation_constant = number_dimensions * std::log(2 * PI);
		double old_log_likelihood = -std::numeric_limits<double>::infinity();
		
		// Main iteration loop.
		for (unsigned int step = 0; step < maximum_steps_; ++step) {
			//// Expectation stage. ////

			// Calculate unnormalised responsibilities.
			for (unsigned int k = 0; k < number_components_; ++k) {
				// Add 1e-15 * I to avoid numerical issues.
				tmp_matrix = invert_symmetric_positive_definite_matrix(covariances_[k] + epsilon);
				const auto mean = means_.col(k);
				auto component_weights = responsibilities.col(k);
				for (Eigen::Index i = 0; i < sample_size; ++i) {
					centred_datapoint = data.col(i) - mean;
					component_weights[i] = std::exp(-0.5 * centred_datapoint.transpose() * tmp_matrix * centred_datapoint);
				}
				component_weights *= mixing_probabilities_[k] / std::sqrt((covariances_[k] + epsilon).determinant());
			}
			log_likelihood_ = responsibilities.rowwise().sum().array().log().mean() - log_likelihood_normalisation_constant;

			// Normalise responsibilities for each datapoint.
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				auto datapoint_responsibilities = responsibilities.row(i);
				const double sum_weights = datapoint_responsibilities.sum();
				datapoint_responsibilities /= sum_weights;
			}

			if (verbose_) {
				std::cout << "Step " << step << "\n";
				std::cout << "Log-likelihood == " << log_likelihood_ << "\n";
				std::cout << "Mixing probabilities == " << mixing_probabilities_.transpose() << "\n";
				for (unsigned int k = 0; k < number_components_; ++k) {
					std::cout << "Mean[" << k << "] == " << means_.col(k).transpose() << "\n";
				}
				std::cout << "Responsibilities (first 10 rows):\n";
				std::cout << responsibilities.topRows(std::min(sample_size, static_cast<Eigen::Index>(10)));
				std::cout << std::endl;
			}

			if (step > 0) {
				const double ll_change = std::abs(log_likelihood_ - old_log_likelihood);
				if (ll_change < absolute_tolerance_ + relative_tolerance_ * std::max(std::abs(old_log_likelihood), std::abs(log_likelihood_))) {
					return true;
				}
			}
			old_log_likelihood = log_likelihood_;

			

			//// Maximisation stage. ////

			// Calculate new means.
			means_ = data * responsibilities; // Unnormalised!
			assert(means_.rows() == number_dimensions);
			assert(means_.cols() == number_components_);

			// Calculate new covariances.
			for (unsigned int k = 0; k < number_components_; ++k) {
				auto& covariance = covariances_[k];
				covariance.setZero();
				const auto component_weights = responsibilities.col(k);
				const auto sum_component_weights = component_weights.sum();

				// Normalise the mean.
				auto mean = means_.col(k);
				mean /= sum_component_weights;

				// Accumulate covariance.
				for (Eigen::Index i = 0; i < sample_size; ++i) {
					centred_datapoint = data.col(i) - mean;
					tmp_matrix = centred_datapoint * centred_datapoint.transpose();
					covariance += component_weights[i] * tmp_matrix;
				}
				
				covariance /= sum_component_weights;
				mixing_probabilities_[k] = sum_component_weights / static_cast<double>(sample_size);
				assert(covariance.rows() == number_dimensions);
				assert(covariance.cols() == number_dimensions);				
			}
		}

		return false;
	}

	std::vector<Eigen::Index> EM::sample_datapoints_without_replacement(Eigen::Index sample_size)
	{
		std::vector<Eigen::Index> all_indices(sample_size);
		std::iota(all_indices.begin(), all_indices.end(), 0);
		std::vector<Eigen::Index> sampled_indices;
		std::sample(all_indices.begin(), all_indices.end(), std::back_inserter(sampled_indices), number_components_, prng_);
		return sampled_indices;
	}

	Eigen::MatrixXd EM::calculate_sample_covariance(const Eigen::MatrixXd& data)
	{
		const Eigen::MatrixXd centred = data.colwise() - data.rowwise().mean();
		const Eigen::MatrixXd covariance = (centred * centred.adjoint()) / (static_cast<double>(data.cols() - 1));
		assert(covariance.rows() == covariance.cols());
		assert(covariance.rows() == data.rows());
		return covariance;
	}

	Eigen::MatrixXd EM::invert_symmetric_positive_definite_matrix(const Eigen::MatrixXd& m)
	{
		assert(m.rows() == m.cols());
		// Uses Cholesky algorithm. Assumes m is symmetric positive definite.
		return m.llt().solve(Eigen::MatrixXd::Identity(m.rows(), m.cols()));
	}
}