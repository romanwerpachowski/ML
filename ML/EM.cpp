#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <Eigen/Cholesky>
#include "EM.hpp"

namespace ml
{
	EM::EM(const unsigned int number_components, const double absolute_tolerance, const double relative_tolerance)
		: mixing_proportions_(number_components), means_(1, number_components_), covariances_(number_components_), absolute_tolerance_(absolute_tolerance), relative_tolerance_(relative_tolerance), number_components_(number_components)
	{
		if (absolute_tolerance < 0) {
			throw std::domain_error("Negative absolute tolerance");
		}
		if (relative_tolerance < 0) {
			throw std::domain_error("Negative relative tolerance");
		}
		if (!number_components) {
			throw std::invalid_argument("At least one component required");
		}		
	}	

	void EM::set_seed(unsigned int seed)
	{
		prng_.seed(seed);
	}

	double EM::fit(const Eigen::MatrixXd& data)
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
		
		if (sample_size == number_components_) {
			// An exact deterministic fit is possible.
			// Center each Gaussian on a different sample point, and set variance to 0.
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				means_.col(i) = data.col(i);
				covariances_[i].setZero(number_dimensions, number_dimensions);
			}
			return std::numeric_limits<double>::infinity();
		}

		std::fill(mixing_proportions_.begin(), mixing_proportions_.end(), 1. / static_cast<double>(number_components_));		

		// Initialise means and covariances to sensible guesses.
		std::vector<Eigen::Index> initial_means_indices(sample_datapoints_without_replacement(sample_size));
		const Eigen::MatrixXd sample_covariance(calculate_sample_covariance(data));
		for (unsigned int k = 0; k < number_components_; ++k) {
			means_.col(k) = data.col(initial_means_indices[k]);
			covariances_[k] = sample_covariance;
		}

		Eigen::MatrixXd responsibilities(sample_size, number_components_);

		// Temporary variables.
		Eigen::MatrixXd inverse_covariance(number_components_, number_components_);
		Eigen::VectorXd centred_point(number_components_);

		// Main iteration loop.
		while (true) {
			//// Expectation stage. ////

			// Calculate unnormalised responsibilities.
			for (unsigned int k = 0; k < number_components_; ++k) {
				// Add 1e-15 * I to make sure covariance matrix can be inverted.
				inverse_covariance = invert_symmetric_positive_definite_matrix(covariances_[k] + 1e-15 * Eigen::MatrixXd::Identity(number_dimensions, number_dimensions));
				const auto mean = means_.col(k);
				auto kth_component_weights = responsibilities.col(k);
				for (Eigen::Index i = 0; i < sample_size; ++i) {
					centred_point = data.col(i) - mean;
					kth_component_weights[i] = std::exp(-0.5 * centred_point.transpose() * inverse_covariance * centred_point);
				}
			}
			// Normalise responsibilities.			
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				const double sum_weights = responsibilities.row(i).sum();
				responsibilities.row(i) /= sum_weights;
			}

			//// Maximisation stage. ////
			means_ = data * responsibilities;
		}

		return 0;
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
		return (centred.adjoint() * centred) / (static_cast<double>(data.cols() - 1));
	}

	Eigen::MatrixXd EM::invert_symmetric_positive_definite_matrix(const Eigen::MatrixXd& m)
	{
		assert(m.rows() == m.cols());
		// Assumes m is symmetric positive definite.
		return m.llt().solve(Eigen::MatrixXd::Identity(m.rows(), m.cols()));
	}
}