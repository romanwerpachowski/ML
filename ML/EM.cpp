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
	EM::EM(const unsigned int number_components, std::shared_ptr<const MeansInitialiser> means_initialiser)
		: means_initialiser_(means_initialiser)
		, mixing_probabilities_(number_components)
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
		if (!means_initialiser) {
			throw std::invalid_argument("Null means initialiser");
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

	void EM::set_maximum_steps(unsigned int maximum_steps)
	{
		if (maximum_steps < 2) {
			throw std::invalid_argument("At least two steps required for convergence test");
		}
		maximum_steps_ = maximum_steps;
	}

	const Eigen::MatrixXd& EM::covariance(unsigned int k) const {
		if (k >= number_components_) {
			throw std::invalid_argument("Bad component index");
		}
		return covariances_[k];
	}

	bool EM::fit(const Eigen::Ref<const Eigen::MatrixXd> data)
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
		responsibilities_.resize(sample_size, number_components_);
		mixing_probabilities_.fill(1. / static_cast<double>(number_components_));
		
		if (sample_size == number_components_) {
			// An exact deterministic fit is possible.
			// Center each Gaussian on a different sample point, and set variance to 0.
			responsibilities_ = Eigen::MatrixXd::Identity(sample_size, sample_size);
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				means_.col(i) = data.col(i);
				covariances_[i].setZero(number_dimensions, number_dimensions);
				log_likelihood_ = std::numeric_limits<double>::infinity();
			}
			return true;
		}

		// Initialise means and covariances to sensible guesses.
		means_initialiser_->choose(data, prng_, number_components_, means_);
		const Eigen::MatrixXd sample_covariance(calculate_sample_covariance(data));
		assert(sample_covariance.rows() == number_dimensions);
		assert(sample_covariance.cols() == number_dimensions);
		for (unsigned int k = 0; k < number_components_; ++k) {
			covariances_[k] = sample_covariance;
		}

		// Work variables.
		
		work_matrix_.resize(number_dimensions, number_dimensions);
		work_vector_.resize(number_dimensions);
		const Eigen::MatrixXd epsilon(1e-15 * Eigen::MatrixXd::Identity(number_dimensions, number_dimensions));
		const auto log_likelihood_normalisation_constant = number_dimensions * std::log(2 * PI);
		double old_log_likelihood = -std::numeric_limits<double>::infinity();
		
		// Main iteration loop.
		for (unsigned int step = 0; step < maximum_steps_; ++step) {
			//// Expectation stage. ////

			expectation_stage(data);

			if (verbose_) {
				std::cout << "Step " << step << "\n";
				std::cout << "Log-likelihood == " << log_likelihood_ << "\n";
				std::cout << "Mixing probabilities == " << mixing_probabilities_.transpose() << "\n";
				for (unsigned int k = 0; k < number_components_; ++k) {
					std::cout << "Mean[" << k << "] == " << means_.col(k).transpose() << "\n";
				}
				std::cout << "Responsibilities (first 10 rows):\n";
				std::cout << responsibilities_.topRows(std::min(sample_size, static_cast<Eigen::Index>(10)));
				std::cout << std::endl;
			}			

			maximisation_stage(data);

			if (step > 0) {
				const double ll_change = std::abs(log_likelihood_ - old_log_likelihood);
				if (ll_change < absolute_tolerance_ + relative_tolerance_ * std::max(std::abs(old_log_likelihood), std::abs(log_likelihood_))) {
					return true;
				}
			}
			old_log_likelihood = log_likelihood_;
		}

		return false;
	}

	Eigen::MatrixXd EM::calculate_sample_covariance(Eigen::Ref<const Eigen::MatrixXd> data)
	{
		const Eigen::MatrixXd centred = data.colwise() - data.rowwise().mean();
		const Eigen::MatrixXd covariance = (centred * centred.adjoint()) / (static_cast<double>(data.cols() - 1));
		assert(covariance.rows() == covariance.cols());
		assert(covariance.rows() == data.rows());
		return covariance;
	}

	Eigen::MatrixXd EM::invert_symmetric_positive_definite_matrix(Eigen::Ref<const Eigen::MatrixXd> m)
	{
		assert(m.rows() == m.cols());
		// Uses Cholesky algorithm. Assumes m is symmetric positive definite.
		return m.llt().solve(Eigen::MatrixXd::Identity(m.rows(), m.cols()));
	}

	void EM::expectation_stage(Eigen::Ref<const Eigen::MatrixXd> data)
	{
		const auto number_dimensions = data.rows();
		assert(number_dimensions);
		const auto sample_size = data.cols();
		assert(sample_size >= number_components_);

		static constexpr double epsilon = 1e-15;
		static const double log_2_pi = std::log(2 * PI);
		const auto log_likelihood_normalisation_constant = number_dimensions * log_2_pi;

		// Calculate unnormalised responsibilities.
		for (unsigned int k = 0; k < number_components_; ++k) {
			// Add epsilon * I to avoid numerical issues.
			work_matrix_ = invert_symmetric_positive_definite_matrix(covariances_[k] + epsilon * Eigen::MatrixXd::Identity(number_dimensions, number_dimensions));
			const auto mean = means_.col(k);
			auto component_weights = responsibilities_.col(k);
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				work_vector_ = data.col(i) - mean;
				component_weights[i] = std::exp(-0.5 * work_vector_.transpose() * work_matrix_ * work_vector_);
			}
			component_weights *= mixing_probabilities_[k] / std::sqrt((covariances_[k] + epsilon * Eigen::MatrixXd::Identity(number_dimensions, number_dimensions)).determinant());
		}
		log_likelihood_ = responsibilities_.rowwise().sum().array().log().mean() - log_likelihood_normalisation_constant;

		// Normalise responsibilities for each datapoint.
		for (Eigen::Index i = 0; i < sample_size; ++i) {
			auto datapoint_responsibilities = responsibilities_.row(i);
			const double sum_weights = datapoint_responsibilities.sum();
			datapoint_responsibilities /= sum_weights;
		}
	}

	void EM::maximisation_stage(Eigen::Ref<const Eigen::MatrixXd> data)
	{
		const auto number_dimensions = data.rows();
		assert(number_dimensions);
		const auto sample_size = data.cols();
		assert(sample_size >= number_components_);

		// Calculate new means.
		means_ = data * responsibilities_; // Unnormalised!
		assert(means_.rows() == number_dimensions);
		assert(means_.cols() == number_components_);

		// Calculate new covariances.
		for (unsigned int k = 0; k < number_components_; ++k) {
			auto& covariance = covariances_[k];
			covariance.setZero();
			const auto component_weights = responsibilities_.col(k);
			const auto sum_component_weights = component_weights.sum();

			// Normalise the mean.
			auto mean = means_.col(k);
			mean /= sum_component_weights;

			// Accumulate covariance.
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				work_vector_ = data.col(i) - mean;
				work_matrix_ = work_vector_ * work_vector_.transpose();
				covariance += component_weights[i] * work_matrix_;
			}

			covariance /= sum_component_weights;
			mixing_probabilities_[k] = sum_component_weights / static_cast<double>(sample_size);
			assert(covariance.rows() == number_dimensions);
			assert(covariance.cols() == number_dimensions);
		}
	}

	EM::MeansInitialiser::~MeansInitialiser()
	{}

	void EM::Forgy::choose(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, const unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const
	{
		std::vector<Eigen::Index> all_indices(data.cols());
		std::iota(all_indices.begin(), all_indices.end(), 0);
		std::vector<Eigen::Index> sampled_indices;
		std::sample(all_indices.begin(), all_indices.end(), std::back_inserter(sampled_indices), number_components, prng);
		means.resize(data.rows(), number_components);
		for (unsigned int i = 0; i < number_components; ++i) {
			means.col(i) = data.col(sampled_indices[i]);
		}
	}

	void EM::RandomPartition::choose(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, const unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const
	{
		means.resize(data.rows(), number_components);
		means.setZero();
		std::vector<unsigned int> counters(number_components, 0);
		std::uniform_int_distribution<unsigned int> dist(0, number_components - 1);
		for (Eigen::Index i = 0; i < data.cols(); ++i) {
			const auto k = dist(prng);
			means.col(k) += (data.col(i) - means.col(k)) / static_cast<double>(++counters[k]);
		}
		assert(std::accumulate(counters.begin(), counters.end(), 0) == data.cols());
	}

	void EM::KPP::choose(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, const unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const
	{
		means.resize(data.rows(), number_components);		
		std::vector<double> weights(data.cols());
		for (unsigned int n = 0; n < number_components; ++n) {
			if (n) {
				for (Eigen::Index i = 0; i < data.cols(); ++i) {
					double min_distance_squared = std::numeric_limits<double>::infinity();
					for (unsigned int k = 0; k < n; ++k) {
						const double distance_squared = (data.col(i) - means.col(k)).squaredNorm();
						min_distance_squared = std::min(min_distance_squared, distance_squared);
					}
					weights[i] = min_distance_squared;
				}
			} else {
				std::fill(weights.begin(), weights.end(), 1);
			}
			std::discrete_distribution<Eigen::Index> dist(weights.begin(), weights.end());
			const auto new_mean_idx = dist(prng);
			means.col(n) = data.col(new_mean_idx);
		}
	}
}