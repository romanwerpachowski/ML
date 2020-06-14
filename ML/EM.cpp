#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include "Clustering.hpp"
#include "EM.hpp"

#define PI 3.14159265358979323846

namespace ml
{
	EM::EM(const unsigned int number_components)
		: means_initialiser_(std::make_shared<Clustering::Forgy>())
		, responsibilities_initialiser_(std::make_shared<Clustering::ClosestMean>(means_initialiser_))
		, mixing_probabilities_(number_components)
		, covariances_(number_components)
		, absolute_tolerance_(1e-8)
		, relative_tolerance_(1e-8)
		, number_components_(number_components)
		, maximum_steps_(1000)
		, verbose_(false)
		, maximise_first_(false)
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

	void EM::set_maximum_steps(unsigned int maximum_steps)
	{
		if (maximum_steps < 2) {
			throw std::invalid_argument("At least two steps required for convergence test");
		}
		maximum_steps_ = maximum_steps;
	}

	void EM::set_means_initialiser(std::shared_ptr<const Clustering::MeansInitialiser> means_initialiser)
	{
		if (!means_initialiser) {
			throw std::invalid_argument("Null means initialiser");
		}
		means_initialiser_ = means_initialiser;
	}

	void EM::set_responsibilities_initialiser(std::shared_ptr<const Clustering::ResponsibilitiesInitialiser> responsibilities_initialiser)
	{
		if (!responsibilities_initialiser) {
			throw std::invalid_argument("Null responsibilities initialiser");
		}
		responsibilities_initialiser_ = responsibilities_initialiser;
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

		if (maximise_first_) {
			responsibilities_initialiser_->init(data, prng_, number_components_, responsibilities_);
			for (unsigned int k = 0; k < number_components_; ++k) {
				covariances_[k].resize(number_dimensions, number_dimensions);
			}
			maximisation_stage(data);			
		} else {
			// Initialise means and covariances to sensible guesses.
			means_initialiser_->init(data, prng_, number_components_, means_);
			const Eigen::MatrixXd sample_covariance(calculate_sample_covariance(data));
			assert(sample_covariance.rows() == number_dimensions);
			assert(sample_covariance.cols() == number_dimensions);
			for (unsigned int k = 0; k < number_components_; ++k) {
				covariances_[k] = sample_covariance;
			}
		}

		// Work variables.
		work_matrix_.resize(number_dimensions, number_dimensions);
		work_vector_.resize(number_dimensions);
		double old_log_likelihood = -std::numeric_limits<double>::infinity();
		
		// Main iteration loop.
		for (unsigned int step = 0; step < maximum_steps_; ++step) {			

			expectation_stage(data);			

			maximisation_stage(data);

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
}