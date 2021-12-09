/* (C) 2020 Roman Werpachowski. */
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
#include "LinearAlgebra.hpp"

#define PI 3.14159265358979323846

namespace ml
{
	EM::EM(const unsigned int number_components)
		: means_initialiser_(std::make_shared<Clustering::Forgy>())
		, responsibilities_initialiser_(std::make_shared<Clustering::ClosestCentroid>(means_initialiser_))
		, mixing_probabilities_(number_components)
		, covariances_(number_components)
		, inverse_covariances_(number_components)
		, covariance_decompositions_(number_components)
		, sqrt_covariance_determinants_(number_components)
		, absolute_tolerance_(1e-8)
		, relative_tolerance_(1e-8)
		, number_components_(number_components)
		, maximum_steps_(1000)
		, verbose_(false)
		, maximise_first_(false)
		, converged_(false)
	{
		if (!number_components) {
			throw std::invalid_argument("EM: At least one component required");
		}
	}	

	void EM::set_seed(unsigned int seed)
	{
		prng_.seed(seed);
	}

	void EM::set_absolute_tolerance(double absolute_tolerance)
	{
		if (absolute_tolerance < 0) {
			throw std::domain_error("EM: Negative absolute tolerance");
		}
		absolute_tolerance_ = absolute_tolerance;
	}

	void EM::set_relative_tolerance(double relative_tolerance)
	{
		if (relative_tolerance < 0) {
			throw std::domain_error("EM: Negative relative tolerance");
		}
		relative_tolerance_ = relative_tolerance;
	}

	void EM::set_maximum_steps(unsigned int maximum_steps)
	{
		if (maximum_steps < 2) {
			throw std::invalid_argument("EM: At least two steps required for convergence test");
		}
		maximum_steps_ = maximum_steps;
	}

	void EM::set_means_initialiser(std::shared_ptr<const Clustering::CentroidsInitialiser> means_initialiser)
	{
		if (!means_initialiser) {
			throw std::invalid_argument("EM: Null means initialiser");
		}
		means_initialiser_ = means_initialiser;
	}

	void EM::set_responsibilities_initialiser(std::shared_ptr<const Clustering::ResponsibilitiesInitialiser> responsibilities_initialiser)
	{
		if (!responsibilities_initialiser) {
			throw std::invalid_argument("EM: Null responsibilities initialiser");
		}
		responsibilities_initialiser_ = responsibilities_initialiser;
	}

	const Eigen::MatrixXd& EM::covariance(unsigned int k) const {
		if (k >= number_components_) {
			throw std::invalid_argument("EM: Bad component index");
		}
		return covariances_[k];
	}

	bool EM::fit(const Eigen::Ref<const Eigen::MatrixXd> data)
	{
		converged_ = false;
		const auto number_dimensions = static_cast<unsigned int>(data.rows());
		const auto sample_size = static_cast<unsigned int>(data.cols());
		if (!number_dimensions) {
			throw std::invalid_argument("EM: At least one dimension required");
		}
		if (sample_size < number_components_) {
			throw std::invalid_argument("EM: Not enough data ");
		}

		means_.resize(number_dimensions, number_components_);
		responsibilities_.resize(sample_size, number_components_);
		mixing_probabilities_.fill(1. / static_cast<double>(number_components_));
		labels_.resize(sample_size);
		
		if (sample_size == number_components_) {
			// An exact deterministic fit is possible.
			// Center each Gaussian on a different sample point, and set variance to 0.
			responsibilities_ = Eigen::MatrixXd::Identity(sample_size, sample_size);
			for (unsigned int i = 0; i < sample_size; ++i) {
				means_.col(i) = data.col(i);
				covariances_[i].setZero(number_dimensions, number_dimensions);
				log_likelihood_ = std::numeric_limits<double>::infinity();
				labels_[i] = i;
			}
			converged_ = true;
		} else {
			if (maximise_first_) {
				responsibilities_initialiser_->init(data, prng_, number_components_, responsibilities_);
				for (unsigned int k = 0; k < number_components_; ++k) {
					covariances_[k].resize(number_dimensions, number_dimensions);
				}
				maximisation_step(data);
			} else {
				// Initialise means and covariances to sensible guesses.
				means_initialiser_->init(data, prng_, number_components_, means_);
				const Eigen::MatrixXd sample_covariance(calculate_sample_covariance(data));
				assert(static_cast<unsigned int>(sample_covariance.rows()) == number_dimensions);
				assert(static_cast<unsigned int>(sample_covariance.cols()) == number_dimensions);
				for (unsigned int k = 0; k < number_components_; ++k) {
					covariances_[k] = sample_covariance;
				}
				process_covariances(number_dimensions);
			}

			// Work variables.
			work_vector_.resize(number_dimensions);
			double old_log_likelihood = -std::numeric_limits<double>::infinity();

			// Main iteration loop.
			for (unsigned int step = 0; step < maximum_steps_; ++step) {

				expectation_step(data);

				maximisation_step(data);

				if (verbose_) {
					std::cout << "Step " << step << "\n";
					std::cout << "Log-likelihood == " << log_likelihood_ << "\n";
					std::cout << "Mixing probabilities == " << mixing_probabilities_.transpose() << "\n";
					for (unsigned int k = 0; k < number_components_; ++k) {
						std::cout << "Mean[" << k << "] == " << means_.col(k).transpose() << "\n";
					}
					std::cout << "Responsibilities (first 10 rows):\n";
					std::cout << responsibilities_.topRows(std::min(sample_size, 10u));
					std::cout << std::endl;
				}

				if (step > 0) {
					const double ll_change = std::abs(log_likelihood_ - old_log_likelihood);
					if (ll_change < absolute_tolerance_ + relative_tolerance_ * std::max(std::abs(old_log_likelihood), std::abs(log_likelihood_))) {
						calculate_labels();
						converged_ = true;
						break;
					}
				}
				old_log_likelihood = log_likelihood_;
			}
		}

		return converged_;
	}

	void EM::assign_responsibilities(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u) const
	{
		if (x.size() != means().rows()) {
			throw std::invalid_argument("Wrong x size");
		}
		if (u.size() != static_cast<Eigen::Index>(number_components())) {
			throw std::invalid_argument("Wrong u size");
		}
		for (unsigned int k = 0; k < number_components_; ++k) {
			u[k] = std::exp(-0.5 * LinearAlgebra::xAx_symmetric(inverse_covariances_[k], x - means_.col(k))) * mixing_probabilities_[k] / sqrt_covariance_determinants_[k];
		}
		u /= u.sum();
	}

	void EM::expectation_step(Eigen::Ref<const Eigen::MatrixXd> data)
	{
		const auto number_dimensions = data.rows();
		assert(number_dimensions);
		const auto sample_size = static_cast<unsigned int>(data.cols());
		assert(sample_size >= number_components_);
		
		static const double log_2_pi = std::log(2. * PI);
		const auto log_likelihood_normalisation_constant = static_cast<double>(number_dimensions) * log_2_pi / 2;		

		// Calculate unnormalised responsibilities.
		for (unsigned int k = 0; k < number_components_; ++k) {
			const auto mean = means_.col(k);
			auto component_weights = responsibilities_.col(k);
			const auto& inverse_covariance = inverse_covariances_[k];
			for (unsigned int i = 0; i < sample_size; ++i) {
				work_vector_ = data.col(i) - mean;
				component_weights[i] = std::exp(-0.5 * LinearAlgebra::xAx_symmetric(inverse_covariance, work_vector_));
			}			
			component_weights *= mixing_probabilities_[k] / sqrt_covariance_determinants_[k];
		}
		log_likelihood_ = responsibilities_.rowwise().sum().array().log().mean() - log_likelihood_normalisation_constant;

		// Normalise responsibilities for each datapoint.
		for (unsigned int i = 0; i < sample_size; ++i) {
			auto datapoint_responsibilities = responsibilities_.row(i);
			const double sum_weights = datapoint_responsibilities.sum();
			datapoint_responsibilities /= sum_weights;
		}
	}

	void EM::maximisation_step(Eigen::Ref<const Eigen::MatrixXd> data)
	{
		const auto number_dimensions = data.rows();
		assert(number_dimensions);
		const auto sample_size = static_cast<unsigned int>(data.cols());
		assert(sample_size >= number_components_);

		// Calculate new means.
		means_.noalias() = data * responsibilities_; // Unnormalised!
		assert(means_.rows() == number_dimensions);
		assert(static_cast<unsigned int>(means_.cols()) == number_components_);

		// Calculate new covariances.
		for (unsigned int k = 0; k < number_components_; ++k) {
			auto& covariance = covariances_[k];
			covariance *= 0;
			const auto component_weights = responsibilities_.col(k);
			const double sum_component_weights = component_weights.sum();

			// Normalise the mean.
			auto mean = means_.col(k);
			mean /= sum_component_weights;

			// Accumulate covariance.
			for (unsigned int i = 0; i < sample_size; ++i) {
				work_vector_ = data.col(i) - mean;
				LinearAlgebra::add_a_xxT(work_vector_, covariance, component_weights[i]);
			}

			covariance /= sum_component_weights;

			static constexpr double epsilon = 1e-15;
			// Add epsilon * I to avoid numerical issues with positive-definites.
			for (Eigen::Index i = 0; i < number_dimensions; ++i) {
				covariance(i, i) += epsilon;
			}
			mixing_probabilities_[k] = sum_component_weights / static_cast<double>(sample_size);
			assert(covariance.rows() == number_dimensions);
			assert(covariance.cols() == number_dimensions);
		}

		process_covariances(number_dimensions);
	}

	Eigen::MatrixXd EM::calculate_sample_covariance(Eigen::Ref<const Eigen::MatrixXd> data)
	{
		const Eigen::MatrixXd centred = data.colwise() - data.rowwise().mean();
		const Eigen::MatrixXd covariance = (centred * centred.adjoint()) / (static_cast<double>(data.cols() - 1));
		assert(covariance.rows() == covariance.cols());
		assert(covariance.rows() == data.rows());
		return covariance;
	}

	void EM::process_covariances(const Eigen::Index number_dimensions)
	{
		// Decompose and invert covariance matrices.
		for (unsigned int k = 0; k < number_components_; ++k) {
			auto& llt = covariance_decompositions_[k];
			llt.compute(covariances_[k]);
			inverse_covariances_[k] = llt.solve(Eigen::MatrixXd::Identity(number_dimensions, number_dimensions));
			double sqrt_covariance_determinant = 1;
			for (Eigen::Index i = 0; i < number_dimensions; ++i) {
				sqrt_covariance_determinant *= llt.matrixL()(i, i);
			}
			sqrt_covariance_determinants_[k] = sqrt_covariance_determinant;
		}
	}

	void EM::calculate_labels()
	{
		assert(labels_.size() == static_cast<size_t>(responsibilities_.rows()));
		for (Eigen::Index i = 0; i < responsibilities_.rows(); ++i) {
			double max_responsibility = -1;
			Eigen::Index label = -1;
			const auto row = responsibilities_.row(i);
			for (Eigen::Index k = 0; k < responsibilities_.cols(); ++k) {
				if (responsibilities_(i, k) > max_responsibility) {
					max_responsibility = row[k];
					label = k;
				}
			}
			labels_[i] = static_cast<unsigned int>(label);
		}
	}
}