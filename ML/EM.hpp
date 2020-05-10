#pragma once
#include <random>
#include <vector>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	/** Gaussian Expectation-Maximisation algorithm.

	Iterates until 
	*/
	class EM {
	public:
		/** Construct an EM ready to fit.
		@param number_components Number of Gaussian components, > 0.
		*/
		CLASS_DECLSPEC EM(unsigned int number_components);

		CLASS_DECLSPEC void set_seed(unsigned int seed);

		CLASS_DECLSPEC void set_absolute_tolerance(double absolute_tolerance);

		CLASS_DECLSPEC void set_relative_tolerance(double relative_tolerance);

		CLASS_DECLSPEC void set_maximum_steps(unsigned int maximum_steps);

		/**
		@param data Matrix with a data point in every column.
		*/
		CLASS_DECLSPEC bool fit(const Eigen::MatrixXd& data);

		auto number_components() const {
			return number_components_;
		}

		const auto& means() const {
			return means_;
		}

		const auto& covariances() const {
			return covariances_;
		}

		CLASS_DECLSPEC const Eigen::MatrixXd& covariance(unsigned int k) const;

		const auto& mixing_probabilities() const {
			return mixing_probabilities_;
		}

		const auto& responsibilities() const {
			return responsibilities_;
		}

		double log_likelihood() const {
			return log_likelihood_;
		}

		void set_verbose(bool verbose) {
			verbose_ = verbose;
		}
	private:
		std::default_random_engine prng_;
		Eigen::VectorXd mixing_probabilities_;
		Eigen::MatrixXd means_;
		Eigen::MatrixXd responsibilities_;
		std::vector<Eigen::MatrixXd> covariances_;
		double absolute_tolerance_;
		double relative_tolerance_;
		double log_likelihood_;
		unsigned int number_components_;
		unsigned int maximum_steps_;
		bool verbose_;

		std::vector<Eigen::Index> sample_datapoints_without_replacement(Eigen::Index sample_size);

		static Eigen::MatrixXd calculate_sample_covariance(const Eigen::MatrixXd& data);

		static Eigen::MatrixXd invert_symmetric_positive_definite_matrix(const Eigen::MatrixXd& m);		
	};
}
