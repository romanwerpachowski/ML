#pragma once
#include <random>
#include <vector>
#include <Eigen/Core>

namespace ml 
{
	/** Gaussian Expectation-Maximisation algorithm.

	Iterates until 
	*/
	class EM {
	public:
		/** Construct an EM ready to fit.
		@param num_gaussians Number of Gaussian components, > 0.
		@param absolute_tolerance Absolute tolerance.
		*/
		EM(unsigned int number_components, double absolute_tolerance, double relative_tolerance);

		void set_seed(unsigned int seed);

		/**
		@param data Matrix with a data point in every column.
		*/
		double fit(const Eigen::MatrixXd& data);
	private:
		std::default_random_engine prng_;
		std::vector<double> mixing_proportions_;
		Eigen::MatrixXd means_;
		std::vector<Eigen::MatrixXd> covariances_;
		double absolute_tolerance_;
		double relative_tolerance_;
		unsigned int number_components_;

		std::vector<Eigen::Index> sample_datapoints_without_replacement(Eigen::Index sample_size);

		static Eigen::MatrixXd calculate_sample_covariance(const Eigen::MatrixXd& data);

		static Eigen::MatrixXd invert_symmetric_positive_definite_matrix(const Eigen::MatrixXd& m);
	};
}
