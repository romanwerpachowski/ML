#pragma once
#include <memory>
#include <random>
#include <vector>
#include <Eigen/Core>
#include "dll.hpp"

// For Python API.
using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace ml 
{
	namespace Clustering
	{
		class CentroidsInitialiser;
		class ResponsibilitiesInitialiser;
	}

	/** Gaussian Expectation-Maximisation algorithm.

	Iterates until log-likelihood converges.
	*/
	class EM {
	public:		
		/** Constructs an EM ready to fit.
		@param number_components Number of Gaussian components, > 0.		
		*/
		DLL_DECLSPEC EM(unsigned int number_components);

		/** Sets PRNG seed. */
		DLL_DECLSPEC void set_seed(unsigned int seed);

		/** Sets absolute tolerance for convergence test. */
		DLL_DECLSPEC void set_absolute_tolerance(double absolute_tolerance);

		/** Sets relative tolerance for convergence test. */
		DLL_DECLSPEC void set_relative_tolerance(double relative_tolerance);

		/** Sets maximum number of E-M steps. */
		DLL_DECLSPEC void set_maximum_steps(unsigned int maximum_steps);

		/** Sets means initialiser.
		@param means_initialiser Pointer to MeansInitialiser implementation.
		*/
		DLL_DECLSPEC void set_means_initialiser(std::shared_ptr<const Clustering::CentroidsInitialiser> means_initialiser);

		/** Sets responsibilities initialiser.
		@param responsibilities_initialiser Pointer to ResponsibilitiesInitialiser implementation.
		*/
		DLL_DECLSPEC void set_responsibilities_initialiser(std::shared_ptr<const Clustering::ResponsibilitiesInitialiser> responsibilities_initialiser);

		/** Switches between verbose and quiet mode. */
		void set_verbose(bool verbose)
		{
			verbose_ = verbose;
		}

		/** Switches between starting with E or M step first. */
		void set_maximise_first(bool maximise_first)
		{
			maximise_first_ = maximise_first;
		}

		/** Fits the model.
		@param data Matrix (column-major order) with a data point in every column.
		*/
		DLL_DECLSPEC bool fit(Eigen::Ref<const Eigen::MatrixXd> data);

		/** Fits the model to data in row-major order.
		@param data Matrix (row-major order) with a data point in every row.
		*/
		DLL_DECLSPEC bool fit_row_major(Eigen::Ref<const MatrixXdR> data);

		/** Returns the number of components. */
		auto number_components() const
		{
			return number_components_;
		}

		const auto& means() const 
		{
			return means_;
		}

		const auto& covariances() const
		{
			return covariances_;
		}

		DLL_DECLSPEC const Eigen::MatrixXd& covariance(unsigned int k) const;

		const auto& mixing_probabilities() const 
		{
			return mixing_probabilities_;
		}

		const auto& responsibilities() const 
		{
			return responsibilities_;
		}

		double log_likelihood() const 
		{
			return log_likelihood_;
		}		

		std::shared_ptr<const Clustering::CentroidsInitialiser> means_initialiser() const
		{
			return means_initialiser_;
		}		
	private:
		std::default_random_engine prng_;
		std::shared_ptr<const Clustering::CentroidsInitialiser> means_initialiser_;
		std::shared_ptr<const Clustering::ResponsibilitiesInitialiser> responsibilities_initialiser_;
		Eigen::VectorXd mixing_probabilities_;
		Eigen::MatrixXd means_; /**< 2D matrix with size number_dimensions x number_components. */
		Eigen::MatrixXd responsibilities_; /**< 2D matrix with size sample_size x number_components. */
		Eigen::MatrixXd work_matrix_;
		Eigen::VectorXd work_vector_;
		std::vector<Eigen::MatrixXd> covariances_; /**< Vector of number_components 2D matrices with size number_dimensions x number_dimensions. */
		double absolute_tolerance_;
		double relative_tolerance_;
		double log_likelihood_;
		unsigned int number_components_;
		unsigned int maximum_steps_;
		bool verbose_;
		bool maximise_first_;

		static Eigen::MatrixXd calculate_sample_covariance(Eigen::Ref<const Eigen::MatrixXd> data);

		void expectation_stage(Eigen::Ref<const Eigen::MatrixXd> data);

		void maximisation_stage(Eigen::Ref<const Eigen::MatrixXd> data);
	};
}
