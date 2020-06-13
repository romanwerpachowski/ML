#pragma once
#include <memory>
#include <random>
#include <vector>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml 
{
	/** Gaussian Expectation-Maximisation algorithm.

	Iterates until log-likelihood converges.
	*/
	class EM {
	public:
		/** Chooses initial locations of means. */
		struct MeansInitialiser
		{
			DLL_DECLSPEC virtual ~MeansInitialiser();

			DLL_DECLSPEC virtual void choose(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const = 0;
		};

		/** Construct an EM ready to fit.
		@param number_components Number of Gaussian components, > 0.
		@param means_initialiser Pointer to MeansInitialiser implementation.
		*/
		DLL_DECLSPEC EM(unsigned int number_components, std::shared_ptr<const MeansInitialiser> means_initialiser);

		DLL_DECLSPEC void set_seed(unsigned int seed);

		DLL_DECLSPEC void set_absolute_tolerance(double absolute_tolerance);

		DLL_DECLSPEC void set_relative_tolerance(double relative_tolerance);

		DLL_DECLSPEC void set_maximum_steps(unsigned int maximum_steps);

		/**
		@param data Matrix with a data point in every column.
		*/
		DLL_DECLSPEC bool fit(Eigen::Ref<const Eigen::MatrixXd> data);

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

		void set_verbose(bool verbose) 
		{
			verbose_ = verbose;
		}

		/** Chooses random points as new means. */
		struct Forgy : public MeansInitialiser
		{
			DLL_DECLSPEC void choose(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const override;
		};

		/** Assigns points to clusters randomly and then returns cluster means. */
		struct RandomPartition : public MeansInitialiser
		{
			DLL_DECLSPEC void choose(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const override;
		};

		/** See https://en.wikipedia.org/wiki/K-means%2B%2B */
		struct KPP : public MeansInitialiser
		{
			DLL_DECLSPEC void choose(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const override;
		};
	private:
		std::default_random_engine prng_;
		std::shared_ptr<const MeansInitialiser> means_initialiser_;
		Eigen::VectorXd mixing_probabilities_;
		Eigen::MatrixXd means_;
		Eigen::MatrixXd responsibilities_;
		Eigen::MatrixXd work_matrix_;
		Eigen::VectorXd work_vector_;
		std::vector<Eigen::MatrixXd> covariances_;		
		double absolute_tolerance_;
		double relative_tolerance_;
		double log_likelihood_;
		unsigned int number_components_;
		unsigned int maximum_steps_;
		bool verbose_;

		static Eigen::MatrixXd calculate_sample_covariance(Eigen::Ref<const Eigen::MatrixXd> data);

		static Eigen::MatrixXd invert_symmetric_positive_definite_matrix(Eigen::Ref<const Eigen::MatrixXd> m);

		void expectation_stage(Eigen::Ref<const Eigen::MatrixXd> data);

		void maximisation_stage(Eigen::Ref<const Eigen::MatrixXd> data);
	};
}
