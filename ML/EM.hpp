#pragma once
/* (C) 2020 Roman Werpachowski. */
#include <memory>
#include <random>
#include <vector>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "Clustering.hpp"
#include "dll.hpp"


namespace ml 
{	
	/** @brief Gaussian Expectation-Maximisation algorithm.

	Iterates until log-likelihood converges.
	*/
	class EM: public Clustering::Model
	{
	public:		
		/** @brief Constructs an EM ready to fit.
		@param[in] number_components Number of Gaussian components.
		@throw If `number_components == 0`.
		*/
		DLL_DECLSPEC EM(unsigned int number_components);

		/** @brief Sets PRNG seed. 
		@param[in] seed PRNG seed.
		*/
		DLL_DECLSPEC void set_seed(unsigned int seed);

		/** @brief Sets absolute tolerance for convergence test. 
		@param[in] absolute_tolerance Absolute tolerance.
		@throw std::domain_error If `absolute_tolerance < 0`.
		*/
		DLL_DECLSPEC void set_absolute_tolerance(double absolute_tolerance);

		/** @brief Sets relative tolerance for convergence test. 
		@param[in] relative_tolerance Relative tolerance.
		@throw std::domain_error If `relative_tolerance < 0`.
		*/
		DLL_DECLSPEC void set_relative_tolerance(double relative_tolerance);

		/** @brief Sets maximum number of E-M steps. 
		@param[in] maximum_steps Maximum number of E-M steps.
		@throw std::invalid_argument If `maximum_steps < 2`.
		*/
		DLL_DECLSPEC void set_maximum_steps(unsigned int maximum_steps);

		/** @brief Sets means initialiser.
		@param[in] means_initialiser Pointer to Clustering::CentroidsInitialiser implementation.
		@throw std::invalid_argument If `means_initialiser` is null.
		*/
		DLL_DECLSPEC void set_means_initialiser(std::shared_ptr<const Clustering::CentroidsInitialiser> means_initialiser);

		/** @brief Sets responsibilities initialiser.
		@param[in] responsibilities_initialiser Pointer to ResponsibilitiesInitialiser implementation.
		@throw std::invalid_argument If `responsibilities_initialiser` is null.
		*/
		DLL_DECLSPEC void set_responsibilities_initialiser(std::shared_ptr<const Clustering::ResponsibilitiesInitialiser> responsibilities_initialiser);

		/** @brief Switches between verbose and quiet mode.
		@param[in] verbose `true` if we want verbose output.
		*/
		void set_verbose(bool verbose)
		{
			verbose_ = verbose;
		}

		/** @brief Switches between starting with E or M step first.
		@param[in] maximise_first `true` if we want to start with the E step.
		*/
		void set_maximise_first(bool maximise_first)
		{
			maximise_first_ = maximise_first;
		}

		/** @brief Fits the model.
		@param[in] data Matrix (column-major order) with a data point in every column.
		@return `true` if fitting converged successfully.
		*/
		DLL_DECLSPEC bool fit(Eigen::Ref<const Eigen::MatrixXd> data) override;

		/** @brief Returns the number of components. 
		*/
		auto number_components() const
		{
			return number_components_;
		}

		unsigned int number_clusters() const override
		{			
			return number_components();
		}

		/** @brief Returns a const reference to matrix containing fitted component means. 
		@return Const reference to `number_dimensions` x #number_components() matrix.
		*/
		const auto& means() const 
		{
			return means_;
		}

		const Eigen::MatrixXd& centroids() const override
		{			
			return means();
		}

		/** @brief Returns a const reference to fitted component covariance matrices. 
		@return Const reference to std::vector<Eigen::MatrixXd> with length equal to #number_components(). Each matrix has size `means().rows()` x `means().rows()`.
		*/
		const auto& covariances() const
		{
			return covariances_;
		}

		/** @brief Returns a const reference to fitted k-th component's covariance matrix.
		@return Const reference to symmetric matrix with size `means().rows()` x `means().rows()`.
		*/
		DLL_DECLSPEC const Eigen::MatrixXd& covariance(unsigned int k) const;

		/** @brief Returns a const reference to fitted component mixing probabilities.
		@return Const reference to vector with size #number_components().
		*/
		const auto& mixing_probabilities() const 
		{
			return mixing_probabilities_;
		}

		/** @brief Returns a const reference to resulting component responsibilities. 
		@return Const reference to `sample_size` x #number_components() matrix.
		*/
		const auto& responsibilities() const 
		{
			return responsibilities_;
		}

		/** @brief Returns a const reference to maximised log-likelihood of training data. */
		double log_likelihood() const 
		{
			return log_likelihood_;
		}		

		/** @brief Returns a shared pointer to means initialiser implementation. */
		std::shared_ptr<const Clustering::CentroidsInitialiser> means_initialiser() const
		{
			return means_initialiser_;
		}

		/** @brief Given a data point x, calculate each component's responsibilities for x and save them in u.

		@param[in] x Data point with correct dimension.
		@param[out] u Vector for responsibilities with length equal to #number_components().
		@throw std::invalid_argument If `x.size() != means().rows()` or `u.size() != number_components()`.
		*/
		DLL_DECLSPEC void assign_responsibilities(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u) const;		

		const std::vector<unsigned int>& labels() const override
		{
			return labels_;
		}

		bool converged() const override
		{
			return converged_;
		}
	private:
		std::default_random_engine prng_;
		std::shared_ptr<const Clustering::CentroidsInitialiser> means_initialiser_;
		std::shared_ptr<const Clustering::ResponsibilitiesInitialiser> responsibilities_initialiser_;
		Eigen::VectorXd mixing_probabilities_;
		Eigen::MatrixXd means_; /**< 2D matrix with size number_dimensions x number_components. */
		Eigen::MatrixXd responsibilities_; /**< 2D matrix with size sample_size x number_components. */
		Eigen::VectorXd work_vector_;
		std::vector<Eigen::MatrixXd> covariances_; /**< Vector of `number_components_` 2D matrices with size number_dimensions x number_dimensions. */
		std::vector<Eigen::MatrixXd> inverse_covariances_; /**< Inverses of `covariance_` matrices. */
		std::vector<Eigen::LLT<Eigen::MatrixXd>> covariance_decompositions_; /**< Cholesky decompositions of `covariance_` matrices. */		
		Eigen::VectorXd sqrt_covariance_determinants_; /**< Square roots of determinants of `covariance_` matrices. */
		std::vector<unsigned int> labels_; /**< Labels assigned to feature vectors by choosing the component with maximum responsibility. */
		double absolute_tolerance_;
		double relative_tolerance_;
		double log_likelihood_;
		unsigned int number_components_;
		unsigned int maximum_steps_;
		bool verbose_;
		bool maximise_first_;
		bool converged_;

		static Eigen::MatrixXd calculate_sample_covariance(Eigen::Ref<const Eigen::MatrixXd> data);

		void process_covariances(Eigen::Index number_dimensions);

		void expectation_step(Eigen::Ref<const Eigen::MatrixXd> data);

		void maximisation_step(Eigen::Ref<const Eigen::MatrixXd> data);

		void calculate_labels();
	};
}
