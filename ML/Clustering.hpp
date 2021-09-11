#pragma once
/* (C) 2020 Roman Werpachowski. */
#include <memory>
#include <random>
#include <vector>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
	/** @brief Methods and classes for clustering algorithms. */
	namespace Clustering
	{
		/**
		 * @brief Abstract clustering model.
		*/
		class Model
		{
		public:
			/**
			 * @brief Virtual destructor.
			*/
			DLL_DECLSPEC virtual ~Model();

			/** @brief Fits the model.
			@param[in] data Matrix (column-major order) with a data point in every column.
			@return `true` if fitting converged successfully.
			@throw std::invalid_argument If `data` has no rows, or if the sample size (number of columns in `data`) is too low.
			*/
			virtual bool fit(Eigen::Ref<const Eigen::MatrixXd> data) = 0;

			/** @brief Returns the number of clusters. 
			* 
			* Value make sense only if fitting converged successfully.
			*/
			virtual unsigned int number_clusters() const = 0;

			/** @brief Returns a const reference to resulting cluster labels for each datapoint.
			
			Values make sense only if fitting converged successfully.
			*/
			virtual const std::vector<unsigned int>& labels() const = 0;

			/**
			 * @brief Returns a const reference to the matrix of cluster centroids (in columns). 
			 * 
			 * Values make sense only if fitting converged successfully.
			*/
			virtual const Eigen::MatrixXd& centroids() const = 0;
		};

		/** @brief Chooses initial locations of centroids. */
		class CentroidsInitialiser
		{
		public:
			/** @brief Virtual destructor. */
			DLL_DECLSPEC virtual ~CentroidsInitialiser();

			/** @brief Initialises location of centroids. 

			@param[in] data Data matrix with data points in columns.
			@param[in,out] prng Pseudo-random number generator.
			@param[in] number_components Number of centroids. Must be less or equal to `data.cols()`.
			@param[out] centroids Destination matrix for centroid locations, with `data.rows()` rows and `number_components` columns.
			*/
			DLL_DECLSPEC virtual void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> centroids) const = 0;
		};

		/** @brief Chooses initial component responsibilities. */
		class ResponsibilitiesInitialiser
		{
		public:
			/** @brief Virtual destructor. */
			DLL_DECLSPEC virtual ~ResponsibilitiesInitialiser();

			/** @brief Initialises component responsibilities.

			@param[in] data Data matrix with data points in columns.
			@param[in,out] prng Pseudo-random number generator.
			@param[in] number_components Number of centroids. Must be less or equal to `data.cols()`.
			@param[out] responsibilities Destination matrix for component responsibilities, with `data.cols()` rows and `number_components` columns.
			*/
			DLL_DECLSPEC virtual void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> responsibilities) const = 0;
		};

		/** @brief Chooses random points as new centroids. */
		class Forgy : public CentroidsInitialiser
		{
		public:
			DLL_DECLSPEC void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> centroids) const override;
		};

		/** @brief Assigns points to clusters randomly and then returns cluster means. */
		class RandomPartition : public CentroidsInitialiser
		{
		public:
			DLL_DECLSPEC void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> centroids) const override;
		};

		/** @brief Implements the K++ algorithm.
		
		See https://en.wikipedia.org/wiki/K-means%2B%2B */
		class KPP : public CentroidsInitialiser
		{
		public:
			DLL_DECLSPEC void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> centroids) const override;
		};

		/** @brief Initialises centroids and then assigns the responsibility for each point to its closest centroid. */
		class ClosestCentroid : public ResponsibilitiesInitialiser
		{
		public:
			/** @brief Constructor.
			 @param centroids_initialiser Non-null pointer to CentroidsInitialiser implementation used to initialise the centroids.
			 @throw std::invalid_argument If `centroids_initialiser` is null.
			*/
			DLL_DECLSPEC ClosestCentroid(std::shared_ptr<const CentroidsInitialiser> centroids_initialiser);

			void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> responsibilities) const override;
		private:
			std::shared_ptr<const CentroidsInitialiser> centroids_initialiser_;
		};
	}
}