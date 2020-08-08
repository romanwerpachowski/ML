#pragma once
#include <random>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
	/** Methods and classes for clustering. */
	namespace Clustering
	{
		/** Chooses initial locations of centroids. */
		class CentroidsInitialiser
		{
		public:
			DLL_DECLSPEC virtual ~CentroidsInitialiser();

			/** Initialises location of centroids. 

			@param data Data matrix with data points in columns.
			@param prng Pseudo-random number generator.
			@param number_components Number of centroids. Must be less or equal to data.cols().
			@param[out] centroids Destination matrix for centroid locations, with data.rows() rows and number_components columns.
			*/
			DLL_DECLSPEC virtual void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> centroids) const = 0;
		};

		/** Chooses initial component responsibilities. */
		class ResponsibilitiesInitialiser
		{
		public:
			DLL_DECLSPEC virtual ~ResponsibilitiesInitialiser();

			/** Initialises component responsibilities.

			@param data Data matrix with data points in columns.
			@param prng Pseudo-random number generator.
			@param number_components Number of centroids. Must be less or equal to data.cols().
			@param[out] responsibilities Destination matrix for component responsibilities, with data.cols() rows and number_components columns.
			*/
			DLL_DECLSPEC virtual void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> responsibilities) const = 0;
		};

		/** Chooses random points as new centroids. */
		class Forgy : public CentroidsInitialiser
		{
		public:
			DLL_DECLSPEC void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> centroids) const override;
		};

		/** Assigns points to clusters randomly and then returns cluster means. */
		class RandomPartition : public CentroidsInitialiser
		{
		public:
			DLL_DECLSPEC void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> centroids) const override;
		};

		/** See https://en.wikipedia.org/wiki/K-means%2B%2B */
		class KPP : public CentroidsInitialiser
		{
		public:
			DLL_DECLSPEC void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> centroids) const override;
		};

		/** Initialises centroids and then assigns the responsibility for each point to its closest centroid. */
		class ClosestCentroid : public ResponsibilitiesInitialiser
		{
		public:
			DLL_DECLSPEC ClosestCentroid(std::shared_ptr<const CentroidsInitialiser> centroids_initialiser);

			void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> responsibilities) const override;
		private:
			std::shared_ptr<const CentroidsInitialiser> centroids_initialiser_;
		};
	}
}