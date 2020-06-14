#pragma once
#include <random>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
	/** Methods and classes for clustering. */
	namespace Clustering
	{
		/** Chooses initial locations of means. */
		class MeansInitialiser
		{
		public:
			DLL_DECLSPEC virtual ~MeansInitialiser();

			DLL_DECLSPEC virtual void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const = 0;
		};

		/** Chooses initial responsibilities. */
		class ResponsibilitiesInitialiser
		{
		public:
			DLL_DECLSPEC virtual ~ResponsibilitiesInitialiser();

			DLL_DECLSPEC virtual void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> responsibilities) const = 0;
		};

		/** Chooses random points as new means. */
		class Forgy : public MeansInitialiser
		{
		public:
			DLL_DECLSPEC void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const override;
		};

		/** Assigns points to clusters randomly and then returns cluster means. */
		class RandomPartition : public MeansInitialiser
		{
		public:
			DLL_DECLSPEC void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const override;
		};

		/** See https://en.wikipedia.org/wiki/K-means%2B%2B */
		class KPP : public MeansInitialiser
		{
		public:
			DLL_DECLSPEC void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const override;
		};

		/** Initialises means and then assigns the responsibility for each point to its closest mean. */
		class ClosestMean : public ResponsibilitiesInitialiser
		{
		public:
			DLL_DECLSPEC ClosestMean(std::shared_ptr<const MeansInitialiser> means_initialiser);

			void init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> responsibilities) const override;
		private:
			std::shared_ptr<const MeansInitialiser> means_initialiser_;
		};
	}
}